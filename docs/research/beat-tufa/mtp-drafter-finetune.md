Verdict: feasible within the budget (about 2 engineering days, about 3.5 Kaggle GPU-hours) if we train only the MTP's 90.6M BF16 dense weights, with the 512 INT4 experts frozen as albucino ships them, on target hidden states dumped by a small Pennyroyal patch. Expected accept length is +5-10% (lossless 2.78 → ~2.9-3.05, ceiling 3.51). Ship it with lossless or near-lossless acceptance: under the 0.5/0.5 rule a better draft also changes what the model writes.

# Fine-tuning the MTP draft head on our ARC traffic (written 2026-10-08, CPU container, nothing run on a GPU)

Labels:
- **[verified]**: checked in this session against the real artifact; how is in section 8.
- **[source]**: read in the Pennyroyal wheel (`dfranzen/pennyroyal-v253`, `sglang-0.5.19+gd00d88efc8d6`, sha256
  `d0620216...`, the file the REAP patch was written against) or in HF transformers 5.19.0, but not executed.
- **[measured]**: from our own run outputs, named.
- **[published]**: from a paper or model card, named.
- **[estimate]**: derived; the derivation is given.

## 0. What this session established

| Fact | Label |
|---|---|
| The full **BF16 original MTP** is already an input of our notebooks. It is Intel's `model_extra_tensors.safetensors` in `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1`: 1,565 tensors, 5,214,301,696 B, per-expert names. Its header is identical to HF `Intel/...W4A16-AutoRound@4c67bf68` | [verified] |
| albucino's 29 dense `mtp.*` tensors equal that BF16 original. Five sampled tensors were bit-identical, and so were 3 × 1 MiB slices of its `lm_head` copy | [verified] |
| albucino's experts are plain RTN of the same BF16 weights. The recipe in section 4 reproduces 100% of the INT4 codes and BF16 scales of two sampled matrices. The INT4 experts carry about **10% relative RMS weight error** (9.8%, 10.1%) | [verified] |
| Swift-1.5 ships Qwen's original MTP, not one retrained for Swift. The first 9 dense tensors (52.6 MB) of `model_mtp.safetensors` in `phuongncn/arc3-qwen38-swift-w4a16-autoround/PyTorch/w4a16-b/1` are bit-identical to Intel's. The experts and later tensors were not compared | [verified] |
| On 154 real ARC requests (runs/fidelity-base, greedy, lossless), today's draft has accept length **2.743** per verify step, token-weighted (2.784 as a per-request mean). 58.1% of proposed drafts are accepted (18,327 of 31,524) | [measured] |
| A perfect draft (always the target's mode) at our sampling settings (T 0.7, top-k 20, top-p 0.95, lossless) would reach **3.51**. Under the 0.5/0.5 relaxed rule it would reach **3.95**. Today the figures are **2.78** and **3.10** | [estimate from measured logprobs, section 6.1] |
| SGLang *can* return the MTP's exact input for prompt tokens. It does so as JSON float lists, which is unusable at 60-120k tokens. A binary dump patch is needed | [source] |

## 1. The MTP architecture of Qwen3.8-Flash-Next

Sources:
- [source] `models/qwen4_exp_mtp.py`, `models/qwen3_5_mtp.py`, `models/qwen4_exp.py`, `layers/hyperconnection.py`,
  `speculative/eagle_worker_v2.py` and `layers/attention/qwen_sparse_attn_backend.py` in the wheel;
- [source] HF `transformers/models/qwen4_exp/modeling_qwen4_exp.py` (5.19.0), which matches;
- [verified] the safetensors headers and `config.json` (`hidden_size` 2560, `hc_count` 4, `hc_lowrank` 320,
  `num_experts` 512, `num_experts_per_tok` 10, `moe_intermediate_size` 640, `shared_expert_intermediate_size` 640,
  `num_attention_heads` 24, `num_key_value_heads` 2, `head_dim` 256, `partial_rotary_factor` 0.25, M-RoPE sections
  11/11/10, `rope_theta` 1e7, `indexer_*` 4/1/128/2048/4, `vocab_size` 248,320, `mtp_num_hidden_layers` 1,
  `mtp_use_dedicated_embeddings` false).

**Inputs and output.** Notation:
- `x_j` is the token at position j.
- `H_j` is the target's final hyper-connection state at position j. It is the output of the last decoder layer and the input of the target's final mixer: 4 streams × 2560 = 10,240 BF16 values.
- `e(x)` is the target's token embedding.

The MTP block at position j takes `(H_j, e(x_{j+1}))` and produces:
- logits for `x_{j+2}`;
- its own 10,240-wide output state `G_j`.

`embed_tokens` and `lm_head` are not its own. At start-up the NEXTN worker replaces them with the target's
(`set_embed_and_head`, eagle_worker_v2.py:298-369). The head is sliced to the 65,536 rows of the FR-Spec map.
albucino's 2.5 GB of `lm_head`/`embed_tokens` copies are never loaded: `load_weights` skips every name without "mtp" [source].

**The three-step chain** [source]:
1. **Draft-extend.** This runs after the prefill and after every verify. The MTP runs over every newly accepted
   position j, with the target's `H_j` and the input ids shifted left by one (eagle_worker_v2.py:917-925). It writes the MTP
   layer's KV cache for those positions. The draft KV uses the target's token slots, so prefix-cached history keeps
   its draft KV.
2. **Step 1** at the last accepted position t: the input is `(H_t, e(x_{t+1}))`, where `x_{t+1}` is the bonus token the target
   just sampled. The output is `x̂_{t+2}` = argmax over the hot vocabulary (topk 1).
3. **Step 2** at t+1: the input is `(G_t, e(x̂_{t+2}))`, so the hidden input is the MTP's *own* previous output (`hidden_states =
   logits_output.hidden_states`, eagle_worker_v2.py:836). The output is `x̂_{t+3}`. **Step 3** works the same way and gives `x̂_{t+4}`.
4. **Verify.** The target scores `[x_{t+1}, x̂_{t+2}, x̂_{t+3}, x̂_{t+4}]` in one forward. Accept length = 1 + accepted drafts, at most 4.
   QSA's pending index-key ring refuses more than `indexer_compress_ratio` = 4 draft tokens
   (`_require_chain_speculation`, qwen_sparse_attn_backend.py:291-306), so 3 steps is the maximum and trees are impossible.

**Inside the block** (one `full_attention` decoder layer, no PLE) [source]:
1. **Input fusion** (`_fuse_residual_linear_shared`):
   - `u = fc_embedding(RMSNorm_2560(e))`.
   - One RMSNorm runs over all 10,240 values of H. Its result is viewed as 4 streams, and `fc_hidden` (one 2560×2560 matrix shared by the streams) is applied to each.
   - Stream i = `fc_hidden(stream_i) + u`.
   - All norms here are Gemma-style, (1 + w)·x̂.
2. **Gated residual around attention** (`GatedResidual`):
   - *Mix*:
     - `Hn` = a per-stream RMSNorm of H (weight 10,240, groups of 2,560).
     - Gates `w = sigmoid(Up(silu(Down(Hn)/4)))`, with Down 10,240→320 and Up 320→10,240.
     - The block input is the mean over streams of `w ⊙ Hn` (2,560 values).
   - *Combine*: `H' = H + out ⊗ 2·sigmoid(Inject(Hn)/4)`, with Inject 10,240→4, giving one injection scale per stream.
3. **Attention:**
   - 24 query heads × 256, each with a sigmoid output gate (`q_proj` 2560→12,288 = queries + gates).
   - 2 KV heads (GQA 12:1).
   - Per-head q/k RMSNorm.
   - Partial RoPE on 64 of the 256 dims, in the interleaved M-RoPE layout.
   - `o_proj` 6,144→2,560.
4. **Qwen Sparse Attention (QSA) indexer:**
   - `index_qk_proj` 2560→640, i.e. 4 index query heads plus 1 shared key head, each of 128.
   - Keys are mean-pooled over 4-token blocks, RMS-normed, and rotated at the block start.
   - Blocks are scored by `Σ_h relu(q·k)/√128`. The top 512 blocks (2,048 tokens) are kept, plus the incomplete tail block.
   - With 2,048 visible tokens or fewer, every token is kept, so attention is dense (HF `Qwen4ExpTextQSAIndexer`; SGLang's
     all-visible prefill path).
   - In draft decode steps the indexer does not run. Steps 2-3 reuse the selection captured at the last accepted
     draft-extend row (qwen_sparse_attn_backend.py:1407-1531).
5. **Gated residual around the MoE:**
   - Router 512×2560: softmax, top-10, renormalized.
   - 512 SwiGLU experts, 2560→640→2560.
   - One shared SwiGLU expert (640), scaled by `sigmoid(shared_expert_gate · x)`.
6. **Final mixer** (a gated residual without combine) → 2,560 values → the target's `lm_head`. The block's pre-mixer state `G`
   feeds the next step. There is no separate final norm: the mixer's per-stream RMSNorm plays that role.

**Parameters** [verified, headers]:

| Group | Params | BF16 |
|---|---|---|
| Attention (q/k/v/o, q/k norms, indexer) | 51.4M | 102.9 MB |
| Gated residuals (attention, MoE, final mixer) | 19.8M | 39.5 MB |
| Input fusion (`fc_embedding`, `fc_hidden`, 2 pre-fc norms) | 13.1M | 26.2 MB |
| Router | 1.3M | 2.6 MB |
| Shared expert + its gate | 4.9M | 9.8 MB |
| **Dense total** (29 `mtp.*` tensors) | **90.6M** | **181.1 MB** |
| Routed experts | 2,516.6M | 5,033 MB BF16; **1,416 MB** in albucino's INT4 g32 file |

Per token, the dense part is about 65% of the active MTP parameters: 90.6M of 139.7M, the rest being 10 experts × 4.9M.

## 2. Training data

### 2.1 What the MTP consumes
- **`H_j`.** These are the 10,240 BF16 values per token of the target's final-layer hyper-connection state. They come
  before the target's final mixer. `Qwen4ExpForConditionalGeneration.forward` overwrites
  `LogitsProcessorOutput.hidden_states` with `last_hc_hidden_states` for every token of every forward
  (qwen4_exp.py:1953-1958) [source].
- **Token ids**, for `e(x_{j+1})`, and the **M-RoPE positions** (3 per token; images get 2-D positions).
- **One multimodal detail** [source; not verified at runtime]:
  - In a prefill chunk that contains an image, the draft takes the target's input embeddings (`mm_input_embeds`,
    with vision features at image rows). Only the last row is replaced by the shifted token (`_prepare_input_embeds`).
  - The other rows therefore look unshifted, unlike the text-only path.
  - This touches only the MTP's KV for prompt rows in image-bearing chunks, never the assistant tokens we train on.
  - The dump records image rows so the trainer can mimic it.

### 2.2 Does SGLang return them? Yes, but not usably at our lengths [source]
- **Switches.** `--enable-return-hidden-states` (equivalent to `--return-hidden-states-mode full`) or
  `--return-hidden-states-mode last`, plus the request field `return_hidden_states: true` or `"last"`.
- **Prompt coverage.** In FULL mode `_append_prefill_hidden_states` (batch_result_processor.py:562-586) appends the rows of
  every extend chunk, so prompt tokens are covered.
  - **Radix-cache hits are not.** Cached prefix tokens are not recomputed, so we must flush (`/flush_cache`) or run
    with `--disable-radix-cache`.
  - Because of the Qwen4-Exp override, the rows are exactly `H_j`.
  - Decode tokens are appended per accepted token.
- **Why it is unusable.** Every chunk goes through `.cpu().clone().tolist()` and comes back as JSON. A 64k-token prompt
  is 655M floats, roughly 10 GB of JSON and 20 GB of Python objects. `"last"` returns only one row.

### 2.3 Can HF transformers run the W4A16 target instead? Not at our scale
- **What exists.** Qwen4-Exp modeling was added in **transformers 5.16.0** (PR #48337) [published, release notes]. In
  5.19.0's `modeling_qwen4_exp.py` [source]:
  - MTP weights are dropped on load (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`).
  - The QSA indexer is a Python loop over every query position.
  - Experts are fused 3-D tensors, looped over in Python.
  - GDN needs the `flash-linear-attention` kernel (ms-swift's Flash-Next page asks for >= 0.5.2 [published]) and
    `causal-conv1d` to be fast.
- **What the Intel target would also need:**
  - AutoRound's GPTQ-format MoE kernels mapped onto HF's fused expert tensors (not checked);
  - the 102 GB BF16 PLE table in host RAM;
  - a transformers >= 5.16 wheel attached as a dataset (the Kaggle image has 5.0.0).
- **Verdict.** It cannot prefill 60-120k-token prompts in our budget. Its modules are still useful as an
  independent CPU reference for the trainer's MTP block (section 3.7).

### 2.4 The dump patch (the recommended path)
**The edit.** One anchored, env-gated edit to the same file the REAP patch checks (`qwen4_exp.py`, sha256 `35a1785c...`), in the
style of `scripts/sglang_reap_patch.py`:
- **Where.** Right after `output = super().forward(...)` in `Qwen4ExpForConditionalGeneration.forward`.
- **When.** If `ARC3_HC_DUMP` is set and the batch is an EXTEND.
- **What it does:**
  - It takes `self.model.last_hc_hidden_states` and selects rows with a token-id state machine:
    - an assistant span starts after `<|im_start|>`(248045) `assistant`(74455) `\n` `<think>`(248068) `\n`;
    - the span ends at `<|im_end|>`(248046);
    - ids are from the served tokenizer, sha256 `06b95093...`.
  - It also keeps the C = 256 rows before each span from a ring buffer, as attention context.
  - Per chunk it writes:
    - the kept rows as FP8 e4m3 with one BF16 scale per row;
    - token ids for all positions;
    - absolute positions and M-RoPE positions;
    - the input-embedding row for any image row inside a kept range.

**Dump server.** It uses the serving target configuration, including REAP-448 at load if that is what we serve. Settings:
- no speculative decoding (faster start, no draft);
- `--disable-cuda-graph`, so the Python hook runs during prefill;
- `--max-running-requests 1`;
- `--disable-radix-cache`;
- chunk 8192.

**Driver.** It replays snapshots as chat completions with `max_tokens` 1 and temperature 0, using the logged messages, tools and `chat_template_kwargs`. The fidelity probe's replay code already does this.

### 2.5 How many tokens, and where they fit
**Cost per row.** One row is 20 KiB in BF16 or about 10 KiB in FP8 plus its scale. Prefill costs about 2.2-2.5 tokens per kept row, assuming assistant spans are about 40% of a snapshot with reasoning preserved [estimate].

**Target size.** Aim for 2-3M loss rows, about 3-3.5M rows including context. That is 31-36 GB in FP8 and about 7-8M prefill tokens. At the 10.7k tok/s
measured on Franzen's serve.log (requests with more than 4k new tokens, serving.md) it takes about 12 minutes [estimate].

**Why that is enough** [estimate]:
- We adapt a head that is already trained over multiple steps.
- SambaNova's domain drafters got most of their gain from 2k-19k samples [published].
- FastMTP used 389k samples from a much weaker starting point [published].

**Supply is not the limit:**
- Franzen's 10-game, 25-minute demo produced 925k output tokens [measured].
- ar25 produced 157-253k per 121-minute pass [measured].
- So one full-length run yields about 4M output tokens [estimate].

**Where it fits on Kaggle:**

| Store | Capacity | FP8 rows |
|---|---|---|
| `/kaggle/working` (also the session's output) | 19.5 GB, less ~4.1 GB for the new draft | ~1.4M |
| Host RAM as tmpfs while SGLang serves | 106-119 of 176.9 GB used (BF16 PLE is pinned), so ~55 GB free; `/dev/shm` size unverified | ~5M |
| `/` | our census lines show 7,102 of 8,157 GB used (a shared host disk); write quota unverified | ? |

Session A starts with a 10 GB write test of `/dev/shm` and `/tmp` and sizes the dump to what passes. After the server is
killed, the trainer has the whole 177 GB of RAM.

### 2.6 Building sequences from our logs
- **Use the request logs, not the text logs:**
  - Use `<game>_p0_requests.jsonl`: exact messages, images inline as base64 PNG, the template kwargs.
  - Franzen's harness writes them on Save & Run: exp-073b's log shows `save_request_logs=True`. Mount them from our
    run notebooks as `kernel_sources`; nothing needs downloading here.
  - The text logs (`runs/*/kernel-output/prompts/*.log`, about 6,000 segments and 8.1M output tokens) have no images and
    no message structure. Use them only for statistics, as `scripts/frspec_map.py` does.
- **Snapshots.** Per game pass, take the maximal snapshots: each request that the next request does not extend
  (the last one before each history trim) and the final request. Franzen's demo had 1-2 trims per game [measured].
  Because past reasoning is rendered (`preserve_thinking`), the snapshots together cover every assistant turn except
  each game's last reply.
- **Hygiene** (section 6.4 explains why):
  - Use only loop-free runs: exp-073, exp-073b and exp-075 had 0-1 exact repeats per ~560 turns. Never use exp-076.
  - Drop assistant turns that exactly repeat an earlier turn, and turns containing "stuck in a loop".
  - Prefer lossless runs. Their tokens are the target's own samples, not draft-biased ones.
- **Game split.** Hold out the 11 games of the fidelity probe sample (ar25, ft09, lp85, r11l, re86, sb26, sc25, tr87,
  tu93, vc33, tn36) and train on the other 14. Retrain on all 25 only after the gate passes. The hidden set is 110
  other games, so a game holdout is the honest generalization test.

## 3. The training loop

### 3.1 What to train (v1)
v1 trains all 90.6M dense tensors except the indexer. The routed experts, the indexer, `embed_tokens`, `lm_head` and the target's mixer stay frozen.

The experts stay exactly albucino's INT4: dequantized to BF16 in the trainer, these are the numbers the server multiplies by. Reasons:
- **The expert file stays byte-identical.** No requantization, and nothing new on the serving path. The dense weights are served in BF16, as trained.
- **The dense weights learn around the INT4 experts.** They can compensate for the experts' 10% RTN error.
- **Most of the capacity is dense.** Dense weights are 65% of the per-token active parameters. Attention and fusion carry most of the context-to-token mapping that domain shift breaks [estimate].
- **The indexer only matters beyond 2,048 tokens.** Keeping it frozen keeps Qwen's block selection at serving.

**Not in v1:**
- **LoRA on experts.** Rank 8 on gate/up/down would be about 39M parameters.
- **Full experts with quantization-aware training.** Fake-quant RTN g32 in the forward, with a straight-through gradient.
  This is 2.52B parameters, about 45 GB of weights, Adam state and gradients. It fits but is slower.
- Training experts *without* QAT and then applying RTN would lose most of the update in the 10% quantization noise.
  Use these only if v1 plateaus.

### 3.2 Starting point
- **Start from albucino as is.** Its dense tensors are the BF16 original (section 0), and its experts are the INT4 the server uses.
- **If Swift becomes the target,** the starting weights are the same, because Swift ships the original MTP. What changes is the
  source of `H`: dump with Swift as the served target. A draft must be trained on the target it will serve with.

### 3.3 The chain in training ("training-time test")
This is the same construction as EAGLE-3, SpecForge and AngelSpec. For every origin t in a window:
- **Step 1** uses `(H_t, e(x_{t+1}))`.
- **Step k** uses `(G^{(k-1)}_t, e(x_{t+k}))`. Here x are the realized tokens: teacher forcing is exact, because step k counts only when steps < k matched the realized tokens.
- **Positions:** step k sits at position t + k − 1.
- **Mask.** Step 1 is causal over the step-1 keys. Step k attends to step-1 keys at positions <= t plus its own k − 1 chain keys. This is the layout SGLang produces: draft KV conditioned on the target up to t, chain KV after it.

**Windows** are one assistant span (about 1,350 tokens on average) plus up to 256 rows of preceding context, at most 2,048 rows. Longer spans are split with overlap. Within 2,048 rows QSA is exactly dense, so the trainer needs no indexer.

Optionally, about 10% of windows can be 8k long, with the frozen indexer's top-512 block selection emulated. This addresses the long-context risk (section 6.5).

### 3.4 Loss
Per step, use the forward KL from the target's distribution to the draft's, over the 65,536-row hot vocabulary.

**Target distribution.** `p_k = softmax(lm_head_hot(Mixer_T(H_{t+k})))` is computed on the fly from the stored rows:
- `lm_head` and the target's final mixer are both BF16 in Intel's `model-00017-of-00017.safetensors` (1.28 GB) [verified, header].
- So distillation targets cost no extra storage.

**Weighting:**
- Step weights are (0.51, 0.31, 0.18), i.e. β = 0.6 decay, normalized, as in FastMTP and vLLM speculators [published].
- Count only positions whose target token lies inside an assistant span and inside the map.

**Choice of loss.** AngelSpec's ablation reports accepted length 3.80 with CE, 3.92 with KL and 3.95 with their LK loss [published]. Start with KL; LK is a v2 option.

### 3.5 Optimizer, memory, steps
**Optimizer:** AdamW, peak lr 5e-5 (FastMTP's), betas (0.9, 0.95), weight decay 0, 5% warmup, cosine decay to 10%, gradient clip 1.0.

**Batch and steps:** batches of 8 windows (about 13k loss tokens). Two epochs over about 3M rows is about 450 steps.

**Memory, well under 40 GB of the 96 GB** [estimate]:
- the trainable 90.6M (BF16 weights, FP32 master, Adam m and v, FP32 gradients): 1.6 GB;
- frozen experts dequantized to BF16: 5.0 GB;
- `embed_tokens`: 1.3 GB;
- hot `lm_head` rows: 0.34 GB;
- activations with per-step recomputation and chunked logits: 15-25 GB.

### 3.6 Time
- **Compute** [estimate]. About 5-7 GFLOP per row per epoch: three steps forward and backward plus the target logits, with the 64k-row `lm_head` dominating. That gives about 36 PFLOP for 3M rows × 2 epochs, 5-10 minutes of pure compute.
- **Budget.** Allow 30-45 minutes, because a simple PyTorch expert loop is launch-bound.

### 3.7 Replica fidelity
- **Port.** Write the block from SGLang's code. The trainer's own module goes in a scratch notebook cell, not in the repo.
- **CPU unit test.** Check it against HF 5.19's `Qwen4ExpTextDecoderLayer`, `GatedResidual` and router with random weights.
- **Expert numerics.** Dequantize experts exactly as `q × scale` in BF16.
- **GPU check.** The gate is the replica check of section 5.2.

## 4. Requantization and the launcher's draft view

**v1 needs no requantization.**
- The export is albucino's 12 files with `mtp-dense.safetensors` rewritten.
- The rewrite keeps the same 31 names, shapes, dtypes and tensor order. Every frozen tensor stays byte-identical, checked by sha256.
- `config.json`, `model.safetensors.index.json` and `mtp-routed-experts-int4.safetensors` are byte-identical copies.

**What `prepare_draft_view` in cell 12 requires**, checked against `kaggle/franzen/arc-agi-3-milestone-2-solution.ipynb`:
1. Exactly one `config.json` anywhere under `DRAFT_MODEL_DIR` (searched recursively). Its `quantization_config.quant_method` must be
   `compressed-tensors`, and `config_groups` must contain `mtp_routed_experts`. So never mount two drafts under one directory.
2. The group's weights must be `num_bits` 4, `group_size` 32, `symmetric` true. `targets` must be `["RoutedExperts"]` or the
   launcher's `EXPERT_TARGET` regex. `ignore` must be `[]` or `DENSE_IGNORE`. The view rewrites both to the regex forms.
3. `model.safetensors.index.json` must be non-empty and contain at least one `mtp.layers.0.mlp.experts.` key. Every shard
   it names must exist, be at least 8 bytes long, and have a relative path without `..`.
4. Tokenizer, chat template, preprocessor and generation config are linked from the target. The draft loads with
   `--speculative-draft-model-quantization compressed-tensors`. The FR-Spec and tokenizer sha checks are unaffected.

A file set with byte-identical config and index passes all four.

**Delivery.**
- Session A's `/kaggle/working` output (about 4.1 GB) is attached to later sessions through `kernel_sources`, with
  `DRAFT_MODEL_DIR` pointing at it. Resolve both mount layouts (lesson 0030).
- A builder option `--draft SRC` that swaps the albucino model source is new code. It is not written here.
- For the final submission, turn the output into a private Kaggle model.

**If experts are ever trained (v2):** albucino's exact recipe [verified, 100% of codes and scales reproduced on 2 matrices], with `w` a BF16 [out, in] matrix:
```python
g = w.reshape(out, -1, 32)                                            # groups of 32 along the input dim
scale = (g.float().abs().amax(-1) / 7.5).to(torch.bfloat16)          # weight_scale, BF16 [out, in/32]
q = torch.clamp(torch.round(g / scale.unsqueeze(-1)), -8, 7)          # BF16 division, round half to even
packed = ((q.long() + 8).reshape(out, -1, 8) << (4 * torch.arange(8))).sum(-1)  # weight_packed, int32 [out, in/8], low nibble first
# weight_shape = torch.tensor([out, in], dtype=torch.int64); same names per expert/projection as albucino
```
This is compressed-tensors' symmetric min/max observer, with bit range 15, hence /7.5. The unit test should reproduce
albucino bit-exactly from Intel's BF16 for all 1,536 matrices before any trained weights are packed.

## 5. Evaluation

### 5.1 Offline (session A, free)
On held-out-game windows, compare the original and fine-tuned weights on identical rows. Report per step:
- KL;
- top-1 agreement with the target's argmax;
- expected accept length, greedy and under the processed T 0.7 distribution.

### 5.2 Replica check (session A, before training; the go/no-go for everything else)
**Setup:**
- Dump 16 held-out probe requests over the full prompt plus the greedy output that runs/fidelity-base recorded (`token_ids`, ≤192 per request).
- Compute the replica's greedy chain accept length per request and compare it with SGLang's `spec_accept_length` from the same run.

**Pass criteria.** The mean difference must be ≤ 0.05 and the per-request correlation ≥ 0.9.

**Two variants:**
- full context with QSA selection emulated;
- context cut to 2k.

The difference between them measures how much the 2k training windows give up.

### 5.3 Probe gate (session B, about 25 minutes) [measured noise]
**Setup:**
- `--probe` with the new draft, greedy and lossless, on the 154 requests of the 11 held-out games.
- Compare per request with runs/fidelity-base (or runs/fidelity-reap448 if REAP is served).

**Noise floor.** fidelity-base2 against fidelity-base gives a paired difference with mean +0.025, SD 0.229 and SE 0.018. So a shift of about +0.05 (2%) is detectable.

**Pass:**
- mean Δ ≥ +0.08 with a bootstrap 95% CI above 0;
- both passes (seq and conc) agree;
- 0 failed requests;
- the result reported by prompt-length bucket.

Under greedy lossless decoding the outputs cannot depend on the draft except through batching nondeterminism, so this measures speed only.

### 5.4 Production gate (session C)
**Setup.** 25 games × 25 minutes at the candidate configuration, lossless, with the new draft.

**What to read:**
- serve.log: `accept len`, and `gen throughput` at `#running-req` ≥ 10;
- from the prompt logs: the exact-repeat-turn rate and the number of output tokens per request.

**Baseline.** Compare with the matching exp-072 gate.

### 5.5 Score runs
- **Lossless:** none needed. The output distribution is the target's.
- **Any relaxed setting:** at least 2 full public-25 runs plus the loop gate (section 6.4). These are outside the 4 GPU-hours.

## 6. Expected gain, relaxed acceptance, risks and plan

### 6.1 Measured baseline and ceilings
Ceilings use the fidelity probe's greedy top-5 logprobs: 154 requests, ≤192 tokens each, 28,819 tokens. The target's processed probability p' of its own mode is estimated from the top-5 at T 0.7 with top-k 20 and top-p 0.95. Tail bounds give ±0.015. A perfect draft's chain acceptance is then averaged along the greedy path [estimate from measured data].

| Acceptance rule (single, acc) | Today | Perfect-draft ceiling | Mean boost of the draft's token per position (perfect draft) |
|---|---|---|---|
| Greedy (probe) | 2.74 [measured] | 4.00 | 0 |
| Lossless (1.0, 1.0), T 0.7 | 2.78 [measured, production] | **3.51** | 0 (output = target) |
| (0.9, 1.0) | not run | 3.53 | 0.004 |
| (0.8, 1.0) | not run | 3.58 | 0.014 |
| (0.7, 1.0) | not run | 3.64 | 0.027; 5% of tokens boosted ≥ 0.2 |
| (0.5, 1.0) | not run | 3.85 | 0.066; 15% boosted ≥ 0.2 |
| (0.5, 0.5), our candidate | 3.10 [measured] | 3.95 | 0.083; 19% boosted ≥ 0.2 |

**How confident the target is.** Its T=1 top-1 probability has mean 0.862 and median 0.992. 65% of tokens are ≥ 0.9 and 9.3% are < 0.5.

**What a better draft can win.** The lossless gap a draft can close is 0.73 accept length (+26%). REAP-448 barely moves greedy acceptance (2.784 vs 2.774), so no retraining is needed for pruning itself.

### 6.2 Evidence from elsewhere
- **AngelSpec** (arXiv 2607.25852, Tencent Hy3) [published]:
  - Setup: one shared MTP block, D = 3, fine-tuned on target rollouts with training-time test.
  - Mean accept length 2.58 → 2.99 at T 0 and 2.54 → 2.90 at T 0.9.
  - The gains came at deep positions (0.799/0.518/0.266 → 0.814/0.653/0.524). The base model had weak deep positions.
  - "Increasing the configured sequence length alone cannot compensate for the absence of representative long-context trajectories."
- **FastMTP** (arXiv 2509.18362, MiMo-7B) [published]:
  - Setup: MTP head only (210.8M, shared across steps), 389k self-distilled samples, weighted CE with β 0.6, 3 epochs at lr 5e-5, under one day on one H20 server.
  - Per-position acceptance 70/10/~0% → 80/56/36%, speed-up 1.21× → 1.81×; 2.03× with a vocabulary cut.
  - The base was trained for one step and reused, which is weaker than Qwen's.
- **SambaNova domain drafters** (arXiv 2503.07807) [published]:
  - A generic draft loses up to 38% acceptance under domain shift.
  - Offline KD beats online KD by 11-25%, and forward-KL KD beats SFT.
  - Synthetic prompts recover 80-93% of the gain from real queries.
- **vLLM speculators MTP guide** [published]: fine-tune the existing MTP layers on in-domain data, with `embed_tokens` and `lm_head` frozen and shared, and β 0.6 step decay.
- **Intel notes (intel.md)** [published, relayed there]:
  - PixelML's DFlash drafter, trained on generic chat, beat Flash-Next's own MTP by only 3.87%, so generic data buys little.
  - Son Pham "retrained for the cut model" its draft head as part of a 25 → 40 step. That step is confounded with temperature and a restart rule, and no acceptance numbers were published.
  - Yi-Chia trained DFlash drafters for AIMO. DFlash and EAGLE-3 blocks of 7-16 tokens do not fit QSA's 4-token verify cap, so the MTP is the only drafter lever on our path.

### 6.3 Estimate [estimate]
**Assumption.** Closing 20-35% of the 0.73 lossless gap is plausible: our domain is narrow and repetitive (1.2-1.5% of our output tokens lay outside the generic 64k map), but the base MTP is already trained over multiple steps.

**Accept length:**
- lossless: 2.78 → **2.92-3.05 (+5-10%)**;
- greedy probe: 2.74 → 2.95-3.15;
- (0.5, 0.5): 3.10 → 3.25-3.40, which is lossy (section 6.4).

**Throughput and score.**
- The step cost is unchanged: same architecture, same INT4 experts, same map. So decode tok/s should scale with accept length (+5-10%).
- At our measured elasticity of about 0.7, that is about **+3.5-7% score**.
- A weak outcome (+2-3%) is possible if the single layer is capacity-bound. The probe gate shows this before any score run.

**The cleanest win.** The fine-tuned draft with lossless acceptance should come close to today's 0.5/0.5 speed (3.10) without its distortion.

### 6.4 How a fine-tuned draft interacts with relaxed acceptance, and safeguards
**The rule** [source, server_args.py:2241-2250]:
- "Accept a draft token if its probability in the target model is greater than [threshold_single]."
- "The accept probability of a draft token is raised from its target probability p to min(1, p / threshold_acc)."

So the draft's token d is emitted with probability at least `A(p) = 1 if p ≥ s else min(1, p/a)` instead of its target probability p. Only (1.0, 1.0) leaves the output equal to the target's. At (0.5, 0.5) the boost reaches 0.5 per position, and **the draft chooses which token gets it.**

**What exp-076 shows** [measured, research_log 2026-10-08 18:55]:
- Setup: Swift-1.5 as the target, the albucino draft (Qwen's original MTP, i.e. the base model's), (0.5, 0.5), REAP ids unverified.
- Accept length 3.70, against 3.09 for the Intel target in exp-075.
- 411 of 595 assistant turns (69%) were exact repeats of an earlier turn, in 14 games. The model wrote "I'm stuck in a loop" 226 times.
- Intel runs under the same rule had 0-1 repeats per ~560 turns.
- The causes are not yet separated; exp-076g (lossless, no REAP) is queued to separate them.

**Mechanism** [estimate]:
- A draft trained on another model proposes that model's continuations.
- In a multi-turn harness with preserved reasoning, the most predictable continuation of a turn that starts like an earlier one is a copy of it.
- Doubling (a = 0.5) turns a copy the target rates 0.3 into an acceptance probability of 0.6.
- Once the target itself rates the copy ≥ 0.5, snapping (s = 0.5) accepts it every time. That removes the 1 − p chance of escaping that sampling gives at every token.
- Each copied token raises the next copy's probability for both models, so the loop absorbs.

**What fine-tuning changes:**
1. **The mismatch part shrinks.** Proposals move toward the target's own mode. If Swift is adopted, a Swift-trained draft is the principled fix; for Intel it is harmless.
2. **The sharpening part grows.** More proposals are the mode, so more positions are snapped. Output moves from T 0.7 sampling toward greedy: a perfect draft at (0.5, 0.5) boosts 19% of tokens by at least 0.2. Qwen's cards for thinking models advise against greedy decoding because of "endless repetitions" [published, Qwen3 model cards]. So under (0.5, 0.5) a better draft can *raise* loop risk even for Intel.
3. **Contaminated data reinforces itself.** Training on looping or relaxed runs teaches the draft to predict copies with confidence, which relaxed acceptance then enforces.
4. **Accept length stops measuring speed.** Under relaxed rules a jump can mean the text became repetitive.

**Safeguards:**
- **S1. Ship lossless first:** fine-tuned draft + (1.0, 1.0). The output distribution is provably the target's, so only the throughput gate is needed.
- **S2. If more speed is wanted, prefer (0.8, 1.0) to (0.5, 0.5).** Keeping acc = 1.0 never doubles low-probability proposals, which is the mismatched-draft path, and snapping happens only when the target is ≥ 80% sure. The ceiling is 3.58 with a mean boost of 0.014, against 3.95 and 0.083. This is worth a gate with *today's* draft as well (section 7).
- **S3. Any non-lossless setting must pass a loop gate:**
  - exact-repeat turns ≤ 1 per 500;
  - "stuck in a loop" mentions, output tokens per request and the `finish_reason=length` share no worse than the lossless arm;
  - then at least 2 full public-25 runs.
- **S4. Treat an accept-length jump without a draft change as an alarm, not a win.** One example is 3.09 → 3.70 on a target swap. Another is anything above the lossless ceiling for that target.
- **S5. Keep the training data clean.** Use only loop-free runs and drop exact-repeat turns (section 2.6). Distill from the target's logits, so draft-biased sampled tokens never become labels.
- **S6. Pair every draft with its own target.** Never serve a draft trained on one target's hidden states with another target under relaxed rules. Thresholds are server flags, so a game in a loop cannot be switched to lossless mid-run.
- **S7. Add a harness-side breaker** that is independent of the draft. If a turn's first ~200 tokens repeat an earlier turn, abort and re-ask.

### 6.5 Risks
1. **Relaxed-acceptance behaviour** (section 6.4). This is the only quality risk, and it is avoided by S1.
2. **The replica differs from SGLang:**
   - QSA selection beyond 2k tokens;
   - multimodal draft embeddings;
   - M-RoPE positions of images;
   - online MXFP8, if serving enables it, since the draft's eligible linear layers convert too.

   Offline gains would then not transfer. The section 5.2 check catches this before training.
3. **Long context.** Training sees ≤ 2k of dense context; serving sees sparse attention over 60-120k. The 5.2 variants measure the gap, and the 8k windows of section 3.3 are the mitigation.
4. **The gain is small** because one layer is capacity-bound. The 25-minute probe shows it.
5. **Kaggle logistics:**
   - dump storage (section 2.5);
   - the RTX queue after the daily reset (lesson 0031): push in the evening UTC window;
   - every new target or REAP list needs a retrain, about 1.3 GPU-hours.
6. **Overfitting to 25 public games.** The game holdout covers this.
7. **Licence.** The weights are a derivative of Qwen's MTP, the same status as albucino and Intel (rule 5.a.3, thread 745079).

### 6.6 Plan
| Step | Where | Work | GPU-h |
|---|---|---|---|
| 0 | CPU, day 1 | Dump patch (anchored, sha-checked, env-gated) with tests on the wheel's file. Driver: snapshot selection, span rule, loop filter, game split | 0 |
| 1 | CPU, days 1-2 | Trainer: MTP block, INT4 dequantization, chain masks, KL loss, loader. CPU tests against HF 5.19 modules and albucino's dequantized weights. Exporter with byte-identity checks. Builder options for session A and `--draft` | 0 |
| 2 | Kaggle A | Storage test. Dump about 3M rows (about 6 minutes to boot plus about 12 of prefill). Replica check (about 10 minutes). Train (30-45 minutes). Offline eval. Export | ≈1.3 |
| 3 | Kaggle B | Probe gate: greedy, lossless, 154 held-out requests, against fidelity-base | ≈0.45 |
| 4 | Kaggle C | Production gate: 25 × 25 minutes, candidate configuration, lossless, new draft (accept length, tok/s, loops) | ≈0.7 |
| 5 | Contingency | One retrain, or the (0.8, 1.0) gate | ≈1.0 |
| | | **Total** | **≈3.5** |

**Stop rules:**
- After step 2, if the replica check fails, stop and add about 1 day of work.
- After step 3, if the probe gain is below +0.08, stop and drop the arm.

## 7. Smaller alternatives
1. **Flags only (no training, one gate each):**
   - Run (0.8, 1.0) with today's draft. It should land between lossless 2.78 and the relaxed 3.10, with a mean boost of 0.014 against 0.083.
   - The ARC FR-Spec map is already built: coverage 98.8% → 99.9%, worth about +0.5-1% [estimate]. It rides in exp-077.
2. **Fine-tune the projections only:**
   - Train only `fc_embedding`, `fc_hidden`, the two pre-fc norms and the router (14.4M parameters), with step-1 loss only (no chain masks).
   - Use ≤ 1M rows in `/kaggle/working`, in one session.
   - Cost: about 1 day and about 1.5 GPU-hours. Expected gain about half of the full plan, +2-5% [estimate].
   - The dump patch and the replica are the irreducible core either way.
3. **Better INT4 without training:**
   - Requantize the experts from Intel's BF16 with an MSE-optimal clip per group: search 0.8-1.0 × absmax/7.5.
   - Same format and size, zero runtime cost, CPU only. Expected 0-2%; how much INT4 costs acceptance is unknown.
   - A BF16-expert draft is a one-off way to measure that cost. It needs +3.6 GB of VRAM and is slower.
4. **Vocabulary:** a 32k map halves the draft's `lm_head` read, about 1% of step time. FastMTP found 32k best for English chat. Low value.

## 8. How this was checked (scratch tools were deleted afterwards; the numbers above are their outputs)
- **Wheel.** Downloaded alone with `kaggle datasets download dfranzen/pennyroyal-v253 -f wheels/sglang-0.5.19+gd00d88efc8d6-...whl`
  (sha256 `d0620216...`, same as the REAP work). Files were read from the unzipped `sglang/srt` tree.
- **Headers** via `scripts/kaggle_model_files.py header`:
  - albucino `mtp-dense.safetensors` (31 tensors) and `mtp-routed-experts-int4.safetensors` (4,608);
  - Swift `model_mtp.safetensors` (31);
  - Intel `model_extra_tensors.safetensors` (1,565).
  - All matched the HF headers (albucino@c6af9fee, Intel@4c67bf68).
- **Byte comparisons.**
  - HF range reads of Intel's and albucino's tensors: 5 dense tensors and 3 `lm_head` slices.
  - For Swift: the first 53 MB of its Kaggle file, streamed, against Intel's.
- **RTN reproduction.** CPU torch on Intel's BF16 `experts.0.gate_proj` and `experts.511.down_proj` against albucino's packed
  codes and scales. Of the variants tried, only BF16 division with round-half-even matched 100%; FP32 division matched 99.45%.
- **Probe data.** From `runs/fidelity-base/kernel-output/fidelity.json`:
  - accept-length counts and target-confidence statistics;
  - the ceilings, from the top-5 logprobs as in section 6.1;
  - paired noise against `runs/fidelity-base2` and `runs/fidelity-reap448`.
- **Request logs.** Counted in scratch copies of Franzen's demo output and v3 output (`*_requests.jsonl` usage records).
- **HF transformers 5.19.0.** The wheel from PyPI, read for `modeling_qwen4_exp.py`.
- **Literature** via web search and fetch: AngelSpec, FastMTP, SambaNova 2503.07807, the vLLM speculators MTP guide, and the
  transformers 5.16.0 release notes.
- **exp-076 facts.** `docs/research_log.md`, entry 2026-10-08 18:55.

## 9. Step 1 implemented: the dump patch and the snapshot driver (2026-10-08, CPU only, nothing run on a GPU)

Asked for as "section 7"; numbered 9 because sections 7 and 8 exist and are cited above.

**Outcome.** Step 0 of the plan in 6.6 is built and tested on CPU: the dump patch, the driver, the span rule, the loop filter and the game split. Nothing has run on a GPU. So the dump's speed, the FP8 error on real `H` and the disk use are still unmeasured, and session A measures them first.

### 9.1 What was built
| File | Role |
|---|---|
| `scripts/sglang_hc_dump_patch.py` | Runtime module (installed as `sglang/srt/arc3_hc_dump.py`) and installer (`apply`, `revert`, `check-wheel`), in the style of the REAP patch |
| `scripts/hc_dump_driver.py` | Plans maximal snapshots from `<game>-<id>_p<k>_requests.jsonl`, applies the hygiene rules and the game split, replays the snapshots with `max_tokens` 1, and checks the dump of every request |
| `tests/test_sglang_hc_dump_patch.py` | 36 tests: span rule, ring buffer, number formats, files, the SGLang hooks, the installer alone and with REAP |
| `tests/test_hc_dump_driver.py` | 13 tests: planning on samples of the bed's logs and on synthetic logs; replay against fake servers, one of which dumps through the real `Dumper` |
| `tests/fixtures/hc_dump_driver/` | 181 KB: shortened samples of the bed's exp-078 logs (ls20, sb26); how they were cut is in its NOTICE.md |

**The patch** (`python -I scripts/sglang_hc_dump_patch.py apply --site-packages SP`):
- **Two anchored edits to `qwen4_exp.py`; 20 lines added, none changed:**
  - A module flag right before `class Qwen4ExpForConditionalGeneration`. `ARC3_HC_DUMP` is read once, at import; the dump module is imported only when the variable is set.
  - Two `if _ARC3_HC_DUMP:` checks in that class's `forward`. The first, before `super().forward`, copies the batch's input ids, because the multimodal embedding clamps image rows in place. The second, after it, hands `self.model.last_hc_hidden_states` to `capture`.
  - With the variable unset, the two checks are the only work added (a test runs the edited method with the flag off and on).
- **Accepted base files:** the wheel's file (sha256 `35a1785c...`) and that file after the REAP patch (`37515e12...`); both hashes are pinned.
- **Apply REAP first:**
  - The REAP installer accepts only the pristine file. So it refuses a file this patch has edited, and it also refuses to run again on the combined file.
  - `revert` undoes this patch.
  - The edits do not overlap, so both orders give the same text (tested on the real file).
- **`capture`:**
  - Dumps EXTEND and MIXED batches only. It skips DECODE, TARGET_VERIFY and draft batches, ranks other than 0, and CUDA-graph capture.
  - A chunk that fails is logged and recorded in the index, and that request is no longer dumped. The server keeps running.

**Row selection** (`scan_spans`, `select_rows`; pure functions):
- **Span rule.** A span starts after `<|im_start|> assistant \n <think> \n` and ends at `<|im_end|>`, inclusive. Edge cases:
  - Empty reasoning: `<think>` is followed by `\n\n`, which is one token (271). The span starts at that token.
  - A header without `<think>`: the span starts right after `assistant \n`.
  - A `<|im_start|>` inside a span ends the span at the row before it.
  - The generation prompt opens a last span one position past the prompt. It has no rows.
- **Context.** Each span that has rows in a chunk gets the `ARC3_HC_DUMP_CONTEXT` rows (default 256) before its first row.
  - Context rows from earlier chunks come from a per-request ring buffer of the last 256 rows.
  - So the result does not depend on how the prompt was chunked. A test compares 32 chunkings, down to one token per chunk.
- **Plan files.** `<dump>/plans/<rid>.json` (`{"loss_spans": [...]}`) limits the span rows to the turns the trainer will use. The driver writes it before each request. The other spans' rows are kept only where they serve as context.
- `ARC3_HC_DUMP_KEEP=all` keeps every row, for the replica check.

**Files.** One file per request chunk, `<dump>/<rid>/c<chunk>-p<start>.safetensors`, in the plain safetensors layout. It is written without the library, and `safetensors.torch.load_file` reads it (tested). Contents:
- **Tensors:**
  - `token_ids` I32 [n]. Image rows keep SGLang's pad value, which is ≥ 1,000,000.
  - `positions` I32 [n] and `mrope_positions` I32 [3, n].
  - `hc_pos` I32 [k] and `hc_role` U8 [k] (1 for a span row, 0 for a context row).
  - `hc` F8_E4M3 [k, 10240] with `hc_scale` BF16 [k], one scale per row as planned. Two options: `ARC3_HC_DUMP_SCALE_GROUPS=4` gives [k, 4], one scale per stream; `ARC3_HC_DUMP_DTYPE=bf16` stores raw BF16 rows.
  - `img_pos` I32 [m] and `img_embeds` BF16 [m, 2560], for the image rows among the kept rows.
- **Self-description:**
  - Every file's `__metadata__` describes its tensors.
  - `format.json` describes the dump.
  - `index.jsonl` has one line per chunk: spans, counts, `qerr_*`, bytes.
  - `load_request` merges the chunks of one request.
- **Size:**
  - `ARC3_HC_DUMP_MAX_GB` stops writing before the cap is passed (a tmpfs is RAM).
  - A kept FP8 row takes 10,247 bytes, and every prefill token adds 20.

**The driver** (`python -I scripts/hc_dump_driver.py --logs DIR ... [--dry-run]`):
- **Dependencies.** Stdlib only. It loads `fidelity_probe.py` and `fidelity_sample.py` from its own folder.
- **Snapshots.** A snapshot is a request that the next request does not extend, plus the last request. "Extends" means the same tools and template kwargs, with this request's messages (compared without `_arc3_control`) as a prefix of the next one's.
- **Turns.** A turn is identified by its tool-call ids, which are unique per generation in real logs, or else by its JSON.
- **Excluded from the loss:**
  - exact repeats of an earlier turn's text ("repeat");
  - turns that mention "stuck in a loop", case-insensitive ("stuck");
  - unless `--keep-duplicates` is given, turns that an earlier snapshot already has ("covered"). History trims keep recent turns, and the first snapshot holds the context they were generated in.
  - A snapshot with no loss turn left is dropped.
- **Split.** `--split train` (the default) leaves out the 11 holdout games of 2.6; `holdout` and `all` are the other choices.
- **Replay:**
  - Each logged request goes through `fidelity_probe.build_body` with `max_tokens` 1, temperature 0, and the snapshot id as `rid` (SGLang's chat endpoint accepts `rid`).
  - Before each request the driver writes the plan file. After it, the driver checks the request in the index: rows must cover positions 0 to the server's `prompt_tokens`, and the number of assistant headers must equal the assistant messages + 1.
  - It stops at the first request that is not dumped, after three failures in a row, or before `--max-dump-gb`, `--max-prefill-tokens` or `--max-minutes` would be passed.
  - It flushes the prefix cache before each request unless the server reports `disable_radix_cache`.
- **Outputs:**
  - `plan.json`;
  - `snapshots.jsonl`: rid → game, split, source line, turns and loss spans; this is the trainer's map;
  - `replay.jsonl` and `replay-summary.json`.

### 9.2 Facts checked or corrected in this step
| Fact | Label |
|---|---|
| The hook point is `Qwen4ExpForConditionalGeneration.forward(self, *args, **kwargs)` (qwen4_exp.py:1952-1958). `self.model` is a `Qwen4ExpVLModel`, whose `forward` sets `last_hc_hidden_states` (None for idle batches). That value is the last decoder layer's `mlp_hyper_connection.combine` output, [tokens, 10240] BF16: the input of `hyper_connection_mixer.mix`. The model runner calls `model.forward(forward_batch.input_ids, forward_batch.positions, forward_batch, **kwargs)`. The MTP class (`Qwen4ExpForCausalLMMTP`) is not a subclass, so draft batches never reach the hook | [source] |
| The header ids `248045 74455 198 248068 198` and `<|im_end|>` = 248046 hold for the served tokenizer (sha256 06b95093) and chat template (sha256 c3cf9e34). Tested on 9 snapshots from Franzen's M2 demo logs (ft09, tu93) and the bed (ls20): every assistant turn has exactly this header, and the span count is assistant messages + 1. Empty reasoning renders as `<think>` followed by token 271 (`\n\n`) | [verified] |
| SGLang replaces `<|image_pad|>` (248056) with pad values 1,000,000 + hash mod 2^30. During the forward, `embed_mm_inputs` clamps the batch's `input_ids` to 248,319 **in place**. So the ids must be copied before `super().forward`, which the patch does. The tokenizer's largest id is 248,076 | [source] / [verified] |
| **Correction to 2.1: unshifted draft embeddings are not limited to chunks that contain an image.** `contains_mm_inputs()` is decided per request: every extend batch carries each request's `multimodal_inputs`. So every prefill chunk of a request with images uses `mm_input_embeds` unshifted, except each chunk's last row. Every request in our traffic has images. So at serving, every *prefilled* prompt row gets that draft KV (user, tool and system turns, and past turns not served from the cache); only decoded tokens get the shifted embedding. To match serving, a trainer should give context rows the unshifted embedding (with the vision features from `img_embeds` at image rows) and span rows the shifted one. The dump records each chunk's boundaries and its `mm_embeds` flag for this | [source] |
| `mm_input_embeds` can be read after the forward. Without speculative decoding it is the same tensor the language model received, and Qwen4ExpModel never writes it in place: its first layer concatenates it into the 4 streams | [source] |
| In this version `--disable-cuda-graph` is a deprecated alias that turns off both decode and **prefill** CUDA graphs. Prefill graphs exist here (breakable, tc_piecewise, full), and under one the Python hook would run only at capture time. The patch skips batches under capture and logs this once | [source, server_args.py] |
| The chat endpoint accepts `rid` and passes it to the scheduler, so the dump directories carry the driver's snapshot ids | [source] |
| Span share in maximal snapshots, measured with the real tokenizer on 9 snapshots of ar25, ft09, re86 and tu93: spans are 43-81% of each snapshot's prefill, 57% overall. 27% of those span rows are turns that an earlier snapshot already holds. Skipping them gives **2.38 prefill tokens per loss row**, inside the 2.2-2.5 of 2.5 | [measured] |
| Driver dry run on Franzen's M2 demo logs (10 holdout games, 25 minutes): 22 snapshots, 527 loss turns, 184 covered occurrences skipped, no repeats or loop mentions, 2.26M prefill tokens. That gives about 0.92M loss rows and 1.06M kept rows, about 10.9 GB in FP8. On 4 games the loss-row estimate (from completion tokens) is within 3 tokens of the tokenized count: 374,709 against 374,706 | [measured] |
| **FP8 codec.** The numpy codec is bit-identical to torch 2.14 on 2M values. torch 2.14 saturates overflow to ±448 where older releases gave NaN; the dump clamps before converting, so both agree. On Gaussian rows the relative RMS error is 2.6% per row | [verified] |
| **One scale per row.** It stays accurate for streams within about 10^3× of each other, since e4m3 spans 2^-9 to 448. At 10^5× the small streams are lost while the row error still reads 3%. The target's final mixer normalizes each stream on its own (`hc_per_branch_norm`), so the index also reports `qerr_stream_max`. Real `H` has not been measured | [estimate] |

### 9.3 Running it in session A (not yet tested on a GPU)
1. **Inputs.** Franzen's notebook inputs, plus:
   - the request logs of the loop-free runs named in 2.6, attached as `kernel_sources` (resolve both mount layouts, lesson 0030);
   - the four scripts (the patch, the driver, `fidelity_probe.py`, `fidelity_sample.py`) in one folder, as `%%writefile` cells or a small dataset.
2. **Patch.** After his install cell, apply REAP first if REAP is served. Then run `python -I .../sglang_hc_dump_patch.py apply --site-packages SP` on the same site-packages. It exits non-zero on any mismatch.
3. **Storage test** (2.5). Write 10 GB to `/dev/shm` and to `/tmp`, choose the dump directory, and set `ARC3_HC_DUMP_MAX_GB` below what passed.
4. **Server.** Start it with cell 12's environment and arguments, with these changes:
   - add `ARC3_HC_DUMP=<dir>` to `env`;
   - drop the `--speculative-*` arguments;
   - add `--disable-cuda-graph` (or `--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled`) and `--disable-radix-cache`;
   - set `--max-running-requests 1` and `--chunked-prefill-size 8192`.

   Keep his `--chat-template` and `--default-chat-template-kwargs`. Keep `SGLANG_SM120_ONLINE_MXFP8` as he serves it, because the target's numerics must match serving. Flags that only make sense with a radix cache, such as `--mamba-radix-cache-strategy`, may need to go; check serve.log.
5. **Driver.** First a dry run: `python -I hc_dump_driver.py --logs <log dirs> --split train --dry-run`. Then the real run with `--out /kaggle/working/hc-run --dump-dir <dir> --max-dump-gb <cap> --max-minutes <budget>`.
6. **Checks.**
   - In `replay-summary.json`, `dump_ok` must equal `sent`.
   - Then read `qerr_stream_max` in the index, now on real `H`. If it is above about 0.05, dump again with `ARC3_HC_DUMP_SCALE_GROUPS=4`, which costs 6 bytes more per row.

### 9.4 What remains
- **The trainer (section 3):**
  - a reader of this format (`load_request` returns per-request arrays sorted by position);
  - the replica of the MTP block;
  - INT4 dequantization, chain masks and the KL loss;
  - CPU tests against HF 5.19.
- **The replica check of 5.2.** It needs a driver mode that is not built: the 16 probe requests and their recorded greedy outputs, dumped with `ARC3_HC_DUMP_KEEP=all` (BF16 advisable). The outputs have to be sent as a continuation, for example to `/generate` with `input_ids` and the images.
- **Export and gates.** Section 4 (export) and the gates of 5.3-5.5.
- **A builder option for session A.** Not written: `build_franzen_nb.py` was out of scope for this step, so the cells in 9.3 are manual.

### 9.5 Open questions
1. Should the trainer reproduce the unshifted draft embeddings for context rows in the prompt (9.2)? I'd say yes, because that is what serving does. The replica check decides.
2. One FP8 scale per row (as planned) or one per stream? Decide from `qerr_stream_max` in session A.
3. Duplicated turns. By default only the first snapshot, the generation-time context, carries a turn as loss. Also training on the post-trim copies (`--keep-duplicates`) would add about 36% more span rows on the demo logs; that is an ablation.
4. Which runs are both loop-free and lossless? The driver cannot tell from the logs, so the list of runs to replay must be chosen by hand.
