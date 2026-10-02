Summary: on Franzen's stack, decode is bound by reading the INT4 expert weights, so tokens/hour rises with tokens per step (more streams, higher MTP acceptance). The KV pool (1.01 M tokens, 95% full with 10 x 128 Ki streams) is what stops more streams. The cheapest win is to turn on Pennyroyal's online MXFP8, which Franzen left off (frees ~3.9 GiB and speeds the BF16 half of each step), and spend the memory on 12-13 streams: est. +20-28% decode tok/s, flags only. Next come a REAP-448 W4A16 checkpoint (16-18 streams, est. +30-40%, quality risk) and relaxed MTP acceptance (est. +8-15%, lossy). No stronger model fits the card before Qwen 4.

# Serving research for beating Tufa (written 2026-10-02, CPU container, nothing run on a GPU)

Scope: the five questions in the task (forks and upstream, speculative decoding, quantization and capacity, stronger
models, startup), answered against Daniel Franzen's Milestone 2 stack (our base since lesson 0026). Every number
is labelled **[measured: source]** or **[published: source]**. A number with no label is **[estimate]**, and the
estimate says how it was derived. Nothing here was run on a GPU, pushed, or uploaded. Clones and downloads are in
`/tmp/claude-0/-home-user-Arc-Agi-3-Kaggle-comp/d342458e-03bd-545b-8a6d-06bca061963e/scratchpad/serving-research/`
(`pennyroyal/` @ `12846e8`, `sglang-qwen38fn-sm120-turbo/` @ `c6cd506`, `sglang-flashnext-sm120/` @ `67d2f92`,
pulled Kaggle notebooks, `parse_serve.py` and `fit_model.py`, which reproduce every serve.log number below).

## 1. Measured facts from Franzen's run (serve.log, 2026-09-30, 10 public games, 25 min)

Source: `scratchpad/m2/franzen-output/serve.log` and the notebook log beside it. Parsed with
`serving-research/parse_serve.py`. This run is his 10-game demo (10 streams, 25 min per game), not a 9 h, 110-game
run, so contexts reach only ~120k late in the run.

| Quantity | Value |
|---|---|
| Server config (from `server_args`) | Pennyroyal 0.5.19+gd00d88efc8d6, Intel W4A16 AutoRound (Marlin MoE), albucino INT4-g32 MTP draft, NEXTN 3 steps / 4 draft tokens, FR-Spec 64k, lossless thresholds 1.0/1.0, FP8 KV, page 64, ctx 139,264, max-running 10, chunk 8192, mem-frac 0.96, mamba cache 60, `extra_buffer`, lpm schedule, online MXFP8 **off** |
| GPU memory | target weights 69.85 GB, draft 3.79 GB, KV 11.58 GB (target) + 0.96 GB (draft), Mamba 3.22 GB SSM + 0.13 GB conv, graphs ~1.1 GB, **4.25 GB still free after graph capture** |
| KV pool | 1,011,264 tokens (~12.4 KB/token incl. draft KV, i.e. ~80k tokens per GiB of VRAM) |
| Game requests | 537 in 1,495 s; prompt mean 63.6k, p90 103.7k, max 119k; output mean 1,742, p50 1,026, p90 4,110 |
| Prefix reuse | 34.14 M prompt tokens, 32.16 M cached (**94.2%**); 1.98 M new prefill tokens |
| Trim re-prefills | 12 requests above 40k with less than 50% cached, 653k tokens = 33% of all new prefill (mean 56k each) |
| Aggregate generation | **626 tok/s** over the span (935.5k output tokens); notebook summary says 588 tok/s of job wall-clock |
| Decode by running count (scheduler `gen throughput`, p50 / p90) | 5: 473 / 605; 6: 570 / 714; 7: 667 / 742; 8: 729 / 783; 9: 751 / 822; **10: 766 / 855**; max 946 |
| Per-request decode | p50 74.5 tok/s; **flat across context** (0-20k 73, 50-80k 75, 100-130k 76 tok/s): long history costs memory, not decode speed |
| MTP acceptance | mean accept length **2.66** of 4 (p10 2.48, p90 2.86); accept rate 0.55-0.72 |
| Slot occupancy | time-weighted mean 8.75 running of 10 (slots idle during tool calls); queue almost always 0 |
| KV pressure | `#full token` peaked at 965k = **0.95 of the pool** with 10 streams (872k at 10 running, 08:46) |
| Prefill speed | p50 10.7k tok/s on requests with more than 4k new tokens; ~13% of GPU time is prefill (1.98 M / ~10.6k tok/s / 1,495 s) |
| Output composition (request logs) | 86% of completion tokens are reasoning (796k of 925k); only 20.5% of 6-word shingles in tool-call code already occur in the context |

**Startup (notebook log + serve.log): 531 s from notebook start to `/health`:**

| Phase | Time |
|---|---|
| harness patch + wheel precache | 13 s |
| uv install (3 s) + the torch/GPU import probe | ~39 s total (13.6 → 52.9 s), ~33 s of it the probe |
| torch distributed init | 8 s |
| target weight load, 38 shards (185 GB dataset incl. 95 GB BF16 PLE; the precache thread read 185 GB in 235 s) | **270 s** |
| draft load + KV allocation | 3 s |
| target-verify CUDA graph capture | **145 s**: the first batch size (bs=10) took 139 s (one-off compile work: TileLang logs a data-race check and a kernel compile at its end; the rest is not itemised), the other six took ~6 s |
| draft graphs + warmup | 23 s |

**Where a decode step goes.** This is an estimate: a 3-parameter fit to the p75 throughput per running count n=5..10
(`fit_model.py`, rmse 0.4 ms). It gives step ≈ 13.5 ms + 0.75 ms·n + 20 ms·U(4n), where U(t) = 1-(1-10/512)^t is the
share of experts a step of t verify tokens touches. At n=10 the step is ~32 ms; the expert-read term grows
sub-linearly while tokens per step grow linearly. It agrees with gabriel's profile ([published]: at C1 the BF16
dense stack is ~85% of per-step bytes; NVFP4 MoE "at floor"). Implications:
- **More concurrent verify tokens are nearly free.** The fit predicts +10% decode tok/s at 12 streams, +15% at 13,
  +20% at 14, +28% at 16, +36% at 18 (decode phase only). External peaks agree: in Tong Hui Kang's table (forum
  744792) Franzen's 10 streams peak at 946 tok/s, Lord Han Solo's 14 at 1,135 (vLLM), sirikilohit's 16 at 1,159
  (Pennyroyal v2.5.0).
- **Raising acceptance length is proportional**, since tokens per step = n x accept length.
- **The constant 13.5 ms is mostly the BF16 dense stack** (GDN 4.17 GB, QSA 1.34, HyperConnection 1.32,
  shared experts 0.61, lm_head 1.27 GB [published: gabriel `docs/PERF_CEILING.md`]). Intel's checkpoint leaves
  exactly these in BF16: its `extra_config` keeps `linear_attn`, `self_attn`, `shared_expert`, `hyper_connection`,
  `indexer`, routers and lm_head at 16 bits (verified in its `config.json`). FP8 storage halves those bytes.

**Context vs streams.** The pool, not compute, limits streams. At Franzen's policy (trim at ~118k down to ~58k)
a stream averages ~87k resident tokens. With ~15% headroom plus 4-6 Mamba slots of 54 MiB each, one stream
needs ~1.5 GiB. 1.01 M tokens is exactly 10 streams. Running 12-16 streams on today's pool would make LRU evict
the prefixes of streams that are out on a tool call (KV is unlocked while a stream executes Python). Each eviction
costs a 60-118k re-prefill (6-11 s of GPU) and wipes out the gain. So **streams must be bought with memory**
(arms 1-3, 5) **or with shorter windows** (a harness choice: Franzen found longer context helps, and rellik13
credits more history with 14.5 → 22.5 [published: forum 744792]).

## 2. Forks and upstream since v2.5.3 (question 1)

| Source | State on 2026-10-02 | Speedups Franzen's build lacks |
|---|---|---|
| **Pennyroyal** `jpezzulli/sglang-rtxpro6000` (Apache-2.0) | No runtime release after v2.5.3 (`d00d88e`, 09-27). `v2.5.3-setup1` (`468e63a`) and branch head `12846e8` (09-28) change only setup scripts, docs and traffic reports (`git diff --stat d00d88e 12846e8`) | None new. But v2.5.0's **online MXFP8** (`SGLANG_SM120_ONLINE_MXFP8=true`) is already in his wheel and he sets it to 0. [published, RESULTS.md, NVFP4 checkpoint]: C1 +28% (161 → 207 tok/s), 128k C1 +26%, **128k C4 +15% (367 → 422)**, 490k C4 +39%, **+3.86 GiB free after graphs**. Also in the wheel: `--prefill-decode-interval`, `--kv-cache-dtype nvfp4`, HiCache |
| **mratsim** `sglang-qwen38fn-sm120-turbo` r24 `c6cd506` (09-17, Apache-2.0) | Patches 0001-0008 on day-0 image `4ccff141`; 0020-0025 add support for the local-inference-lab QAD checkpoint (ported from kanadaj) | [published, power-capped 360 W, memory overclocked +3000 MT/s, short prompts]: C4 800, **C8 1,170 tok/s** on RadixArk NVFP4 with online MXFP8 + `flashinfer_cutlass` MoE; TP2 QAD C16 1,600-1,800 with accept length 3.2-3.5. Not comparable to Kaggle clocks or our 87k contexts |
| **gabriel** `sglang-flashnext-sm120` `67d2f92` (08-31, **no licence**: ideas only) | Unchanged since 08-31 | Already in Franzen: 0004 low-M GEMM and the 0002 budget fix. Not in Franzen: 0005/0006 FP8 dense + lm_head (superseded by Pennyroyal's online MXFP8) and relaxed acceptance 0.3. [published]: C1 lossless 179 → 231 at 0.3, C4 620-657, C8 758 (all temp 0.6) |
| **Lord Han Solo** (Kaggle `lordhansolo/vllm-main-e975732-arc3` v3, Apache-2.0; LB 23.84) | vLLM 0.29.1rc1 nightly `e975732` + align-state retention, prompt-tail state caching, mixed-checkpoint PLE guard, pinned-host embed_tokens; model `primitive-ai/...mixed-NVFP4-FP8` + BF16 PLE | [published, forum 744792]: **KV pool 1,417,100 tokens at gpu-util 0.98** (1.90 GB left free), 14 streams x 147k, peak 1,135 tok/s, ready at 544 s. His notes say v3 is faster than the scoring runtime but has not beaten its score. README/VALIDATION not read: dataset downloads returned HTTP 429 all session. Switching engines would drop Franzen's SGLang prefix-cache patches; not recommended |
| **sirikilohit** (3rd in M2, LB 22.53) | Pennyroyal v2.5.0 build (`sirikilohit/sglang-penny-build-qwen`), Intel W4A16 + RadixArk FP8 PLE, NVFP4 MTP draft on `flashinfer_cutlass` | Flags (read from his notebook): mem 0.97, chunk 4096, mamba 80, 16 streams x 69,632, `--enable-hierarchical-cache --hicache-size 48 --hicache-write-policy write_through`, prefetch 16 threads + multithread load 8; [published]: pool 1,004,288, peak 1,159, ready at 615 s |
| **keithtyser** | Nothing newer than `qwen38-flash-next-vllm-nvfp4-runtime-v1` (08-30) | n/a |
| **Upstream SGLang** | 0.5.20 (09-18), 0.5.21 (10-01) | Flash-Next items are B200/GB300-centric: #40041 fused PLE gate for target verify, #41166 fused graph input copies, #34012 agentic tail-optimized LRU eviction. No sm120 numbers. Rebasing off Pennyroyal is not worth it before Nov 2 |
| **Upstream vLLM** | 0.30.0 (09-22), 0.31.0rc5 | #54890 FP8 QSA indexer cache, #54517 fused PLE kernels, #52228 adaptive verification, #54646 graph-capture GC freeze (12 s → 2 s on H200). No sm120 Flash-Next numbers |

## 3. Speculative decoding (question 2)

- **Draft length is capped at 4 tokens.** Pennyroyal's `qwen_sparse_attn_backend.py:_require_chain_speculation`
  raises if `speculative_num_draft_tokens` exceeds the QSA compress ratio. The ratio is 4: Intel config
  `indexer_compress_ratio: 4` and gabriel's dead-end note. topk must be 1, so there are no trees. The ceiling is 4
  tokens per verify, against today's 2.66. This also rules out DFlash/DSpark/EAGLE-3 trees (blocks of 7-16) unless
  someone rewrites the QSA index-key ring. No public Flash-Next DFlash or EAGLE-3 drafter exists (HF search
  2026-10-02).
- **Acceptance seen elsewhere** [published]:

  | Setup | Accept length |
  |---|---|
  | Pennyroyal native NEXTN | 2.58 (52.7%) |
  | WonderRico Automation Bench, online FP8 | 2.75-3.18 |
  | gabriel greedy lossless | 2.08-2.22 |
  | gabriel relaxed 0.3 | 2.5-3.0 |
  | mratsim QAD checkpoint, TP2 | 3.2-3.5 |
  | Franzen on ARC [measured] | 2.66 |

- **Relaxed acceptance** (`--speculative-accept-threshold-single/acc` below 1.0) is the only lever on accept
  length that is just a flag. It is lossy at temperature above 0: a draft token whose target probability exceeds
  the threshold is always accepted, which sharpens sampling toward the MTP draft.
- **N-gram / lookahead / suffix decoding: not worth it here.** 86% of output tokens are reasoning, and only ~20%
  of tool-code 6-grams repeat from context [measured], so at most a few percent of tokens are copyable. It also
  cannot be combined with NEXTN under the 4-token QSA cap.
- **FR-Spec** is already on (64k hot tokens). An ARC-specific token map would save at most the 64k-row draft
  lm_head read (~0.3 GB x 3 draft steps ≈ 0.5 ms, under 2% of a step). Low value.
- **Draft precision:** albucino's INT4-g32 MTP experts against an NVFP4 or BF16 MTP. No published acceptance
  difference. It would cost +1.4 GB for BF16. Low priority.

## 4. Quantization and capacity (question 3)

| Option | VRAM effect | Speed effect | Quality | Evidence |
|---|---|---|---|---|
| W4A16 (Intel, now) vs NVFP4 (RadixArk/NVIDIA) | W4A16 is ~5.6 GB smaller [estimate from 4.125 vs 4.5 bits on 120.8B expert params] | NVFP4 prefill faster (FP4 tensor cores: Pennyroyal 14.8k tok/s cold 64k vs Franzen's ~10.6k), decode similar | similar | [published] Franzen: "similar quality and throughput, more VRAM for KV". Keep W4A16 |
| Online MXFP8 of the BF16 dense stack | **frees ~3.86 GiB** | halves ~8.7 GB of per-step BF16 reads (est. 2.5-3.5 ms of the ~32 ms step) | small (MXFP8 weights + dynamic MXFP8 activations; the LIL QAD release ships MXFP8 attention by design) | [published] Pennyroyal v2.5.0 table above; [verified in source] works on any BF16 `UnquantizedLinearMethod` linear, which is what AutoRound's 16-bit `extra_config` layers get (`auto_round.py:336-342`); lm_head and HyperConnection become row-wise FP8; `tie_word_embeddings` is false |
| mem-fraction 0.96 → 0.975 | +~1.4 GiB | none | none | [published] LHS 0.98 profiled, sirikilohit 0.97 + chunk 4096, mratsim 0.975 ("images and JIT need spare VRAM"); Franzen has 4.25 GB unused after graphs |
| REAP-448 experts | −7.25 GiB for W4A16 (64/512 of ~58 GiB of experts); NVFP4 card says −7.9 GiB | ~equal per step (slightly fewer bytes) | calibration-dependent; REAP-288 kept coding but lost knowledge (dev.to "expert-pruning trap"); Son Pham's REAP-384 cost ~8% at equal lanes (19.7 vs 21.4, one run each) and gained a lot from 16 lanes (33.9) | [published] `lee-chang-93/Qwen3.8-Flash-Next-NVFP4-REAP-k448` (Kaggle `boristown/...-reap-448e`; its "~20 GiB" subtitle is wrong, the HF card says 7.9 GiB), Son Pham chart (our sglang-serving-plan.md §1) |
| FP4 KV (`--kv-cache-dtype nvfp4`) | ~2x KV tokens | similar | xz reports "very little NLL penalty" (forum 744792, short contexts, runtime unknown) | The flag exists in Pennyroyal, but the QSA path is FP8-specific: Triton packed sparse gather + `trtllm_decode` + sparse prefill. Expect boot failure or kernel work |
| BF16 PLE → FP8 PLE | none on GPU; −47.7 GB host RAM; −48 GB to read at startup | none | Franzen kept BF16 deliberately; sirikilohit and RadixArk use FP8 | albucino / RadixArk FP8 tables; Franzen's launcher accepts either |
| 3 bpw experts (klee100 AutoRound) | −15 GB | faster MoE | **KLD 0.12, "arithmetic consistency issue"** | [published] model card. Rejected |

**Streams that fit at Franzen's window** (~1.5 GiB per stream, as derived in section 1) [estimate]:

| Configuration | Pool tokens | Streams |
|---|---|---|
| Today | 1.01 M | 10 |
| + online MXFP8 | ~1.32 M | 12 |
| + mem-frac 0.975 | ~1.43 M | 13 |
| + REAP-448 | ~2.0 M | 17-18 |
| + FP4 KV instead of REAP | ~2.6 M | ~20 |

More streams help aggregate throughput (section 1 fit and the external peaks). At constant memory they hurt cache
residency. Mamba cache must stay at ~6 x max-running: gabriel reports that below that the speculative graphs
silently cap at bs 4. Each extra slot is 54 MiB, already counted above.

## 5. Stronger models (question 4)

- **Nothing stronger fits.** Checked on the HF API, 2026-10-02:
  - GLM-5.3-Flash: 321B parameters (NVFP4 ~185 GB).
  - DeepSeek-V4.1-Flash: 763B.
  - DeepSeek-V4-Flash-Vision-Exp: 305B.
  - MiniMax-H3: a video model.
  - Qwen-AgentWorld-35B-A3B: smaller than Flash-Next.
  - Son Pham (forum 742788): DeepSeek Flash after pruning and quantization "the token efficiency just wasn't equal
    to Flash Next"; two others report slow decode and broken vision.
  - **Qwen 4** is "in training, very soon" (Qwen lead, 09-22; a prediction market gives 74% before Nov 1). Day-0
    sm120 support for Flash-Next took about a week of community forks, so a late-October Qwen 4 is a contingency
    plan, not a planned arm. Keep the build pipeline ready.
- **Flash-Next derivatives** that run on the current SGLang path:
  - `ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AutoRound` (09-22), the same format as ours. [published]
    31-63% fewer thinking tokens on GPQA/MMLU-Pro/LCB/AIME, but **+11.9% total output on Terminal-Bench 2.1**
    (agentic). HelixML pilot: 1.40x end-to-end on two AIME items. Licence: Swift Open License v1.0 (free under
    US$1 M revenue) + Qwen Community. **Check prize eligibility before use** (the rules want an open-source
    licence; see forum thread 745079 on the Qwen licence). Lord Han Solo and michaelpoluektov mirrored the NVFP4
    build on Kaggle, so others are testing it.
  - `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (QAD, updated 10-02): [published] AA-LCR 79.4 vs 77.5 NVFP4-PTQ,
    GPQA-D 89.9 with 9% fewer tokens, MXFP8 attention, NVFP4 PLE (~24 GB host). It needs mratsim's 0020-0025 loader
    patches on SGLang (not in Pennyroyal). Its quality relative to Intel W4A16 is unknown.
  - `huikang/qwen-38-flash-next-finetune` v15 (Tong Hui Kang, Apache-2.0): a LoRA merged into RadixArk NVFP4. No
    published gain; his LB is 20.53.
- On this evidence, model quality is not the cheapest lever before Nov 2. With a same-notebook LB SD of 3.93
  (Son Pham, forum 745062), a model swap must clear about 8 points to be visible on the LB. Local public-25
  repeats are required.

## 6. Startup (question 5)

Today it is 531 s, 1.6% of 9 h. Achievable cuts [estimate]:
1. **Warm JIT caches**: up to −~2 min, if the 139 s first-capture time is cacheable compile work. Verify this by
   booting twice on the local box with the same cache directory.
   - Runtime Triton compiles also stall serving ("`_fwd_kernel` took 1.05 s to compile after serving started").
   - Point TileLang's cache (env `TILELANG_CACHE_DIR`; verify the name for the bundled version), Triton,
     FlashInfer and SGLang JIT dirs at one directory.
   - Tar it into `/kaggle/working` after a run on the Kaggle image, attach it as a small private dataset, and
     untar it before launch. A cache miss just recompiles, so it fails safe.
2. **FP8 PLE**: −~60-70 s, since 48 GB less is read at ~0.7-0.8 GB/s, and +48 GB host RAM.
   - Assemble the Intel target with the RadixArk/albucino `model-plefp8-*` shards and set
     `text_config.ple_embedding_dtype=float8_e4m3fn` (sirikilohit's "M82" cell does exactly this).
   - Needs the FP8 PLE shards as a Kaggle input: keithtyser's RadixArk model, or Son Pham's
     `sonphamorg/arc3-flashnext-ple-fp8-v1`.
   - Quality: untested here.
3. **Drop the torch import probe** in the launcher: −~30 s.

Together this is up to ~3.5 min, about +0.7% play time. Small, but free.

## 7. Ranked serving arms

Each arm builds on the previous one unless it says otherwise. Every arm first passes the replay benchmark
(section 9) on the local box, then gets repeated public-25 runs against the unchanged Franzen copy (the exp-070/071
baselines). Gains are in decode tok/s at Franzen's shape unless stated.

### Arm 1. Online MXFP8 + spend the memory on 12-13 streams (flags only)
- **Change** (notebook cell 4 env and cell 12 `CFG`, same `dfranzen/pennyroyal-v253` bundle, no rebuild):
  - `SGLANG_SM120_ONLINE_MXFP8=true` in cell 12's `env.update(...)`. It hard-codes `"0"` today, as does the bundle's
    `runtime_env.sh`.
  - Step a: `MAXREQ=12`, `CUDAGRAPH_MAXBS=12`, `MAMBA_CACHE=72`, `ARC3_MAX_ACTIVE_STREAMS=12`.
  - Step b, with arm 2: 13 / 13 / 78 / 13.
  - `graph_bs` then pads 11 to 12. Optionally add 11 and 12 to the set.
  - Boot-log checks: `Flash-Next online MXFP8 projection ready` lines; `max_total_num_tokens` ≥ ~1.30 M; `avail mem`
    after graphs.
- **Package:** nothing new.
- **Expected:**
  - [published] +15% C4 at 128k and +3.86 GiB on the NVFP4 checkpoint.
  - [estimate] +5-8% per step at n=10 from halving ~8.7 GB of BF16 reads; +10-15% from 12-13 streams (fit).
    **Combined +20-28% decode**, ~+15-25% generated tokens per hour after prefill and slot idle.
- **Risk:**
  - Franzen turned it off for an unknown reason: possibly the builder's default, possibly a failure he hit.
  - First use with AutoRound + Marlin MoE + the albucino draft. MXFP8 conversion of the GDN/QSA projections is a
    numerics change.
  - Pennyroyal's quoted-tool-markup probe was 7/10 with online FP8. That is not isolated to FP8, but watch the
    harness's malformed-call counters.
- **Cheapest test** (~45 GPU-min on the local box):
  - Boot and read the log (10 min).
  - Replay at n=10 (MXFP8 on vs off), then n=12. Pass: ≥ +5% tok/s at n=10, ≥ +12% at n=12, 0 retractions, cache
    hit ≥ 90%.
  - Then 2+ public-25 runs.

### Arm 2. Memory accounting: mem-fraction 0.975 (and chunk 4096 if needed)
- **Change:** `MEMFRAC=0.975`. If the stress replay OOMs, use `CHUNK=4096`, `MAX_PREFILL=8192`. Smaller chunks cost
  ~4% prefill speed because each chunk re-reads the experts.
- **Expected:** +~1.4 GiB ≈ +110k tokens ≈ +1 stream (12 → 13) [estimate].
  [published] LHS 0.98 (1.9 GB free), sirikilohit 0.97, mratsim 0.975.
- **Risk:** an OOM during an image or prefill spike kills the server. That would be catastrophic in a rerun, so
  keep Franzen's watchdog and startup fallbacks.
- **Cheapest test:** a 30-min replay at the arm 1 + 2 stream count with the largest prompts and images; read peak
  `nvidia-smi` and server errors.

### Arm 3. REAP-448 W4A16 checkpoint → 16-18 streams at 128k
- **Change:** a new checkpoint. Intel W4A16 sliced to 448 routed experts per layer; router rows sliced to match;
  `num_experts: 448`. Leave the MTP layer at 512: it is separate, and albucino's draft has its own config. Boot
  with arm 1 and arm 2 and `MAXREQ=16-18`, `MAMBA_CACHE=6xMAXREQ`.
- **Expert list:** either
  - (a) recover `lee-chang-93/...REAP-k448`'s kept indices by matching its BF16 router rows to RadixArk's (fetch only
    the router tensors with HTTP range reads), or
  - (b) better: our own **ARC-calibrated REAP**. Accumulate router-weighted expert-output norms per expert while
    replaying Franzen's request logs (~34 M prompt tokens with images, already on disk) through a hooked
    SGLang/HF forward on the local GPU. REAP keeps what its calibration data uses, and this calibration data is
    our task.
- **Package:** a private Kaggle model holding only the re-sliced non-PLE shards (~62 GB). Assemble at runtime with
  symlinks to Franzen's PLE shards, as cihanatak's Swift notebook does. That needs the owner (dataset upload).
- **Expected:** −7.25 GiB → ~2.0 M-token pool → 16-18 streams.
  - [estimate] fit: +28-36% decode over today, more with arm 1's per-step gain.
  - [published, different harness] Son Pham's 384-expert run reached 33.9 at 16 lanes vs 21.4 unpruned at 7.
- **Risk:** quality. The REAP-288 knowledge collapse shows uncovered skills vanish; calibrate on image-bearing ARC
  prompts. SGLang may need the draft to keep 512 experts (vLLM needs `VLLM_MTP_NUM_EXPERTS=512`). About 1-2 days of
  CPU slicing plus ~2 GPU-hours of calibration.
- **Cheapest test:** route-only statistics first. Log top-10 router choices over the replay and measure how much
  routed weight the 64 dropped experts carry per layer (REAP-k448's card retained 92.65% of REAP mass). Then boot,
  replay, and run 3 public-25 runs at 16 streams against arm 1 at 12-13.

### Arm 4. Relaxed MTP acceptance (flags, lossy)
- **Change:** `SPEC_ACCEPT_SINGLE=SPEC_ACCEPT_ACC=0.5`, then 0.3 (cell 12 `CFG`).
- **Expected:**
  - [published, gabriel C1, temp 0.6]: lossless 179 → 0.5: 203 → 0.3: 231 tok/s; accept length 2.5-3.0.
  - [estimate] at n=10, tokens per step scale with accept length: 2.66 → ~2.9-3.1 ≈ +8-15%.
- **Risk:** distribution shift. Confident tokens are taken from the draft, roughly a lower effective temperature.
  Effect on exploration is unknown; it must be judged on score, not tok/s.
- **Cheapest test:** a replay gives accept length and tok/s in 25 min. Score needs ≥3 public-25 runs per arm,
  given the run-to-run spread.

### Arm 5. FP4 KV cache (high upside, likely kernel work)
- **Change:** `KVDTYPE="nvfp4"` (and `--speculative-draft-kv-cache-dtype`).
- **Expected:** ~2x pool → ~20 streams or 256k windows [estimate]; xz claims a small NLL penalty [published, forum].
- **Risk:** high. QSA's packed sparse gather, `trtllm_decode` and the Triton sparse prefill handle FP8 explicitly;
  FP4 probably fails at boot or produces garbage. Making it work is kernel work of days or more, then needle and NLL
  validation at 60-120k.
- **Cheapest test:** one 10-min boot attempt. If it serves, a 64k/110k needle test plus a replay accept-length
  comparison. Do not invest more unless arms 1-3 fall short.

### Arm 6. Startup cuts (section 6)
- **Expected:** up to −~3.5 min, about +0.7% play time.
- **Package:** a JIT-cache dataset (≤1 GB), and the FP8 PLE shards (an existing public Kaggle copy).
- **Risk:** low. A cache miss recompiles. FP8 PLE is a numerics change.
- **Cheapest test:** one Kaggle save comparing READY seconds.

### Arm 7 (secondary). Model swaps on the same stack
Swift-1.5 W4A16-AutoRound (licence check first), LIL QAD NVFP4 (needs loader patches), THK LoRA. Each is
quality-uncertain and needs ≥3 public-25 runs to see. Do these only after arms 1-4 are settled.

## 8. Not recommended (and why)

- Deeper MTP, DFlash/DSpark, EAGLE-3 trees: blocked by the QSA 4-token verify window. There are also no drafters.
- N-gram / suffix speculation: little copyable text (86% reasoning). It does not compose with NEXTN.
- Switching to vLLM (LHS) or upstream SGLang 0.5.21: loses Franzen's tested prefix-cache patches. No sm120
  Flash-Next numbers favour either.
- HiCache host tier: Franzen re-queues a game only at a trim, which already invalidates its prefix. HiCache helps
  only if games are parked mid-cycle. Revisit if arm 3 changes admission.
- 3 bpw experts (KLD 0.12); GLM-5.3-Flash / DeepSeek V4.x (do not fit); TensorRT-LLM (no hybrid QSA support
  evidence); FP8 SSM state (recurrent precision risk).
- Spec steps 2 vs 3: the fit says neutral (fewer verify tokens vs lower accept). Sweep only if spare time.

## 9. The test bench every arm needs: request-log replay (CPU work to build, ~25 GPU-min per arm)

Franzen's output already has the inputs: `*_p0_requests.jsonl` for 10 games, 531 responses with full messages,
tools, `chat_template_kwargs` and usage.
- **Streams.** Run N concurrent "game streams". Each replays one game's requests in order, sending request i+1
  after i returns plus the logged tool latency.
- **Fixed outputs.** Set `max_tokens` = the logged `completion_tokens` and `ignore_eos: true`, so outputs are fixed
  and arms compare tokens/s rather than behaviour.
- **More than 10 streams.** Clone games with a per-stream nonce at the very start of the system prompt. That keeps
  prefixes unique, as in the real run.
- **Record** `gen throughput`, accept length, `#full token` peak, retractions, cache hit, prefill seconds, and
  per-request decode. Use `parse_serve.py` on the server log.
- **Throughput only.** This is a throughput gate. Score effects (arms 3, 4, 7) still need public-25 runs, read
  through the 1.5-2.3x same-code spread.

Take `flock /tmp/arc-gpu.lock` for every boot.

## Sources

- Franzen: `/home/user/da-fr/arc-agi-3-solution` (WRITEUP §1-2, `serving/README.md`,
  `serving/build_bundle_pennyroyal.sh`); `kaggle/franzen/arc-agi-3-milestone-2-solution.ipynb` cells 4/6/12/16;
  `scratchpad/m2/franzen-output/{serve.log, arc-agi-3-milestone-2-solution.log, *_p0_requests.jsonl, summary.txt}`.
- Pennyroyal `RESULTS.md`, `FP8.md`, `CHANGES.md`, `COMMUNITY-RESULTS.md`; source files
  `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`, `.../quantization/auto_round.py`,
  `.../models/qwen4_exp.py`, `.../kernels/ops/gemm/sm120_online_fp8.py`, `.../server_args.py`.
- mratsim README and launchers; gabriel `README.md`, `docs/PERF_CEILING.md`, `docs/STATUS.md`,
  `results/results_8h_final.json`.
- Kaggle:
  - forum 744792 (M2 comparison table + comments), 745062 (LB SD 3.93), 742788 (new models), 742801;
  - notebooks `lordhansolo/built-on-tufa-labs-duck-harness-milestone-2`, `sirikilohit/arc-agi-3-duck-18-1gc-submit`,
    `cihanatak/arc3-b32-reap-v030-kv14-c12-ctx48-r1`;
  - models `boristown/qwen3-8-flash-next-nvfp4-reap-448e`, `huikang/qwen-38-flash-next-finetune`,
    `nvidia/qwen3-8-flash-next-nvfp4`.
- Hugging Face cards/API:
  - [Intel W4A16 config](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound),
    [albucino](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE);
  - [LIL QAD](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4),
    [REAP-k448](https://huggingface.co/lee-chang-93/Qwen3.8-Flash-Next-NVFP4-REAP-k448);
  - [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4),
    [klee100 3bpw](https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP),
    [todiadiyatmo Attn8](https://huggingface.co/todiadiyatmo/Qwen3.8-Flash-Next-W4A16-Attn8-FP8PLE).
- Web:
  - [HelixML Swift pilot](https://helix.ml/blog/swift-flash-next-four-gpu-evaluation),
    [expert-pruning trap](https://dev.to/kiarina/running-qwen38-flash-next-on-a-128-gb-mac-the-expert-pruning-trap-and-a-memory-mapped-n-gram-182m);
  - [SGLang v0.5.21](https://github.com/sgl-project/sglang/releases/tag/v0.5.21),
    [v0.5.20](https://github.com/sgl-project/sglang/releases/tag/v0.5.20),
    [vLLM v0.30.0](https://github.com/vllm-project/vllm/releases/tag/v0.30.0);
  - [Qwen 4 status](https://www.yottalabs.ai/post/qwen-4-release-date-what-is-known-how-to-prepare-2026).
- Not obtained: Lord Han Solo's write-up (JS-only page) and his runtime README/VALIDATION (Kaggle downloads
  returned 429 all session). His figures above come from Tong Hui Kang's table.
