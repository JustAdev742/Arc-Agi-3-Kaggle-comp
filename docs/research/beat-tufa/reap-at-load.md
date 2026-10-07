Summary: REAP-448 can be applied to Franzen's unchanged Intel W4A16 checkpoint at load time. The public REAP-k448 build's kept experts were recovered exactly (all 48 x 448 router rows match Intel's bit for bit). A 13-line anchored patch to the installed Pennyroyal sglang plus two server flags (`--reap-kept` in scripts/build_franzen_nb.py) drops the other 64 experts per layer, freeing 7.31 GiB, which buys 16 streams at today's pool pressure. Nothing has run on a GPU yet.

# REAP-448 at load time (written 2026-10-07, CPU container, nothing run on a GPU)

serving.md arm 3 assumed a new 62 GB checkpoint, which we cannot build or ship. This arm keeps Franzen's checkpoint
and prunes while loading. Labels: **[verified]** = checked in this session against the real artifact (command
below), **[source]** = read in the Pennyroyal wheel's Python sources but not executed, **[estimate]** = derived.

## 1. What was done

| Deliverable | File | Status |
|---|---|---|
| Kept experts per layer | `kaggle/franzen/reap448_kept_experts.json` (`{"<layer>": [448 sorted original ids]}`, sha256 `e7e6a28b27b1...`), provenance and router fingerprints in `reap448_kept_experts.meta.json` | [verified] exact |
| Recovery script | `scripts/reap_kept_experts.py` (HTTP range reads of the router tensors only) | ran once, 2 min, 307 MB read |
| Loader patch (runtime module + installer) | `scripts/sglang_reap_patch.py` | [verified] on the wheel's file and on real router data with CPU torch; [source] for the GPU load path |
| Notebook option | `scripts/build_franzen_nb.py --reap-kept FILE` | [verified] builds; its cell-12 steps executed in tests |
| Tests | `tests/test_sglang_reap_patch.py`, 5 new tests in `tests/test_build_franzen_nb.py` | 38 pass, 1 skip (torch absent from .venv) |
| Gate arm (built, not pushed) | `scratchpad/franzen/exp072e/arc3-dprime-gate-reap448-mxfp8-r16.ipynb` | differs from exp-072c only by the REAP cells and 14 -> 16 streams |

## 2. The kept experts

**Source of truth.** The public build `lee-chang-93/Qwen3.8-Flash-Next-NVFP4-REAP-k448` (revision `8d565c90`) ships no
kept-index metadata: its config.json says only `num_experts: 448`, hf_quant_config.json and README have no list, and
`sitecustomize.py` is a vLLM MTP hook. Its card says the routers are "bit-exact rows of the original" and the
experts are "renumbered 0..447".

**Method** (`scripts/reap_kept_experts.py`). For each layer, read the safetensors header and then only the bytes of
`model.language_model.layers.{i}.mlp.gate.weight` from both the pruned build and Intel's
`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` (revision `4c67bf68`, the checkpoint Franzen loads; BF16 routers). Then
map every pruned row to the unique unpruned row with identical bytes.

**Result [verified]:**
- In all 48 layers, each of the 448 pruned rows matches exactly one of Intel's 512 rows, bitwise, with 448 distinct
  ids.
- In every layer the pruned build lists its kept experts in ascending original-id order. So "slot = rank in the
  sorted list", which the patch uses, is the build's own numbering.
- The MTP router (`mtp.layers.0.mlp.gate.weight`, 512 x 2560) is identical in both checkpoints, consistent with the
  card's "MTP untouched".
- Bytes read: 148 MB (pruned) + 159 MB (Intel), headers included. No shard was downloaded.

**Same checkpoint as Franzen's Kaggle copy.** Franzen's Kaggle model `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround`
has the same non-PLE shard names and byte sizes as the HF revision; only the 102 GB PLE shard is resharded into 22
files. The CLI cannot download single files from a Kaggle model, so this is not checked by hash. The patch enforces
it at load instead: the meta file holds each layer's sha256 of the full 512-row router, and the server refuses to
start if a loaded router differs.

**Routed-weight share** [published, model card, not reproducible here: it needs activations]:
- 92.65% of REAP saliency mass retained; mean 7.35% removed per layer, worst layer 15 at 9.11%, for a 12.5% cut in
  expert count.
- Calibration: 16.49 M tokens of the model's own agentic production traffic, including the multimodal path (626
  images).
- 28 dead experts were pruned first and 5 "super experts" force-kept.
- The calibration is not ARC traffic. It was measured on NVIDIA's NVFP4 quantization, and Intel's W4A16 shares the
  same experts and router.

## 3. The loader patch

All code references are to the Pennyroyal wheel `sglang-0.5.19+gd00d88efc8d6` from the dataset `dfranzen/pennyroyal-v253`
(wheel sha256 `d0620216...`, downloaded alone, 24.8 MB). The source clone at `12846e8` differs from the wheel in 12
files under `srt/`. Those include `model_loader/loader.py`, `model_loader/weight_utils.py`,
`startup_weight_load.py`, `gptq/schemes/gptq_moe.py` and `server_args.py`, so everything below was read in the wheel.
`models/qwen4_exp.py` is identical in both.

**How the model is built [source]:**
- `Qwen4ExpForConditionalGeneration` (qwen4_exp.py:1930) takes `self.config = config.text_config` (qwen3_vl.py:1291).
- Every one of its 48 layers builds `Qwen2MoeSparseMoeBlock` (qwen3_5.py:867/1125), which sizes three things from
  `config.num_experts`:
  - `FusedMoE(num_experts=...)`;
  - the router `gate = ReplicatedLinear(hidden, config.num_experts)` (qwen2_moe.py:323-352);
  - `TopK(top_k=10, renormalize=norm_topk_prob)`, where `norm_topk_prob` defaults to True (configs/qwen3_next.py:214)
    and Intel's config does not set it.
- Shared-expert fusion needs aiter or `enable_cuda_shared_expert_fusion`, which Qwen3.5 layers do not pass. So the
  shared expert stays a separate MLP and is untouched.
- The target's MoE method is `GPTQMarlinMoEMethod` (serve.log); its weights are per-expert `[E, ...]` parameters.

**How the weights arrive [source]:**
- The checkpoint names experts `model.language_model.layers.{i}.mlp.experts.{j}.{gate,up,down}_proj.{qweight,qzeros,scales}`
  (9 tensors per expert, 2,553,600 B, [verified] from Intel's headers).
- `load_weights` (qwen4_exp.py:2028) maps names through
  `FusedMoE.make_expert_params_mapping(num_experts=config.num_experts)`. It then calls
  `weight_loader(param, w, name, shard_id, expert_id=j)`, which silently drops ids outside the local range
  (fused_moe_triton/layer.py:906-946).
- The router goes through the generic path, whose loader requires the exact shape.
- Franzen's server uses `startup_weight_load_mode: serial`; speculative decoding rules out the overlap mode
  (startup_weight_load.py:359). So `DefaultModelLoader` calls `model.load_weights(iterator)` once with the whole
  stream (loader.py:963-1015).

**The patch (two anchored edits, active only with `ARC3_REAP_KEPT_EXPERTS` set):**
1. The first statement of `Qwen4ExpForConditionalGeneration.load_weights` becomes
   `weights = arc3_reap.wrap_weights(weights, self.config)`. The generator does the following:
   - renames `experts.{j}.` to `experts.{slot}.` for kept `j` and drops pruned `j`;
   - replaces each router with `w.index_select(0, kept)`;
   - passes everything else through, including the `mtp.*` tensors, which the target skips itself.
   It also checks, and raises on failure:
   - before loading: `num_experts == 448` and 48 layers;
   - per router: more rows than the largest kept id, and the full router's sha256 equal to the meta file;
   - at the end of the stream: every layer had a router, 448 kept slots and 64 pruned experts, each with the same
     tensor set.
   It then logs one `ARC3 REAP: kept 448 of 512 ...` line.
2. `get_model_config_for_expert_location` returns None, so there is no global expert-location map. The MTP draft
   still has 512 experts and its MoE layer has `layer_id` 0. Its `FusedMoE.weight_loader` looks ids up in the
   *target's* global map (`logical_to_all_physical`, eplb/expert_location.py:347-378). The draft worker never builds
   its own map (model_runner.py:692-695). A 448-wide map would raise IndexError for draft experts 448-511. With no
   map, both models take the loader's `metadata is None` branch, which yields the same local ids at EP 1. Other
   consumers of the map in this configuration:
   - `ExpertLocationDispatchInfo`: DeepEP or mega-MoE only;
   - the distribution recorder: mode None, a no-op;
   - EPLB/LPLB/elastic EP: off (serve.log server_args).

**Installed how.** `python -I scripts/sglang_reap_patch.py apply --site-packages SP --kept kept.json` does the
following:
- copies itself to `SP/sglang/srt/arc3_reap.py` (a namespace package, imported as the wheel imports
  `sglang.srt.platforms`);
- refuses the edit if `qwen4_exp.py`'s sha256 is not `35a1785c...` (the analysed file) or an anchor is not found
  exactly once;
- compiles the result, removes the stale `.pyc`, and is a no-op on a tree it already patched;
- exits 1 on any failure.

The notebook cell calls it with the launcher's own `run(..., check=True)`, so a failure raises in cell 12 before the
server starts.

**Model size and the draft.** These are server flags, not code:
- `--json-model-override-args '{"text_config": {"num_experts": 448}}'`. Pennyroyal merges a dict into the
  `text_config` sub-config (utils/hf_transformers/config.py:266-275).
- `--speculative-draft-model-override-args '{}'` is required. When it is unset, the draft inherits
  `--json-model-override-args` (configs/model_config.py:612-620, arg_groups/speculative_hook.py:20-27) and would be
  built 448 wide against its 512-expert checkpoint.
- The albucino draft's own config says `num_experts: 512` [verified, HF copy]. It is `Qwen4ExpForCausalLMMTP`,
  loading through `Qwen3_5ForCausalLMMTP.load_weights`, which the patch does not touch.

**What assumes 512 [source]:**

| Component | Finding |
|---|---|
| Top-k routing (CUDA) | Triton `moe_fused_gate` (topk.py:878-890) pads the expert dimension to `next_power_of_2(448)` = 512 with masking (moe_fused_gate.py:349). Same block size as today. With `norm_topk_prob` True, the top-10 among the kept experts is renormalised, which equals masking the pruned ones. The JIT CUDA `topk_softmax` has a non-power-of-two path with a workspace, but it is not on the CUDA route. |
| `moe_align_block_size` | Generic up to 4,096 experts (moe_align_kernel.cu). |
| Marlin MoE | `E = w1.shape[0]`. Only the `block_size_m` heuristic uses E (`M*topk/E`), so its choice can shift slightly. The kernel has no expert-count constraint. |
| Router GEMM | `n=448, k=2560` (2.3 MB) is above the 512 KB threshold of Pennyroyal's low-M BF16 Triton GEMM (m <= 32). 448 % 16 = 0, so N tiles are full. |
| Online MXFP8 | The converter skips any child named `gate` and any FusedMoE (sm120_online_fp8.py:236), so the router stays BF16, as today. |
| AutoRound config | No target-expert entries in `extra_config`. FusedMoE gets the default 4-bit g128, the router stays 16-bit via `.*mlp\.gate.*`. Renumbering changes neither. |
| EP / `num_local_experts` | EP 1, `ep_num_redundant_experts` 0. FusedMoE computes 448 local experts. |
| CUDA graphs | Shape-only. Capture sizes come from `MAXREQ` / `CUDAGRAPH_MAXBS`. |
| FR-Spec | The hot-token map and the draft vocabulary are independent of experts. The "late json_model_override_args" comment in eagle_draft_extend_cuda_graph_runner.py:222 is stale: no code writes it. |
| Shared expert, `shared_expert_gate` | Separate modules, untouched. |
| MTP layer and its 512 experts | Separate draft model, untouched (see above). |

## 4. Memory estimate (MXFP8 on, `MAMBA_CACHE = 6 x MAXREQ`, mem-fraction 0.96 unless noted) [estimate]

Inputs from Franzen's serve.log (2026-09-30):
- KV pool 1,011,264 tokens in 11.58 + 0.96 GiB, i.e. 80.6k tokens per GiB.
- Mamba 3.40 GiB for 60 slots, i.e. 0.0567 GiB per slot.
- Peak `#full token` 965k at 10 streams, i.e. 96.5k per stream, 95% of the pool.

Freed by this arm:
- REAP: 7.31 GiB. This is 64 x 48 x 2,553,600 B from Intel's headers, of 58.45 GiB of routed experts. The g_idx
  buffers that the Marlin repack frees anyway are not counted.
- Online MXFP8: 3.86 GiB [published, Pennyroyal on the NVFP4 checkpoint; exp-072b/c measure it on W4A16].

Freed memory goes into the static pool one for one.

| Streams | Mamba slots (+GiB vs 60) | KV pool | Peak need (x 96.5k) | Pool use | Pool @ MEMFRAC 0.975 | Use | Decode fit vs today |
|---|---|---|---|---|---|---|---|
| 10 (today) | 60 | 1.011 M | 0.965 M | 95% | | | 0 |
| 14 | 84 (+1.36) | 1.80 M | 1.35 M | 75% | 1.92 M | 70% | +22% (+33% with MXFP8's -3 ms) |
| **16** | 96 (+2.04) | 1.75 M | 1.54 M | **88%** | 1.86 M | 83% | +31% (+42%) |
| 18 | 108 (+2.72) | 1.69 M | 1.74 M | 103% | 1.81 M | **96%** | +40% (+51%) |

Decode fit: serving.md's step model, 13.5 ms + 0.75 ms·n + 20 ms·(448/512)·U'(4n), with
U'(t) = 1-(1-10/448)^t. Pruning makes each step read slightly fewer expert bytes. At equal streams, the fit gives
+22/+31/+40% with REAP against +20/+28/+36% without, so almost all of the gain comes from the extra streams the
memory buys. The second figure assumes MXFP8 removes 3 ms of the constant (serving.md: 2.5-3.5 ms).

For comparison, exp-072c (MXFP8 + 14 streams, no REAP) has a ~1.21 M pool against 1.35 M peak need (111%). REAP is
what makes 14-16 streams fit.

**Recommendation:** 16 streams at mem-fraction 0.96 (88%, below today's 95%). 18 streams only together with
`--cfg MEMFRAC=0.975` (serving.md arm 2; 96%, today's level).

Not estimated: CUDA-graph memory at batch size 16-18. Today the graphs take 1.14 GiB at bs <= 10 and 4.25 GiB stay
free after them. Read `available_gpu_mem` in the first run.

## 5. The gate arm (built, not pushed)

```bash
.venv/bin/python scripts/build_franzen_nb.py --base dprime --full25 25 \
  --out "$SCRATCH/franzen/exp072e" --slug arc3-dprime-gate-reap448-mxfp8-r16 \
  --reap-kept kaggle/franzen/reap448_kept_experts.json \
  --server-env SGLANG_SM120_ONLINE_MXFP8=true \
  --cfg MAXREQ=16 --cfg CUDAGRAPH_MAXBS=16 --cfg MAMBA_CACHE=96 --env ARC3_MAX_ACTIVE_STREAMS=16 \
  --note "exp-072e: serving gate on D', REAP-448 at load + online MXFP8 + 16 streams (docs/research/beat-tufa/reap-at-load.md)"
```

`$SCRATCH` is the session scratchpad; this was run once and the output is in `scratchpad/franzen/exp072e/`. "072e" is
a proposed number.

Compared with the queued exp-072c notebook, the cells differ only in:
- the note;
- 14 -> 16 for `MAXREQ`, `CUDAGRAPH_MAXBS`, `MAMBA_CACHE` and `ARC3_MAX_ACTIVE_STREAMS`;
- three `%%writefile` cells before the launcher (`/kaggle/arc3-reap-patch.py`, `/kaggle/arc3-reap-kept.json`, its
  `.meta.json`);
- one block after `env.update({...})` (patch the installed sglang, set `ARC3_REAP_KEPT_EXPERTS`);
- one block before the prefetch flag (the two override flags).

Kernel metadata is identical apart from id, title and file: same inputs, Franzen's pinned GPU image. The 18-stream
variant adds `--cfg MEMFRAC=0.975` and uses 18/18/108/18. Run it against exp-072c, after exp-072b/c show that
online MXFP8 is stable on this checkpoint.

## 6. Risks

- **Quality (the main one).**
  - REAP-288 (44% pruned) kept coding but lost knowledge (serving.md section 4, "expert-pruning trap").
  - Son Pham's REAP-384 (25%) scored ~8% lower at equal lanes, but much higher with 16 lanes (one run each).
  - 448 is milder (12.5%, 7.35% of REAP mass), but the calibration is agentic production traffic, not ARC
    games. ARC needs vision, grid reasoning and code, and the card's 626 images are not ARC boards.
  - The gate measures throughput. Score needs repeated public-25 runs against the unpruned arm at the same stream
    count, read through the 1.5-2.3x same-code spread (lesson 0018).
- **MTP acceptance.** The 512-expert draft was trained on the unpruned target's hidden states. A pruned target
  shifts the token distribution a little, so the accept length (2.66 today) may drop. That would eat into the
  stream gain, which is proportional to it.
- **Numerics.** For tokens whose top-10 contains no pruned expert, routing is unchanged up to float rounding of the
  renormalisation. Tokens that used a pruned expert get the next-ranked kept expert instead; this is what REAP's
  quality numbers measure. No kernel numerics change: Marlin, the router GEMM and Triton routing are the same
  kernels with E=448.
- **Not verifiable without a GPU** (the first run must check these):
  - that Pennyroyal's CLI accepts `--speculative-draft-model-override-args '{}'`. The field exists
    (server_args.py:2171) and flags are named `"--" + field.replace("_", "-")` (arg_groups/arg_utils.py:216), but
    the flag was not run through the parser here. The draft sees the same `{}` as today, because Franzen's server
    has `json_model_override_args: '{}'` and the draft inherits it;
  - that the scheduler subprocess inherits `ARC3_REAP_KEPT_EXPERTS` (it inherits the Popen environment like every
    other `SGLANG_*` variable the launcher sets);
  - the actual freed memory, KV pool and graph memory;
  - that the `ARC3 REAP` log line reaches serve.log (sglang configures the root logger at `info` with `force=True`);
  - that no other code path calls `get_model_config_for_expert_location` expecting a non-None value (none found by
    search).
- **Startup failure mode.** If the server refuses to start (a fingerprint mismatch, a check failure), Franzen's cell 12
  prints the log tail and the notebook goes on to the benchmark against a dead server. That is existing behaviour
  for any server failure. Check the first minutes of the run's log rather than waiting for its end.
- **Version lock.** The patch applies only to the analysed `qwen4_exp.py` (sha256 `35a1785c...`). A new Pennyroyal
  wheel stops the notebook at cell 12 until the loader is re-read; `--allow-other-version` exists for that review.

## 7. What the first GPU run must check

In the notebook log, cell 12:
- `arc3 REAP: .../qwen4_exp.py patched; ... 48 layers x 448 experts, router sha256 for every layer`.

In serve.log:
1. `server_args`:
   - `'json_model_override_args': '{"text_config": {"num_experts": 448}}'`;
   - `'speculative_draft_model_override_args': '{}'`;
   - `max_running_requests` 16, `max_mamba_cache_size` 96.
2. `ARC3 REAP: kept 448 of 512 routed experts in each of 48 layers (193536 expert tensors loaded, 27648 pruned tensors
   skipped); routers sliced to 448 rows, router sha256 verified`.
3. No `Parameter ... not found while loading Qwen4-Exp VL weights` warnings and no tracebacks.
4. `Load weight end ... type=Qwen4ExpForConditionalGeneration`: mem usage about 69.85 - 7.3 GiB, and about 3.9 GiB
   less again with MXFP8 (~58.7 GiB).
5. The draft's `Load weight end ... type=Qwen4ExpForCausalLMMTP ... mem usage=3.79 GB` unchanged. This is the proof
   that the draft kept 512 experts; a 448-wide draft would be ~3.3 GB or fail.
6. `Flash-Next online MXFP8 projection ready` lines.
7. `Mamba Cache is allocated ... max_mamba_cache_size: 96`.
8. `KV Cache is allocated ... #tokens:` about 1.75 M, on both the target and draft lines.
9. Graph capture: bs list `[1, 2, 4, 7, 8, 9, 10, 16]`; `available_gpu_mem` in the final `max_total_num_tokens=...`
   line (expect >= 3 GiB; today 4.25).
10. Under load (scripts/franzen_report.py):
    - MTP `accept len` against 2.66;
    - `gen throughput` at `#running-req` 16 against exp-072c at 14 and exp-072a at 10;
    - `#full token` peak as a share of the pool;
    - retractions or preemptions, HTTP errors;
    - the harness's malformed tool-call count.

## 8. How this was checked (this session)

- `python -I scripts/reap_kept_experts.py --out kaggle/franzen/reap448_kept_experts.json --pruned-index ... --base-index ...`
  gave 48 lines of "448 of 512 rows matched exactly" and "MTP router identical: True".
- `python -I scripts/sglang_reap_patch.py check-wheel <the Pennyroyal wheel>` reported the sha256 match, both anchors
  found once, and that the patched file compiles. `apply` was run twice on a copy of the wheel's file ("patched", then
  "already patched"), with a diff of 13 added lines.
- Real-data runtime check (`scratchpad/reap/tools/real_router_check.py`, CPU torch 2.14; it reads the two
  checkpoints' `model.safetensors.index.json`, which were deleted afterwards for disk space and must be re-fetched to
  rerun it). Each of Intel's 48 BF16 routers was fetched and put through `wrap_weights` in a stream with synthetic
  per-expert tensors named as in Intel's checkpoint. Results:
  - every sliced router equals REAP-k448's router bytes;
  - 48 x 448 experts landed in their slots with 9 tensors each, and 27,648 pruned tensors were skipped;
  - the router fingerprints matched, and a one-element change to one router was refused.
- `.venv/bin/python -m pytest -q tests/test_build_franzen_nb.py tests/test_sglang_reap_patch.py`: 38 passed, 1
  skipped (the torch test; torch is not in .venv). `ruff check` was clean on all touched files.

The builder tests execute the arm's own cell-12 code. The launcher's `run()` plus our block:
- on a file that is not the analysed one, stop the cell with `Command failed`;
- on the real wheel file, patch it, set the server env, and are a no-op on a rerun.

The two tests that need the wheel read it from `$PENNYROYAL_WHEEL` (default: this session's scratchpad copy) and skip
without it.
