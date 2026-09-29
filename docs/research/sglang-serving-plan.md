# SGLang serving arm for our Duck fork: what Son Pham's team does, and the plan (written 2026-09-29)

One-line summary: an upstream SGLang 0.5.20 server can be installed offline from public Kaggle wheelhouses and serves
our exact checkpoint with an FP8 KV cache (about 3x the tokens our 7.75 GiB BF16 vLLM cache holds), but it silently drops
our harness's past reasoning unless the client sends `reasoning_content`. Two 12-minute stress notebooks on Oct 3 (about
2 GPU-hours) decide whether a scored pair follows.

Everything below was read on 2026-09-29 from a CPU container. Nothing was run on a GPU, pushed to Kaggle or submitted.
Son Pham's repository has no licence, and neither does gabrielolympie/sglang-flashnext-sm120, so both were read for
facts only. No code from them is copied here or should be copied into our notebook. The launcher described in section 4
is our own design.

## 1. How Son Pham & Mark Barney serve Flash-Next with SGLang

Sources: `sonpham-org/arc-3`, branch `origin/docs/deepseek-v41-ceiling-run`, commit `c86b108a3` (2026-09-25);
`origin/main` commit `3b3a3e5d8` (2026-09-27, `docs/trace-findings/2026-09-27-moe-expert-pruning-review.md`); the Kaggle
CLI.

**Leaderboard (checked with the Kaggle CLI today).** Son Pham & Mark Barney show 12.21, last submission 2026-09-28
03:20 UTC. On Sep 21 their repo recorded 7.36. Tufa Labs now shows 45.33 and Yi-Chia Chen 36.73.

**Where it runs.** Every SGLang file in the repo drives GCP Spot `g4-standard-48` VMs, which have the same GPU as
Kaggle's rtx6000. Nothing in the repo is a Kaggle notebook. The Sep 27 pruning review on `main` names Son's Kaggle
notebook `arc3-flash-next-clean-return-sglang` as "the J' SGLang arm" (7 games x ~103k context, unpruned). The notebook
is private: `kaggle kernels list --user sonphamorg` shows only four AmnesiaBench kernels and a Feb 2026 traces kernel,
none of them ARC-3. We cannot tell whether the 12.21 came from that notebook.

**Their Kaggle datasets (public, all from late August, all from the vLLM era):**
- `sonphamorg/arc3-flashnext-gcp-runtime-exact-v1`: 6.0 GB, a `site-packages.tar.zst` of their GCP container plus
  its freeze list.
- `sonphamorg/arc3-flashnext-serving-part-{a,b,c}-v1`: 53.3, 40.8 and 41.6 GB. These hold the RadixArk checkpoint
  re-sharded per layer and expert range, and part A also carries the same site-packages tarball.

No SGLang wheel or build is public under their account. The pattern they used for vLLM (the whole container's
site-packages packed into a dataset) suggests their SGLang Kaggle runtime is a private dataset built the same way. That
is an inference, not verified.

**SGLang build (on GCP).**
- Base: official `sgl-project/sglang`, branch `qwen4-main-squashed`, taken from the branch head at VM boot. The commit
  is written to `sglang_commit.txt` at runtime, so the repo does not pin it.
- Patches: the six patches from `gabrielolympie/sglang-flashnext-sm120`, also taken from the head at boot (head today:
  `67d2f92`, 2026-08-31, no licence file). They are `0001b` (RecoverSSM/WY MTP verify on sm120, which adds
  `--gdn-mtp-cache-mode`), `0002` (FP8-KV tile dequant for QSA sparse prefill), `0003` (fp32 prefill state), `0004`
  (Triton low-M GEMM), `0005` (fp8 weight-only copies of the dense bf16 weights, about 3.6 GB) and `0006` (fp8
  HyperConnection mix and lm_head).
- Build: `uv pip install -e python` with the `docs.sglang.ai/whl/cu130` index, inside
  `nvidia/cuda:13.0.3-devel-ubuntu24.04`, then `docker commit`.
- This is not the Pennyroyal fork (`jpezzulli/sglang-rtxpro6000`). The gabriel fork builds on Pennyroyal's sm120 work,
  and their bench runs keep the served name `pennyroyal`.

**Checkpoint.** `RadixArk/Qwen3.8-Flash-Next-NVFP4` @ `7b719225242aacd3dbd3f9407468c2ee9a9d2594`
(`arms/cv5cr_sgl_c96k_w5/startup.sh` lines 26-27). This is the same checkpoint we serve from Keith Tyser's Kaggle model.
It is the build intended for SGLang: its PLE n-gram table is stored in FP8, and SGLang keeps it in pinned host RAM
(47.7 GiB).

**Launch flags of the scored arms** (`startup.sh`, `serve.sh` heredoc):
- Model and memory: `--tp 1 --dtype bfloat16 --quantization modelopt_fp4 --mem-fraction-static 0.95`,
  `--context-length <lanes-dependent> --kv-cache-dtype fp8_e4m3 --page-size 64`.
- Scheduling: `--max-running-requests <lanes> --cuda-graph-max-bs <lanes> --chunked-prefill-size 4096`.
- Mamba state: `--mamba-ssm-dtype bfloat16 --max-mamba-cache-size 48 --mamba-radix-cache-strategy extra_buffer`,
  `--mamba-track-interval 64 --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer`.
- Model loading and chat: `--ple-offload-embedding --trust-remote-code --chat-template <model>/chat_template.jinja`.
- Parsers and service: `--reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-metrics --watchdog-timeout 1800`.
- Speculative decoding: `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1`,
  `--speculative-num-draft-tokens 4 --speculative-draft-model-quantization unquant`,
  `--speculative-token-map hot_tokens_64k.pt` (FR-Spec, from the fork), `--gdn-mtp-cache-mode none`, and acceptance
  thresholds 1.0/1.0 (lossless).
- Host cache: `--enable-hierarchical-cache --hicache-size 64 --hicache-write-policy write_through`,
  `--hicache-io-backend kernel --hicache-mem-layout page_first`.
- Environment: `SGLANG_SM120_LOWM_FP8_WEIGHT=1 SGLANG_SM120_LM_HEAD_FP8=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1`.
- The radix prefix cache is on (the default).
- The vision tower is on: the serving gate sends an image, and nothing sets `language_model_only`.

**Parsers and workarounds.**
- No parser patches. The gate requires a tool call to come back parsed with no `<tool_call>` markup left in `content`.
  It does not send `tool_choice`.
- A small proxy on :1234 in front of SGLang on :1235 does three things. It answers `/tokenize` for their harness by
  running a 1-token chat completion and returning `usage.prompt_tokens`. It adds `max_model_len` to `/v1/models`. It
  injects `preserve_thinking: true` as a default `chat_template_kwargs`.
- Observation: their `/tokenize` path renames the vLLM alias `reasoning` to `reasoning_content`, but the chat path does
  not. Their gate's usage-side fixture already sends `reasoning_content`, so it never tests `reasoning` on the chat path.
  Their harness stores past reasoning as `reasoning` (`ARC3-Inference/inference/agent/tool_agent.py:2410` on that
  branch). If their SGLang branch behaves like 0.5.20 (section 2), those arms ran without past reasoning in history.
  Their own vLLM ablation `nopreserve` scored 2.68 against about 18. We cannot resolve this from the repo.

**Lanes x context and measured throughput** (GCP, `gcp/controllers/sglang-flashnext/README.md`). All figures are
aggregate generated tokens/s. The GPU KV pool is about 520k tokens at mem 0.95 (about 12.4 KB/token in fp8). Setting
0.98 runs out of memory during graph capture. Setting 0.965 runs out of memory on the first request when the fp8 copies
are on.

Shape bench, MTP lossless:

| Shape | Aggregate tok/s | Note |
|---|---|---|
| 7 x 39k | 759 | 130 per stream |
| 4 x 39k | 738 | |
| 1 x 39k | 248-273 | |
| 7 x 78k | collapses | exceeds the pool |

Slots bench (games over decode slots, harness-like turns):

| Configuration | Aggregate tok/s |
|---|---|
| 7 resident at 39k | 627 |
| 12 games over 5 slots at 45k, host cache | 619 |
| 12 games over 8 slots at 45k | 556 |
| 100k contexts | 305-475 |

Their vLLM profile is about 310 at 7 x 103k.

Rules they derived:
- `slots x (context + generation)` must fit the pool, or running requests get retracted (p90 turn 80-90 s).
- Parking games in host RAM lifts throughput 15-50%. About 2x games over slots is the sweet spot, and "28 in flight
  collapses" (measured at 100k contexts).
- A 64 GB host cache fits on the 176 GB host, 96 GB fits, and 128 GB fails.
- The gabriel fork adds that `--max-mamba-cache-size` must be about 6x `max-running-requests`, or the speculative CUDA
  graphs silently cap at batch 4.

**Scored arms** (launched 2026-09-26 00:45 UTC; the results are not in the repo):

| Arm | Lanes x context | Seconds per game |
|---|---|---|
| cv5-CR | 5 x 98,304 | 1584 |
| cv5-CR | 6 x 80,896 | 1584 |
| cv5-CR | 7 x 67,584 | 2061 |
| clean-return, no swap | 7 x 39,936 | 2061 |
| clean-return, no swap, 11 in flight over 7 slots | 11 x 39,936 | 2640 |

Relayed from Son's chart in the Sep 27 pruning review, all on the 132-minute all-25 suite on GCP:

| Configuration | Score |
|---|---|
| J' (SGLang, 7 x 103k, unpruned) | 21.4 |
| Expert-pruned (384/512), 16 x 103k | 33.9 |
| Pruned, 7 games | 19.7 |
| 34% pruned, 28 x 57k | 28.7 |

The pool grew from 736k to 1.97M tokens at mem 0.985 when pruned. Their vLLM reference on the same suite is LA-CR
18.01 / 19.24 at 7 x 103k (`gcp/controllers/lacr-serving-variants/README.md`). These are one run each.

**Startup (GCP).** Docker build about 8 min, weights about 7 min, MTP CUDA-graph capture about 13 min, gates and
self-tests about 15 min. The startup asserts uptime < 106 min before play. A fresh server runs 20-30% slow for its first
minutes because of Triton JIT per shape.

**Failure modes they recorded.**
- An invalid PNG in the gate returned HTTP 400 and killed the whole first wave at the serving gate.
- Memory-fraction OOMs, as above.
- Retraction when the pool is exceeded: 7 x 60k fell from about 520 to 328.
- A 128 GB host cache failed.
- The "fp8 MTP drafter" flag was useless.
- GCP Spot revoked the GPU cgroup (a GCP-only problem).
- On vLLM: MTP under spec decode has no prefix cache for this hybrid model, n-gram drafting corrupted a tool call, and
  turning `preserve_thinking` off collapsed the score.

## 2. What upstream SGLang 0.5.20 gives us (verified from the wheel)

The public dataset `aaravbajya/arc-agi-sglang-workspace` has 233 files, 10.84 GB, uploaded 2026-09-24. Its Kaggle
licence is "unknown" and it has no description, while `archive/dataset-metadata.json` inside says CC0-1.0. Its
`archive/sglang-0.5.20-cp312-cp312-manylinux_2_34_x86_64.whl` is byte-identical to PyPI's sglang 0.5.20 (sha256
`ffaced7e...`, released 2026-09-18). `flashinfer_python-0.6.18` and `humming_kernels-0.1.12` also match PyPI.

Read from the wheel, and from importing its `protocol.py` with a stubbed `sglang.utils`:

**Model support.** `qwen4_exp.py`, `qwen4_exp_mtp.py` and `qwen4_exp_ple_table.py` exist
(`Qwen4ExpForConditionalGeneration`). QSA forces `page_size 64`, and PLE offload is on by default: FP8, 47.7 GiB of
pinned host memory. Keith's vLLM already offloads the same table to host RAM on Kaggle ("1 PleOffloadLayer" in our
logs).

**Flags present.** `mamba-radix-cache-strategy` (auto / no_buffer / extra_buffer / extra_buffer_lazy),
`linear-attn-{decode,prefill}-backend`, `ple-offload-embedding`, the hierarchical-cache flags, `speculative-token-map`,
`speculative-accept-threshold-*` (default 1.0, lossless), `speculative-draft-model-quantization`,
`max-mamba-cache-size`, `mamba-track-interval`, `default-chat-template-kwargs` (per-request keys win),
`enable-cache-report`, `json-model-override-args` (for `language_model_only`) and `watchdog-timeout`.

**Absent (patch-only).** `--gdn-mtp-cache-mode`, `SGLANG_SM120_LOWM_FP8_WEIGHT` and `SGLANG_SM120_LM_HEAD_FP8`. So we
get upstream kernels without the fork's 0004-0006 speed work, and without 0001b as a flag. A line-match of the patches
against 0.5.20 suggests 0003 is essentially upstream, 0001b and 0002 partly, and 0004 not at all. 0.5.20 has its own
FP8-pool dequant in QSA sparse attention.

**Past reasoning is dropped.** `ChatCompletionMessageGenericParam` has only `reasoning_content`. An assistant history
message that carries `reasoning` loses it at request parsing (verified: `reasoning` came back absent,
`reasoning_content` came back None). The Flash-Next template renders only `message.reasoning_content`. The Duck stores
past reasoning as `assistant_message["reasoning"]` (anim bundle `tool_agent.py` about line 2276), so on stock SGLang
every past turn would render as an empty `<think></think>`. That is the failure mode of exp-035 (7.86 to 3.58, lesson
0022). Responses carry `reasoning_content`, which the Duck already reads as a fallback.

**Other client-facing behaviour.**
- Tool calls: the `qwen3_coder` detector is registered.
- Reasoning: the `qwen3` detector handles the template's prefilled `<think>`.
- Vision: the default vision attention backend on sm120 is `triton_attn`, not fa4.
- Request fields: `top_k`, `seed`, `min_p`, `chat_template_kwargs`, `tool_choice` and `reasoning_effort` are accepted.
- Over-length errors read "The input (N tokens) is longer than the model's context length (M tokens)." The Duck's
  `_is_context_length_error` does not match that phrase; it matches only the second message ("...maximum context
  length...").
- The Duck needs no `/tokenize` or `max_model_len`, because it estimates tokens locally at 3 characters per token. So
  we need no proxy.

**Wheelhouse sufficiency.** Nearly sufficient, with one real gap and several decoys:
- **Missing: `torchvision` matching torch 2.13.** sglang 0.5.20 and `sglang_kernel 0.4.7` pin `torch==2.13.0`, but
  the wheelhouse's `torchvision-0.29.0+cu130` requires `torch==2.14.0`. SGLang's Qwen VL processor imports torchvision
  at module import, so a mismatch can stop the multimodal model from starting. Fix: `torchvision-0.28.0` from the public
  `nick2187/qwen38-vllm0272-cu130-wheelhouse-v1` (`wheels/...`), which is byte-identical to PyPI (sha256 `028a3d48...`),
  a CUDA 13 build that requires `torch==2.13.0`. That dataset's licence is also "unknown". The `saltb0x` wheelhouse's
  0.28.0 is a CUDA 12 build, so do not use it.
- **Missing but not needed:** `build` and `smg-grpc-servicer`. The latter is imported only in gRPC mode. Install with
  `--no-deps` from an explicit list.
- **Present and consistent:** `torch-2.13.0+cu130` and its pins (triton 3.7.1, cuDNN 9.20.0.48, NCCL 2.29.7, NVSHMEM
  3.4.5, cusparselt 0.8.1), `sglang_kernel-0.4.7+cu130`, flashinfer python/cubin/jit-cache 0.6.18 (the cu130
  jit-cache wheel without an arch suffix), `transformers==5.12.1` and `tokenizers==0.22.2`.
- **Do not install:** torch 2.14.0, transformers 5.17.0, the `flashinfer_jit_cache*` 0.7.0 wheels (there is no
  flashinfer_python 0.7.0 in the set), `flashinfer_cubin-0.6.13`, `sgl_kernel-0.3.21` (the old package name), `b12x`,
  and the duplicate numpy, protobuf, fsspec and tokenizers versions.
- **Platform:** Kaggle glibc 2.35 satisfies manylinux_2_34, and Python 3.12.13 matches cp312. Driver 580.159.04 is
  CUDA 13.0, fine for cu130 wheels. The pip `nvidia-cuda-nvcc` and `nvjitlink` are 13.4, newer than the driver, so PTX
  JIT through them may fail. Keith's runtime unpacks a CUDA 13.0 toolkit (`.../usr/local/cuda-13.0`) that we can point
  `CUDA_HOME` at.

## 3. Options for the engine

| Option | Build | Licence | Kaggle effort | Expected speed |
|---|---|---|---|---|
| A. Upstream 0.5.20 from the two public wheelhouses | none | Apache-2.0 code; datasets "unknown", wheels hash-checkable | our launcher only | unknown; the sm120 speed patches are absent |
| B. Pennyroyal (`jpezzulli/sglang-rtxpro6000`, head `12846e83`) | native: CUDA 13.3, GCC 15, Rust | Apache-2.0 | a wheel or site-packages built on another machine, uploaded as our private dataset | published C4 428 / C8 632 on short prompts |
| C. Son's recipe (upstream branch + gabriel patches) | Docker build | patches unlicensed: not usable | n/a | 440-755 at harness shapes |

Start with A. B needs a machine with a GPU toolchain and a private dataset upload, which is an owner decision, and is
only worth it if A passes but falls short.

## 4. The SGLang arm of our notebook (design; our own code)

**Datasets to attach:**
- `keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1` (model, unchanged).
- `aaravbajya/arc-agi-sglang-workspace` (wheels).
- `nick2187/qwen38-vllm0272-cu130-wheelhouse-v1` (torchvision 0.28.0 only).
- Keith's two datasets stay attached: his runtime provides the CUDA 13.0 toolkit, and the vLLM fallback needs them.
- The anim TAAF source, unchanged.

**Launcher** (new `arc3/serve_sglang.py`, inlined into the notebook by a builder flag `--serving sglang`), in order:
1. **Hash check.** Verify an sha256 manifest of the wheels we install. We generate it once from PyPI JSON plus the files
   we downloaded; the big binaries are hashed in parallel, about 1 min.
2. **Install.** Create a clean venv (`python3.12 -m venv /tmp/sgl`, no system site-packages, so Kaggle's torch 2.10
   cannot leak in) and run `pip install --no-index --no-deps` on the explicit list. Log `pip check`, which is
   informational only.
3. **Import probe.** In the venv, import torch, torchvision, flashinfer, sglang_kernel, sglang and
   `sglang.srt.multimodal.processors.qwen_vl`. Print versions and `nvidia-smi`.
4. **Environment.**
   - Point `CUDA_HOME` at Keith's CUDA 13.0 toolkit (untar its layer from his runtime blobs, as his setup does), or the
     pip nvcc as a fallback, and add `/usr/local/nvidia/lib64` to the library path.
   - Put all JIT caches under `/tmp/sgl-cache`: Triton, flashinfer workspace, sglang JIT and inductor. Seed that
     directory from a previous run's output if one is attached as a kernel source, and copy it to
     `/kaggle/working/sgl-cache` at the end.
   - Set `MAX_JOBS=8`, `OMP_NUM_THREADS=8`, `TOKENIZERS_PARALLELISM=false`, `HF_HUB_OFFLINE=1` and
     `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
5. **Server on 127.0.0.1:1234**, served name `Qwen/Qwen3.8-Flash-Next-NVFP4`, so the notebook's existing asserts and
   `LOCAL_ANALYZER_BASE_URL` stay unchanged. Flags for rung 1:
   - Model and memory: `--tp 1 --dtype bfloat16 --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 --page-size 64`,
     `--mem-fraction-static 0.94`.
   - Context and scheduling: `--context-length 32768` (the harness window) `--max-running-requests R`,
     `--cuda-graph-max-bs R --chunked-prefill-size 4096`.
   - Mamba and linear attention: `--mamba-radix-cache-strategy extra_buffer`, `--mamba-ssm-dtype bfloat16`,
     `--mamba-track-interval 64 --max-mamba-cache-size 6R`, `--linear-attn-decode-backend flashinfer`,
     `--linear-attn-prefill-backend flashinfer`.
   - Loading and chat: `--ple-offload-embedding --trust-remote-code`, `--chat-template <model>/chat_template.jinja`.
   - Parsers and service: `--reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-metrics`,
     `--enable-cache-report --watchdog-timeout 1800`.
   - Speculative decoding: `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1`,
     `--speculative-num-draft-tokens 4` with lossless acceptance (the defaults).
   - Radix cache on. No token map at first; FR-Spec would be our own map later. No `--default-chat-template-kwargs`,
     for parity with today's vLLM runs, where the template defaults apply and P11 passes knobs per request. Vision tower
     on.
6. **Rung ladder** on failure. Each rung has a timeout, and the log tail is saved for each:
   1. The rung-1 flags above.
   2. Linear-attention backends switched to `triton`. This is the fork's own fallback when its WY patch is missing.
   3. Rung 2 with no speculative decoding, mamba cache 3R.
   4. BF16 KV (`auto`), no MTP, R=8, mem 0.90.

   A rung counts as up only after `/health` passes, plus one text completion and one real 256x256 board-image
   completion.
7. **Fallback to vLLM.** If no rung is up 50 min after notebook start, run Keith's original `setup_commands.json`.
   A submission must never be left without a server.
8. **Watchdog thread.** Probe `/health` every 30 s. Three misses restart the same rung, logged to
   `sglang-watchdog.jsonl`.
9. **Persisted environment.** Write the same analyzer keys Keith's setup writes: `LOCAL_ANALYZER_BASE_URL`,
   `LOCAL_ANALYZER_MODEL_ID`, `LOCAL_ANALYZER_PROVIDER=vllm` (this keeps `top_k`, `seed` and `chat_template_kwargs` in
   the payload, all of which SGLang accepts), and `OPENAI_BASE_URL`.
10. **Teardown.** Snapshot `/metrics` to `sglang-metrics-final.prom`, save the server log, and kill the process group.

**Harness changes** (patches in `scripts/taaf_ours_patch.py`, active only when `OURS_SERVING=sglang`, so vLLM arms stay
byte-identical):
- **P30 (required).** Store past reasoning as `reasoning_content` instead of `reasoning`. Every reader in our patches
  (P1 note extraction, the `_STRIP_PAST_REASONING` path) must accept both keys. Test: render a history through the
  real template and check that the reasoning text appears.
- **P30b.** Add "longer than the model's context length" to `_is_context_length_error`, so an over-length prompt trims
  and retries instead of failing the turn.
- Unchanged: 28 lanes, the 32k window, `tool_choice: "auto"`, the markup-recovery fallback for unparsed tool calls,
  and one board image per turn.

**Memory budget (Kaggle: 176.9 GiB RAM, 97.9 GB GPU).**
- Host RAM: 47.7 GiB of pinned PLE, plus the host cache if enabled (32 GB plus about 30% mamba overhead), plus the
  harness and 28 sandboxes. Keep total pinned memory at or below about 95 GiB.
- GPU pool: without the fork's fp8 weight copies, about 3.6 GB more should go to KV than in Son's 520k-token pool at
  0.95. The 6R mamba cache takes back about 56 MB per entry. We measure the result as `sglang:max_total_num_tokens`.
- At R=12 x 32k = 393k tokens, the running set should fit with room for cached prefixes. R=16 needs about 524k.

## 5. Stress test (`scripts/build_sgl_stress_nb.py`, modelled on `build_kv_stress_nb.py`)

The notebook keeps cells 0-8 of the base notebook, replaces Keith's setup command with the launcher, and imports the
solver only to get the Duck's system prompt. Then it runs two phases.

**Functional gate** (about 3 min; each check is written to `sgl_stress.json`):
1. `/v1/models` returns the served name.
2. **Vision:** a real 256x256 Duck board PNG gives a non-empty answer. Record the prompt tokens with and without the
   image, to compare image token counts with vLLM.
3. **Tool call** (tools=[python], `tool_choice: auto`, thinking on): a parsed `tool_calls[0]` with valid JSON
   arguments and no `<tool_call>` markup in `content`.
4. **Reasoning round-trip:** send the same history twice, once with 1,000 words of `reasoning_content` and once with
   `reasoning`. Record the prompt_tokens difference. It must be about 1.3k tokens for `reasoning_content`, and about 0
   for `reasoning`, which documents the drop. Then run one call through the patched harness path.
5. **Response split:** a thinking-on response has non-empty `reasoning_content`, and its `content` has no `</think>`.
6. **Over-length:** a 40k-token prompt returns an error string that the P30b matcher accepts.
7. **Concurrency:** R concurrent 24k-token prompts all return.

**Load** (12 min, identical to the vLLM baseline so the numbers compare): 28 clients, the Duck system prompt, a
15,000-word user text plus a 256x256 image, `max_tokens 1500`, thinking on, `top_k 20`.
- Every 10 s, sample `sglang:num_running_reqs`, `num_queue_reqs`, `token_usage`, `num_retracted_reqs`,
  `spec_accept_length`, `cache_hit_rate`, `gen_throughput`, `generation_tokens_total`, `kv_available_tokens` and
  `mamba_usage`, plus `nvidia-smi` memory and host RAM.
- The summary also holds each startup phase's time (install, load, graphs, first token), the rung used, the flags,
  the versions, and the latency p50/p90.
- The baseline to beat, kvstress-7g75-b2k (runs/kvstress-7g75-b2k): 421 completions, 152.9 generated tokens/s,
  5.65 running, 18,335 prompt tokens, 49.8 s mean latency, ready after 940 s, 31 min for the whole notebook.

**Two notebooks, run in parallel on Oct 3:**
- `arc3-sgl-stress-r12`: R=12, no host cache.
- `arc3-sgl-stress-r16-hic32`: R=16, mamba cache 96, `--enable-hierarchical-cache --hicache-size 32`, write_through,
  kernel I/O, page_first layout.

## 6. Go/no-go (to be copied into the research log before the runs)

**GO for a scored arm** if one profile meets all of the following:
1. The server is up within 45 min of notebook start, on a rung that keeps the FP8 KV cache.
2. All seven functional checks pass.
3. Zero request errors and zero server restarts during the load.
4. Generated tokens/s is at least 1.25 x 152.9 = 191 on the identical load, with mean running at least 8. The margin
   covers about 15 extra minutes of startup and the risk of a new engine.
5. Retracted requests are under 1% of completions.
6. Peak host RAM is at most 150 GiB, and the GPU does not run out of memory.

If both profiles pass, the one with more tokens/s goes forward. **NO-GO** otherwise: record it and keep vLLM. If A only
fails on speed (within 10% of 191), Option B becomes the owner's call.

**Scored arm.** exp-062 = exp-054's patches + P30/P30b on the passing profile. Public 25, 28 lanes, 132 min, as a pair.
- Read first on the mechanism: past reasoning must be about 35% of prompt tokens, as in exp-054, and requests per run
  must be at least 1.2x exp-054's 1,609 / 1,627.
- Then on score: keep it if the pair's mean z-sum is not below exp-054's (+12.45) by more than 2, and no game lost its
  server.
- An LB draw follows only on the owner's word.
- P28 (stable-prefix trim) and longer contexts (48-64k at R=8-12) are later arms. Son's results point at context, but
  one change at a time.

## 7. Main risks

1. **Past reasoning dropped (verified).** Without P30 the arm reruns exp-035's collapse. The gate's round-trip check
   exists to catch this.
2. **Speed without the sm120 patches is unmeasured.** Pennyroyal's unpatched-by-gabriel numbers (C8 632) and Son's
   patched 440-755 are both short-prompt or GCP figures. Our 21k-token prompts with images may land well below them.
3. **FP8 KV in QSA sparse prefill on upstream.** Pennyroyal notes its correction covers only unit-scale FP8, and PR
   #36644 is open. A quality loss would not show in tokens/s, only in the scored pair.
4. **torchvision and the toolchain.** If the 0.28.0 wheel or the 13.4 pip toolchain misbehaves, the processor import or
   a JIT kernel fails at startup. The ladder, the CUDA 13.0 toolkit and the vLLM fallback bound the cost.
5. **Startup.** Likely 25-40 min against vLLM's 16 min, which is 2-4% of the 9 h budget; wave-fit absorbs it. The first
   minutes run slow because of JIT.
6. **Host RAM:** pinned PLE plus the host cache. An OOM kill zeroes the rerun, so the host cache stays at 32 GB or less
   until measured.
7. **28 games over R slots.** Son saw collapse at 28 in flight (at 100k); at 32k it should hold, so we measure queue
   and retractions.
8. **Tool parser differences** from vLLM's `qwen3_coder`: Pennyroyal found quoted markup parsed as calls in 3 of 6
   probes. Watch the harness's malformed/recovered-call counters.
9. **Licence and supply chain.** Both wheel datasets are "unknown" on Kaggle. The wheels are upstream, and three were
   hash-matched to PyPI, but a prize submission should re-host the exact files as our own dataset. That needs the owner:
   dataset creation was blocked once by the permission check.
10. **Son's evidence is indirect.** Their Kaggle notebook is private, the 12.21 is not tied to it, and each GCP score is
    one run.

## 8. GPU budget for Oct 3

| Step | GPU minutes |
|---|---|
| `arc3-sgl-stress-r12` (install ~6, start 20-30, gate 3, load 12, teardown 2) | 45-60 |
| `arc3-sgl-stress-r16-hic32` (runs in parallel) | 45-60 |
| A fallback rung, if needed (MTP off or triton linear) | +30-60 |
| **Serving validation total** | **about 90-120, at most 180** |
| exp-062 pair, if GO (2 x (132 play + ~35 start + ~5)) | about 345 |
| **Oct 3 total with the scored pair** | **about 7.5-9 h of the 30 h week** |

A CPU-only preflight (install and import in a Kaggle CPU session) costs no GPU quota. It needs one private push before
Oct 3.

## 9. Decisions for the owner

1. **Datasets:** may private runs attach the two third-party public wheel datasets with "unknown" licences? For a final
   submission, may we re-host the exact wheels as our own private dataset?
2. **Option B:** if A passes the gate but falls short, is a Pennyroyal (Apache-2.0) build worth making? It needs the
   local workstation or another build machine, plus a private dataset upload.
