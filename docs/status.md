# Status

Last updated: 2026-09-16 (session 1, remote CPU container: 4 vCPU, 15 GB RAM, no GPU; Kaggle CLI authenticated as `scottmahony`).

## Competition facts: verified vs. unconfirmed

| Item | Value used | Status | Source |
|---|---|---|---|
| Metric | RHAE: per level `min((baseline/actions)^2, 1.15)`; per game level-index-weighted mean, capped at the weighted fraction of levels completed (so never above 100%, as the Kaggle page says); total = mean over games; unsolved levels = 0 | **VERIFIED** | `arc_agi/scorecard.py` (arc-agi 0.9.9) and ARC-AGI-3 Technical Report §4.1, eq. 1–3. `tests/test_scoring.py` checks parity with the toolkit |
| Human baseline | upper-median best human action count per level; per-level `baseline_actions` ship in each game's `metadata.json` | VERIFIED | docs.arcprize.org/methodology, downloaded metadata |
| Action counting | every non-RESET action costs 1; RESET also costs 1 (`Card.inc_reset_count`); per-level count = actions between successive level completions | VERIFIED (toolkit) | `arc_agi/scorecard.py` |
| RESET in competition | competition mode forces *level* resets; game resets become level resets; a RESET at a level start is a billed no-op; one `make()` per environment; one scorecard | VERIFIED | docs.arcprize.org/toolkit/competition_mode and `arc_agi/api.py`; `arc3/env.py` mirrors it, `tests/test_env.py` checks it |
| Games / levels | 25 public games, 6–10 levels each (188 levels); ids and per-level baselines in `environment_files/*/metadata.json` | VERIFIED | download on 2026-09-15 |
| Runtime limit | **9 h** for CPU and GPU notebooks ("CPU Notebook <= 9 hours run-time, GPU Notebook <= 9 hours run-time"). Governor default 9 h minus a 15 min reserve | **VERIFIED** | Kaggle Code Requirements, pasted by the user on 2026-09-16 |
| Daily submissions | 5 | UNCONFIRMED (two secondary sources agree) | starter README, third-party summary |
| Milestone 2 | closes 2026-09-30, 23:59 UTC; prizes $25,000 / $7,500 / $5,000; notebook must be public under an open-source license by then | VERIFIED | Kaggle overview (pasted 2026-09-16) |
| Entry / team merge | 2026-10-26, 23:59 UTC | VERIFIED | Kaggle overview (pasted 2026-09-16) |
| Final submission | 2026-11-02, 23:59 UTC; winners announced 2026-12-04 | VERIFIED | Kaggle overview (pasted 2026-09-16) |
| License for prizes | "open source license" (no specific license named); prize-eligible entries that do not open-source are removed | VERIFIED (wording) | Kaggle overview (pasted 2026-09-16) |
| Hardware | `rtx6000` = GCP `g4-standard-48`: RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, SM 12.0, 48 vCPU, 176 GB RAM, 20 GB writable disk; image: driver 580.159.04 (CUDA 13.0), nvcc 12.8, torch 2.10.0+cu128, Python 3.12.13. Notebook accelerator id `nvidiaRtxPro6000` (CLI `--accelerator NvidiaRtxPro6000`) | **VERIFIED** | diag run v2 log, 2026-09-16 |
| Hidden set size | unknown; the Duck's Kaggle validation ran 16 games at 16 concurrent, 75 min each in a 90-min kernel | UNKNOWN | Tufa Labs README |
| Kaggle concurrency | the framework's `Swarm` plays **all games in parallel threads** against the gateway | VERIFIED | `ARC-AGI-3-Agents/agents/swarm.py` |

## What this container could and could not do

Could: install `arc-agi` 0.9.9 (Python 3.12), download the 25 games with an anonymous key,
run the local engine at ~1 ms/action, clone the Kaggle starter, the agents framework and the
Duck harness (MIT classifier in its pyproject; no LICENSE file at repo root; treat reuse as
"rebuild from the write-up", which is what `arc3/agents/repl_agent.py` does).

Could not: run any model (no GPU), push a Kaggle kernel (no token), read Kaggle's own
competition pages (JS-only), confirm the runtime limit or license.

## Models and serving assets (decided 2026-09-16, per CLAUDE.md)

| Role | Asset | Status |
|---|---|---|
| Primary | **Qwen3.8-27B-FP8** (official). Kaggle dataset `saltb0x/qwen3-8-27b-fp8`: 81 files, 30.89 GB, byte-for-byte equal to HF `Qwen/Qwen3.8-27B-FP8` @ `017b9c7a` (includes `mtp.safetensors`) | VERIFIED |
| A/B arm | **NVFP4**: HF `nvidia/Qwen3.8-27B-NVFP4` @ `dbb8f445` (19 files, 21.95 GB, Apache-2.0), size-verified against HF, private Kaggle dataset **`scottmahony/qwen3-8-27b-nvfp4-nvidia`** (21 files incl. provenance, 21.95 GB) | UPLOADED 2026-09-16 |
| Specialist arm | **Qwen3-VL-8B-Instruct-NVFP4** (`JEILDLWLRMA/Qwen3-VL-8B-Instruct-NVFP4` @ `243f10e2`, 16 files, 7.57 GB, Apache-2.0), size-verified against HF, private Kaggle dataset **`scottmahony/qwen3-vl-8b-instruct-nvfp4`** (18 files incl. provenance) | UPLOADED 2026-09-16 |
| Baseline reproduction | Qwen3.6-27B-FP8 as used by the Duck: `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` | available |
| vLLM wheelhouse | `saltb0x/arc3-vllm-wheelhouse-v0271-cu129` (vLLM 0.27.1, CUDA 12.9, flashinfer 0.6.16, built for the ARC3 duck harness). Alternative: `nick2187/qwen38-vllm0272-cu130-wheelhouse-v1` (vLLM 0.27.2, CUDA 13, needs a newer driver) | diag run pending |
| Tool-call parser | Qwen3.8's chat template emits `<tool_call><function=...><parameter=...>` XML: vLLM `--tool-call-parser qwen3_coder`, `--reasoning-parser qwen3`; template knobs `enable_thinking`, `reasoning_effort` in {xhigh (default), medium, low}, `preserve_thinking` | VERIFIED (template) |

## Leaderboard calibration (public LB, 2026-09-16)

Tufa Labs 18.81, then Ebi 8.68, Lord Han Solo 8.44, NVARC3 8.40, Third Intelligence 8.21, ... (15th place 5.85).
The score is in percent of human-level RHAE. The user's target of 65-100 is 3.5-5x the current leader; every
architectural step therefore has to be measured on dev and val, not assumed.

## Kaggle validation runs this milestone (user granted unlimited private runs on 2026-09-16)

1. `scottmahony/arc-prize-2026-arc-agi-3-arc3-agent` v1, **CPU**, 2026-09-16: explorer notebook, Save & Run All. Installed
   arc-agi from the bundle, unpacked arc3, played ls20 + vc33 offline (60 actions each), wrote submission.parquet. PASS. GPU quota: 0.
2. `scottmahony/arc3-gpu-diag` v1, 2026-09-16: serving-stack probe. **Landed on 2x Tesla T4, not the RTX 6000**: the starter kit's accelerator id
   `nvidiaRtx6000` is unknown to Kaggle; the right id is `nvidiaRtxPro6000` (fixed in both builders, plus `--accelerator` on push). Facts obtained anyway: driver
   580.159.04 (CUDA 13.0 capable), nvcc 12.8, Python 3.12.13, preinstalled torch 2.10.0+cu128 / transformers 5.0.0,
   /kaggle/working 20 GB, 4 vCPU / 31 GB RAM on the T4 box. vLLM 0.27.1 + flashinfer 0.6.16 from
   `saltb0x/arc3-vllm-wheelhouse-v0271-cu129` installed in 231 s (upgrades torch to 2.13.0, transformers 5.15.0; pip's
   resolver warnings are noise). vLLM then refused FP8 KV cache on SM75 (T4) as expected; `arc3.serve` now picks the KV
   dtype from compute capability. 6 min wall, T4 quota only.

3. `scottmahony/arc3-gpu-diag` v2, **RTX PRO 6000 confirmed** (`NVIDIA RTX PRO 6000 Blackwell Server Edition`, 97,887 MiB,
   compute capability 12.0, driver 580.159.04 / CUDA 13.0, 48 vCPU, 176 GB RAM, 20 GB /kaggle/working), 2026-09-16.
   vLLM 0.27.1 installed in 166 s; the FP8 model loaded in 120 s (28.95 GiB), MTP draft detected, 53.2 GiB KV cache
   (1.0 M tokens, 31x concurrency at 32k), server ready **345 s** after launch on the first attempt with MTP + FP8 KV.
   The first request then failed inside FlashInfer (auto-selected attention backend; needs SM120 cubins from NVIDIA's
   artifactory, unreachable offline).
4. `scottmahony/arc3-gpu-diag` v3, RTX, 2026-09-16: env var `VLLM_ATTENTION_BACKEND=TRITON_ATTN` changed nothing; vLLM 0.27
   logs "Unknown vLLM environment variable" and still picked FlashInfer. 9 min of RTX quota, same failure.
5. `scottmahony/arc3-gpu-diag` v4, RTX, 2026-09-16: `--attention-backend TRITON_ATTN` fixed the main model ("Using
   AttentionBackendEnum.TRITON_ATTN backend") but the MTP draft model still auto-selected FlashInfer (vLLM never inherits
   the target backend for drafts) and the first request failed the same way. 9 min of RTX quota.
6. `scottmahony/arc3-gpu-diag` v5, RTX, 2026-09-16: draft backend pinned via `--speculative-config {..., "attention_backend":
   "TRITON_ATTN"}`; start-up now proves itself with one real completion and retries without MTP otherwise. Result pending.

## Open items (need you)

1. Daily submission limit: paste the "Submission limits" lines from the Kaggle **Rules** page (still
   UNCONFIRMED; 5 per day from two secondary sources).
2. For future sessions put the Kaggle token in the Claude Code environment as `KAGGLE_API_TOKEN` (this
   session keeps it in git-ignored `.kaggle/access_token`). Consider regenerating the token after this
   project since it passed through a chat upload.

## Follow-ups noticed (not fixed on purpose)

- `arc_agi` logs at INFO through the root logger; the harness silences it with a level filter.
- The starter's `build_notebook.py` writes the agent to `/tmp/my_agent.py`; ours bundles a
  package instead (see `scripts/build_notebook.py`).

## Session 1 outcome (2026-09-16)

Built and tested (all in `tests/`, green on this container via `make test`):

- Evaluation harness with the toolkit scorer, fixed dev/val split, run records, crash isolation.
- Exact perception library and the local env wrapper mirroring the gateway's reset billing.
- Two CPU baselines measured (research log exp-000..002): random 0.19, explorer 0.06 on all 25 games.
  Both confirm that blind search is worth ~0 under RHAE (`docs/lessons/0003-*`).
- The REPL agent (Duck-style, persistent sandbox, image + ASCII + helpers, eviction, governor,
  explorer fallback) with a scripted-model end-to-end test. **Not yet run against a real model.**
- The council agent (`arc3/agents/council.py`): the user's six-specialist + coordinator design as an ablation arm,
  specialists run concurrently on a second vLLM server (Qwen3-VL-8B-NVFP4) or on the coordinator model; mock-tested,
  **not yet run against real models**. `scripts/build_eval_notebook.py` runs any agent on a split on Kaggle's RTX.
- Kaggle path: framework adapter with shared deadline and crash recovery, notebook builder that
  embeds the package and optionally starts vLLM, tests for both.

Next on the GPU box, in order:
1. `make setup && make games && make test`; start vLLM with Qwen3.8-27B-FP8 (`arc3.serve.build_vllm_command`)
   and fix the speculative-decoding flag for the installed vLLM version.
2. exp-003: REPL agent on `smoke` then `dev` with `TIME=900`; record s/action, tokens, VRAM.
3. A/B: Qwen3.6-27B-FP8 (Milestone-1 winner's model) vs Qwen3.8-27B-FP8, same harness.
4. Build the vLLM wheelhouse + model datasets on Kaggle; `make notebook`; one private validation run.
