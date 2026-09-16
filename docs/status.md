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
| Specialist arm, primary since exp-010 | **Qwen3-VL-8B-Instruct-FP8** (official `Qwen/Qwen3-VL-8B-Instruct-FP8` @ `9cdc6310`, 13 files, 10.6 GB, Apache-2.0, `quant_method: fp8` = the same vLLM path the working 27B coordinator uses), size-verified against HF, private Kaggle dataset **`scottmahony/qwen3-vl-8b-instruct-fp8`** (14 files incl. provenance). The NVFP4 build (compressed-tensors `nvfp4-pack-quantized`) failed vLLM engine-core init twice in exp-010 (`runs/_kaggle_output/arc3-eval-dev-council/vllm-specialist.log` once pulled); the council notebook now tries FP8 tuned -> FP8 conservative -> NVFP4 tuned -> NVFP4 conservative (`arc3.serve.specialist_attempts`) and only then shares the coordinator | UPLOADED 2026-09-16 11:15 UTC |
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
6. `scottmahony/arc3-gpu-diag` v5, RTX, 2026-09-16: **PASS**. Draft backend pinned via `--speculative-config {...,
   "attention_backend": "TRITON_ATTN"}`. vLLM install 141 s, server ready 316 s (first attempt, MTP + FP8 KV), probe OK.
   Tool-call completion with image: 1.3 s, 80 completion tokens, code parsed. Throughput: 94 tok/s single stream,
   308 tok/s at 8 concurrent; MTP mean acceptance length 2.4-3.0. REPL smoke (ls20 + vc33, 5 min, 150-action cap):
   0 levels; model took 3 and 7 actions in 20 and 16 calls (p50 11 s/call, ~15k prompt tokens/call); the explorer
   fallback then spent the remaining ~145 actions in seconds. Files: `runs/kaggle-diag-v5-repl-smoke/`. 18 min RTX.

7. `scottmahony/arc3-eval-smoke` v1, RTX, 2026-09-16: exp-003b smoke, **first model-driven level solved** (ls20 L1 in 19
   actions vs human 22). 18 min RTX. See research log.
8. `scottmahony/arc3-eval-dev` v1, RTX, 2026-09-16: exp-003 control arm on the 19 dev games, 1200 s/game, 8 concurrent:
   **dev 0.53**, 6/142 levels (all level 1, four at the human cap). Context-overflow bug sank 4 games (see research log).
   58 min RTX.
9. `scottmahony/arc3-eval-dev` v2, RTX, 2026-09-16: exp-003c, same settings with the fixes: **dev 0.56**, 5/142 levels, zero
   errors. 67 min RTX. This is the control.
10. `arc3-eval-dev` v3 (exp-005, dev 1.11, 8 levels), `arc3-eval-dev-b` v1 (exp-007, 0.84, 5), `arc3-eval-dev-c` v1 (exp-008,
    0.73, 5), `arc3-eval-dev-d` v1 (exp-009, 0.84, 6), `arc3-eval-dev-e` v2 (exp-009b repeat, 0.87, 6),
    `arc3-eval-dev-council` v1 (exp-010, 0.28, 4; specialists never started): about 60-70 min RTX each, all
    2026-09-16; details and per-game tables in `docs/research_log.md` and `runs/kaggle-*/summary.json`.
11. `arc3-eval-dev-council-b` v1, `arc3-eval-dev-f` v1, `arc3-eval-dev-g` v1 (12:00-12:30): **wasted, landed on Tesla T4s**
    because the pushes omitted `--accelerator NvidiaRtxPro6000` (the 27B model cannot load in 15 GB; both attempts OOM,
    fallback to the rules agent, ~10 min each). `scripts/push_eval.py` now always passes the flag and refuses a notebook
    whose metadata does not name the RTX. Re-pushed 12:30 as v2 (exp-010b council-b, exp-011 f); exp-004 (g) follows.

## Rule library coverage (exp-006a, code-only, 2026-09-16)

`scripts/rule_coverage.py` plays each public game blind for 60 actions and measures the fraction of entity events the
fitted rule set explains: mean 0.50 over 25 games; ar25 1.00, ft09 1.00, m0r0 1.00, sp80 1.00, tn36 1.00, ls20 0.98,
re86 0.96, cn04 0.86, g50t 0.72, dc22 0.70, ka59 0.63; two games at 0. Lower bound, not a score. Details: `runs/rule-coverage/summary.json`,
research log exp-006a.

## Open items (need you)

- **Kaggle allows 2 concurrent batch GPU sessions and the queued exp-005 kernel (`arc3-eval-dev` v3, queued since
  04:53 UTC while later kernels ran) holds one of them.** The CLI cannot cancel a queued kernel. If it is still
  "Queued" when you look, cancel it from the Kaggle UI (Your Work -> arc3-eval-dev -> Cancel run) so two runs can
  proceed in parallel.

1. Daily submission limit: paste the "Submission limits" lines from the Kaggle **Rules** page (still
   UNCONFIRMED; 5 per day from two secondary sources).
4. **Qwen3.8-Flash-Next-NVFP4 (requested 2026-09-16 evening):** cannot be staged from this container (132.7 GB, one
   53.7 GB file, 18 GB of writable disk here; Kaggle has no server-side Hugging Face import in the CLI). Recipe and
   metadata for an upload from the workstation: `docs/models/qwen3-8-flash-next-nvfp4/UPLOAD.md`. Before it can
   serve on one RTX PRO 6000: NVFP4 language weights ~79 GB on the GPU, the 51 GB n-gram/PLE table offloaded to host
   RAM, a vLLM newer than the 0.27.1 wheelhouse (needs commit d4d703c, "Fix FP8 PLE loading in mixed ModelOpt
   checkpoints"), and NVFP4 kernels on SM120 (unverified). The vLLM recipe page lists only 4-GPU configurations.
3. **Milestone 2 (closes 2026-09-30):** the submission notebook (single 27B REPL agent, exp-011 harness) passed its
   private Save & Run All on the RTX on 2026-09-16. Two decisions are yours: (a) the open-source license for the public
   copy (MIT or Apache-2.0 for the code; the model weights are Apache-2.0 already); (b) the go-ahead to make the
   notebook public and to press Submit to Competition (each real submission is one of the daily allowance and about 9 h
   of the hidden-set run). I will not publish or submit without that OK.
0. **Milestone 2 (2026-09-30)** needs the notebook public under an open-source license by then. Say which license
   (MIT/Apache-2.0 for the code) and I will prepare the public copy once the control arm and one Save & Run All of the
   submission notebook on the RTX are green; publishing itself waits for your explicit OK.
2. For future sessions put the Kaggle token in the Claude Code environment as `KAGGLE_API_TOKEN` (this
   session keeps it in git-ignored `.kaggle/access_token`). Consider regenerating the token after this
   project since it passed through a chat upload.

## Follow-ups noticed (not fixed on purpose)

- Measurement noise: four single runs of similar arms span 0.56-1.11 with 5-8 levels; the three levels every arm
  solves (ar25 L1, lp85 L1, sb26 L1) are the only stable signal. Plan: repeat exp-009 unchanged (exp-009b) to
  measure run-to-run spread, then require a difference larger than that spread before keeping a change.

- Human play data (342 replays for the 25 public games, arcprize.org/blog/arc-agi-3-human-dataset): the download short
  link (dub.link/vfwCqvb) answers 429 from this container and the replay pages load their data through an endpoint I
  could not find in 15 minutes. If you can download the "full Public Demo dataset" archive from that blog post and
  drop it into `data/human/` (or a private Kaggle dataset), plan-100 §4.1 (goal priors from winning steps) becomes
  a CPU-only job.

- `arc_agi` logs at INFO through the root logger; the harness silences it with a level filter.
- The starter's `build_notebook.py` writes the agent to `/tmp/my_agent.py`; ours bundles a
  package instead (see `scripts/build_notebook.py`).

## Session 3 outcome (2026-09-16, midday; in progress)

- **Council fix (user's request):** exp-010 ran without its specialists (NVFP4 vLLM server failed engine-core init twice,
  silent fallback). Fixed in `arc3/serve.py` (attempt ladder over checkpoints and flag sets, gpu_mem sized from the
  memory the coordinator left, image probe, `LAST_START` record) and both notebook builders; the official
  Qwen3-VL-8B-Instruct-FP8 build is uploaded as `scottmahony/qwen3-vl-8b-instruct-fp8` and is the first rung. exp-010b
  (`arc3-eval-dev-council-b`, run `kaggle-council-dev-010b`) is built at HEAD and pushes when a Kaggle slot frees.
- **exp-011 bundle (from the exp-009 transcripts, all in the harness, measured code-only, not yet on Kaggle):** walkable
  terrain by colour, sprite companions, avatar hand-over, turn-tracking ids, edge-riding markers as compound parts, live
  move/rules predictors that follow `PLAN['optimistic']`, model retirement after three misses, idle-batch stop, inline
  events on act() results, no duplicate result echo, helper restore after rebinding, lazy rule fitting, downscale() at
  tile size, goal_candidates() on level 1. Code-only coverage 0.50 -> 0.547 (wa30 0.30 -> 1.0, tu93 0.33 -> 0.5,
  sk48 0.29 -> 0.43); code-only rules agent 0.199 -> 0.224 (exp-006c). Notebook `arc3-eval-dev-f` (run
  `kaggle-repl-dev-011`) is built at HEAD for the next free slot.
- exp-010 finished 11:57: **dev 0.278, 4 levels** with all six roles on the coordinator (specialist server never
  started: flashinfer's cutlass FP4 JIT has no SM120 kernels, `vllm-specialist.log`); the shared-model council costs
  more GPU time than it gives (coordinator p50 latency up ~40%, 743 actions vs 987). exp-010b (specialist ladder,
  FP8 first, NVFP4 with Marlin last) pushed 11:58 as `arc3-eval-dev-council-b`.
- exp-009b finished 12:14: **dev 0.870, 6 levels** (exp-009: 0.836, 6) with only 3 of 9 solved games in common: score
  stable to about +-0.05, per-game outcomes churn. Keep/revert threshold set (research log). exp-011 pushed 12:15.
- **exp-011 finished 14:02: dev 1.229, 9 levels** (best so far; exp-005 1.11/8, exp-009 repeats 0.84-0.87/6): the
  transcript-driven harness fixes are KEPT. exp-004 (thinking off) pushed 14:03 as `arc3-eval-dev-g` v2.
- exp-010b finished 14:05: **dev 0.744, 5 levels, specialists still absent**: the FP8 Qwen3-VL-8B build died in
  DeepGEMM at load ('Unknown SF transformation'), the NVFP4 build in flashinfer's FP4 JIT (no SM120 kernels). Fix
  pushed as env vars on the rungs (VLLM_USE_DEEP_GEMM=0 for FP8, Marlin for NVFP4); exp-010c is the retry.
- exp-010c finished 15:23: **specialist server started (FP8, DeepGEMM off; the council bug is fixed) and the council
  scored dev 0.498, 5 levels** vs exp-011's 1.229/9 on the same harness. Third loss in three council runs:
  the arm is parked (code kept), see research log. exp-011b (noise repeat) pushed 15:25 as `arc3-eval-dev-f` v3.
- exp-004 (thinking off) finished 15:44: **dev 0.324, 6 levels, 3837 actions**: cheap calls made the model spray
  actions (lp85 819, wa30 786); REVERTED, low-effort thinking stays. exp-012 pushed 15:45 as `arc3-eval-dev-h` v1.
- exp-011b (repeat) finished 16:45: **dev 1.372, 9 levels**, 7 of 8 solved games shared with exp-011 (1.229/9): the
  exp-011 harness is the confirmed base (pair mean 1.30). Submission notebook (REPL agent, harness 699825b) pushed
  16:46 as a private Save & Run All on the RTX: `arc-prize-2026-arc-agi-3-arc3-agent` v2 (Milestone 2 check).
- exp-012 finished 16:59: **dev 0.608, 6 levels**, below the exp-011 pair (1.23/1.37, 9); no mechanism visible in the
  transcripts (describe_events calls fell 126 -> 40 as intended). Repeat (exp-012b) decides; until then the
  submission base is the exp-011 harness 756a87e. exp-013 (medium effort) running since 17:00.
- **Submission notebook Save & Run All PASSED on the RTX** (17:03, `arc-prize-2026-arc-agi-3-arc3-agent` v2, private,
  harness 699825b): vLLM up on the first attempt, offline smoke on ls20 + vc33 (vc33 L1 in 8 actions), `submission.parquet`
  written, 17 min of RTX. This is the Milestone 2 candidate; publishing needs your OK (open items). exp-012b pushed 17:04.
- exp-013 (medium effort every call) finished 18:08: **dev 1.251, 10 levels** (su15 L2, vc33 L2), the most levels of
  any run; score inside the exp-011 band. Repeat exp-013b pushed 18:10. exp-012b still running.
- exp-012b finished 18:13: **0.692, 6 levels** (exp-012: 0.608/6) vs the exp-011 pair (9/9): the events line in every
  tool output is off by default now (`tool_events_line`); the error fixes stay. exp-014 (medium effort, line off)
  pushed 18:20 as `arc3-eval-dev-j` to separate the two effects; exp-013b running.
- exp-013b finished 19:19: **0.866, 11 levels** (cd82 and ka59 solved for the first time) but with many actions per
  level; medium-effort pair mean 1.06 vs the adaptive-low pair 1.30. Submission config stays adaptive low (exp-011).
  First validation-split run of the base config pushed 19:22 as `arc3-eval-val-a` (exp-011v).
- exp-014 (medium effort, events line off) finished 19:23: **0.631, 7 levels**. Effort ablation closed: adaptive
  low (exp-011 config) stays; medium's three runs average 0.92 vs 1.30. exp-015 (earlier raise to medium) pushed 19:27.
- **Validation split, first run (exp-011v, 19:49): val 0.794, 1/41 levels** (sp80 L1). Dev 1.30 vs val 0.79: dev has
  been tuned on, val has not; g50t had a perfect world model and no level (goal inference, like dc22 on dev).
  exp-016 (base config on all 25 games) pushed 19:52 as `arc3-eval-all-a`.
- exp-015 (raise to medium after 3 stagnant actions) finished 20:32: **0.421, 4 levels**, REVERTED. Low-effort runs on
  the post-exp-011 harness now read 6, 6, 4 levels against 9, 9 on 756a87e: the harness commits are suspect; exp-016's
  dev subset decides whether to bisect. The observation text is unchanged between the two (same length, no 'unknown'
  roles printed), so no mechanism is identified yet. exp-011c (the 756a87e notebook, third sample) pushed 20:36 as
  `arc3-eval-dev-f` v4 to run alongside exp-016 as the paired control.
- exp-016 (all 25 games, current harness) finished 21:18: **dev 1.041 (6 levels), val 0.0**, all-25 0.791; ft09 L1
  solved for the first time. exp-018 (the submission's operating point: 25 games concurrently, 3 h each) pushed
  21:19 as `arc3-eval-all-long-a`; exp-011c still running.
- exp-011c finished 21:43: **0.775, 6 levels** on the exp-011 commit itself. Harness question closed: the two
  triples (1.23/1.37/0.78 vs 0.61/0.69/1.04) overlap; noise is +-0.3 in score and +-2 levels between identical runs.
  exp-018 (relaunched 21:28 after a note-quoting bug) and exp-017 (pushed 21:46) are the long-horizon runs.
- 14:09: exp-010c (`arc3-eval-dev-council-c` v1, harness 7de7f95) running; exp-004 v2 queued. Queue after them, in
  order: exp-011b (the exp-011 notebook unchanged, per the noise rule), exp-012 (harness 7de7f95: events line in every
  tool output, single results iterate, ascii(region) degrades to tiles, role always present), then a private Save &
  Run All of `notebooks/submission.ipynb` (REPL agent) on the RTX as the Milestone 2 candidate check.
- exp-011 transcripts: 587 code cells, 46% without an act() (exp-009: ~55%); 31 tool errors, the three harness-side
  ones fixed in 7de7f95; describe_events() still called in 77 inspection-only cells, hence the events line.

## Session 2 outcome (2026-09-16, afternoon)

- **exp-007 dev 0.839** vs control 0.56 (Kaggle, same settings; provisional: two levels of difference, half the actions).
- Rule library built and measured code-only: coverage 0.10 -> 0.50 mean over the 25 public games in one day of
  perception and fitter fixes (occlusion, compound sprites, cell-level walkability, required colour, invisible walls,
  slides, click effects, return-to-start, colour-aware matching). 11 games >= 0.63 explained; 2 at 0.
- Code-only rules agent (exp-006b): 0.20 on all 25 games at 120 s/game; now the REPL fallback (explorer was 0.06).
- Transcripts + report tool; post-mortem: ~35 model calls per game is the binding constraint (lesson 0009). Harness
  changes for the next arms: per-turn rules summary, goal hints, probe suggestions, optimistic planners, tile map,
  first-step nudge.
- Queued on Kaggle: exp-005 (running), exp-008 (fitter v5 + summary + hints, commit 840bd4b). Ready to push when a
  slot frees: exp-009 (HEAD, efficiency bundle) and exp-010 (council).
- Later the same day: exp-008 0.734, exp-009 0.836 (6 levels), exp-009b (noise repeat) and exp-010 (council) running.
  **exp-010 ran without its specialists**: the Qwen3-VL-8B-NVFP4 vLLM server failed engine-core initialisation on both
  attempts (~150 s) and the notebook silently fell back to the coordinator model for the six roles, so exp-010 measures
  the *shared-model* council only. Fix in `arc3/serve.py` (attempt ladder over checkpoints and flag sets, gpu_mem sized
  from the memory the coordinator actually left, image probe before declaring the server ready, `LAST_START` record) and
  both notebook builders (coordinator 0.60 / specialist 0.30 split, `--specialist-dataset` preference list). The
  official FP8 Qwen3-VL-8B build is uploaded as the first rung. exp-010b (`arc3-eval-dev-council-b`, run name
  `kaggle-council-dev-010b`) is built and pushes when a Kaggle slot frees.

## Session 1 outcome (2026-09-16)

Built and tested (all in `tests/`, green on this container via `make test`):

- Evaluation harness with the toolkit scorer, fixed dev/val split, run records, crash isolation.
- Exact perception library and the local env wrapper mirroring the gateway's reset billing.
- Two CPU baselines measured (research log exp-000..002): random 0.19, explorer 0.06 on all 25 games.
  Both confirm that blind search is worth ~0 under RHAE (`docs/lessons/0003-*`).
- **Control arm on Kaggle RTX: exp-003c dev 0.56** (single 27B REPL agent, 20 min/game, 8 concurrent, no errors). exp-003
  (0.53) had the context-overflow bug. Single runs are noisy at the 1-2 level scale (research log).
- **exp-005 dev 1.108** (tracker + navigation planner + Method, no rule library): 8 levels, 924 actions; best single run.
- **exp-007 dev 0.839** (same + rule library v3 in the Method): 5 levels like the control but with half the actions
  (755 vs 1415); single-run noise is about two levels, so the exp-005/007 order is not yet a conclusion.
- **exp-008 dev 0.734** (fitter v5 + per-turn rules line + goal hints + probe suggestions): 5 levels, 816 actions; the
  automatic rules line did not change how the model plays. exp-009 (navigation-first Method + all helpers) is running;
  a repeat run follows to measure noise; exp-010 (council v2) is queued. Transcripts show the binding constraint: ~35 model
  calls per game, 2.5 inspection-only calls per turn (`docs/postmortems/exp007-inspection-budget-2026-09-16.md`).
  exp-008 (fitter v5, per-turn rules summary, goal hints, probe suggestions) is queued on Kaggle.
- The REPL agent (Duck-style, persistent sandbox, image + ASCII + helpers, eviction, governor,
  explorer fallback). **First real-model run 2026-09-16 (diag v5 smoke)**: plumbing works end to end on Kaggle;
  the model acts too rarely per turn and prompts are too large (see research log exp-003a).
- Entity tracking (`arc3/entities.py`, plan-100 II.2.A): persistent ids across frames, per-action events (moved/appeared/
  disappeared/recoloured/reshaped), roles (static, hud, avatar with its key map), multi-part sprite groups, tile size;
  in the REPL as `ents()/events()/avatar()/roles()` and in every per-action summary the model reads. Unmeasured (exp-005).
- Navigation planner (`arc3/planner.py`, plan-100 II.2.C first slice): avatar move rule and obstacles fitted from evidence
  (including bumps into invisible walls), BFS `plan_to_entity`/`plan_to`, and a predictor the verifier checks; the system
  prompt now prescribes the procedure (keys once, target, plan, verified execution). Unmeasured (exp-005).
- Executable world-model hooks in the sandbox: `set_model(predict)` checks every real action against the model's own
  predictor (mismatches stop batched actions and are reported each turn); `transitions()` logs the level's
  (before, action, after) triples; `verify_model(predict)` replays them and returns counter-examples (plan-100 §3.2).
  Unmeasured on a real model.
- The council agent (`arc3/agents/council.py`): the user's six-specialist + coordinator design as an ablation arm,
  specialists run concurrently on a second vLLM server (Qwen3-VL-8B-NVFP4) or on the coordinator model; mock-tested,
  **not yet run against real models**. Reworked 09-16 afternoon for exp-010: specialists read the coordinator's
  full observation (entities, key map, auto-fitted rules, notes, tile map); rounds are event-driven (level start,
  mismatch, idle turn, stagnation, periodic floor) and run concurrently with the coordinator's call, their reports
  injected before its next call; prompts rewritten around the helpers. Coordinator: adaptive reasoning effort
  (raised only on stagnation), level-completion and stagnation notices, per-call "# goal | learned | now" header,
  optional preserve_thinking. `scripts/build_eval_notebook.py` runs any agent on a split on Kaggle's RTX.
- Kaggle path: framework adapter with shared deadline and crash recovery, notebook builder that
  embeds the package and optionally starts vLLM, tests for both.
- Rule library (`arc3/dsl.py`, plan-100 II.2.B): rule types Move (walkable/blocking colours per cell, required colour),
  Push, Drift, Vanish, OnOverlap, Recolor, OnClick (move/vanish/recolor/goto), periodic Counter; `fit` enumerates
  parameters and keeps contradiction-free rules, `auto_rules` selects a set, `explain` reports coverage and the
  unexplained (transition, entity, event) items, `predictor` feeds `set_model`, `plan` searches simulated frames,
  `goal_predicates` lists win conditions consistent with completed levels. Tracker: occlusion-aware events, static
  layer, compound sprites, canonical terrain state. In the REPL as `auto_rules()/plan_rules()/rules_predictor()/
  goal_candidates()`; the prompt's Method uses them. Code-only coverage gate: mean 0.33 over the 25 public games
  (11 games >= 0.63, two at 0). Regression-guarded in `tests/test_rule_coverage.py`. Model-driven effect: exp-007 (running).
- Per-game transcripts (`runs/<name>/<game>.transcript.jsonl`) and `scripts/transcript_report.py` for post-mortems.

Next on the GPU box, in order:
1. `make setup && make games && make test`; start vLLM with Qwen3.8-27B-FP8 (`arc3.serve.build_vllm_command`)
   and fix the speculative-decoding flag for the installed vLLM version.
2. exp-003: REPL agent on `smoke` then `dev` with `TIME=900`; record s/action, tokens, VRAM.
3. A/B: Qwen3.6-27B-FP8 (Milestone-1 winner's model) vs Qwen3.8-27B-FP8, same harness.
4. Build the vLLM wheelhouse + model datasets on Kaggle; `make notebook`; one private validation run.
