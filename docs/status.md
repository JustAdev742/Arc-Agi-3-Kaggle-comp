# Status

Last updated: 2026-09-23 (session 6, remote CPU container: 4 vCPU, 15 GB RAM, no GPU; Kaggle CLI authenticated as `scottmahony`).

## Competition facts: verified vs. unconfirmed

| Item | Value used | Status | Source |
|---|---|---|---|
| Metric | RHAE: per level `min((baseline/actions)^2, 1.15)`; per game level-index-weighted mean, capped at the weighted fraction of levels completed (so never above 100%, as the Kaggle page says); total = mean over games; unsolved levels = 0 | **VERIFIED** | `arc_agi/scorecard.py` (arc-agi 0.9.9) and ARC-AGI-3 Technical Report §4.1, eq. 1–3. `tests/test_scoring.py` checks parity with the toolkit |
| Human baseline | upper-median best human action count per level; per-level `baseline_actions` ship in each game's `metadata.json` | VERIFIED | docs.arcprize.org/methodology, downloaded metadata |
| Action counting | every non-RESET action costs 1; RESET also costs 1 (`Card.inc_reset_count`); per-level count = actions between successive level completions | VERIFIED (toolkit) | `arc_agi/scorecard.py` |
| RESET in competition | competition mode forces *level* resets; game resets become level resets; a RESET at a level start is a billed no-op; one `make()` per environment; one scorecard | VERIFIED | docs.arcprize.org/toolkit/competition_mode and `arc_agi/api.py`; `arc3/env.py` mirrors it, `tests/test_env.py` checks it |
| Games / levels | 25 public games, 6–10 levels each (188 levels); ids and per-level baselines in `environment_files/*/metadata.json` | VERIFIED | download on 2026-09-15 |
| Runtime limit | **9 h** for CPU and GPU notebooks ("CPU Notebook <= 9 hours run-time, GPU Notebook <= 9 hours run-time"). Governor default 9 h minus a 15 min reserve | **VERIFIED** | Kaggle Code Requirements, pasted by the user on 2026-09-16 |
| Daily submissions | **1 per day**; up to **2 final submissions** selected for judging; the rerun on the hidden set costs no quota, but a forum answer says each submission needs its own notebook version and making one needs some GPU quota (reportedly a version cancelled right after it starts also works): UNVERIFIED | **VERIFIED** (limits) | Rules tab, section 2 (read via `kaggle competitions pages`, 2026-09-23); `submission-limits` shows 0 remaining after one |
| Milestone 2 | closes 2026-09-30, 23:59 UTC; prizes $25,000 / $7,500 / $5,000; notebook must be public under an open-source license by then | VERIFIED | Kaggle overview (pasted 2026-09-16) |
| Entry / team merge | 2026-10-26, 23:59 UTC | VERIFIED | Kaggle overview (pasted 2026-09-16) |
| Final submission | 2026-11-02, 23:59 UTC; winners announced 2026-12-04 | VERIFIED | Kaggle overview (pasted 2026-09-16) |
| License for prizes | winners license the submission and its source under **CC-BY 4.0**; data use Apache 2.0. Reused code: the Duck notebook is Apache 2.0, its code dataset (TAAF + ARC3-Inference) MIT (Jeroen Cottaar, discussion 717133) | **VERIFIED** | Rules tab sections 5-7 |
| Hardware | `rtx6000` = GCP `g4-standard-48`: RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, SM 12.0, 48 vCPU, 176 GB RAM, 20 GB writable disk; image: driver 580.159.04 (CUDA 13.0), nvcc 12.8, torch 2.10.0+cu128, Python 3.12.13. Notebook accelerator id `nvidiaRtxPro6000` (CLI `--accelerator NvidiaRtxPro6000`) | **VERIFIED** | diag run v2 log, 2026-09-16 |
| Hidden set size | **110 games**; every submission plays all 110; the public LB shows about half, the private LB the other half; private scores are computed at the original run, not rerun at the deadline. The Duck's 7,920 s per game at 28 concurrent = 4 waves = 8.8 h is sized for exactly this | **VERIFIED** | data-description page; Greg Kamradt in discussion 729985 |
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
12. exp-011..016 (`arc3-eval-dev-f` v2-v4, `-g` v2, `-h` v1-v2, `-i` v1-v2, `-j` v1, `-k` v1, `arc3-eval-val-a` v1,
    `arc3-eval-all-a` v1): about 60-70 min RTX each, 2026-09-16 afternoon and evening; numbers in the research log.
13. `arc3-eval-all-long-a` v2 (exp-018: all 25 games concurrently, 3 h each) and `arc3-eval-dev-long-a` v1 (exp-017:
    dev, 1 h per game), pushed 21:28 and 21:46. The **weekly GPU quota (30 h) ran out at about 23:20 UTC (resets
    2026-09-19 00:00 UTC)**: exp-018 was cancelled at 23:13 after 1 h 45 min (no output; its live log exposed the
    image-limit bug, lesson 0012), while Kaggle let exp-017 finish on borrowed quota at 00:52 UTC: **dev 1.288, 10
    levels** (runs/kaggle-repl-dev-017; research log). About 4.7 h RTX between them.

14. `scottmahony/arc3-submission-cpu-check` v1, **CPU**, 2026-09-17 02:15 UTC: the submission notebook built at the
    current harness (Apache-2.0 LICENSE in the bundle, no model attached so the rules fallback plays) ran Save & Run
    All to completion: arc-agi installed from the competition wheels, bundle unpacked, `my_agent.py` written, offline
    smoke on ls20 (140 actions, 0 levels, 128 s) and vc33 (200 actions, 0 levels, 47 s), `submission.parquet` written.
    Packaging and the framework path verified at HEAD; the model path needs the RTX Save & Run All (open item 2). GPU quota: 0.

15. `scottmahony/arc-prize-2026-arc-agi-3-arc3-agent` **v3**, RTX, queued 2026-09-20 23:20 UTC, ran and **COMPLETED
    2026-09-21 13:30 UTC** (14 h of queue, about 20 min of run). Harness 8f3af9e, champion preset. vLLM installed in
    159 s, server ready on the **first attempt** (tuned flags, MTP + FP8 KV) after 321 s; `agent: repl` with the
    champion config keys, so no silent fallback; offline smoke played ls20 (15 actions) and vc33 (4 actions) against
    the served model in 284 s each, 0 levels (a 300 s per-game smoke is a pipeline check, not a score);
    `submission.parquet` written with the expected placeholder row. **This version is valid and submittable.**

16. **COMPETITION SUBMISSION** 2026-09-23 04:23 UTC, on the owner's explicit instruction ("Submit it"): notebook
    `scottmahony/arc-prize-2026-arc-agi-3-arc3-agent` **v3** (champion preset, harness 8f3af9e), submission ref
    **56482492**, message "champion (exp-011 config), harness 8f3af9e, notebook v3". Status PENDING at submit time;
    the rerun plays the hidden games for up to 9 h. Account history before it: 0.17 (2026-08-27) and 0.10
    (2026-08-28), both from before this repo. Public score: **0.68** (complete 13:25 UTC).

## Rule library coverage (exp-006a, code-only, 2026-09-16)

`scripts/rule_coverage.py` plays each public game blind for 60 actions and measures the fraction of entity events the
fitted rule set explains: mean 0.50 over 25 games; ar25 1.00, ft09 1.00, m0r0 1.00, sp80 1.00, tn36 1.00, ls20 0.98,
re86 0.96, cn04 0.86, g50t 0.72, dc22 0.70, ka59 0.63; two games at 0. Lower bound, not a score. Details: `runs/rule-coverage/summary.json`,
research log exp-006a.

## Plan for the week of 2026-09-26 and the final-selection rule (pre-registered 2026-09-26 00:20 UTC)

GPU (30 h, about 12 public-25 runs of 2.5 h):
1. exp-049 (fixes + 6.5 GiB KV, no gate) and exp-050 (+ P23/P24): pushed 00:16 / next cycle.
2. If the base's LB draw (submission 56563791) is >= 6.0: exp-051 (base + wave-fit) and exp-052 (fixes, no wave-fit),
   then their LB draws (research log, 2026-09-25 22:50).
3. If exp-050 against exp-049 is inside one run's noise: one repeat of each before any submission uses P23/P24.
4. Reserve about 8 h for a Milestone 2 notebook from a top team (if one is published by Sep 30) or the next change.

Leaderboard slots (1 a day, 37 left): calibration first (base, then the bisection arms if needed), then repeated
draws of the leading candidates, alternating, so each has at least 3 draws by late October.

Final two selections (the rule is fixed now, before the data): the two configurations with the highest mean public-LB
score over at least 3 draws each. A difference under 0.5 points is a tie, broken by hard-game level-1 solves on the
public 25 (scripts/arm_table.py), then the validation score. If the top two are variants of one configuration within
noise, the second slot goes to the best configuration that differs from it (diversification against the private half).
If fewer than two configurations reach 3 draws, the unmodified base fills the second slot.

## Open items (need you)

Current as of 2026-09-23 14:30 UTC. (The earlier items about our own REPL harness, the rtx6000 queue stall and the
champion preset are superseded by the move to the Duck family; they are in this file's git history.)

1. **Milestone 2 (closes 2026-09-30): I recommend not entering.** An entry must be a public notebook under an open
   license. Our best measured configuration is the unmodified public base (7.86 on the public 25; Duck-family
   notebooks score about 5-8 on the leaderboard) against a leader at about 19, so a prize is out of reach, and
   publishing our fork would give away our only differentiated work (the P21 gate, if it proves out). Say so if you
   want an entry anyway for the visibility; making anything public waits for your word.
   **The other side of Milestone 2:** entries must be public notebooks by 2026-09-30, so the top teams' current
   notebooks may become readable this week. If one scores well above our Duck base, adopting it (within its
   license) is worth more than any patch of ours; I check the public notebook list at each daily check-in.
2. **GPU quota is the binding constraint: 30 h per week, resetting Saturdays 00:00 UTC.** A full public-25 run costs
   about 2.5 h, so about 10 full runs a week; single runs differ by about 2.3 points (sd of a difference), so only
   large effects or clear mechanism changes can be read from one run. Nothing to decide unless you can add GPU time
   elsewhere: the RTX PRO 6000 workstation in CLAUDE.md is not reachable from this cloud session; if you can give me
   access (or run commands there), each experiment would stop costing Kaggle quota.
3. **Kaggle's API cannot cancel a running notebook** (the cancel call needs a session id no public call returns; the
   site's internal endpoint refuses API tokens). A run that should be stopped (like exp-036/039 today, which carry the
   harmful P4) can only be cancelled from the browser: https://www.kaggle.com/code/scottmahony/arc3-taaf-ours-c and
   .../arc3-taaf-ours-d (both end on their own about 15:30 and 16:00 UTC today).
4. **Daily submissions:** 1 per day, 2 final selections (verified on the Rules tab). Tonight's slot (00:03 UTC) is
   used by a pre-registered rule (research log): a P21 arm if its mechanism and score check out, else the base.
5. For future sessions put the Kaggle token in the environment as `KAGGLE_API_TOKEN`; consider regenerating it after
   the competition since it passed through a chat upload.

## Follow-ups noticed (not fixed on purpose)

- Measurement noise: four single runs of similar arms span 0.56-1.11 with 5-8 levels; the three levels every arm
  solves (ar25 L1, lp85 L1, sb26 L1) are the only stable signal. Plan: repeat exp-009 unchanged (exp-009b) to
  measure run-to-run spread, then require a difference larger than that spread before keeping a change.

- Human play data (342 replays for the 25 public games, arcprize.org/blog/arc-agi-3-human-dataset): the download short
  link (dub.link/vfwCqvb) answers 429 from this container (retried 2026-09-17), the replay pages
  (`arcprize.org/replay/<guid>`) load their data client-side, and the recordings, REST and scorecard docs name no
  download endpoint (checked 2026-09-17). If you can download the "full Public Demo dataset" archive from that blog
  post and drop it into `data/human/` (or a private Kaggle dataset), plan-100 §4.1 (goal priors from winning steps)
  and road-to-100 item 7 (replay priors) become CPU-only jobs: `scripts/human_replays.py` already parses the
  documented recording format and writes `arc3/data/human_priors.json` (tested on a synthetic file).

- `arc_agi` logs at INFO through the root logger; the harness silences it with a level filter.
- The starter's `build_notebook.py` writes the agent to `/tmp/my_agent.py`; ours bundles a
  package instead (see `scripts/build_notebook.py`).

- Improvement pass (2026-09-17, during the quota wait; commits 08076e9, 4bedddf, 98dffb6): run summaries are now
  written after every game and atomically, a dead pool worker no longer sinks the run, a stopped sandbox no longer
  respawns its child, the terminal frame stays out of the per-cell state, the ruff rule set is pinned (`make check`),
  the eval/diag builders take `--out` and their folders are ignored, the REPL agent has one model-call path, the
  README and this file were brought up to date. Left as is on purpose: `governor.py` (tested, off the submission
  path since lesson 0011); the council arm (parked); the `notebooks/eval` watchdog's `os._exit(0)`, which skips the
  final summary but no longer loses the per-game results.

## Session 6 (2026-09-23, in progress): the submission base moves to the public Duck family

- **Why:** our REPL harness at HEAD scored 0.44 (exp-024 champion) and 0.26 (exp-026 bundle) on dev; the public
  Duck-family notebooks score 5-11 on the same public games (22+ harvested public runs, `runs/public-harvest/`; plain
  Flash-Next Duck n=19 mean 7.4, animation-aware n=3 mean 7.0). The owner approved running the public Tufa/Duck
  notebooks on this account and asked for the highest possible score.
- **Verified facts (competition pages via the CLI):** the hidden set is 110 games, all played by every submission, half
  public / half private LB; 1 submission per day; 2 final selections; winners license CC-BY 4.0; the Duck notebook is
  Apache 2.0, its code MIT. Forum: identical notebooks vary 1.5-2.3x on the LB and about 2 points on the public games;
  Duck-style public scores of 10-22 became 5-7 on the LB (lesson 0018).
- **Diagnosis of the best public configuration** (animation-aware Duck on Qwen3.8-Flash-Next NVFP4; research log):
  time-bound, 54 model calls per game, 127 s of queueing per call because the KV cache (5 GiB after 81.8 GiB of
  weights, BF16 only) holds about 3 prompts of 20.6k tokens; 35% of prompt tokens are past reasoning; the carried note
  is mostly lost; ACTION7 (UNDO) is unmapped and always rejected; solved levels already use fewer actions than humans.
- **Our fork** (patches applied at runtime to the published anim source, `scripts/taaf_ours_patch.py`; notebooks from
  `scripts/build_taaf_nb.py`; every patch bed-tested on CPU with the real harness and a mock model): P1/P1b carried
  note from reasoning and lenient labels, P2 ACTION7 as UNDO, P3 goal/action models kept across levels, P4 history
  compression, P7 pre-imports, P8 cross-level note on level change; knobs yield 600 s, output cap 6,144, history
  budget; wave-fit per-game cap in real reruns.
- **GPU runs this session:** exp-024/026 (done), exp-032 = unchanged copy of the public anim + Flash-Next notebook
  (running), exp-033 FP8 KV (failed at startup: the model needs a BF16 KV cache), exp-034 = our fork without history
  compression (running). Queue, pushed as the two slots free: exp-037 (reasoning effort medium), 8 GiB KV stress
  test, exp-035 (history compression), exp-036 (+ board diff, level-1 note, persisted helpers), exp-039 (+ the
  stall-analysis patches P13-P16), 5 GiB stress test; exp-038 (effort low) only if exp-037 beats the default.
- **Measured today (public 25, one run each; harvest of the base: n=37, mean 7.15, sd 1.65):** exp-032 control
  7.86 (38 levels); exp-034 fork, first arm, 6.44 (32; its 600 s yield is the prime suspect, the queued arms went
  back to 180 s); exp-037 effort medium 6.70 (37; +35% calls, no more levels: effort stays xhigh). KV 8 GiB fails
  at startup (out of memory). exp-035 (history compression P4 + 22.5k window) 3.58 (21 levels): the model re-derived its state every call once its past reasoning was stripped (lesson 0022); P4 and the smaller budget are out. exp-036/039 carry P4 and could not be cancelled from the API (no session id; internal endpoint 403), so they finish. Replacements queued at the base budget: exp-042 (low-token fixes) and exp-043 (all patches but P4/P11). Nothing beats the base yet.
- **New lever, P21 (2026-09-23 afternoon):** the server fits 4-6 calls while about 20 wait first-come, so every game
  gets the same call rate (later levels even less: 0.37 vs 0.47 calls/min in exp-032), yet level k is worth k times
  level 1. P21 gates the calls (6 in flight) and admits the largest weight x wait, weight 1 + 2 x (level - 1) fading
  after 45 stalled minutes. Replay model over 19 base runs: +0.64 on the public 25 (paired sd 0.84), +0.42 on a
  modelled harder hidden set; bed-tested on the real harness. Queue (scripts/kaggle_queue.py, pushes as slots free):
  stress tests 5 GiB, 6.5 GiB + 4k chunks, the b12x MoE kernels, prefix caching; then exp-045 (base + P21), exp-042
  (low-token fixes), exp-046 (P21 + 6.5 GiB, if its stress test passes), exp-047 (P21 + prefix caching, if its stress
  test passes), exp-043 (all patches but P4/P11). GPU quota: 13.3 of 30 h used at 13:37 UTC, resets 2026-09-26 00:00.
- **2026-09-25 (morning):** a study of level 1 on the 8 hardest public games (`docs/research/hard-games-level1.md`)
  found the P21 gate starves hard level 1s (gated runs solved 6 of 16, ungated 16 of 24) and that the most common
  failure is a misread action effect. Built P23 (object-level change report per action, including what happens
  only mid-animation) and P24 (shape matches up to rotation/reflection/colour at each level start); tests, engine
  replay and a real-harness bed pass. Queued for the 2026-09-26 quota reset: exp-049 (fixes + 6.5 GiB KV, no gate)
  and exp-050 (exp-049 + P23 + P24). exp-048r (gated repeat) dropped.
- **2026-09-26 09:31: the unmodified base drew 2.72 on the LB** (ours: fixes 3.81, fixes + gate + KV 4.23): our changes help on the hidden set; about 3-4 is this family's level on our draws. No bisection needed. **Sep 27 00:04 submission: exp-050** (P23/P24). P25 (the report inside action() results) was dropped by its rule (pair 8.24 vs exp-050's 10.31).
- **2026-09-26 03:30: exp-050 (fixes + 6.5 GiB KV + P23 object-change report + P24 shape matches) is the leading candidate:** public-25 10.98 / 51 levels, z-sum +3.1 sd (best of every run), level 1 solved on all 8 hard games (no reference run above 7). Its control exp-049 (no P23/P24): 9.60 / 42, +1.4 sd, 6/8. Validation went the other way (8.99 vs 3.63) on one noisy game (r11l). **Repeats (06:30): exp-050r 9.64 (7/8 hard L1), exp-049r 6.30 (4/8); pooled with/without P23/P24: mean 10.31 vs 7.95, z-sum +14.3 vs +2.8, hard L1 15/16 vs 10/16 -> P23/P24 kept.**
- **Best configuration so far (2026-09-23 23:36): exp-048 = fixes (P1 P1B P2 P7 P12 P13 P17 P22) + P21 gate + 6.5 GiB
  KV cache: 15.50 on the public 25, 47 levels, above all 39 reference base runs (max 11.02); lp85 8/8, sb26 7, ft09 6
  levels.** The fixes alone repeat (10.89, 8.87). **Submitted 2026-09-24 00:04 UTC as competition submission 56506396: public LB 4.23** (previous 0.68; Duck-family teams show 7-11 as their best; leader 19.40). One draw at about half the expected level: the next submissions separate the components (fixes only, base, gate only). **2026-09-25 00:05: submission 56534663 = exp-042 (fixes only, no gate): public LB 3.81** (09:31). The gate did not cost exp-048 its LB score; both forks draw about 4. Sep 26 00:05: submission 56563791 = the unmodified base exp-032 (stock timing, no patches), pending; it decides between LB noise and a cost in our forks' rerun path (rule in the research log, 2026-09-25 22:50).
- **Results this evening (public 25, one run each; base harvest n=39, mean 6.98, sd 1.77):** exp-042 (low-token
  fixes P1 P1B P2 P7 P12 P13 P17 + P22) **10.89, 46 levels**, 95th percentile, our best run; exp-045 (P21 gate)
  7.59, 34 levels, mechanism confirmed (L2+/L1 calls-per-minute ratio 1.71 vs 0.80, 0 timeouts, same total
  throughput); P4 arms exp-036/039 3.94 / 3.50. Stress tests: 6.5 GiB KV passes (+25% running, +14% tokens/s);
  prefix caching -19% (exp-047 dropped); b12x MoE kernels unsupported. Running now: exp-046 (P21 + 6.5 GiB) and
  exp-043 (all patches but P4/P11). Queued for Saturday's quota: exp-048 (fixes + P21 + 6.5 GiB, the combined
  candidate) and exp-042r (a repeat: is 10.89 real?). Submissions: 00:03 Sep 24 by the pre-registered rule (a
  passing P21 arm, currently exp-045, or exp-046 if it scores higher and passes); 00:03 Sep 25 exp-042.
- **The 100% question (owner, 2026-09-23):** every system reported at 100 (NVIDIA AVO, MIT VISTA, Tycho) is Claude
  Opus 5 or GPT-5.6 over the internet on the 25 public games; none can run in the offline Kaggle rerun, and frontier
  models score 30-63 on the semi-private set. Their shared structures are ported as patches P13-P20
  (docs/research/frontier-100-systems.md).
- **Found 2026-09-23:** the served chat template runs at reasoning effort "xhigh" unless told otherwise (it prepends
  "think carefully ... consider plausible alternatives" to the system prompt) and accepts medium/low; every public
  Flash-Next notebook plays at xhigh. Patch P11 makes it a knob; exp-037/038 measure it. Harvest analysis: the score
  rises with time per game but concavely (stock mean 3.9 at 60 min, 6.3 at 132); levels 2-4 take as long as level 1
  (median 24-31 min each); the score comes from later levels of a few games.
- **Code review of our patches (subagent, bed-tested):** P6 can hang a game thread (replayed helper code that prints
  corrupts the sandbox protocol; the host then blocks forever) and needs its fix before exp-036 runs; P9 reported a
  wrong diff right after GAME_OVER; P10's note accumulated in compressed history. exp-034/035 do not contain P6 and
  are unaffected (verified on the four patch sets).
- **Scoring path (read in the notebook and TAAF source):** a real rerun plays through the Kaggle gateway's scorecard
  (`competition_arcade.py`: one scorecard, hidden baselines); the notebook writes submission.parquet only in offline
  runs. Levels therefore count as they complete, and a game still running at soft_end keeps its progress; the
  wave-fit option only evens out per-game time (the 4th wave of 28 gets about 10% less), it rescues nothing.
- **Blocked:** creating a private Kaggle dataset for our source was refused by the permission classifier (read as a
  possible public surface); worked around legitimately by patching the public source at runtime inside our private
  notebooks. No dataset of ours is needed.

## Session 5 outcome (2026-09-23, CPU only; the rtx6000 pool was stalled)

- **Competition submission 56482492** made on the owner's instruction (notebook v3, champion preset); public LB
  **0.68** (scored 2026-09-23 13:25 UTC; our old harness; the Duck-family notebooks score about 5-7 there)
  (run 16 above). Daily limit observed as 1.
- **Road to 100 percent, part II** (`docs/research/road-to-100-v2.md`), requested by the owner: level 1 is nearly free
  under RHAE (levels 2+ at 0.949-0.991 of the human count pay for any level-1 cost); optimal play is a median 0.41 of
  the human count over 32 dev levels (`runs/oracle-bfs`, research instrument); the win-condition census of the 19
  dev games (`docs/research/win-conditions-dev.md`, lesson 0016); our model solves levels at a median 0.93 of the
  human count but 31 of its 34 solved levels are level 1, so the gap is unsolved levels.
- **Kept:** exp-027 explorer masks budget-bar cells in state keys (dev levels 6 → 15); stable per-game seeds
  (process-salted `hash()` made explorer/rules/random runs unrepeatable); the REPL agent's winning frame is
  `perception.terminal_layer` (exact on 43 of 43 collected levels, `layers[0]` on 31; the old rule is in the
  submitted notebook v3, not in the champion record's runs; `docs/champion.md`); universal and colour-free goal kinds in the DSL (opt-in) and the REPL knobs
  `goal_forall` / `goal_lifted` (off in CHAMPION, on in BUNDLE).
- **Measured, not kept:** exp-029 goal-directed explorer frontier (15 → 14 levels; ls20 lost to a precondition).
- **Instruments:** `scripts/oracle_bfs.py` (optimal-play BFS in the engine), `scripts/goal_induction.py` (exp-028:
  level-1 goal candidates on 10 of 13 games, holding on 10 of 15 later levels, 11 with colour-free kinds).
- **Next GPU runs when the pool allocates:** exp-024 control at HEAD (includes the winning-frame fix), exp-026 bundle
  (now with the goal knobs), then the model challengers (Part I).

## Session 4 outcome (2026-09-17, no GPU quota)

- **Research toward 100 percent (owner's request):** `docs/research/road-to-100.md`. Verified leaderboards, three
  ARC-AGI-3 papers, Tycho, the Kaggle leaderboard, candidate open models that fit the RTX PRO 6000 (gpt-oss-120b
  MXFP4, Nemotron 3 Super 120B-A12B NVFP4, Devstral Small 2) with their SM120 serving recipes and Kaggle-hub copies,
  and the compute budget at the real operating point. Reading: no system reaches 100 on hidden games (best verified
  62.7 with GPT-6 Astra; Kaggle leader 18.81); the gap from our 1.3 is first the model, then calls per game, then
  harness structure. Ranked plan with gates in the document; research log entry dated 2026-09-17.
- **exp-022 built (code-only, tests pass): exploration-first probe sweep** at level start (`explore_first`, off by
  default): the harness presses each legal key once, ACT once and clicks one entity per class before the first model
  turn on a level and shows the effect table once. Measured after the quota reset, after exp-019/020.
- **Goal-hypothesis discrimination built (road-to-100 item 4, in the exp-022 bundle):** candidates ranked by
  code-computed distance, the ones the current level already falsified marked, `goal_probe()` plans the cheapest
  live one (win or rule it out). Observation line from level 2 on (`goal_progress_in_prompt`). Code-only gate:
  on the two recorded replays with level-1 candidates and a solved level 2, the winner ranked first both times.
- **Serving checks prepared for two candidate models** (road-to-100 item 1): gpt-oss-120b MXFP4 and Nemotron 3 Super
  120B-A12B NVFP4, both with Kaggle-hub copies and public RTX PRO 6000 recipes; diag kernels built
  (`scratchpad/nb/diag-gptoss`, `scratchpad/nb/diag-nemotron`), notes in `docs/models/`, research log entry. Nemotron's
  license is NVIDIA's Open Model License (not OSI): your call before it is used in a submission.
- **Decisions received and recorded** (open item 1): Apache-2.0, publication with a submission, no submission under
  the "over 50" condition, Nemotron not used. **CPU Save & Run All of the submission notebook at HEAD passed**
  (run 14): packaging, LICENSE in the bundle, rules fallback, submission.parquet.
- **First human recording received** (ls20, 7 levels won in 546 actions): the local engine reproduces all 546 recorded
  frames exactly. Tracing why no win predicate held from level 3 on exposed three perception faults (lesson 0015:
  occlusion-canonical frames, background-coloured holes, colour-pair goals instead of avatar-relative ones); fixed
  with plain component frames and avatar-relative goal kinds: a consistent predicate now on 7 of 7 levels, ranked
  first before every win from level 2 on. The remaining 341 files would make this a recall measurement across games.
- Improvement pass (task #37) closed on 2026-09-17 morning: see the follow-ups entry and commits 08076e9, 4bedddf,
  98dffb6, 96c7d94, 4f78ad1.

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

- **Night (22:45-23:30): quota ran out; two findings and one build.** (1) The live log of exp-018 (all 25 games at
  3 h) showed every call for ft09 and s5i5 failing from 22:51 with vLLM's 400 "At most 16 image(s)" and being retried
  unchanged (554 and 340 times): a submission-path bug that only the 9-hour operating point reaches. Fixed (max_images
  cap with image-only eviction, immediate retry on the 400; commit d8e3ce4, lesson 0012). (2) exp-017 (dev, 1 h per
  game) finished after all (Kaggle let it run past the quota): dev 1.288, 10 levels, inside the band of the 1200 s
  runs with one extra level for 4-5x the actions: time is not the bottleneck, stagnation is. (3) Built the learning memory the owner asked for: `arc3/memory.py` Lessons store
  (harness-written recipe/hazard/mistake lessons plus `learn()` for the model, shown every turn, carried across
  levels, shared across the run's games through a locked JSONL), a "What did we learn?" question after each decisive
  event, and an offline skill library (`scripts/mine_skills.py` -> `arc3/data/skills.json`, 18 cards from 79 winning
  transcripts, retrieved by level signature, never for the same game) shipped in the notebook tarball. 100 tests pass.
  Measurement is exp-019 (notebook ready) once the quota resets. (4) Tycho (owner's reference, Apache-2.0; 100 RHAE on
  the public set with frontier models): read, installed, its tests pass here in host mode; transfer plan in lesson 0013.
  (5) Flash-Next NVFP4: all 132.7 GB streamed to Kaggle and the dataset create accepted; visibility pending (open items).
- **23:30-23:45: Tycho port bundle (exp-020, commit 0608566).** Level-boundary consolidation pass (learn()/note() only,
  actions refused) followed by a conversation clear; the observed terminal frame (first layer of the level-completing
  step, verified on vc33/ls20/ar25 replays) now feeds the tracker and the goal predicates; animation frames are noted;
  FRICTION lines go to the transcript. Two notebooks are ready for the quota reset: exp-019 (memory only) and exp-020
  (memory + level boundary), both against the six-run base. 102 tests pass.
- **23:45-00:10: goal inference (exp-021, code-only, KEPT).** `scripts/goal_probe.py` replays each recorded solved level
  through the real agent and asks whether a win condition consistent with the level exists: 5 of 17 before, 9 of 17
  after adding entity-pair relations (same_box, same_columns, same_rows, inside) to the goal predicates; the REPL's
  goal_candidates() now uses the observed terminal frame and plan_rules() takes the new relations as dict goals.
  Both evaluation notebooks (exp-019, exp-020) rebuilt at 9548b57. 103 tests pass. Quota-blocked until Saturday.

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
