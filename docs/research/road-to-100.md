# Road to 100 percent: what the evidence says and the ranked plan (research, 2026-09-17)

Requested by the owner on 2026-09-17 ("do a research to get 100 percent"). This is the evidence gathered in that
session and the plan it supports. Every number below is either from a run in `runs/` (cited by name), from a
document read on the date given, or marked as vendor-reported. Nothing here is a measured gain of ours until the
research log says so. `docs/plan-100.md` remains the design; this document says what the field has shown since,
what the gap is made of, and in what order to attack it.

## 0. Outcome first

- **Nobody is at 100 percent on hidden games.** ARC Prize's verified leaderboard (read 2026-09-17, dated 2026-09-14)
  has GPT-6 Astra at 62.7 percent, Claude Opus 5 at 30.2, GPT-5.6 Sol at 7.8, with their own harness and thousands of
  calls per game. The Kaggle public leaderboard leader is 18.81 (Tufa Labs, read 2026-09-16). The two systems that
  report 100.00 (Tycho with Opus 5 or GPT-5.6 Sol) do so on the **public** set, which those labs' own analysis calls
  saturated, at 600 to 1,800 calls and 3,000 to 6,000 USD per 25-game run.
- **Our position:** six-run dev base 0.61 to 1.37 RHAE (6 to 9 levels of 19 games), val 0.79 and 0.0; the 3,600 s
  arm (exp-017) 1.29 with 10 levels. Public-leaderboard percent and our dev RHAE are on the same scale.
- **What the gap is made of, in order:** the model (a frontier model with typed evidence alone reaches 79 in Tycho,
  a 27B open model with a comparable harness reaches about 1), the operating point (calls per game and output tokens
  per call: 10 to 30x fewer for us at 9 hours on one GPU), then harness structure (worth about +9 RHAE at the
  frontier: orchestrator over direct reasoning), then learning across games.
- **Honest target:** 100 percent on the hidden set is not reachable with any open model that fits one 96 GB GPU at
  today's evidence; the grand prize rule makes it the direction, not the November plan. The measurable steps that
  move us toward it, each cheap to test once GPU quota returns on 2026-09-19, are in section 6. The first milestone
  that is realistic by 2026-11-02 is passing the Kaggle public leader (about 19), which needs a 10 to 15x gain over
  our base; the plan below is ordered by expected gain per GPU hour.

## 1. What 100 percent means (recap of plan-100 section 1)

Every level of every hidden game completed, at or above human efficiency on the level-weighted average
(`min((baseline/actions)^2, 1.15)` per level, unsolved = 0, later levels weigh more). The 1.15 cap gives a little
slack for an inefficient level; an unfinished level cannot be compensated. Compute is free until the 9-hour clock
binds; actions are the cost.

## 2. The field, with dates

| Source (read 2026-09-17 unless noted) | Result | What it tells us |
|---|---|---|
| ARC Prize verified leaderboard, 2026-09-14 | GPT-6 Astra 62.7; Claude Opus 5 30.2; GPT-5.6 Sol 7.8 (hidden games, ARC Prize's harness) | the frontier ceiling on hidden games is far below 100 even with unlimited calls |
| ARC Prize semi-private failure analysis | GPT-5.5 0.43 percent, Opus 4.7 0.18 percent with three failure modes: a local effect learned without the global rule; the wrong abstraction imported from other games; a level solved without the game being learned | the failures are inference failures, not perception; the same three appear in our post-mortems (dc22, g50t, tn36) |
| Kaggle public leaderboard, 2026-09-16 | 18.81 leader (Tufa Labs), 8.7 second, 5.9 at 15th | the offline, one-GPU, 9-hour regime is a different game; 19 is the bar |
| Tycho, arXiv 2607.28287 and repository (lesson 0013) | Opus 4.8: 79.07 direct, 85.36 single, 88.49 orchestrator, 83.07 trigger; Opus 5 and GPT-5.6 Sol 100.00, all on the 25 public games | harness structure is worth about +9 at the frontier; objective inference is the residual failure; the level boundary and the outcome contract are the transferable parts (exp-020, exp-021) |
| arXiv 2605.05138 (executable world models) | GPT-5.5 high 58.12 on the public set, 15 of 25 games, with a 1,500-action cap per level | "premature commitment" to a goal is the named failure; a world model helps only when the goal is right |
| arXiv 2607.15439 (Rodionov) | model capability and reasoning effort dominate harness architecture; verification is the best structure but the most expensive; the public set is saturating | spend the next GPU hours on the model and the effort knob before on more harness |
| arXiv 2605.25931 ("Explore Before You Solve") | 24 of 25 public games are solvable by systematic exploration alone; their AERA scores 0.21 public and 0.30 private on 55 games | exploration is cheap and general; a harness-owned probe sweep is a candidate that needs no model quality |
| Milestone 1 winner (Tufa Labs, "The Duck") | one 27B model in a persistent REPL, image plus grid, tools hurt | our baseline; its result is the 18.81 class after further work |

Human replay data (342 replays over the 25 public games): the blog's download link (`dub.link/vfwCqvb`) answers 429
from this container; the replay pages (`arcprize.org/replay/<guid>`, 250 guids linked from the blog) load their
data client-side and neither the REST nor the scorecard docs (`docs.arcprize.org/recordings.md`, `rest_overview.md`,
`scorecards.md`, read 2026-09-17) name a download endpoint. Blocked; see section 8.

## 3. Decomposition of the gap

Tycho's four Opus 4.8 arms share one model and differ only in harness: 79 to 88. Our harness has the same
ingredients in reduced form (typed evidence, executable model with verification, level boundary, lessons) and
scores about 1 with a 27B model at 30 to 60 calls per game. The difference between 79 and 1 is therefore mostly
not harness. It splits into:

1. **Model capability at the task** (writing a correct simulator and a correct goal predicate from a few frames).
   Open models that fit one 96 GB GPU are two generations behind the frontier on this; the strongest candidates
   are listed in section 5. This is the largest term and the cheapest to test (swap the served model).
2. **Operating point.** Tycho: 600 to 1,800 calls per game, 24k output tokens per call, xhigh effort. Ours on
   Kaggle: all hidden games concurrent for 9 hours on one server. Measured (diag v5, exp-003): 94 tok/s single
   stream, 308 tok/s aggregate at 8 concurrent with the 27B FP8 and MTP. Section 4 does the arithmetic.
3. **Harness structure.** Worth about +9 RHAE at the frontier; unmeasured at our scale (exp-019/020 are the tests).
4. **Learning.** Lessons, cross-game sharing and the skill library exist (exp-019 arm); human-replay priors are
   blocked on data.

## 4. The compute budget on Kaggle

Assumptions: N hidden games all concurrent (unknown; plan for 30 to 60), 9 h minus 15 min reserve. Aggregate
throughput T tok/s from one server. Tokens available per game = T x 31,500 s / N.

| T (tok/s) | N | tokens per game | calls per game at 2k output | at 8k output |
|---:|---:|---:|---:|---:|
| 308 (measured at 8 concurrent) | 30 | 323k | 160 | 40 |
| 308 | 60 | 162k | 80 | 20 |
| 900 (unmeasured; vLLM scales sublinearly with batch) | 30 | 945k | 470 | 118 |
| 900 | 60 | 472k | 236 | 59 |

Tycho used 600 to 1,800 calls per game at up to 24k output tokens. So even the optimistic row is an order of
magnitude below the regime where a frontier model reaches 100 on the public set. Levers that raise T: batch
concurrency (measure at 16 and 32), MTP acceptance (2.4 to 3.0 measured), prefix caching, FP8 KV, an MoE model
with fewer active parameters (section 5). Levers that lower tokens per call: reasoning effort, the image, the
observation size. Every model choice must be judged on RHAE per GPU hour at the concurrency of the real run, not
single-stream tok/s (CLAUDE.md).

## 5. Candidate models that fit one RTX PRO 6000 (96 GB, SM 12.0, offline)

All facts vendor- or forum-reported unless marked measured; gathered 2026-09-17. "Hub" is whether a Kaggle model
or dataset copy already exists (no upload needed).

| Model | Size on GPU | Vision | Why it might help | Serving recipe on SM120 | Hub | Risk |
|---|---|---|---|---|---|---|
| Qwen3.8-27B-FP8 (current) | 29 GB weights, 53 GB KV | yes | baseline; MTP, effort knob | vLLM 0.27.1 wheelhouse, TRITON_ATTN, MTP, FP8 KV (measured, diag v5) | yes | known ceiling (~1 RHAE) |
| Qwen3.8-27B-NVFP4 | ~22 GB | yes | more KV, more concurrency, same model | same flags; NVFP4 kernels on SM120 unverified | uploaded (private) | accuracy loss; kernel support |
| gpt-oss-120b (MXFP4, MoE 5B active) | ~63 GB | no | strongest open reasoner and coder that fits; effort knob low/medium/high; fast per token (few active params) | `--quantization mxfp4 --mxfp4-backend CUTLASS --attention-backend FLASHINFER` with `FLASHINFER_CUDA_ARCH_LIST=12.0f` (JIT for SM120), `--enforce-eager` reported for stability; ~4,630 tok/s aggregate reported on one RTX PRO 6000 | `danielhanchen/gpt-oss-120b` | FlashInfer JIT offline (our diag v2 to v4 failed on FlashInfer cubin download; JIT needs nvcc 12.8, present); text-only: loses the image |
| Nemotron 3 Super 120B-A12B (NVFP4, MoE 12B active, hybrid Mamba; **NVIDIA Nemotron Open Model License, not OSI open source: owner's call before any submission use**) | 77 to 80 GB | no | strong open agentic model; fast per token | vLLM 0.20+ flags `--kv-cache-dtype fp8 --mamba-ssm-cache-dtype float16 --max-num-seqs 32 --reasoning-parser super_v3`; MTP OOMs on 96 GB; ~94 tok/s single stream reported | `sivavoleti/nemotron-3-super-120b-a12b-nvfp4` (faithful mirror with the parser plugin) | 16 to 19 GB left for KV: low concurrency; text-only; license |
| Devstral Small 2 (24B dense, Apache-2.0) | 48 GB BF16 / 24 GB FP8 | no | 68 percent SWE-bench Verified (vendor): the best open coder at this size | standard vLLM; FP8/NVFP4 community quants | no (hub has 2505/2507 only): upload needed | text-only; no MTP; unproven on game inference |
| Qwen3.8-Flash-Next-NVFP4 | 133 GB checkpoint, needs PLE offload | yes | newer architecture | vLLM newer than 0.27.1; unverified | dataset uploaded (docs/models) | may not serve at all on 96 GB |

Reading: the two MoE models trade the image for 5 to 10x more tokens per second, which per section 4 is the
lever we are shortest on; Tycho found the text grid sufficient with a frontier model, the Duck found the image
helped a 27B. Both are A/B questions, not decisions. Order of the serving checks (task #40): gpt-oss-120b (recipe
and hub copy both exist; the MoE backend on SM120 is the thing to confirm), then Devstral (needs an upload and a
quant); Nemotron 3 Super was dropped on 2026-09-17 (section 8: license and KV headroom). A serving check is one diag kernel (about
15 min of quota); a dev run is 3 h. Decision rule: a model earns a dev run only if its aggregate tok/s at 8
concurrent is at or above the 27B's 308 and its two-game REPL smoke solves at least what the 27B did.

## 6. Ranked plan (expected gain per GPU hour, highest first)

Each item is a hypothesis with a gate; the six-run dev base and the noise rule (three runs a side, ±0.3 RHAE) apply.

1. **Model swap A/B (section 5).** Hypothesis: a stronger open reasoner at more tokens per second raises levels solved
   more than any harness change. Cost: 15 min per serving check, 3 h per dev run. Gate: dev RHAE and levels above
   the base with val not worse; seconds per action at the real concurrency.
2. **Effort and budget arm at the real operating point.** Hypothesis (Rodionov): reasoning effort medium or high with
   the per-level action-budget notice (exp-020) converts our unused wall-clock into levels; exp-017 showed time is
   not the binding constraint at 3,600 s per game (4,634 actions, 10 levels). Cost: one 3 h run per setting. Gate: levels
   solved and actions per solved level.
3. **Exploration-first probe sweep (harness-owned, code-only, buildable now).** Hypothesis (2605.25931 and our
   post-mortems): a systematic sweep at level start (each legal key once, one click per entity class, capped) recorded
   as an effect table costs 5 to 20 actions and removes the premature-commitment failure; every public level's human
   baseline exceeds that. Built as a config flag off by default; task #39. Gate: actions to the first goal-directed
   sequence and levels solved on dev; no loss on levels with baselines under 20.
4. **Goal-hypothesis discrimination.** Hypothesis (Tycho's residual failure, 2605.05138): keep competing goal
   predicates, score them by partial-progress signals (a counter moving, an entity changing state) and propose the
   cheapest probe on which they disagree; verify against observed terminal frames (exp-021 built the predicates;
   recall 9 of 17). Gate: goal-probe recall and levels solved.
5. **Level boundary and lessons (exp-019/020, built, unmeasured).** Run first after the reset because the notebooks
   are ready; three runs a side.
6. **Actor-requested builder role.** Port of Tycho's orchestrator on the parked council code, after 1 to 4. Gate:
   beats the best single-model arm on dev, holds on val, fits the time budget.
7. **Human-replay priors.** Blocked on data (section 8); once available: probing order and first-goal-directed-move
   statistics per game type as prompt priors and as a click-target ranker (CLAUDE.md item 6).
8. **Fine-tune on (game, program) pairs.** Only if item 1 shows the open models cannot write the simulators at all;
   needs the local workstation for weeks. Not before October.

## 7. What would change this plan

- If gpt-oss-120b or Nemotron serves offline at the reported throughput and solves more dev levels than the 27B,
  the image ablation and the effort arm run on that model and the 27B becomes the fallback.
- If no candidate beats the 27B, the effort arm (item 2) and the sweep (item 3) are the remaining levers and the
  November target is the leaderboard bar, not 100.
- If the hidden set turns out to be much larger than 60 games, the budget in section 4 halves again and the
  harness must park games deliberately (plan-100 section 3.9).

## 8. Blocked and follow-ups

- Human replays: needs a browser download of the dataset from the ARC Prize blog (the short link is rate-limited
  from this container) dropped into a private Kaggle dataset or the repo's `data/` (git-ignored). Owner action.
- Serving checks (task #40) need the 2026-09-19 quota reset; exp-019/020 go first (six runs, about 18 h of the 30). The
  two diag kernels are built (`scratchpad/nb/diag-gptoss`, `scratchpad/nb/diag-nemotron`; research log 2026-09-17).
- The NVFP4 27B A/B remains unrun (dataset uploaded 2026-09-16).
- Decisions 2026-09-17 (owner delegated): code license Apache-2.0; Nemotron 3 Super not used for the submission (license,
  KV headroom); gpt-oss-120b is the candidate model; no competition submission unless the code is verified end to end,
  and the owner's "over 50" condition cannot be met by any run of ours (section 0), which is stated in `docs/status.md`.
