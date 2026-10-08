Summary: under D′ every game gets the same 532-minute deadline and slots change hands only at context trims, priced by A·M·C + B·φ. Levels that are never solved take 36% of all slot-time, and the solve hazard does fall with time on a level (≈0.08 per active minute for the first 10 minutes, ≈0.02-0.03 after 30). Even so, in a rerun simulator that reproduces our three full-length runs, no give-up or level cap beats D′: cut-offs at ≤40 active minutes lose 1.7-29 public-25 points, and cut-offs at 50-60 minutes land within ±0.3 with no stable sign. Do not build it; the expected gain is far under 1 point on either scale.

# Time allocation under D′ in the competition rerun: give-ups and level caps do not pay

Research note, 2026-10-08. CPU only: no GPU, no Kaggle, no repo code changed.

- **Measured** means computed from the three full-length runs by the scripts below.
- **Modelled** means taken from the rerun simulator described in §4. §4.1 shows that it reproduces those runs.

## 0. Bottom line

1. **How D′ spends the 9 hours (code).**
   - Every game's deadline is fixed at 532 minutes from the start, and it is the same for all 110 games. Nothing recomputes it from the time left.
   - A game keeps its slot until its next context trim: the first comes after ≈62k generated tokens and later ones every ≈37k, i.e. ≈18 and ≈11 slot-minutes.
   - At a trim the game re-queues. The slot goes to the highest A·M·C + B·φ, which may be the same game.
   - A game that wins releases its slot to the best waiter at once.
   - Nothing in the harness ends a game for lack of progress.
   - B (16/14/10/0 by levels left) dominates until the tail fade, which starts at minute 319. So a game on its last level waits behind every game with levels left.
   - In the modelled rerun with 14 slots, half the games get their first slot after minute 126 and 21% after minute 240. With 10 slots, 2.3 games per rerun never start.
2. **Measured hazard.**
   - Of 391 level attempts, 336 were solved and 55 were censored at the run end.
   - The median solved level took 6.5 active minutes; the slowest solve took 47.7.
   - Kaplan-Meier gives P(solved) = 0.56 by 10 active minutes, 0.77 by 20, 0.85 by 30 and 0.91 by 60 (plateau).
   - After 30 minutes without a solve, P(solve within 30 more) = 0.39 [0.25, 0.55], against 0.85 for a fresh level. After 40 minutes it is 0.23 [0.07, 0.40]. None of the 10 attempts still open at 50 minutes was solved.
   - The hazard falls from 0.08/min in the first 10 minutes to 0.027/min at 30-45 minutes. A censored Weibull fit gives shape 0.90 [0.81, 1.01].
   - Levels never solved took **36.4%** of all generated tokens (25.8-43.4% per run). The time they took beyond 30 / 40 / 50 active minutes is only 9.6% / 5.1% / 2.5% of all tokens.
3. **Modelled rerun** (110 games, 532 min, 14 slots; 200 cluster-bootstrap replicates × 4 rerun draws, paired).
   - "End the game after X active minutes on one level" changes the mean by:

     | X | Change [90% interval] |
     |---|---|
     | 20 min | −20.0 |
     | 30 min | −6.9 [−13.5, −1.7] |
     | 40 min | −1.7 [−4.0, −0.2] |
     | 50 min | −0.2 [−0.8, +0.2] |
     | 60 min | −0.03 [−0.25, +0.14] |

   - The soft cap (re-queue at the lowest priority) is −1.3 to −2.7.
   - With 10 slots the results are the same.
   - **Why it fails.** D′ already rations stuck games: a game that has spent 50 minutes on a level would get only ≈19k more tokens. The time a stuck game does get is worth ≈1.9-2.6 points per 1M tokens to it, against ≈1.0-1.35 per 1M for whoever receives it; the marginal value of capacity is 1.05 per 1M. Short cut-offs also collapse demand, so most freed time goes unused.
4. **Recommendation: no patch.**
   - The best threshold (50-60 active minutes) is worth between −0.9 and +0.3 public-25 points across every variant tried, i.e. −0.5 to +0.2 on the LB at 0.55-0.62×. Its sign depends on assumptions the data cannot settle.
   - The one scheduler tweak with a positive point estimate, giving the last level B = 10 instead of 0, is +0.26 [−0.35, +1.09] in the main model and ≈0 in the harder "hidden-like" worlds. Also no.
   - Two warnings:
     - A public-25 A/B (25 games × 121 min) would show these give-ups as *gains* (+0.7 to +1.2), the opposite sign to the rerun model.
     - The large prize is not in time-based rules. A perfect detector of hopeless levels at level entry would be worth +8.1 [+3.7, +11.7] (oracle bound), and time-on-level cannot provide that signal.

## 1. Sources and method

- **Runs.** All three are D′ with REAP-448 and 14 slots, playing all 25 public games at 121 minutes per game:
  - exp-073 (lossless);
  - exp-073b (MTP acceptance 0.5);
  - exp-075 (exp-073b plus the sandbox fix).

  Files: `runs/<run>/kernel-output/benchmark.json` (per action: action, `generated_tokens` since the previous action, `wallclock_seconds` since the game start), `serve.log` and the notebook log.
- **Code.**
  - Franzen's harness tree is rebuilt with `scripts/franzen_tree.py build` (tag `franzen`). Citations of the form `tool_agent.py:N` and `solver.py:N` are paths under `ARC3-Inference/inference/{agent,framework}/` in that tree.
  - The D′ notebook is `kaggle/dprime/affectify-arc-31-54-in-a-single-sub.ipynb`. Cells are 0-based and are one higher than in Franzen's notebook: D′ cell 5 = his cell 4, D′ cell 17 = his cell 16.
  - D′'s priority module is embedded as a string in cell 21. `embedded:N` cites lines of that module, extracted verbatim to the scratch directory as `ours_form_priority.py`.
- **Scripts.** All are in `/tmp/claude-0/-home-user-Arc-Agi-3-Kaggle-comp/d342458e-03bd-545b-8a6d-06bca061963e/scratchpad/timealloc/`, with outputs in `out/`. All run with `.venv/bin/python -I`.
  - `common.py`: the loaders, the scorer and per-level records. The scorer reproduces `final_score` for all 75 game runs.
  - `rates.py`, `gaps.py`, `quanta.py`, `decode.py`: the token clock, parked time, trim quanta and decode throughput.
  - `km.py`, `weibull.py`, `levelprop.py`: section 3.
  - `sim.py`: the gate simulator. `check_ports.py` verifies its priority functions against the real code: 0 mismatches in 20,000 random states, against D′'s `d_priority` and Franzen's `priority_value`.
  - `validate.py`, `composition.py`, `bootcheck.py`, `rerun.py`, `rerun_diag.py`, `policy_decomp.py`, `oracle.py`, `sens_quanta.py`, `hard_world.py`, `where_time_goes.py`: section 4.
  - `nbdiff2.py`: the D′-versus-Franzen cell alignment. The only code differences are cell 17 (save-run budget) and the inserted cells 21 and 23.
- **The clock is tokens.** A held slot generates 52.6, 56.5 and 56.4 tokens per second in the three runs (`rates.py`: total tokens ÷ (14 × run length); all 14 slots are held whenever ≥14 games are unfinished, which was always the case).
  - An independent check finds the same thing. `gaps.py` marks an inter-action gap as parked when it is more than 110 s longer than its tokens explain at 70 tok/s. The remaining active time sums to 1,651, 1,672 and 1,685 of the 1,694 available slot-minutes.
  - Generated tokens are therefore slot-time. **Active minutes** in this note means tokens ÷ (the run's per-slot rate × 60). Wall minutes include time parked at the gate.
- **Level records.**
  - A level's tokens are the `generated_tokens` of the actions in its index range, taken from cumulative `actions_per_level`.
  - The final unsolved level also gets the tokens generated after the last action: the `tokens=` total in `solver_note` minus the history sum.
  - Censoring happens at the run end.

## 2. How D′ allocates time and capacity in the rerun (Q1)

### 2.1 Mechanics (code)

| Question | Answer | Where |
|---|---|---|
| Per-game time cap in the rerun | **532 min for every game, fixed.** The rerun branch sets `bm.solver.concurrency = 120` and `max_runtime_s_per_game = 532*60`. The task brief's `532*60*concurrency//110` is not the rerun setting: it is a comment on Franzen's non-submission branch, and it is the formula behind our `--full25 121` emulation (25 games → 121 min). | D′ cell 17 lines 12-17 (his cell 16); his cell 16 line 21 |
| Do all games start their clock together? | Yes. All 110 tasks run at once (semaphore and thread pool sized 120), and each session's clock starts at construction. So every game's deadline is ≈ start + 532 min, and a game waiting at the gate ages. | `solver.py:1524-1527`, `solver.py:1512`, `solver.py:480`, `solver.py:521-526` |
| Is the cap recomputed from the remaining wall clock? | No. The soft deadline is `None` in a rerun. The per-request timeout is min(900 s, the game's remaining time). Nothing divides the time left among the games left. | D′ cell 22 lines 76-81; `solver.py:536-553` |
| Admission gate | `ARC3_MAX_ACTIVE_STREAMS` slots: 10 in D′ and Franzen, 14 in ours (exp-074t log: "priority gate active: 14 concurrent streams"). `play()` takes a slot before the first turn and releases it in `finally`. The first 14 (or 10) games to arrive are admitted at once, whatever their priority. | D′ cell 5 line 128; `solver.py:603-634` (acquire 625, release 633-634); `tool_agent.py:2080-2117` |
| When slots change hands | **Only at a context trim, or when a game ends.** `_maybe_handover` runs before every request but acts only when the trimmer has evicted history. It re-prices the game and calls `handover`, which frees the slot and re-enqueues the game in one critical section; the best waiter takes the slot, and that may be the same game. | `tool_agent.py:6372-6416`, called at `6744` after the trim at `6743`; flag set at `6307-6311`; `tool_agent.py:2068-2078` |
| Quantum length | The context budget is 128 Ki − 12 Ki output reserve − 512 ≈ 118k tokens, drained to 58 Ki at a trim. Measured from the runs' parked gaps: first tenures that ended at the first trim cluster at 50-80k generated tokens (48% of first tenures; median 62k), and single later quanta are 37k (mean 36.7k, SD 7.5k, n = 32). That is ≈18 and ≈11 slot-minutes at 56 tok/s. | `tool_agent.py:3506-3509`; D′ cell 5 lines 83-85; `quanta.py` |
| Waiting queue | Waiters keep the snapshot taken when they queued: level ℓ, actions a and tokens t on the level, pace, N. Whenever a slot frees, every waiter is re-scored, but only φ changes. The highest integer priority wins, and ties go to the earliest arrival. | `tool_agent.py:2099-2117` |
| Never-started games | Franzen queues them in a top band, 2,000,000 − dispatch index (fresh first, in dispatch order). D′ replaces `acquire` so that they are priced by the formula with ℓ = 1, a = t = 0 and no pace: `#OURS_FRESH replaced=25` in all three runs. | `tool_agent.py:2149-2150`, `solver.py:598-601`; embedded:217-233 |
| Freed slot-time | A game that wins all levels (or crashes) calls `release()`. The best waiter is admitted at once, so the time is not lost. In the rerun there are waiters until the last hour. | `tool_agent.py:2063-2066`; `solver.py:633-634` |
| Can the harness give up on a game? | **No, not for lack of progress.** `should_stop` ends a game on WIN, at the runtime limit, at `max_actions_per_game` (None), at `max_generated_tokens_per_game` (None: a per-game token cap exists but is unset), or on the stop event. GAME_OVER triggers an automatic RESET. The only other exit is 10 consecutive failed analyzer calls, a guard against the endpoint dying. | `solver.py:579-596`, `solver.py:555-577`, `solver.py:655-661`, `solver.py:105`, `solver.py:690-710`; D′ cell 17 line 5 |

### 2.2 The D′ priority

priority = max(1, int(1000 · (A·M·C + B·φ))) (embedded:130-169, installed at embedded:196-201). With ℓ the current level (from 1), N the level count (clamped to 6..10 inside norm), and a, t the actions and generated tokens spent on the current level:

- **A** (depth × efficiency) = (1 + 0.5(ℓ−1)) · norm(N) · (300/(300+a))^2.5, where norm(N) = 55/(N(N+1)/2).
- **M** (pace) = clip((30,000/p)^0.4, 0.25, 4), where p is the mean generated tokens per level cleared so far (D′'s `FormPace`, embedded:45-63). M = 1 before the first clear. The text adds M as a separate factor, but in the code it is folded into A.
- **C** (patience) = 0.1·max(0.1, 1 − a/115) + 0.9·max(0.1, 1 − t/T), where T = 225,000 · clip(p/30,000, 0.5, 2)^0.5 (T = 225k before the first clear). C falls linearly with tokens on the level and reaches its floor of 0.1 at 0.9 T, i.e. ≈200k tokens ≈ 60 active minutes at average pace.
- **B** (room to grow) = 16, 14, 10 or 0 when ≥3, 2, 1 or 0 levels remain after the current one (embedded:150).
- **φ** (tail fade) = clip(remaining / window, 0, 1), with window = 0.4 × run length. D′ sets the fraction to 0.4, overriding the 0.2 in cell 5 (embedded:201; `tool_agent.py:2018-2057`). In the rerun the fade starts at minute 319.2 and B·φ reaches 0 at minute 532. The three test runs logged "tail fade phase" at 73.4, 74.7 and 72.8 minutes: the first re-pricing after 0.6 × 121 = 72.6 minutes.

What follows from the formula:

- **B dominates until the fade.**
  - A·M·C is 1.53 for a fresh 8-level game and at most ≈27 (level 8 of 8 with M = 4).
  - Never-started games queue at 16 + norm(N), i.e. 18.6 for N = 6 down to 17.0 for N = 10. Games with fewer levels start first; in the runs, the 6-level sp80, cd82, ft09 and tr87 were the first to start after the initial 14.
  - A running game with ≥3 levels left keeps its slot at a trim while its A·M·C exceeds norm(N) of the best fresh waiter. A stalled level-1 game yields at its first trim, while a progressing deep game does not.
  - A game on its last level (B = 0) is outranked by every game with a level left until φ is small. exp-073b measured this as last levels parked 70% of their game time (exp073b-failure-analysis.md).

### 2.3 What this does in the rerun (modelled; `rerun.py`, `rerun_diag.py`)

| | 14 slots | 10 slots |
|---|---|---|
| Games whose first slot comes after 120 / 240 min (of 110) | 56.6 / 23.2 | 59.4 / 29.1 |
| Median first-slot time over all games; latest (mean over draws) | 126 min; 343 min | 143 min; 398 min |
| Games that never get a slot (mean per rerun) | 0.1 | 2.3 |
| Share of slot-time on the last level / 1 left / 2 left / ≥3 left | 10% / 21% / 12% / 57% | 11% / 20% / 12% / 57% |
| Slot-time on levels still unsolved at the deadline | 33.3% | 32.5% |

These rows come from 200 × 4 draws in `rerun.py`, except the share rows, which come from 30 draws in `rerun_diag.py`.

In the public-25 replay, the same code's parked share by levels left (≥3 / 2 / 1 / 0) is 25% / 8% / 44% / 63%; exp-073b measured 21% / 0% / 41% / 70%. The first pass through 110 games takes most of the run: about a fifth of the games (23 of 110) get their first slot only after minute 240, in the last 4.9 hours.

## 3. Level timelines in the three runs (Q2, measured)

### 3.1 Time to solve

The data: 391 level attempts, of which 336 were solved and 55 were final levels censored at the run end. Per run: exp-073 108 of 126, exp-073b 124 of 141, exp-075 104 of 124.

| Solved levels | p10 | p25 | p50 | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
| Active minutes | 1.5 | 3.0 | 6.5 | 13.9 | 21.7 | 47.7 |
| Generated tokens (k) | 5.2 | 10.1 | 21.8 | 46.7 | 71.1 | 161.5 |
| Wall minutes (parked time included) | 1.9 | 3.6 | 9.0 | 21.1 | 41.5 | 98.0 |

The censored levels had run for a median 34.6 active minutes (p10 12.5, p90 56.0), or 48.4 wall minutes, when the runs ended.

Kaplan-Meier, P(solved within x active minutes):

| | 2.5 | 5 | 10 | 15 | 20 | 30 | 40 | 60 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All levels (n = 391) | 0.16 | 0.34 | 0.56 | 0.67 | 0.77 | 0.85 | 0.88 | 0.91 |
| Level 1 (n = 75) | 0.23 | 0.53 | 0.76 | 0.85 | 0.93 | 0.95 | 0.96 | 0.97 |
| Levels 2+ (n = 316) | 0.15 | 0.29 | 0.51 | 0.63 | 0.73 | 0.83 | 0.87 | 0.89 |

The last solve came at 47.7 active minutes, where S = 0.089. Attempts still at risk at 30 / 45 / 60 minutes: 45 / 15 / 5.

### 3.2 P(solve within t more active minutes | T spent without solving)

All levels; 90% cluster bootstrap over the 25 games, with all runs of a game resampled together.

| T \ t | 5 | 10 | 20 | 30 | At risk |
|---|---|---|---|---|---:|
| 0 | 0.34 [0.25, 0.42] | 0.56 [0.47, 0.64] | 0.77 [0.71, 0.82] | 0.85 [0.81, 0.89] | 391 |
| 5 | 0.34 [0.27, 0.40] | 0.51 [0.44, 0.57] | 0.73 [0.65, 0.79] | 0.81 [0.76, 0.86] | 257 |
| 10 | 0.26 [0.18, 0.34] | 0.48 [0.39, 0.56] | 0.67 [0.59, 0.74] | 0.74 [0.67, 0.80] | 170 |
| 15 | 0.29 [0.21, 0.36] | 0.44 [0.35, 0.53] | 0.62 [0.54, 0.70] | 0.68 [0.58, 0.77] | 121 |
| 20 | 0.21 [0.15, 0.28] | 0.36 [0.28, 0.45] | 0.50 [0.40, 0.60] | 0.62 [0.50, 0.73] | 81 |
| 30 | 0.16 [0.06, 0.27] | 0.21 [0.10, 0.33] | 0.39 [0.25, 0.55] | 0.39 [0.25, 0.55] | 45 |
| 40 | 0.10 [0.00, 0.23] | 0.23 [0.07, 0.40] | 0.23 [0.07, 0.40] | 0.23 [0.07, 0.40] | 22 |
| 50 | 0 of 10 at risk solved | | | | 10 |

- **By level.**
  - Level 1 alone: 0.95 within 30 minutes from T = 0, 0.82 [0.60, 1.00] from T = 15 (11 at risk).
  - Levels 2+: 0.83 from T = 0, 0.61 from T = 20, 0.38 [0.23, 0.55] from T = 30 (`out/km.txt`).
- **On the wall clock** (parked time counts): P(solve within 30 more) = 0.74 at T = 0, 0.51 at T = 30, 0.50 at T = 45 and 0.38 at T = 60 (`out/weibull.txt`). The wall clock understates the decline, because parked minutes are not attempts.

### 3.3 Is the hazard falling with time on the level?

Yes, but only after about 20 minutes. Solves per active minute at risk, with a 90% cluster bootstrap:

| (min] | 0-5 | 5-10 | 10-15 | 15-20 | 20-30 | 30-45 | 45-60 | 60-120 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hazard /min | 0.080 | 0.083 | 0.061 | 0.067 | 0.046 | 0.027 | 0.016 | 0 |
| 90% CI | 0.058-0.105 | 0.065-0.106 | 0.041-0.084 | 0.048-0.089 | 0.034-0.060 | 0.013-0.046 | 0-0.035 | (60 min at risk) |
| Solves | 131 | 86 | 44 | 34 | 28 | 11 | 2 | 0 |

- A censored Weibull fit gives shape 0.90 [0.81, 1.01] for all levels, 0.95 [0.84, 1.24] for level 1 and 0.92 [0.81, 1.05] for levels 2+. That is mildly decreasing overall, and the decline is concentrated after 20 minutes.
- So a level not solved in 30 active minutes is about half as likely per minute to be solved as a fresh one. It is not hopeless: 39% are solved in the next 30 minutes. After 45-50 minutes the evidence runs out: 2 solves in 129 minutes at risk at 45-60, then none.
- **Hard level or unlucky run?** Both.
  - The solve times of the same level in two runs correlate at r = 0.68 on log active minutes (291 pairs, `levelprop.py`).
  - When a run spent 20 or more minutes on a level, the same level in the other runs was solved in 90 of 139 attempts, 28 of them in under 10 minutes.
  - Of the 55 final unsolved levels, 44 were reached in another run, and 35 of those were solved there.
  - 15 (game, level) pairs were reached but never solved in any run (`out/levelprop.txt`).

### 3.4 Slot-time on levels that were never solved

- **Share.** Final unsolved levels took **36.4%** of all generated tokens: exp-073 40.2% (2.16M of 5.36M), exp-073b 25.8% (1.49M of 5.75M), exp-075 43.4% (2.50M of 5.76M). Since all 14 slots were held the whole time, this is also the share of wall time × concurrency.
- **The same runs with a cut.** Suppose each game had stopped once it had spent X active minutes on one level, with no reallocation (per run, `km.py`):

| X | Tokens after the cut | …of it on levels never solved | Games stopped | Levels lost | Score lost directly |
|---|---|---|---:|---:|---:|
| 20 min | 2.40M (42.7%) | 0.79M (14.0%) | 20.0 of 25 | 28.7 | 16.36 |
| 30 min | 0.99M (17.7%) | 0.54M (9.6%) | 14.0 | 9.0 | 5.69 |
| 40 min | 0.44M (7.8%) | 0.29M (5.1%) | 7.3 | 1.7 | 0.51 |
| 50 min | 0.14M (2.5%) | 0.14M (2.5%) | 3.3 | 0.0 | 0.00 |

Most of the 36% is spent before 30 minutes, where it cannot be told apart from the time solved levels need.

## 4. Reallocation policies in the modelled rerun (Q3)

### 4.1 The model (`sim.py`), and how well it reproduces the runs

**Scheduling** follows §2.1 exactly:

- 110 threads with a common 532-minute deadline;
- the first S games in a random dispatch order admitted on arrival;
- handover only at trims (first after N(62k, 6k) generated tokens, then N(37k, 7.5k) each);
- the D′ priority, ported and checked against the real code;
- φ re-evaluated at every pump, with ties going to the earliest arrival;
- `release` to the best waiter when a game wins.

**Throughput.** Each held slot generates 56.4 tok/s when 14 are held (exp-073b/075). With k active games the aggregate scales along the measured decode curve, the median SGLang decode tok/s by running requests at 0.88 running per held slot (`decode.py`). That gives 790 tok/s for 14 slots, 719 for 10, 539 for 7 and 150 for 1, so slots left idle by a give-up return part of their throughput to the games still running.

**Games.** Each of the 110 games is one of the 25 public games, drawn at random, with one of its three recorded runs:

- Level costs (tokens to solve) and per-action token profiles are taken as recorded.
- The level a run ended on gets a solve cost drawn from the pooled Kaplan-Meier (§3) conditional on exceeding its censoring point.
- Beyond the last observed solve (161k tokens) the hazard is held constant at the rate fitted on all exposure beyond 100k tokens (≈30 active minutes): 6.2 per M tokens = 0.021 per active minute.
- Levels its run never reached are taken from another run of the same game that did reach them. Where no run did, the pooled KM is used, with the score of a donor level.
- Actions on an imputed level continue at the level's own rate. Scoring is the toolkit's.

**The counterfactual, explicitly.**

- A slot freed by a policy goes to whichever waiter D′ ranks highest, exactly as the code does.
- That game converts the extra tokens into levels along its own recorded costs. Once it is past what its run recorded, it follows the same hazard curve as §3.
- So the value of extra time to the receiving games is not assumed. It comes out of the simulation: **+1.05 points per extra 1M tokens** at the margin (+1.32 points for +5% capacity, `policy_decomp.py`). A rerun has 25.2M tokens in all.
- Each policy and D′ are run on the same draws (common random numbers).

**Replay validation** (`validate.py`). Each run was replayed with its own trajectories and dispatch order: 25 games, 121 minutes, 14 slots, 40 draws.

| Run | Observed | Simulated [5-95%] | Per-game tokens, corr / mean abs diff | Per-game score MAE |
|---|---|---|---|---:|
| exp-073 | 49.45, 108 levels | 49.92 [48.89, 50.83], 109.1 | 0.99 / 8k of 214k | 0.8 |
| exp-073b | 56.00, 124 | 57.56 [56.00, 59.89], 125.7 | 0.97 / 11k of 230k | 1.7 |
| exp-075 | 42.89, 104 | 43.89 [42.23, 46.02], 105.2 | 0.95 / 12k of 230k | 1.3 |

The start times of the queued games match closely. For example, in exp-073: sp80 17.8 observed vs 17.2 simulated, cd82 18.2 vs 18.3, ft09 20.3 vs 19.1, g50t 32.0 vs 33.0, ar25 44.6 vs 43.7. The parked shares by levels left also match (§2.3). The simulator runs about 1 point high, because imputed levels are sometimes solved when a game gets slightly more time than it really had.

**The rerun level.**

- 110 games drawn from the full pool score 56.8 in the model (`bootcheck.py`, 10 × 2 draws).
- The cluster bootstrap gives 55.7, against 49.4 observed on the public-25 for the same games: +6.3.
- Of the rerun's score, 6.8 points come from imputed levels (beyond a run's recorded horizon), against 1.6 in the replay (`composition.py`). The longer horizon lets the gate concentrate time on games that go past where their 121-minute run stopped.
- That level is therefore extrapolation. **Only paired differences are used below.**

### 4.2 Results

Mean-score points on the public-25 scale. Each row is the mean of 4 draws per replicate, with the [5%, 95%] interval over 200 cluster-bootstrap replicates. "SD 1 run" is the standard deviation of the difference in a single rerun.

| Policy | 14 slots: Δ vs D′ | P(Δ>0) | SD 1 run | 10 slots: Δ vs D′ | P(Δ>0) |
|---|---|---:|---:|---|---:|
| D′ (score) | 56.53 [45.20, 68.52] | | | 53.54 [43.73, 63.93] | |
| Give up at 15 active min on one level | −29.01 [−37.30, −22.07] | 0.00 | 5.20 | −27.11 [−34.84, −20.91] | 0.00 |
| Give up at 20 | −20.02 [−26.58, −14.29] | 0.00 | 4.52 | −17.95 [−23.97, −13.27] | 0.00 |
| Give up at 30 | −6.91 [−13.50, −1.69] | 0.01 | 4.08 | −5.41 [−11.27, −0.61] | 0.01 |
| Give up at 40 | −1.74 [−3.96, −0.16] | 0.03 | 1.70 | −1.05 [−2.71, +0.37] | 0.13 |
| Give up at 50 | −0.19 [−0.75, +0.23] | 0.35 | 0.63 | −0.09 [−0.58, +0.23] | 0.43 |
| Give up at 60 | −0.03 [−0.25, +0.14] | 0.32 | 0.33 | −0.02 [−0.16, +0.08] | 0.27 |
| Park at 20 (re-queue at priority 1) | −2.67 [−4.75, −1.02] | 0.00 | 1.66 | −3.31 [−5.76, −1.37] | 0.01 |
| Park at 30 | −1.37 [−3.16, +0.28] | 0.07 | 1.47 | −1.98 [−4.39, +0.18] | 0.07 |
| Park at 40 | −1.32 [−3.00, −0.12] | 0.03 | 1.32 | −0.98 [−2.57, +0.37] | 0.14 |
| Give up at 20, only while a never-started game waits | −6.82 [−11.47, −3.22] | 0.00 | 3.53 | −7.15 [−12.03, −2.91] | 0.00 |
| Give up at 30, only while a never-started game waits | −1.61 [−4.98, +0.70] | 0.23 | 2.23 | −2.13 [−5.65, +0.58] | 0.14 |
| Give up at 30, levels 3+ only | −4.53 [−8.54, −1.36] | 0.01 | 2.71 | −4.07 [−7.91, −0.69] | 0.01 |
| *Reference: Franzen's gate (fresh-first, (A+B)·C, fade 0.2)* | −1.19 [−2.56, −0.08] | 0.04 | 1.24 | −2.16 [−4.24, −0.47] | 0.01 |
| *D′ with last-level B = 10 (instead of 0)* | +0.26 [−0.35, +1.09] | 0.69 | 0.70 | +0.23 [−0.26, +0.85] | 0.76 |
| *D′ with C floor 0 (instead of 0.1)* | −0.03 [−0.26, +0.19] | 0.43 | 0.31 | −0.03 [−0.22, +0.13] | 0.34 |

### 4.3 Why give-ups lose: where the freed time goes

Decomposition on common draws (`policy_decomp.py`: 60 reruns of 110 games from the full pool, 14 slots; mean-score points per rerun):

| Policy | Games ended | Their change | Tokens they gave up | Others' change | Tokens others gained | Net | Quitters' points per 1M freed tokens | Receivers' points per 1M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Give up at 30 | 75.2 | −7.59 | 3.39M | +0.59 | +0.18M | −7.00 | −2.24 | +3.20 |
| Give up at 40 | 58.6 | −2.77 | 1.06M | +1.00 | +0.74M | −1.76 | −2.62 | +1.35 |
| Give up at 50 | 14.9 | −0.54 | 0.28M | +0.28 | +0.28M | −0.26 | −1.92 | +0.99 |
| Give up at 60 | 3.1 | −0.06 | 0.05M | +0.06 | +0.05M | +0.01 | −1.23 | +1.35 |

Three things drive the result:

1. **D′ already starves stuck games.** A game that reaches 50 active minutes on a level would get only ≈19k more tokens (0.28M over 14.9 games). C has decayed, and fresh or deeper games outrank it. There is little time left to free: what levels unsolved at the deadline get beyond 40 minutes is 3.7% of rerun slot-time (`rerun_diag.py`).
2. **The time stuck games do get is productive.** Most of it goes to deep levels with large weights. At 30-45 minutes the hazard is still 0.027/min (§3.3), so it returns ≈1.9-2.6 points per 1M tokens, against ≈1.0-1.35 for the receivers, the marginal value of capacity.
3. **Short thresholds collapse demand.** Giving up at 30 minutes ends 75 of 110 games. The 3.39M tokens they release mostly go unused (+0.18M reaches others), because fewer games want slots than there are slots, and aggregate throughput falls with fewer streams.

### 4.4 Sensitivity

Δ vs D′ on the public-25 scale, 14 slots, [5%, 95%] over cluster-bootstrap replicates. Rows marked * use 60 × 3 draws; the others use 120 × 4.

| Variant | D′ | Give up at 40 | Give up at 50 | Give up at 60 | Park at 40 | Last-level B = 10 |
|---|---:|---|---|---|---|---|
| Main (fitted tail, KM for unreached levels; 200 × 4) | 56.53 | −1.74 [−3.96, −0.16] | −0.19 [−0.75, +0.23] | −0.03 [−0.25, +0.14] | −1.32 [−3.00, −0.12] | +0.26 [−0.35, +1.09] |
| No solves beyond the last observed one (tail hazard 0) | 55.40 | −0.91 [−2.63, +0.44] | +0.32 [+0.06, +0.69] | +0.06 [0.00, +0.20] | −0.62 [−2.06, +0.73] | +0.01 [−0.46, +0.65] |
| Tail hazard doubled | 56.70 | −2.16 [−4.38, −0.61] | −0.47 [−1.40, +0.08] | −0.06 [−0.31, +0.07] | −1.72 [−3.42, −0.42] | +0.44 [−0.28, +1.44] |
| Levels no run reached are never solved | 53.87 | −0.55 [−1.86, +0.84] | +0.27 [−0.22, +0.73] | +0.14 [−0.03, +0.50] | −0.28 [−1.47, +0.98] | +0.52 [−0.10, +1.47] |
| Later quanta 25k* | 55.34 | −1.70 [−3.84, −0.35] | −0.17 [−0.77, +0.25] | 0.00 [−0.30, +0.19] | −1.34 [−2.99, −0.29] | +0.40 [−0.19, +1.29] |
| Later quanta 50k* | 55.54 | −1.88 [−3.87, −0.08] | −0.17 [−0.79, +0.41] | −0.02 [−0.28, +0.18] | −1.52 [−3.01, −0.08] | +0.26 [−0.39, +1.04] |
| First quantum 45k* | 55.54 | −1.84 [−3.86, −0.21] | −0.16 [−0.61, +0.43] | −0.01 [−0.22, +0.14] | −1.46 [−2.88, −0.21] | +0.32 [−0.10, +1.06] |
| Harder games: every level ×2 the tokens* | 32.07 | −1.38 [−4.93, +0.61] | +0.19 [−0.12, +0.59] | −0.02 [−0.19, +0.12] | −1.36 [−4.93, +0.61] | −0.09 [−0.51, +0.19] |
| Harder games: ×2.5* | 27.29 | −1.52 [−4.04, +0.51] | −0.89 [−2.85, +0.27] | −0.01 [−0.27, +0.12] | −1.46 [−3.92, +0.51] | −0.12 [−0.48, +0.16] |
| Harder games: 12% of levels unsolvable* | 33.70 | −2.37 [−3.73, −1.18] | −0.40 [−1.27, +0.31] | −0.12 [−0.63, +0.21] | −1.67 [−2.73, −0.55] | +0.03 [−0.19, +0.37] |
| Harder games: 18% unsolvable* | 25.98 | −2.23 [−3.53, −1.11] | −0.57 [−1.58, +0.29] | −0.25 [−0.84, +0.19] | −1.44 [−2.51, −0.59] | +0.02 [−0.20, +0.21] |

The harder worlds put D′ at 26-34, about the LB's 0.55-0.62 of public-25 (`hard_world.py`). Across all eleven rows:

- Giving up at 40 is never positive.
- Giving up at 50 or 60 stays within ±0.9, and its sign flips with the tail assumption.
- The last-level B = 10 tweak is +0.0 to +0.5 and vanishes in the harder worlds.

### 4.5 A public-25 A/B would mislead

The same policies were replayed in our test setting (25 games × 121 min, all started at once, 14 slots; 3 runs × 20 draws; `sens_quanta.py`):

| Policy | Δ vs D′ | SD |
|---|---|---:|
| Give up at 40 | **+1.20** | 1.54 |
| Give up at 50 | +0.74 | 1.08 |
| Give up at 60 | +0.12 | 0.30 |
| Park at 40 | +0.84 | 1.30 |
| Last-level B = 10 | +0.35 | 1.03 |

The sign is the opposite of the rerun model for give-up at 40 and park at 40. The reason is where the freed time goes (`where_time_goes.py`, give-up at 40):

- **Public-25 replay.** Of the 0.35M tokens per run it frees, other games receive 0.16M, and 69% of that goes to games with 0 or 1 levels left, the ones the B cliff had parked.
- **Rerun model.** Of the 0.97M it frees, others receive 0.63M, and 84% goes to games with 3 or more levels left, the 96-game first pass.

Scheduling changes must therefore be judged in a rerun-shaped model (or on the LB), not on public-25 runs. One 25-game run's mean also has an SD of about 4.5 points.

### 4.6 The prize a better signal could win (oracle bound)

Suppose the policy knew, on entering a level, that the level would need more than Y active minutes, and ended the game there (`oracle.py`; 60 × 3 draws; Δ vs D′):

| Y | Δ vs D′ |
|---|---|
| 30 | −7.18 [−13.50, −0.55] |
| 45 | +2.94 [−1.38, +8.35] |
| 60 | **+8.07 [+3.69, +11.72]** |
| 90 | +5.13 [+1.62, +10.70] |

The value of reallocating away from hopeless levels is real. But it needs a signal at level entry or early on the level, and time spent on the level is not that signal: by the time it separates hopeless levels from slow ones (40-50 minutes), D′ has already stopped feeding them.

## 5. Recommendation (Q4)

**Do not implement a give-up or level-cap patch.**

- **Gain.** The best thresholds (50-60 active minutes on one level, ≈170-200k tokens) are worth between −0.89 and +0.32 public-25 points across all variants. The main estimate is −0.19 [−0.75, +0.23], with a single-rerun SD of 0.63. On the LB scale (× 0.55-0.62) that is about −0.5 to +0.2.
- **Risk.** Every threshold of 40 minutes or less loses 1.0-29 points with high confidence, and the soft cap ("park") loses 1-3.
- **Below the bar.** The expected gain is under 1 point by a wide margin.

If this question comes back, these are the places the code would change. They are recorded only so the same analysis is not repeated:

- **A give-up:** `_HarnessGameSession.should_stop` (`solver.py:579-596`) would return True when the analyzer's tokens on the current level exceed a threshold. The analyzer counts them as `_session_generated_tokens − _tokens_at_level_start` (`tool_agent.py:6390-6394`, reset at `3863-3864`). The slot is then released in `play()`'s `finally` (`solver.py:633-634`).
- **The B cliff:** the knob is the `0.0` in `B = (0.0, b1, b2, b3)` (D′ cell 21, embedded:150). Its estimate of +0.26 [−0.35, +1.09] public-25 (≈ +0.15 LB) disappears in the harder worlds. Not worth a submission slot.

What the numbers do support:

- Keep D′'s gate as it is. It beats Franzen's original gate by 1.2 [0.1, 2.6] points (14 slots) and by 2.2 [0.5, 4.2] points (10 slots) in the same model.
- Keep 14 streams rather than 10. In the model, D′ scores +2.46 [+1.33, +3.76] more with 14 (paired, `where_time_goes.py`), using the measured decode curve for the 10-slot throughput; with 10 streams, 2.3 games per rerun never get a slot.
- If time allocation is revisited, look for an early hopeless-level signal, not a timer (§4.6: up to +8 with perfect foresight). The cross-run evidence bears on the fresh-start patch (ours-07): of the 44 final unsolved levels that another run reached, 35 were solved there, so a stuck level is often a stuck *run*.

## 6. What this could not determine

- **No gate logs.** Active and parked time come from token rates. They are consistent in aggregate (active minutes sum to 97-99% of slot capacity) but are not measured per handover. The trim quanta are calibrated from parked gaps, and the results are insensitive to them (§4.4).
- **The hidden games are not the public 25.** The model draws hidden games from the public-25 trajectories. The harder-world rows (§4.4) are stress tests, not a model of the hidden set.
- **Level costs are assumed independent of scheduling.** The model assumes a level's token cost does not depend on when it is played, on parking, or on trims. A late start or many handovers could make levels costlier, which would favour neither policy in particular.
- **Extrapolation.** About 7 of the rerun model's ≈56 points come from levels beyond a run's recorded horizon. The tail assumption moves the give-up-at-50 sign (§4.4), which is why no give-up is recommended in either direction.
- **The 9-hour total.** This note did not check how 532 minutes of play plus about 9 minutes of notebook setup fits inside Kaggle's 9-hour limit. D′'s and our LB submissions completed, so it fits in practice.
- **Single runs are noisy.** Per-game SD between runs is ≈22 points (research log, exp-075). No run-level score difference in this note comes from a single run; the policy deltas are all model outputs.
