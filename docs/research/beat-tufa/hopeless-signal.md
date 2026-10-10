Summary: at level entry, the game's own history (mostly how long the previous level took) ranks levels by their chance of needing more than 60 active minutes with leave-one-game-out AUC 0.68-0.70 (0.59-0.62 on the three plain runs alone), and nothing the agent does in its first 5-15 minutes on the level adds to it. The flag rules a policy could use are right 8-27% of the time, against the ~61-66% a give-up needs. In the rerun model all 48 end/park policies built on the predictions lose to D′ (best −0.22 [−1.81, +1.13]), demotion loses too, and a detector would need about AUC 0.9 at ≤1% false positives to pay. No priority patch.

# Early signals of a hopeless level under D′: a weak one at entry, none on the level, and no policy that pays

Research note, 2026-10-10. CPU only: no GPU, no Kaggle, no repo code changed. It follows
`docs/research/beat-tufa/time-allocation.md` (the prior study). Its §4.6 put the prize for a perfect hopeless-level
detector at level entry at +8.07 [+3.69, +11.72] public-25 points.

- **Measured**: computed from eight full-length runs by the scripts in §7.
- **Modelled**: taken from the prior study's rerun simulator (110 games, 532 minutes, 14 slots). It is extended only
  to look up predictions, and it still reproduces the prior study's oracle table exactly (§3.1).
- **Terms.** "Hopeless" means the level needs more than 60 active minutes. "Active minutes" are the prior study's:
  generated tokens ÷ the run's tokens per held slot-second. Intervals are 90% (5th-95th percentile).

## 0. Bottom line

1. **Is there an early signal?** A weak one at level entry, and none from the first 5-15 minutes on the level.
   - **At entry**, the game's own history ranks levels with leave-one-game-out (LOGO) AUC 0.68 [0.60, 0.75]
     (logistic) and 0.70 [0.63, 0.76] (boosted trees). This is on the six runs without a mid-level reset. The same
     pipeline on permuted features gets 0.45 (90% range 0.38-0.51). On the three plain runs alone, the entry AUC is
     0.59-0.62.
   - **Most of it is the previous level's time** (levels 2+: AUC 0.65-0.69 on its own). After a previous level in the
     fastest third, 2-3% of levels are hopeless; after a slower one, 11-16%. The level index carries nothing (0.49),
     and level 1 is rarely hopeless (2.3%).
   - **On the level, nothing.** The features of what the agent does there (actions, rate, tokens per action, deaths,
     action types, clicks, repeats, stretches without an action) stay inside the null range on their own at 5, 10 and
     15 minutes, and they add nothing to the history. By 10-15 minutes no feature set is reliably above chance
     (0.48-0.58).
2. **The flags a policy could use are mostly wrong.** On the plain runs, the rules the simulation uses flag attempts
   of which 8-27% are hopeless. Half or more of the flagged attempts were solved, at a median of 5-13 active
   minutes into the level for the entry and 5-minute rules.
3. **Modelled: no policy built on the signal beats D′.** Main rerun model, 200 × 4 paired draws:
   - **End or park.** All 48 predictor policies have negative point estimates: they end or park the game, at entry or
     at 5, 10 or 15 minutes, for the top 5, 10 or 20% of attempts. The best is **−0.22 [−1.81, +1.13]**: boosted
     trees, top 5% at 10 minutes, ending about 11 games per rerun. Entry policies lose 4-21 points, and 5-minute
     policies 1.6-8.
   - **Demote.** Lowering a flagged level's priority instead of ending the game gives −0.35 to −7.7, no better than
     demoting at random.
   - **Other worlds.** There are eight sensitivity worlds: two tail assumptions, never-reached levels unsolvable, no
     solves beyond a run's censoring point, games 2-2.5× harder, and 12-18% of levels unsolvable. In each, the best
     policy lies between −0.63 and +0.63, and its interval never excludes zero.
   - **The signal is real, just far too weak.** The predictor beats random flags at the same rate by +1 to +9 points
     at entry and 5 minutes.
4. **The bar.**
   - **The arithmetic.** A right flag gains about +0.17 points, the freed time. A wrong flag costs about 0.27-0.33
     points, the level and the rest of the game. So a flag must be right about 61-66% of the time.
   - **Synthetic detectors.** At entry they pay only at about AUC 0.9 with ≤ 1% false positives
     (+1.18 [−0.40, +3.06]). At 10 minutes the bar is AUC ≈ 0.86-0.92. At AUC 0.76, which is above anything measured
     here, they lose or break even.
5. **Recommendation: no priority patch** (end, park or demote).
   - D′'s pace factor M already uses the same history, and in the model it is worth about nothing either way: off
     +0.19 [−0.45, +0.91], doubled −0.10 [−0.84, +0.65].
   - On the LB scale (× 0.55-0.62), the best policy is about −0.1.

## 1. Data and method

### 1.1 Runs and pools

All eight runs are D′ with 14 LLM slots playing the 25 public games at 121 minutes per game (each log: "priority gate
active: 14 concurrent streams", `max_runtime_s_per_game=7260.0`). Scores and counts are from `hcommon.py`.

| Run | Harness | Mean score | Level attempts (solved) | Tokens per held slot-second |
|---|---|---:|---:|---:|
| exp-073 | base configuration, lossless decoding | 49.45 | 126 (108) | 52.6 |
| exp-073b | base, relaxed draft acceptance | 56.00 | 141 (124) | 56.5 |
| exp-075 | exp-073b + sandbox fix | 42.89 | 124 (104) | 56.4 |
| exp-077 | + budget meter, search helper, win ledger, level memory; RESET exposed | 48.96 | 125 (107) | 57.6 |
| exp-080 | the same four helpers, RESET not exposed | 46.49 | 126 (107) | 58.9 |
| exp-081 | exp-080 + perception helpers | 50.00 | 131 (113) | 57.6 |
| exp-078 | seven patches incl. the fresh start at 20 min (ours-07), RESET exposed | 39.73 | 115 (95) | 59.9 |
| exp-079 | seven patches incl. the fresh start, RESET not exposed | 45.02 | 120 (102) | 60.8 |

Three pools:

- **plain** = exp-073, 073b, 075: the prior study's runs, unpatched harness. 391 attempts.
- **6-run** (main for the predictors) = plain + exp-077, 080, 081. These add helpers that did not move the score
  beyond noise, and none of them resets a live level. 773 attempts, 663 solved.
- **8-run** = all. exp-078/079 clear the conversation on a level after 20 minutes, which roughly halves the later
  solve rate of the levels it touches (lesson 0037), so their labels after 20 minutes are not D′'s. Sensitivity only.
  1,008 attempts, 860 solved, 148 censored.

Every result is also given on the plain runs alone.

### 1.2 Level attempts, the clock and the checkpoints

- One record per level attempt: every solved level, plus the level each game run ended on (censored at the run end).
- The clock is the prior study's. A level's tokens are the `generated_tokens` of the actions in its index range
  (cumulative `actions_per_level`); the final unsolved level also gets the tokens generated after the last action.
  **Active minutes** = tokens ÷ (the run's tokens per held slot-second × 60). The per-run rates for the three plain runs
  (52.6, 56.5, 56.4) are the prior study's.
- Checkpoints t = 0 (level entry), 5, 10 and 15 active minutes on the level. An attempt has a row at t if it is still
  unsolved and still observed at t.

### 1.3 Features (only what the harness has at that moment)

- **Game history** (known at entry): level index, level count, share of levels done, levels left, last-level and
  first-level flags; the previous level's active minutes, actions, deaths and tokens per action; the mean ("pace") and
  the maximum active minutes of earlier levels; level 1's active minutes; the game's active minutes so far; the
  previous level relative to the mean of the earlier ones.
- **On the level so far** (t > 0): actions; actions per minute; tokens per action; RESETs (deaths: in the runs without
  RESET exposure every RESET is the automatic one after GAME_OVER); actions per model turn; distinct action types;
  click share and distinct click targets; repeated actions; minutes since the last action; the longest stretch without
  an action; actions in the last 5 minutes; "no action yet".
- **Not used.**
  - Human baselines: the harness does not have them for hidden games. They were used for nothing here.
  - Wall-clock features (minutes since the game started, parked share): they encode the 121-minute run and its parking,
    which the 532-minute rerun does not share.
  - "Actions with no visible effect": `benchmark.json` holds no frames. Whole-game transcripts exist for three patched
    runs only (exp-077/078/081), and they show diff images, not a per-action no-op flag. Not attempted.

### 1.4 The label and censoring

- **Hopeless** = the level needs more than 60 active minutes (the oracle threshold that was worth +8.07).
- Known when the attempt was solved (by 60 → 0; after 60 → 1) or was censored at 60 minutes or later (→ 1).
- Most hopeless attempts were censored earlier, because a game gets about 68 active minutes in all in a 121-minute
  run (14 slots × 121 minutes ÷ 25 games). In the 6-run pool only 13 attempts were observed past 60 minutes, while the Kaplan-Meier (KM) puts the
  hopeless share at 8.6%.
- An attempt censored at c < 60 gets the soft label P(T > 60 | T > c). This comes from the pool's KM on active
  minutes, with a constant hazard beyond the last solve fitted on all exposure beyond 30 minutes (the analogue of the
  simulator's tail beyond 100k tokens: 0.021-0.023 per minute).
- **Fitting**: each such row is used twice, as y = 1 with weight w and as y = 0 with weight 1 − w.
- **Evaluation**: the soft AUC, which is the AUC expected under the soft labels (pairs weighted p_i (1 − p_j)). It
  equals the ordinary AUC when the labels are 0/1 (checked against sklearn). Harrell's C on (T, solved) is also
  reported.
- **Check without soft labels**: a discrete-time hazard model fitted on 5-minute person-periods, which uses the
  censored attempts directly (§2.5).

### 1.5 Models and validation

- One model per checkpoint:
  - L2 logistic regression, with C chosen by grouped 5-fold CV inside the training games;
  - a small boosted-tree model (depth 2, 150 rounds, learning rate 0.05, at least 25 rows per leaf).
- **Leave-one-game-out (LOGO).** All runs of the held-out game are held out together, so no prediction ever saw its
  own game. The hidden set has different games, so this is the honest split.
- **90% intervals**: a cluster bootstrap over games of the out-of-fold predictions (1,000 replicates).
- **Null reference.** LOGO is biased below 0.5 when there is no signal: holding out a game rich in hopeless levels
  lowers the training base rate for exactly that game. An intercept-only model scores 0.22-0.31. The reference used
  below is the same logistic pipeline on feature rows permuted across the pool (20 permutations, `null_logo.py`).

## 2. Is there an early signal? (measured)

### 2.1 Base rates

Soft P(the level needs more than 60 active minutes), among attempts still unsolved at t (number at risk in brackets):

| Pool | Entry | 5 min | 10 min | 15 min | Attempts seen past 60 min |
|---|---|---|---|---|---:|
| plain | 0.072 (391) | 0.109 (257) | 0.164 (170) | 0.223 (121) | 5 |
| 6-run | 0.086 (773) | 0.129 (508) | 0.202 (321) | 0.275 (230) | 13 |
| 8-run | 0.092 (1,008) | 0.138 (660) | 0.212 (424) | 0.302 (292) | 19 |

Level 1 is rarely hopeless: 2.3% in both the plain and the 6-run pool, against 8.4% and 10.1% for levels 2+.

### 2.2 Prediction quality by checkpoint

Soft AUC, leave-one-game-out, [90% game-cluster bootstrap]; logistic / boosted trees. 6-run pool:

| Checkpoint | Level index only | Game history | On-level only | History + on-level | Null (permuted features, 90% range) |
|---|---|---|---|---|---|
| Entry | 0.50 [0.43, 0.58] / 0.55 [0.48, 0.63] | **0.68 [0.60, 0.75] / 0.70 [0.63, 0.76]** | – | – | 0.45 (0.38-0.51) |
| 5 min | – | 0.63 [0.55, 0.71] / 0.64 [0.58, 0.71] | 0.48 [0.40, 0.57] / 0.55 [0.50, 0.61] | 0.63 [0.56, 0.70] / 0.67 [0.62, 0.73] | 0.47 (0.37-0.55) |
| 10 min | – | 0.58 [0.50, 0.65] / 0.57 [0.51, 0.64] | 0.51 [0.44, 0.57] / 0.50 [0.44, 0.55] | 0.56 [0.51, 0.62] / 0.51 [0.46, 0.57] | 0.48 (0.41-0.52) |
| 15 min | – | 0.50 [0.44, 0.57] / 0.51 [0.44, 0.58] | 0.51 [0.44, 0.59] / 0.49 [0.43, 0.56] | 0.54 [0.49, 0.60] / 0.48 [0.43, 0.53] | 0.52 (0.45-0.58) |

The same models in the other pools (best of logistic / trees for the history and history + on-level sets):

| Checkpoint | plain runs only (LOGO within the 3 runs) | 6-run models scored on the plain runs' attempts | 8-run pool |
|---|---|---|---|
| Entry | 0.59 [0.50, 0.68] / 0.62 [0.54, 0.69] | 0.62 [0.53, 0.72] / 0.67 [0.60, 0.75] | 0.66 [0.59, 0.74] / 0.65 [0.58, 0.72] |
| 5 min | 0.55 [0.46, 0.63] / 0.56 [0.49, 0.64] | 0.59 [0.50, 0.70] / 0.62 [0.54, 0.69] | 0.60 [0.54, 0.66] / 0.63 [0.58, 0.68] |
| 10 min | 0.42-0.51 | 0.48-0.53 | 0.53-0.58 |
| 15 min | 0.35-0.40 | 0.41-0.49 | 0.50-0.54 |

Reading:

- **Entry.** A real but weak signal: 0.68-0.70 in the 6-run pool against a null that tops out at 0.51, and 0.59-0.67
  on the plain runs. The models lean on the previous level's time and level 1's time; their largest standardized
  logistic coefficients are log previous-level minutes +0.40, log level-1 minutes +0.32 and level count +0.23.
- **5 minutes.** The history still carries 0.63-0.64. What happened on the level adds nothing: the on-level features
  alone are inside the null range (0.48-0.55), and history + on-level is 0.63-0.67, the same as history alone within
  the intervals.
- **10-15 minutes.** No feature set is reliably above the null. On the plain runs, several values fall below 0.5
  (0.35-0.44 at 15 minutes); that is the low end of the null range for a pipeline with no transferable signal, not
  a reversed signal.
- **Where the hopeless mass sits.** 22 of the 25 games have some, and 18 of them hold 95% of it
  (`games_breakdown.py`). The entry prediction ranks pairs within the same game (0.68) about as well as pairs across
  games (0.70), so it is not only a "this game is hard" flag.

### 2.3 Level entry: does the game's history predict a hopeless level? (the open question)

Each history feature is used as a score on its own, with the sign fixed in advance. There is no fitting, so no
cross-validation is needed. Levels 2+; soft AUC [90% game bootstrap], and the soft hopeless rate by tercile of the
feature (fastest / middle / slowest third):

| Feature | plain: AUC | plain: rate by tercile | 6-run: AUC | 6-run: rate by tercile |
|---|---|---|---|---|
| Previous level's active minutes | **0.65 [0.57, 0.72]** | 2.7% / 11.5% / 11.0% | **0.69 [0.62, 0.76]** | 2.1% / 12.3% / 15.9% |
| Level 1's active minutes | 0.63 [0.54, 0.72] | 4.7% / 8.1% / 12.7% | 0.64 [0.54, 0.73] | 4.8% / 11.0% / 14.7% |
| Mean active minutes per cleared level (the input to D′'s pace factor M) | 0.62 [0.52, 0.71] | 5.6% / 9.4% / 10.2% | 0.67 [0.58, 0.75] | 5.2% / 9.7% / 15.4% |
| Slowest earlier level | 0.58 [0.49, 0.67] | | 0.63 [0.56, 0.72] | |
| Tokens per action on the previous level | 0.58 [0.46, 0.67] | | 0.60 [0.50, 0.67] | |
| Game's level count | 0.54 [0.44, 0.64] | | 0.57 [0.47, 0.67] | |
| Level index | 0.49 [0.39, 0.62] | | 0.49 [0.41, 0.60] | |
| Share of levels already solved | 0.49 [0.37, 0.62] | | 0.48 [0.38, 0.59] | |
| Fewer deaths on the previous level | 0.49 [0.45, 0.53] | | 0.47 [0.44, 0.50] | |

So yes: a slow previous level predicts a hopeless level better than chance, on the plain runs alone too. The useful
part is the negative. After a fast previous level, only 2-3% of levels are hopeless. After a slow one, 11-16% are,
which is not enough to act on (§2.4, §3). The level index, and the level relative to the game's length, carry
nothing. D′'s pace factor M is built from the third row and already lowers a slow game's priority.

### 2.4 Calibration and precision at the thresholds a policy would use

- **Calibration.** 6-run pool, entry, boosted trees, mean prediction vs mean soft label by quintile: 0.01/0.01,
  0.02/0.06, 0.06/0.10, 0.11/0.09, 0.22/0.17. The ranking is right at the bottom and flat at the top. No attempt
  gets a prediction above 0.5 except two, and both were solved.
- **Precision.** The flag rules the rerun simulation uses: the 6-run models' out-of-fold predictions on the plain
  runs' 391 attempts, flagging the top q of attempts at risk at t (`auc_subsets.py`).

| Rule | Flagged | Share hopeless (base) | Solved later | Their median solve time (active min from entry) |
|---|---:|---|---:|---:|
| Entry, trees, top 5% | 20 | 0.16 (0.07) | 11 | 5.2 |
| Entry, trees, top 10% | 39 | 0.18 (0.07) | 20 | 6.6 |
| Entry, trees, top 20% | 78 | 0.14 (0.07) | 48 | 11.9 |
| Entry, logistic, top 10% | 39 | 0.08 (0.07) | 28 | 10.9 |
| 5 min, trees, top 5% | 13 | 0.27 (0.11) | 5 | 6.9 |
| 5 min, trees, top 20% | 52 | 0.19 (0.11) | 30 | 13.4 |
| 10 min, logistic, top 10% | 17 | 0.15 (0.16) | 10 | 21.9 |
| 15 min, logistic, top 10% | 12 | 0.15 (0.22) | 7 | 24.1 |

- **Across the 6-run pool.** At a threshold of 0.3 at 5 minutes the boosted model flags 35 attempts, 34% of them
  hopeless; 16 of them were solved later, at a median of 11.3 minutes.
- **The pattern.** Whatever the rule, half or more of the flagged attempts were solved, usually within a few more
  minutes.

### 2.5 Robustness of the measurement

- **Without soft labels.** A discrete-time hazard model (`hazard.py`) fits the censored attempts directly. Its
  soft AUC is 0.68 / 0.64 / 0.54 / 0.53 at entry / 5 / 10 / 15 minutes (6-run) and 0.64 / 0.57 / 0.45 / 0.42 (plain),
  the same picture as §2.2.
- **Other horizons** (6-run, boosted trees). For needs > 45 minutes: 0.70 at entry and 0.67 at 5 minutes. For > 90
  minutes: 0.68 and 0.64. Again the same.
- **Harrell's C** (ranking of time to solve, censoring-aware) is 0.60-0.63 at entry and 0.48-0.56 at 10-15 minutes.
- **Plain vs patched runs.** The 6-run models score 0.62-0.67 at entry on the plain runs and 0.72 on the patched
  runs (exp-077/080/081). The signal is weaker on the unpatched harness, which is the one the submission uses.

## 3. Policies in the rerun model (modelled)

### 3.1 Setup

- **Simulator.** The prior study's rerun model (`timealloc/sim.py`): 110 games, 532 minutes each, all started together,
  14 slots, D′'s gate and priority ported and checked against the code. Each simulated game replays one recorded run
  of a public game; a level its run never solved gets an effort drawn beyond its censoring point.
- **What changed (`psim.py`).** Two things only:
  - `make_instance2` makes the same random draws as the original and tags each simulated level with the recorded
    attempt it came from;
  - `simulate2` lets the policy act at a per-level token count instead of one global threshold.
- **Check.** `validate_psim.py` runs the prior study's oracle script with its seed. It reproduces its table exactly:
  D′ 55.96; oracle at 30 / 45 / 60 / 90 minutes −7.18 / +2.94 / +8.07 / +5.13.
- **Predictions.** Each simulated level gets the out-of-fold prediction of the attempt it came from, from the 6-run
  LOGO models, so a game's predictions never saw that game. The simulation pool is the prior study's (the three plain
  runs).
  - If the source attempt was censored before the checkpoint, there is no prediction and the policy does nothing
    there. The same holds for levels no run reached.
  - A checkpoint at t minutes acts at t × (the source run's tokens per slot-second) × 60 tokens on the level, in the
    same units as the features.
- **Policies.** At checkpoint t ∈ {entry, 5, 10, 15 minutes}, flag the attempt if its predicted P(hopeless) is in the
  top q (5, 10 or 20%) of the attempts at risk at t. Then either:
  - **end** the game (its slot goes to the best waiter), or
  - **park** it (re-queue at priority 1 for the rest of the level, as in the prior study), or
  - **demote** it (§3.4).

  Nothing touches a live level's context (lesson 0037).
- **Controls.**
  - Random flags at the same checkpoint and rate, drawn afresh per replicate.
  - The oracle (end the game on entering a level that needs more than 60 minutes).
  - Synthetic detectors of known quality (§3.5).
- **Statistics.** Paired cluster bootstrap as in the prior study: 200 replicates (the 25 games resampled with all their
  runs; KM and donors refit) × 4 rerun draws for the main world, with every policy on the same draws. The stress
  worlds use 60 × 3.

### 3.2 Main result: every predictor policy loses to D′

D′ scores 55.75 in this model (replicate 5-95%: 44.90-67.04). Δ vs D′ on the public-25 scale [5%, 95% over
replicates]. The action is "end"; park is within 0.5 of end at the top 5 and 10%, and at the top 20% it loses less
but still loses:

| Checkpoint | Model | Top 5% | Top 10% | Top 20% |
|---|---|---|---|---|
| Entry | logistic | −4.29 [−8.01, −1.30] | −9.16 [−15.12, −3.74] | −20.99 [−30.56, −12.03] |
| Entry | trees | −4.08 [−6.71, −1.76] | −7.94 [−11.34, −4.16] | −14.40 [−22.39, −7.30] |
| 5 min | logistic | −2.47 [−5.67, +0.03] | −3.12 [−6.67, −0.43] | −8.01 [−14.23, −3.28] |
| 5 min | trees | −1.56 [−3.93, +0.20] | −4.63 [−8.78, −1.42] | −7.25 [−14.28, −1.83] |
| 10 min | logistic | −0.38 [−1.57, +0.38] | −1.67 [−3.90, +0.35] | −2.72 [−6.59, +0.39] |
| 10 min | trees | **−0.22 [−1.81, +1.13]** | −2.18 [−4.87, −0.11] | −3.51 [−7.83, +0.19] |
| 15 min | logistic | −0.38 [−1.73, +0.20] | −0.36 [−1.63, +0.50] | −2.42 [−5.23, −0.06] |
| 15 min | trees | −1.48 [−4.02, +0.09] | −2.76 [−6.50, −0.11] | −3.83 [−7.48, −0.41] |
| *Random flags, same rate* | entry / 5 / 10 / 15 min | −6.82 / −3.85 / −1.99 / −1.07 | −11.49 / −6.54 / −3.94 / −2.52 | −23.44 / −12.76 / −7.00 / −3.95 |
| *Oracle (> 60 min, at entry)* | | +7.96 [+4.38, +11.09] | | |

- **The best policy** of the 48 (flag the top 5% at 10 minutes with the boosted model, end the game) is
  −0.22 [−1.81, +1.13], P(Δ > 0) = 0.43. That is the best of 48 tries, and it is still below zero.
- **How often they fire.** The policies end 19-27 games per rerun at the top 5% at entry, 47-51 at the top 10%, 4-11
  at the top 5% at 10 minutes, and 3-7 at 15 minutes. Of the levels they end, at most 22% would have needed more than
  60 minutes in the simulated world; the oracle's share is 100%.
- **Park vs end.** The rerun has more than 14 games waiting until late in the run, so a parked game rarely gets a slot
  back, and parking is almost the same as ending. At the top 20% at entry, parking loses less (−6.82 and −8.98 against
  −20.99 and −14.40) because the parked games do resume once the queue thins. It never gains.
- **10-15 minutes.** These policies lose little only because they act on few games: few levels last that long, and D′
  already gives those games little time.

### 3.3 The predictor does carry information, just not enough

Δ(predictor policy) − Δ(random flags at the same checkpoint, rate and action), paired per replicate:

- **Entry and 5 minutes**: +1.4 to +9.0, for example +2.74 [−2.92, +8.38] for trees at the top 5% at entry, and
  +5.50 [−2.79, +15.33] for trees at the top 20% at 5 minutes.
- **10-15 minutes**: −0.4 to +4.3.

The ranking is better than chance, which matches §2. But random flags lose 7-23 points at entry, and the predictor
recovers only part of that.

**Why.**

- **The value of a right flag.** Ending a game on a hopeless level gains about +0.17 points: the oracle's +7.96 over
  46.2 games per rerun.
- **The cost of a wrong flag.** Ending a game on a level it would have solved costs about 0.27-0.33 points: the level
  and the rest of the game. This is backed out of the random-flag policies, which are 7% right.
- **Break-even.** A flag must be right about 61-66% of the time to break even (`summarize.py`). The rules in §2.4 are
  right 8-27% of the time.

### 3.4 Softer uses of the same information

`pace_variants.py`, main world, 200 × 4 (D′ 55.75):

- **D′ already uses game history, and that use is worth nothing either way.** Its pace factor M (from the mean
  tokens per cleared level, AUC 0.62-0.67 for hopelessness in §2.3):
  - off: +0.19 [−0.45, +0.91];
  - half strength: +0.07 [−0.38, +0.50];
  - doubled: −0.10 [−0.84, +0.65].
- **Demotion.** A flagged level keeps its slot until its next trim, then re-queues with A·M·C × f for the rest of the
  level. Flags are the top 10 or 20% at entry or at 5 minutes:

| Demotion | Predictor flags (6 rules) | Random flags at entry, top 10% / top 20% |
|---|---|---|
| A·M·C × 0.5 (B kept) | −0.35 to −1.58 | −0.26 [−1.37, +0.60] / −0.55 [−1.88, +0.57] |
| A·M·C × 0 (B only) | −2.13 to −7.66 | −4.63 / −8.21 |
| A·M·C and B × 0.5 | −1.59 to −4.29 | −2.86 / −5.18 |

  - None is positive, and the mildest is no better than demoting at random.
  - Halving B as well puts a flagged game behind every fresh game for most of the run, and that acts like parking.

### 3.5 What detector quality would pay

`synth_grid.py`, main world, 200 × 4 (D′ 56.09 on these draws; oracle +7.69 [+3.01, +11.28]).

- **The detector.** A synthetic score d·[the level needs > 60 min] + N(0, 1); the game ends when the score exceeds k.
  It acts at entry, or at 10 minutes on levels still unsolved then.
- **Its quality.** AUC = Φ(d/√2) and false-positive rate (FPR) = 1 − Φ(k).

| AUC | Entry, FPR 5% | Entry, FPR 1% | Entry, FPR 0.1% | 10 min, FPR 5% | 10 min, FPR 1% | 10 min, FPR 0.1% |
|---|---|---|---|---|---|---|
| 0.76 | −5.02 [−8.30, −1.87] | −0.97 [−2.22, +0.31] | −0.04 [−0.41, +0.31] | −1.00 [−2.80, +0.93] | −0.08 [−0.96, +0.65] | +0.04 [−0.21, +0.30] |
| 0.86 | −3.42 [−7.02, −0.06] | −0.19 [−1.70, +1.18] | +0.18 [−0.26, +0.66] | +0.24 [−1.63, +2.61] | +0.53 [−0.48, +1.46] | +0.20 [−0.14, +0.53] |
| 0.92 | −1.92 [−6.31, +2.01] | +1.18 [−0.40, +3.06] | +0.71 [+0.08, +1.56] | +1.78 [−0.45, +4.09] | +1.53 [+0.32, +2.87] | +0.60 [+0.13, +1.25] |
| 0.96 | −2.12 [−7.92, +3.31] | +3.16 [+0.88, +5.52] | +1.74 [+0.75, +3.00] | +3.00 [+0.31, +5.55] | +2.93 [+1.14, +4.73] | +1.37 [+0.49, +2.44] |
| 0.98 | −3.18 [−9.64, +3.25] | +5.12 [+1.87, +7.83] | +3.36 [+1.76, +5.12] | +3.22 [+0.06, +6.22] | +4.49 [+2.00, +6.58] | +2.57 [+1.30, +4.05] |

- **The break-even.** A detector starts to pay once about 60-70% of the levels it ends are truly hopeless.
  - At entry, that takes AUC ≈ 0.9 at a false-positive rate of 1% or less.
  - At 10 minutes the bar is a little lower, AUC ≈ 0.86-0.92: fewer levels are at risk and more of them are hopeless.
- **The measured signal** is AUC 0.68-0.70 at entry and 0.51-0.58 at 10 minutes. It sits on the first row of this
  table or below it.

### 3.6 Sensitivity: tail assumptions and harder, hidden-like worlds

Δ vs D′; the main world uses 200 × 4 draws and the others 60 × 3 (`policy_rerun.py`, `summarize.py`).

- **tail 0**: no solves beyond the last observed one.
- **tail ×2**: the tail hazard doubled.
- **never**: levels no run reached are never solved.
- **no late solves**: an attempt its run never solved is never solved.
- **×2 and ×2.5**: every level needs that many times the tokens. The policies act at the same relative progress.
- **12% and 18% unsolvable**: that share of levels, at random, can never be solved. The predictor cannot see them.

The last four put D′ at 26-34, about the LB's 0.55-0.62 of public-25.

| World | D′ | Best of the 48 predictor policies | Entry, trees, top 5%, end | 5 min, trees, top 5%, end | 10 min, trees, top 5%, end | Oracle | Synthetic AUC 0.92, FPR 1% |
|---|---:|---|---|---|---|---|---|
| main | 55.75 | −0.22 [−1.81, +1.13] | −4.08 [−6.71, −1.76] | −1.56 [−3.93, +0.20] | −0.22 [−1.81, +1.13] | +7.96 [+4.38, +11.09] | +1.24 [−0.21, +2.70] |
| tail 0 | 53.70 | −0.58 [−1.88, +0.18] | −4.87 [−7.71, −2.49] | −1.85 [−4.34, +0.40] | −0.70 [−1.89, +0.40] | +6.34 [+1.93, +11.40] | +0.72 [−1.27, +2.35] |
| tail ×2 | 54.45 | +0.12 [−1.50, +1.91] | −3.13 [−5.89, −0.47] | −0.92 [−2.79, +1.20] | +0.12 [−1.50, +1.91] | +8.37 [+4.22, +12.25] | +1.25 [−0.15, +2.92] |
| never | 53.15 | −0.56 [−2.10, +0.14] | −5.18 [−8.36, −2.37] | −1.93 [−3.99, −0.07] | −0.65 [−1.99, +0.73] | +6.07 [+2.70, +9.55] | +0.22 [−1.62, +1.63] |
| no late solves | 48.34 | −0.63 [−1.53, +0.18] | −4.19 [−6.56, −2.03] | −1.73 [−3.39, −0.37] | −0.63 [−1.53, +0.18] | +0.89 [+0.07, +2.41] | −1.32 [−2.59, +0.18] |
| ×2 | 33.45 | +0.63 [−3.52, +4.01] | −0.78 [−2.37, +0.69] | +0.08 [−0.46, +0.68] | +0.08 [−0.63, +0.88] | +9.69 [+6.12, +12.84] | +2.45 [+0.52, +4.81] |
| ×2.5 | 26.89 | +0.04 [−0.02, +0.20] | −0.68 [−1.99, +0.44] | −0.02 [−0.29, +0.21] | +0.04 [−0.21, +0.34] | +8.81 [+6.24, +12.50] | +1.67 [+0.43, +2.92] |
| 12% unsolvable | 34.15 | −0.37 [−1.36, +0.14] | −2.55 [−4.22, −1.09] | −1.39 [−2.71, −0.26] | −0.75 [−1.95, +0.22] | +2.88 [+0.80, +6.02] | +1.95 [−0.05, +3.45] |
| 18% unsolvable | 26.53 | −0.36 [−0.99, +0.00] | −1.96 [−3.49, −0.56] | −1.17 [−2.41, −0.31] | −0.74 [−1.65, +0.06] | +1.15 [−0.32, +2.69] | +1.22 [−0.22, +2.78] |

- **No world gives a predictor policy a positive estimate with an interval that excludes zero.** The best per world
  ranges from −0.63 to +0.63. The three positive ones are +0.12 [−1.50, +1.91] (tail ×2), +0.63 [−3.52, +4.01] (×2)
  and +0.04 [−0.02, +0.20] (×2.5, a policy that acts on about one game per rerun).
- **"No late solves" is not the case most favourable to give-ups.** It raises the share of flagged levels that are
  hopeless (up to 55%, against at most 22% in the main world). But it also makes freed time nearly worthless,
  because the games that would receive it end on unsolvable levels too: even the oracle gains only +0.89. The 12% and
  18% unsolvable worlds show the same effect (oracle +2.88 and +1.15).
- **In the scaled worlds the entry policies lose less** (−0.7 to −1.3). They fire on 3-11 games instead of 19-27,
  because fewer flagged deep levels are ever reached in 532 minutes.

### 3.7 Other pools

Both variants use 200 × 4 draws.

- **Predictions from the plain runs only** (LOGO within the three runs; same simulation, D′ 55.75): all 48 policies
  are negative. The best is −0.34 [−1.36, +0.40] (logistic, top 5% at 15 minutes), and entry policies lose 1.8-19.4.
- **The six runs as the simulation pool**, each with its 6-run LOGO predictions (D′ 53.99): the best are
  +0.09 [−0.99, +0.99] (logistic, top 10% at 15 minutes, about 11 games per rerun) and +0.01 [−1.09, +1.30] (trees,
  top 5% at 10 minutes). The other 44 are negative, and entry policies lose 2.1-14.6.

## 4. Why an early signal does not pay here

1. **The signal is in the game's history, and it is weak where it matters.** A slow previous level raises the hopeless
   share from 2-3% to 11-16% (§2.3). A give-up needs about 60% (§3.3).
2. **In their first 15 minutes, hopeless levels look like slow solvable ones.** Nothing the agent does on the level
   separates them (§2.2). Half or more of the flagged attempts are solved a few minutes later (§2.4).
3. **D′ already uses the history, softly.** Its pace factor M is built from the mean tokens per cleared level, one of
   the three best single predictors here (AUC 0.62-0.67). M lowers a slow game's A, and the patience term C decays
   with time on the level. In the model that use is worth nothing in either direction (§3.4): switching M off
   gives +0.19 [−0.45, +0.91] and doubling it −0.10 [−0.84, +0.65]. Giving the history more weight is not where
   the points are.
4. **The rerun makes a wrong flag expensive and a right one cheap.**
   - An ended game loses its current level and every later level.
   - The slot-time freed by a right flag goes mostly to fresh games in the queue, at about 1 point per 1M tokens (prior
     study §4.3).
   - That is why the oracle needs perfect foresight to make +8, and a detector that is right half the time still loses.

## 5. Recommendation

**Do not build a priority patch (end, park or demote) on these signals.**

- **Gain.** No predictor policy has a positive point estimate in the main model. The best is
  −0.22 [−1.81, +1.13] public-25 points, about −0.1 on the LB at 0.55-0.62×. Softer demotion does not help (§3.4), and
  the harder worlds do not change the sign (§3.6).
- **The bar for a future signal.** At level entry, a detector has to be right about 60-75% of the time on the levels it
  flags (§3.3, §3.5); in synthetic terms, AUC ≈ 0.9 at a ≤1% false-positive rate. The measured 0.68-0.70 AUC and
  8-27% precision are far from that. Lesson 0036 still applies: such a detector would have to be judged in the rerun
  model, not on a public-25 run.
- **Where such a signal might come from, if anyone looks again.** Information the harness does not keep in
  `benchmark.json` today:
  - per-action effects (the no-op share);
  - whether the agent's own notes contain a testable hypothesis about the goal.

  The prior study's cross-run finding (35 of the 44 final unsolved levels another run reached were solved there)
  says many hopeless attempts are a bad draw of the run rather than a hard level. So the information is more likely in
  the run's state of understanding than in the level. A mid-level reset is not the remedy (lesson 0037).
- **Keep D′'s gate as it is.** The give-up conclusion of time-allocation.md stands with a better signal than time
  alone.

## 6. What this could not determine

- **The labels are mostly inferred.** In the 6-run pool, 13 attempts were observed past 60 minutes, against 66.4 soft
  positives. The rest come from attempts censored at the run end, weighted by the pool KM's P(T > 60 | T > c). The
  hazard model, which fits the censored attempts directly, gives the same AUCs (§2.5).
- **The simulator imputes the remaining effort of a censored attempt from the pooled KM, independent of its
  features.** If stuck attempts with a slow history are more hopeless than average, the model understates how often
  the flags are right. The "no late solves" world (§3.6) makes every such attempt hopeless. The policies still lose
  there, because freed time is then worth little too.
- **No frame-level features.** `benchmark.json` holds no frames. The whole-game transcripts exist for three patched
  runs and give diff images, not a no-op flag. Not attempted.
- **Selection.** 48 predictor policies were tried, plus demotions. The best is still negative, so selection can only
  have flattered it.
- **The predictor is not refit per bootstrap replicate.** Predictions are out-of-fold by game. The bootstrap resamples
  the games of the simulated world but reuses those predictions.
- **The hidden set is harder** (LB ≈ 0.55-0.62 × public-25). The stress worlds are stress tests, not a model of it.
  - In the scaled worlds the policies act at the same relative progress as on the public games, as if the predictor
    were recalibrated to the hidden set's pace. That favours them.
  - In the unsolvable worlds the extra hopeless levels are invisible to the features.
- **Run mix.** The predictions come from models trained on six runs, three of them with helper patches. On the plain
  runs alone the entry signal is weaker (0.59-0.62).
- **Wall-clock features were excluded** on purpose (§1.3). Wall time on a level mostly measures parking in a 121-minute
  run.

## 7. Scripts and outputs

All scripts are in `/tmp/claude-0/-home-user-Arc-Agi-3-Kaggle-comp/d342458e-03bd-545b-8a6d-06bca061963e/scratchpad/hopeless/`,
with outputs in `out/`. They run with `skvenv/bin/python -I`, a scratch venv with the repo's numpy plus scikit-learn
(the repo venv has no scikit-learn).

- `runall.sh` runs the fits, the measured analyses and every simulation in about 36 minutes on 4 CPUs (01:33-02:09
  UTC).
- `null_logo.py`, `check_metric.py` and `validate_psim.py` were run on their own.
- The prior study's `timealloc/sim.py` and `common.py` are imported unchanged.

| Script | What it does | Output |
|---|---|---|
| `hcommon.py` | The eight runs, per-run token rates (timealloc/rates.py's method), level records with their actions | (printed) |
| `dataset.py` | Checkpoint rows and features; the KM with a tail used for the soft labels | `rows.json` |
| `model.py POOL Y` | LOGO logistic and boosted-tree models per checkpoint; soft AUC, Harrell's C, calibration, precision | `model_<pool>_Y<Y>.txt`, `oof_<pool>_Y<Y>.json` |
| `null_logo.py` | Null references for the LOGO AUC (intercept-only; permuted features) | `null_logo.txt` |
| `check_metric.py` | The soft AUC against sklearn (0/1 labels) and a brute-force pair sum | `check_metric.txt` |
| `hazard.py POOL` | Discrete-time hazard model (no soft labels in the fit) | `hazard_<pool>.txt` |
| `entry_history.py` | Single history features at level entry, rates by tercile | `entry_history.txt` |
| `auc_subsets.py` | 6-run models on plain vs patched runs; precision of the simulated flag rules | `auc_subsets.txt` |
| `games_breakdown.py` | Hopeless mass by game; AUC within and across games | `games_breakdown.txt` |
| `psim.py` | timealloc/sim.py's instance builder and gate with per-level fire points, end/park/demote, synthetic detectors | |
| `validate_psim.py` | Reproduces the prior study's oracle table exactly | `validate_psim.txt` |
| `policy_rerun.py WORLD B R SIMPOOL MODELPOOL` | Predictor, random-flag, oracle and synthetic policies against D′ | `policy_<world>_<simpool>_<modelpool>.txt/.json` |
| `pace_variants.py` | D′'s pace factor M off/half/doubled; predictor-driven demotion | `pace_variants.txt` |
| `synth_grid.py` | Synthetic detectors at entry and at 10 minutes | `synth_grid.txt` |
| `summarize.py` | The cross-world table and the per-game value arithmetic | `summary.txt` |

Outputs from before a feature fix (`prev_resets` counted level 1's warm-up RESET as a death, which only 8 of 200 game
runs record) are kept in `out/v1_prevresets_bug/`. The fix moved no AUC by more than 0.06, and that largest move was
in the no-signal region at 15 minutes (plain runs, 0.44 → 0.39). Every number above comes from the corrected run.
