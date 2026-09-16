# Research log

One entry per change, newest at the bottom. Every number must point at a `runs/<name>`
directory and a commit. Format:

```
## <date> · exp-NNN · <one-line change> · KEPT|REVERTED|BASELINE
Why:      <problem observed>
Expected: <effect>
Measured: dev <a> → <b>, val <c> → <d> (run <runs/...>, commit <sha>); <s/action>; <tokens>
Notes:    <what did not move, links to post-mortems>
```

Fixed evaluation settings unless stated: seed 0, `ONLY_RESET_LEVELS=true`, per-game
wall-clock and action caps as listed, split as in `arc3/splits.py` (dev 19 / val 6).

## 2026-09-16 · exp-000 · uniform-random agent (Kaggle starter policy) · BASELINE
Why:      Calibrate what blind action-taking is worth under RHAE.
Settings: all 25 games, seed 0, 5000 actions/game, 120 s/game, run `runs/exp000-random-all-s0`, commit 0ae5623.
Measured: total 0.193 (dev 0.002, val 0.798); 8/183 levels; 0 games solved; 125,000 actions in 24 s wall.
          The whole score comes from sp80 level 1 (26 actions vs 39 baseline = 4.76 game pts); the other
          7 levels took 333–3434 actions and are worth ~0 after squaring.
Notes:    RHAE makes blind search worthless: a level solved at 10x the human count scores 1%.

## 2026-09-16 · exp-001 · no-LLM novelty explorer v1 (BFS over frame hashes, up to 48 click targets) · SUPERSEDED
Why:      Control arm: how far does blind state-graph search get, and what does it cost in actions?
Settings: all 25 games, seed 0, 3000 actions/game, 240 s/game, run `runs/exp001-explorer-all-s0`, commit 0ae5623.
Measured: total 0.025 (dev 0.002, val 0.097); 5/183 levels; below random (exp-000, which had 5000 actions).
Notes:    Branching factor (≈50 candidates per state) keeps the search shallow; keyboard games need depth.

## 2026-09-16 · exp-002 · explorer v2: probe keys before clicks, per-shape click priors, 16 click targets · KEPT (as fallback only)
Why:      Cut branching so the graph search goes deeper on keyboard games.
Expected: More level-1 completions on keyboard games.
Measured: total 0.057 (dev 0.044, val 0.097), run `runs/exp002-explorer-v2-all-s0`, commit 0ae5623; 5/183 levels;
          lp85 L1 in 31 actions (baseline 17, 0.84 pts), g50t L1 in 193 (0.58 pts); other completions at
          1370–2298 actions are worth ~0. ~10 ms/action on 4 CPUs.
Decision: Keep as the crash/idle fallback inside the REPL agent and on Kaggle; stop iterating on it.
          Blind search cannot reach RHAE-relevant action counts; the model-driven harness is the only path.

## 2026-09-16 · exp-003a · first real-model run: single Qwen3.8-27B-FP8 REPL agent, Kaggle RTX PRO 6000 smoke · BASELINE (plumbing)
Why:      Prove the serving stack and the agent loop end to end on the competition hardware before spending a dev run.
Settings: ls20 + vc33, 300 s/game, 150-action cap, reasoning_effort=low, max_output 3072, image scale 6 + ASCII board,
          vLLM 0.27.1 (Triton attention, MTP 2 tokens, FP8 KV), kernel `scottmahony/arc3-gpu-diag` v5, commit 82be16d,
          files `runs/kaggle-diag-v5-repl-smoke/`.
Measured: 0/14 levels. ls20: 20 model calls, 3 model actions, 146 fallback actions; vc33: 16 calls, 7 model actions,
          141 fallback. p50 11 s per call; 303k and 213k prompt tokens (~15k per call); 690 completion tokens per call.
          Server: 94 tok/s single, 308 tok/s at 8 concurrent, MTP acceptance ~2.5.
Diagnosis: (1) prompt bloat: ASCII board every turn plus the model printing ascii()/objects() into tool outputs;
          (2) the model inspects for up to 12 tool steps per turn and rarely calls act(); (3) when time runs short the
          explorer fallback fires and spends actions at 1 ms each, which is exactly what RHAE punishes.
Next:     exp-003b: observation diet (no ASCII when the image is attached, objects summary in the user text, 2k-char
          tool output cap), turn policy (act within 3 inspection steps, max 8 steps), fallback only on errors with a
          per-game cap; re-run the smoke, then dev.

## 2026-09-16 · exp-003b · observation diet + turn policy + capped fallback (smoke, Kaggle RTX) · KEPT
Why:      exp-003a showed 15k-token prompts, turns without actions, and a fallback that spent 146 actions in seconds.
Change:   no ASCII board when the image is attached; 16-object summary in the user text; 2.5k-char tool output cap;
          image scale 4; act-within-3-inspection-steps nudge; max 8 tool steps; fallback only after 3 idle turns, bursts
          of 2, capped at 30 per game; clean stop when <45 s remain; 60 s floor on the model call timeout.
Settings: ls20 + vc33, 600 s/game, 300-action cap, reasoning_effort=low; kernel `scottmahony/arc3-eval-smoke` v1,
          commit ff5a723; files `runs/kaggle-repl-smoke-003b/`.
Measured: **first model-driven level: ls20 level 1 in 19 actions (human 22) -> level score 115 (cap), game 3.57**;
          vc33 0 levels, 54 actions of which 16 RESETs. Total 1.79 over the two games. 0 fallback actions of 92,
          0 model errors, 0 idle turns. ls20: 34 calls / 37 actions, p50 8.8 s; vc33: 27 calls / 54 actions, p50 18.4 s.
          Prompt tokens still ~13k per call (accumulated history within the 32k budget; prefix-cached), ~930 completion
          tokens per call: decode dominates latency.
Notes:    Next: exp-003 on the 19 dev games (control arm), 1200 s/game, 8 concurrent. Then ablate thinking budget.

## 2026-09-16 · exp-003 · control arm: single Qwen3.8-27B-FP8 REPL agent on the 19 dev games (Kaggle RTX) · BASELINE
Settings: dev split, 1200 s/game, 8 games concurrent, 2000-action cap, reasoning_effort=low, max_output 3072,
          image scale 4, no ASCII; vLLM 0.27.1 Triton attention + MTP 2 + FP8 KV; kernel `scottmahony/arc3-eval-dev` v1,
          commit ff5a723 (+ world-model hooks, unused); files `runs/kaggle-repl-dev-003/`. 58 min RTX (441 s setup).
Measured: **dev 0.53**; 6/142 levels (all level 1): lp85 in 10 actions (human 17, cap), sb26 12 (18, cap), su15 17 (22,
          cap), tu93 23 (19), re86 50 (26), vc33 106 (7). 0 games solved; every game ended by the clean stop.
          Per game: 26-47 model calls, p50 10-55 s per call, median 1,435 completion tokens per call; aggregate server
          generation 347 tok/s median with 8 running requests. The model never used set_model/verify_model.
Failures: 40 of 48 model errors were HTTP 400 "maximum context length": the eviction estimate undercounts (images 576
          tokens, code tokenizes worse than chars/3). 5 consecutive errors flipped 4 games (sb26, vc33, cd82, wa30) into
          "server dead" mode and the fallback spent 400 actions each. 8 errors were read timeouts at 180 s under load.
Decision: bugs, not architecture: (1) calibrate the token estimate against the server's prompt_tokens, 2k margin,
          evict-and-retry on a context error; (2) timeouts widen the timeout instead of counting toward death, and
          death requires a failed liveness check; (3) dead-server fallback capped at 40 actions; (4) error-ended turns
          are not idle turns. Re-run as exp-003c before any architecture change. The time budget per game is the
          binding constraint: ~35 calls in 20 min at 8-way concurrency; thinking length is the next ablation (exp-004).

## 2026-09-16 · exp-003c · control arm re-run with the exp-003 bug fixes · KEPT (this is the control number)
Settings: identical to exp-003 (dev, 1200 s/game, 8 concurrent, reasoning_effort=low); kernel `scottmahony/arc3-eval-dev` v2,
          commit b0c0a4e; files `runs/kaggle-repl-dev-003c/`. 67 min RTX (440 s setup).
Measured: **dev 0.56**; 5/142 levels: ar25 L1 in 20 actions (human 32, cap), lp85 11 (17, cap), sb26 12 (18, cap),
          su15 31 (22 -> 50), vc33 12 (7 -> 34). 0 model errors, 0 context overflows, fallback 0-1 actions per game
          (the stop step). 28-71 model calls per game, p50 15-48 s, ~1,400 completion tokens per call.
Notes:    exp-003 solved tu93 and re86 but not ar25; exp-003c the reverse: single 20-min runs are noisy at the level of
          1-2 levels. Deltas smaller than ~0.3 on dev will not be trusted without a repeat. The model still never uses
          set_model/verify_model on its own; exp-005 tests the tracker + planner + prescribed procedure.

## 2026-09-16 · exp-006a · rule library + fitter over the symbolic transition log, code-only coverage probe · BASELINE (gate metric)
Why:      plan-100 component B. The model never wrote a verified world model on its own (exp-003/003c). Code can enumerate
          rule parameterisations and verify them against every transition; the model should only choose rule types.
What:     arc3/dsl.py (Move with walkable/blocking colours and a required colour, Push, Drift, Vanish, OnOverlap, Recolor,
          OnClick, Counter; fit/auto_rules/explain/simulate/predictor/plan/goal_predicates), tracker: occlusion filtering,
          static layer, compound sprites, HUD strips that shift along their axis; sandbox helpers auto_rules/plan_rules/
          rules_predictor/goal_candidates; prompt Method steps 3-5.
Measured: scripts/rule_coverage.py, fixed blind probe (60 actions on level 1: keys in pairs, ACT, clicks on entity
          centres), fraction of observed entity events explained by the fitted rule set (HUD excluded):
            first version           mean 0.103 over 25 (runs/rule-coverage/probe_all.log, commit 032cf88)
            + occlusion, cells, requires, compound sprites, onclick:
                                    mean 0.230 over 25; dev 0.279, val 0.074 (probe_all_v2.log, commit 38f0da8)
            + periodic counters, per-id move classes, HUD-masked prediction checks, split/merge artefacts:
                                    mean 0.283 over 25 (probe_all_v5.log, commit 4b28d5b)
            + canonical terrain under occlusion, click 'goto' effect, region-restricted classes, shape-only click targets:
                                    mean 0.325 over 25 (probe_all_v7.log, commit 0ada112)
            + HUD bars remembered after they run out:
                                    mean 0.375 over 25 (probe_all_v8.log, commit 570412f)
            + sliding moves, invisible walls from bumps, click-to-cell and swap effects:
                                    mean 0.417 over 25 (probe_all_v9.log, commit c8bb157); + region-restricted click targets, 12 click rules: mean 0.427 (runs/rule-coverage/summary.json, probe_all_v10.log)
            + Return rule (entities jump back to their start on ACT), move rules silent on ACT/clicks:
                                    mean 0.488 over 25 (probe_all_v11.log, commit 8abe231); + colour-aware matching and same-box
                                    re-identification of sprites whose mask changes while moving: mean 0.500 (probe_all_v12.log)
            per game >= 0.5: ar25 1.00, ft09 1.00, m0r0 1.00, sp80 1.00, tn36 1.00, ls20 0.98, re86 0.96, cn04 0.86,
                             g50t 0.72, dc22 0.70, ka59 0.63
            0.0: s5i5, vc33; below 0.1: lf52, r11l, su15, tu93 (tile-swap puzzles where identical tiles exchange places
                             and the tracker loses the avatar's id; blind clicks that do nothing)
          CPU only: play + fit under 4 s per game. No GPU run yet: exp-005 (tracker + planner) is still queued on Kaggle.
Notes:    this is a lower bound (blind probe, one level) and not a score. It is the exp-006 gate metric: the fraction of
          observed mechanics the library can express. Remaining unexplained kinds: 'resized' (bars/counters that change by
          varying amounts, growth bars), 'moved' on click games (a marker jumping to the clicked column: OnClick 'goto'),
          scrolling worlds (sk48: everything but the avatar moves; needs a class exclusion), and 'appeared' (spawns).
          Lessons: docs/lessons/0008-terrain-vs-sprites.md. Next: exp-005 result; then the same probe driven by the model
          (does auto_rules() + plan_rules() reduce actions per solved level on ls20/re86/ar25/dc22?).

## 2026-09-16 · exp-007 · tracker + planner + rule library (v3) + prescriptive Method, model-driven · KEPT (provisional)
Why:      the control model never built a verified world model on its own; exp-005/007 give it entity tracking, a
          navigation planner, the rule fitter (auto_rules/plan_rules/rules_predictor/goal_candidates) and a Method.
Expected: fewer probing actions per level, more levels via planned paths.
Measured: dev 0.56 -> 0.839 (runs/kaggle-repl-dev-007, Kaggle kernel arc3-eval-dev-b v1, harness commit 4b28d5b,
          same settings as exp-003c: Qwen3.8-27B-FP8, low reasoning, 32k context, 1200 s/game, 8 workers).
          Levels 5/142 (same count as the control: ar25 L1 19 actions, ls20 L1 32, sb26 L1 12, vc33 L1+L2 16+17);
          actions 755 (control 1415); prompt tokens 9.7M (10.7M); 0 model errors; wall 3587 s; server setup 430 s.
          Transcripts (scripts/transcript_report.py): auto_rules called in 17/19 games, a planner in 2, set_model in 2;
          reported coverage mostly 0.16-0.69, so the model followed Method step 4 (probe) instead of planning.
          Per-call p50 17-57 s; 5-10 turns per game; most calls are manual grid inspection (printing rows) that
          ents()/describe_events already answer.
Notes:    the gain is two levels (vc33 L2, ls20 L1) at the noise level of a single run; the halved action count is the
          more robust signal. exp-005 (no rules) is still queued on Kaggle after 2 h. Next: exp-008 = fitter v5 (coverage
          0.10 -> 0.43 code-only) + automatic per-turn rules summary + goal_hints + probe_suggestions, same settings.

## 2026-09-16 · exp-006b · code-only rules agent (probe -> auto_rules -> goal -> BFS plan -> verified execution) · BASELINE
Why:      an end-to-end test of components A-C on real games without a model, and a fallback when the server dies.
Measured: six games, 150 s each (runs/smoke-rules3, commit 8ce2e7e): m0r0 L1 solved by a plan in 43 actions (human 30),
          nothing else. All 25 games, 120 s each, 4 workers (runs/exp006b-rules-all-s0, commit b9f878d): score 0.199
          (dev 0.011, val 0.794), 3 levels of 183: r11l L1 in 14 actions (human 22, during the probe phase), m0r0 L1 by
          plan (215 actions, after 6 mismatched optimistic plans), tn36 L1 (180 actions). Explorer (exp-002) 0.06,
          random (exp-000) 0.19 on all 25. Fits take 0.1-3 s per call and were run after every action, so the
          120 s budget bought only 25-250 actions per game.
Notes:    goal inference without a model is the limit (unique-colour / avatar-sized / collectible heuristics); ls20 loops
          on an optimistic plan against an invisible wall (fixed: bumps stay in optimistic plans, failed goals are banned).

## 2026-09-16 · exp-005 · tracker + navigation planner + prescriptive Method (no rule library) · KEPT (best single run so far)
Why:      the arm between the control and exp-007: entity tracking, move_model/plan_to_entity and the Method procedure,
          without auto_rules/plan_rules in the prompt (harness commit f76f29c; queued 2.5 h on Kaggle, ran 07:28-08:35).
Measured: dev 0.56 -> 1.108 (runs/kaggle-repl-dev-005, kernel arc3-eval-dev v3, same settings as exp-003c/007).
          Levels 8/142: ar25 L1, lp85 L1, s5i5 L1, sb26 L1, su15 L1, tn36 L1, vc33 L1+L2 (11+20 actions); actions 924
          (control 1415, exp-007 755); 0 model errors; wall 3563 s; setup 460 s.
          Transcripts: a planner in 2 games, set_model in 3, auto_rules 0 (not available); 2.19 inspection-only calls
          per turn (exp-007 2.48), mean latency 33 s per call.
Notes:    exp-005 (no rules) > exp-007 (rules v3, 0.839, 5 levels) in single runs: the rules-v3 Method (auto_rules first,
          coverage mostly < 1, probe) may have displaced the simpler plan_to_entity procedure, or it is noise (the two
          runs share 3 of their solved levels: ar25 L1, sb26 L1, vc33 L1-L2). Both arms halve the control's actions.
          exp-008 (rules v5 + automatic per-turn summary, queued) and exp-009 (HEAD) decide; if the rules summary does
          not beat 1.1, the Method reverts to exp-005's wording with the rules as optional helpers.

## 2026-09-16 · exp-008 · fitter v5 + automatic per-turn rules summary + goal_hints + probe_suggestions · REVERTED as a Method change (helpers kept)
Why:      exp-007 showed the model calling auto_rules but rarely planning; the harness now fits rules every turn and
          shows coverage and unexplained items, plus structural goal hints and untested-action suggestions.
Measured: dev 0.734 (runs/kaggle-repl-dev-008, kernel arc3-eval-dev-c v1, harness 840bd4b, same settings; queued 2 h,
          ran 09:12-10:19). Levels 5/142: ar25 L1, lp85 L1, ls20 L1 (23 actions), sb26 L1, su15 L1; actions 816; 0 errors.
          Four arms now: control 0.56 (5 levels), exp-007 0.84 (5), exp-005 1.11 (8), exp-008 0.73 (5). All four solve
          ar25 L1, lp85 L1, sb26 L1; the rest is 1-3 levels of spread. Transcripts: auto_rules in 12 games, a planner in
          3, set_model in 4, goal_candidates called in 16 games although it is empty on level 1 (wasted calls); 2.45
          inspection-only calls per turn (unchanged), mean latency 34 s, 13.8k prompt tokens per call. vc33 took 98
          actions without a level (exp-005: 2 levels in 31), tn36 128 actions (exp-005: L1 in 17).
Notes:    the per-turn rules line did not change how the model plays; the rules-first Method wording (exp-007/008) has
          twice come out below the navigation-first wording (exp-005). exp-009 (running) uses the navigation-first
          Method with all helpers; its repeat (exp-009b) will measure run-to-run noise before any further keep/revert.
          Follow-up: make goal_candidates() say "no level completed yet" instead of [] and steer the model away from it
          on level 1 (done in the prompt for exp-011).

## 2026-09-16 · exp-009 · navigation-first Method + all helpers (rules v6, per-turn rules line, goal hints, probe suggestions, optimistic planners, tile map, first-step nudge, rules-agent fallback) · KEPT (pending the noise repeat)
Measured: dev 0.836 (runs/kaggle-repl-dev-009, kernel arc3-eval-dev-d v1, harness 9c22b80, same settings; ran 09:55-11:03).
          Levels 6/142: ar25 L1, lp85 L1, ls20 L1, sb26 L1, su15 L1, vc33 L1 (5 actions); actions 987; 0 errors.
          Transcripts: a planner in 5 games, set_model in 5 (dc22: 49/49 predictions correct, level still not solved),
          auto_rules in 7; 2.15 inspection-only calls per turn (exp-005 2.19), mean latency 33 s.
          Five arms: control 0.56 (5 levels) < exp-008 0.73 (5) < exp-007 0.84 (5) = exp-009 0.84 (6) < exp-005 1.11 (8).
Notes:    every harness arm beats the control; among them the order is inside single-run noise. exp-009b (the same
          notebook pushed again as kernel arc3-eval-dev-e, 11:06) measures that noise. The per-call cost has not moved
          (about 35 calls per game); the ascii() tile-map change and the stagnation/level notices came after this build.

## 2026-09-16 · exp-009b · exp-009 repeated unchanged (run-to-run noise) · MEASURED
Measured: dev 0.870 (runs/kaggle-repl-dev-009b, kernel arc3-eval-dev-e v2, same notebook and harness 9c22b80 as exp-009,
          same seed and settings; ran 11:06-12:14). Levels 6/142: ar25 L1, bp35 L1, m0r0 L1, sb26 L1, vc33 L2;
          actions 794 (exp-009: 987); model calls 636 (exp-009: 684); 0 errors.
          exp-009 was 0.836 with 6 levels. Levels solved in both runs: ar25, sb26, vc33; only in exp-009: lp85, ls20, su15;
          only in exp-009b: bp35, m0r0.
Notes:    the total score moved by 0.03 while the set of solved levels changed on 6 of 9 games: the score is stable
          to about +-0.05 but per-game outcomes churn. Keep/revert rule from here: a change is a gain only when it
          beats the best of the two repeats by more than 0.1 AND solves at least as many levels; anything smaller
          is repeated once before deciding. Per-game claims need the level to be solved in both runs.

## 2026-09-16 · exp-010 · council arm, six roles on the coordinator model (specialist server failed) · REVERTED as run; council decided by exp-010b
Why:      the user's six-specialist + coordinator design as an ablation arm; council v2 (event-driven rounds, rich
          observation, async injection) on Qwen3.8-27B-FP8 with Qwen3-VL-8B-NVFP4 as the specialists.
Measured: dev 0.278 (runs/kaggle-council-dev-010, kernel arc3-eval-dev-council v1, harness 9c22b80 + council v2, same
          settings as exp-009; ran 10:44-11:57). Levels 4/142: sb26 L1, su15 L1, tn36 L1, vc33 L1; actions 743; 0 model
          errors. **The specialist server never started**: vLLM chose FlashInferCutlassNvFp4LinearKernel for the NVFP4
          checkpoint and flashinfer's JIT raised "No supported CUDA architectures found for major versions [12]"
          (SM120) on both attempts (`runs/_kaggle_output/arc3-eval-dev-council/vllm-specialist.log`), so all six roles ran
          on the 27B coordinator: 2-7 rounds per game, 12-42 extra 27B calls per game on the same GPU. Coordinator
          latency p50 rose to 23-60 s (exp-009: 14-57 s), calls per game fell to 21-35, actions 987 -> 743.
          vs exp-009 0.836 (6 levels): the shared-model council costs more than it gives.
Notes:    the 14 'gave_up' failures are a label artefact (exp-009 had 16): the agent stops when less than one turn of
          time is left; eval.py now calls that 'timeout'. Fix for the specialist server in `arc3/serve.py` (attempt
          ladder: official FP8 Qwen3-VL-8B build first, NVFP4 with VLLM_NVFP4_GEMM_BACKEND=marlin last, gpu_mem from free
          memory, image probe). exp-010b (`arc3-eval-dev-council-b`, run kaggle-council-dev-010b, harness 33f758e) is the
          real council measurement; it also carries the exp-011 harness, so its control is exp-011, not exp-009.
          Council prompt fixes after reading the transcripts (commit d7916b7, not in exp-010b): a level's first round
          waits for the first action (the specialists had written 'no actions taken; no transitions to analyze' in
          every game) and each report is written out once, later turns only say it is unchanged (the full text was
          repeated on every turn). These go into exp-010c if exp-010b shows the specialists helping at all.
          Report usage: 70 rounds, ~400 chars per report; the coordinator's cells referred to a specialist or report
          in 6 of 19 games, once each. The reports restate the observation the coordinator already has (entities,
          Rules line, goal hints); their marginal information is low by construction. A specialist that adds value
          must see or do something the coordinator does not (the full transition log, a search, an executed probe).

## 2026-09-16 · exp-010b · council with the specialist start ladder (FP8 then NVFP4 Qwen3-VL-8B) on the exp-011 harness · specialists still absent; council REVERTED as an arm until a specialist server runs
Measured: dev 0.744 (runs/kaggle-council-dev-010b, kernel arc3-eval-dev-council-b v2, harness 33f758e; ran 12:49-14:05 on
          the RTX; v1 landed on a T4 and was wasted). Levels 5/142: ar25 L1, lp85 L1, sb26 L1, su15 L1, tn36 L1; actions
          732; 0 model errors. **All four specialist rungs failed in 240 s** (`vllm-specialist.log`): the official FP8
          build died at load in DeepGEMM ("Assertion error (deepgemm layout.hpp:60): Unknown SF transformation", tuned
          and conservative alike), the NVFP4 build in flashinfer's cutlass FP4 JIT ("No supported CUDA architectures
          found for major versions [12]", the run predates the Marlin env). GPU fraction free before the ladder: 0.39, so
          memory was not the problem. The six roles ran on the coordinator again.
          Same harness, same day: exp-011 (plain REPL) 1.229 with 9 levels. The shared-model council costs 0.5 of score
          for the extra 27B calls (coordinator p50 latency 15-69 s vs 12-51 s).
Notes:    fix for the FP8 rungs: VLLM_USE_DEEP_GEMM=0 (FP8 linear layers take the CUTLASS/Triton path; the 27B
          coordinator's FP8 checkpoint uses a different scale layout and loads fine). NVFP4 rungs carry
          VLLM_NVFP4_GEMM_BACKEND=marlin. Both unverified until exp-010c. exp-010c also carries the council prompt fixes
          (first round after the first action, reports written once). If the specialist server still does not start,
          the council arm is parked: two runs show it only costs when the roles share the coordinator.

## 2026-09-16 · exp-010c · council WITH its own specialist server (Qwen3-VL-8B-Instruct-FP8, DeepGEMM off) on the exp-011 harness · REVERTED as an arm (code kept)
Measured: dev 0.498 (runs/kaggle-council-dev-010c, kernel arc3-eval-dev-council-c v1, harness 7de7f95 + council fixes d7916b7;
          ran 14:09-15:23 on the RTX). Levels 5/142: ar25 L1, bp35 L1, sb26 L1, su15 L1, tu93 L1; actions 925; 0 model errors.
          **The specialist server started on the first rung** (FP8 tuned, VLLM_USE_DEEP_GEMM=0: ready in 120 s, image
          probe OK, gpu_mem 0.30 next to the 0.60 coordinator), so this is the first measurement of the user's design
          with a real second model: 67 specialist rounds, 402 specialist calls, 0 timeouts, 491 s of round time,
          67 late injections; the coordinator's own cells referred to a report 19 times in 8 games.
          Same harness, same day: exp-011 (plain REPL) 1.229 with 9 levels; exp-010b (roles on the coordinator) 0.744 with 5.
          Coordinator p50 latency 12.45-63.91 s per game (exp-011: 12-51 s): the 8B server shares the GPU's compute
          with the 27B, and the report block adds prompt tokens on every turn.
Notes:    three council runs, three losses (0.28 shared model on the exp-009 harness, 0.74 shared model and 0.50 with real
          specialists on the exp-011 harness) against the single-model REPL agent measured the same day. The design's
          premise, that six advisory passes over the same observation help the coordinator, does not hold with this
          model: the reports restate what the observation already says and their cost is paid in GPU time and prompt
          tokens. The code stays (arc3/agents/council.py, the specialist ladder in arc3/serve.py); the arm is parked.
          A specialist that could still pay: one that does something the coordinator cannot, e.g. a search over the
          fitted rules or an executed probe on a copy of the game, measured as a helper call, not a prose report.

## 2026-09-16 · exp-004 · thinking off (enable_thinking false, max_output 2048, fixed effort policy) on the exp-011 harness · REVERTED
Why:      the per-call cost is the binding constraint (lesson 0009); a call without thinking is 3-5x faster, so the
          question was whether more, cheaper calls beat fewer, considered ones.
Measured: dev 0.324 (runs/kaggle-repl-dev-004, kernel arc3-eval-dev-g v2, harness 756a87e = exp-011's; ran 14:36-15:44).
          Levels 6/142: ar25 L1, bp35 L1, lp85 L1, m0r0 L1, s5i5 L1, sb26 L1; actions 3837; model calls 2475 (exp-011: 1180 actions, 9 levels);
          p50 latency 3.6-16 s per call (exp-011: 12-51 s). The model acted far more and far worse: lp85 took 819 actions for
          its level (exp-011: 45), wa30 786 without one; the levels it did solve scored 0.04-1.66 each because RHAE is
          the square of the action ratio. Same harness: exp-011 1.229 with 9 levels.
Notes:    low-effort thinking stays; it is what keeps the model from spraying actions. The remaining knob worth a run is
          the other direction (medium effort on every call, fewer calls) once the noise repeat of exp-011 is in.

## 2026-09-16 · exp-013 · medium reasoning effort on every call (fixed policy, max_output 4096) on harness 699825b · NOT KEPT for the submission (more levels, lower RHAE)
Why:      exp-004 showed that no thinking makes the model spray actions; the other direction, more thinking per call and
          fewer calls, had not been measured.
Measured: dev 1.251 (runs/kaggle-repl-dev-013, kernel arc3-eval-dev-i v1, harness 699825b = the exp-012 code plus
          the eval label fix; ran 17:00-18:08). **Levels 10/142**: ar25 L1, m0r0 L1, s5i5 L1, sb26 L1, su15 L2, tn36 L1, tu93 L1, vc33 L2; actions 2463; model calls 866;
          0 model errors. Against the exp-011 pair (1.229 / 1.372, 9 levels): one more level (su15 L2 and vc33 L2 both
          solved, tu93 L1 again), score inside the pair's band; the extra actions went to games that scored zero anyway
          (bp35 328, cd82 397, tr87 348). Per-call p50 latency 11-32 s, not higher than exp-011's, and 25-66 calls per
          game: with this template "medium" does not lengthen the calls much but the plans it produces are longer batches.
          Also relevant to exp-012: this run carries the same events-line/iterable-result code and did not lose levels,
          which points at noise rather than harm for exp-012's 0.608.
Decision: repeat (exp-013b, kernel arc3-eval-dev-i v2) before keeping, per the noise rule; if it holds at 10 levels the
          submission config moves to fixed medium effort.
Repeat:   exp-013b dev 0.866, **11 levels** (runs/kaggle-repl-dev-013b, arc3-eval-dev-i v2, 18:19-19:19): ar25, cd82 (first
          solve, 55 actions), ka59 (first solve, 146 actions), lp85, s5i5, sb26, su15, tn36, tu93, vc33 L2; actions 2359.
          7 of the solved games shared with exp-013. Medium effort finds more levels but spends more actions on them
          (vc33 L2 in 340 actions scores 0.32; tn36 208; s5i5 128), so the pair scores 1.251 / 0.866 (mean 1.06, 10 / 11
          levels) against the low-effort adaptive pair 1.229 / 1.372 (mean 1.30, 9 / 9). RHAE is the metric: the
          submission keeps exp-011's adaptive policy (low, raised to medium when stagnant). Worth a later run: adaptive
          with the raise triggered earlier, to get medium effort's levels without its action cost. exp-014 (medium,
          events line off) is still running and only separates the events-line effect.

## 2026-09-16 · exp-014 · medium effort on every call with the events line off (harness 4647ba8) · CLOSES the effort ablation: adaptive low effort stays
Measured: dev 0.631 (runs/kaggle-repl-dev-014, kernel arc3-eval-dev-j v1; ran 18:15-19:23). Levels 7/142: ar25 L1, lp85 L1, s5i5 L1, sb26 L1, su15 L1, tu93 L1, vc33 L1; actions 2613.
          Medium effort, three runs: 1.251 (10 levels, line on), 0.866 (11, line on), 0.631 (7, line off): mean 0.92.
          Adaptive low effort (exp-011 config), two runs: 1.229 and 1.372 (9 and 9): mean 1.30.
          Medium effort spends 2.1-2.6k actions per run against 0.9-1.2k, and the extra actions land on levels it does
          solve (low per-level RHAE) as well as on games it does not. The events line shows no effect at medium effort
          (7 levels without it, 10 and 11 with it) and a negative one at low effort (6 and 6 against 9 and 9); the level
          count of one config varies by up to 4 between runs, so only the score means are trusted here.
Decision: submission config = exp-011's (low effort, raised to medium on stagnation), events line off. Next knob, if a
          slot is spare: raise to medium earlier (stagnation_actions 3 instead of 6) to buy medium's extra levels
          without paying its action cost everywhere (exp-015).

## 2026-09-16 · exp-011v · base config (adaptive low effort, events line off, harness 4647ba8) on the VALIDATION split · first held-out number
Measured: val 0.794 (runs/kaggle-repl-val-011, kernel arc3-eval-val-a v1; 6 games, 1200 s each, 6 workers; ran 19:21-19:49).
          Levels 1/41: sp80 L1 in 39 actions (at the cap); actions 643; 0 model errors. g50t: 30/30 world-model
          predictions correct and no level (the same failure as dc22 on dev: a correct model of the mechanics without the
          goal); cn04 8/10 correct; r11l 473 actions without a level; lf52 (10 levels) and sc25 nothing.
          For scale: the code-only rules agent scored the same 0.794 on val (exp-006b, r11l L1), and the dev pair of this
          config is 1.229 / 1.372. The dev number carries a day of tuning on dev transcripts; val does not.
Notes:    val transcripts are not used for harness changes (the split exists to stay untuned). The gap says the next
          gains must come from general mechanisms, above all goal inference when the world model is already right
          (dc22 on dev is the training case for it), not from more per-game fixes. exp-016 = the base config on all 25
          games in one run (dev third sample, val second sample) is next.

## 2026-09-16 · submission notebook check · private Save & Run All on the RTX PRO 6000 · PASSED
Measured: kernel `arc-prize-2026-arc-agi-3-arc3-agent` v2 (private), harness 699825b, REPL agent, datasets FP8 27B + wheelhouse.
          vLLM ready on the first attempt (MTP + FP8 KV); offline smoke on ls20 + vc33 at 300 s/game, workers 1: vc33 L1
          in 8 actions, ls20 0 levels in 6 actions (runs/kaggle-submission-check-v2, smoke score 1.786); submission.parquet
          written; about 17 min of RTX in total. The competition rerun branch (framework against the gateway) is
          exercised only by a real submission.
Notes:    Milestone 2 candidate is ready technically; license and publication are the owner's call (status open items).

## 2026-09-16 · exp-012 · events line in every tool output, iterable single results, ascii(region) degrades to tiles, role key always present, DeepGEMM off for FP8 specialists · events line REVERTED (off by default), error fixes KEPT
Why:      exp-011 transcripts: 77 inspection-only calls spent on describe_events() after an act, nine refused
          ascii(region) calls, KeyError 'role' nine times, `for r in act('ACT')` iterating dict keys.
Measured: dev 0.608 (runs/kaggle-repl-dev-012, kernel arc3-eval-dev-h v1, harness 7de7f95, same settings; ran 15:51-16:59).
          Levels 6/142: ar25 L1, bp35 L1, s5i5 L1, sb26 L1, su15 L1, vc33 L1; actions 1181; 0 model errors.
          Against the exp-011 pair (1.229 / 1.372, 9 levels each): lost lp85, m0r0, tn36 (all solved in both exp-011 runs),
          gained bp35. Transcripts: describe_events() calls 126 -> 40 (the events line did its job), inspection-only share
          0.40 (pair: 0.46 / 0.41), tool errors 38 (31 / 45), mean tool output 631 chars (456 / 474), code cells 512
          (587 / 586), turns 141 (152 / 157). No mechanism visible in the transcripts for a loss of three levels; the
          longer tool outputs cost about 12% of the calls.
Decision: not kept yet. The base for the submission stays the exp-011 harness (756a87e) until exp-012b, a repeat of this
          notebook, lands: two runs below the pair mean revert the events line (keep the three error fixes, which
          cannot lower a score); one run inside the pair's band means noise and the bundle is kept.
Repeat:   exp-012b dev 0.692, 6 levels (runs/kaggle-repl-dev-012b, arc3-eval-dev-h v2, 17:12-18:13; ar25, lp85, ls20, sb26,
          su15, vc33; 900 actions; p50 latency 16-67 s). Two runs at 6 levels against two at 9: the events line is now
          off by default (`tool_events_line` config, commit below); the error fixes stay. Confound: exp-013 (medium
          effort, same code with the line on) reached 10 levels, so exp-014 = medium effort with the line off separates
          the two; exp-013b (running) repeats medium effort with the line on.

## 2026-09-16 · exp-011b · exp-011 repeated unchanged (noise repeat) · CONFIRMS exp-011
Measured: dev 1.372 (runs/kaggle-repl-dev-011b, kernel arc3-eval-dev-f v3, same notebook and harness 756a87e as exp-011, same
          seed and settings; ran 15:37-16:45). Levels 9/142: ar25 L2 (8.33 on its two levels), lp85 L1, m0r0 L1, s5i5 L1,
          sb26 L1, su15 L1, tn36 L1, vc33 L1; actions 938 (exp-011: 1180); 0 errors.
          Agreement with exp-011: 7 of the 8 solved games in common (only tu93 in exp-011, only tn36 here); the exp-009
          pair had 3 of 9 in common. Score pair 1.229 / 1.372, both 9 levels.
Notes:    the exp-011 harness is confirmed as the base: mean of the pair 1.30 with 9 levels, against 0.85 with 6 for the
          exp-009 pair and 1.11 with 8 for exp-005 (single run). Per-game agreement doubled, which suggests the fixes
          removed real failure modes rather than luck. exp-012 (harness 7de7f95: events line, iterable results, region
          degrade, role key) is measured against this pair.

## 2026-09-16 · exp-011 · verified-navigation fixes from the exp-009 transcripts (walkable floor entities, sprite companions, avatar hand-over, live move model, model retirement, inline events, batched click probes) · KEPT
Why:      exp-009 transcripts (`scripts/transcript_report.py runs/kaggle-repl-dev-009`): ka59 registered
          set_model(move_model().predict) and then took one action per call for 14 consecutive prediction mismatches;
          su15 called describe_events 33 times (one click per call, 1.5 actions per call); every act() result was echoed
          twice. A local replay of ka59's first six actions (`scratchpad/replay_ka59.py`, environment_files) showed the
          cause: the strict move model marked the white floor entity (colour 1, which the avatar had already stood on)
          as an obstacle, so the strict plan was None, the optimistic plan was executed, and the strict predictor
          "mismatched" every correct step (19 wrong cells = the sprite drawn twice). When the avatar then merged into
          #7, avatar() kept returning the dead id.
Change:   planner: cells whose static-layer colour the avatar has stood on are walkable wherever they occur; parts that
          always move with the avatar (tracker groups: ka59's eye, ar25's pupils) move with it in predictions; the
          sprite is erased with the terrain colour, not the background. tracker: avatar() prefers an alive entity that
          moved on the most recent key presses (control hand-over); 1-cell-thick edge strips that keep resizing count as
          HUD (ka59's growing "used" half of the bottom bar). sandbox: set_model(move_model().predict) registers a live
          predictor that re-fits from the evidence before every prediction and follows PLAN['optimistic']; three
          consecutive mismatches retire the model with 'pred_retired' in the result; every act() result carries its
          entity 'events'. REPL: no second echo of a printed result. Prompt: click three or four entities per act() call.
          Second pass over wa30 and ft09: the tracker keeps an id across a sprite turn (same colour and cell count,
          transposed box; wa30's 3x4/4x3 body was reborn on every turn, so avatar() reported a dead id and the model
          spent 282 actions and several RESETs), avatar() breaks ties by size (the body, not the strip riding on it),
          downscale() returns the tile-sampled board when the engine has no pixel upscale (ft09: the model rebuilt an
          11x11 tile map by hand over four calls), goal_candidates() explains itself on level 1 instead of returning [].
          Code-only coverage after the bundle: mean 0.505 over 25 games (v13 log; 0.50 before); after the turn-tracking
          pass wa30 fell to 0.05 (its old 0.30 came from spurious vanish rules on the reborn sprite); merging markers
          that ride on a sprite's edge into its compound (tracker _attached_parts) gives wa30 an exact 4-px move rule
          and a touch-recolour rule: mean 0.547 over 25 games (v15 log; changed: bp35 0.191->0.217, cn04 0.872->0.897, dc22 0.7->0.718, r11l 0.062->0.054, re86 0.957->0.951, sc25 0.436->0.429, sk48 0.287->0.432, tu93 0.333->0.5, wa30 0.296->1.0).
          Replay after the fix: strict plan ['RIGHT','RIGHT'] exists, prediction exact except one HUD cell (masked),
          avatar() hands over to #7 after its first key move. Tests: tests/test_planner.py (FloorWorld, hand-over),
          tests/test_sandbox.py (events, live model, retirement), tests/test_repl_agent.py (echo).
          Third pass (dc22, tn36, tool errors): dc22 ran an optimistic plan_rules() plan under the strict rules predictor,
          so 48 blocked moves were 'predicted correctly' as standing still (49/49 correct, no level). Now an optimistic
          plan is verified against the optimistic rules (the first blocked step is a mismatch) and any batch stops after
          three actions that changed nothing. Sandbox: helpers the model rebinds (`for act in plan`, `rules = ...`, 8
          calls lost) are restored with a note; user functions that shadow state names (wa30 def state()) survive;
          plan_rules()/rules_predictor() fit rules lazily (7 calls lost to 'call auto_rules() first').
          Code-only rules agent on all 25 games after these changes (exp-006c, runs/exp006c-rules-all-s0, 120 s/game,
          8 workers): 0.224, 2 levels (m0r0 L1 in 149 actions, r11l L1 in 54) vs exp-006b 0.199, 3 levels (tn36 L1
          lost, m0r0 0.09 -> 0.85): no regression on the code-only path.
Expected: fewer calls per action on navigation games (ka59, ls20, re86, dc22 style) and on click games (su15, sb26);
          no change on games without an avatar. Measured against exp-009/exp-009b with the same settings.
Measured: **dev 1.229** (runs/kaggle-repl-dev-011, kernel arc3-eval-dev-f v2, harness 756a87e, same settings and seed as
          exp-009; ran 12:55-14:02 on the RTX; v1 landed on a T4 and was wasted). Levels 9/142: ar25 L1, lp85 L1, m0r0 L1,
          s5i5 L1 (first solve of s5i5), sb26 L2, su15 L1, tu93 L1 (first solve of tu93), vc33 L1; actions 1180; 0 model
          errors; 19 of 19 games used their full 1200 s. Against the exp-009 repeats (0.836 / 0.870, 6 levels each) this
          is +0.36 over the better one with 3 more levels, past the keep threshold set by exp-009b; it is also above
          exp-005 (1.108, 8 levels). Six arms: control 0.56 (5) < exp-008 0.73 (5) < exp-007 0.84 (5) = exp-009 0.84 (6)
          ~ exp-009b 0.87 (6) < exp-005 1.11 (8) < exp-011 1.23 (9). Median latency per call fell on most games (12-27 s
          against 18-63 s in exp-009b) and calls per game rose to 27-56.
Kept:     yes. Follow-up: a repeat (exp-011b) when a slot is free, per the noise rule.
