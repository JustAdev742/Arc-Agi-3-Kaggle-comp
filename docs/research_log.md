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

## 2026-09-16 · exp-011c · exp-011 notebook, third sample (harness 756a87e) · CLOSES the harness question: no bisect
Measured: dev 0.775, 6 levels (runs/kaggle-repl-dev-011c, arc3-eval-dev-f v4; ran 20:42-21:43): ar25, lp85, sb26, su15, tn36,
          vc33; actions 1012; 0 model errors.
          The exp-011 notebook now reads 1.229 / 1.372 / 0.775 (9 / 9 / 6 levels); the current harness at the same
          config reads 0.608 / 0.692 / 1.041 (6 / 6 / 6). Means 1.13 vs 0.78 with three samples each and a spread of
          0.6 within the first triple: the difference is inside the noise and the "9 levels twice" was the lucky pair.
          No bisect; the current harness (events line off, error fixes in) stays the base.
Notes:    the practical noise model for this stack at 1200 s/game: score +-0.3 and levels +-2 between identical runs.
          Single-run decisions are only safe for effects larger than that (exp-004 thinking off, the council, medium
          effort on score); everything smaller needs three runs a side, which the long-horizon runs (exp-017/018)
          will also be subject to.

## 2026-09-16 · exp-016 · base config (adaptive low effort, events line off, harness ec12191) on ALL 25 public games · dev third sample, val second sample
Measured: all-25 0.791; **dev 1.041, val 0.000** (runs/kaggle-repl-all-016, kernel arc3-eval-all-a v1; 8 workers, 1200 s each;
          ran 19:50-21:18). Levels 7/183: ar25 L1, ft09 L1, lp85 L1, sb26 L2, su15 L1, tn36 L1; actions 1277; 0 model errors. ft09 L1 solved for the
          first time (54 actions); sp80 L1 (solved in exp-011v) missed, so val is 0 of 41 this time.
          Dev samples on the post-exp-011 harness at low effort now read 6 (exp-012), 6 (012b), 4 (015, different config)
          and 6 levels here with 1.04, against 9 and 9 (1.23, 1.37) on 756a87e. exp-011c (756a87e, running) is the
          third sample of the pair; if it lands at 8-9 levels the harness commits after 756a87e get bisected
          (candidates: single-result iteration, region degrade, role key, cell events in the sandbox message).
Notes:    val across two runs: 0.79 and 0.0 (one level or none): the held-out games are hard for this agent and the
          number is noise-dominated at n=6; the dev/val gap stands. exp-018 (all 25 games concurrently, 3 h each,
          the submission's operating point) pushed 21:19 as arc3-eval-all-long-a.

## 2026-09-16 · exp-015 · adaptive effort with the raise to medium after 3 stagnant actions (instead of 6), harness ec12191 · REVERTED
Measured: dev 0.421 (runs/kaggle-repl-dev-015, kernel arc3-eval-dev-k v1; ran 19:24-20:32). Levels 4/142: lp85 L1, s5i5 L1, sb26 L1, su15 L1;
          actions 933; 0 model errors. Lowest of the day for the REPL agent: ar25 L1 and vc33 L1, solved in every
          other low-effort run, were missed.
Notes:    an earlier raise to medium buys nothing here. But the run adds to a pattern on the post-exp-011 harness: at low
          effort, exp-012 6, exp-012b 6 and exp-015 4 levels against 9 and 9 for exp-011/011b on harness 756a87e. The
          config differences are small (events line, then off; stagnation_actions), so the harness commits between
          756a87e and ec12191 are suspect (candidates: 'role': 'unknown' now printed for every entity in the
          observation; single-result iteration; region degrade). exp-016 (base config, harness ec12191, all 25 games)
          is the clean test on dev; a bisect follows if its dev subset lands at 6 levels or fewer.

## 2026-09-16 · exp-011v · base config (adaptive low effort, events line off, harness 4647ba8) on the VALIDATION split · first held-out number
Measured: val 0.794 (runs/kaggle-repl-val-011, kernel arc3-eval-val-a v1; 6 games, 1200 s each, 6 workers; ran 19:21-19:49).
          Levels 1/41: sp80 L1 in 39 actions (at the cap); actions 643; 0 model errors. g50t: 30/30 world-model
          predictions correct and no level (the same failure as dc22 on dev: a correct model of the mechanics without the
          goal); cn04 8/10 correct; r11l 473 actions without a level; lf52 (10 levels) and sc25 nothing.
          For scale: the code-only rules agent scored the same 0.794 on val (exp-006b, r11l L1), and the dev pair of this
          config is 1.229 / 1.372. The dev number carries a day of tuning on dev transcripts; val does not.
Notes:    val transcripts are not used for harness changes (the split exists to stay untuned). The gap says the next
          gains must come from general mechanisms rather than more per-game fixes. exp-016 = the base config on all 25
          games in one run (dev third sample, val second sample) is next.
          Local check of the 'goal sweep' idea on dc22 (plan to every goal hint, verified, one call): no strict plan
          exists for any hint because the avatar's walkable colour forms islands (docs/postmortems/dc22-islands-2026-09-16.md);
          the optimistic plans each cost 1-2 actions; a naive sweep after model retirement wasted 82 unverified actions.
          Not built: no game in hand where it would have completed a level. Lesson recorded in the post-mortem.

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

## 2026-09-16 · exp-018 · all 25 games concurrently, 3 h per game (submission operating point, lesson 0011) · KILLED (quota); found and fixed the image-limit bug
Why:      the submission plays every hidden game in parallel for ~9 h, about 6x the calls of a 1200 s dev run (lesson 0011);
          no run had been watched at that horizon.
Setup:    kernel arc3-eval-all-long-a v2 (v1 died on a SyntaxError from an apostrophe in the run note; fixed with
          note={a.note!r} + a compile test), 25 workers, time_budget_s 10800, watchdog 215 min, harness af19f73,
          same model/serving as exp-011; started 21:28, vLLM ready after 449 s.
Measured: cancelled at 23:13 (1 h 45 min in; the weekly GPU quota was down to 26 min with two sessions running) before
          any game finished, so no score. Live log (Kaggle GetKernelSessionLogsStream, saved in the scratchpad):
          from 22:51 every model call for ft09 and then s5i5 failed with vLLM 400 "At most 16 image(s) may be provided
          in one prompt" and was retried unchanged (554 and 340 failures). Cause and fix in docs/lessons/0012: the
          eviction only counted tokens, terse turns let 17+ board images accumulate, and a 400 is not a dead server.
          Fix (commit d8e3ce4): max_images=12 with image-only eviction of old observations, and an image-limit 400
          halves the cap and retries at once. Tests: test_image_cap_keeps_prompt_under_the_server_limit,
          test_image_limit_error_strips_images_and_retries_without_counting_an_error.
Kept:     the fix, yes (submission-path bug: at 9 h per game every game would have died this way). The measurement
          itself must be repeated after the quota resets (2026-09-19 00:00 UTC), with the fix in.

## 2026-09-16 · exp-017 · dev split at 3600 s per game, 8 workers (long-horizon check) · MEASURED (finished on borrowed quota)
Why:      does triple the per-game time buy levels, or does the agent stagnate? (Decides how much the 9 h operating point
          is worth versus a better harness.)
Setup:    kernel arc3-eval-dev-long-a v1, harness af19f73, exp-011 settings except time_budget_s 3600; started 21:44.
Measured: first batch of 8 games, all used the full hour: lp85 L1 (300 actions), ar25 L1 (116), dc22 0 (206 actions),
          ka59 0 (93), ft09 0 (52), cd82 0 (235), bp35 0 (321), ls20 0 (126): 2 levels. In the 1200 s runs the same
          eight games gave 1-2 levels (exp-011: ar25 L1, lp85 L1; exp-011b: ar25 L2, lp85 L1). Second batch (live log
          until the quota kill): s5i5 L1 at action 19 (541 s), tn36 L1 at action 69 (583 s), the rest unknown.
          Kaggle let the kernel run past the quota; it completed at 00:52 UTC on 2026-09-17 (runs/kaggle-repl-dev-017,
          kernel arc3-eval-dev-long-a v1, harness 59dad5c, 2 h 59 min wall). **dev 1.288, 10/142 levels**, 4634 actions,
          every game used its full hour: ar25 L1 (20 actions), lp85 L1 (10), m0r0 L1 (55), s5i5 L1 (19), sb26 L1 (17),
          su15 L1 (20), tn36 L1 (69), tu93 L1 (34), vc33 L2 (57, 16). 62-148 calls per game, 0 model errors, p50 latency
          16-65 s (8 concurrent games).
Notes:    against the six 1200 s runs of the same config family (1.229/1.372/0.775/0.692/0.608/1.041, 9/9/6/6/6/6 levels),
          triple the time per game gives a score inside the band and one extra level (vc33 L2, the only level 2 in the
          run) for 4-5x the actions (4634 vs 938-1180): tn36 spent 796 actions on its level 1, wa30 498 and tr87 355 on
          nothing. Time is not the bottleneck; the agent keeps acting without learning. This is the case for the memory
          and level-boundary arms (exp-019/020) and for an action budget per level, not for more throughput.

## 2026-09-16 · exp-019 · learning memory: harness-written and model-written lessons shown every turn, shared across the run's games, offline skill library · PLANNED (needs GPU quota, resets 2026-09-19 00:00 UTC)
Why:      exp-017 shows the agent stagnating with time to spare; exp-009/011 transcripts show the same mistakes repeated
          within a game (ka59: 14 mismatches of one stale model; dc22: 48 blocked moves) and across levels (re-probing a
          key map already known). Reflexion/ExpeL-style verbal memory is the cheapest known remedy; the Milestone 1
          winner refreshed a reflection memory every ~10 steps.
Change:   arc3/memory.py Lessons store (kinds recipe/hazard/mistake/mechanic/goal/strategy; dedupe by text; cap 40;
          rendered every turn as "Lessons (this game)"; shareable kinds appended under a file lock to
          <run>/shared_lessons.jsonl and rendered for the other games as "From other games in this run"). Harness-written
          lessons: level completed (recipe: actions and the last moves, the consistent win condition), game over
          (hazard), world model retired and batch stopped after three no-ops (mistake), each followed once by a
          "What did we learn?" question; STAGNATION notices ask for learn(). Sandbox: learn(text, kind) helper; the
          final message carries lessons and the cell's decisive flags. Prompt: the learn() line and Method step 7.
          Offline: scripts/mine_skills.py -> arc3/data/skills.json (18 cards from 79 winning transcripts: signature,
          the model's goal statement when it won, actions, compressed sequence), retrieved by code-computed level
          signature and never for the same game (no leakage on dev); shipped in the notebook tarball.
          Commits 1e6d44f, 44f0c90; tests tests/test_memory.py and the lessons test in tests/test_repl_agent.py.
Expected: fewer repeated no-op actions and fewer re-probes after a level change; a small cost in prompt tokens
          (~10 lines). Risk: a wrong lesson persists; mitigated by evidence-only harness lessons and the "hint, not
          fact" framing of cross-game lines.
Plan:     dev, 1200 s, 8 workers, same seed and settings as exp-011..015 (notebook scratchpad/nb/exp019, slug
          arc3-eval-dev-m, run name kaggle-repl-dev-019, rebuilt at HEAD 0608566 with level_consolidation off so it
          isolates the memory from the exp-020 level boundary); three runs a side per the noise rule (the base is the
          six-run set exp-011/011b/011c/012b/013b/015: 1.229/1.372/0.775/0.692/0.608/1.041). Config knobs for the
          ablation: memory (all off), memory_shared, skills, level_consolidation.
Measured: not run. GPU quota for the week is exhausted (used 29 h 34 min of 30 h at 23:10 UTC).

## 2026-09-16 · exp-020 · Tycho port bundle: level-boundary consolidation + conversation clear, observed terminal frame, animation note, friction line · PLANNED (needs GPU quota)
Why:      lesson 0013 (owner's reference, Tycho): the one harness structure they show adding score is a per-level
          conversation with a consolidation pass at the boundary (scribe writes level insights; the chat is cleared;
          notes, programs and plans persist on disk). Our agent keeps one growing conversation across levels and evicts by
          tokens, so stale reasoning from level 1 stays in the prompt on level 3, and the winning frame of a completed
          level was simulated rather than observed (the engine returns it as the first layer of the level-completing
          step: verified on vc33/ls20/ar25 replays).
Change:   repl agent: after a completed level, one or two "consolidation" model calls (no actions; act() is refused with
          a message) ask for learn()/note() entries on the goal, mechanics, recipe and mistakes plus an optional
          FRICTION line (kept in the transcript meta, never shown), then the conversation is cleared to the system prompt
          and the next level starts from a fresh observation with the lessons and notes. The tracker sees the observed
          terminal frame, so the level archive and the harness-side goal predicates use real evidence; the winning move
          is described in the level notice. Transient animation frames are noted as "[animation: N frames]" in the
          turn log. Config: level_consolidation (default True), consolidation_calls (2). Prompt Method step 5 updated.
          Tests: test_level_consolidation_records_lessons_refuses_actions_and_clears_history,
          test_consolidation_is_skipped_when_disabled_or_out_of_time.
Expected: fewer wasted actions on levels 2+ (no re-probing of a known key map, no stale plans), at most 2 extra calls
          per completed level. Risk: a level solved late leaves no time for the pass (skipped under 3 turns of time).
          Added after exp-017 (2026-09-17): a per-level ACTION BUDGET notice (config level_action_notice = N, off by
          default; shown at N actions on a level and at every doubling). Evidence: tn36, wa30 and tr87 spent 796, 498
          and 355 actions on one level with 85-98 percent of them changing the board, so the stagnation notice never
          fired; the model's own note on tn36 says "431-action loop was a waste". Every public level's human baseline is
          under 200, so past that a level is worth little and the notice points the model at the untested goal
          hypotheses instead (Tycho caps at 5x the baseline per level; the hidden set has no baselines, hence a fixed N).
Plan:     two arms against the six-run base: exp-019 (memory on, level_consolidation off, no action notice) and exp-020
          (memory + level boundary + level_action_notice 120), dev, 1200 s, 8 workers, three runs each. Notebooks built
          from HEAD in the scratchpad.
Measured: not run (GPU quota exhausted until 2026-09-19 00:00 UTC).

## 2026-09-16 · exp-021 · goal predicates over entity pairs (same_box, same_columns, same_rows, inside) · KEPT (code-only gate)
Why:      Tycho's residual failure and ours is objective inference (lesson 0013; dc22, g50t post-mortems). The harness
          lists win conditions consistent with every completed level, but scripts/goal_probe.py (replays the recorded
          winning actions of each solved level through the real agent and the observed terminal frame) found a
          consistent predicate on only 5 of 17 solved levels: ar25's "avatar parked on the target" fails
          overlap/touch because the sprite already touches the target one frame earlier.
Change:   dsl.goal_predicates adds relations between two distinct entities (same colour allowed): same_box (a sprite
          exactly on its slot), same_columns / same_rows (aligned in x or y without overlapping), inside (a smaller
          entity within a larger one of another colour); all still required to be true only at the terminal frame.
Measured: goal-predicate recall 5 -> 9 of 17 solved levels (ar25 L1/L2 same_box(4,11), lp85, s5i5, su15 L1 gained);
          bp35, cd82, ka59, m0r0, sb26, vc33 still none. Some new candidates are spurious (su15 same_columns(0,0)); the
          model sees them as candidates, the cross-level consistency check prunes them from level 2 on.
          The REPL's own goal_candidates() now uses the observed terminal frame too (the harness attaches it to the
          level-completing act() result; the sandbox pops it before the model sees the result and archives the real
          frame): on the ar25 replay the REPL lists same_box(colour 4, colour 11) first, and plan_rules accepts the
          relations as dict goals ({'same_box': (a, b)}, {'same_columns': ...}, {'same_rows': ...}, {'inside': ...}).
Kept:     yes (code-only; no model run). Follow-up: vc33-style alignment of a sub-part with a marker needs
          part-level entities.

## 2026-09-17 · research · road to 100 percent: evidence survey and ranked plan · DOCUMENT (no run)
Why:      the owner asked for research toward 100 percent. Gathered: ARC Prize verified leaderboard (2026-09-14: GPT-6
          Astra 62.7, Opus 5 30.2, GPT-5.6 Sol 7.8 on hidden games), ARC Prize's semi-private failure analysis (three
          inference failure modes), Tycho (100.00 on the public set with frontier models at 600-1,800 calls per game;
          +9.4 RHAE from harness structure), arXiv 2605.05138 (executable world models, 58.12 public, "premature
          commitment"), 2607.15439 (capability and effort dominate architecture), 2605.25931 (24 of 25 public games
          solvable by exploration alone), the Kaggle public leaderboard (18.81), candidate open models that fit one
          RTX PRO 6000 with their SM120 serving recipes and Kaggle-hub copies, the compute budget at the real operating
          point, and the state of the human replay data (blocked: rate-limited download link, no API endpoint).
Result:   docs/research/road-to-100.md. Ranked plan: (1) model swap A/B (gpt-oss-120b MXFP4, Nemotron 3 Super NVFP4,
          Devstral Small 2), (2) effort and action-budget arm at 9 h, (3) exploration-first probe sweep (built below as
          exp-022), (4) goal-hypothesis discrimination, (5) exp-019/020 as queued, (6) builder role, (7) replay priors
          (blocked), (8) fine-tune (October at the earliest). Honest reading: no system reaches 100 on hidden games; the
          November bar is the public leader (about 19), 10-15x our base.

## 2026-09-17 · exp-022 · exploration-first probe sweep at level start (config explore_first, off by default) · PLANNED (needs GPU quota)
Why:      arXiv 2605.25931 found 24 of 25 public games solvable by systematic exploration; the post-mortems (dc22, g50t,
          tn36) and Tycho's residual failure are a model committing to a goal before it has pressed every key or clicked
          every class of entity. The prompt's Method step 1 asks the model to do this itself, but the transcripts show it
          skipped or half-done (exp-017: tn36 431-action loop with no untested key). The harness can do it for free in
          calls: no model turn until the sweep is over.
Change:   repl agent: at each level start (once per level index, only while the model has not acted on the level, never
          on a game over, a finished game or with less than a turn of time left) the harness takes at most explore_first
          actions itself: each legal key once, ACT once, then one click per entity class (colour, shape) largest first,
          skipping HUD entities and anything larger than a quarter of the board, up to explore_first_clicks. The effects
          go to a PROBE SWEEP line in the next observation (shown once), not the turn log; a level completed or a game
          over under the sweep drops the rest of it. Stats: sweep_actions. Tests:
          test_explore_first_sweep_is_built_once_per_level_and_reported_once,
          test_explore_first_sweep_precedes_the_first_model_turn_on_a_real_game.
Expected: fewer actions before the first goal-directed sequence, no untested keys on stuck levels; cost 5-12 actions per
          level (every public level's human baseline is above that except a few click levels, where the cap should be
          low). Risk: a click on a hazard ends a level early (the rules agent's identical sweep has not shown this on the
          public games), and a level whose baseline is under 10 loses efficiency.
Plan:     dev, 1200 s, 8 workers, three runs, config explore_first 8, explore_first_clicks 4, on top of the exp-020
          bundle (level_consolidation on, level_action_notice 120), against the six-run base and the exp-020 arm.
Measured: not run (GPU quota exhausted until 2026-09-19 00:00 UTC).

## 2026-09-17 · serving checks prepared for two candidate models (road-to-100 item 1) · READY (needs GPU quota)
Why:      road-to-100 ranks the served model as the largest term of the gap. Two candidates have Kaggle-hub copies and
          public single-RTX-PRO-6000 recipes: gpt-oss-120b MXFP4 (Apache-2.0, 65 GB) and Nemotron 3 Super 120B-A12B
          NVFP4 (NVIDIA Nemotron Open Model License, 80 GB). Devstral Small 2 has no hub copy (upload needed) and waits.
Facts:    vLLM 0.27.1 source (tag v0.27.1): reasoning parsers `openai_gptoss`, `nemotron_v3`, `qwen3`; tool parsers
          `openai`, `qwen3_coder`, `hermes`; `--moe-backend {marlin, triton, flashinfer_cutlass, flashinfer_trtllm, ...}`
          (the MXFP4 oracle otherwise picks by capability; the FlashInfer TRT-LLM kernels are SM100-only and the CUTLASS
          path needed a custom FlashInfer JIT build in the one public SM120 write-up, which is what fails offline for us);
          vLLM accepts the OpenAI request field `reasoning_effort` (injects enable_thinking; harmony models read it).
          Nemotron: model card recipe (fp8 KV, mamba cache float16, 32 seqs, super_v3 plugin shipped in the checkpoint,
          qwen3_coder tools, temperature 1.0 / top_p 0.95); RTX 6000 Pro report: fits in ~77 GB with MTP off, MTP OOMs;
          DGX Spark recipe on the 0.27.1 container uses VLLM_NVFP4_GEMM_BACKEND=marlin + VLLM_USE_FLASHINFER_MOE_FP4=0.
          Kaggle kernel metadata takes models as `model_sources` entries owner/slug/framework/variation/version.
Change:   build_diag_notebook.py: model refs as `model_sources` (the notebook finds the folder holding config.json under
          /kaggle/input), --no-image (text-only probe and smoke), --efforts, --smoke-config, --effort-in-request,
          `{MODEL_DIR}` placeholder in --attempts-json. serve.build_vllm_command: attention_backend kwarg,
          images_per_prompt 0 omits the image limit. llm.ChatClient(effort_in_request=True) and the REPL config key
          effort_in_request. Tests in tests/test_serve.py and tests/test_llm.py. Notes: docs/models/gpt-oss-120b-mxfp4,
          docs/models/nemotron-3-super-120b-a12b-nvfp4.
Plan:     after the 2026-09-19 reset and the exp-019/020/022 runs: push scratchpad/nb/diag-gptoss (ladder: Marlin MoE fp8
          KV; Marlin auto KV eager; Triton MoE) and scratchpad/nb/diag-nemotron (default kernels; Marlin GEMM env; Marlin
          auto KV eager), 50 min budget each. Gate: aggregate tok/s at 8 concurrent >= 308 (27B, diag v5) and the
          ls20 + vc33 smoke not worse than the 27B's; then one dev run for the winner (needs the eval builder to take a
          model source and the text-only config).
Measured: not run.

## 2026-09-17 · exp-022 (bundle addition) · goal-hypothesis discrimination: distances, falsification, goal_probe() · BUILT (code-only)
Why:      road-to-100 item 4 and Tycho's residual failure: the harness lists win conditions consistent with completed
          levels (exp-021: 9 of 17 solved levels have one) but nothing ranks them, nothing says which the current level
          has already ruled out, and the model commits to one (dc22, g50t, tn36 post-mortems). The cheapest test of a
          goal hypothesis is to satisfy it: the level either completes or the hypothesis is falsified.
Change:   dsl: goal_predicates() entries carry kind/args; goal_kind() parses candidate names and plan_rules dict goals;
          goal_distance() is a cheap geometric distance per kind (counts, bbox gaps, box/column/row offsets, stick-out
          for inside, avatar gap for reach); goal_progress() ranks hypotheses (live nearest first, falsified last) with
          the change over the last actions and an incremental falsification check; render_goal_progress() is the line.
          sandbox: goal_progress() (candidates + goal_hints goals) and goal_probe() (the cheapest live hypothesis with
          its plan under the fitted rules, plan_rules with a small node budget; act(plan) wins or falsifies). REPL
          agent: from level 2 on the observation carries "Goal hypotheses (code-computed; N live, M falsified): ...";
          config goal_progress_in_prompt (default on: geometry only, no action, no model call); the falsification
          cache resets with the tracker. Prompt: Method step 2 and the helper listing. Tests: test_dsl
          (kinds, distances, progress, incremental falsification), test_sandbox (GridWorld level 2: ranked live
          hypotheses, goal_probe plan), test_repl_agent (line appears from level 2, falsifies on evidence, knob off).
Expected: on levels 2+, fewer actions spent on a goal the level has already ruled out; the model reaches for
          goal_probe() instead of guessing. Cost: a few ms per turn.
Plan:     rides in the exp-022 bundle (exploration sweep + goal hypotheses) against exp-020, three runs; ablation knobs
          explore_first and goal_progress_in_prompt.
Measured: code-only gate (scripts/goal_probe.py --max-levels 3, 2026-09-17, the latest recorded winning run per game
          in skills.json): 13 solved levels replayed, 7 with at least one consistent predicate after the level. Levels
          >= 2 that had level-1 candidates: 2 (ar25 L2: 4 candidates, su15 L2: 9); in both the eventual winner was
          ranked first among the live hypotheses one frame before the win (goal_probe() would have planned the right
          goal first); no candidate was falsified on these replays, as expected of winning runs that go straight to
          the goal (falsification only pays on the exploring runs the model actually produces). n = 2: a sanity check,
          not evidence of a gain. Model run: not yet (GPU quota exhausted until 2026-09-19 00:00 UTC).

## 2026-09-17 · human replay summariser (road-to-100 item 7, data still blocked) · BUILT (code-only)
Change:   scripts/human_replays.py parses the documented recording JSONL (docs.arcprize.org/recordings: one line per
          action with levels_completed and action_input) into per-level action sequences (RESET billed on level 1, the
          level-completing action on the level it completes) and writes arc3/data/human_priors.json: per game the
          replays, wins, median actions per completed level over winning replays, the first three actions and the
          click fraction; cross-game aggregates for hidden-game priors (per-game figures are for dev-split analysis only:
          using them in play on dev games would be leakage). Test on a synthetic recording (tests/test_human_replays.py).
Blocked:  the dataset itself (docs/status.md follow-ups: browser download by the owner).

## 2026-09-17 · decisions from the owner: Apache-2.0, publication, submission condition, Nemotron · RECORDED
Owner:    license Apache-2.0 (LICENSE at the root, packed into the notebook bundle, named in the notebook header,
          pyproject license field); the notebook may be made public; "you decide" on submitting, with the condition
          "fully verify it will get over 50 or 100 percent and the code works"; "decide if you want to use nemotron".
Decided:  publish together with a submission, not before. No submission under the "over 50" condition: the metric is
          percent of human-level RHAE, our dev base is 0.6-1.4 and the public leader 18.81, so no run of ours can be
          verified above 50; said plainly in docs/status.md with the alternative (an entry at the expected 1-3 for the
          leaderboard position and the end-to-end validation) left to the owner. Nemotron 3 Super not used for the
          submission (NVIDIA Open Model License, not OSI; about 16 GB of KV headroom on 96 GB); its serving check is
          dropped from the queue; gpt-oss-120b (Apache-2.0) stays the candidate. The private Save & Run All of the
          submission notebook at HEAD moves to the front of the quota queue as the Milestone 2 gate.

## 2026-09-17 · human recording replay (ls20, 546 actions, 7 levels) through the agent · MEASURED (code-only)
Why:      the owner uploaded one file of the ARC Prize human dataset (ls20-9607627b, guid 8aed7120, a WIN in 546
          actions; per level 21/123/39/92/54/108/109 against the published baselines 22/123/73/84/96/192/186).
          scripts/goal_probe.py --recordings replays it through the real REPL agent (mock model feeding the recorded
          actions one per call; a batch of six lost the actions after a level completion, fixed) and runs the goal gates.
Measured: engine determinism: 0 of 546 engine frames differ from the recorded ones (the local engine reproduces the
          site's play exactly). Goal predicates: after L1 and L2 the only consistent candidate is count(colour 9) == 4;
          on L3 it came true without completing the level, so the per-level falsification removed it (the mechanism
          works on a real trajectory), and from L3 on NO predicate in the library is consistent with all completed
          levels: ls20's win condition is outside dsl.goal_predicates (the level ends when the moved shape matches the
          reference; a shape-equality relation between two entities is the missing kind). Winner rank: L2 1 of 1,
          L3 none (no candidate survives). Human priors from the same file (scripts/human_replays.py): first actions
          UP x8 then DOWN x2 (a human probes each key several times), click fraction 0.
Follow-up: add "shape_matches(a, b)" (equal masks up to translation) and "count(colour) == count(colour)" style
          relations to goal_predicates and re-run this gate; the rest of the human dataset (341 files) turns this into
          a real recall measurement across games.

## 2026-09-17 · champion record, champion preset, research-status tool, no-op measurement · KEPT (code-only)
Why:      the owner's research brief (2026-09-17): re-establish the champion as an immutable comparison point, make sure
          recent changes did not silently move the baseline, and answer "what is best / what changed / which failure
          dominates" from machine-readable logs.
Found:    the agent's defaults had drifted: memory, level consolidation and the goal-hypotheses line are on by default
          since exp-019/020/022 were built, so the submission notebook and any control run at HEAD would have run an
          unmeasured bundle, not the exp-011 configuration.
Change:   docs/champion.md (exp-011 pair: 1.229 / 1.372 / 0.775 on the same notebook, val 0.794, per-game table, serving
          flags, VRAM, weaknesses). arc3/presets.py: CHAMPION (every knob added since exp-011 off) and BUNDLE; the
          submission builder defaults to CHAMPION (test), the eval builder and scripts/eval.py take --preset.
          scripts/research_status.py: best run, git diff since its commit, score spread, failure categories, seconds per
          action, time hogs, games solved in every run / never, game x run matrix (--matrix), JSON output.
Measured: research_status on 18 dev runs: best kaggle-repl-dev-011b 1.372 / 9 levels; spread 0.32-1.37 (median 0.81);
          never solved in any run: dc22, ft09, sk48, tr87, wa30; solved in every run: sb26; level 3 never reached.
          No-op measurement on runs/kaggle-repl-dev-017 action logs: 4634 actions, 434 changed nothing (9.4 percent),
          293 (6.3 percent) re-sent a (frame hash, action) pair already observed to change nothing (wa30 67, cd82 62,
          s5i5 38, lp85 34, sk48 33); 1349 (29 percent) re-sent a (frame, action) pair of any kind (loops). On solved
          levels the repeats cost score directly (s5i5 L1: 155 actions, 38 known no-ops).
Next:     exp-024 control (champion preset at HEAD) first in the quota queue, before the exp-019/020/022 arms, so every
          arm compares against a control from the same harness; exp-023 no-op memory (task #44).

## 2026-09-17 · exp-023 · no-op memory: known (frame, action) no-ops shown and flagged; exp-023b hard skip · PLANNED (needs GPU quota)
Why:      exp-017 action logs: 293 of 4634 actions (6.3 percent) re-sent a (frame hash, action) pair already observed to
          change nothing (wa30 67, cd82 62, s5i5 38, lp85 34, sk48 33); on a solved level every such action costs
          score directly. The engine is deterministic (the human ls20 recording replayed with 0 of 546 frame
          mismatches), so the outcome of such a pair is known.
Change:   repl agent: per-game map frame hash -> action labels that changed nothing from that frame (recorded in
          observe(); never RESET, never on a level change, game over or an animated step). noop_memory (default on in
          code, off in the champion preset): the observation lists the current frame's known no-ops and a re-sent one
          carries 'known_noop_repeat' in its act() result (stat noop_repeats). noop_skip (default off): the harness
          answers a known no-op from memory without sending it ('skipped_known_noop', stat noop_skipped); act(...,
          force=True) sends it anyway, for games whose hidden timers need repeats. Prompt: act() description.
          Test: test_noop_memory_records_flags_and_optionally_skips.
Expected: fewer actions on solved levels (s5i5 L1: 155 -> 117 in exp-017 terms), fewer wasted actions on stuck
          levels; no effect on levels where the model never repeats. Risk (hard skip only): a game that needs the same
          key repeated from a static frame (hidden counter) stalls; the soft arm has no such risk.
Plan:     exp-023 = champion preset + noop_memory; exp-023b = + noop_skip; three runs each against the exp-024 control
          (champion preset at HEAD), dev, 1200 s, 8 workers.
Measured: not run (GPU quota exhausted until 2026-09-19 00:00 UTC).

## 2026-09-17 · exp-021b · goal library: plain-component frames, background holes, avatar-relative kinds, shape kinds · KEPT (code-only gate)
Why:      the human ls20 recording had no consistent win predicate from level 3 on (entry above). Tracing it exposed
          three perception faults, not one predicate gap (lesson 0015).
Change:   entities.Tracker.plain_frames(): plain connected components per observed grid (grids stored beside the
          symbolic frames), tracker ids lent by overlap so the avatar id carries over, small enclosed background-
          coloured components kept. dsl: goal kinds shape_matches, vanish(colour, shape), avatar_inside(colour),
          avatar_touch(colour) (per-level avatar ids); goal_candidates_dual / goal_progress_dual evaluate candidates on
          the compound frames and on the plain frames (entries tagged rep). Harness and sandbox archive plain frames
          and avatar ids per level; the single-layer WIN frame counts as the observed terminal; goal_probe.py replays
          human recordings with one action per call. Tests in test_entities, test_dsl, test_sandbox, test_repl_agent.
Measured: human ls20 recording (7 levels): a consistent predicate on 7 of 7 levels (was 2 of 7): avatar_inside(colour 5)
          on every level; on the 6 levels >= 2 the eventual winner was the cheapest live hypothesis before the win 6
          times (was 1); the falsification removed 2, 3 and 1 spurious candidates on levels 2, 3 and 5. Engine vs
          recording: 0 of 546 frames differ. Recall on our own recorded runs (goal_probe --max-levels 3): see the next
          entry once the run finishes.
Kept:     yes (code-only; the model-facing change is the goal line and goal_probe() output, part of the exp-022 bundle).

## 2026-09-17 · bg_holes · tracker knob: small enclosed background-coloured islands are entities · BUILT (in the bundle, off in the champion)
Why:      lesson 0015 fault 2: the tracker ignores every background-coloured cell, so a socket or slot drawn in the
          background colour is invisible to ents(), the rule fitter and goal_hints() (ls20 level 7).
Change:   entities.segments(grid, bg, bg_holes): with the knob, background-coloured components that are small (<= 400
          cells) and enclosed (not touching the border) stay; Tracker(bg_holes=...) applies it in reset() and update();
          the REPL agent sets ARC3_BG_HOLES from config so the sandbox child's tracker agrees. Presets: champion off,
          bundle on. Test: test_bg_holes_knob_makes_enclosed_background_islands_entities.
Expected: goal_hints and rules see slots on games that draw them in the background colour; risk: extra entities on
          games with decorative background pockets (the 400-cell and border filters bound it).
Measured: not run (rides in the exp-022 bundle).

## 2026-09-17 · disagreement_probe() · experiment selection over alive hypotheses (brief item 12) · BUILT (code-only)
Why:      the sandbox keeps competing world models (set_models) and kills them on real actions, but nothing told the
          model which action would separate the survivors; probe_suggestions() only lists untested actions.
Change:   sandbox disagreement_probe(actions=None): for each candidate action (legal keys and ACT plus the clicks
          probe_suggestions() proposes, or the model's own list) the number of distinct next grids the alive
          hypotheses predict, best first, with the hypothesis names per outcome; [] with fewer than two alive. Prompt:
          the set_models line points at it. Test: test_disagreement_probe_ranks_actions_that_separate_alive_hypotheses.
Measured: not run (a helper the model may call; part of the exp-022 bundle's prompt).

## 2026-09-17 · postmortem knob · structured post-mortem call at the end of an unsolved game (brief item 15) · BUILT (research data)
Change:   repl agent: with postmortem on and at least postmortem_min_s (20 s) left, close() makes one model call under the
          fixed headings (WHAT DID WE BELIEVE? ... CONFIDENCE:), stores the text in the transcript meta and writes
          <game>.postmortem.md next to the transcript; skipped for a won game, a model that never answered or a dead
          server; never in the champion preset (on in the evaluation bundle). Stat postmortems. Test.
Use:      the per-game post-mortems feed docs/postmortems and the failure matrix (which assumption was wrong, what
          cheaper test would have caught it) without reading whole transcripts.

## 2026-09-17 · skill memory statuses (brief item 16) · BUILT and re-mined (code-only)
Change:   scripts/mine_skills.py cards carry evidence and a status: wins, failures (transcripts of the game that reached
          the level and did not complete it), confidence = wins / (wins + failures), last_validated, status validated
          (>= 2 wins) / candidate (1 win, <= 2 failures) / deprecated (1 win, > 2 failures: a one-off later runs did not
          reproduce). memory.match_skills never returns deprecated cards and ranks validated above candidate;
          render_skills shows the status and counts. Test extended.
Measured: re-mined arc3/data/skills.json from 88 winning transcripts: 18 cards, 12 validated, 0 candidate, 6 deprecated
          (ar25 L2, cd82 L1, ft09 L1, ka59 L1, ls20 L1, su15 L2: each won once in 13-21 attempts). The validated cards
          with the highest confidence: sb26 L1 0.72, su15 L1 0.67, ar25 L1 0.63, lp85 L1 0.50; the lowest: cd82/ft09/ka59/
          ls20 L1 at 0.05. This is the stable-versus-lucky split the game x run matrix showed, now on the cards.

## 2026-09-17 · exp-021b gate on our own recorded runs (goal_probe --max-levels 3, after the re-mine) · MEASURED (code-only)
Measured: 18 solved levels replayed (the re-mined library picks the latest winning run per game), 11 with at least one
          predicate consistent after the level (61 percent; the previous set: 7 of 13, 54 percent). New: ka59 L1
          (vanish(colour 1, shape), count(colour 1) == 2, aligned), lp85 L1 (same_rows), tn36 L1 (vanish(colour 9,
          shape), touch(0, 10)), vc33 L1 (aligned(colour 11)), ls20 L1 (avatar_inside(colour 5)). Still none: bp35,
          cd82, ft09, m0r0, sb26 L1/L2, vc33 L2 (a sub-part aligned with a marker needs part-level entities).
          Levels >= 2 with level-1 candidates: 3; the eventual winner was the cheapest live hypothesis before the
          win on ar25 L2 (1 of 5) and su15 L2 (1 of 26 live; the library over-generates on su15), none survives on
          vc33 L2. Engine determinism holds on every replay.
Kept:     yes. Follow-ups: part-level entities for vc33-style goals; prune the over-generation on colour-0 pairs.
