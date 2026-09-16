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

## 2026-09-16 · exp-011 · verified-navigation fixes from the exp-009 transcripts (walkable floor entities, sprite companions, avatar hand-over, live move model, model retirement, inline events, batched click probes) · PENDING (built, not yet run)
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
Measured: not yet run (kernel queue: exp-010 council and exp-009b occupy both slots).
