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
            per game >= 0.5: ar25 1.00, ft09 1.00, m0r0 1.00, sp80 1.00, tn36 1.00, ls20 0.98, re86 0.96, dc22 0.70, ka59 0.63
            0.0: s5i5, su15, vc33 (blind clicks that do nothing; a swap mechanic the fitter does not yet match)
          CPU only: play + fit under 4 s per game. No GPU run yet: exp-005 (tracker + planner) is still queued on Kaggle.
Notes:    this is a lower bound (blind probe, one level) and not a score. It is the exp-006 gate metric: the fraction of
          observed mechanics the library can express. Remaining unexplained kinds: 'resized' (bars/counters that change by
          varying amounts, growth bars), 'moved' on click games (a marker jumping to the clicked column: OnClick 'goto'),
          scrolling worlds (sk48: everything but the avatar moves; needs a class exclusion), and 'appeared' (spawns).
          Lessons: docs/lessons/0008-terrain-vs-sprites.md. Next: exp-005 result; then the same probe driven by the model
          (does auto_rules() + plan_rules() reduce actions per solved level on ls20/re86/ar25/dc22?).
