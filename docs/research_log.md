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
