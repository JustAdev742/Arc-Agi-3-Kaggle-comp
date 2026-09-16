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
