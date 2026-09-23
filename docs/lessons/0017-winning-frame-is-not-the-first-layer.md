# A completed level's winning frame is the layer before the level switch, not the first layer of the step

Found 2026-09-23 (exp-028): the goal-induction collector wraps the engine's `next_level()` to render the board at the
exact moment a game declares a level won, and compares it with the layers the agent receives for that step.

- The engine renders one layer per internal step of an action (`ARCBaseGame.perform_action`). A win renders the
  finished board; when the level switch happens inside the same action, the next layer is the next level's start.
- Animated wins put the finished board late: cd82's pour is a 16-layer step whose `layers[0]` is still the board
  before the pour (the finished board is `layers[14]`, the next level `layers[15]`); tu93's winning move is 9 layers.
  One-frame wins (most games) have `layers[0]` equal to the finished board, which is why the 2026-09-16 check on
  vc33/ls20/ar25 replays looked fine.
- Rule now used by the REPL harness: `perception.terminal_layer(layers, before)`: if the last consecutive change in
  the step is a jump (2x every earlier change including the move's own, at least 20 cells), the terminal is the
  layer before it; otherwise the last layer. Exact on all 43 dev levels collected in exp-028 (explorer and
  optimal-play data); `layers[0]` is exact on 31. Other long wins: sk48 (39 layers: a 35-frame flash), su15 (15).
- If the switch is still pending when the action completes, the step's last layer is the finished board and the
  next action's first layer is the new level (not seen on the dev levels so far; the rule handles it).

Why it matters: the level archive and every goal predicate built from it were fed the board before the winning move
on animated games, so the win condition could never be found there.
