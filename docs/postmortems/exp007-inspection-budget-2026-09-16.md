# exp-007: the model spends its 20 minutes looking, not acting

Run: `runs/kaggle-repl-dev-007` (dev, Qwen3.8-27B-FP8, low reasoning, 1200 s/game, 8 concurrent).

## What the agent believed
That inspection is cheap. The prompt said "each python call costs roughly 10 seconds".

## What happened
- 580 tool calls over 19 games, 277 of them with an `act(...)`; 122 turns; 2.5 inspection-only calls per turn.
- Mean model latency per call 23-49 s (8 concurrent requests share the GPU), so a game gets ~35 calls in 1200 s
  and takes 3-91 environment actions. lp85 took 3 actions in 20 minutes; ft09 took 8.
- Typical inspection call: print rows of `grid`, sample cells, list `objects()` again, although the observation
  already carried `ents()` with ids and roles and the per-action entity events.
- The rule loop worked when used: ls20 `auto_rules()` found the move rule after 4 key presses (coverage 0.5, the
  rest HUD growth the v3 fitter did not exclude); the planners then returned None because the target sits on
  never-walked terrain and both planners were strict. The model spent the remaining budget probing by hand.

## Which assumption was wrong and what showed it
"Calls are cheap" is wrong by a factor of three; transcripts (`scripts/transcript_report.py`) showed it.
"Plans are exact or nothing" is wrong: an optimistic plan whose first wrong step is verified is a cheaper probe
than hand-written probing.

## The cheaper test that would have caught it
Reading one transcript of exp-003c would have shown the inspection pattern before building more helpers; the
transcripts did not exist then (added the same day).

## Fixes (in the harness now, measured in exp-008/009)
- Nudge after the first inspection-only call, quoting the measured per-call cost and the calls left.
- Automatic per-turn "Rules (auto-fitted ...)" line with coverage and unexplained items, so the model does not
  need a call to know whether planning applies.
- `plan_to`/`plan_to_entity`/`plan_rules` fall back to optimistic paths (`PLAN['optimistic']`).
- Fitter v5 excludes HUD bars and occlusion artefacts, so coverage reflects mechanics (code-only 0.10 -> 0.43).

## Is the lesson general?
Yes for every model-in-the-loop arm: wall-clock per call is the binding constraint, not tokens per second.
See `docs/lessons/0009-calls-are-the-budget.md`.
