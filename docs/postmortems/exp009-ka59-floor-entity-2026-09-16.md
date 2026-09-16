# exp-009 ka59: fourteen mismatches against a plan that was right

Run: `runs/kaggle-repl-dev-009/ka59.transcript.jsonl` (dev, Qwen3.8-27B-FP8, low reasoning, 1200 s/game).

## What the agent believed
That `set_model(move_model().predict)` verifies the plan from `plan_to_entity(7)`, so a `pred_ok: False` means the
world surprised it and the model must be revised by hand.

## What happened
- After the four probe keys the strict planner returned None (the floor of ka59 is one large colour-1 entity and the
  move model treated every entity mask as an obstacle), so the optimistic plan `['RIGHT', 'RIGHT']` was executed with
  the *strict* predictor registered. The first step moved the avatar exactly as planned; the strict model predicted
  "blocked"; 19 wrong cells; the batch stopped.
- The model re-issued the same move eleven more times, one action per call, each time reading the same mismatch,
  because the registered predictor was a snapshot of the stale fit. 14 mismatches, 21 actions, 37 calls, no level.
- Halfway through, the avatar merged into another sprite of its colour (#7) and control passed to it; `avatar()` kept
  reporting the dead id 6, so `move_model()` raised and the model probed by hand.

## Which assumption was wrong and what showed it
"An entity is an obstacle unless the avatar has already occupied those exact cells" is wrong for floors drawn as
objects: the evidence that colour 1 is walkable should apply to every colour-1 cell. A local replay
(`scratchpad/replay_ka59.py` over `environment_files`) reproduced it in one minute: strict plan None, relaxed plan
found, strict prediction wrong on a correct move.

## The cheaper test that would have caught it
Replaying the first six actions of each dev game through the tracker and move model after every planner change
(the ka59 replay is now the pattern; `tests/test_planner.py::test_floor_entity_the_avatar_stood_on_is_walkable` keeps
the case).

## Fixes (in the harness for exp-011)
- Walkable colours from the static layer under the avatar's history; companions move with the sprite; erase with
  terrain colour; thin resizing edge strips are HUD.
- `set_model(move_model().predict)` becomes a live, re-fitting predictor that follows `PLAN['optimistic']`.
- Three consecutive mismatches retire a model (`pred_retired`) instead of stopping every batch after one action.
- `avatar()` hands over to the entity that moves now.

## Is the lesson general?
Yes: every learned obstacle map should generalise by terrain colour, not by visited cells, and a verifier that can
be wrong forever must have a stop rule. See lesson 0010.
