# Tycho (NIMI Research, 2026): what a 100 percent public-set harness does that ours does not, and what transfers to a 27B run offline

Reference supplied by the owner on 2026-09-16: github.com/NIMI-research/Tycho (Apache-2.0) and arXiv 2607.28287
("Tycho: Active Abstraction with Programmatic World Models for ARC-AGI-3"). Clone and extracted paper text in the
session scratchpad; its 164 tests pass here in host-sandbox mode with our venv's exact dependency pins
(arc-agi 0.9.9, arcengine 0.9.3).

## The numbers that matter for us

| Run (25 public games, 183 levels) | RHAE | games / levels | scored actions | model calls | peak prompt tokens |
|---|---:|---:|---:|---:|---:|
| Opus 4.8, no world model (direct reasoning + typed evidence) | 79.07 | 19 / 157 | 12,997 | 22,856 | 850k |
| Opus 4.8, single (actor edits world_model.py) | 85.36 | 19 / 162 | 11,576 | 25,640 | 851k |
| Opus 4.8, orchestrator (actor calls a builder subagent) | 88.49 | 21 / 166 | 10,354 | 24,054 | 852k |
| Opus 4.8, trigger (harness calls the builder on verification failure) | 83.07 | 18 / 162 | 9,442 | 44,391 | 852k |
| GPT-5.6 Sol, orchestrator | 100.00 | 25 / 183 | 7,766 | 26,505 | 219k |
| Opus 5, orchestrator | 100.00 | 25 / 183 | 6,641 | 15,103 | 300k |

Settings: reasoning effort xhigh, 24,000 output tokens per call, 3,500 calls per game, 40 tool steps per turn, cost
cap 750-1,500 USD per game (2.99k-5.78k USD per 25-game run). Our best dev run is 1.37 RHAE with a 27B model at
32k context, about 30-60 calls per game, 1200 s per game.

Reading: the gap is first the model (a frontier model with only typed evidence and notes already reaches 79), and
second the operating point (600-1,800 calls per game, 1M output tokens per game). Neither is available offline
on one GPU in 9 hours for all hidden games. What is available is the harness structure, which they show adds
+9.4 RHAE (orchestrator over direct reasoning, interval excluding zero) and 19-36 percent fewer actions.

## Structures Tycho has that arc3 lacks (candidates, each an ablation)

1. **Frame roles.** Every recorded grid is tagged decision / transient (animation) / level-complete / game-over /
   reset-init / next-level-init. The actor sees only decision frames; animation, terminal and reset evidence
   are separate typed records the code can query. arc3 shows the model whatever frame the engine returned and
   diffs consecutive frames, so animation frames pollute the entity log and the rule fitter.
2. **Executable model contract with an outcome function.** `init_state(frame, level)`, `transition(state, action)`,
   `render(state)` (may abstain per cell with -1), `outcome(state)` in {ongoing, level_complete, game_over};
   optional `actions`, `subgoals`, `heuristic`. Verification replays the level's attempt and reports accepted
   transition match, known-cell accuracy, coverage, and terminal recall separately, because "a model may
   reproduce motion while misidentifying the goal or a hazard". arc3's `set_model(predict)` checks only the next
   grid; the goal lives in a separate `plan_rules(goal)` argument and is never verified against terminals.
3. **Level-boundary consolidation.** The conversation is cleared at every level boundary; a consolidation pass
   writes a level summary first; notes, helper programs, the model and validated plans persist in an on-disk
   workspace; resets archive attempts instead of erasing them. arc3 evicts by token budget and keeps one
   growing conversation; the new Lessons store (exp-019) is the first step toward the consolidation pass.
4. **Guarded plan execution.** `plan.py` writes a validated route with an expected-frame hash per step; the
   harness surfaces the next action only while the model is unchanged, the observed frame matches the hash and
   the action is available; the actor commits one action at a time. arc3 executes a whole batch and stops on the
   first mismatch (equivalent for the mismatch case, worse for re-planning: no hash, no route artifact).
5. **Actor-requested builder (orchestrator).** The actor decides when to delegate model construction to a
   focused subagent with the full evidence; automatic triggering (trigger policy) produced the most exact models
   (99.97 percent accepted match) and a lower score (83.07 vs 88.49) with 8x the builder calls. Lesson for our
   parked council: specialists must be called by the actor when it wants them, not on every event.
6. **Objective inference is the residual failure.** sk48: exact transition model, four plausible goal predicates,
   314 actions, no level. Our dc22 and g50t failures are the same shape (post-mortems). Their partial remedies:
   verify outcomes against recorded terminals, discriminate hypotheses by cheap probes, and keep competing goal
   hypotheses in notes rather than committing to one.
7. **Meta-reflection.** Actor and builder are asked for one or two sources of friction per run; the notes are
   read only after the run and turned into harness changes by the authors. Equivalent to our post-mortems, but
   the model writes the first draft. Cheap to add to the transcript (one extra field at close), never in-prompt.

## What does not transfer

- The context regime (200k-850k prompt tokens, 24k output tokens per call) and the call counts. At 32k context
  and ~300 tok/s aggregate, one Tycho game would take longer than our whole 9-hour budget.
- Docker isolation (the host runtime exists; Kaggle has no Docker; our subprocess sandbox is the equivalent).
- Their five harness-development games (tr87, vc33, r11l, bp35, ft09) overlap our splits (vc33, bp35, ft09 are
  in dev; r11l is in val), so their per-game numbers are not a clean reference for our val split.

## Decision

Not a replacement: running Tycho unchanged with the 27B is outside the time budget by an order of magnitude.
Port, in this order, each as an ablation against the six-run base: (1) frame roles and animation evidence in
the tracker and the observation; (2) `outcome` in the model contract with terminal verification and a
goal-hypothesis list in the observation; (3) level-boundary consolidation on top of the Lessons store (clear the
conversation, keep a level summary, notes, model and plans); (4) guarded routes with frame hashes. Keep the
transport option open: Tycho's `openai` backend plus `TYCHO_SANDBOX_RUNTIME=host` can drive our vLLM server,
which makes a reduced-context Tycho run on one or two dev games a cheap calibration of how much of the gap is
the harness (needs GPU quota).

## Verified in the code (two read-throughs of the repository, 2026-09-16 night)

- One board image per turn (current frame only), `image_cap` 4 hard ceiling "must be <= server --limit-mm-per-prompt";
  the full 64x64 grid as spaced text every turn (`text_grid: full`) plus a compact lossless diff of the last action;
  Qwen models get lossless 32 px cells (2048 px images), others 6 px lossy. Old images are evicted first (keep 3).
- One conversation per level: on level completion a scribe pass of at most 3 calls writes `notes/level_L_insights.md`
  ("Do not take an action"), then the history is cleared; the next turn says "Your conversation was cleared at the
  boundary; carry-forward is on disk." Emergency compaction only above 850k prompt tokens.
- Tools: ls/read/write/edit, run_python (fresh subprocess per call, 15 s default, `wmlib` preloaded), set_verbosity,
  invoke_builder (orchestrator), take_action (one action per turn; the enum is narrowed to the frame's legal actions).
  Tool steps per turn 25 (paper 40); calls per game 1100 (paper 3500); action budget 5x the human baseline per level.
- GAME_OVER: the harness resets by itself, archives the attempt, and attaches the death evidence (fatal action,
  pre/post diff, two PNGs) to the next turn with "avoid equivalent actions unless deliberately probing".
- Animation frames: deterministic filter (bbox, max step, border fractions) decides whether keyframes are summarised
  by a separate model call; the actor gets 1-3 sentences of text, never the frames.
- World model contract enforced by code: forward simulation from the level's first frame (not teacher-forced), a
  no-op that the model predicts as motion counts as a mismatch, coverage priced separately (`-1` abstain), outcome
  graded separately with a render bridge on the observed terminal frame, planner target = own `outcome()`, validated
  plan artifact with per-step frame hashes re-checked every turn.
- Builder report format (8 fixed fields: confidence, model, dynamics, outcome, outcome_verified, plan,
  recommended_action, note) and the instruction "Outcome is a first-class inference, not an afterthought... keep
  competing outcome hypotheses and the probe that would separate them".
- `TYCHO_SANDBOX_RUNTIME=host` runs agent code as a plain subprocess; `LLM_BACKEND=openai` + `LLM_BASE_URL` targets any
  Chat Completions server (our vLLM). Its 164 tests pass here with our venv's pins (arc-agi 0.9.9, arcengine 0.9.3).

Ported so far (exp-020 bundle, commit 0608566): the level boundary (consolidation pass + clear), the observed terminal
frame in the level archive, the animation note, the friction line. Not ported: per-level action budget (no baselines
on the hidden set), the tool-schema narrowing (our actions go through act() in code), the builder role.
