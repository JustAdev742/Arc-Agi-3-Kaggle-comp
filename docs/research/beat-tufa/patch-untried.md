Summary: patch ours-10 (flag OURS_UNTRIED) adds two short lines to the next user prompt: what has not been tried on a level that has stalled for 10 minutes (I1: unused action types and up to 6 object kinds no click has hit), and an object that just took a colour new to the game (I2). Replayed with the shipped module over all 25 games of exp-073b, exp-075 and exp-083, I1 fires 98-112 times per run (about 105 tokens each) and I2 12-13 times (about 41 tokens). Between them the lines name all 7 objects that the post-mortem's 7 losing game-runs assumed inert, and also the two that exp-084, a run outside the post-mortem, lost on. With the flag off the harness is byte-identical. Nothing has run on a GPU, so the effect on score is unknown.

# Untried-object lines I1 and I2 for Franzen's harness (ours-10, 2026-10-10)

Patch: `kaggle/franzen/patches/ours-10-untried.patch` (sha256
`a3a7d89fbb2cb253afcfc8eb9b4d43f763c45ea8b8624139db54949e7bc6a52e`, 483 lines, 22,845 bytes).

| File | Added | Removed |
|---|---:|---:|
| `inference/utils/ours_untried.py` (new, stdlib only) | 453 | 0 |
| `inference/agent/tool_agent.py` (2 call sites, 3 lines each) | 6 | 0 |

**One file for both stacks.** `scripts/franzen_tree.py check` reports:
- the patch alone: "ok: 1 patch(es) apply on top of his";
- on exp-083's stack (his patch, then `ours-sandbox-timeout-keeps-work.patch`): "ok: 2 patch(es) apply on top of his";
- on exp-084's stack (sandbox, 02, 04, 03b, 05, 08b, in that order): "ok: 7 patch(es) apply on top of his", with line
  offsets only.

The hooks land in the same place on both stacks (a test checks this), so no `ours-10b` file was made.
`build_franzen_nb.py --base dprime` builds an arm with it and `--env-add OURS_UNTRIED=1` (local build only, nothing
pushed).

Everything below ran on the CPU in this session. Nothing ran on Kaggle or a GPU, and nothing was committed.

## 1. What it does

`OURS_UNTRIED` is read at every call:
- unset, or any other value: off, and the harness is byte-identical (section 4);
- `1`, `on`, `true` or `yes`: I1 and I2;
- `i1` or `i2`: that line alone, for separate arms.

Both lines are facts about the level. They give no advice beyond the I1 sentence the post-mortem specified, and they
spend no action.

**I1, "not yet tried on this level".**
- **When it fires.**
  - A level has run 10 minutes of wall clock without a level-up. Parked time counts.
  - Then again every 10 minutes after the last I1 on that level.
  - After each GAME OVER on that level, once it is past the 10-minute mark. The post-mortem's trigger says "it fires
    again ... after each GAME OVER", so a GAME OVER in the first 10 minutes fires nothing and is not remembered.
- **What it names.**
  - The valid action types not used on this level. RESET and UNDO are never listed; the D′ notebook advertises UNDO.
  - When MOUSE is valid, up to 6 object kinds on the current board that no click on this level has hit. Each is given
    as a colour letter, a box and a copy count. The box gives the size: a pixel count was left out to save tokens.
  - Order: kinds missing from the previous level's first board come first, then the fewest copies, then the larger.
    Kinds beyond the 6 are counted.
- **The sentence.** Every I1 ends "Each is a one-action test of whether it is interactive."
- **Example.** Replayed exp-083, sp80 level 2 at 10.0 minutes. That run never clicked the red bars (they are in all 9
  of its I1 lines on the level) and never solved it:

```
Not yet tried on this level: DOWN, MOUSE never used; no click has hit R r16-19 c8-19 x2, b r24-27 c4-23, M r56-59 c40-43, c r60-62 c40-43, Y r4-11 c12-23 x3. Each is a one-action test of whether it is interactive.
```

**I2, "new colour".**
- **When it fires.** After an action, an object other than the one that action clicked shows a colour seen nowhere
  earlier in this game.
- **What is excluded.**
  - Level switches: their colours count as seen.
  - Game-over frames, and RESETs.
  - An object clicked between the event and the next prompt.
- **What it names.** The object's old colour (the colour that held 60% of its cells before; it reads "appeared" when
  that colour was the background), its box and its new colour. It names at most 3 objects, then "(and n more)".
- **Example.** Replayed exp-083, vc33 level 4 at 5.3 minutes. This is the gate turning orange, which that run never
  clicked:

```
After MOUSE(row=61, col=15), w r43-54 c12-14 turned O, a colour new to this game; no click has hit it since.
```

**Where the lines go.** They go in the next user prompt, after the valid-actions line (and after the death-ledger
lines when `ARC3_DEATH_LEDGER` is on), just before "Only tool: `python`.". From the bed:

```
Current state: step 4, level 2.
Valid actions right now: UP, DOWN, LEFT, RIGHT, MOUSE, UNDO.
Not yet tried on this level: UP, DOWN, LEFT, RIGHT, MOUSE never used; no click has hit Y r12-14 c12-14, b r2-3 c2-3, R r5-8 c20. Each is a one-action test of whether it is interactive.
Only tool: `python`. It receives `current_frame`, ...
```

**Definitions.** The module docstring is the specification.
- **Objects.** An object is a 4-connected one-colour component of the frame. Areas are dropped: more than 300 cells, or
  at least 75% of the rows or columns. Bars at most 2 cells thick lying within 2 cells of an edge are dropped as HUD.
- **Kinds.** A kind is a colour plus a shape up to rotation, so copies collapse into one entry.
- **Clicks.** A click hits the object under its cell, or else the smallest object whose box, grown by one cell,
  contains the cell. The kind of the object on those cells right after the click also counts as hit, because a
  selection often recolours the object.
- **Clock.**
  - A level starts at the wall time at which the `action()` call that reached it returned. The first level starts at
    the game's first prompt.
  - Every `action()` call is stamped with its last action number when it returns.

## 2. Why

`docs/postmortems/bimodal-games-2026-10-10.md` (accepted) found the following in tn36, vc33, sp80 and cn04:
- 7 of the 8 losing game-runs never clicked, or clicked 48 minutes late, an object the model had assumed inert.
- Every run that won those levels had clicked it first.
- Three losing runs had written the probe down and put it off.

The same mode is the top stuck mode in `levels2plus-exp054.md` (15 of 48) and `exp073b-failure-analysis.md` (8 of 17).
I1 and I2 are that post-mortem's two interventions, built as specified.

## 3. Wiring

- **`tool_agent.py`, 2 call sites, each a lazy import and one call:**
  - In `_handle_action`, right after `raw_payload = self._step_env_callback(...)` and its type check: `stamp(self,
    raw_payload)` records the call's return time under its `action_num`. Refused calls are not stamped.
  - In `_build_user_prompt`, after the death-ledger block: `lines.extend(prompt_lines(self, history_entries,
    _normalize_valid_actions(valid_actions)))`.
- **The tracker.** One `Tracker` per game, kept on the agent and keyed by `_session_runtime_dir`, the per-game key
  that `_ensure_session` already resets on.
  - It processes each history entry once.
  - It computes components only for clicks, for frames with a colour new to the game, and when I1 fires.
- **Off.** Both functions return before touching the agent, so no attribute is set.
- **Errors.** An exception in `prompt_lines` logs a warning and turns the lines off for that game. A failed `stamp`
  logs and is otherwise ignored. Neither can stop a game.
- **Host time.** Over exp-073b's 3,405 reconstructed prompts: mean 2.4 ms per prompt, p99 47 ms, max 167 ms (lp85, a
  long click batch); 8.2 s for the whole 25-game run.

## 4. How it was tested

**`tests/test_ours_untried_patch.py`: 33 tests, all passed (about 14 s).** The module is loaded from the patch text.
- **The flag** (11 cases).
- **Objects:**
  - kinds, with rotation collapse (an L and its 90° turn are one kind; its mirror image is not);
  - ragged and empty grids;
  - HUD, panel and area exclusion;
  - click targets.
- **I1:**
  - the ordering: new kinds first, then fewest copies, then larger;
  - the cap of 6 and the "2 more kinds" tail;
  - the empty and actions-only texts;
  - RESET and UNDO never listed;
  - kinds listed only when MOUSE is valid.
- **I2:**
  - fires: a recolour, an object appearing on background, 4 objects;
  - silent: the click target, a level switch (whose colours then count as seen), a game-over frame, a colour already
    on the board, an object clicked before the prompt;
  - the i1-only mode drops it.
- **Timing, with explicit clock values:**
  - nothing at 599 s, I1 at 600 s, again at 1,200 s;
  - a GAME OVER past the mark fires at once, 100 s after the last I1;
  - a GAME OVER before the mark fires nothing and is forgotten;
  - a level's start comes from the stamp (t=50) and not from the prompt that first saw it (t=649);
  - stamps keep only increasing action numbers.
- **Glue:**
  - flag off: no state on the agent;
  - one tracker per session directory;
  - an internal error turns the lines off for the game and logs.
- **The patch** (alone, on 083's stack, on 084's):
  - tool_agent.py loses no line and gains exactly the 6 patch lines;
  - the shipped module equals the patch text;
  - the prompt hook sits between the death-ledger call and the tool rules;
  - the stamp sits right after the step callback.
- **The real ToolAgent in the bed venv,** with the notebook's cell-4 environment, on both stacks.
  - Setup: a 24×24 fake game, driven as `analyze()` drives it (`_ensure_session`, then `_build_user_prompt` from the
    runtime-state file), with 4 real `_run_python_tool` calls through the sandbox (a click, RIGHT, a level-up DOWN, a
    fatal LEFT plus the solver's automatic RESET) and a fake clock.
  - Flag unset or `0`: the system prompt, all 6 user prompts and all 4 tool results are byte-identical to the tree
    without the patch. `OURS_UNTRIED=1` on the tree without the patch changes nothing.
  - Flag `1`: each prompt equals the base prompt with exactly the expected lines inserted before "Only tool:":
    - I2 after the click;
    - I1 at 600 s on level 1;
    - nothing at 595 s on level 2, and I1 at 600 s;
    - I1 "game over" after the fatal LEFT;
    - the tool results are unchanged.
  - Level 2's clock starts at the DOWN call's return (t=1,610), not at the next prompt (t=2,205), which shows that the
    stamp path works.
  - `i1` and `i2` each give only their own lines.

**ruff** (`.venv/bin/ruff check`): clean on the module, the test file and `scripts/untried_replay.py`.

## 5. CPU replay with the shipped module

**How it works.** `scripts/untried_replay.py`:
- replays each recorded run's actions through the local engine and rebuilds the harness's history;
- drives the patch's own `Tracker` as tool_agent.py does: `stamp()` when an action call returns, `lines()` at every
  user prompt;
- loads the module from the patch text.

Runs keep no prompt times, so they are rebuilt from the recorded clock and tokens. An action with tokens opens a call,
and one prompt is placed after each call plus one per 2,048 generated tokens at 55 tok/s. Lines can therefore appear up
to one turn earlier than in a real run.

The run took about 1 minute per 4 runs. Tokens were counted with `scratchpad/frspec/tokenizer.json`, which has the
served model family's 248,044-entry vocabulary.

| Run | Levels reached / solved | I1 firings (on solved / unsolved levels) | Unsolved levels with an I1 | I1 stall / repeat / game over | I1 items, median | I1 tokens, median / p90 | I2 firings (solved / unsolved) | Unsolved levels with an I2 | I2 tokens, median / p90 |
|---|---|---|---|---|---:|---|---|---|---|
| exp-073b | 141 / 124 | 98 (67 / 31) | 13 of 17 | 52 / 39 / 7 | 6 | 106.5 / 117 | 13 (12 / 1) | 1 of 17 | 42 / 58 |
| exp-075 | 124 / 104 | 112 (44 / 68) | 16 of 20 | 46 / 53 / 13 | 6 | 107 / 115 | 12 (7 / 5) | 4 of 20 | 40.5 / 53 |
| exp-083 | 134 / 116 | 108 (56 / 52) | 13 of 18 | 45 / 47 / 16 | 6 | 101 / 115 | 12 (11 / 1) | 1 of 18 | 41 / 58 |
| exp-073 (for the critical check) | 126 / 108 | 107 (53 / 54) | 11 of 18 | 48 / 46 / 13 | 6 | 105 / 114 | 12 (8 / 4) | 4 of 18 | 42 / 58 |
| exp-084 (extra; not in the post-mortem) | 133 / 114 | 106 (63 / 43) | 13 of 19 | 51 / 43 / 12 | 6 | 106 / 114 | 12 (10 / 2) | 2 of 19 | 40.5 / 54 |

- **Where I1 fires.**
  - It fires on 45-53 levels per run, and fires at least once on 22-23 of the 25 games.
  - Per level it fires a median of 1-2 times, and at most 6-10 times (exp-083: bp35 L2 10, sp80 L2 9, vc33 L4 8).
- **What I1 holds.** Of the I1 lines (exp-073b, exp-075, exp-083):
  - 63-64 list kinds only;
  - 25-40 list both kinds and actions;
  - 8-14 list actions only (keyboard games, for example "LEFT never used").
- **I2.** Each run fires the same few events: m0r0 L3, bp35 L1, cn04 L2 (2), tu93 L7, wa30 L1-L2, vc33 L4, lf52 L1-L2,
  su15 L4-L5 and g50t L1. They land mostly on solved levels; 24 of the 37 come within the level's first 6 minutes.

**The post-mortem's critical cases.** All 7 reproduce, and exp-084 adds two. Times are minutes into the level. The
post-mortem's "28.8" and "17.4" for vc33 were game-clock minutes; these are the same events.

| Losing game-run | Object assumed inert | I1 names it | I2 names it | Post-mortem |
|---|---|---|---|---|
| tn36 exp-073 L2 | the demo's two boxes | 10.3 (b r54-62 ring ranked 1st) | – | yes, ring 1st |
| vc33 exp-073 L4 | orange gate | no (9 firings) | 18.0 | no / yes |
| vc33 exp-083 L4 | orange gate | no (8 firings) | 5.3 | no / yes |
| sp80 exp-073 L2 | red bars | 10.3, plus "MOUSE never used" | – | yes |
| sp80 exp-075 L2 | red bars | 10.4, plus "MOUSE never used" | – | yes |
| sp80 exp-083 L2 | red bars | 10.0, plus "MOUSE never used" | – | yes |
| cn04 exp-075 L2 | N and b "sockets" | 10.3 (ranked 2nd and 4th) | – | yes, 2nd and 4th |
| vc33 exp-084 L4 (lost, 76 min) | orange gate | – | 9.3 | (not analysed) |
| sp80 exp-084 L2 (lost, 70 min) | red bars | 10.6, then 6 more times, with MOUSE never used on the level | – | (not analysed) |

## 6. What a GPU arm should read for each firing

Run one arm: an LB candidate's build (exp-083 or exp-084) plus this patch last plus `OURS_UNTRIED=1`, against the
same build without it. Score alone cannot show this: a gain of about 1 level is inside the noise (lesson 0018). Read
the mechanism per firing.

**Getting the firings.** Take the actual firings from the arm's prompt logs:
- the user messages in `*_requests.jsonl` or the transcripts containing "Not yet tried on this level:" or "a colour
  new to this game";
- the step from that prompt's "Current state: step N" line.

Then read the next actions and the level-ups from `benchmark.json`. `untried_replay.py` on the arm's `benchmark.json`
recomputes the firings with the same code, but its prompt times are reconstructed.

**The three readings, per firing.** The baselines come from the recorded runs, which had no lines. Pooled over
exp-073b, exp-075 and exp-083 (318 I1, 37 I2):

| Reading | I1 baseline | I2 baseline |
|---|---|---|
| A named object clicked within 3 actions | 32 / 318 (10%) | 5 / 37 (14%) |
| ...and that click changed the board (the frame without its 4-cell edge band) | 23 (72% of those clicks) | 5 (all) |
| A named unused action used within 3 actions | 41 / 318 (13%) | – |
| A level-up within 15 minutes | 114 / 318 (36%) | 20 / 37 (54%) |

Per run, I1 clicked within 3 / changed / action used / level-up within 15 minutes:

| Run | Clicked within 3 | Changed | Action used | Level-up |
|---|---|---|---|---|
| exp-073b | 9 | 8 | 14 | 46 |
| exp-075 | 12 | 9 | 11 | 34 |
| exp-083 | 11 | 6 | 16 | 34 |
| exp-073 | 14 | 9 | 13 | 43 |
| exp-084 | 9 | 7 | 17 | 46 |

Per run, I2 clicked within 3 / changed / level-up within 15 minutes: exp-073b 4 / 4 / 9 of 13, exp-075 1 / 1 / 6,
exp-083 0 / 0 / 5, exp-073 2 / 1 / 5, exp-084 2 / 2 / 8 of 12.

- **Uptake.** If the model acts on the lines, the clicked-within-3 rate should rise well above 10-14%. If it does not,
  the lines are ignored and the arm says nothing about the idea.
- **Informative probes.** The share of those clicks that changed the board shows whether the probes were informative.
- **Level-ups.** Level-ups within 15 minutes matter most on the critical levels: vc33 L4, sp80 L2, cn04 L2 and tn36
  L2, in every run that reaches them.
- **Cost on solved levels.** Also read the extra actions on levels that were solved anyway.
- **Seconds per action.** It should be unchanged: host time is milliseconds per prompt.

## 7. Risks and limits

- **I1 is longer than the target.**
  - The median is 101-107 tokens (p90 114-117), against "~80 typical". That is 6 kinds at about 12 tokens each,
    because digits tokenize one per token, plus the fixed text.
  - At 4 kinds it would be about 80.
  - Per run the cost is about 110 × 105 ≈ 11.5k prompt tokens in all, which is negligible. Kept at 6 as specified;
    the cap is the `MAX_KINDS` constant.
- **Probes on levels that were solved anyway.** 44-67 I1 firings per run fall on levels the run solved. If half draw
  one probe, that is about 25-35 actions per run, at about 2-4% of a 50-100-action level's score each (post-mortem
  estimate).
- **Noise in the lists.** The object filter is crude, so walls and water columns get listed (vc33 "B r31-63 c42-44"),
  as do large regions (r11l "S r32-54 c1-28") and decorations (bp35 "G r4 c7 x110").
- **Repeats.** The same block repeats every 10 minutes on a long stall (up to 10 times on one level, exp-083 bp35 L2).
- **Keyboard games.** These get actions-only lines ("LEFT never used"). That matches the spec, and each costs at most
  one action.
- **The edge rule hides compact edge objects.**
  - Thin things within 2 cells of an edge are treated as HUD.
  - A scan of exp-073b's frames found, besides border pixels:
    - 2×2 squares at the edge in su15 (S, M, p, Y, one of each per frame);
    - sc25 (p, Y);
    - ls20 (R about 2 per frame, and G).
  - If any of these is a control, I1 never lists it. I kept the rule because the alternative lists border dots and
    HUD squares.
- **I2 noise.** cn04 fires 2 per run (grey tips, the purple deselected object), and wa30 L1-L2 and lf52 fire too.
  These come early, mostly on solved levels; each costs at most one click.
- **The GAME OVER reading.** A GAME OVER fires I1 only past the 10-minute mark (section 1). If "after each GAME OVER"
  was meant at any time, change one condition in `Tracker.lines`.
- **Replay times are approximate.** Prompt times are reconstructed. The real first-level start also includes the time
  before the first prompt, so a real run can fire a little later on level 1.
- **No system-prompt line.** The model is not told in advance what these lines are; they are meant to read on their
  own.
- **Uptake is unknown.** Nothing here shows the model will act on the lines. The floor is 0 (post-mortem).

## 8. Reproduce

```bash
P=kaggle/franzen/patches
.venv/bin/python scripts/franzen_tree.py check --patch $P/ours-sandbox-timeout-keeps-work.patch --patch $P/ours-10-untried.patch
.venv/bin/python -m pytest tests/test_ours_untried_patch.py
.venv/bin/python -I scripts/untried_replay.py runs/exp073b-dprime-reap448-r14-accept05-full \
    runs/exp075-dprime-r14a05-sandbox-full runs/exp083-dprime-r14a05-arcmap-draft-full \
    runs/exp073-dprime-reap448-r14-full --tokenizer <tokenizer.json> --json replay.json
```

Building an arm means adding `--patch $P/ours-10-untried.patch --env-add OURS_UNTRIED=1` to the candidate's
`build_franzen_nb.py` command, with the patch last. Use `OURS_UNTRIED=i1` or `i2` for single-line arms.

## GPU results (2026-10-10)

Two full-length runs of exp-085 (exp-084 + this patch, OURS_UNTRIED=1), read with `scripts/untried_read.py` (the lines
as the model saw them; research log 14:31 and 16:59):

| Run | Score / levels | I1 firings | Named object clicked within 3 actions | I2 firings / clicks | Critical levels won |
|---|---|---|---|---|---|
| v1 | 53.77 / 119 | 101 | 20 (19.8%) | 12 / 1 | 4 of 4 |
| v2 (slow boot) | 47.51 / 111 | 99 | 23 (23.2%) | 12 / 1 | 3 of 4 |
| pooled | | 200 | 43 (21.5%; without the lines 32/318 = 10.1%) | 24 / 2 | 7 of 8 |

The model acts on I1 about twice as often as it clicks those objects without it, and I2 draws almost no clicks. Clearest
case: sp80 v1, step 43, the line named the red bar "R r16-19 c8-19" and the next action clicked it; the L3 and L4 lines
drew red-bar clicks followed by level-ups. exp-085 replaced exp-084 in the LB rotation. A later build should keep I1
only (OURS_UNTRIED=i1).
