Summary: patch ours-06 keeps, for every level, an exact table of what each action did. Keys get one line each and MOUSE gets one line per clicked cell, with uses, uses with no change, game overs, and the object changes with their counts. The table also lists what changed objects overlapped at each game over and objects whose positions repeated with a fixed period. The model reads it with `effects()` in the sandbox, documented in 4 system-prompt lines. Flag OURS_EFFECT_TABLE=1 (=note also adds a one-line pointer to tool results). Replaying all 6,350 actions of exp-073b through the local engine, a second implementation checked 4,819 table lines and 448 notes against the frames: 0 errors, 2.3 ms per action. The dc22 red-bar click shows in the table right after its first use. With the flag off the harness is byte-identical. Nothing has run on a GPU, so its effect on score is unknown.

# Effect table: an exact per-level action-effect table for Franzen's harness (ours-06, 2026-10-08)

Patch: `kaggle/franzen/patches/ours-06-effect-table.patch` (sha256 `3544ec37a1354fa8518b2170d81d03b07956dec68db8558d4bedc0b28602ecb6`).
It has 3 files:
- a new `inference/utils/ours_effect_table.py` (869 lines, stdlib only);
- 3 call sites of 3 lines each in `inference/agent/tool_agent.py`;
- a 20-line helper and 2 lines in `inference/agent/python_tool_sandbox.py`.

It applies with `git apply` after his patch and `ours-sandbox-timeout-keeps-work.patch` (`scripts/franzen_tree.py check
--patch .../ours-sandbox-timeout-keeps-work.patch --patch .../ours-06-effect-table.patch`: "ok: 2 patch(es) apply on
top of his"). It also applies after the lead's bundle order 01, 02, 04, 03b with line offsets only ("ok: 5 patch(es)
apply on top of his").

Everything below ran on the CPU in this session. Nothing ran on Kaggle or a GPU.

## 1. What it does

`OURS_EFFECT_TABLE` is read at call time:
- unset (or anything else): off, and the harness is byte-identical (section 4);
- `1`: the sandbox gets `effects(level=None)`, documented in 4 system-prompt lines;
- `note`: the same, plus at most one line in a python tool result (below).

`effects()` returns the current level's table; `effects(k)` returns level k's. The return value is a `str` whose repr
is the text, so a bare `effects()` at the end of a snippet prints as plain text. Functions that call it are retained
like any other (`ARC3_PERSISTENT_FUNCTIONS`). Reading it spends no action.

Real example: dc22 right after action 10, the first use of the click that turns the red bar. In exp-073b the model
found this out 168 actions later, and in the meantime it credited LEFT with the effect.

```
Level 1: 10 actions. Board = frame without the 4-cell edge band; from the frames before and after each action; positions are (row,col) top-left corners.
UP x2: no change x1; moved N 4px (-2,+0) x1 [step 3: (40,8)->(38,8)]
LEFT x2: no change x1; moved N 4px (+0,-2) x1 [step 1: (40,10)->(40,8)]
RIGHT x2: moved N 4px (+0,+2) x2 [last step 7: (38,10)->(38,12)]
MOUSE(row=19, col=48) x1 (on R 47px at (17,42)): rotated R 24px 90 (-6,+6) x1 [step 10: (30,12)->(24,18)]; disappeared 2x W 40px x1 [step 10]
MOUSE with no change: 3 uses at 3 cells (row,col), latest first: (42,12) (38,12) (21,24)
Edge band changed on 6 of 10 actions.
```

Two more real lines, from the replayed exp-073b runs:
- tu93 level 2, the game over that the analysis traced to a patroller:

  ```
  Game overs: after RIGHT at step 23: ambiguous W 9px objects 1->2; moved R 8px (+0,-6) (27,36)->(27,30) onto W:8; disappeared b 8px at (27,24), its cells now W:8; disappeared c 1px at (28,26), its cells now W:1; disappeared p 1px at (28,36), its cells now W:1; appeared Y 1px at (28,30) onto W:1, edge band changed
  ```

- ls20 level 6, the moving pluses that the model timed by hand ("Both bounce with period 8"):

  ```
  Objects whose places and turns repeated, period in actions: ... | R 2px every 8 (steps 535-559), last 8 corners: (32,22) (27,22) (22,22) (22,27) (22,32) (27,32) (32,32) (32,27) | ...
  ```

### What every line states

Everything is computed from the frame before and the frame after each action. The module docstring is the
specification; `scripts/effect_table_replay.py` implements it a second time.

- **The board.** The board is the frame without the cells within `ARC3_NOOP_GUARD_BORDER` (4) of an edge, where timer
  and step bars sit.
  - "no change" means no board cell changed. That is the harness's own `gameplay_changed=False`.
  - How often the edge band changed is counted once per level, so a ticking bar never enters an action's effects.
  - Automatic RESETs after game overs are counted in the header but are not actions here.
- **Objects.** Objects are maximal 4-connected one-colour cell sets on the board, as in `.segmentation`.
  - Positions are bounding-box top-left corners, as in `frame_diff()`.
  - Objects of 512 cells or more are areas and are never listed.
- **Pairing.** The objects present in only one of the two frames are paired by these rules, in order:
  1. Same cells, new colour: **recolored**.
  2. The only one of its colour and shape (up to rotation) on each side: **moved** by the corner's offset, or
     **rotated** by 90/180/270 degrees clockwise (with that offset).
  3. n of one colour and shape on each side whose places all differ by one offset: **moved n times**.
  4. Otherwise, objects of that colour and shape on both sides: **ambiguous**. Both position lists are given when there
     are at most 6 objects.
  5. Of the rest:
     - same-colour objects that share cells are **reshaped** (sizes before and after). This is listed only if the
       cluster contains no area and holds a changed cell that no other change covers, so the floor a player walks
       across is not listed.
     - objects none of whose cells keep their colour **disappeared**; objects none of whose cells had it
       **appeared**.
  6. Changed cells covered by none of these are counted by colour transition (`cells c->B`).
  7. More than 12 distinct changes in one action become one **many changes** entry with counts per kind, for example
     a scrolled view or lp85's ring of 22 recoloured tiles.
- **Per action.** Each action line gives:
  - uses, uses with no change, game overs, level completions;
  - up to 4 distinct changes, most frequent first, each with the number of uses it occurred in and the step and
    positions of the last one, then `+k other changes`.
  - A MOUSE cell also names the object under the cursor ("on", or "last on" if that varied).
  - Cells that never changed anything share one line, latest first (24 cells at most, then `+k more cells`).
  - The table is capped at 40 lines.
- **Game overs.** The last 3 game overs on the level, with each change of the fatal action. For a single changed
  object the line also gives what it overlapped: the colours its cells held before (moved, rotated, appeared) or hold
  after (disappeared). The line also says whether the edge band changed.
- **Periodic objects.** Objects of one colour, size and bounding-box shape, at most 4 of them, whose places and poses
  repeated with a fixed period.
  - The period must hold over at least two periods of one attempt. The stretch ends at the attempt's last frame.
  - The object must have moved differently on two uses of one action. This leaves out an avatar walked back and forth.
  - For a single object, its last p corners are listed.

### The note (OURS_EFFECT_TABLE=note)

The note is one line under `effects_note` in a python tool result. It fires when an action of that snippet, already
used on its level, made a kind of change (moved, rotated, recolored, appeared or disappeared, on a colour) that has
not been seen before on that level. Example from the bed:

```
UP at step 6 (used 2x before on this level): disappeared W 3px at (31,21); disappeared 2x w 1px, a kind of change not seen before on level 1. effects() has the table.
```

It is a separate value because of its rate: per action, it would fire on 448 of the 6,350 replayed actions (7%). Two
looser and tighter variants measured 7-11%. Since it carries a cost, it should be a separate arm.

### System-prompt lines (flag on only)

```
- `effects(level=None)` returns the action-effect table of the current level (or of `level`), computed by the harness from the frames before and after each action.
- One line per action (a key, or a MOUSE cell): its uses, the uses with no change, and each object change (moved, rotated, recolored, appeared, disappeared, reshaped) with the number of uses it occurred in.
- It leaves out cells within 4 of the edge, and lists changes it cannot pair object to object as ambiguous.
- It also lists game overs with the colours the changed objects overlapped, and objects whose positions repeated with a fixed period.
```

They follow his Python-tool guidance and come before the `frame_diff()` lines. They describe only; there is no advice
to call it.

## 2. Why (the evidence)

The evidence is pattern P2 of `exp073b-failure-analysis.md`:
- On the 17 games exp-073b did not win, 69 of 331 retained tool calls (21%) loop over `history` or `transitions` to
  recover what actions did, against 9% on the won games.
- dc22 sent the red-bar click at action 10 and identified it at minute 99.7.
- ka59 scanned transitions for moves that did not match the key.
- tu93, su15 and ls20 rebuilt patroller periods by hand, and tu93's phase bugs came from exactly that.
- bp35 asked what killed the player.

What his harness already gives per action:
- the action echo line;
- `gameplay_changed`, NO-OP verdicts and `action_trace` for batches;
- a diff image per turn;
- the animation summary;
- `frame_diff()` on request.

Nothing aggregates by action over a level, so the model rebuilds that by hand. On the old Duck base, exact
object-change reports were measured and kept: P23/P25, exp-050, hard-game level-1 solves 10/16 → 15/16 over two paired
runs. Lesson 0024 (never state a false fact) is why every rule above is exact and why "ambiguous" exists.

## 3. How it is wired into his harness

- `tool_agent.py`, 3 call sites, each a lazy import plus one call, so the module's import header is untouched:
  - `_build_system_prompt`: adds the 4 lines (an empty string when off).
  - `_run_python_tool`: wraps the snippet's `_serialized_runtime_state` closure. The state sent to the sandbox, at the
    start and after every `action(...)`, then carries `ours_effects` = `{level, tables}`. With the flag off,
    `attach()` returns the closure itself.
  - After `_record_retained_functions`: `note()` adds `effects_note` in note mode.
- `python_tool_sandbox.py`: the bootstrap defines the helper. `_refresh_state` binds `effects` only when the state
  carries `ours_effects`, so with the flag off the name does not exist and `dir()` is unchanged.
- **The ledger.** One `EffectLedger` per ToolAgent (one per game), stored on the agent.
  - It processes each new history entry once: a run-length union-find labels the frame in about a millisecond.
  - It re-renders only the level that changed.
  - It keeps only small per-frame object inventories for the period search, not frames.
  - It restarts if the history is not a continuation.
  - An exception disables it for that game. `effects()` then says "Effect table unavailable: internal error (...)"
    and the game goes on.

## 4. How it was tested

**Replay check** (`scripts/effect_table_replay.py`, CPU, about 1 minute):
- It replays all 25 games of exp-073b (`benchmark.json`: 6,350 actions with click coordinates) through the local
  engine, after checking that every game reproduces its recorded per-level action counts.
- It builds the harness's history from the frames, feeds it to the patched `EffectLedger` one action at a time, and
  checks every line against the frames.
- The checker is a second implementation of the docstring's rules: breadth-first objects on numpy arrays, written from
  the text, sharing no code with the module. It recomputes:
  - every count, change, multiplicity, last step and position;
  - click targets, game-over overlaps, periods and corner lists, edge counts and the no-change line;
  - every note.
- Tables are checked at every 25th action, whenever a level ends, and for all levels at the end.

| Run | Games | Actions | Table lines checked | Notes checked | Errors | Update + render per action |
|---|---:|---:|---:|---:|---:|---|
| exp-073b, from the patch files | 25 | 6,350 | 4,819 | 448 | **0** | mean 2.3 ms, p99 5.6 ms, max 126 ms (an earlier run: 2.4 / 6.7 / 150 ms) |
| the note-mode bed run below (its own benchmark.json) | 3 | 266 | 82 | 3 | **0** | mean 1.3 ms, max 15 ms |

The checker catches wrong lines. A use count, an offset, a click target, a no-change count, a period, a last step, an
edge count or a change count, each altered by one, is flagged in every case on dc22 and ls20 (also a unit test).

**The dc22 question.** Yes: the table after action 10 holds the click's line, shown above. Before any later click it
says the click at (19,48) turned R 24px by 90 degrees and moved it (-6,+6), and that LEFT moved N 4px (+0,-2) and
nothing else.

**Unit tests** (`tests/test_ours_effect_table_patch.py`, 16 tests, about 5 s):
- **Synthetic boards:** moves, no change and edge band; recolor, rotate, appear and disappear; group moves and
  ambiguous pairs; reshape side effects; game-over overlaps with recolor precedence; level completion and automatic
  RESET; periodic movers and the walked-back-and-forth avatar; click targets and the no-change line.
- **Ledger and glue:** incremental equals batch; a rewritten history restarts; flag modes of `attach`/`note`; an
  internal error disables the table without raising; under 0.5 s per call on a 400-action 64×64 history.
- **Recorded frames:** dc22's 10 recorded actions replayed through the engine, with the checker reporting 0 errors and
  catching planted errors.
- **Real harness, in the bed venv,** with the notebook's cell-4 environment and a scripted 64×64 game driven through
  the real `ToolAgent._run_python_tool` and sandbox:
  - flag off: the system prompt and all 5 tool results are byte-identical to the tree without the patch;
  - flag on: the prompt lines, `effects()` in snippets (printed and as a bare expression), `effects(7)`, a retained
    function that calls it, and the note only in note mode.

**Full suite:** `pytest tests/` gives 408 passed, 2 skipped, 2 deselected (the slow bed tests). ruff is clean.

**CPU bed** (`scripts/franzen_bed.py`, ls20/vc33/sb26, 120 s per game). The new optional program `--program effects`
acts once, then calls `effects()` and checks that the first line names the current level.

- **Flag 1** (`--set OURS_EFFECT_TABLE=1 --expect "returns the action-effect table of the current level" --program
  effects`): 15/15 checks.
  - 295 requests; the prompt text was in all 295 system prompts.
  - `effects()` was called 8 times and returned the current level's table 8 times.
  - 52 retained-function reuses, 43 `frame_diff` calls, 39 batch no-op stops, 3 stale-state refusals, 6 UNDOs,
    52 prefix breaks, 3/3 overflows recovered, 0 tracebacks.
- **Flag note:** 14/15 checks; 317 requests; `effects()` 7/7. The failed check is "UNDO executed": sb26, the only game
  offering UNDO, never reached the undo step in 120 s (no `bed:undo` marker in its transcript).
  - The model received two notes, found in `*_requests.jsonl`: ls20 step 6 inside the 13-move solve batch, and sb26
    step 6. These are exactly the first notes the replay predicts for those action sequences.
  - `transcripts/*.txt` show tool results as stdout only, so grep the request logs for `effects_note`.

## 5. Cost

- **System prompt:** 613 characters, about 140 tokens, flag on only. They are static, so they are prefix-cached.
- **Tool output:** only when the model prints the table. Over the 140 level tables of the replay:
  - median 1,599 characters (about 370 tokens);
  - p90 3,044 (about 710);
  - max 8,200 (about 1,900), against his 3,072-token tool-output budget.
- **Note mode:** about 200 characters (about 50 tokens) per note, at most one per tool result.
- **Host CPU:** about 2.3 ms per action, run on every state serialization. The state sent to the sandbox grows by the
  tables, a few kB against a history payload of megabytes.
- **No actions.**

## 6. Known limits

- **The edge band is not board.** An object that lives within 4 cells of an edge is invisible to the table. This is
  the same convention as his `gameplay_changed`.
- **One-colour objects.** A multi-colour sprite is several objects: tu93's player is a b 8px body plus a c 1px eye, and
  ls20's plus is R, b and O parts. A diagonal sprite falls apart into pixels, which become group moves or "ambiguous".
  The table says so rather than guessing.
- **Ambiguity is common where identical objects move differently.** 1,234 of the 6,331 replayed uses (19%) have an
  ambiguous change, for example tu93's identical patrollers, su15's pieces and re86. Position lists are given up to
  6 objects.
- **Pairing follows the rules, not semantics.**
  - An object that lands exactly on a same-shaped object of another colour reads as "recolored", because rule 1 comes
    first (pinned by a test).
  - A tile that swaps places with the player reads as "moved W 9px" next to the player's own move.
- **"many changes".** 171 replayed uses (2.7%) hide their details behind kind counts, for example lp85's tile rings and
  bp35's scrolls.
- **Periods.** A period is reported only for at most 4 objects of a class, over two full periods inside one attempt.
  Only 3 of 140 replayed levels produced a line. tu93's four identical patrollers (periods 8, 4 and 6 by hand) have a
  joint period of 24, longer than its attempts, so nothing is reported there.
- **Final frames only.** The table compares the final frames of each action, so transient animation changes are not
  in it, just as they are not in `history`.
- **Uptake is unknown.** The mock bed shows that the mechanics work, not that the model will call `effects()`. The
  note is the nudge, in its own arm.
- **No Kaggle or GPU run** was made, and the tables were checked on public games only.

## 7. Building an arm

```bash
.venv/bin/python scripts/build_franzen_nb.py --out build/effects --slug arc3-franzen-effect-table \
    --patch kaggle/franzen/patches/ours-sandbox-timeout-keeps-work.patch \
    --patch kaggle/franzen/patches/ours-06-effect-table.patch --env-add OURS_EFFECT_TABLE=1
```

This builds; it was checked in this session. For the note arm, use `--env-add OURS_EFFECT_TABLE=note`. To stack it on
the lead's bundle, add the patches in the order 01, 02, 04, 03b, 06. Add `--full25 121` for a full public-25 arm
(engineering.md).

## 8. Reproduce

```bash
.venv/bin/python -I scripts/effect_table_replay.py runs/exp073b-dprime-reap448-r14-accept05-full/kernel-output/benchmark.json
.venv/bin/python -m pytest tests/test_ours_effect_table_patch.py
.venv/bin/python scripts/franzen_bed.py --games ls20,vc33,sb26 --seconds 120 \
    --patch kaggle/franzen/patches/ours-sandbox-timeout-keeps-work.patch \
    --patch kaggle/franzen/patches/ours-06-effect-table.patch \
    --set OURS_EFFECT_TABLE=1 --expect "returns the action-effect table of the current level" --program effects
```
