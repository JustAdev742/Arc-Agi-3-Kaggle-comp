Summary: OURS_PERCEPTION=1 (patch ours-08) hands the model four exact helpers, computed by code from the frames the harness already holds: a whole-view shift estimate with a running `view_offset` (and a `[view]` line when the view scrolls), `left_view` for objects that left the view, `logical_grid()` on the frame's cell lattice, and `frame.segmentation8`. Replayed against the engine's own camera on every recorded run (39 runs x 25 games, 107,083 frames), all 285 scrolls were measured exactly, no scroll was claimed on 103,477 still steps, and no offset disagreed with the engine (188 frames, 0.18%, were left unknown). With the flag off the harness is byte-identical. Everything was checked on the CPU only, so its effect on score is unknown.

# Patch ours-08: exact perception helpers (2026-10-08)

Patch: `kaggle/franzen/patches/ours-08-perception.patch` (sha256
`f202d54b50637fcad86640ed02d60fa29c3092412b453284bc94c19c9a42a5c1`, 971 lines, 50,641 bytes). Files:

| File | Added | Removed |
|---|---:|---:|
| `inference/utils/ours_perception.py` (new, host side) | 607 | 0 |
| `inference/utils/ours_perception_sandbox.py` (new, spliced into the sandbox bootstrap) | 174 | 0 |
| `inference/agent/python_tool_sandbox.py` | 31 | 1 |
| `inference/agent/tool_agent.py` | 13 | 0 |
| `inference/utils/segmentation.py` | 10 | 2 |

It applies after his patch and ours-01, 02, 04, 03b, 05, 06b, 07, in that order, with no offsets.
`scripts/franzen_tree.py check` with all eight reported "ok: 8 patch(es) apply on top of his". Flag
`OURS_PERCEPTION=1` (also `on`, `true`, `yes`), read at call time, default off.

**Variant for exp-080's stack.** ours-08 does not apply without 06b, because two of its hooks sit next to 06b's lines.
exp-080's set (01, 02, 04, 03b, 05; 06b and 07 dropped) therefore gets
`kaggle/franzen/patches/ours-08b-perception-on-01-02-04-03b-05.patch` (sha256
`5745781e03d3967da1c12f6f7643969b723a5e6024e197d8fa98bbf41ca22c78`). Every file gains and loses exactly the same
lines as in ours-08; only the hook positions differ, and a test asserts this. `check` on that stack reported "ok: 6
patch(es) apply on top of his".

Everything below was run on the CPU in this session. Nothing ran on Kaggle or a GPU, and nothing was committed.

## 1. What it adds

All four are computed from frames, with no model calls and no actions.

**1. View shift and `view_offset`.** After each action, the harness compares the frame before and the frame after
(same level, not a RESET) and decides whether the whole view moved. `view_offset` is (dr, dc) with world (row, col) =
frame (row, col) + `view_offset`. It is (0, 0) at a level's first frame and after every RESET, and None when the
harness could not tell. Every history frame and the current frame carry `.view_offset`. The action's echo gets a line
when the view moved during that `action()` call, for example (lf52, exp-073b's step 155):

```
[view] scrolled at step 155: content moved (+0,-8); view_offset now (+0,+8)
```

A call with several scrolls lists up to 3 (`scrolled 3 times (steps 1, 2, 6): content moved (-6,+0), (+0,+8),
(+0,-6); view_offset now (+0,+6)`). In a game that has already scrolled, a step whose evidence is too weak to claim
and too strong to ignore makes the offset unknown, and the echo says `[view] the frame changed at step k in a way
that may be a scroll; view_offset is unknown until the next RESET or level`. Before a game's first scroll that case
is silent (the offset becomes None without a line).

**2. `left_view`.** When a step leaves no cell of some colour in view, an object of that colour in the frame before
is recorded if:
- the claimed shift carried all of it into the 4-cell edge band or past the edge; or
- it touched an edge it was moving toward. One step earlier an object of its colour overlapped it across the move,
  lay further from that edge and was at least as long along the move.

An object that vanishes in place (a blinking marker), stays in place in a new colour, or lies inside the edge band
(HUD) is not recorded. The record is kept while its colour is out of view on that level. Example (lf52, exp-073b's
step 204, the cart that P3 says was hunted with single actions):

```
[view] left the view: Y 14px at the right edge (last seen step 203, rows 23-28, cols 59-63); see left_view
left_view == [{'colour': 'Y', 'pixels': 14, 'step': 203, 'box': [23, 59, 28, 63], 'world_box': [23, 59, 28, 63], 'edge': 'right'}]
```

**3. `logical_grid(frame=None, cell=None, origin=None)`.** It finds the frame's square cell lattice and returns one
character per cell: the cell's colour, or `*` for a cell holding several (`.mixed` lists their counts). With
`cell=(h, w), origin=(row, col)` the lattice is set by hand. It returns None when no lattice is found. The object has
`.rows`, `.cell`, `.origin`, `.shape`, `.mixed`, `.box(i, j)` and `.cell_of(row, col)`. ls20, level 1:

```
cells 5x5 px from (0,4): 12 rows x 12 cols; '*' = mixed (15, see .mixed)
  0 cccccccccccc
  1 ccccc***cccc
  ...
  5 ccGGGGGGGGcc
```

**4. `frame.segmentation8`.** His `segmentation`, but with 8-connected objects, so cells touching at a corner join.
His `segment_layer` gained `connectivity=4|8`; the default path is unchanged (section 4.7).

**Prompt cost.** Four lines (853 characters) after the effect-table lines. With the flag off, they are absent.

## 2. Why

P3 of `exp073b-failure-analysis.md` lists five games that lost calls this way:
- bp35 confused positions across camera moves;
- lf52's cart left the view and was hunted with single actions (about 8 calls);
- ls20 read its maze lattice by hand (about 5 calls; its first action on level 6 came 6.9 minutes in);
- cn04 misread its block grid ("the crop was offset");
- su15's diagonal sprite fell apart into 1-pixel objects under 4-connectivity.

The estimated gain is +0.5 to +1.5 levels. That estimate is from the analysis and was not measured here.

## 3. How the shift is decided

The rules are in the module docstring. In short:
- **Candidates.**
  - Pure vertical shifts under which the most rows of A (their part outside the edge band) reappear unchanged in B:
    at least 3 rows, and at least half of the overlapping rows that are not one colour.
  - The same for columns.
  - Up to 4 shifts voted for by 4x4 blocks of A that occur at most 3 times in B.
- **Evidence for a shift s.** It is taken only from cells p where A(p) != A(p+s); those are the only cells where "the
  view moved" and "nothing moved" predict different things.
  - **support:** B(p+s) == A(p).
  - **contradict:** B(p+s) == A(p+s).
  - **ratio** = support / (support + contradict).
  - **explained** = support / every cell that changed or should have changed. A sprite's new cells count against s.
  - Each is counted over the whole overlap and again without the 4-cell edge band.

| Rule | Whole overlap: ratio / explained / span | Without the band: ratio / explained / span | Also |
|---|---|---|---|
| CLAIM | 0.8 / 0.75 / 0.5 | 0.9 / 0.75 / 0.25, only if >= 16 band cells contradict s (a status bar that stays put) | >= 64 supporting cells over >= 8 rows and >= 8 columns; >= 32 changed cells |
| CLAIM_AFTER_SCROLL (once the game has scrolled under CLAIM) | 0.7 / 0.65 / 0.5 | 0.8 / 0.65 / 0.25, same band condition | the same |
| UNSURE: offset becomes None, silently | 0.75 / 0.65 / 0.5 | 0.8 / 0.65 / 0.25 | >= 32 supporting cells over >= 8 rows and >= 8 columns |
| UNSURE_AFTER_SCROLL: the same, with the line above | 0.6 / 0.5 / 0.5 | 0.6 / 0.5 / 0.25 | the same |

"Span" is the smaller of the row and column extents of the supporting cells, relative to the overlap.

**How the rules got here.** These are measurements in this session on the recorded runs (section 4.2), not
predictions:

| Version | Scrolls | Exact | Unsure | Missed | False claims | Offset disagreements |
|---|---:|---:|---:|---:|---:|---:|
| Block-voted candidates only, no per-game prior | 271 | 250 | 5 | 16 | 0 | 195 |
| + row/column votes, + the per-game prior and the unsure rules | 271 | 271 | 0 | 0 | 24 | 372 |
| + explained share, + 8-cell extent, + band condition (final) | 285 | 285 | 0 | 0 | 0 | 0 |

- **Block-voted candidates only.** 13 of the 16 misses were bp35's 36-row scrolls and 3 were lf52's 6-column
  scrolls. Every miss left a silently wrong offset.
- **+ row/column votes, + the per-game prior.** All 24 false claims were in ar25: shifts of 42 to 60 cells over small
  overlaps, where a periodic background mapped onto a moved object's vacated cells.
- **Final.** The explained share and the 8-cell extent removed the ar25 claims. The band condition stops a lone large
  sprite on an empty floor from being claimed. The runs of exp-078 and exp-079 reached `runs/` before the last
  measurement, hence 285 scrolls. On the earlier 37 runs the final rules also gave 271 of 271 exact and 0 false claims.

**Caveat: there is no holdout.** The thresholds were tuned on these same runs. The random walks (4.3) reach other
states of the same two games. The hidden games may scroll in ways these 25 do not.

## 4. Validation

The engine oracle is the engine's own scroll state:
- bp35: the scene camera (`frame = world - camera`);
- lf52: the world node (`frame = world + node`);
- every other game: the engine camera, which none of them moves.

The tracker is fed one action at a time, as the harness serializes state.

### 4.1 exp-073b's bp35 and lf52, at every step

| Game | Frames | Engine scrolls | Claimed exactly | Wrong / missed / unsure | False claims | Offsets agreeing |
|---|---:|---:|---:|---:|---:|---:|
| bp35 | 119 | 27 | 27 | 0 / 0 / 0 | 0 | 119 / 119 |
| lf52 | 215 | 12 | 12 | 0 / 0 / 0 | 0 | 215 / 215 |

This is unit test `test_recorded_scrolls_agree_with_the_engine_at_every_step`.

### 4.2 Every recorded run (39 runs under `runs/`, every game with a history)

| | Frames | Still steps | Engine scrolls | Claimed exactly | Wrong | Missed | False claims | Unsure on still steps | Frames with unknown offset | Offsets disagreeing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bp35 | 1,869 | 1,528 | 237 | 237 | 0 | 0 | 0 | 0 | 0 | 0 |
| lf52 | 3,498 | 3,367 | 48 | 48 | 0 | 0 | 0 | 0 | 0 | 0 |
| ar25 | 5,722 | 5,571 | 0 | - | - | - | 0 | 6 | 129 | 0 |
| cn04 | 3,272 | 3,172 | 0 | - | - | - | 0 | 4 | 59 | 0 |
| the other 21 games | 92,722 | 89,839 | 0 | - | - | - | 0 | 0 | 0 | 0 |
| **all** | **107,083** | **103,477** | **285** | **285** | **0** | **0** | **0** | **10** | **188 (0.18%)** | **0** |

In all 10 unsure steps (ar25 and cn04), the action moved objects by one 3-pixel cell in its own direction, and that
move was read as a possible 3-pixel scroll. Each leaves the offset None until the next RESET or level, silently,
because neither game ever scrolled before.

**Time per tracker update** (one history entry, pure Python, over the 107,083 updates, with no other job of this
session running):
- mean 1.71 ms;
- p50 2.27 ms;
- p99 3.74 ms;
- max 59.7 ms.

The slow test `test_every_recorded_run` reruns this with its own counter. It asserts 0 wrong, 0 missed, 0 false claims
and 0 disagreeing offsets.

### 4.3 Random walks on every level of bp35 and lf52

For each level: `set_level`, a level RESET, then 300 random actions, compared with the engine at every frame. Where
clicks are offered, about 40% of actions are clicks, mostly on rare-coloured cells.

| Game | Levels | Frames | Engine scrolls | Claimed | Offsets agreeing | Unknown |
|---|---:|---:|---:|---:|---:|---:|
| bp35 | 9 | 2,709 | 116 | 116 | 2,709 | 0 |
| lf52 | 10 | 3,010 | 0 | 0 | 3,010 | 0 |

The engine scrolls per bp35 level are 5, 12, 0, 14, 3, 7, 49, 7 and 19. lf52 scrolls only in particular states (a
cart pushed far enough), which random play does not reach.

### 4.4 Synthetic frames (unit tests)

- **Camera moves over a textured 200x200 scene** (9 shifts, 3 seeds): (0,6), (0,-8), (18,0), (-24,0), (6,6), (-42,0),
  (0,40), (3,-5), (48,0).
  - Plain: all exact.
  - With a status bar in the bottom row and a 5x5 player the camera follows:
    - exact once the game has scrolled;
    - before that, exact or "unsure" (4 of 27 cases), never wrong.
- **Never a claimed scroll:**
  - one sprite stepping over a textured scene;
  - five sprites moving together;
  - a sprite moving by the period of a dotted floor;
  - a 20x20 sprite in a walled room;
  - a 20x20 sprite alone on an empty floor. This one is pixel for pixel a scroll, so it may come out "unsure".
- **Other cases:**
  - a toroidal roll (exact);
  - 4 rows of overlap (no claim);
  - a full-frame flash (no claim);
  - differently shaped frames (no claim).
- **Tracker:**
  - RESETs and level changes;
  - a rewritten history is processed again from the start;
  - the line texts;
  - the payload the sandbox binds.

### 4.5 `left_view` records on the recorded runs: 39

- **bp35 (18):** objects that scrolls carried into the bottom band (2 by 12-row, 13 by 24-row and 3 by 30-row scrolls).
- **lf52 (14):**
  - the cart driving off the right edge (exp-073b step 204, exp-077 step 172);
  - a 4-piece `g` marker that 30-column scrolls carried out (it splits into four 2-pixel objects under
    4-connectivity, so it gives 4 records per scroll).
- **ar25 (7):** an 18-pixel piece of colour c driven off the left edge (6) or the top edge (1).
  - One case was checked by eye (exp-076, steps 45-47): an L shape at cols 3-11, then partly out at cols 0-5, then
    gone.

Earlier versions of this rule also recorded things that had not left the view, found by reading the records:
- a marker blinking at lf52's bottom edge;
- sb26's panel closing;
- sp80's bar turning red at the right edge;
- re86's crosshair changing colour.

The motion and recolour conditions above remove them, and unit tests pin each case.

### 4.6 Lattice on the first frame of every level (183 level starts, 25 games)

Found on 71 starts in 13 games; detection takes 0.33 ms per frame.

| Game | Found (cell @ origin) |
|---|---|
| ls20 | all 7: 5 @ (0,4) |
| cn04 | all 6: 3 @ (2,2), no mixed cells |
| tu93 | all 9: 3 |
| ar25 | all 8: 3 @ (0,0) |
| ft09 | all 6: 2 @ (0,0) |
| sp80 | all 6: 4 @ (0,0) on L1-3, 3 @ (2,2) on L4-6 |
| ka59 | 6 of 7: 3 |
| vc33 | 6 of 7: 4, 2 or 3 |
| wa30 | 6 of 9: 4 @ (0,0) |
| dc22 | 5 of 6: 2 @ (0,0) |
| m0r0 | L1 5 @ (4,4), L3 4 @ (2,2), L5 2 |
| lp85 | L1 and L5: 2 |
| tn36 | L1: 4 @ (1,2) |

None was found in bp35, cd82, g50t, lf52, r11l, re86, s5i5, sb26, sc25, sk48, su15 or tr87. Those are textured or
irregular scenes, and the detector returns None rather than a doubtful lattice.

These were read by eye in ASCII next to the frame: ls20 L1, tu93 L1, cn04 L1, ka59 L3, vc33 L1 and L4, ar25 L2, tn36
L1, wa30 L2, m0r0 L1 and L3, sp80 L1, dc22 L1 and lp85 L1. In each, the cells line up with the drawn pieces. Mixed
cells come from 1-pixel walls or borders, for example dc22's room outlines and lp85's 1-pixel left border column.

The unit test pins ls20, cn04, tu93 and ft09 on all their levels and checks that bp35 and lf52 get no lattice. In the
real sandbox, the bed ran `logical_grid()` 14 times without an error.

### 4.7 `segmentation8`

His default (4-connected) output is unchanged. `segment_layer(frame)` and `segment_layer(frame, connectivity=4)` equal
the unpatched function on the first frame of all 25 games.

On a synthetic diagonal sprite like su15's, 8-connectivity gives 1 object where 4-connectivity gives 5. A diamond ring
of diagonal steps still encloses its centre.

## 5. How it is wired into his harness

- **System prompt.** `_build_system_prompt` appends `system_prompt_lines()`, which is "" when the flag is off.
- **`_run_python_tool`.**
  - It reads the flag once per call.
  - It wraps the state serializer (`attach`), so the state carries `ours_view`: the offsets, the current offset and
    `left_view`.
  - It wraps `_handle_action` (`wrap_actions`), which appends the `[view]` lines to `action_echo`.
  - It passes `perception=` to `run_sandboxed_python`.
- **`python_tool_sandbox.py`.**
  - Three placeholders in the bootstrap template are filled only in the flag-on variants (with and without the search
    helper).
  - The flag-off bootstraps are his strings minus the placeholders. They differ from the seven-patch tree only by the
    segmentation source, which gained the `connectivity` option; a test asserts this.
- **Failure handling.** An internal error turns the helpers off for that game (`view_offset` None, logged) and never
  stops a game.

## 6. Tests

`tests/test_ours_perception_patch.py` has 36 tests and all 36 passed in this session:
- 32 run under `pytest -m "not slow"`;
- 2 are the variant tests, run in a later invocation;
- 2 are slow: `test_every_recorded_run` (section 4.2) and `test_bed_with_all_eight_patches_and_flags`, which took 6.6
  minutes together.

`tests/test_franzen_bed.py`'s fast tests also pass with the bed's new `perception` program.

The tests cover:
- the patch's shape, and that it applies after the seven with its hooks flag-gated;
- the flag-off bootstraps;
- `segment_layer`'s default path on real frames;
- the shift estimator (9 camera moves x 3 seeds, objects-only motion, wraparound and edges, unsure, the tracker,
  `left_view`);
- the lattice on synthetic boards and on real level starts;
- exp-073b's bp35 and lf52 against the engine at every step;
- in the bed venv, with his real `ToolAgent._run_python_tool` and sandbox:
  - flag off or "0": the system prompt and every tool result are byte-identical to the tree without the patch, with
    the other flags off, on (exp-078) and as the next arm (exp-079, RESET not exposed);
  - flag off: three full `analyze()` drives (ls20 as the next arm, ls20 off, vc33 on) produce identical requests,
    history and transcript;
  - flag on:
    - bp35's scroll line and offsets match the engine camera;
    - ls20's `logical_grid()` and `.segmentation8` work in the sandbox, including from a retained function;
    - lf52's scroll and the cart that left the view, from exp-073b's actions;
- that the flag name does not collide with cell 4;
- the exp-080 variant:
  - it changes the same lines as ours-08;
  - on 01, 02, 04, 03b and 05, flag off, the prompt and tool results are byte-identical;
  - on that stack, flag on, bp35's first scroll gives the same line.

## 7. Bed

`scripts/franzen_bed.py` gained a `perception` program, an additive change; with it unset the bed is unchanged and
its tests pass. Its snippet, which takes its turn in the bed's program cycle, makes three moves (or a click) and then
runs four checks.
- `view_offset` must equal the current frame's `.view_offset`.
- `left_view` must be a list.
- `logical_grid()` must run.
- `.segmentation8` must never hold more objects than `.segmentation`.

**All eight patches, every flag on, RESET not exposed** (exp-079's configuration plus this patch; 4 games x 120 s):

```bash
.venv/bin/python scripts/franzen_bed.py --games ls20,vc33,sb26,bp35 --seconds 120 \
  --patch kaggle/franzen/patches/ours-sandbox-timeout-keeps-work.patch --patch kaggle/franzen/patches/ours-02-budget-meter.patch \
  --patch kaggle/franzen/patches/ours-04-search-helper.patch --patch kaggle/franzen/patches/ours-03b-win-ledger-on-02-04.patch \
  --patch kaggle/franzen/patches/ours-05-level-mem.patch --patch kaggle/franzen/patches/ours-06b-effect-table-on-02-04-03b-05.patch \
  --patch kaggle/franzen/patches/ours-07-fresh-start.patch --patch kaggle/franzen/patches/ours-08-perception.patch \
  --env-add OURS_BUDGET_METER=1 --env-add OURS_SEARCH_HELPER=1 --env-add OURS_WIN_LEDGER=1 --env-add OURS_LEVEL_MEM=1 \
  --env-add OURS_EFFECT_TABLE=1 --env-add OURS_FRESH_START=1 --env-add OURS_PERCEPTION=1 \
  --program search --program mem --program effects --program perception --expect "[view] scrolled at step" --expect-in any
```

**Result: 27 of 27 checks ok.**
- The child exited 0 after 127 s, with 326 requests.
- No game crashed and the log has 0 tracebacks.
- RESET was executed 0 times; it is not exposed.
- ls20 and vc33 each won their scripted level 1.

The checks:
- harness made requests; no game crashed; a scripted level-1 win; the priority gate admitted more than once per game;
  games waited at the gate;
- history trimmed; retained function reused; `frame_diff` called; batch no-op guard; stale-state guard; context
  overflow recovered; no tracebacks; UNDO executed;
- the `--expect` text `[view] scrolled at step` was in a request;
- `search()` found a plan; the longer tool time limit; a search ended with a best_partial; `run_plan` stopped at the
  planted mismatch;
- the five `mem` checks; `effects()` returned the current level's table;
- `view_offset` equals the current frame's `.view_offset` (14 of 14 snippets); `logical_grid()` ran (14);
  `segmentation8` never had more objects than `segmentation` (14 of 14).

bp35's scripted moves scrolled the view three times. Each tool result carried a line such as `[view] scrolled at step
28: content moved (+18,+0); view_offset now (-18,+0)`; the others were at steps 50 and 96. The slow test
`test_bed_with_all_eight_patches_and_flags` runs the same bed and passed.

**exp-080's stack plus ours-08b** (01, 02, 04, 03b, 05, 08b; budget meter, search helper, win ledger, level mem and
perception on; RESET not exposed; programs search, mem and perception; same games and seconds): **26 of 26 checks
ok.**
- The child exited 0 after 125 s, with 338 requests.
- No crash, 0 tracebacks, and RESET was executed 0 times.
- The perception checks passed 15 of 15 times.
- `[view] scrolled at step 68` and `at step 86` appeared.
- That bed's recorded games were replayed through the engine:
  - bp35's 2 scrolls were claimed exactly; the second came after an automatic RESET at step 78.
  - All 109 of bp35's offsets agree with the engine camera.
  - ls20, vc33 and sb26 had no scroll and no claim.

## 8. Cost

- **Prompt.** 853 characters (four lines), present only with the flag on.
- **Per action.** One tracker update per new history entry: mean 1.71 ms, p99 3.74 ms, max 59.7 ms. The payload adds
  one `[dr, dc]` or `null` per history entry to the state file.
- **On demand.** `logical_grid()` detection takes 0.33 ms per frame. `.segmentation8` costs one segmentation, cached
  per frame object.
- **Notebook.** Both arms were built with `--compact` in this session; lesson 0033 has the size limit.
  - The eight-patch arm is 824,259 bytes.
  - exp-080's set plus ours-08b is 797,239 bytes, against exp-080's own 772,503.

## 9. Known limits

- **A lone large object on an otherwise empty floor** is pixel for pixel a scroll. It is not claimed, but it may make
  the offset unknown, silently.
- **A large object moving in an arena whose walls lie in the 4-cell edge band** could be claimed as a scroll, because
  the band rule discounts the band when something there stays put. None of the 25 games does this.
- **A scroll with too little evidence** goes unnoticed and leaves the offset silently wrong: fewer than 64 supporting
  cells, or spread over fewer than 8 rows or columns. None of the 285 recorded scrolls was like this. No holdout exists
  (section 3).
- **Unknown offsets on still steps:** 10 times in ar25 and cn04 (section 4.2). The model then sees None until the next
  RESET or level.
- **`left_view` gaps.**
  - It records only objects whose colour is gone from the whole view; an object leaving while another of its colour
    stays in view is not recorded.
  - It uses 4-connected pieces, so a ring may give several records.
  - An object that the scroll carried out and that also vanished at the same moment cannot be told apart from one that
    left.
- **`logical_grid` limits.**
  - Square cells only.
  - The origin may include a 1-pixel border strip, shown as `*` cells (lp85).
  - No lattice in textured or irregular scenes.
- **Not done from the brief.** No item was dropped. Item 2 is the restricted form above. The brief's "view moved by
  (dr, dc)" became "content moved (dr, dc); view_offset now (...)", which says which way the numbers point.

## 10. Building an arm

**On exp-080's set, the current candidate.** After today's runs, 06b and 07 were dropped and RESET is not exposed. Both
commands below were built locally in this session; neither was pushed.

The build log shows ours-08b's sha256 `5745781e03d3`; the notebook is 797,239 bytes.

```bash
.venv/bin/python scripts/build_franzen_nb.py --base dprime --out build/perception5 \
  --slug arc3-dprime-r14a05-harness5-perception-noreset-full \
  --input-fallback --wait-inputs 120 --reap-kept kaggle/franzen/reap448_kept_experts.json \
  --cfg MAXREQ=14 --cfg CUDAGRAPH_MAXBS=14 --cfg MAMBA_CACHE=84 --cfg SPEC_ACCEPT_SINGLE=0.5 --cfg SPEC_ACCEPT_ACC=0.5 \
  --env ARC3_MAX_ACTIVE_STREAMS=14 --hot-tokens kaggle/franzen/hot_tokens_64k_arc.pt --fail-fast --full25 121 --compact \
  --patch kaggle/franzen/patches/ours-sandbox-timeout-keeps-work.patch \
  --patch kaggle/franzen/patches/ours-02-budget-meter.patch \
  --patch kaggle/franzen/patches/ours-04-search-helper.patch \
  --patch kaggle/franzen/patches/ours-03b-win-ledger-on-02-04.patch \
  --patch kaggle/franzen/patches/ours-05-level-mem.patch \
  --patch kaggle/franzen/patches/ours-08b-perception-on-01-02-04-03b-05.patch \
  --env-add OURS_BUDGET_METER=1 --env-add OURS_SEARCH_HELPER=1 --env-add OURS_WIN_LEDGER=1 --env-add OURS_LEVEL_MEM=1 \
  --env-add OURS_PERCEPTION=1
```

**The control arm** is exp-080 itself; with the flag off, ours-08b is inert, as the variant test shows.

**On the seven-patch bundle, as the brief specified** (exp-079's configuration). The build log shows ours-08's sha256
`f202d54b5063`; the notebook is 824,259 bytes. Use the same command with these patch lines and these flags instead:

```bash
  --patch kaggle/franzen/patches/ours-sandbox-timeout-keeps-work.patch \
  --patch kaggle/franzen/patches/ours-02-budget-meter.patch \
  --patch kaggle/franzen/patches/ours-04-search-helper.patch \
  --patch kaggle/franzen/patches/ours-03b-win-ledger-on-02-04.patch \
  --patch kaggle/franzen/patches/ours-05-level-mem.patch \
  --patch kaggle/franzen/patches/ours-06b-effect-table-on-02-04-03b-05.patch \
  --patch kaggle/franzen/patches/ours-07-fresh-start.patch \
  --patch kaggle/franzen/patches/ours-08-perception.patch \
  --env-add OURS_BUDGET_METER=1 --env-add OURS_SEARCH_HELPER=1 --env-add OURS_WIN_LEDGER=1 --env-add OURS_LEVEL_MEM=1 \
  --env-add OURS_EFFECT_TABLE=1 --env-add OURS_FRESH_START=1 --env-add OURS_PERCEPTION=1
```

**What to read in the run's logs:**
- `[view]` lines in bp35 and lf52, and none in games that do not scroll;
- the use of `view_offset`, `left_view`, `logical_grid(` and `segmentation8` in snippets;
- the actions to the first move on ls20's and cn04's levels;
- lf52's single-action probes after its cart leaves the view.

## 11. Reproduce

```bash
.venv/bin/python -m pytest tests/test_ours_perception_patch.py            # all, including the two slow tests
.venv/bin/python -m pytest tests/test_ours_perception_patch.py -m "not slow"
```

The section-4 tallies were made with scratch scripts that were not added to the repo. `test_every_recorded_run` (4.2)
and `test_recorded_scrolls_agree_with_the_engine_at_every_step` (4.1) recompute the same comparisons. To rebuild the
patch, build a tree with `scripts/franzen_tree.py build DIR` and the seven patches, apply this one, and edit there.
Then run `scripts/franzen_tree.py diff DIR > kaggle/franzen/patches/ours-08-perception.patch` and `check` with all
eight. For ours-08b, do the same on 01, 02, 04, 03b and 05; the hooks go at the places shown in its diff.
