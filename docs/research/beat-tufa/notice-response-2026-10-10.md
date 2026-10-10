Summary: the model ignores most I1 lines for two reasons. In 70% of firings it is busy with its own plan and never mentions the line. And the lines mostly name inert things: only 13% of the named entries react to a click (19% at rank 1). Even when the model did act on the line (32 of 299 firings), 26 of its 32 clicks changed nothing. Two changes are supported. Cap the list at 3 kinds: all 6 useful clicks were on ranks 1-2, and all 12 clicks on ranks 4-6 hit inert objects. And fire the first line at 5 minutes when MOUSE is still unused on the level: those lines drew a click citing the line 8 of 18 times, against 24 of 247 for other lines.

# Why the model ignores most I1 lines, and two changes (2026-10-10)

CPU only. No GPU, no Kaggle, no repo code changed. The scripts are in the session scratchpad `notice2/` and are not committed.

## Sources and method

- **Runs.** exp-085, exp-085r and exp-086 (`runs/exp085*-untried-full`, `runs/exp086-*-untried-full`). `scripts/untried_read.py --json` gives the 299 I1 firings the model saw (101 / 99 / 99) and the named clicks within 3 actions (20 / 23 / 17).
- **Response window.** For each firing: the turn whose prompt carried the line, up to and including the first turn that executed an action. "Next turns" means the turns after that, until 6 more actions (used for deferrals).
- **Classes.** A script found every window that referred to the line ("not yet tried", "the hint/note/reminder says", "no click has hit", a named box). I then read by hand:
  - all 124 windows with a hit that drew no named click;
  - the passages around the hits in the 60 windows that did draw one;
  - 30 random windows from the 115 with no hit. None of the 30 mentions the line in the window itself.
  - All 115 were also scanned for references in the next turns, which found 4.
- **Probe.** For each firing, the engine state the prompt saw (local replay) was deep-copied. The probe then clicked once on every unclicked kind, at the cell of its first object nearest the box centre, and once on background.
  - A kind is "interactive" when the board inside the 4-cell edge band differs after the two clicks.
  - 21 firings are left out of the probe statistics: in su15, r11l and m0r0 a background click also changes the board.

## 1. What the model did after each line (all 299 classified)

| Run | Lines | (a) clicked a named object, citing the line | (a) clicked one, no mention | (b) read, dismissed | (c) planned a probe, deferred | (e) acted otherwise | (d) no sign it was read |
|---|---:|---:|---:|---:|---:|---:|---:|
| exp-085 | 101 | 12 | 8 | 4 | 3 | 5 | 69 |
| exp-085r | 99 | 11 | 12 | 1 | 7 | 1 | 67 |
| exp-086 | 99 | 9 | 8 | 1 | 6 | 3 | 72 |
| All | 299 | 32 (11%) | 28 (9%) | 6 (2%) | 16 (5%) | 9 (3%) | 208 (70%) |

- **Uncited clicks are the base rate.** Clicks with no mention (28/299, 9.4%) match the 10% rate at which runs without the line click those objects (32/318). The line's own effect is the 32 cited clicks.
- **Slow-boot artefact.** 13 exp-085r lines fired at step 1-2 of level 1, because the level-1 clock counted the slow boot (13.2-13.9 min "on the level"). 5 of that run's 23 hits are these. None of them cites the line.
- **(a), citing the line.**
  - sp80 L2, exp-085, step 43: "Maybe the paddles can also be moved by MOUSE clicks (the 'not yet tried' hints!)". It clicked the red bar and won the level.
  - m0r0 L5, exp-085, step 199: "The hold model says the merge is impossible ... The hint says ... Let me test MOUSE on a colored block!"
- **(b), dismissed.**
  - lf52 L3, exp-085, step 140: "these are the blue shadow objects (b) and white (W). Those are just shadows/borders, not interactive."
  - vc33 L4, exp-086, step 95: "clicking a wall might do nothing".
  - sc25 L6, exp-085, step 283: "did clicking on gray do anything? Probably not."
  - The probe agrees: 5 of the 6 dismissed lists held no interactive entry.
- **(c), deferred.**
  - dc22 L3, exp-085, step 168: "Let me try clicking the purple pod icon (42,17) and the pink dot (16,6). Hmm, but I should think more." It never clicked them.
  - vc33 L4, exp-086, step 79: "That's ~10 actions. The budget is 27. Hmm, that's a lot but might be necessary." It made 1 of the 8 planned clicks.
  - Of the 16 deferred probes, 9 were made 4-66 actions later; only one of those changed the board (sp80's red bar). The other 7 were never made.
- **(e), acted otherwise.** bp35 L1, exp-085, step 12: "The hint says ... Let me click on the green block". The green block is not on the list. 4 of the 9 used a named unused action instead (SPACE, UP, LEFT twice).
- **(d), no sign it was read.** 29 of the 30 hand-read windows are class (d); the 30th mentions the line only in a later turn. Of the 29:
  - 15 were executing a plan, route or search result, e.g. "Let me execute it with run_plan" (m0r0 L5, exp-085, step 213);
  - 9 were testing their own hypothesis;
  - 3 were the first turns of a game (the exp-085r slow boot);
  - 2 were reading their own records.

**Why ~80% of lines draw no named click:**
1. **Mid-plan.** 70% of lines arrive while the model is following its own plan, and it never mentions them. Nothing in the line says the level has stalled.
2. **Inert content, and the model is mostly right.** 87% of the named entries do not react to a click (section 2). Dismissals were correct 5 times out of 6. Only 6 of the 32 cited clicks changed the board, against 19 of the 28 uncited ones.
3. **Repeats and late lines.** Repeats drew half the cited uptake of first lines: 10/148 against 22/151 (p = 0.04). Cited clicks came at a median of 12.0 min into the level. Ignored lines came at a median of 23 min.
4. **Probes feel costly.**
   - Deferrals cite the budget.
   - 6 of the 32 cited clicks (checked by hand) were spending the last actions of an attempt already given up: "Budget: 3 actions. I can't win. Let me use the remaining 3 actions to learn the most valuable thing" (su15 L2, exp-085, step 53).
   - This is not the usual habit. Over 8 runs, first clicks on a never-clicked kind are 2.6% of the last 5 actions of lost attempts (25/971), against 5.5% elsewhere.

## 2. Which named entries were interactive

| | exp-085 | exp-085r | exp-086 | All |
|---|---:|---:|---:|---:|
| Named entries (254 firings, probe statistics) | 462 | 419 | 463 | 1,344 |
| Interactive in the probe | 51 (11%) | 55 (13%) | 67 (14%) | 173 (13%) |

- **Check against real clicks.** 169 named entries were of a kind the run itself clicked on that level at some time. The probe matches the real click for 160 of them: both changed the board for 51, neither for 109. The other 9 changed the board only in the real click; these controls depend on state.
- **What the inert 87% are:** walls, water, shadows, panels, HUD pieces, the avatar, and controls that are dead in their current state (vc33's grey gate).
- **By game.**
  - cn04: 82% interactive.
  - sc25, sp80, tn36, sb26: 24-26%.
  - All other games: 7% or less (ka59 4%, dc22 2%, vc33 1%, sk48 0%, ft09 0%).
- **Precision of the current order.**
  - By rank: rank 1 49/254 (19%), rank 2 31/243 (13%), rank 3 23/234 (10%), ranks 4-6 9-13%.
  - At least one interactive entry in the top 1/2/3/6: 49, 61, 73 and 99 of 254 firings.
  - Among all unclicked kinds (up to 40 probed per firing): 134 of 254. So in 47% of firings nothing unclicked was interactive.

Feature tests: interactive share among the 1,344 named entries, when the feature holds and when it does not.

| Feature (computable by the harness) | Holds | Does not hold |
|---|---:|---:|
| Colour of an object whose real click changed the board earlier in the game | 88/359 (25%) | 85/985 (9%) |
| Thin (one side 2 cells or less) | 26/341 (8%) | 147/1003 (15%) |
| Solid (fills at least 90% of its box) | 76/774 (10%) | 97/570 (17%) |
| 100 px or more | 34/159 (21%) | 139/1185 (12%) |
| New on this level (current first key) | 136/974 (14%) | 37/370 (10%) |
| One copy | 129/1001 (13%) | 44/343 (13%) |
| Its cells changed earlier on the level | 76/656 (12%) | 97/688 (14%) |
| Within 3 cells of the last 5 actions' changes (stands in for "near the avatar") | 75/633 (12%) | 98/711 (14%) |
| Colour also present in the edge band (HUD colour) | 96/710 (14%) | 77/634 (12%) |

**Orderings tried** (rank-1 precision; firings with an interactive entry in the top 3, out of 254):
- current: 19%, 73;
- known-control colour first: 21%, 81;
- thin kinds dropped: 19%, 72;
- unchanged-on-level first: 20%, 84. This one moves sp80's red bars from rank 1 to rank 3-4 in 22 of 30 replayed sp80 L2 lines, because the bars swap colour.

No simple feature lifts rank-1 precision above about 21%. **Where the cited clicks went:** rank 1 13, rank 2 7, ranks 4-6 12. All 6 that changed the board were at ranks 1-2. None of the 12 at ranks 4-6 changed it (6/20 against 0/12, p = 0.06).

## 3. Timing

- **When uptake happened.**
  - Cited clicks by time into the level: before 15 min 19/123 (15%), 15-30 min 5/63 (8%), 30 min and later 8/113 (7%).
  - The 6 useful ones came at 10.3-28.1 min. 5 of them came from the first line on their level.
- **Replay at 5 minutes for all levels.** `timing.py` runs the shipped module with STALL_SECONDS = 300 and the repeat kept at 600, over 8 runs (exp-073, 073b, 075, 083, 084, 085, 085r, 086).
  - Lines on solved levels go from 440 to 755 (+39 per run). Lines on unsolved levels go from 388 to 434.
  - Levels solved anyway that get a line rise from 29-41 to 48-72 per run.
- **Same objects at 5 minutes?** On average, 59% of the 10-minute names are already in the 5-minute line. The lists are identical on 118 of 245 levels in the baseline runs and on 68 of 142 in the GPU runs. In the post-mortem's 7 losing game-runs:

| Losing game-run | Critical object | Line at 10 min | Line at 5 min |
|---|---|---|---|
| sp80 L2: exp-073, 075, 083 | red bars | 10.0-10.4 min, rank 1, MOUSE never used | 5.1-5.4 min, rank 1, MOUSE never used |
| tn36 L2: exp-073 | demo box ring | 10.3 min, rank 1 | 5.3 min, rank 1 |
| cn04 L2: exp-075 | N socket | 10.3 min, rank 2 | 5.5 min, rank 3 |
| vc33 L4: exp-073, 083 | orange gate | not named: the gate is grey at the firings (exp-073 names it at 91 min, rank 7) | not named |

- **Finding.** 10 minutes is not what stops uptake. The post-mortem's losing runs spent 67-110 min on those levels, so they saw the line many times. A 5-minute line for every level names the same critical objects at the same rank. But it adds ~39 lines per run on levels solved anyway. At the measured rates (11% cited, 81% of cited clicks inert) that costs about 3-4 inert clicks per run.

## 4. Wording

| Line | Lines | Cited named click | Useful (board changed) | No sign (d) |
|---|---:|---:|---:|---:|
| MOUSE listed as never used (step > 3) | 18 | 8 (44%) | 3 | 8 |
| MOUSE not listed (step > 3) | 247 | 24 (10%) | 3 | 174 |
| First line on the level | 151 | 22 (15%) | 5 | 101 (67%) |
| Repeat (10-minute or game over) | 148 | 10 (7%) | 1 | 107 (72%) |
| 1-3 kinds named | 37 | 6 (16%) | – | 23 |
| 4-6 kinds named | 238 | 26 (11%) | – | 163 |
| Game-over trigger | 34 | 4 (12%) | – | 26 |
| Actions only (24 lines) | 24 | – (2 used the named action) | – | 22 |

- **"MOUSE never used" is the line that works.** p = 0.0004 for 8/18 against 24/247. Those 18 lines also produced half the useful clicks (3 of 6).
- **Length does not matter.** The cited share does not depend on the number of kinds (p = 0.4) or on characters (200+ characters: 11%; 120-199: 14%).
- **Repeats are ignored more**, but the one critical-level win came from a third line (sp80 L2, exp-085, 24 min in).
- **The model has no name for the line.** It calls it "the hint", "the note", "the reminder", "the harness" or "the system hint".
- **It takes the list literally.** "So the harness suggests clicking the gray box's gray cells ... That seems odd but cheap to test" (dc22 L2, exp-085, step 62). The box's frame and its icon are separate components, so one control appears as several entries.
- **I2 (noted only).** 36 lines drew 4 named clicks within 3 actions. In 8 of them the object no longer had the new colour on the board the prompt saw: vc33's gate in exp-085 and exp-086, tu93 three times, wa30 three times. exp-086 then dismissed the gate: "The harness's note about 'O' must refer to a transient change" (vc33 L4, step 75). If I2 stays, it should check the colour at prompt time.

## 5. Proposals (at most two)

**P1. Cap the object list at 3 kinds (MAX_KINDS 6 → 3), same order.**
- **Evidence.**
  - Section 2: the 6 useful cited clicks were all at ranks 1-2; the 12 cited clicks at ranks 4-6 were all inert; ranks 4-6 are 9-13% interactive.
  - Section 4: a shorter list does not lower uptake.
- **Replay** (`timing2.py`, the shipped module text-patched, 8 runs).
  - The same 828 lines fire.
  - Per-run median length drops from 209-215 to 164-166 characters (about 105 → 70 tokens, at the patch doc's ~12 tokens per kind).
  - In the 5 losing game-runs where the 10-minute line names the critical object, it stays inside the cap: tn36 1, sp80 1 (three runs), cn04 2.
- **What is lost.**
  - cn04's second socket (b, rank 4).
  - In the GPU runs, tn36's box-2 yellow (rank 4-5, interactive), which the model never clicked.
  - 26 of 254 firings had their only interactive entries at ranks 4-6. Cited clicks on those ranks never changed the board.
- **Expected effect.** About 4 fewer inert clicks per run (12 over the 3 runs), and none of the 6 useful clicks lost.
- **Read in a GPU arm:** cited clicks per line, and their board-change rate.

**P2. When MOUSE is valid and unused on the level, fire the first I1 at 5 minutes.**
- **Rule.** Everything else stays: the 10-minute stall for other levels, repeats 10 minutes after the last line, and game over past the mark.
- **Evidence.**
  - Section 4: these lines drew cited clicks 8/18 against 24/247, and 3 of the 6 useful clicks.
  - Section 3: uptake is highest on first and early lines.
  - In all three losing sp80 L2 runs the line read "MOUSE never used" with the red bars at rank 1.
- **Replay** (8 runs): 828 → 902 lines (+9 per run). 111 levels get the line before 10 minutes:
  - 39 solved within 10 min (new lines);
  - 50 solved later (line 5 min earlier);
  - 22 unsolved (line 5 min earlier).
- **Critical levels.**
  - sp80 L2 in exp-073/075/083 gets "MOUSE never used; no click has hit R r16-19 c8-19 ..." at 5.1-5.4 min.
  - cn04 L2 in exp-075 gets the N socket at rank 3 at 5.5 min.
- **Cost.** If uptake holds (44%) and 5 of 8 such clicks stay inert, the 9 extra lines per run cost ~2-3 inert clicks per run (9 × 0.44 × 5/8). The lines fall mostly in ar25, m0r0, sk48 and ka59 (22, 20, 14 and 13 of the 111 levels), whose named entries were 0-7% interactive.
- **Read in a GPU arm:** the cited-click rate on the 5-minute lines, and sp80 L2.

**Not proposed:**
- a 5-minute line for every level (+39 lines per run on solved levels, no new objects named);
- unchanged-first ordering (it demotes the red bars);
- dropping repeats (the sp80 L2 win came from a third line);
- known-control-colour-first ordering (+2 points at rank 1, too little to justify alone).

The data do not support any ordering change.

## Limits

- **Classes.** One reader, one pass. The 115 windows with no reference were checked by a sample of 30.
- **Probe.**
  - One click at one cell, in the state the prompt saw. Controls that only work in some states count as inert (vc33's grey gate).
  - "Interactive" is not the same as useful. 25 probe clicks ended in GAME OVER and are counted as inert.
- **Replay timing.** Prompt times in the replays are reconstructed, as in `scripts/untried_replay.py`.
- **P2's evidence is 18 lines.** These lines come when the model has not clicked at all, so part of their uptake may be the novelty of a whole action type. Whether that novelty still pulls at 5 minutes is untested. Neither change has run on a GPU.
- **Follow-up (not fixed).** Start the level-1 clock at the first action() return, not the first prompt. The slow boot made 13 exp-085r lines fire at step 1-2.
- **Scripts.** In the session scratchpad `notice2/`, not committed:
  - `extract.py`, `refs.py`, `classes.py`: windows and classes (the hand labels are in `classes.py`);
  - `probe.py`, `probe_stats.py`, `orderings.py`: the probe and its statistics;
  - `crosstab.py`, `endprobe.py`: Q1/Q4 cross-tabs and end-of-attempt probing;
  - `timing.py`, `timing2.py`: the replay variants.
