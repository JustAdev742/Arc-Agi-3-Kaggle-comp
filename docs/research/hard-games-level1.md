# Level 1 on the hard public games: what the Duck believed, where it went wrong, and which harness change would help

Research note, 2026-09-25. Read-only analysis; no GPU. Follow-up: R1 and R2 were built as P23 and P24 (scripts/taaf_ours_patch.py) and queued as exp-050 against exp-049; their engine replay is `hard-games-level1/p23_p24_replay.txt`.

Data: transcripts of 9 games x 5 runs (exp032 base, exp042 and exp042r fixes, exp045 base + P21 gate, exp048 fixes + gate
+ 6.5 GiB KV), `runs/<run>/summary.json`, game source under `environment_files/`, and three offline engine checks. The
analysis scripts are in `docs/research/hard-games-level1/` (exploratory, not maintained): `extract.py` (belief snapshots at 15/60/120 min),
`metrics.py` (level-1 actions, calls, idle stretches, first use of each action), `calls.py` (share of tool calls that act
or diff), `asst.py` (reasoning in a time window), and `tr87_check.py`, `g50t_check2.py`, `tn36_check.py` (engine checks).
Times are minutes from the game's first turn. "Calls" means model responses. Level score = min((baseline/actions)^2, 1.15).

## 1. Outcome on level 1 (40 runs; su15 listed separately)

| game | L1 solved | actions when solved (baseline) | minutes to solve | L1 calls | longest no-action stretch |
|---|---|---|---|---|---|
| sk48 | 0/5 | none; 70-170 spent (61) | none | 34-55 | 11-30 min |
| bp35 | 2/5 | 44, 50 (21): scores 0.23 and 0.18 | 106, 94 | 35-54 | 18-47 |
| g50t | 2/5 | 50, 64 (78): 1.15 | 131, 117 | 33-53 | 13-80 |
| dc22 | 2/5 | 81, 45 (59): 0.53 and 1.15 | 81, 60 | 29-54 | 10-26 |
| tn36 | 4/5 | 31, 138, 25, 16 (32) | 50, 130, 39, 50 | 21-54 | 10-26 |
| ls20 | 5/5 | 43, 92, 38, 24, 20 (22): 0.26, 0.06, 0.34, 0.84, 1.15 | 51, 106, 61, 42, 33 | 16-44 | 11-25 |
| tr87 | 3/5 | 80, 57, 46 (54): 0.46, 0.90, 1.15 | 68, 127, 131 | 31-52 | **43-113** |
| m0r0 | 4/5 | 23, 32, 38, 21 (30) | 29, 39, 51, 105 | 17-56 | 8-45 |
| su15 | 5/5 | 13, 15, 19, 22, 9 (22) | 11-32 | 10-19 | 3-5 |

- **su15 is not a level-1 problem.** Level 1 was solved in all 5 runs at or under the baseline within 32 minutes; its low
  score comes from level 2 onward. It is left out of the counts below.
- Of the 40 other level-1 attempts, **18 were never solved** and 13 were solved slowly (after 90 min or more, which
  leaves no time for level 2) or inefficiently (level score under 0.5). Only 9 were clean.
- Many solves land late: g50t at 117 and 131 min, tr87 at 127 and 131, bp35 at 94 and 106, tn36 (exp042) at 130.
- On hard level 1s the model acts less. 38% of level-1 tool calls contain `action(` (1,478 calls), against 51% on the
  other 16 games (1,380 calls). On tr87 it is 13%. 24% of hard level-1 calls read `previous_frame`, `before_frame` or
  `history[`, mostly to write diffs by hand.
- **The P21 gate starves level 1.** On the 8 hard games, the gated runs solved level 1 in 6 of 16 attempts and the
  ungated runs in 16 of 24 (Fisher p = 0.11). The direction holds within both pairs: base 4/8 without the gate against
  2/8 with it; fixes 5/8 and 7/8 without it against 4/8 with it. Gated runs averaged 30-31 level-1 calls per hard game;
  ungated runs averaged 37-44. Solved hard level 1s took a median of about 30 calls, and many took more. This supports
  hypothesis (2) of the 2026-09-24 research-log entry: on a hidden set where most games stay on level 1, the gate removes
  the calls those games need.

## 2. Per game: the level-1 mechanic (from source) and the model's belief over time

### sk48: 0/5 (baseline 61; oracle 14)
**Mechanic** (`sk48.py` level 1 and `gvtmoopqgy`). A magenta head on a left rail carries an arm. RIGHT extends the arm
by one 6 px segment, LEFT retracts it, and UP/DOWN move head and arm between lanes. Blocks lying on the arm's segments
are pushed, pulled and carried, and pushes cascade. The win is checked only after an animated arrow move: the blocks on
the arm, read from the head outward, must match the reference arm drawn in the bottom strip (R, N, b). The move budget
is 196.

**Belief:**
- At 15 min, every run had movement right and the goal wrong. Examples: "Open question: how targets are collected"
  (exp042); "connect the pink base's bar to red, then green, then blue (a chain)" (exp048); "extend arm right to touch
  red" (exp045).
- At 60 min, the push/pull rules were half learned. "Docking all three at the left wall did not win, so the goal is
  still unknown" (exp042, 57 min). "sort blocks left→right in that order (all-equal columns didn't win)" (exp042r).
- At 120 min, 3 runs had roughly the right goal but a wrong rule model. "HUD order R,N,b = packed order in player's
  lane … Never tried exact order" (exp045, 116 min). At 111 min the same run was "testing R,b,N first since it's
  cheapest". "My rule model is still off in details … build simulator + BFS" (exp045, 122 min, with 10 minutes left).

**Where it went wrong.** The goal was found late, and the plans were long open-loop batches of 15-41 actions built on a
rule model that each batch then contradicted ("Lesson learned: when the arm passes a lane vertically, it pushes/pulls
EVERY block…", exp045, 102 min). Runs spent 70-170 actions and 34-55 calls.

### bp35: 2/5 (baseline 21; code solution 15)
**Mechanic** (census; the source is 4,565 obfuscated lines). A character moves LEFT/RIGHT and gravity pulls it upward.
Clicks break or toggle terrain. The camera scrolls, and the goal gem starts off-screen. The level is won when the
character reaches the gem.

**Belief:**
- 15 min: dig to collect the green "gems" (exp042); "Goal = collect the 7 green blocks … or reach somewhere?" (exp045).
- 60 min: the camera scroll was confusing. "the board looks completely different … whether this is a level
  reset/regeneration" (exp042). "It is a rising-bubble ascent puzzle with generated content above" (exp048).
- 120 min, in the unsolved runs: "Now stitching all frames into a single world map" (exp048, after 43 minutes without an
  action and 11 actions in total). "greens are removable blocks … so the buoyant fish floats up to the top room/pink" (exp045, 110 min).

**Where it went wrong.** The model misread the goal: it treated the green blocks as collectibles and never had the gem on
screen early. Both solves were partly accidental: "Level 1 SOLVED … by pressing LEFT 3× so the ship's column matched
the pink sparkle's column" (exp042r, 94 min). The scroll also confused its perception.

### g50t: 2/5 (baseline 78; validation game)
**Mechanic** (`g50t.py` `step` and `pmlawcgvcp`, checked in the engine). A 5x5 block moves one cell per arrow key, and a
laser barrier blocks the route to the goal ring. SPACE rewinds the player to the start. From the next action on, a gray
clone replays the recorded path one step per action, whatever key is pressed. The clone standing on the emitter holds
the barrier open. Engine check: after RIGHT x4 and SPACE, the player is back at (8,14); on DOWN, a gray 5x5 appears at
(8,20); on the next DOWN it moves to (8,26), to the right.

**Belief:**
- 15 min: "Goal = blue ring … Red = laser path … Plan: probe SPACE" (exp032). "SPACE looks like an undo/reset-to-start
  toggle" (exp042r).
- 60 min: in exp042, a sokoban-style crate hypothesis ("push the white crate onto the source button"). In exp042r:
  "The gray block tracks the blue block's position from ~5 actions earlier … it's a delayed echo" (57 min).
- 120 min, exp042: "Gate opens only while the block covers the source … Block can't be in both places ⇒ my map/goal
  reading must be wrong." exp048: "top-left is a life/attempt counter … SPACE spent one, so avoid SPACE."
  exp045: no action between 55 and 132 min (6 actions in total).

**Where it went wrong.** SPACE was tried early (5-18 min) in 3 runs and late (55, 76 min) in the gated runs, but its
effect is delayed and spread over later actions. In all 5 runs it was understood late or never. The solves came at 117
and 131 min.

### dc22: 2/5 (baseline 59; oracle 20)
**Mechanic** (`dc22.py:10891`). Arrow keys move a green 2x2 avatar, but only onto walkable pieces. Two buttons on the
right panel toggle a red bar and a blue block between two placements, which changes the walkable paths. The level is won
when the avatar's top-left equals the yellow marker's, checked after arrow moves only.

**Belief:**
- 15 min: "maybe the right panel is a goal showing the final configuration" (exp042).
- 60 min: "All 36 (green × red × blue) states tried — no completion" (exp048). "all 9 knob positions × toggles tested"
  (exp042). exp045 called the avatar a "knob/joystick".
- 120 min: "Green walks only on gray and is permanently trapped … All 36 configs … score 0" (exp032, after 201 actions).
  "the goal must be an event: a button press at a specific dot cell" (exp048, 158 actions).

**Where it went wrong.** The model read the goal as a static switch configuration and never tested walking toward the
yellow in the other toggle states. The solving run said: "exit box2 via blue solid, cross bar in R0, toggle R1+B1, climb ladder, enter box1 → yellow
… Goal = reach the yellow" (exp042r, 60 min).

### tn36: 4/5 (baseline 32)
**Mechanic** (census, and the engine). Clicks toggle instruction bits, and clicking the ball runs the program. The piece
moves during the run animation and snaps back unless it ends matching the target. Engine check: one run shows the piece
(colour 11) at rows 13, 17, 17, 21, 21, 21, then back at 13 on the final frame; the final board differs from the start
in one cell.

**Belief:**
- Solved runs: they measured the piece's descent per configuration from `animation()` ("with 1 removed the descent is
  12 < 20", exp048, 43 min).
- Failures: "Ball animation is fully canned (identical 7 frames/256 transient px for different configs) → pure
  demo/distractor" (exp042, 58 min; that run then took 138 actions, 2 game-overs and 130 min). "the plug lands in the
  socket for frames 2-5 but reverts; no level completion" (exp045, 117 min; unsolved).

**Where it went wrong.** The action's effect was visible only mid-animation, and pixel-count summaries misled the model.

### ls20: 5/5 but costly (baseline 22; oracle 13)
**Mechanic.** A 5x5 key-block moves 5 px per step. Landing on the "+" rotates the key, which is shown in the bottom-left
box. The socket accepts the key only when its orientation matches the socket glyph. The bar allows 42 moves, and running
out resets the attempt.

**Belief:**
- 15 min: "Goal appears to be landing the block on the top door box" (exp042).
- 60 min: exp042 had the rotator ("the plus is a BUTTON that cycles the icon").
- 84 min: "I matched the dial to the door glyph at step 26 but never probed the door then" (exp042).
- After that, the planning went wrong. The model's BFS map treated its own pixels as wall and missed a J-shaped
  obstacle ("the map treated the block's own blue pixels ('b') as walls"). The bar ran out and the attempt reset.
  Level 1 took 92 actions (score 0.06). exp042r also passed through the matching state without testing the box (38
  actions).

### tr87: 3/5, all slow (baseline 54; code solution 15)
**Mechanic** (`tr87.py` level 1). Six rule pairs A→B are drawn at the top. The bottom shows a source row of 5 A-glyphs
and an answer row of 5 B-glyphs. LEFT/RIGHT move the cursor, and UP/DOWN cycle the selected answer glyph through 7
values. The level is won when answer[i] = rule(source[i]), checked after UP/DOWN. **Every glyph sprite is drawn at a
random rotation**, so identities must be matched up to rotation.

**Belief:**
- 60 min: exp032: "Each bottom blue glyph is a dihedral variant of an example blue" (solved at 68 min, with 80
  actions). exp042: "Still unknown: how to derive the answer for the 4 unseen blue glyphs … Test CA consistency". The
  same note was still there at 118 min, and the run stopped acting after action 11, 112 minutes before the end.
- 120 min: "Suspect identity mislabeling (orbit collisions or window offsets)" (exp042r, solved at 127 min). "Big
  finding: the bottom rows' glyph hashes now coincide" (exp045, 108 min; it took no action after 56 min).

**Where it went wrong.** This is a perception problem, and every run also stalled: the longest stretch without an action
was 43-113 minutes in all five. **Engine check (`tr87_check.py`):** on the first frame, the segmentation's
translation-only shape key pairs 2 of the 5 source glyphs with a rule-table glyph. A rotation-invariant key pairs all 5
source glyphs with their rule's left glyph, and 4 answer-row glyphs with rule right glyphs.

### m0r0: 4/5 (baseline 30; oracle 15)
**Mechanic** (`m0r0.py:863`). Two mirrored pieces move together, with horizontal moves reversed for one of them. The
level is won when they collide and merge.

**Belief:**
- exp032 (unsolved): "blocks can only cross by merging, a true identity swap is likely impossible" (117 min). The
  model read merging as a way for the pieces to cross, not as the goal, and went 45 minutes without an action.
- exp045: "blocks are now adjacent at the top-center … no completion" (51 min). It then spent 54 minutes on other probes
  (SPACE and MOUSE, which do nothing, then DOWN). The resolution: "the goal was to make the two blocks OVERLAP … RIGHT achieved it" (105
  min).

## 3. Failure modes (31 problem runs: 18 unsolved, 13 slow or inefficient)

| mode | primary | any role | runs (game run) |
|---|---|---|---|
| A. Misread goal / goal found late | 11 | 15 | sk48 x5, bp35 x5, dc22 x4 (032, 042, 048, 045), m0r0 032 |
| B. Key action's effect misunderstood (delayed, mid-animation, far-away display) | 9 | 16 | g50t x5, sk48 x4, tn36 042/045, dc22 x3, bp35 045, ls20 032 |
| E. Misperceived objects (rotated identity, own sprite as wall, HUD misread, camera scroll) | 5 | 8 | tr87 x5, bp35 032/048, g50t 048 |
| D. No action for 40 min or more (re-inspection loop) | 2 | 9 | tr87 x5, bp35 032/048, g50t 045, m0r0 032 |
| C. Right goal, bad plan or execution (open-loop batch on a wrong model, bad map, budget reset) | 2 | 3 | sk48 045, ls20 042, tn36 042 |
| F. Goal state reached or adjacent but not confirmed with the completing action | 2 | 4 | m0r0 045, ls20 042/042r, ls20 032 |
| G. Gated run (fewer calls) | none | 12 | all gated problem runs |

- **Never discovered a key action** is rare as a cause, although late first uses are common: in about 15 runs some
  action was first tried 25-95 minutes in, mostly a direction whose effect follows by symmetry. The late first uses
  that mattered were SPACE in g50t in the two gated runs (55 and 76 min; both unsolved). In m0r0 exp045, late probes of
  SPACE and MOUSE were wasted (both do nothing on level 1).
- **Resets and game-overs** were minor: 6 game-overs on level 1, 3 of them in tn36 (2 in exp042, 1 in exp045).

## 4. Harness changes, per failure mode (general; nothing game-specific)

| change | modes | observed failures it plausibly prevents or shortens | action cost | token/latency cost | risk |
|---|---|---|---|---|---|
| **R1. Object-level effect report after every action, including animations**: per action, list objects (colour+shape hash) that moved (from→to), appeared, vanished or changed colour; mark objects that moved during the animation but returned; detect a whole-scene shift (camera); keep edge-bar changes on one line | B, part of A and E | 8-12: g50t x5 (the clone appears and replays, verified), tn36 x2 (the piece moves then returns, verified), ls20 032 (key rotates on "+"), bp35 x3 (scroll), sk48 (which block moved), partly | 0 | ~100-200 tokens per turn (<1% of a 20.6k prompt); saves some of the 24% of calls that write diffs | noise when many objects change (cap it, collapse shifts); P9 (a cell-level final-board diff) exists but was tested only inside the harmful P4 arms |
| **R2. Rotation- and reflection-invariant shape keys on segmentation nodes, plus a level-start line listing objects equal up to rotation, reflection or colour** | E, part of A and F | 6-9: tr87 x5 (verified: 5/5 source glyphs paired, against 2/5 today), ls20 042/042r (the door glyph now matches the dial), m0r0 032 (twin pieces), dc22 partly (the avatar and the yellow are both 2x2, but so are 11 other objects) | 0 | ~50-150 tokens once per level | false matches on tiny shapes (limit to 5 px or more; cap the list) |
| R3. Idle guard: after N calls or M minutes on a level with no action, ask for a 1-3 action probe with a stated prediction (P15 is already built) | D | 3-5 (bp35 048, g50t 045, m0r0 032; tr87 only if R2 is missing) | 1-3 actions per trigger | small | spends actions; the idle is often a symptom (of perception in tr87, of gate starvation in g50t 045); P15 is untested outside P4 |
| R4. List the actions never tried on this level in each prompt after K calls without progress, as text only | B (discovery) | 2 (g50t 048/045) | 0 | ~20 tokens | little |
| R5. Automatic "try every action once" at level start | B (discovery) | about 2 | 4-6 actions per level for keyboard games; click games cannot be enumerated | small | on a 22-action baseline, 5 extra actions cut a perfect level from 1.0 to 0.66; the late first uses seen were mostly harmless directions; **not recommended** |
| R6. Batch stop-on-surprise: `action()` takes an optional expected result (for example a predicate on the next frame) and stops the batch at the first mismatch | C | 2-3 (sk48 045, ls20 042, tn36 042) | saves actions | small | relies on the model writing predictions |
| R7. Prompt prior: when the goal seems met and nothing happens, repeat the last move once, then try each other action kind once (census finding: 5 of 19 games gate the win check by action kind) | F | 2-3 (m0r0 045, ls20 042/042r) | 1-3 | ~40 tokens | small |
| R8. Do not starve level-1 games (drop P21, or give every game a floor share) | G | ~4-5 level-1 solves over 16 gated attempts (6/16 against 16/24) | 0 | none | scheduling, not harness perception; already being tested on the leaderboard |

## 5. Recommendation

1. **R1: an exact, object-level report of what each action changed, including inside animations.** This is the most
   common primary failure (B, 9 of 31, and involved in 16). Two offline engine checks show it would have stated the
   missing fact directly:
   - g50t: "after SPACE, a gray 5x5 appears and moves right while you press DOWN". Runs took 57 minutes to never to find
     this.
   - tn36: "the piece moved from row 13 to 21 during the run, then returned". exp042 dismissed the animation as "canned".

   It costs no actions and under 1% of prompt tokens. It also targets the calls the model spends writing diffs by hand
   (24% of hard level-1 calls). It follows the brief's first priority: never ask the model for what code can compute.
2. **R2: rotation- and reflection-invariant shape keys, with a level-start match line.** The evidence is narrower but
   verified: tr87 failed or stalled in 5 of 5 runs on exactly this, and the invariant key pairs all its glyphs from the
   first frame. Rotated copies are a common ARC device (tr87, ls20, tn36 also match rotation), and the cost is near zero.

Beyond these two, **do not carry the P21 gate onto a hard hidden set** without level-1 floors. This is the largest
measured effect in these data (6/16 against 16/24), although it is confounded by run and p = 0.11.

**What the data cannot decide.**
- Every estimate above is a retrospective plausibility judgement from single runs per arm, not an ablation.
- The model may ignore a report it is given. In tn36 exp042 it had `animation()` and misread it.
- Goal misreads (A, 11 primary) remain the largest family, and neither change fixes bp35's off-screen gem or sk48's
  mechanics-heavy planning.
- R3 (idle guard) could help or hurt: the data has no clean arm for it.

**Cheapest test before a GPU run.** Replay each hard game's recorded action sequence through the local engine with R1 and
R2 attached, and check whether the first report after the key action contains the fact the model later wrote down (the
g50t echo, the tn36 descent, the ls20 legend rotation, the tr87 pairing). Then run one ablation arm (fixes + R1 + R2,
stock serving) against exp042 and exp042r on level-1 solves of the 8 hard games plus the dev score.
