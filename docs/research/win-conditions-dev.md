# Win conditions of the 19 dev games, read from the game source (research census, 2026-09-23)

Three read-only agents traced each dev game's win check, lose checks and controls in
`environment_files/<game>/<version>/<game>.py` (the engine's `next_level()` / `win()` / `lose()` call sites), and
quoted the lines. The six validation games (cn04, g50t, lf52, r11l, sc25, sp80) were not opened. Group B also ran a
code-derived level-1 solution for each of its six games in the real engine (in memory, nothing written), which
confirms those readings.

This is a research instrument. The agent never reads game source: hidden games are only frames. The census is used
to decide which goal kinds the goal grammar must be able to express and which priors are safe; any prior taken from
it must be general (engine-level), not a per-game rule, or it overfits the public set.

## 1. Findings that hold for all 19 games

| Finding | Count | Consequence for the harness |
|---|---|---|
| The kind of win condition is the same on every level of a game; later levels add mechanics, not new goals | 19 of 19 | A goal inferred on level 1 is the strongest prior for every later level; relearn mechanics per level, not goals |
| The goal is drawn on screen from the first frame (target outline, marker, reference picture or strip, clue tiles) | 17 of 19 fully; bp35 partly (the gem starts off-screen, the camera scrolls); tr87 partly (the rule table is shown, the answer must be computed); m0r0 has no marker (the target is the other twin) | Goal inference is mostly perception plus a relation, not guessing: the frame names the target |
| Every game has a per-level action budget drawn as a HUD bar; running out loses the attempt (ls20: a life; tu93 also loses on enemy contact; bp35 also on spikes) | 19 of 19 | The bar is the volatile cell region the explorer must mask; its length is the attempt budget; a RESET refills it and costs one action |
| The win is checked only after certain action types (dc22 arrows only; cd82 pour or stamp; m0r0 piece moves; sk48 successful arrow moves; tr87 UP/DOWN) | at least 5 of 19 | A state can satisfy the goal without the level ending; a goal test must be tried after the right action kind |
| At least one level's human baseline exceeds the per-attempt budget (m0r0 L3/L5/L6, ka59 L7, tu93 L4, wa30 L9, tn36 L2, vc33 L4, re86 L7, s5i5 L5) | 8 of 19 | Baselines include failed attempts; those levels are lenient (the 115 cap is within reach) |
| A code-derived solution beats the human baseline on level 1 | 6 of 6 checked: ft09 4 vs 43, lp85 5 vs 17, sb26 9 vs 18, tr87 15 vs 54, ar25 15 vs 32, bp35 15 vs 21 | Knowing the rules and the goal is worth 1.4x to 11x fewer actions than a human; the headroom is real (see the oracle table in `road-to-100-v2.md`) |

## 2. Per game

Categories: AVATAR_REACH (a controlled sprite must reach a marker), PLACE_IN_SLOTS (pieces must sit in visible
targets), MATCH_REFERENCE (an editable region must equal a shown reference), ALIGN, COUNT, ALL_STATE (every tile
satisfies a local constraint), OTHER. A game can carry two.

| Game | Levels | Category | Win condition (engine check) | Controls | Budget per attempt |
|---|---:|---|---|---|---|
| ar25 | 8 | PLACE_IN_SLOTS + OTHER | every target dot is covered by a movable shape or one of its mirror images (reflections drawn live in colour 4) | click selects a shape or mirror, ACTION5 cycles selection, 1-4 move one cell, 7 undo | 64 to 320 moves (clicks and undo free) |
| bp35 | 9 | AVATAR_REACH | the character walks into or falls onto the gem | LEFT/RIGHT, click edits terrain (break, toggle, flip gravity, spread), undo; gravity pulls up | 64 or 128 actions; spikes; a rising hazard row on levels 1-3 |
| cd82 | 6 | MATCH_REFERENCE | the 10x10 canvas equals the reference picture pixel for pixel, both diagonals ignored | click a palette colour, arrows move a bucket around 8 slots, ACTION5 pours a half or a triangle; later a notch stamp | 99 actions |
| dc22 | 6 | AVATAR_REACH | the 2x2 avatar's top-left equals the 2x2 goal marker's (checked after arrow keys only) | arrows move 2 px on walkable tiles; letter buttons toggle bridges; pads teleport; later a crane carries bridges | 128 to 1024 steps; a fall costs 20 |
| ft09 | 6 | MATCH_REFERENCE + ALL_STATE | every clue tile is satisfied (a 3x3 miniature says which of its 8 neighbours must be its centre colour) | click cycles a tile's colour; later Lights-Out style linked tiles | 32 to 128 clicks |
| ka59 | 7 | PLACE_IN_SLOTS | every target outline holds a player block of exactly matching size 1 px inside; every plus outline holds a passive plus block | click selects a block, arrows move it 3 px; bumping kicks the other block 15 px; later timed bombs | 100 to 200 steps |
| lp85 | 8 | PLACE_IN_SLOTS | every 4x4 bracket holds a matching 2x2 piece | clicks on L/R buttons rotate every sprite on a ring by one position | 13 to 150 button clicks |
| ls20 | 7 | AVATAR_REACH + MATCH_REFERENCE | the avatar steps onto every socket while the key's shape, colour and rotation match that socket; used sockets vanish; all used = win | arrows move one 5 px cell; changer tiles cycle the key; push pads; refills; fog on level 7 | a 42-cell bar, 1 or 2 per move; 3 lives |
| m0r0 | 6 | AVATAR_REACH (twin variant) | the two mirrored pieces meet and merge; no unmerged piece left | arrows move both pieces, left/right mirrored; click selects a movable block; plates open doors | 150 actions; hazards reset positions |
| re86 | 8 | MATCH_REFERENCE + PLACE_IN_SLOTS | every target dot is covered by a stroke of its colour on the canvas | ACTION5 selects the next piece, arrows move it 3 px; walls reshape; paint pads recolour | 100 to 400 steps |
| s5i5 | 8 | PLACE_IN_SLOTS | every diamond target has a dot at exactly its x, y | clicks lengthen or shorten all segments of a colour; later rotate buttons | 50 to 200 clicks |
| sb26 | 8 | MATCH_REFERENCE + PLACE_IN_SLOTS | the program's emitted colours equal the reference strip, in order | click a tile then a slot to place or swap; ACTION5 runs; undo | 64 energy |
| sk48 | 8 | MATCH_REFERENCE | each play arm carries the same colour sequence as its reference arm | click selects an arm; arrows extend, retract, slide on rails; undo | 196 moves |
| su15 | 9 | PLACE_IN_SLOTS + COUNT | exactly n pieces of merge level k (and, later, enemies) are inside the goal zones | click pulls everything within radius 8; equal pieces merge; undo free | 32 to 48 steps with penalties |
| tn36 | 7 | AVATAR_REACH + MATCH_REFERENCE | after the program runs, the piece matches the target outline's x, y, scale, rotation and colour | clicks toggle instruction bits, then Run; a demo panel shows instructions | a 61-pixel bar, 1 per click |
| tr87 | 6 | MATCH_REFERENCE + OTHER | the answer row equals the source row translated through the rule table | LEFT/RIGHT move a cursor, UP/DOWN cycle the symbol; later levels edit rules | 128 or 256 |
| tu93 | 9 | AVATAR_REACH | the player is on the exit at the end of the enemies' turn | arrows move one maze node; moving onto an enemy squashes it | 20 to 60 steps; enemy contact |
| vc33 | 7 | ALIGN + PLACE_IN_SLOTS | every floater sits at the height of a marker of its colour on a bordering wall | clicks on valves move liquid between columns; later gates swap floaters | 50 to 200 clicks |
| wa30 | 9 | PLACE_IN_SLOTS | every box is inside a goal zone and not held | arrows move and turn; ACTION5 grabs, releases or destroys | 70 to 200 steps |

Totals (a game can count twice): PLACE_IN_SLOTS 9, MATCH_REFERENCE 8, AVATAR_REACH 6, ALIGN 1, COUNT 1,
ALL_STATE 1, OTHER 2, COLLECT_ALL 0.

## 3. What the goal grammar needs (gap analysis against `arc3/dsl.py` goal kinds)

The existing kinds (`goal_predicates`, 2026-09-17) are existential over colour classes: `same_box(a, b)` holds when
*some* a-entity has the box of *some* b-entity; `inside`, `overlap`, `touch`, `shape_matches` likewise; plus
`none_left`, `count`, `aligned`, `vanish_shape` and the avatar-relative kinds.

1. **Universal quantifier.** Every multi-target game (ka59, lp85, s5i5, wa30, re86, vc33, ar25, ls20, su15) needs
   *every* target satisfied. Level 1 usually shows one target, where "some" and "every" agree, so level 1 cannot
   tell them apart; on level 2 an existential goal is met by the first filled slot and a planner stops early. The
   census gives the tie-break: prefer the universal form when level 1 cannot separate them (9 of 9 games).
2. **Relations the grammar lacks:** same top-left (dc22, s5i5, tu93: `same_box` covers equal-size sprites only),
   "covered by pixels of" (ar25 dots under a shape or its reflection, re86 dots under a stroke), inside with a
   1 px margin and equal size (ka59), and region equality under a mask (cd82 canvas vs reference without diagonals,
   sk48 arm vs reference arm, sb26 and tr87 sequences).
3. **Out of reach for a fixed grammar:** tr87 (the answer is a computed translation), sb26 (program semantics),
   ft09 (a local constraint read from clue miniatures), su15 (counts by merge level). These need a model that
   reads the picture and writes the predicate as code; they are the case for the LLM-written goal predicate.

## 4. Priors this census supports (general, not per game)

- Goal kind is constant within a game: infer it on level 1 and carry it; revise only on contradiction.
- Targets are drawn: sprites that never move and have a partner of matching shape, size or colour are target
  candidates; the relation is one of a handful (same position, inside, covers, equals).
- The budget bar is on an edge row or column, shrinks by a fixed amount per action and refills on RESET; it is not
  part of the goal and should be masked from state keys and goal tests.
- Win checks can be gated by action kind: when the goal looks satisfied and nothing happens, try the other action
  kinds once before doubting the goal.
