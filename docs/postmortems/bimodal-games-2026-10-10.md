Summary: each bimodal game splits on one level; in 7 of the 8 losing game-runs the agent never clicked (or clicked 48 minutes late) an object it had assumed inert (tn36's demo boxes, vc33's orange gate, sp80's red bars, cn04's "sockets"), while every run that won the level clicked it first; a 10-minute "not yet clicked on this level" line plus a "new colour appeared" notice would have named that object in all 7.

# Bimodal public games (tn36, vc33, sp80, cn04): what separates winning and losing runs of one configuration

Analysis of 2026-10-10. CPU only. No GPU, no Kaggle, no repo code changed.

## Sources and method

- **Runs.** exp073, exp073b, exp075 and exp083 (`runs/exp073-dprime-reap448-r14-full`, `runs/exp073b-dprime-reap448-r14-accept05-full`,
  `runs/exp075-dprime-r14a05-sandbox-full`, `runs/exp083-dprime-r14a05-arcmap-draft-full`): D′ with Franzen's harness.
- **Transcripts.** The per-turn solver_analysis HTML of each game (session scratchpad `sa/<run>/`). "Turn bN" is the N-th
  analyzer turn of that game, counted from 0. Quotes are the model's reasoning, copied verbatim.
- **Replay.** Every recorded action (`kernel-output/benchmark.json`) replayed through the local engine on the CPU; per-level
  action counts match `report.json` for all 16 game-runs. Objects are 4-connected single-colour components without
  backgrounds, panels (over 300 px, or 48+ wide or tall) and edge HUD bars. A click counts against the object under it, else
  the smallest object whose box contains it.
- **Units and labels.** "Min in" is minutes since the level started (benchmark wall clock, parked time included). Tokens are
  generated tokens on that level. Scripts are in `pm/` in the session scratchpad, not in the repo. Plain statements come from
  a command output; "(inference)" marks a judgement.

## Where the runs split (measured)

| game | level where runs split | won it: actions / min / tokens | lost it: actions / min / tokens / budget deaths |
|---|---|---|---|
| tn36 | L2 (61 clicks per attempt; human 72) | exp073b 50 / 21.6 / 70k; exp083 98 / 96.3 / 150k | exp073 85 / 75.6 / 119k / 1; exp075 107 / 75.6 / 181k / 1 |
| vc33 | L4 (attempts died on the 51st click; human 61) | exp073b 47 / 9.9 / 35k; exp075 100 / 21.0 / 73k / 1 | exp073 102 / 109.8 / 240k / 2; exp083 130 / 71.9 / 211k / 2 |
| sp80 | L2 (human 58) | exp073b 15 / 20.9 / 77k | exp073 81 / 66.8 / 170k / 2; exp075 72 / 71.1 / 256k / 3; exp083 104 / 79.7 / 281k / 2 |
| cn04 | L2 (human 54) | exp073 59 / 26.4 / 89k; exp073b 53 / 10.4 / 41k; exp083 90 / 75.5 / 188k | exp075 61 / 105.0 / 236k / 0 |

Earlier levels went alike in all runs. After winning these levels the runs reached: tn36 7 of 7 (exp073b) and 5 (exp083);
vc33 7 of 7 (exp073b, exp075); sp80 4 (exp073b); cn04 6 of 6 (exp073) and 5 (exp073b). exp083's cn04 then ran out of time
on L3, which it entered with 42 minutes left after a 75-minute L2.

## tn36 L2

**What the agent believed.** exp073: the left panel, its two boxes included, is a static picture. exp075: each
bar-and-stem pair in a row is one instruction, as on level 1.

**What happened.**
- **Mechanic** (winning transcript and the census). The right switch grid is a program read column by column; the run
  cursor visits 4 positions, and each column's bars and stems make one move. The left panel is a demo machine whose two
  boxes load and run example programs. Box B's program, row-1 bars and row-3 stems in all 4 columns, moves the demo piece
  up 4.
- **exp073b (won).** It clicked both boxes at L2 actions 2-3 (1.7 min in, 6.7k tokens). b11: "Let me try clicking the
  devices and see what happens. That's the most direct way to learn." About 35 single-part probes followed (bar = left,
  stem = right, both = down); none gave "up". b28 (22.4 min): "Let me now think about the LEFT machine as the model to
  copy, since the left machine clearly produced 'up 4' with the pattern {row1 bars ×4, row3 stems ×4}." It tested one
  column (b30, 25.0 min): "The pattern {row1 bar c1, row3 stem c1} produced: the plug moved UP 1". It won 7 actions later
  (L2 action 50, 70k tokens).
- **exp073 (lost).** At b10 (4.5 min) one click on a left-panel stem (L2 action 1) did nothing: "So the left panel icons
  are NOT clickable ... maybe the left panel is a 'display'". b11 planned "probe clicks on (58,8) yellow in box1, (55,22)
  yellow in box2 ... 3 actions", then clicked the run ball. It never clicked a box in 85 actions; it decoded single parts
  (b19: bar2 → nothing, stem2 → "scale ×2", bar3 → "transform"). It ran out of budget at 32.5 min, was parked 36.7 min, and
  ended with L2 unsolved.
- **exp075 (lost).** It clicked the boxes at L2 actions 7-9 (8.5-10.8 min) and saw the demo program change. It read the
  program with level 1's unit. b13 (9.4 min): "the left program = row1: 4×(B,w)=Left; row3: 4×(w,B)=Right. Net: 4 lefts + 4
  rights = 0? Hmm." From b16 (12.7 min) it pursued "the position in the program grid determines the direction". It never
  ran the demo's pattern on its own panel (107 actions).

**Which assumption was wrong, and what showed it.**
- exp073 took one no-op on one part as proof that the whole panel was inert. Its own b9 (4.1 min) had seen the link: "left
  panel ← tube → two boxes; right panel ← white tube → blue ball", and it had just used that ball to run the right panel.
- exp075's row code had no "up" (b11, 8.0 min: "all four combinations are used! ... So where's Up?"). Its reading of the
  demo, 4 left plus 4 right, contradicted the demo piece's visible move up.

**Cheaper test.** exp073: click each box once (2 actions). exp075: copy one column of the demo pattern onto its own panel
and run it (3 actions), as exp073b did at b29-b30.

**General?** exp073's miss is the cross-game pattern below. exp075's miss (decoding a worked example with the old unit)
fits only games that show an example; I count it as not general.

## vc33 L4

**What the agent believed.** The losing runs saw a wall segment turn orange and read it as a passive valve for the water.

**What happened.**
- **Mechanic** (exp073b, exp075). A light-grey wall segment turns orange, with two white marks, while the water on both
  sides stands at its bottom edge. Clicking the orange segment moves the key through the wall. It turns grey again when
  the levels change.
- **exp073b (won).** b27 (16.4 min) listed the key heights not yet tried and equalized the two sides. The gate turned
  orange after L4 action 32 (17.2 min). b28: "Let me click the orange bar." b29 (17.8 min): "EXCELLENT! The key MOVED
  HORIZONTALLY!" It won 14 actions later (action 47, 35k tokens).
- **exp073 (lost).** The gate turned orange after L4 action 27 (28.8 min). b35: "MAJOR EVENT! ... an ORANGE block". Its
  readings: b38 "the plug does not get carried through the gate"; b42 "Maybe the orange gate + strips is a 'siphon'"; b60
  (55.9 min) "So the gate's only effect is visual (orange + slits)." It clicked that segment only while grey (12.5 and
  91.9 min), and both clicks were no-ops. 102 actions, 2 budget deaths, 240k tokens.
- **exp083 (lost).** The gate turned orange after L4 action 25 (17.4 min). b28 (20.2 min): "Orange likely means 'the window
  is being used / water passing through'. So the window IS a channel". It never clicked it (130 actions).
- **exp075 (won late).** The gate turned orange after L4 action 24 (13.0 min); the budget ran out at action 51 (19.0 min).
  Its first orange click came at action 80 (28.1 min): "BREAKTHROUGH! Clicking the orange gate moved the KEY!" It won at
  action 100: the level scored (61/100)² = 0.37, against the cap for exp073b.

**Which assumption was wrong, and what showed it.** An object that changes colour as a side effect is only an indicator.
The change itself, at the moment it appeared, said otherwise. All four runs saw it.

**Cheaper test.** One click on the orange segment (1 action).

**General?** Yes. An object that changes colour after some other action is a candidate new control.

## sp80 L2

**What the agent believed.** The losing runs believed only the paddle moves; the red bars are fixed obstacles.

**What happened.**
- **Mechanic** (exp073b). The arrows move the selected (blue) bar, and SPACE releases ink, which rises, splits on bars and
  fills cups. Clicking a red bar selects it: it turns blue and the old one turns red. To win, all three cups must fill with
  no ink reaching the ceiling.
- **exp073b (won).** One launch (L2 action 2, 1.6 min), then 10 turns without an action (b6-b15, 10.8 min) writing an
  ink-flow simulator, checked against the launch: "The observed run is reproduced". Its search over paddle positions
  printed "n solutions 0" (b15, 11.7 min): "My model says no static paddle position works." b16 (12.4 min): "Idea: maybe
  the paddle is not the only movable object ... maybe MOUSE clicks can move the red bars ... Let me test clicking on a red
  bar." The click (L2 action 3, 11.6 min in, 44k tokens) selected the bar. Then a 3-bar search, one failed fill-all attempt
  and a "no waste" condition; it won at action 15 (77k tokens).
- **exp073, exp075, exp083 (lost).**
  - They tested their stream model with real launches (11, 15 and 13) and hit the budget 2-3 times each.
  - exp073 never used MOUSE on L2. exp083 clicked once, on the ink block. exp075's first click that selected a bar came at
    L2 action 56 (48.5 min in), after two budget deaths; it never solved the level.
  - exp075 wrote the probe and put it off. b12 (5.7 min in): "Maybe MOUSE clicks select other objects? Let me test MOUSE on
    the cannon. Let me test the timer first (cheap, no action) and then decide."

**Which assumption was wrong, and what showed it.**
- Replay: a launch with the paddle in the top bar row moved the blue (selected) colour to red bar B in all three losing
  runs: L2 actions 50 and 63 (exp073), 16 and 31 (exp075), 21, 46 and 96 (exp083).
- Two runs saw it about 12 min into L2 and explained it away. exp075 b18: "the blue bar and the red bar 2 SWAPPED colors!
  ... the bar becomes 'used' (red), and the next unused bar (red) becomes the new blue bar". exp083 b23: "They swapped?!"
- exp073 also misread an occlusion as destruction. b11: "the red block DISAPPEARED! The paddle absorbed/destroyed red B".
  In the replay the red pixels came back as the paddle moved off.

**Cheaper test.** One MOUSE click on a red bar (1 action), after the first launch that did not win or right after the swap.

**General?** Yes. (Inference) The winner questioned the controls because a checked simulator returned zero solutions; the
losers got "no win" from real launches and blamed the stream model instead.

## cn04 L2

**What the agent believed.** exp075 labelled two of the four objects "sockets" and treated them as fixed.

**What happened.**
- **Mechanic** (exp073). Four objects carry red 3×3 blocks. A click selects any object, which can then be moved and
  rotated. A red block on a red block turns grey (connected). The level is won when every red block is paired.
- **exp073 (won).** It connected both "plugs" (grey), and the level did not end. b15 (12.1 min): "Cheap probes (each 1
  step, 22 left): 1. Click on the yellow socket's body". b16 (14.4 min, L2 action 31, 38k tokens): "Big discovery: clicking
  the YELLOW SOCKET made it SELECTABLE ... So the sockets are also movable objects!" b17 (16.1 min): "the win could be: ALL
  red blocks are connected in pairs!" A search over rotations and offsets followed; it won at L2 action 59 (89k tokens).
  exp073b clicked the green object 8.1 min in and won at 10.4 min.
- **exp075 (lost).** b7: "2 plugs (W, Y) and 2 sockets (N green, b blue)"; it clicked only W and Y. b16 (15.7 min): "(they're
  part of the socket, static)". With N and b fixed, full pairing could not be reached. b27 (44.8 min): "So 'all the prongs
  touched' is impossible → the win condition is not that." It also worked from a wrong tip list until b33 (52.7 min):
  "That's the bug — the tip set was wrong." 105 min in all, 46.8 of them parked.
- **exp083 (won late).** It first clicked N 72 min into L2 and won 3.5 min later.

**Which assumption was wrong, and what showed it.** "Socket" was the model's own label: exp073 called the yellow object a
socket and the blue one a plug, and exp075 had them the other way round. By b15 (14.2 min) exp075 knew a click on Y had
selected it ("the active plug is white, the inactive is purple"). Nothing but the label set N and b apart.

**Cheaper test.** One click on N or on b (1 action). **General?** Yes.

## What did not separate the runs (measured)

- **Slot waits.** sp80 waited 16-18 min for its first slot in the three losing runs and not in exp073b. The losers still
  had 67-80 min on L2, while the winner used 21.
- **Parking.** Losing runs were parked 0-47 min on the stuck level (gaps over 5 min): tn36 36.7 and 25.1; vc33 38.8 and
  11.6; sp80 16.1, 6.1 and 0; cn04 46.8. Parking started 13-48 min into the level, and winners were never parked on these
  levels: it is a consequence, not a cause.
- **Other.** The longest action-free stretch belongs to the sp80 winner (10.8 min, building its simulator). Tool errors ran
  at 0-0.2 per call on both sides. The first budget death came 11-40 min into the stuck level, after the stall had begun.

## Cross-game pattern (4 of 4 games)

The object a losing run assumed inert, and when it was clicked:

| losing game-run | object assumed inert | first click on it | named by I1 at 10 min? | named by I2? |
|---|---|---|---|---|
| tn36 exp073 | the demo's 2 boxes | never | yes (ring ranked 1st) | no event |
| vc33 exp073 | orange gate | never while orange | no (grey at each firing) | yes, 28.8 min |
| vc33 exp083 | orange gate | never | no | yes, 17.4 min |
| sp80 exp073 | red bars | never; MOUSE unused on L2 | yes, plus "MOUSE unused" | no |
| sp80 exp075 | red bars | 48.5 min in | yes, plus "MOUSE unused" | no |
| sp80 exp083 | red bars | never | yes, plus "MOUSE unused" | no |
| cn04 exp075 | N and b "sockets" | never | yes (ranked 2nd and 4th) | no |
| tn36 exp075 | (instruction unit, not an object) | n/a | no | no |

- **The click comes first.** Every run that won one of these levels clicked the object before winning: tn36 1.7-1.9 min
  in, vc33 8.5 and 20.0, sp80 11.6, cn04 8.1-72.
- **The click is not always enough.** vc33 and cn04 were won 1-16 min after the click in 5 of 5 cases. tn36 and sp80 also
  needed a rule insight: exp073b won them 20 and 9 min later, exp083 won tn36 94 min later, and exp075 solved neither.
- **The probe was often already planned.** Three losing runs wrote it down and put it off: tn36 exp073 b11, sp80 exp075 b12
  and cn04 exp075 b16.
- **It is the common failure mode.** `levels2plus-exp054.md` counts "new element/mechanic never found" as the top stuck mode
  (15 of 48), and `exp073b-failure-analysis.md` counts 8 of 17.

## Proposed interventions (each adds zero actions)

### I1. "Not yet clicked on this level" line

- **Trigger.** A level has run 10 minutes without a level-up. It fires again every 10 minutes and after each GAME OVER.
- **Behaviour.** The next user prompt gets one block:
  - the valid action types not used on this level;
  - up to 6 object kinds on the board that no click on this level has hit, each with colour, size and box. A kind is one
    colour and shape, rotation ignored; repeated tiles collapse into one entry, and HUD bars and panels are left out. Kinds
    missing from the previous level's first board come first, then the kinds with the fewest copies;
  - the sentence "Each is a one-action test of whether it is interactive."
- **Prior work.** ours-06 already records every clicked cell per level, so I1 adds a kind table and the line. R3 in
  `levels2plus-exp054.md` proposed the same list; I found no record that it was built.
- **Counts** (replay; 16 game-runs, 67 levels; the three sp80 L1 levels whose clock included the slot wait are excluded):
  - It fires on 11 of 12 unsolved levels, and on 21 of 55 solved levels (57 firings).
  - It names the object on 5 of the 8 losing critical levels (table above).
  - In the two slow wins it names the object long before the run's own click: 62 min earlier in cn04 exp083, and 10 min
    earlier in vc33 exp075 (the gate happened to be orange at minute 10).
  - In the fast winners it fires 1-2 min before their own click (sp80, cn04 exp073), after it (tn36, cn04 exp073b), or not
    at all (vc33 exp073b, 9.9 min).
- **Risk.** It invites probe clicks on solved levels that ran past 10 minutes; each extra action costs about 2-4% of a
  50-100-action level's score. If half the 57 firings draw one probe, that is about 28 actions, an estimated 0.6-1.1
  level-equivalents across the 16 game-runs. Decorations get listed too (tn36's checker tiles), and I chose the ordering
  rule after seeing these cases.

### I2. "New colour" notice

- **Trigger.** After an action, an object that was not the click target shows a colour that has not appeared earlier in
  this game. Level switches are excluded.
- **Behaviour.** The next user prompt gets one line: "New: the <old colour> object at rows a-b, cols c-d is now <colour>
  (first time in this game). It has not been clicked since."
- **Counts.** 12 firings in 16 game-runs, none in tn36 or sp80.
  - vc33: 4, one per run, all on the L4 gate. The winner clicked within 1 action; exp075 clicked after 56 actions and a
    budget death; exp073 and exp083 never did.
  - cn04: 8, two per run (grey connected tips; the purple deselected object).
- **Rejected variant.** "Any side-effect recolour" would also catch sp80's selection swap, but it fires on 1-22 actions per
  affected level in all four games.
- **Risk.** The cn04 noise: 8 lines, each costing at most one click. A click on a grey tip only selects its plug.

### Estimated levels recovered (inference)

- **Upper bound.** Suppose every named object were clicked within 3 actions and the level then went as in the fastest
  winner. Across the 16 game-runs that recovers vc33 2, cn04 1, sp80 up to 3 and tn36 1 critical levels, plus up to about
  20 follow-on levels in the time those runs had left.
- **Expected.** Suppose the model acts on a line in a quarter to a half of firings (unmeasured, though the three deferred
  probes suggest a nudge can tip it). That gives about 4-9 levels across these 16 game-runs, or 1-2 per full run on these
  four games. Most would come from vc33 (I2) and cn04 (I1), where the click alone was enough.
- **Floor.** 0, if the model ignores prompt text. Facts in P23's reports were used in 61-75% of turns
  (`levels2plus-exp054.md`).

### Checks before a GPU run

1. **CPU replay.** Run the same replay over all 25 games of exp073b, exp075 and exp083, as `scripts/effect_table_replay.py`
   does. Confirm the lines name the 7 objects at the times given here, and measure list lengths and firings on solved levels.
2. **GPU pair.** Run one D′ pair, base against base + I1 + I2. Read the mechanism from the transcripts first: was the named
   object clicked within 3 actions? A gain of about 1 level is inside the score noise (lesson 0018).

## Limits

- One analyst, 16 game-runs, one configuration family.
- Object identity in the replay is approximate: a selection recolour counts as clicked; a gate that changes colour does not.
- The model's response to such lines is unmeasured. P14, an untried-actions line in the old TAAF harness, ran only in
  exp-039, confounded with P4.
