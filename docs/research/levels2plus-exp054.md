# Levels 2+ in the exp-054 pair: where they stall, and which harness change would add levels

Research note, 2026-09-26 (subagent study, read-only; no GPU). Runs: `runs/exp054-fix-kv775-obj` (A: 12.87, 49 levels)
and `runs/exp054r-fix-kv775-obj-r2` (B: 10.70, 46 levels). Sources: `summary.json`, `kernel-output/benchmark.json`, the
50 transcripts, and the game source under `environment_files/`. Scripts and per-game notes: `levels2plus-exp054/`.
"Calls" means model responses; minutes are counted per game from its first turn; level score = min((baseline/actions)^2,
1.15). The "stuck level" is the level a game was on when the run ended.

## Bottom line

1. **The stuck level takes half the budget:** 54% of all game-minutes and 50% of calls (1,616 of 3,236). Level 1 took
   26% (A) and 30% (B) of the minutes; solved later levels 20-24%.
2. **Discovery, not efficiency, is where levels are lost.** Primary mode of the 48 stuck level-2+ levels: 15 a new
   element or mechanic never found or decoded; 11 out of time while progressing; 7 a misread rule or map; 6 started too
   late; 5 an old goal reused after the level changed it (its count or scope grew); 4 lost or wrong knowledge of the
   previous level. Solved later levels are efficient: median 0.73x baseline actions, mean level score 0.98; bringing every
   solved level to baseline would add only +0.57 (A) and +1.10 (B).
3. **P23's reports are used on later levels:** the reasoning explicitly refers to 28-31% of them and uses content from
   61-63% (75% on level 1). The failures are in interpretation: in sb26 the report said "W 3x3 ... moved as far as down 14
   ... back in place" and the model read it as the frames "merging".
4. **20% more calls would plausibly add about 3 levels per run** (re86 L4 and s5i5 L3 in both runs, dc22 L3 in A,
   su15 L2 in B had a correct plan when time ran out): about +1.0 to +1.4 on the mean. An extrapolation.
5. **Recommendation R1:** at each level start, pin an exact record of how the previous level was won (extends P16 with
   P23/P24-style object facts). Then R2 (a move, death and budget ledger), R3 (objects never interacted with on this
   level), R4 (a run-clock line).

## Stuck levels by game

Cells: stuck level with minutes / calls / actions÷baseline; "(k)" = levels solved before it; GO = game-overs.

| game | run A | run B | what the stuck level required (source) | mode A; B |
|---|---|---|---|---|
| ar25 | L6 (5): 27/13/19÷159 | L6 (5): 22/10/25÷159 | set the mirror axes, then place pieces so they and their mirrors cover the targets | E; E |
| bp35 | L2: 85/37/16÷48, 2 GO | L2 (L1 took 96 min): 37/17/11÷48, 4 GO | rise through the open right shaft to the off-screen gem; the 3 purple/yellow objects are spikes | B; B |
| cd82 | L3 (2): 63/27/52÷41 | L3 (2): 70/29/160÷41 | L3 adds a notch stamp to the 8 orbit stamps | B; B |
| cn04 | L2: 118/49/15÷54 | L2: 85/41/77÷54 | every red peg of all 4 pieces overlaps another piece's peg (L1 had 2 pieces) | A; D |
| dc22 | L3 (2): 36/16/36÷67 | L2: 95/43/100÷102 | toggle bridges so the avatar reaches the yellow marker | E; D |
| ft09 | L5 (4): 85/40/74÷65 | L5 (4): 65/29/10÷65 | 7 new clue sprite kinds and a new palette on a 32-cell grid | B; B |
| g50t | L2: 76/32/47÷175 | **L1**: 132/63/68÷78 | SPACE rewinds the player; a gray clone replays the path and holds the emitter open | B; B |
| ka59 | L2: 105/50/92÷109 | L3: 1/0 (L2 solved at 131 min) | bump a ring so it slides 15 px through the purple wall | C; F |
| lf52 | L2: 113/52/79÷81 | L2: 92/42/96÷81 | carry a peg on the pipe's pad and merge down to 1 peg (inferred) | B; B |
| lp85 | L5 (4): 8/3/2÷41 | L6 (5): 30/14/15÷60 | ring rotations on new 8-cycle stars | F; E |
| ls20 | L2: 114/53/174÷123, 1 GO | L2: 120/54/224÷123, 1 GO | rotator 3 times (L1 needed 1), 2 bar cells per move, then the socket | A; A |
| m0r0 | L2: 99/43/38÷111 | L2: 70/32/41÷111 | the mirrored pair must meet and merge; the red checker resets positions | C; D |
| r11l | L2: 97/47/69÷33, 1 GO | L4 (3): 44/21/19÷26 | a body re-centres on its leg endpoints; each ring needs a body with its colour set | D; C |
| re86 | L4 (3): 36/16/58÷108 | L4 (3): 37/15/35÷108 | paint pads: a shape takes the colour of the last pad it sweeps | E; E |
| s5i5 | L3 (2): 66/28/50÷106 | L3 (2): 31/13/64÷106 | more arms, knobs and split pad buttons | E; E |
| sb26 | L2: 129/63/79÷28 | L2: 114/52/66÷28 | red slot 3 is a call into the green frame | B; B |
| sc25 | L4 (3): 33/16/25÷83 | **L1**: 132/63/84÷36, 2 GO | L4: a pad pattern launches the piece at a plug (inferred) | E; A |
| sk48 | L2 (L1 took 104 min): 28/13/26÷177 | L2: 108/47/114÷177 | the arm's blocks, read from the head, must be R,O,b,N | F; D |
| sp80 | L2 (L1 took 80 min): 53/25/6÷58 | L2 (L1 took 85 min): 48/25/34÷58, 1 GO | a click selects which bar moves; liquid must reach all 3 cups | A; A |
| su15 | L2: 116/50/46÷42, 1 GO | L2: 103/45/39÷42, 1 GO | merge up to tier 3 (L1's HUD showed tier 2) and bring it into the blob | B; E |
| tn36 | L2 (L1 took 102 min): 30/12/16÷72 | L2: 98/48/211÷72, 3 GO | two program panels with new instruction kinds | F; B |
| tr87 | L6 (5): 4/2/0÷146 | L4 (3): 5/2/0÷45 | censored | F; F |
| tu93 | L3 (2): 94/41/88÷34, 12 GO | L3 (2): 59/25/25÷34, 2 GO | reach the exit; enemies capture | D; D |
| vc33 | L4 (3): 90/43/103÷61, 2 GO | L4 (3): 72/32/51÷61, 1 GO | clicking an open gate moves the floater to the next compartment | B; E |
| wa30 | L2: 92/39/209÷119, 2 GO | L2: 95/44/152÷119, 2 GO | the orange helper carries boxes alone but too slowly; the player must also carry | B; C |

17 of the 50 stuck levels ended at or above the baseline action count; 39 game-overs across 17 stuck levels (12 in tu93 A).

## Time budget

| run | L1 (solved) | L2+ (solved) | stuck level | total |
|---|---|---|---|---|
| A | 848 min (26%), 505 calls | 660 min (20%), 294 calls | 1,794 min (54%), 810 calls | 3,303 min, 1,609 calls |
| B | 738 min (22%), 457 calls | 801 min (24%), 364 calls | 1,764 min (53%), 806 calls | 3,303 min, 1,627 calls |

A solved level 2+ took a median 25.4 min and 11 calls; a solved level 1, 26.6 min and 18 calls. Level 1 took over 55% of
the run in 7 game-runs (sk48 A, tn36 A, sp80 A/B, bp35 B, and g50t B and sc25 B, which never solved it).

## Failure modes (48 stuck levels at L2+)

| mode | primary | any role |
|---|---|---|
| B. New element or mechanic never found or decoded | 15 | 21 |
| E. Out of time while progressing | 11 | 16 |
| D. Rule or map misread; plans kept failing | 7 | 11 |
| F. Started too late (previous level ran past 100 min, or under 10 min left) | 6 | 6 |
| A. Goal changed or grew; the previous goal or recipe was reused | 5 | 13 |
| C. Previous-level knowledge lost or wrong | 4 | 7 |
| G. Action-free stretch of 20 min or more (a symptom) | none | 19 |
| H. False belief that the level was complete | none | 2 |

Mode A levels kept the goal kind but changed its count or scope (one rotation became three; one pair became all pegs;
one cup became all cups; tier 2 became tier 3), consistent with lesson 0016. In about 4 mode-B levels the key element was
never acted on (sp80, tn36 B, wa30 A); in the rest its effect was reported and misread (sb26, g50t, ft09, cd82).

## Evidence

1. sb26 A (B): P23 at +18 min: "SPACE: ... W 3x3 (5 px) at (17,17) moved as far as down 14 (frame 84) and was back in
   place at the end". The model at +122: "the SPACE animation showed the red frame's 4 corner brackets moving DOWN 14 px
   (merging with the green frame)". In the source, the cursor is entering the called green frame.
2. ls20 A (A): the L1 note said "the bottom-left legend glyph rotated 90° CW to match the box glyph". On L2 at +108: "L1
   solved: stepping exactly on the plus ... = 'key collected'. Then attempting to move into the box won." L2 needs three
   rotator visits; one visit was made. P24 cannot pair the legend (drawn at 2x) with the socket glyph (2 components).
3. wa30 B (C): the L1 note said "SPACE grab, RIGHT×3, SPACE drop". On L2 at +23: "Actually in level 1 the boxes never
   moved at all!"; 152 actions against 119 and 2 game-overs followed.
4. tu93 A (D): after 5 deaths (12 in all), at +49: "my coordinate origin was off by one the whole time; new verified map".
5. re86 B (E): at +32 "BFS found X plan (21) and plus plan (16). Executing X plan first"; at +36, as the run ended:
   "nothing consumed by the invalid letter-coded batch". The last 15 minutes mention the run clock in 2 of 252 turns.

## P23 usage

| where | reports | reasoning refers to the report | uses its content (upper bound) |
|---|---|---|---|
| L1 | 469 | 37% | 75% |
| solved L2+ | 249 | 31% | 63% |
| stuck levels | 605 | 28% | 61% |

## Ranked harness changes (all exact facts, 0 actions)

| # | change | failures it plausibly fixes | cost | risk |
|---|---|---|---|---|
| R1 | Pinned previous-level win record at each level start: last actions (P16), P23's net object report over the level's last 3 actions on the frame before the switch, the winning board's shape relations (incl. a 2x/3x scale check), P14's new/removed kinds | C: ka59 A, m0r0 A, wa30 B; contributing g50t A, cd82 A, sp80 A; A in part: ls20, cn04 A, su15 A. About 6-9 of 48 | ~250-400 tokens on the next level's prompts | last-action lists invite spurious rules (r11l B); must be exact (lesson 0024) |
| R2 | Move, death and budget ledger per level | D: tu93, dc22 B, m0r0 B; deaths bp35; budget ls20 A, wa30, vc33 B, r11l A. About 4-7 | ~100-150 tokens per turn | needs generic detection of the controlled object and the bar |
| R3 | Per-level list of objects never acted on and actions untried on this level | B where the element was never touched: sp80, tn36 B, wa30 A. About 3-5 | ~60-100 tokens; may draw 1-2 probes | lists decorations; misses state-dependent elements |
| R4 | Run-clock line (time left, calls left at this game's rate, time on this level) | E with an unexecuted plan: re86, cn04 A, s5i5 A. About 2-4 | ~30 tokens | nudge-like; weakest evidence |

Not recommended: forced probing of every new object (costs actions; most mode-B misses were misreadings); the idle nudge
P15 (idle stretches are a symptom); the supervisor pass P19 (~17k tokens per call).

Cheapest test before a GPU run: replay the pair's recorded actions through the local engine with R1 computed at each level
switch and check that it states the facts the model got wrong (ls20 "rotated 90° CW once; matched the socket at 2x", wa30
"box carried and released", m0r0 "the two blocks merged", su15 "tier-2 piece in the blob, the same as the HUD"). Then one
GPU pair (exp-054 + P14 + P16 + R1) against exp-054's pair, rule pre-registered on levels 2+ solved, minutes on the stuck
level, and z-sum.

## What the data cannot decide

Two runs of one configuration and one labeller; source traced for 20 stuck levels (lf52, sc25 L4, ft09 L5, tn36 L2
inferred); whether the model would use the records (it uses P23 in ~61% of stuck-level turns and still misread sb26 and
g50t; in r11l B a last-action history produced a false rule); mode B, the largest family, is not reached by most of these
proposals; the 20% estimate assumes the late-run completion rate holds; public-25 gains have transferred poorly so far
(leaderboard draws 2.7-4.2).
