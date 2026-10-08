Summary: of the 17 games exp-073b did not win, 8 stopped on a mechanic the agent never decoded, 6 ran out of time while progressing (2 of them began their level with under 11 minutes left), and 3 failed on planner bugs, an off-screen object and a budget-death loop; on the unwon games 21% of tool calls loop over `history` to recover facts the harness holds exactly, data dies between calls, and the D' gate keeps last levels parked 70% of the time, so the cheapest new harness gains are a per-level data store and an exact per-action effect ledger (the budget, planner and trim-ledger evidence belongs to M2, M3 and M4, which are being built).

# exp-073b failure analysis: why 17 games stopped short of a win

Research note, 2026-10-08. Read-only analysis of run data on the CPU. No GPU, no Kaggle and no repo code were used or changed.
**Measured** means computed from the run files by the scripts listed below. **Estimate** means a judgement, and the
reasoning is given each time.

## Sources and method

- **Run.** exp-073b: `scottmahony/arc3-dprime-reap448-r14-accept05-full` v2, 2026-10-08 13:55-15:57 UTC.
  - Configuration: D' slot priority, REAP-448, 14 streams and MTP acceptance 0.5.
  - It played all 25 public games at 121 minutes per game and scored 56.00 with 124 levels and 6,350 actions.
  - It ran Franzen's harness without our sandbox fix. ours-01 is in exp-074t and exp-075, not in this run: the notebook log
    has no "our harness patches applied" line.
- **Files.** All under `runs/exp073b-dprime-reap448-r14-accept05-full/`:
  - `report.txt`;
  - `kernel-output/benchmark.json`: one record per action, holding the action (with click coordinates), the tokens
    generated since the previous action, and the wall clock since the run started;
  - `kernel-output/prompts/<game>_p0.log`: the full input of each game's last model call;
  - the notebook log.
- **What the last-call logs hold.**
  - Each log holds the history retained after context trims: 8-26 user turns and 8-45 tool calls per game, 545 calls in
    all.
  - That covers the final level and at most the end of the level before it. Earlier levels are not visible (for example
    tr87 L4, cd82 L2, ls20 L3-L5).
  - Every count taken from the logs is therefore a lower bound for the whole game, weighted toward each game's last
    minutes.
- **Active versus parked time.**
  - benchmark.json records tokens and wall clock per action. It does not record gate events.
  - A stretch between two actions counts as active for (tokens ÷ 55 tok/s) + 5 s. The remainder counts as parked at the
    slot gate when it is over 60 s.
  - Why 55 tok/s: the 219 stretches longer than 2 minutes that ran above 20 tok/s have a median of 60.3 tok/s and a
    10th percentile of 53.3. So 55 tok/s errs toward calling time active.
  - Waits before a game's first action are listed separately.
- **Categories.**
  - Each game gets one primary category from the brief's list: the cause that consumed the level's time. Secondary
    factors are listed beside it.
  - Every quote is verbatim model reasoning or tool output from the last-call logs, checked with `grep -F`.
- **Scripts.** All are in `scratchpad/fa073b/` (session scratchpad) and were run with `python -I`, read-only on the run data:
  - `digest.py` and `skim.py`: condensed versions of the logs;
  - `levels.py`, `gaps.py` and `timetok.py`: per-level actions, tokens, and active and parked time;
  - `gate.py`: parked time grouped by the number of levels left;
  - `lastcall.py`: the time of each game's last call;
  - `counts.py` and `mining.py`: per-game event counts in the logs;
  - `waste.py`: action efficiency.

## Where the 44 missing points are (measured)

| Part | Mean points |
|---|---:|
| Score | 56.00 |
| Solved levels below human efficiency (if every one scored at least 1.0) | 2.89 |
| The 17 levels being played when each unwon game stopped | 11.35 |
| The 42 levels never reached | 29.76 |
| **Total** | **100.00** |

## 1. Per-game table: the 17 games not won

### 1a. The final level in numbers (measured)

Column notes:
- **Start** is the run minute at which the level began.
- **× median** is the final level's tokens divided by the median tokens of the same game's solved levels.
- **Active / parked** is minutes of active compute and minutes parked at the slot gate on this level, as defined in
  Sources and method.
- **Value** is the mean points the level adds if solved at human efficiency: its weight share × 100 ÷ 25.

| Game | Level | Start | Actions / human | Tokens | × median | Active / parked | Median solved active | Class | Value |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| ar25 | 7/8 | 86.9 | 12/233 | 121,917 | 4.8 | 34.1 / 0.0 | 7.2 | stagnation | 0.78 |
| bp35 | 3/9 | 86.6 | 50/44 | 108,470 | 1.0 | 30.3 / 4.6 | 30.8 | stagnation | 0.27 |
| cn04 | 6/6 | 99.9 | 27/113 | 62,158 | 1.5 | 21.1 / 0.0 | 10.4 | cut off | 1.14 |
| dc22 | 1/6 | 0.6 | 190/59 | 192,774 | – | 55.7 / 66.3 | – | stagnation | 0.19 |
| g50t | 3/7 | 96.4 | 91/179 | 71,613 | 0.6 | 21.1 / 3.5 | 32.9 | cut off | 0.43 |
| ka59 | 6/7 | 72.4 | 111/132 | 103,015 | 2.0 | 33.3 / 15.3 | 15.5 | stagnation | 0.86 |
| lf52 | 4/10 | 106.8 | 16/71 | 40,074 | 0.4 | 14.2 / 0.0 | 27.7 | cut off | 0.29 |
| ls20 | 6/7 | 105.2 | 24/192 | 48,554 | 0.9 | 16.2 / 0.0 | 18.1 | cut off | 0.86 |
| m0r0 | 5/6 | 73.0 | 67/500 | 117,257 | 2.6 | 35.0 / 13.0 | 12.9 | stagnation (broken at 118.4) | 0.95 |
| re86 | 7/8 | 90.8 | 153/424 | 77,213 | 4.5 | 26.4 / 3.8 | 5.0 | cut off | 0.78 |
| s5i5 | 7/8 | 110.5 | 3/86 | 35,080 | 0.7 | 10.5 / 0.0 | 13.2 | started late | 0.78 |
| sc25 | 6/6 | 119.3 | 0/50 | 654 | 0.0 | 0.3 / 1.6 | 8.3 | started late | 1.14 |
| sk48 | 3/8 | 99.5 | 25/101 | 66,757 | 0.7 | 20.5 / 1.1 | 28.9 | cut off | 0.33 |
| sp80 | 5/6 | 44.0 | 52/96 | 126,045 | 3.3 | 37.2 / 39.8 | 11.0 | stagnation | 0.95 |
| su15 | 5/9 | 73.0 | 52/36 | 143,697 | 10.0 | 41.5 / 6.9 | 4.4 | stagnation | 0.44 |
| tu93 | 9/9 | 28.2 | 21/111 | 78,917 | 8.2 | 21.4 / 71.4 | 2.8 | cut off | 0.80 |
| wa30 | 4/9 | 65.7 | 223/98 | 92,479 | 5.6 | 31.5 / 24.2 | 6.6 | cut off | 0.36 |

dc22 never left level 1; its first action came at 0.6 minutes.

### 1b. What the agent was doing at the end, and why the level was not solved

All times are run minutes.

| Game | At the end | Primary category and evidence | Secondary factors |
|---|---|---|---|
| ar25 | Last action at 100.7, last call at 120.7. The final 20.4 minutes (71k tokens) produced no action. The agent was mining level 6's frames from `history` and brute-forcing mirror-axis placements to cover the yellow (`('NONE', None, None, 37, 22)`). | **Mechanic misunderstood**: "Hmm, so the charcoal is NOT the mirror of the object's shape." | **Context lost after trim**: "Let me check the history to recall level 6's mechanics!" It then read the board *before* the winning move and doubted the goal: "So the yellow was NOT fully covered at the win." (lesson 0017). **Sandbox**: "Variables reset between calls (only functions are retained). I need to rebuild everything in one call."; NameError `MPS`; `import time` refused. **Turns**: 8 exchanges ended with no tool call. |
| bp35 | Last call at 116.3, then parked for 5.2 minutes. After deaths at 91.3 and 112.0 it was clicking blocks to learn which ones break and what killed the player, in a scrolling view. | **Mechanic misunderstood**: "Hmm, so what killed the player?" | **Perception**: "Hmm, the display shifted by 6 on the first RIGHT (the camera moved)." **Context lost**: about the goal of levels 1-2, "Hmm, I don't have that context in this conversation." Misread a transition result: "HUGE FINDING: transition 67 was a level_completed with action MOUSE(row=33,col=33) at level 3!" **Budget doubt**: "maybe the death was a timeout (the bar ran out)?" 11 UNDOs. Already over the human count (50 against 44), so a win would score at most 0.77. |
| cn04 | Last call at 120.9. It was assembling pieces with tips. It had just learned that tips that meet turn grey and that moving a piece breaks the link. Its search for an assembly with every tip paired timed out. | **Ran out of time while progressing**: "Excellent! A connection happened!" (step 229 of 236). | **Sandbox**: two 30-s timeouts. "Everything got unretained (because the snippet timed out?" — the bug ours-01 fixes. 4 NameErrors (`VAR6`). **Search bug**: "the overlap checks were done for the mirrored configuration". **Goal unsettled**: all pieces connected, or all tips paired. |
| dc22 | Last call at 102.2, then parked for 19.7 minutes. It was clicking every object to see which responds. The goal was still unknown: "OK, I'm going in circles." | **Mechanic misunderstood (effects attributed to the wrong action)**: "So LEFT is NOT the toggle! Earlier I thought LEFT toggled." then "HUGE finding: `MOUSE(row=19, col=48)` toggles the red bar!" That click was first sent at 3.6 (action 10) and 15 times in all, and was identified at 99.7. | **Budget death read as a hazard**: "Clicking the yellow dot = death. So the yellow dot is a hazard." Later: "The RESET at 109 was a TIMEOUT (bar hit 0), not a hazard." Paragraphs repeated in its reasoning. Parked 66.3 minutes in all. |
| g50t | Last call at 119.1. It had run a 5-run plan that relies on recorded runs ("ghosts") replaying. On run 5 only one ghost replayed, and it was counting ghosts per run from `history`. | **Mechanic misunderstood**: "Only ONE ghost exists in run 5 (at (2,4))."; "the ghost count is: run1=0, run2=1, run3=2, run4=0, run5=1". | One reasoning turn of 36k characters. 7 of its 15 calls mine `history`. The level began 24.6 minutes before the end. |
| ka59 | Last call at 120.5. It was probing every object, with a shrinking bar, to get the block past a band. | **Mechanic misunderstood**: "The band never changes in level 6. It's a static wall." … "So the level appears unsolvable". | **Bar pressure**: "The bar is 29 → I have limited actions." Re-read level 4-5 boards from transitions. Parked 15.3 minutes. Levels 3-4 were played at 2.8× and 1.7× the human count (−14.2 game points). |
| lf52 | Last call at 120.5. It had moved the cart off-screen and the view did not follow. It probed with single actions until a LEFT brought the cart back at step 215. | **Perception error (partial view)**: "No 'Y' or 'O' nodes at all in the current frame! The cart is completely off-screen." | **Guard**: the stale-state guard forced one action per call: "The StaleStateActionError guard blocks batched actions after a no-op, even inside try/except." (8 further stops). `'TransitionView' object has no attribute 'level'`. The level began 14.2 minutes before the end. |
| ls20 | Last call at 120.5. It built a search over a maze with moving pluses that tracks the bar and the coins, and found a 37-move plan. Its own guarded executor stopped at step 1 on a mismatch. | **Ran out of time while progressing**: "Interesting! The screen1 plan with the bar constraint: cost 37, bar left 13." | About 5 calls went on reading the maze lattice at level start: "I'm misreading the image's y offsets." `'HistoryEntryView' object has no attribute 'step'`. Levels 3-5 were played at 1.4-1.8× the human count (−24.2 game points). |
| m0r0 | Last call at 120.5. For about 43 minutes it believed the two rooms could not connect. At 118.4 it found that keys open doors of their own colour, planned 26 moves by BFS and had executed 22 when the run ended. | **Mechanic misunderstood (found too late)**: "Hmm, so the rooms are isolated and the merge is impossible." → "MAJOR DISCOVERY! The left S moved UP onto the 'p' square's cell (50,14)". | **Sandbox in the last 2.6 minutes**: NameErrors `step3` and `W0`; "W0, K, DOOR are not retained (they're data, not functions)." The guard stopped a random-walk probe (33 refusals). Parked 13.0 minutes. |
| re86 | Last call at 120.7. Two of three shapes were on their dots; it was measuring the third (the pink plus). | **Ran out of time while progressing**: "The blue frame is now at rows 18-24, cols 39-57 ✓ — exactly the target, covering both blue dots ✓." | **Search**: a BFS timed out on an unbounded state space. **Model bug**: "I found the bug in my model!" (coordinate mapping). |
| s5i5 | Last call at 120.2. The level began 10.5 minutes before the end. It had built a simulator of a chain of bars and checked one prediction. | **Ran out of time (late start)**: "PERFECT! The prediction was exactly right". | Two reasoning turns of 35k and 19k characters, and 2 exchanges with no tool call, at the level start. NameError `open`. |
| sc25 | Level 6 began at 119.3; the last call came at the same minute. | **Ran out of time (late start)**: "LEVEL 6 completed level 5! Now level 6." | Level 5 took 81 minutes. About 50 of them were parked at the gate (51.4-92.5 and 104.0-115.9), and there was one death. |
| sk48 | Last call at 120.1; the last recorded action was at 110.4. It had validated a transition model and A* returned a 42-move plan, but its own simulation said the plan does not win (`sim win False`). The 20 actions it sent at 120.1 never executed: the board and bar did not change and they are not in benchmark.json. | **Ran out of time while progressing**: "The model now validates 111/111 transitions across levels 2 and 3"; "A* found a 42-action solution". | **Unverified goal**: the win predicate was never checked. **Sandbox**: NameErrors on `S3` (twice) and `KS`. **False bar belief**: "the budget depends on how fast I act". A 10.7-minute tail without an action. |
| sp80 | Last action at 102.2, last call at 119.3. It reasoned at length about how the ink flows and concluded the level cannot be solved. | **Mechanic misunderstood**: "CONCLUSION: The level as I understand it is unsolvable." | 5 exchanges ended with no tool call. One sentence was repeated 5 times: "Let me scan all the transitions' results for 'level_completed' or for the C turning red." Parked 39.8 minutes. An 18.8-minute tail without an action. |
| su15 | Last call at 120.5. After a death at 102.0 and 20 UNDOs, its model of the chasing invaders predicted that every move loses an object; it was inspecting the fatal transition. | **Mechanic misunderstood**: "Zero safe 1-step moves! That means my model predicts that ANY move leads to a bite." | **Perception**: "The invader sprite is split into several 1-pixel components". **Retention**: helper functions were dropped (`objs`; "ctr was not retained: unavailable dependencies"). Already over the human count (52 against 36). Its first action came at 50.2, after a 50-minute wait at the gate. |
| tu93 | Last call at 120.8. It was searching for paths past three patrollers and a chaser. It wrote 16 snippets that rebuild its maze graph and search it; 9 searches found no path. It confirmed the chaser rule in its last minutes: "Confirmed: the dark red is a permanent chaser". | **Right idea, but planning failed**: "Wait, my P1 phase: at step 233 P1 is at (2,2) moving left. I set pat(0,0) = ppos(T1,3,-1,0) = (2,6) — wrong!"; "I compared v to pats(i-1) instead of pats(i)". | **Gate**: parked 71.4 minutes on this level (28.2-100.6). **Chaser rule unsure**: "I must be wrong about the dark red's behavior." |
| wa30 | Last call at 96.8, then parked until the end (24.6 minutes). Its third attempt at level 4 was going to plan when it lost its slot. | **Budget/death loop**: budget deaths at 80.9 and 93.8. The second was forced on purpose: "Let me burn the last 3 moves." … "since the level is unwinnable now, let me just trigger the death." | Parked while progressing. 223 actions on the level against a human 98, so a win would have scored about 0.2. |

### Counts

**Primary categories:**

| Category | Games |
|---|---|
| Mechanic misunderstood | 8: ar25, bp35, dc22, g50t, ka59, m0r0, sp80, su15 |
| Simply ran out of time while progressing | 6: cn04, ls20, re86, sk48, s5i5 and sc25 (the last two started under 11 minutes before the end) |
| Right idea but execution/planning failed | 1: tu93 |
| Perception error | 1: lf52 |
| Budget/death loop | 1: wa30 |
| Wrong goal defended, tool or sandbox trouble, context lost after trim | 0 as primary (all appear as secondary factors) |

**Secondary factors:**

| Factor | Games |
|---|---|
| Final level began less than 25 minutes before the end | 7: cn04, g50t, lf52, ls20, s5i5, sc25, sk48 |
| Parked at the gate for 10 minutes or more on the final level (sc25: on the level before it) | 7: dc22, ka59, m0r0, sc25, sp80, tu93, wa30 |
| NameError on data defined in an earlier call, or a 30-s timeout | 8: ar25, cn04, lf52, m0r0, re86, sk48, sp80, su15 |
| Deaths on the final level | 4 games, 6 deaths: bp35 2, wa30 2, su15 1, dc22 1 |
| Exchanges with no tool call | 7: ar25 8, sp80 5, s5i5 2, sc25 2, su15 2, dc22 1, g50t 1 |
| Explicit loss of earlier-level knowledge | ar25, bp35 |
| Partial or shifted view | lf52, bp35, su15 |

## 2. Time and tokens on the final levels (measured)

- **Final levels.**
  - They took 1.49M of the run's 5.75M generated tokens (26%).
  - The median final level took 78.9k tokens, against 22.1k for the median of the 124 solved levels.
  - The median final level ran 26.4 active minutes and 34.1 wall-clock minutes.
- **Ratio to solved levels.**
  - The final level used far more tokens than the same game's typical solved level in su15 (×10.0), tu93 (×8.2),
    wa30 (×5.6), ar25 (×4.8), re86 (×4.5), sp80 (×3.3) and m0r0 (×2.6).
  - It used fewer in lf52 (×0.4), g50t (×0.6), s5i5 and sk48 (×0.7) and ls20 (×0.9). Those levels ran out of clock
    before they ran out of ideas.

| Class | Games | Active min (median) | Tokens | Parked min | Value (mean points) |
|---|---|---:|---:|---:|---:|
| Long stagnation (no verified progress for 15 or more active minutes at the end of the level) | 7: ar25, bp35, dc22, ka59, m0r0, sp80, su15 | 35.0 | 913k (61% of final-level tokens) | 145.9 | 4.44 |
| Steady progress cut off by time | 8: cn04, g50t, lf52, ls20, re86, sk48, tu93, wa30 | 21.1 | 538k | 104.0 | 4.99 |
| Started too late to matter (under 11 minutes left) | 2: s5i5, sc25 | 10.5 and 0.3 | 36k | 1.6 | 1.92 |

**What puts a level in each class.**
- **Stagnation.** The marks are a long no-action tail, an "unsolvable" verdict, or no way forward:
  - ar25 ended with 20.4 minutes and 71k tokens without an action; sp80 with 18.8 minutes and 47k tokens.
  - ka59 and sp80 both concluded the level was unsolvable.
  - su15's model admitted no safe move.
  - dc22 never cleared level 1 in 190 actions.
  - m0r0 counts here because its 43-minute stall ended only 2.6 minutes before the end.
- **Cut off.** Mechanics were being confirmed, or plan steps were executing correctly, in the last minutes:
  - cn04's connection rule, sk48's 111/111 model check, re86's 2 of 3 shapes, ls20's 37-move plan;
  - tu93's chaser confirmation, after 71 minutes parked;
  - wa30's third attempt, parked for the last 24.6 minutes;
  - lf52 had got its cart back, and g50t was measuring the replay rule from fresh data.

So 10 of the 17 levels were limited by time: 8 cut off and 2 started late. Stagnation consumed 61% of the final-level
tokens.

**The slot gate, by levels left after the current one** (`gate.py`; all 25 games, game-minutes):

| Levels left | Parked | Active | Parked share |
|---|---:|---:|---:|
| Not started yet (waiting for the first slot) | 304.3 | – | – |
| 3 or more | 281.3 | 1,056.6 | 21% |
| 2 | 0.0 | 237.8 | 0% |
| 1 | 210.6 | 297.8 | 41% |
| 0 (the game's last level) | 242.8 | 102.0 | **70%** |
| **All** | **1,039** | **1,702** | **38%** |

**Why the gate behaves this way.**
- This follows from the D' formula (the D' cell in `kaggle/dprime/`): priority = A·M·C + B·φ.
- B is 16, 14, 10 or 0 when 3 or more, 2, 1 or 0 levels remain. A is roughly 1-6 for these games.
- So a game on its last level ranks below every game with levels left, until the tail fade φ starts at 74.7 minutes.

**What it cost.**
- Unwon games:
  - tu93 waited 71.4 minutes on level 9.
  - sc25 waited 49.8 minutes on level 5 and then had 1.9 minutes for level 6.
- Won games still finished late:
  - Their last or second-to-last levels waited 23-61 minutes: r11l L6 55.7, vc33 L7 61.1, tn36 L6 24.4 and L7 23.3,
    tr87 L6 29.6, lp85 L7 30.9.
  - tr87 won 1.3 minutes before its limit.
- Unstarted games waited 304 game-minutes in all. su15 waited 50 minutes, and ar25, re86 and s5i5 about 40, because D'
  removed Franzen's fresh-first rule.

## 3. Cross-game patterns and harness interventions

### Ranking by expected levels gained per engineering day (estimates)

Gains are levels per 25-game public run, as judgement calls with the reasoning in each section below. The 17 final
levels average 0.67 mean points each, but early levels of hard games are worth less.

| Rank | Intervention | Pattern (games) | Levels gained | Engineering days | Levels per day | Overlap |
|---:|---|---|---:|---:|---:|---|
| 1 | Per-level data store (`mem`) | Data lost between calls (10) | +0.5 to +1.5 | 0.5 | ~2 | M1(c), proposed but not built. Not M2/M3/M4. |
| 2 | Per-level action→effect ledger, with mover tracks and a death-overlap fact | Hand-mining `history` (16 games); effects attributed to the wrong action | +1 to +3 | 2 | ~1 | Includes M5's automatic variant (not being built). Complements M4 and M2. Not M3. |
| 3 | View-shift, off-screen and lattice perception | Partial or shifted views, lattice parsing (5) | +0.5 to +1.5 | 2 | ~0.5 | Franzen-analysis lever 6. Not M2/M3/M4. |
| 4 | Stop runaway turns in code | Exchanges with no tool call, repetition loops (7) | 0 to +1 | 1 | ~0.5 | Franzen-analysis lever 8. Risk from lesson 0022. |
| 5 | Remove the gate's B cliff on the last level | Last levels parked 30-71 minutes in total (7 games) | −1 to +2 | 1.5 | ~0.3, high variance | None of M2/M3/M4 (scheduler). |

Rank 5 has the largest possible effect on the hidden set, but its sign is unknown and it needs a gate-binding run. That
is why it ranks last.

### P1. Data does not survive between tool calls (10 games) → a per-level data store

**Evidence (measured).**
- The retained windows hold 19 NameErrors on names the model had defined in an earlier call, in 10 games: ar25 1,
  cn04 4, lf52 1, lp85 3, m0r0 2, sk48 3, sp80 1, su15 1, tr87 2, vc33 1. That is 3.5% of the 545 calls.
- Models also spend reasoning on working around the limit:
  - "Variables reset between calls (only functions are retained). I need to rebuild everything in one call." (ar25,
    which also tried to cache masks in a mutable default argument and found "persist across calls? False");
  - "W0, K, DOOR are not retained (they're data, not functions)." (m0r0);
  - su15's helper was dropped: "Your function ctr was not retained: unavailable dependencies: box".
- ours-01 fixed only the timeout case and the `time` import. Ordinary data still dies with every call, by Franzen's
  design.

**Intervention: M1(c) of `new-methods.md`.** The sandbox keeps a `mem` dict across calls within a level:
- values are JSON-serialisable, capped at about 200 KB, and cleared at a level change;
- each tool result lists its keys and the step each was written, so staleness is visible.

It is exact, needs only code, and costs about 40 static prompt tokens.

**What it would have changed.**
- cn04 lost 4 calls in its last 21 minutes.
- m0r0 lost 2 calls plus a rebuild in its last 2.6 minutes, when its plan was 4 moves short.
- sk48 lost 3 calls in its last 10.7 minutes.
- ar25 rebuilt its masks on every call for 20 minutes.

**Gain (estimate): +0.5 to +1.5 levels.** About 3% of calls are lost outright. They cluster on final levels, where the
score has responded 0.6-0.8 to extra compute (`plan.md`).

**Cheapest offline test.**
1. Unit tests in `scripts/franzen_bed.py`: a value set in call 1 can be read in call 2, it is cleared at a level
   change, and the cap holds.
2. Then compare the NameError count in the logs of the next full run with exp-075.

### P2. The agent re-derives from `history` what the harness knows exactly (16 games) → a per-level action→effect ledger

**Evidence (measured; `mining.py`).**
- Volume:
  - 88 of the 545 retained calls (16%), in 16 games, loop over `history` or `transitions`.
  - In the 17 unsolved games the share is 69 of 331 calls (21%). In the 8 won games it is 19 of 214 (9%), and all 19
    are in lp85.
  - 34 of the 88, in 12 games, search for level wins or deaths.
- Cases where the fact sought had been in the transitions for a long time:
  - **dc22**: the click that toggles the red bar was first sent at 3.6 minutes and identified at 99.7, 168 actions
    later, by a history scan. Before that the agent credited LEFT with the effect.
  - **g50t**: ghosts replayed per run, found only after the 5-run plan failed.
  - **sc25**: "Confirmed: in the previous run, the dark red was removed with the marker at rows 35-36, cols 15-16".
  - **ka59**: scanned the level 5-6 transitions for block moves that did not match the key pressed.
  - **bp35**: the cause of each death.
  - **tu93, su15, ls20**: rebuilt the tracks and periods of autonomous movers by hand ("Periods: 8, 4, 6 → LCM 24";
    "Both bounce with period 8"). tu93's phase bugs (table 1b) came from exactly this.

**Intervention.** For each level, the harness keeps an exact table keyed by action signature, meaning a key, or a click
resolved to the segmented object under the cursor. Each entry records:
- the count of uses;
- the outcome classes seen: no-op, HUD-only, object changes (`frame_diff` entries with offsets), level up, and game
  over with the bar's state at death;
- the first and last step.

It also keeps a track for every object that moved without being the acted object: positions per step, plus period and
phase once they repeat. On a game over it states what the controlled object overlapped in the fatal frame.

All of this is exposed as `effects` and `movers` in the sandbox, which costs no prompt tokens. The action result gets
one line when a signature produces an outcome class not seen before. Every line states only what the frames show.
Lesson 0024 says the report must never state a false fact. P23's matcher and its tests already exist in
`scripts/taaf_ours_patch.py`.

**What it would have changed.**
- dc22: "MOUSE on the red T: red bar rotated" from action 10 onward.
- bp35: the overlap that killed the player.
- g50t: ghosts per run.
- tu93: exact patroller periods and phases, with no hand-written schedule.
- su15: which click merges and which only moves.
- sc25: which marker position made the pink pattern work.
- ka59: what moves on which press.

**Gain (estimate): +1 to +3 levels**, mostly early levels of hard games and mover levels.
- For it: lesson 0024. On the old base, P23 raised hard-game level-1 solves from 10/16 to 15/16 over two paired runs.
- Against it: Franzen saw no clear gain from inserting animation timelines automatically. So the inserted part must stay
  at one line per new outcome, and the rest must be a variable the model chooses to read.

**Cheapest offline test.**
1. benchmark.json holds every one of the 6,350 actions, click coordinates included. Replay them through the local
   engine (`scripts/replay_probe.py`) and build the ledger.
2. Check every ledger line against the frames.
3. Check that the fact is present at the confusion points listed above: dc22 from action 10, g50t's runs 1-5, tu93's
   patroller periods.

This needs the CPU only.

**Overlap.**
- It includes M5's "automatic variant" in `new-methods.md`, which is not being built.
- It complements M4 (win records at trims) and M2 (budget deaths).
- It does not overlap M3.

### P3. Partial or shifted views, and lattices read by hand (5 games) → exact perception helpers

**Evidence.**
- lf52 lost about 8 calls of single-action probes to an off-screen cart (table 1b).
- bp35 confused positions across camera shifts of 6 and 12 cells.
- su15's diagonal sprite came apart into 1-pixel components.
- ls20 spent about 5 calls reading the maze lattice, and its first action on level 6 came 6.9 minutes into the level:
  "I'm misreading the image's y offsets."
- cn04 misread its block grid: "the earlier print was misleading (the crop was offset)".

**Intervention.** All exact and code only:
1. A whole-view shift estimate between consecutive frames. The action result states it when it is not zero ("view moved
   by (dr, dc)"), and the sandbox gets a world-coordinate offset.
2. Last-seen boxes for objects that left the view.
3. Lattice detection (cell size and origin) with a logical-grid view.
4. An 8-connected option for `segmentation`.

**What it would have changed.**
- lf52: the cart's position after it left the view.
- bp35: positions in world coordinates.
- ls20: part of the 6.9 minutes before its first action on level 6 (16.2 minutes in all on that level).
- cn04 and su15: a few calls each.

**Gain (estimate): +0.5 to +1.5 levels.** The savings fall on levels that were cut off by time: ls20, cn04 and lf52.

**Cheapest offline test.**
1. Replay lf52's and bp35's recorded actions locally and compare the shift estimate with the engine's camera offset at
   every step.
2. Run the lattice detector on the first frame of every level of the 25 games.

**Overlap.** Lever 6 of `franzen-run-analysis.md`. Not M2/M3/M4.

### P4. Turns that end without a tool call, and runaway reasoning (7 games) → stop them in code

**Evidence.**
- 21 exchanges ended without a tool call in 7 games' windows: ar25 8, sp80 5, s5i5 2, sc25 2, su15 2, dc22 1, g50t 1.
- There are 43 reasoning turns longer than 15,000 characters.
- Some reasoning looped word for word: sp80's sentence 5 times, and a paragraph in dc22.
- ar25 and sp80 spent their final 20.4 and 18.8 minutes without an action.

**Intervention.** Two code-side rules, with no new wording:
- end a request whose streamed reasoning repeats a sentence of 80 or more characters three times;
- after an exchange with no tool call, cap the next request's output (for example at 4k tokens).

**What it would have changed.** About 20 minutes each in ar25 and sp80, and 2 exchanges at s5i5's late level start.

**Gain (estimate): 0 to +1 level.** The time saved mostly falls on levels that were already stagnating. Lesson 0022 warns
that past reasoning carries weight, so cutting reasoning can hurt.

**Cheapest offline test.** Over the 25 logs, count the turns each rule would cut and check that none of them ended in an
action. This uses the CPU only.

**Overlap.** Lever 8 of `franzen-run-analysis.md` (cap long completions). Not M2/M3/M4.

### P5. The slot gate parks last levels (7 games, 30-71 minutes parked in total) → remove the B cliff on the last level

**Evidence.** Section 2.
- Unwon: tu93 and sc25.
- Won late: r11l level 6 (parked 55.7 minutes, won at 102.4), vc33 level 7 (61.1, 106.9), tn36 levels 6-7 (47.7,
  110.9), tr87 level 6 (29.6, 119.7, 1.3 minutes before its limit) and lp85 level 7 (30.9, 77.5).
- Separately, wa30 and dc22 lost their slot and were parked for their final 24.6 and 19.7 minutes (in this scheduler a
  running game gives its slot back only at a context trim). wa30 was in the middle of a working attempt.

**Intervention.** Remove the step in D''s remaining-levels bonus: give 0 levels left the same B as 1 level left (10), or
make B proportional to the current level's weight share. Everything else stays as it is. It is exact and touches only
the scheduler.

**What it would have changed.** tu93 and sc25 would have got their slots earlier. Some other games would have been
parked instead, and this run cannot say which or at what cost.

**Gain (estimate): −1 to +2 levels on the public 25.** The hidden set's game count and level depth are unknown, so the
sign there is unknown too (lesson 0011).

**Cheapest offline test.**
1. A replay simulation that uses the per-level token costs of exp-073b, plus exp-073 and exp-075, at 14 slots × 121
   minutes, comparing D' with the variant. `scripts/sim_call_share.py` is a starting point.
2. Then one gate-binding full run before the change goes near a submission.

**Overlap.** None of M2/M3/M4.

### Patterns that belong to M2, M3 and M4 (evidence for the builders, no new design)

| Pattern | Evidence (measured) | Owner | Notes for the builder |
|---|---|---|---|
| Budget deaths, forced burns, misread bars | 18 game overs in 10 games; 6 on final levels (bp35 2, wa30 2, su15 1, dc22 1). Misreads: dc22 took a budget death for a hazard; bp35 ("maybe the death was a timeout (the bar ran out)?"); sk48 ("the budget depends on how fast I act", which is false). wa30 burned its bar on purpose twice. ka59 played under bar pressure. | **M2** | wa30's burns are exactly what exposing RESET removes. dc22 shows why the death line must say "budget death" when the bar reads 0. |
| Solved levels well above the human count | Raising every solved level to at least human efficiency adds 2.89 mean points: ls20 +24.2, ka59 +14.2, m0r0 +12.7 and sc25 +5.8 game points among the unwon games. The logs do not cover those levels, so why they were slow cannot be read. | **M2/M3** | More than Franzen's 10-game demo showed (0.73); most of it is in bar games with long levels (ls20, ka59, m0r0). |
| Hand-written planners time out, have bugs, get rewritten | 8 tool timeouts in 5 games (cn04 2, tr87 3, lp85, re86, vc33). Search bugs in the final windows of cn04, re86, tu93 (16 rewrites, 9 searches with no path), m0r0, sk48, ls20 and su15. ls20 and m0r0 wrote their own step-by-step executors that stop on a mismatch. | **M3** | That is `run_plan`'s design, and the model reaches for it unprompted. tu93 suggests `search` should accept a time-indexed state (mover phase) without the model hand-rolling the schedule. |
| Earlier levels re-derived after trims; misread win frames; confusing result fields | bp35: "Hmm, I don't have that context in this conversation." ar25 re-derived level 6 and read the board before the winning move. 34 history scans for level wins or deaths in 12 games. Field confusion: bp35's "transition 67 … at level 3"; lf52 `'TransitionView' object has no attribute 'level'`; ls20 `'HistoryEntryView' object has no attribute 'step'`. | **M4** | The win record must hold the terminal layer of the winning action (lesson 0017), not the board before it. `level_wins` should carry plain level numbers and step indices. |

### Smaller findings (fewer than 3 games; follow-ups, not designed here)

- **The stale-state guard's escape hatch does not work.** The guard's message says the error "can be caught", but
  catching it does not let the snippet continue: every later action is refused too.
  - lf52: "… even inside try/except", with 8 further stops.
  - m0r0: a 40-step random-walk probe made 4 moves; the guard refused 33 more.
- **sk48's last 20 actions were silently dropped.** They were sent at 120.1 minutes and returned `NOT_FINISHED`, but there
  are no `[action]` lines, the board and bar did not change, and benchmark.json ends at action 196 (110.4 minutes). It is
  worth checking what `action()` does near the per-game limit, so that a ready plan is not lost.
- **Blocked modules.** `open` in lp85, r11l and s5i5; `inspect` in r11l and sk48. `time` was blocked in ar25 and is allowed
  since ours-01.

## 4. Won games: what was wasteful (measured)

| Game | Score | Levels above human (actions / human → level score) | Game points lost | Long waits | Notes |
|---|---:|---|---:|---|---|
| cd82 | 96.99 | L2 26/8 → 0.09 | 3.01 | Waited 16.3 minutes for its first slot; level 1 parked 10.0 | Level 2 is not in the retained window, so the cause cannot be seen. |
| ft09 | 100 | none | 0 | Waited 18.3 minutes to start | Every level was below the human count. |
| lp85 | 100 | L4 18/16, L6 70/60 (absorbed by surplus) | 0 | Level 7 parked 30.9 | |
| r11l | 100 | none | 0 | Level 6 parked 55.7; won at 102.4 | |
| sb26 | 96.23 | L8 30/18 → 0.36 | 3.77 | Waited 32.1 minutes to start | Level 8 (weight 8, with no surplus to absorb a loss): 16 clicks and a SPACE on a model of the pointer's walk that the animation then refuted; the fix took 6 swaps (12 clicks). |
| tn36 | 100 | L4 74/40 (absorbed; one death at 45.8) | 0 | Levels 6 and 7 parked 24.4 + 23.3 | |
| tr87 | 95.04 | L4 107/45 → 0.18 | 4.96 | Waited 19.1 to start; level 6 parked 29.6; won 1.3 minutes before its limit | Level 4 took 102.6k tokens, the costliest solved level in any won game. It is not in the window. |
| vc33 | 100 | L1 13/7 (absorbed) | 0 | Level 7 parked 61.1; won at 106.9 | |

- In the won games, inefficiency cost 11.74 game points in all, which is 0.47 mean points.
- Every long stall before a solve (23-61 minutes parked) was gate parking, not thinking. The longest *active* stretches without
  an action on a solved level were short: 9.3 minutes (tr87 L4), 8.7 (lp85 L6) and 7.9 (r11l L2).

## 5. Open questions and what this data cannot tell

1. **One run.** Per-game outcomes are single draws: between exp-073 and exp-073b, vc33 went from 21 to 100 and tn36 from
   4 to 100. exp-075 (the same configuration plus ours-01) will show which of these failures repeat.
2. **The logs hold only the last 50-60k tokens of each game.** Earlier levels' reasoning is gone. The causes behind the
   2.89 efficiency points (ls20 L3-L5, ka59 L3-L4, m0r0 L4, tr87 L4, cd82 L2) cannot be read here. Counts from the logs
   are lower bounds.
3. **Gate events are inferred from tokens, not logged.** The notebook does not log handovers or priorities. Logging
   (game, level, priority, wait) at each gate acquire would make P5 measurable.
4. **Who pays when last levels are un-parked?** This run cannot say. Answering it needs the replay simulation and a
   gate-binding run, and the hidden set's size and depth decide the sign.
5. **Does an inserted effect line help on Franzen's base?** His automatic timeline insertion showed no clear gain, and
   lesson 0024 comes from the old base. The replay test checks truth and coverage only; a paired GPU run decides.
6. **ka59 L6 and sp80 L5 ended with "unsolvable".** Comparing those levels with the human replays in `data/human` would
   show which mechanic or goal the agent missed. That is a cheap CPU check, and it would also tell whether P2's ledger
   could have surfaced it.
7. **Transfer.** Every number here comes from the public games. Lesson 0018 says public gains transfer to the hidden set
   only partly.
