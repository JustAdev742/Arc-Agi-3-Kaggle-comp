# What happens at a level-up: the exp032 control run (Duck harness, Qwen3.8-Flash-Next)

Run: `runs/exp032-anim-flashnext` (25 public games played at once for 132 min; score 7.86). 21 games completed at least one
level, which gives **38 level starts at L2 or later**. 37 of them got at least one model call; g50t L2 began 1 minute before the run ended.
Read-only analysis, one run, one seed. Scripts and dumps: `scratchpad/research/lt/`.

## Bottom line

1. **A new level is not slow to start because of the level-up.** The first action on L2+ comes after a median of **3 model
   calls**, the same as on L1. It takes 8.0 min instead of 1.6 min because each call then takes about **160 s instead of 30 s**:
   the server runs 3 requests with about 22 waiting for most of the run. The same holds for whole levels: a solved L2+ level
   needs **12 calls** (L1: 14). It takes longer in minutes (32.5 vs 22.1) because its calls are slower.
2. **First calls are consistent.** In 37 of 37 level starts, call 1 inspects the new board with code and never acts. In 34 of 37, the first three calls restate how
   the previous level was solved. After that, 20 level starts spend their first action on a probe (median 1 action) and 15 run a plan carried
   over from the previous level (median 7 actions). In 2 the model never acted.
3. **The previous level fills the context for about 20 minutes.** At the new level's first call, 94% of the chat characters
   come from earlier levels. Those turns leave the history a median **19.8 min** (5 turns, 8 calls) after the level-up.
4. **The previous level's knowledge was both used and misleading.** It was reused visibly: in solved later levels, and as copied code (20% of long code lines, against 3% copied from other games). In at least 4 level starts, a stale belief from the previous level cost a
   large share of the level: sp80 L2, tu93 L3, ls20 L2 and wa30 L2. In each case the stale belief was the model's own restatement of the
   previous level, not raw history text that misled it.
5. **Recommendation: (b), weakly.** Add a level-start prompt and keep the history, together with a harness-written summary of how the last level was won. Do not clear the previous level's messages. The data cannot show that any option shortens later levels. The comparison between probing first and applying a carried plan first is small (n = 35), not significant (p = 0.5), and confounded by game.

## Method (so the numbers can be checked)

- **Level start:** the wall-clock time of the action that completed the previous level (`benchmark.json` history, joined with the
  `level_completed` flags in `artifacts/*_events.jsonl`). The times match `summary.json` `level_done_s` exactly.
- **First action:** the first environment action after that point.
- **Calls:** `[MODEL RESPONSE META]` blocks in the transcript turns, counted from the level's first turn through the acting turn.
  An acting call is always the last call of its turn, because the harness ends a turn when it executes an action.
- **Unsolved levels** are censored at the run end (about 132 min).
- **History reconstruction.** The harness (`tests/fixtures/taaf_anim/.../tool_agent.py`) keeps a suffix of the previous history plus
  the messages appended in the turn, trimmed to about 31.7k estimated tokens and at most 30 assistant messages. A turn that ends in a request error keeps the old history. Each turn logs its history length as `history_messages`. Replaying this rule over 922 turns gave 0 inconsistencies. Message counts are therefore exact; character shares are approximate, with each image counted as 900 characters.
- **Classification** of each level start (probe versus carried plan) comes from reading the first calls. I was the only reader; the labels are in `lt/q3.py`.

## 1. Time and calls until the first action and until the level ends

| | n | min to 1st action, median (IQR) | calls to 1st action, median (range) | solved | min to solve (median) | calls to solve (median) |
|---|---|---|---|---|---|---|
| L1 | 25 | 1.6 | 3 (2–5) | 21 | 22.1 | 14 |
| L2+ (all) | 38 | 8.0 (5.8–11.7); 3 never acted | 3 (2–15) | 17 | 32.5 | 12 |
| L2 | 21 | 8.1 | 3 | 10 | 33.7 | 12.5 |
| L3 | 10 | 8.0 | 3 | 4 | 22.5 | 7.5 |
| L4 | 4 | 7.1 | 2.5 | 3 | 45.3 | 15 |
| L5 | 3 | 7.1 (only lp85 acted) | 3 | 0 | – | – |

- **Per-call wall time:** the median seconds per call inside a turn were 124 s over run minutes 0–30, 171 s over 30–60, 174 s over 60–90, 152 s over 90–120 and 148 s after that.
  Up to the first action, a call took 30 s on L1 and 160 s on L2+. The vLLM log shows "Running: 3 reqs, Waiting: 21–22 reqs"
  from about minute 13 to the end.
- **The 4.9-min figure** in `summary.json` measures from the start of the level's first turn to the start of the first acting turn. It reproduces as
  4.9 min. Measured to the action itself, the figure is 8.0 min.
- **Long starts** came from long analysis, not from level-up overhead:
  - cn04 L2: 15 calls and 41 min designing an assembly search, including a shape-normalisation bug.
  - tr87 L2: 15 calls and 40 min parsing glyphs.
  - ls20 L2: 8 calls and 22 min on a BFS plan.
  - wa30 L2: 7 calls; the A* search timed out.
- **Fifteen of the 20 unsolved later levels** that got any calls ended with fewer actions than the human baseline. Wall-clock time, not the action count,
  was the binding constraint.

## 2. What the model does in its first 1–3 calls on a new level

- **Call 1 always inspects:** in 37 of 37 level starts, it runs code that reads `segmentation` or crops `ascii`. None of these call-1 snippets calls `action()`. The median call-1
  reasoning is 762 characters, against 2,262 for all calls.
- **Recap of the previous level:** 34 of 37 restate how the previous level was solved or what mechanic was confirmed, for example
  "Level 1 solved by matching the framed panel to its embedded mini target". The other 3 (sb26 L2, vc33 L3, wa30 L2) use the previous
  representation without stating it.
- **Main pattern of the first three calls:**

| pattern (first 3 calls) | n | level starts |
|---|---|---|
| inspect and recap, then a **probe** within 3 calls | 13 | cd82 L2, ft09 L2, ka59 L2, lf52 L2, lp85 L4, lp85 L5, re86 L4, s5i5 L2, su15 L2, tn36 L2, vc33 L2–L4 |
| inspect and recap, then **re-apply the previous approach** within 3 calls | 9 | ar25 L3, ka59 L3, lf52 L3, sb26 L2, sc25 L2, sc25 L3, sp80 L2, tu93 L2, tu93 L3 |
| inspect and recap, **no action** in the first 3 calls | 15 | first action later a probe in 7 (ar25 L2, cn04 L2, ft09 L3, lp85 L2, lp85 L3, r11l L2, tr87 L2), a carried plan in 6 (ft09 L4, ls20 L2, re86 L2, re86 L3, sb26 L3, wa30 L2), none in 2 (ft09 L5, re86 L5) |

- **Kinds of probe:** 5 probes were systematic, testing every direction or button: ar25 L2, vc33 L2, vc33 L3, re86 L4 (cycling SPACE) and cd82 L2 (all 3 paint
  operations). The other 15 probes tested one element.
- **Stale runtime state wasted reasoning in 5 level starts.** `previous_frame` or `last_action_result` still showed the previous level. Examples:
  - lp85 L5 (+4.7 min): "That diff was level 4→5 transition (previous_frame was level 4)."
  - tr87 L2 (+35.4 min): "The previous_frame is the L1 end screen (sky blue), so the diff is level transition — not useful."
  - The same happened in r11l L2, su15 L2 and ft09 L3.

## 3. Re-applying the previous level's approach versus probing first

| first action on the new level | n (games) | solved | solved: median min / calls | solved: actions ÷ human | unsolved: median min available (n < 30 min) |
|---|---|---|---|---|---|
| carried plan (all) | 15 (11) | 6 (40%) | 31.7 / 12 | 1.0 | 48.4 (4) |
| probe (all) | 20 (14) | 11 (55%) | 35.0 / 13 | 0.8 | 82.4 (1) |
| carried plan within 3 calls | 9 | 3 (33%) | 30.9 / 12 | 1.6 | 36.3 (3) |
| probe within 3 calls | 13 | 7 (54%) | 30.0 / 11 | 0.9 | 81.9 (1) |
| carried plan, ≥45 min available | 10 | 5 (50%) | 32.5 / 12 | 1.0 | – |
| probe, ≥45 min available | 19 | 11 (58%) | 35.0 / 13 | 0.8 | – |

- **Probing first solved slightly more levels** and solved them in about the same time. The difference is not significant: Fisher p = 0.50 overall and 0.71 for levels with at least 45 min available. It survives
  relabelling the borderline cases (lf52 L3, ka59 L3 and tu93 L2 moved to probe, cd82 L2 to carried plan: 38% vs 55%).
- **Confounds:** the result depends on the game. lp85 and vc33 always probed; without them, probing solves 6 of 13 (46%) with a median of 52 min and 18.5 calls. Carried-plan levels also had less time left.
- **When the carried plan was right, it was the fastest route:** sc25 L2 was solved in 14.2 min with 6 calls and 5 actions (human: 6); re86 L3 in 17.6 min with 47 actions (human: 86).
  **When it was wrong, the level stalled:** sp80 L2, tu93 L3, ls20 L2 and wa30 L2 (section 5).

## 4. How much of the context is still about earlier levels

- **Share of the chat history, excluding the fixed system prompt**, that comes from earlier levels, per call on the new level:

  | call | characters | turns | spread (characters) |
  |---|---|---|---|
  | 1 | 94% | 83% | 93–94% |
  | 2 | 90% | 83% | |
  | 3 | 77% | 71% | |

  The history at a turn start holds a median of 5 turns and about 63k characters, matching the server's 20.6k prompt tokens per request. At call 1 it also includes a median of 5 earlier-level user messages, each with an image of a board from that level (range 2–10).
- **Time until the earlier-level turns have left:** a median of **19.8 min** (IQR 18.0–24.8, range 10.7–31.1), after 5 turns or 8 calls; n = 29.
  In 9 level starts they never left before the level ended.
  - 4 of the 17 solved later levels were solved entirely with the previous level still in context. They include the fastest two: vc33 L2 in 8.4 min and sc25 L2 in 14.2 min.
  - Overall, 38% of the minutes spent on L2+ levels ran with previous-level text in the context. The median time until it left was the same for solved and unsolved levels (19.6 and 19.9 min).
- **Cross-level memory barely exists.** The harness clears the carried note at every level-up; only the "Cross-level notes" line survives.
  The note was non-empty at **1 of 38** level starts (tu93 L3: "Level 1 solved; reward ~0.111/level."). Within a level, the model re-creates
  the knowledge: 147 of the 296 non-empty notes on L2+ turns mention an earlier level.
- **The model keeps referring to earlier levels after they leave**, measured with a regex over the model's text:
  - 67% of calls (185 of 277) mention an earlier level while its turns are in the history.
  - 40% of calls (179 of 444) mention one after they have left.

## 5. Knowledge that was needed and used, versus stale assumptions

**Used (same mechanic reused):**
- **sc25 L2, +0.0 min:** "Same mechanics expected: click the 3 white cells to match the socket pattern → unlock tip → move it into the head."
  Solved in 6 calls, 14.2 min and 5 actions (human: 6).
- **ft09 L2, +3.2 min:** "strong confirmation of the level-1 rule (white=on, gray=off)". Solved in 22.1 min. L3 and L4 were then solved with the
  same "established pipeline" (16 actions against the human 23; 25 against 28).
- **re86 L3, +0.0 min:** "Mechanics confirmed: shapes move 3 cells per arrow press; SPACE cycles selection among shapes …" Solved in 17.6 min
  with 47 actions (human: 86).
- **lp85 L3, +5.4 min:** "In L1/L2, red pointer = one direction, green = other". One probe click, then BFS. lp85 L2, L3 and L4 used 12, 26 and 14 actions against the human
  38, 31 and 16.
- **Code carried over:** in their first 3 calls, 35 of 37 level starts reuse a code line or function name from the previous level's last 5 turns. On average
  19.6% of long lines (≥25 characters) are identical to one there (median 14.3%). Against other games' previous levels, the figure is 3.1%.

**Stale assumptions from the previous level (in each case the model's own restatement or assumption, not text quoted from the history):**
- **sp80 L2, +6.0 min:** "Plan (mirroring level 1's solution: catch the block on the paddle → level completes). In level 1, catching alone completed the
  level." The RIGHT, SPACE pair failed. The model then reran the same pair at +46.2 and +56.3 min and spent the time in between re-deriving L1's win from
  animations ("Level 1 won, level 2 didn't", +22.7 min). Unsolved after 74.7 min, with 6 actions against the human 58.
- **tu93 L3, +5.9 min:** it carried an L2 guard rule, "Guard trigger = player stops 1 maze-cell away in same row (vertical safe)". A 9-move plan died on move 4
  (+9.2 min), and the level had 6 GAME_OVERs. At +63.8 min: "I claimed that earlier but it might be wrong. … That was probably my assumption, not
  observation." Unsolved after 93.4 min.
- **ls20 L2:** a 41-step plan with a 1-step margin, which assumed L1's timer rate. It failed at step 22 (+22.4 min: "maybe the level's mechanic is
  entirely different from level 1!"). At +28.9 min: "the timer decreases by 2 cells per step → 21 steps total!" (L1 used 1 cell per step). Unsolved.
- **wa30 L2, +4.6 min:** "the level-1 mechanic was NOT pure Sokoban … pure pushing worked in level 1 too". A 13-move plan at +19.0 min showed from the
  first move that the new orange block moves with the player. Censored at 29.6 min.
**Knowledge needed after it had left the context:**
- **su15 L2, +56.2 min:** "I don't have level 1's details in this session's context." In s5i5 L2, vc33 L4 and sp80 L2 the model went back through `history` frames or
  `animation(action_num=…)` to re-derive earlier wins, 19–67 min into the level.

## 6. Recommendation for later levels

**(b), keeping the history, is the better-supported option, with a harness-written summary added and nothing cleared. The evidence is weak.**

**Against (a) clearing the previous level's messages:**
- The raw previous-level turns are what the first calls work from: the recap in 34 of 37 level starts and the reused code (20% against 3%).
- They leave by themselves after about 20 min.
- No transition was found where the raw text, rather than the model's own recap, misled it.
- A summary would carry the same stale beliefs as the recaps: "catching alone completed the level", "vertical safe".
- exp-035 (lesson 0022) found that removing past reasoning from the history halved the score. That was a different cut from this one, but it points the same way.

**For (b), weak:**
- Two of the four stale-assumption failures would have been caught by one test action before the multi-action plan:
  - ls20: one move shows the bar drops 2 cells per step, not 1.
  - wa30: the first move already showed the orange block following the player.

  The other two would not:
  - sp80: the carried plan was itself only 2 actions. The cost came from re-deriving L1 for about 37 min and retrying the same pair twice.
  - tu93 L3: the guard rule could only be tested by risking a GAME_OVER.
- Probing first solved 55% of levels against 40% (p = 0.5).
- Probes cost a median of 1 action, and actions are not the binding constraint. The real cost is about one extra call, roughly 2.5 min, unless the probe is folded into the
  inspection call. For correct carried plans such as sc25 L2 and re86 L3, the cost is pure loss.

**Neither option touches the largest term.** A later level needs about as many calls as L1; each call takes about 160 s on the saturated server.
Either option can save or lose only a few calls per level start, a few minutes each; that sets the upper limit on the gain.

**Low-risk pieces that do not depend on the (a)/(b) choice:**
- At a level-up, the harness should write the cross-level note itself: the confirmed rules plus the actions that completed the level. Today it is cleared, and it was non-empty at only 1 of 38 level starts.
- At a level start, `previous_frame`, `last_transition` and `last_action_result` should be reset or labelled; 5 level starts spent reasoning on the transition diff.

**What would decide:** a public-25 A/B with at least 2 seeds per arm, measuring:
- L2+ solve rate;
- calls and actions per solved L2+ level;
- the number of level starts whose first multi-action plan fails on a changed element.

## Appendix: all level starts (L2+)

Time columns are in minutes. "Previous-level turns gone after" is the time from the level-up until the previous level's turns have left the chat history.

| game and level | level start (run min) | min / calls to 1st action | 1st action | 1st batch (actions) | solved | min / calls on level | actions / human | previous-level turns gone after (min) |
|---|---|---|---|---|---|---|---|---|
| ar25 L2 | 4.6 | 5.6 / 4 | probe | 4 | yes | 79.2 / 32 | 75 / 50 | 22.3 |
| ar25 L3 | 83.8 | 5.8 / 2 | carried plan | 14 | no | 48.4 / 20 | 74 / 75 | 27.1 |
| cd82 L2 | 50.9 | 7.8 / 3 | probe | 8 | no | 81.1 / 30 | 76 / 8 | 19.8 |
| cn04 L2 | 10.5 | 41.3 / 15 | probe | 1 | no | 122.3 / 45 | 16 / 54 | 24.0 |
| ft09 L2 | 7.6 | 3.2 / 2 | probe | 1 | yes | 22.1 / 9 | 21 / 12 | 13.7 |
| ft09 L3 | 29.6 | 14.6 / 5 | probe | 1 | yes | 22.8 / 8 | 16 / 23 | not before level end |
| ft09 L4 | 52.4 | 11.6 / 4 | carried plan | 7 | yes | 62.1 / 23 | 25 / 28 | 25.9 |
| ft09 L5 | 114.6 | none | none | 0 | no | 17.5 / 6 | 0 / 65 | not before level end |
| g50t L2 | 131.0 | none | no calls | 0 | no | 1.0 / 0 | 0 / 175 | not before level end |
| ka59 L2 | 53.7 | 8.6 / 3 | probe | 1 | yes | 59.2 / 22 | 78 / 109 | 22.5 |
| ka59 L3 | 112.9 | 4.6 / 2 | carried plan | 2 | no | 19.1 / 7 | 23 / 51 | not before level end |
| lf52 L2 | 11.6 | 4.2 / 2 | probe | 1 | yes | 96.3 / 35 | 76 / 81 | 23.9 |
| lf52 L3 | 107.8 | 6.5 / 3 | carried plan | 1 | no | 24.2 / 9 | 16 / 60 | 19.8 |
| lp85 L2 | 8.6 | 8.2 / 4 | probe | 1 | yes | 35.0 / 13 | 12 / 38 | 17.6 |
| lp85 L3 | 43.6 | 10.3 / 4 | probe | 1 | yes | 37.2 / 13 | 26 / 31 | 25.5 |
| lp85 L4 | 80.8 | 8.4 / 3 | probe | 1 | yes | 30.0 / 11 | 14 / 16 | 19.6 |
| lp85 L5 | 110.8 | 7.1 / 3 | probe | 1 | no | 21.8 / 9 | 6 / 41 | not before level end |
| ls20 L2 | 51.4 | 22.3 / 8 | carried plan | 22 | no | 80.9 / 31 | 52 / 123 | 22.4 |
| r11l L2 | 22.1 | 11.7 / 4 | probe | 1 | no | 109.9 / 37 | 37 / 33 | 17.9 |
| re86 L2 | 13.1 | 15.7 / 6 | carried plan | 14 | yes | 32.5 / 12 | 44 / 42 | 29.7 |
| re86 L3 | 45.6 | 17.4 / 6 | carried plan | 47 | yes | 17.6 / 6 | 47 / 86 | not before level end |
| re86 L4 | 63.1 | 5.9 / 2 | probe | 6 | yes | 45.3 / 15 | 140 / 108 | 18.6 |
| re86 L5 | 108.4 | none | none | 0 | no | 23.6 / 9 | 0 / 189 | 17.6 |
| s5i5 L2 | 13.2 | 7.1 / 3 | probe | 3 | no | 119.1 / 45 | 48 / 89 | 22.2 |
| sb26 L2 | 5.9 | 3.3 / 2 | carried plan | 14 | yes | 48.6 / 19 | 48 / 28 | 19.4 |
| sb26 L3 | 54.5 | 13.4 / 4 | carried plan | 14 | no | 77.5 / 29 | 115 / 18 | 10.7 |
| sc25 L2 | 110.3 | 6.0 / 3 | carried plan | 3 | yes | 14.2 / 6 | 5 / 6 | not before level end |
| sc25 L3 | 124.5 | 6.7 / 3 | carried plan | 3 | no | 7.5 / 3 | 3 / 32 | not before level end |
| sp80 L2 | 57.9 | 8.8 / 3 | carried plan | 2 | no | 74.7 / 28 | 6 / 58 | 28.7 |
| su15 L2 | 31.9 | 8.5 / 3 | probe | 1 | no | 100.1 / 37 | 33 / 42 | 31.1 |
| tn36 L2 | 49.7 | 8.0 / 3 | probe | 2 | no | 82.4 / 30 | 46 / 72 | 15.0 |
| tr87 L2 | 68.4 | 40.2 / 15 | probe | 1 | no | 63.7 / 24 | 10 / 58 | 30.2 |
| tu93 L2 | 7.8 | 5.3 / 3 | carried plan | 4 | yes | 30.9 / 12 | 25 / 16 | 16.0 |
| tu93 L3 | 38.6 | 9.2 / 3 | carried plan | 4 | no | 93.4 / 35 | 43 / 34 | 18.2 |
| vc33 L2 | 19.9 | 5.1 / 2 | probe | 4 | yes | 8.4 / 3 | 14 / 18 | not before level end |
| vc33 L3 | 28.4 | 6.0 / 2 | probe | 8 | yes | 22.1 / 7 | 35 / 44 | 18.6 |
| vc33 L4 | 50.5 | 5.0 / 2 | probe | 1 | no | 81.5 / 31 | 102 / 61 | 20.1 |
| wa30 L2 | 102.4 | 19.0 / 7 | carried plan | 13 | no | 29.6 / 10 | 17 / 119 | 19.0 |

**Caveats:**
- One run and one seed; I labelled the first calls alone.
- The regex counts of earlier-level references are rough.
- Character shares count each image as 900 characters.
- Unsolved levels are censored at the run cap.
- The earlier stall analysis used a different run (thui). Its figure of 22 min for the previous level leaving the context agrees with the 19.8 min here. Its claim that new mechanics go unprobed holds here for part of the carried-plan starts (15 of 37), not as the general pattern.
