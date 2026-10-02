# Franzen's Milestone 2 demo run: where the score and the time go

One-line summary: the run is decode-bound (86% of every game's clock is token generation at ~74 tok/s per stream; prefill
1.7%, server queue 0.3%), action efficiency costs almost nothing (0.73 of 36.56 points), and the remaining 62.7 points are
levels never reached. Of the 9 levels being played when time ran out, 3 had a verified plan cut off by the 25-minute
deadline, 2 had started under 6 minutes earlier, 3 were new mechanics never decoded, and 1 had its search code fail. The
demo cannot test the priority gate (10 games, 10 slots), and that gate decides allocation on the hidden set. Ranked
levers are in section 5.

Research note, 2026-10-02, read-only analysis (no GPU used). **Measured** means computed from the run files by the
scripts below. **Estimate** means a projection, and the reasoning is given each time.

## Sources and method

- Run: Franzen's public notebook `dfranzen/arc-agi-3-milestone-2-solution`, last Save & Run output (2026-09-30 08:30-08:55
  UTC; 10 public games, 25 min per game, 10 admission slots, `TRUE_SUBMISSION=False`). Local copy:
  `/tmp/claude-0/-home-user-Arc-Agi-3-Kaggle-comp/d342458e-03bd-545b-8a6d-06bca061963e/scratchpad/m2/franzen-output/`.
  Config (notebook cell 4 and cell 12): Qwen3.8-Flash-Next W4A16 AutoRound + Albucino MTP draft, Pennyroyal SGLang
  0.5.19, FP8 KV (1,011,264 tokens), 60 Mamba slots, MAXREQ 10, SPEC_STEPS 3, 116+12 Ki harness window, 58 Ki drain,
  yield at 2,048 generated tokens, tool timeout 30 s, guards from level 2, UNDO exposed, RESET not exposed, world-model
  notes off.
- Code: `/home/user/da-fr/arc-agi-3-solution` (WRITEUP.md, ARC3-Inference/CONFIGURATION.md, `inference/agent/tool_agent.py`).
- Scripts (all in `/tmp/claude-0/-home-user-Arc-Agi-3-Kaggle-comp/d342458e-03bd-545b-8a6d-06bca061963e/scratchpad/franzen-analysis/`,
  outputs in `out/`):
  - `common.py`: loaders plus the scorer, which reproduces `benchmark.json` `final_score` for all 10 games.
  - `compute.py`: section 1. Matches all 531 harness responses to SGLang `ReqTimeStats` lines on prompt length, cached
    length and output length ±8 tokens. Output: `out/compute.txt`.
  - `decode_curve.py`: decode throughput by batch size (`out/decode_curve.txt`).
  - `levels.py`: section 2 and section 4 (`out/levels.txt`, `out/levels.json`).
  - `transcripts.py`: per-call parse of the 10 transcripts, plus counts of tool use, errors and guard events
    (`out/transcripts.txt`, `out/calls_<game>.json`).
  - `trims.py`: context trims and the long completions (`out/trims.txt`).
  - `stuck.py`: share of calls that act, and the longest stretch without an action, per level (`out/stuck.txt`).
  - `km.py`: Kaplan-Meier time to solve a level (`out/km.txt`).
  - `longreq.py`: where the completions over 6k tokens fall (`out/longreq.txt`).
  - `first_action.py`: time from a level's start to its first action (`out/first_action.txt`).
  - `dump_level.py <game> <first_action>`: condensed view of a level, used to read the 9 stuck levels
    (`out/dump_*.txt`).
- Scoring: per level `min((h/a)^2, 1.15)`, level-index weighted. **The game total is capped at the weight of the
  completed levels** (`min(Σ w·s, Σ_completed w) / Σ w`; this is what reproduces ar25 = 41.67 and sb26 = 93.34). So
  better-than-human levels only offset worse ones inside the same game.

## 0. Bottom line

1. **Time is token generation.** Per request the server spends 0.10 s queued, 0.49 s on prefill and 24.3 s on decode,
   and the harness spends 2.1 s between requests (tool execution plus prompt building). Per game that is 86.2% decode,
   7.4% tool/harness, 1.7% prefill and 0.3% queue. 94.2% of prompt tokens come from the prefix cache. Aggregate decode is
   765 tok/s at 10 running requests (median of the log lines) and 621 tok/s averaged over the wall clock. Per stream it
   is 74 tok/s. The batch curve is almost flat at the top: 729 tok/s at 8 running, 750 at 9, 765 at 10.
2. **A decision is about a minute.** The run made 532 model calls; 47% of them executed at least one action (5.3 actions
   each). Each acting call costs 3,688 generated tokens and 60 s of a game's clock, which works out to 11.4 s per action.
   Calls on the levels that ended unsolved generate 41% more tokens than calls on solved levels (2,198 vs 1,559), and
   fewer of them act (40% vs 49%).
3. **Reach is the whole loss.** The mean is 36.56. With human-level efficiency on every solved level it would be 37.29,
   so solved-level inefficiency costs 0.73 points; levels never solved cost 62.71. The solved levels used a median 0.69x
   of the human action count.
4. **The deadline binds in the demo.** No level was completed in the last 5.2 minutes. Of the 9 games cut off, 3 (ar25 L6,
   re86 L4, tu93 L4) had a verified plan that was mid-execution or ready to send. Adding those three levels would raise
   the demo mean from 36.56 to about 40.2 (estimate). On the hidden set only the ~10 games holding a slot at minute 532
   are exposed to this.
5. **The scheduler is untested here.** With 10 games and 10 slots the priority gate never blocked. On the hidden set
   (110 games, 10 slots, 532 min) the code gives each game its first slot in dispatch order and keeps it until the
   game's first context trim. In this run the first trim came at 17.0 min on average (13.6-20.8). Only after every game
   has started does the (A+B)·C priority choose who runs. A gate-binding run is needed before any scheduler change can
   be judged.
6. **The failure modes are mostly discovery.** Of the 9 unsolved levels, 3 were new mechanics or goals never decoded
   (ft09, sc25, vc33), 3 had a plan when time ran out, 2 started late (under 6 min left), and 1 had decoded the mechanic
   but its planner code failed (r11l: a 30-s tool timeout dropped all 13 retained functions).

## 1. Compute accounting (measured unless marked)

### Requests and tokens

| Quantity | Value | Source |
|---|---:|---|
| Game requests answered (harness responses, each matched to a serve.log `ReqTimeStats` line) | 531, plus 1 warm-up before the run | compute.py |
| Requests in flight at the deadline (server finished, harness had timed out) | 6, holding 9,852 generated tokens | compute.py |
| Requests per game | 34-81, mean 53 (ar25 81, re86 64, sc25 59 … ft09 34) | compute.py |
| Prompt tokens per request | mean 63,235, median 66,710, p90 103,511, max 118,992 | compute.py |
| Image tokens in the prompt | mean 8,912 (14.1% of prompt tokens; ~402 per 640×640 image) | compute.py |
| Generated tokens per request | mean 1,743, median 1,011, p90 4,110, max 12,288 (1 length cut) | compute.py |
| Reasoning share of generated tokens | 86.0% | compute.py |
| Requests generating more than 4k / 6k / 8k tokens | 10.4% / 4.3% / 2.3% of requests, carrying 38.0% / 21.4% / 13.0% of generated tokens | trims.py |
| Prefix-cache share (token-weighted) | **94.20%** (33.58M prompt tokens, 1.95M uncached) | compute.py |
| Requests over 40k tokens with less than 50% cached | 12 of 394, all at context trims | compute.py, trims.py |
| Uncached tokens per request | mean 3,671; effective prefill ~9,900 tok/s | compute.py |
| Total generated | 925,650 tokens in 25 min, about 2.2M per hour | compute.py |

### Server time per request (SGLang `ReqTimeStats`)

| | mean | median | p90 | total over the run |
|---|---:|---:|---:|---:|
| queue | 0.10 s | 0.00 s | 0.00 s | 52 s |
| prefill | 0.49 s | 0.27 s | 0.84 s | 259 s |
| decode | 24.34 s | 13.24 s | 59.35 s | 12,923 s |

- Per-stream decode speed (output tokens over decode time): mean 75.4, median 74.1 tok/s. It does not depend on prompt
  length: 72.8 tok/s for 0-20k-token prompts and 75.9 for 100-120k (the hybrid linear-attention model). Source:
  decode_curve.py.
- Aggregate decode, from the `Decode batch` log lines during the run: mean 671, median 722, p90 833 tok/s. The MTP draft
  averages 2.67 accepted tokens per step (of 4).

| Running requests | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| Median gen tok/s | 473 | 570 | 667 | 729 | 750 | 765 |
| Per stream | 95 | 95 | 95 | 91 | 83 | 77 |

  Each request beyond 8 adds about 2-3% throughput. KV in use peaked at 965,248 of 1,011,264 tokens (95%, mostly radix
  cache); Mamba slots peaked at 0.67 (40 of 60).

### Where each game's 25 minutes went

Measured over 10 games × 1,500 s:

| decode | tool + harness gap | prefill | server queue | before the first request | after the last request |
|---:|---:|---:|---:|---:|---:|
| **86.2%** | 7.4% | 1.7% | 0.3% | 0.3% | 4.0% |

- The gap between consecutive requests of a game (tool execution plus harness work) has mean 2.14 s, median 1.46 s,
  p90 4.0 s, max 31.5 s. The two 30-s tool timeouts are in the tail.
- In the real submission `save_request_logs=False` and diagnostics are minimal (cell 16), so this gap is probably smaller
  there (estimate).
- The 4.0% after the last request is mostly sb26, which won all 8 levels at 17.8 min and left its slot idle; a 10-game
  demo cannot refill a slot.

### Slots busy

- Requests resident in the server (1-s sampling of request intervals): mean 8.80 of 10. All 10 were resident 39.3% of
  the time. The mean was 9.4-9.6 in minutes 0-10, 8.9-9.2 in minutes 10-17.5, and 7.1-8.2 after sb26 finished.
- The `Decode batch` line distribution agrees: 10 running 39%, 9 31%, 8 16%, 7 or fewer 14%.

### The priority gate and waiting (measured: zero)

- With 10 games and `ARC3_MAX_ACTIVE_STREAMS=10`, every acquire and handover was admitted at once, so time waiting at the
  gate is 0. The server queue was at most 1 request.
- The tail fade started at 20.5 min (log line). With every game already admitted it changed nothing.

### Time per decision

- Per request: 27.1 s (queue + prefill + decode + gap).
- Per acting call: 59.8 s. Per action: 11.4 s.
- Per level, before the first action: median 0.87 min (L1 0.56, L2+ 0.97). Summed over all 49 levels this is 58.7
  game-minutes, 24.1% of the 243 active game-minutes (first_action.py).
- 91 of 348 turns ended at the 2,048-token yield.

### Context trims (measured)

- 12 trims, 1-2 per game. Each takes the prompt from 111-119k down to 52-60k tokens. The next request had 0 or 4,672
  tokens cached, so it re-prefilled about 55k tokens (about 5.5 s).
- First trim per game: mean 17.0 min (13.6-20.8). The prompt grows 2.3-3.7k tokens per request.

### Hidden-set implications (estimates from the code and these rates)

- **Slot holding.** `_PriorityGate` (tool_agent.py:1976) hands a slot over only inside `_maybe_handover`
  (tool_agent.py:6372), which fires only after a context trim. Games that have never started wait at priority
  2,000,000 − dispatch index (tool_agent.py:3525, solver.py:625). A started game re-queues at (A+B)·C·100 ≈ 500-1,600.
  So the first pass is strict round-robin in dispatch order.
- **First pass cost.** Each of the 110 games holds a slot for about 17 min, or less if it wins first. That is about
  1,870 slot-minutes, ~35% of 5,320, so the first pass ends around minute 187. During it, a game that trims while making
  progress waits behind every unstarted game.
- **Later quanta.** Re-admissions last about 58k / ~3k tokens per request ≈ 19 requests ≈ 9 min.
  `ARC3_CONTEXT_DRAIN_TOKENS` and `LOCAL_ANALYZER_CONTEXT_WINDOW` therefore also set the scheduler's time quantum.
- **Tokens per game.** About 532 min × 620-765 tok/s = 20-24M generated tokens, or 180-220k per game on average. The
  demo gave each game about 92k.

## 2. Per game and per level (measured)

Cells read `agent actions / human actions → level score [minutes on level, generated tokens, requests]`. `*` marks the
level being played when time ran out. Apart from the harness's warm-up RESET in ar25 (counted as one L1 action), there was
no RESET and no UNDO.

| Game | Levels | Score | Levels in order |
|---|---|---:|---|
| ar25 | 5/8 | 41.67 | 18/32→1.15 [1.8m, 7k, 7]; 26/50→1.15 [5.8m, 23k, 21]; 52/75→1.15 [6.2m, 22k, 20]; 22/37→1.15 [1.7m, 6k, 5]; 32/89→1.15 [4.1m, 15k, 11]; **L6\* 32/159** [5.4m, 19k, 17] |
| ft09 | 4/6 | 47.62 | 4/43→1.15 [1.8m]; 7/12→1.15 [0.6m]; 16/23→1.15 [3.2m]; 24/28→1.15 [2.1m]; **L5\* 42/65** [17.4m, 64k, 15] |
| lp85 | 5/8 | 41.67 | 20/17→0.72 [4.5m]; 9/38→1.15 [6.8m]; 19/31→1.15 [3.4m]; 15/16→1.14 [3.1m]; 11/41→1.15 [2.1m]; **L6\* 38/60** [5.2m, 19k, 7] |
| r11l | 2/6 | 14.29 | 7/22→1.15 [4.1m]; 12/33→1.15 [10.9m, 43k]; **L3\* 1/51** [10.1m, 37k, 16] |
| re86 | 3/8 | 16.67 | 32/26→0.66 [2.3m]; 43/42→0.95 [5.0m]; 59/86→1.15 [3.9m]; **L4\* 93/108** [13.9m, 48k, 27] |
| sb26 | 8/8 | 93.34 | L1-L7 all ≤ human (1.15 each, 0.4-4.7 min each); L8 43/18→0.18 [7.6m, 27k, 11] |
| sc25 | 3/6 | 28.57 | 24/36→1.15 [9.7m]; 5/6→1.15 [1.1m]; 22/32→1.15 [3.8m]; **L4\* 106/83** [10.4m, 38k, 24, 2 deaths] |
| tr87 | 4/6 | 47.62 | 45/54→1.15 [11.5m, 46k]; 51/58→1.15 [4.7m]; 41/40→0.95 [2.3m]; 29/45→1.15 [0.9m]; **L5\* 18/71** [5.6m, 19k, 8] |
| tu93 | 3/9 | 13.07 | 48/19→0.16 [3.9m]; 15/16→1.14 [6.5m, 1 death]; 19/34→1.15 [5.1m]; **L4\* 25/42** [9.6m, 37k, 14, 2 deaths] |
| vc33 | 3/7 | 21.08 | 18/7→0.15 [3.3m]; 12/18→1.15 [1.1m]; 27/44→1.15 [5.8m]; **L4\* 26/61** [14.7m, 55k, 25] |

Mean 36.56, matching summary.txt; the cap is 1.15. The notebook log's interim summary at minute 10 (14.21) also matches
`levels.py`.

### Where the time went

- **Solved vs final level.** 62% of game-minutes and 64% of generated tokens went to the 40 solved levels; 38% and 36%
  went to the 9 levels the games ended on.
- **Cost of a solved level.** Median 3.3 min (L2+) to 3.6 min (L1), about 13-15k generated tokens and 7-10.5 requests.
  The mean is 14.7k tokens.
- **Time to solve** (Kaplan-Meier over 49 attempts, 9 censored at the deadline). P(solved within):

  | 2 min | 5 min | 7.5 min | 10 min | 12.5 min |
  |---:|---:|---:|---:|---:|
  | 0.24 | 0.59 | 0.76 | 0.81 | 0.88 |

  By generated tokens: 0.35 by 10k, 0.62 by 20k, 0.82 by 40k.
- **Score if the run had stopped at minute m** (game cap applied):

  | Minute | 2.5 | 5 | 7.5 | 10 | 12.5 | 15 | 17.5 | 20 | 25 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | Score | 2.17 | 3.75 | 9.60 | 14.21 | 20.90 | 24.95 | 27.78 | 36.56 | 36.56 |

  The last level fell at 19.79 min (lp85 L5). The elasticity of score with respect to time is 0.75-0.81 between
  12.5-15 min and 25 min.

### Where each game was at the deadline

From reading the transcripts (section 3 has the detail):

| Game, level | Minutes on level | State at the deadline |
|---|---:|---|
| ar25 L6 | 5.4 | Mirror-piece placement decoded. Piece 1 was in place and piece 2 was mid-move: 7 of 15 LEFT presses had run when the deadline refused the rest. About 20 actions were left (the rest of the move plus the mirror lines). |
| ft09 L5 | 17.4 | New key glyph kinds not decoded. A checkerboard target derived by CSP failed in both polarities (36 clicks). Still enumerating geometry/mapping hypotheses. |
| lp85 L6 | 5.2 | Started at 19.8 min. Was measuring each arrow's permutation period with click loops (29 actions in one call). |
| r11l L3 | 10.1 | Mechanic decoded (the blob sits at the centroid of its anchors; anchors may sit on walls). Was rewriting a beam/A* planner for the fifth time. 1 action in 10.1 min. |
| re86 L4 | 13.9 | Colour-station mechanic decoded at about 12 min. The X was placed; the plus route was 9 of 21 moves done when cut. |
| sc25 L4 | 10.4 | Trigger for the "clearing" submission not decoded. Third attempt, already 106 actions against a human 83. |
| tr87 L5 | 5.6 | Started at 19.4 min. Enumerating 18 rule-set candidates for the editable legend. |
| tu93 L4 | 9.6 | Patrol model and a 17-move plan verified in its own simulator. The request that would execute it was in flight at the deadline. |
| vc33 L4 | 14.7 | New mechanic (panes/gates) not decoded. Probing the A0..A4 buttons with about 31 clicks of budget left. |

**Estimate: the three cut plans.** Each needed about 1-2 more minutes. Completing them at the planned action counts gives
ar25 +16.7, re86 +11.1 and tu93 +8.9 game points, which lifts the demo mean from 36.56 to ~40.2 (+3.7). This assumes the
plans work as simulated. On the hidden set only the games in a slot at minute 532 are exposed.

## 3. Failure taxonomy for the 9 unsolved levels (read from the transcripts)

| Level | Primary mode | Secondary factors (evidence) |
|---|---|---|
| ar25 L6 | **Plan known, executing when time ran out** | Perception: 4-connected same-colour components split each piece into several parts. Tiling attempts failed ("A on board all B? False", "B placements 0") and a two-action probe was needed to learn which parts move together. About 2.5 min and 8 calls. |
| ft09 L5 | **Goal/clue semantics not decoded** (new key glyph kinds: grayscale and chromatic ring symbols) | **Wrong belief tested at cost:** the "checkerboard" CSP was applied in both polarities, 9 + 27 clicks, with no win. **Tool timeout:** a 2^26 brute force hit the 30-s limit. 7.1-min stretch without an action. 4,235 tokens per request, including two near-12k completions. |
| lp85 L6 | **Out of time while decoding** (started at 19.8 min) | Action-costly measurement: repeated arrow clicks to read permutation periods (29 actions in one call). |
| r11l L3 | **Mechanic decoded, planner code failed** | **Tool timeout** (A* over anchor configurations, 30 s): the harness dropped all 13 retained functions (`plan_chain`, `layout2`, `foot_ok`, …), and the model noted "the search timed out (and all retained functions were cleared!)". **Sandbox:** `open` is not defined; `import time` is blocked. Beam/corridor search bugs (rounding variants coupled; "depth 0 states 1 children 0"). 6.8-min stretch without an action. |
| re86 L4 | **Plan known, executing when time ran out** | **Wrong first belief:** "cover all 6 dots regardless of colour" cost about 30 actions with no win. **Budget bar misread:** "the purple counts are unreliable?!". The animation timeline produced the breakthrough ("orange pulse" = colour transfer). |
| sc25 L4 | **Mechanic not decoded** (what makes the clearing submission work) | **2 budget deaths.** **About 9 actions burned on purpose** (two burns: 3 + 6 actions) to force a reset ("burn the budget and die, then … execute the optimal plan"), because RESET is not exposed. 4 snippets wrapped `action()` in try/except to get past the stale-state guard. Already over the human count (106 vs 83). |
| tr87 L5 | **Out of time while decoding** (started at 19.4 min) | None notable: cheap probes, hypothesis enumeration in code. |
| tu93 L4 | **Plan known, ready to send** | **2 deaths:** one collision probe, and one budget death first misdiagnosed as a collision (the bar dropped 3 units per move, about 21 moves per attempt). Probing had spent most of the attempt's budget. |
| vc33 L4 | **New mechanic not decoded** (gates/panes) | **Lost to a trim:** the trim at 16.4 min (116.5k→56.2k tokens) dropped the earlier button coordinates. At about 18 min the model pressed col 34 "from level 3's layout", read the no-op as a "cap rule" (a false belief that cost about 3 calls), then failed to rebuild level 3's win from `transitions` (6 calls). 6 try/except wrappers around actions. |

### Counts

**Primary modes:**

| Mode | Levels |
|---|---|
| Plan known or ready when time ran out | 3 (ar25, re86, tu93) |
| Out of time while decoding, started under 6 min before the end | 2 (lp85, tr87) |
| Mechanic or goal not decoded after 10-17 min | 3 (ft09, sc25, vc33) |
| Mechanic decoded, planning code failed | 1 (r11l) |

**Contributing factors:**

| Factor | Count |
|---|---|
| Wrong belief tested or defended at cost of actions or calls | 4 levels (ft09, re86, vc33, tu93) |
| Budget-bar trouble: deaths, burns, misreads | 4 levels; 3 of the run's 5 deaths were budget deaths |
| Tool errors decisive | 1 (r11l); contributing 1 (ft09) |
| Context trim lost something that was then needed | 1 of 12 trims (vc33) |
| Perception: segmentation does not give rigid bodies | 1 (ar25) |
| Action-costly probing loops | 2 (lp85, sc25) |
| Goal misread | 1 (ft09); re86's first goal belief was wrong but later fixed |

### Harness events, whole run (transcripts.py, trims.py)

- **Tool errors.** 34 Python exceptions in 532 calls (6.4%): TypeError 12, NameError 10 (3 of them `open`), IndexError 6,
  ValueError 3, KeyError 2, ImportError 1 (`time`). Plus 2 tool timeouts at 30 s (ft09, r11l). Most NameErrors are
  variables lost between snippets.
- **Guards.** StaleStateActionError 6 (ar25, re86, sb26, sc25×2, vc33). TerminalStateActionError 1. Batch stopped at a
  level completion 4.
  - 13 snippets wrapped actions in try/except, mostly to continue through no-ops: vc33 6, sc25 4, tu93 2, re86 1.
  - One false no-op: in re86 an UP that moved the line from row 63 to row 60 was flagged as a no-op because the 4-cell
    border is ignored.
  - The stale-state guard is off on level 1 (`ARC3_GUARDS_FROM_LEVEL=2`). On tu93 L1 a loop of single `action()` calls
    ran 34 moves, 18 of them no-ops (the BFS assumed 1-cell moves; the player moves 2): 48 actions against a human 19.
- **No-op actions** (echo NO-OP: unchanged inside the border): 71 of 1,327 (5.4%). Board fully unchanged: 26.
- **UNDO.** Offered in 2 games (ar25, sb26), used 0 times. RESET is not offered (`EXPOSE_RESET` off).
- **frame_diff.** Called in 41 of 532 calls (7.7%), in 9 games.
- **Animation variables.** Read in 45 calls (8.5%), mostly sb26 (16) and sc25 (18). The transient-pixel hint appeared in
  84 prompts, and re86's breakthrough came from `last_animation_frames`.
- **Budget/bar reasoning.** Appears in 118 of 532 calls' thinking (22%). 63 calls ran bar-reading code (sc25 42).
- **Context trims.** 12. The prompt drops by 56-65k tokens each time. Functions retained for the game scope survive a
  trim; old reasoning and coordinates do not (the vc33 case above).
- **Yields.** 91 of 348 turns ended at the 2,048-token yield. Each one appends a full resume prompt with images, so
  context grows 2.3-3.7k tokens per request.

## 4. Action efficiency (measured)

- **Solved levels.** 40, median 0.69x the human count, mean level score 1.042. 7 were over the human count, 3 over 2x:
  - tu93 L1 48/19 (the 18-no-op loop above);
  - vc33 L1 18/7;
  - sb26 L8 43/18 (a final level, where no surplus can offset it: −6.66 game points).
- **Human efficiency counterfactual.** Raising every solved level to at least human efficiency gives 37.29 instead of
  36.56 (+0.73). Only sb26 (+6.66), vc33 (+0.35) and tu93 (+0.26) move; the completed-share cap absorbs the rest.
- **Where actions go on solved levels.** Exploration probes before the first plan, and execution of plans built on a
  wrong motion model (tu93 L1). Deaths on solved levels: 1 (tu93 L2).
- **Where actions go on unsolved levels.** These matter if the level is later solved, and on bar games they consume the
  attempt budget:
  - deliberate budget burns (sc25 about 9);
  - repeated failed hypotheses (ft09 36 clicks, re86 about 30);
  - period-measurement loops (lp85 29 in one call);
  - stale coordinates (vc33 2 no-op clicks plus probes).
  - sc25 L4 is already past the human count at 106/83, so even a win there would score at most (83/107)^2 ≈ 0.60.
- **Conclusion.** Action efficiency is not where this agent loses score. Reach is. Spending actions to learn faster is
  correct up to the bar budget. The real action costs are budget deaths and burns, because they waste attempts and calls.

## 5. Levers, ranked by estimated LB gain per GPU-hour of testing

### How the gains are estimated

- **Throughput changes.** LB gain ≈ 27.89 × ((1+Δ)^e − 1), where Δ is the change in useful decode throughput and e ≈ 0.75
  (range 0.5-1.0). The elasticity comes from this run's score-vs-time curve (0.75-0.81 between 12.5-15 and 25 min) and our
  exp-054 (0.95 from 60 to 132 min; `docs/research/strategy-sep29/compute-and-allocation.md`).
- **Behavioural fixes.** Gains are judgement calls tied to how many stuck levels the fix plausibly converts, scaled down
  for the hidden set.
- **Test cost** is GPU-hours on one RTX PRO 6000:
  - demo-10 run: ≈ 0.6 h (8-min server start plus a 25-min run);
  - hidden-shape run (25 games × 121 min, gate binding): ≈ 2.2 h;
  - local replay bench: ≈ 0.4 h per serving config.
- **Noise.** Identical runs differ by 1.5-2.3x on the LB (lesson 0018). Only the serving bench and the event counts
  (deaths, burns, timeouts) can be read from a single run.

### The ten levers

| # | Lever | Evidence (sections above) | Arm | Test, cost | Gain [range] | Gain/GPU-h |
|---|---|---|---|---|---:|---:|
| 1 | **Tool-sandbox hygiene:** tool timeout 30→90 s; keep retained functions when a call times out; allow `time` and a read-only `open` substitute | 2 timeouts in 532 calls. The r11l L3 timeout dropped 13 retained functions (`function_retention`: "Your previous function plan_chain is no longer retained …"); the level then had 1 action in 10.1 min. 4 sandbox import/name errors. | Cell 4: `LOCAL_ANALYZER_TOOL_TIMEOUT='90'`. Code: `tool_agent.py` `_record_retained_functions` (line 4614) keeps `previous` when the result has no `keepable_functions` because of a timeout; `python_tool_sandbox.py` `SAFE_MODULES` (line 50) gains `time`. | Unit test plus ride-along smoke in any demo run; ~0.3 GPU-h | +0.2 [0, +0.5] | ~0.7 |
| 2 | **Serving micro-tuning:** MTP steps 2/3/4, accept thresholds, 10→12 streams and slots | Decode is 86% of each clock. Accept length 2.67 of 4. tok/s 729/750/765 at 8/9/10 running. 7.4% of stream-time is tool gaps, during which the batch drops to 9. | Cell 12: `SPEC_STEPS`, `SPEC_ACCEPT_SINGLE/ACC`, `MAXREQ`, `CUDAGRAPH_MAXBS`. Cell 4: `ARC3_MAX_ACTIVE_STREAMS`. | Replay the 531 recorded requests from `*_requests.jsonl` at matching concurrency on the local box; tok/s and seconds per request are exact. 3 configs ≈ 1.0 h, plus 1 confirm demo 0.6 h. | +0.8 [0, +2.4] (Δ 0-12%) | ~0.5 |
| 3 | **Expose RESET** (then a harness budget readout) | 3 of 5 deaths were budget deaths (sc25×2, tu93). sc25 burned about 9 actions on purpose to force a reset; re86 planned a burn. Bar reasoning appears in 22% of calls. | Cell 4: `EXPOSE_RESET='on'` (ARC3_ACTION_INFO is already on, so RESET gets described). Follow-up code: a bar tracker in the action result (`tool_agent.py`, action-result builder), reporting monotone border-cell counts per action and actions left. | 1 demo run: count burns, deaths and RESET use; ~0.6 GPU-h | +0.3 [0, +0.8] | ~0.5 |
| 4 | **Endgame flush:** a time-left line, and "execute your verified plan now" in the last ~3 min | 3 of 9 cut levels had a verified plan (worth +3.7 demo points). Hidden: only the ~10 games in a slot at the end. | Code: the resume and opener prompt in `tool_agent.py` adds `time_remaining_seconds` (already in action results) once under ~5 min. | Behaviour check only; ~0.3 GPU-h. **Inflates demo scores far more than LB.** Never read a demo A/B of this as hidden evidence. | +0.15 [0, +0.4] (LB) | ~0.5 |
| 5 | **Stale-state guard from level 1** | tu93 L1: a 34-move loop with 18 no-ops, 48 actions against a human 19. The guard would have stopped it at the first no-op. Franzen chose 2, reason unrecorded. | Cell 4: `ARC3_GUARDS_FROM_LEVEL=1`. | Ride-along with #3, or 1 demo run; 0.6 GPU-h | +0.1 [−0.2, +0.4] | ~0.17 |
| 6 | **Perception helpers:** co-moving component groups ("parts that moved together") in `frame_diff`, and a cell-lattice detector | ar25 L6 spent about 2.5 min and 2 actions finding which components form a piece. 5 games wrote their own lattice parsers. Orientation before the first action is 24.1% of active time. | Code: `inference/utils/frame_diff.py` (group `moved` entries with equal displacement); a `grid_cells()` helper next to `segmentation`; prompt line under `ARC3_FRAME_DIFF_HINT`. | Offline replay of recorded transitions (CPU), then 3 demo runs ≈ 1.8 GPU-h | +0.4 [0, +1.2] | ~0.2 |
| 7 | **Level-win record kept outside the trimmed region** | vc33: the trim at 16.4 min dropped the button coordinates, followed by a stale col-34 press, a false "cap rule", and a failed rebuild of the L3 win from `transitions`. 12 trims each drop 56-65k tokens. | Code: `tool_agent.py` `_trim_messages_for_context` (line 6454) keeps a harness-built record per completed level (winning action list, frame_diff of the last actions, button and object coordinates) right after the system prompt. It is fixed at level-up, so the prefix stays cacheable. | Offline check plus 3 demo runs ≈ 1.8 GPU-h | +0.4 [0, +1.0] | ~0.2 |
| 8 | **Cap the long-completion tail** (max output 12k→6k, with the effort ladder) | 10.4% of requests carry 38% of tokens; 13 of the 25 completions over 6k were on levels that were solved, and 15 of 25 led to an action. The sign is unclear; lesson 0022 says past reasoning is load-bearing. | Cell 4: `LOCAL_ANALYZER_MAX_OUTPUT=6*1024`, `ARC3_REASONING_EFFORT_LADDER`. | 4 demo runs ≈ 2.4 GPU-h; read tokens per level and levels | +0.3 [−2, +2] | ~0.13 |
| 9 | **Gate: hand over at a yield when a level has stalled** | Handover happens only at trims, so a stuck game keeps its slot for the whole quantum (about 17 min first, about 9 min later). r11l L3 had 1 action in 10.1 min; ft09 L5 had a 7.1-min stretch with no action. But 3 of the 5 levels that passed 40k tokens were solved or nearly solved, so the sign is unclear. | Code: `tool_agent.py` `_maybe_handover` also fires on a `turn_token_budget` yield when C(level) < ~0.6 and a waiting snapshot beats this game by a margin. | Needs a binding gate: 2 hidden-shape runs ≈ 4.4 GPU-h | +0.5 [−1, +2] | ~0.11 |
| 10 | **Drain target as a scheduler quantum** (58k → 70k) | The quantum after the first trim is about (116k − drain)/3k per request. A larger drain keeps more history and makes handovers more frequent. Untested at 110 games. | Cell 4: `ARC3_CONTEXT_DRAIN_TOKENS`. | 2 hidden-shape runs ≈ 4.4 GPU-h | +0.3 [−0.5, +1.5] | ~0.07 |

### Why the ranking comes out this way

- **The big lever is token throughput, not harness text.** The data points to useful tokens per hour on every axis:
  - the run is decode-bound;
  - 3 of 9 unsolved levels needed only minutes;
  - the score curve is still steep at 20 min.
  
  Nothing else in this list is worth more than ~1 LB on its own. Levers 1, 3, 4 and 5 are cheap hygiene, and each
  removes a failure seen in this run. Lever 2 is the only one with both a measurable effect and a plausible gain above
  +1.
- **Tufa's 52.5 is 1.88x Franzen's 27.89.** At e = 0.75 that would take about 2.3x more useful tokens per level-hour
  (1.88^(1/0.75)). None of the knobs above gets near that. A step that size would have to come from the model or
  engine (draft model, quantization, kernels) or from cheaper decisions: 3,688 tokens per acting call, 53% of calls not
  acting, and long thinking on new-mechanic levels.
- **Before any of 9 or 10:** run one hidden-shape calibration (25 games × 121 min, the planned exp-071, with
  `ARC3_DIAG_CONCURRENCY=1`). It measures what this demo cannot: gate wait times, quantum lengths, and how many slot-minutes
  go to games that never pass level 1.

### Suggested order

1. A CPU-only patch for #1 and #7, plus offline replay checks for #6 and #7.
2. The serving bench for #2 on the local box.
3. One demo run with the hygiene bundle #1 + #3 + #5, read on events (timeouts, burns, deaths, RESET use), not on score.
4. The hidden-shape calibration run.
5. Arms #6 and #7, then #8, as repeated demo pairs.
6. Gate arms #9 and #10 last, against the calibration run.

## 6. What this data cannot tell us

- **One run, 10 games.** Per-game scores are single draws. The failure taxonomy is one reader's labelling of 9 levels,
  checked against our exp-054 notes for the same games (`docs/research/levels2plus-exp054/notes.md`), whose ground truth
  agrees for ar25, re86, tu93, vc33 and ft09.
- **The demo is deadline-dominated; the hidden set is gate-dominated.** Demo A/B differences on endgame behaviour (#4)
  or per-game time do not transfer.
- **No gate data.** Every hidden-set scheduling statement above comes from reading the code and applying this run's
  rates. None is measured.
- **The serving gains are unmeasured** until the replay bench runs. The Franzen write-up reports his own config search,
  so the headroom may be small.
- **The elasticity** (score per unit of time) comes from public-game curves. Lesson 0018 shows public gains transfer to
  the LB only partially, so treat every LB figure in section 5 as an order of magnitude.
