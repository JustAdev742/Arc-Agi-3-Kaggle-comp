Summary: Franzen's agent loses tokens to rebuilding planners that time out, guessing its own action budget, and re-deriving facts its 116 Ki window already trimmed away, so the best value per GPU-hour comes from six harness additions. Cheapest first: fix the sandbox so a timeout keeps its work; an exact budget-bar meter (CPU check: found the bar in 9 of 10 games and read "0 left" before all 3 budget deaths); a harness search service with guarded execution; an exact ledger re-pinned at every context trim; hypothesis checks with a discriminating probe; and a goal monitor. Together they are worth about +2 to +5 LB, which does not close the gap to Tufa on its own.

# New methods on Franzen's base: deciding better per token (research, 2026-10-02)

Scope: read-only research for the brief "take Franzen's solution (LB 27.89) past Tufa Labs (52.51)". No GPU, no
Kaggle, nothing committed. Sources: Franzen's `WRITEUP.md`, `ARC3-Inference/CONFIGURATION.md` and harness code
(`/home/user/da-fr/arc-agi-3-solution`, commit as vendored); his notebook's env settings; one real run of his agent
(the 10-game, 25-minute demo of 2026-09-30: transcripts, request logs, per-action boards in `m2/franzen-output/`);
our analyses (`strategy-sep29/agent-model-training.md`, `levels2plus-exp054.md`, `hard-games-level1.md`,
`road-to-100-v2.md`, `frontier-100-systems.md`, `win-conditions-dev.md`, `stall-analysis-thui.md`, lessons
0001-0026). Scratch work (digests, timelines, two CPU prototypes) is in
`scratchpad/new-methods/` (paths at the end).

Caveat on the run: it is one seed, 10 games, 25 minutes each (about half the ~48 slot-minutes a hidden game gets),
on public games. It shows how the agent spends tokens. It cannot rank arms.

## 1. How Franzen's agent spends its tokens (10-game demo run, read first-hand)

Score 36.56 on the 10 games; 40 levels solved. Measured from the transcripts and request logs
(`scratchpad/new-methods/timeline.txt`, `toolstats.py`):

| Fact | Number | What it means |
|---|---|---|
| Solved levels at or above 1.0 of the human score | 33 of 40 (the 7 below are mostly level 1s) | It loses on reach, not on efficiency. This matches exp-054. |
| Generated tokens spent on the level each game ended on | 41% (ft09 69%, vc33 64%, tu93 58%, re86 54%) | The stuck level is where tokens go. |
| Wall time in turn sections with no action | 87 of 238 game-minutes (37%) | Analysis is where the time goes. |
| Turns that execute one already-planned action with less than 1k reasoning characters | 1.8 game-minutes in total | "Plan known, execution slow" is **not** a loss mode here. |
| Requests over 4k generated tokens | 10% of requests, 38% of tokens; median request 1,010 tokens | The long requests are mostly level-start analysis (sb26 L8, lp85 L2, r11l L2). |
| Context trims (116 Ki down to ~55 Ki) | at least one in **every** game within 25 minutes (12 in total) | On the hidden set (~48 min a game) expect 3-4 per game. The previous level leaves the context early. |
| Tool calls with an error | 41 of 531 (7.7%); 10 NameErrors (a variable from an earlier call); 2 tool timeouts | Each timeout loses the call's output **and all retained functions** (code read, section 3, M1). |
| Tool calls that write a search (BFS/beam/A*/brute force) | 28 of 531; 9 of r11l's 44 | The model writes its own planners, often under a 30-s limit. |
| Game overs | 5, of which **3 were budget deaths** (sc25 L4 twice, tu93 L4 once) | The agent misjudges its own action budget. |

The loss episodes behind these numbers (quotes are the model's reasoning):

- **r11l L3: a planner that never ran (10 minutes, 1 action, level lost).** The model decoded the mechanic
  correctly ("blob = floor(mean(anchors))", "anchors CAN be placed on walls") and then spent six minutes and four
  rewrites on search code: "The search timed out (and all retained functions were cleared!). The A* with 4 anchors ×
  400 candidates is too slow." → "My beam_plan draft is messy (the reconstruction logic is broken)" → "No solution at
  depth 8 with step 3" → "'stuck' at some depth". The run ended there. Level 3 is worth 3/21 of the game (+14 points
  on r11l).
- **tu93 L4: hypotheses tested one LLM turn per probe, then a budget death.** Three competing models of the
  orange chaser ("chase+stay", "chase+fallback", "patrol") were probed with one or two actions per turn, about 1 to 1.5
  minutes each: "I can do 1 action per call and re-evaluate". It also had a BFS bug ("I used `c=st` instead of the
  current state"). The last death: "The bar decreased by 3 per move … the death was a BUDGET death (the step limit
  ran out), not a collision!"
- **sc25 L4: the budget read by eye, then actions burned.** "Bar: 58 white rows → only 6 green rows left ≈ 6
  actions (if 1 row/action) or 3 (if 2 rows/action)" → "Death cause: budget depleted … I was 4 LEFTs short". Later:
  "Let me burn the budget now with LEFT presses … then execute the optimal plan after the reset". RESET is not
  exposed in his submission (`EXPOSE_RESET=off`), so a hopeless attempt costs 5-10 actions instead of 1. It also
  fails to find the cause of an effect: "In level 3, the pink column submission (2nd time) cleared the ring + r
  blocks. In level 4 it didn't … So what triggered the clearing in the earlier attempt?"
- **re86 L4: a whole plan executed on a wrong model, with the budget misread.** "Both objects are now at their
  target positions … All 6 dots should be covered — but the level did NOT complete" (23 actions). Then "the pattern
  suggests it should be 38 at step 198, yet I'm reading 44 from the bar"; "37 remain → 27 actions consumed since the
  level start (step 135 → 219 = 84 steps?!)". The bar ticks once every ~3 actions.
- **ft09 L5: brute force timed out, and costly whole-hypothesis tests.** "Timed out (2^26 brute force is too slow)"
  → union-find. Then 9 clicks and 27 clicks testing the two polarities: "Neither polarity completed!" It also had a
  12,288-token reasoning turn that ended on the length limit with no tool call.
- **vc33 L4: lost knowledge of the previous level after the trim at step 24.** "Let me look at the level-3
  completion frame (transition 29)" → "my memory of the level-3 geometry is fuzzy" → "WAIT. This is the frame after
  transition 18, which was the first action of LEVEL 3". That was three calls spent reconstructing a fact the
  harness had exactly.

These match our earlier studies on the old base: discovery dominates (levels2plus: 15/48 stuck levels "mechanic
never decoded", 7 rule misreads; hard L1: goal misread 11/31, key effect misunderstood 9/31), solved levels are
efficient, and knowledge is lost at context boundaries (stall analysis: 73% of stuck time ran without the previous
level's raw context). Franzen's 128 Ki window delays that loss; it does not remove it.

## 2. Ranking (expected LB gain per GPU-hour of testing)

Gains are judgement calls from the evidence above, on top of Franzen's 27.89. Identical notebooks spread
1.5-2.3x on the LB, so no LB draw can confirm a +1. "Test GPU-h" is the dedicated GPU time to a keep/kill decision,
using the level-snapshot bed in section 4 where possible.

| Rank | Method | Main loss mode | Expected LB | Test GPU-h | Gain per GPU-h |
|---:|---|---|---:|---:|---:|
| 1 | M1 Failure-safe sandbox (a timeout keeps its functions and partial output; per-level `mem`) | lost work, NameErrors | +0.1 to +0.4 | ~0 (CPU tests, rides along) | highest |
| 2 | M2 Exact budget meter (+ the existing `EXPOSE_RESET=on`) | budget deaths, burned actions, misread bars | +0.3 to +1.0 | ~1.5 | ~0.4 |
| 3 | M3 Harness search service + guarded plan execution | planners rewritten, buggy or timed out; open-loop batches on a wrong model | +0.5 to +1.5 | ~3 | ~0.33 |
| 4 | M4 Exact ledger re-pinned at every context trim (level win records, attempts) | knowledge lost at trims; goal grew / old recipe reused | +0.3 to +1.5 | ~3 | ~0.27 |
| 5 | M5 Hypothesis check against history + discriminating probe | one probe per LLM turn; competing mechanic models | 0 to +1.5 | ~3 | ~0.2 (high variance) |
| 6 | M6 Goal monitor (model-registered goal predicates, harness-evaluated) | wrong goal kept; falsified goals retried | 0 to +1.5 | ~3 | ~0.17 |

If they stack, expect +2 to +5 LB (about 30-33). That is not 52. Section 7 says what that implies.

## 3. The methods

### M1. Failure-safe sandbox: timeouts keep their work, and a small per-level data store

**Loss mode and evidence.** On a tool timeout `run_sandboxed_python` returns `{"error": "Tool timed out after
30s", "stdout": ""}` and no `keepable_functions`. `ToolAgent._record_retained_functions` then sets
`self._kept_functions = {}`, which drops **every** retained function of the game, not only the new ones, and all
printed output is lost. r11l L3: "The search timed out (and all retained functions were cleared!)". Separately,
ordinary variables die between calls (by design), giving 10 NameErrors in 531 calls. The model also rebuilds large
structures each call: "I need to rebuild FREE/CORR each call (they're local)" (r11l); "Nfill/Ofill were locals.
Redefine inline." (re86).

**Mechanism.**
(a) On timeout, keep the previous retained functions and add the new definitions parsed from the code text
(`collect_functions` already works from source).
(b) Return partial stdout: the child arms `signal.setitimer(ITIMER_REAL, timeout-1)` and a soft `RLIMIT_CPU` with a
SIGXCPU handler, which raise inside the snippet, so `main()` still sends `final` with the stdout so far and the
keepable functions.
(c) An explicit `mem` dict that persists only within the current level, holds JSON-serialisable values (capped at
~200 KB) and is cleared at level change. The tool result lists its keys and the step at which each was written,
so staleness is visible. This keeps Franzen's reason for discarding variables (stale board data) while removing
the rebuild cost.

**Cost.** About 40 tokens of static system prompt (prefix-cached); no actions.
**Expected effect.** Reach: fewer lost turns (each timeout costs a turn plus re-deriving its helpers); a few
percent fewer code tokens. Efficiency: none.
**Risk.** Franzen's design note: variables were dropped on purpose; `mem` is opt-in and level-scoped. There is no
negative result against keeping functions on a timeout; dropping them is a bug.
**Attachment points.** `inference/agent/python_tool_sandbox.py`: `_set_limits` (soft CPU limit), `main()` (the
timeout path sends `final`), `run_sandboxed_python` (the timeout branch returns partial stdout).
`inference/agent/tool_agent.py`: `_record_retained_functions` (keep `previous` when `keepable_functions` is
missing) and `_run_python_tool` (`mem` round-trip through `_serialized_runtime_state`; cleared on level change).
**Cheapest test.** CPU pytest only: a snippet that defines `f` and loops forever keeps `f` and earlier functions
and returns its first prints; `mem` survives two calls and is empty after a level change. Ship it with the next
GPU run; it needs no arm of its own.

### M2. Exact budget meter (and let a hopeless attempt RESET)

**Loss mode and evidence.** 3 of the run's 5 game overs were budget deaths (sc25 L4 twice, tu93 L4). The model reads
bars by eye and gets the rate wrong when a bar ticks 2-3 cells per action or one cell per 2-3 actions. Quotes are in
section 1 (sc25 "if 1 row/action or 3 (if 2 rows/action)"; re86 "84 steps?!"; ft09 "maybe the bar decrements per
action but slowly … Whatever"). It burns 5-10 actions to end an attempt it knows is hopeless (sc25). The census
found a budget bar in 19 of 19 dev games (lesson 0016). levels2plus proposed a budget ledger (R2); Franzen's harness has
nothing like it.

**Mechanism.** A frame-only detector. For each attempt (from level start or RESET), track per-colour cell counts on
every row and column within 4 cells of the edge. The bar is the line whose count only falls, apart from rare
refills, with at least 3 drops. Rate = cells lost per action since the last refill (it can be fractional); remaining
= count ÷ rate. Each user prompt gets one line, after `Current state: step N, level L.`, for example: "Budget bar
(row 63): ~3 actions left in this attempt (it loses ~3.2 cells per action; refilled +16 at step 67)". The same
estimate is exposed as `budget` in the sandbox. Nothing is printed until the detector has seen 3 consistent drops;
the rate is carried from the previous attempt or level as a prior. Pair it with the existing `EXPOSE_RESET=on`: then
a RESET costs 1 action instead of burning the bar. Franzen built RESET but left it off, and the meter gives the model
the number it needs to choose it.

**CPU check already done** (`proto/budget_bar2.py` on the recorded per-action boards of all 10 games).
- It finds a bar in 9 of 10 games (all but sb26): sc25 col 62 at ~1.9 cells/action; tu93 row 63 at 1.25 → 3.2 by
  level; re86 row 63 at 0.32 (one cell per ~3 actions); tr87 at 0.5; ft09 at 0.48-0.69; vc33, ar25, lp85 and r11l
  at 1.0.
- Before each of the 3 budget deaths it read 3.1 → 2.1 → 1.0 → "0 left" (sc25 actions 90-93, death at 94; the same
  pattern before 129) and 3.1 → 1.9 → 0.9 → "0 left" (tu93, death at 106).
- At the moment the model wrote "37 remain … 84 steps?!", it would have said "~110 actions left at 1 cell per 3
  actions".

**Cost.** ~30-40 prompt tokens per turn (~3k per game, cached after first prefill); no actions.
**Expected effect.** Efficiency: fewer repeated attempts on budget-binding levels (each costs up to a full bar: 32
actions in sc25 L4, 21 in tu93 L4) and RESET instead of burning. Reach: 1-3 fewer diagnosis and replanning calls per
death, and plans sized to the budget before they start. Hidden games, which are harder and probe more, should hit
budgets more often than this public run did.
**Risk.** Franzen found "death-ledger guidance" not clearly helpful. That was a log of past deaths; this is a
forward-looking number. A false reading is worse than none (lesson 0024). Bars that are not edge lines (sb26 here)
or refill often must stay silent; a two-colour bar needs care (count the shrinking colour only).
**Attachment points.** New `inference/utils/budget_bar.py`, called from `ToolAgent._build_user_prompt` with
`history_entries` since the last RESET or level start. Expose it in `_run_python_tool._serialized_runtime_state`
and the sandbox `_refresh_state` (`runtime_globals['budget']`). `ToolAgent._game_over_diff_lines`: when the bar
reads 0 at the fatal action, state "budget death" exactly, which shortens his diagnosis prompt. Notebook:
`EXPOSE_RESET=on`.
**Cheapest test.**
1. CPU: turn the prototype into unit tests on these recorded boards (bar found, deaths predicted, silent on sb26)
   and run it on exp-071's boards when that run exists.
2. GPU: one snapshot pair on budget-binding dev levels (~1.5 GPU-h). Read budget deaths per attempt, actions per
   solved level, minutes to solve.

### M3. Harness search service and guarded plan execution

**Loss mode and evidence.** His system prompt already tells the model to "write an explicit search algorithm such as
BFS". It does (28 search snippets), and pays three ways:
- It times out at the 30-s tool limit: r11l, ft09.
- It writes reconstruction bugs: tu93 "I used `c=st`", r11l "the reconstruction logic is broken".
- It re-plans a level-sized problem from scratch in tokens: four rewrites in r11l L3.

Plans are then run open-loop: re86 L4 executed 23 actions on a wrong model. The hard-games study saw the same in
sk48, ls20 and tn36 (open-loop batches of 15-41 actions on wrong rules; R6 there). Planning in a correct model is
the cheap route to below-human action counts (road-to-100: the optimum is a median 0.41 of the human count).

**Mechanism.** Two library functions in the sandbox, documented in about 10 static system-prompt lines.

- `search(start, step, is_goal, actions, key=None, heuristic=None, beam=None, max_nodes=300_000,
  time_limit=60)` → `{status: found|exhausted|timeout, plan, expanded, best_partial}`. It runs BFS, or A*/beam
  when given a heuristic or beam width. It dedupes on a hashable `key(state)` and has correct parent tracking.
  `step` returning `None` means illegal, and `'dead'` means fatal. The model's functions run in the same process,
  so nothing is serialised; they become retained functions.
- `run_plan(plan, observe, predicted)` executes one action at a time and compares the model's own
  `observe(current_frame)` with the predicted state. It stops at the first mismatch and returns `{executed,
  mismatch_at, predicted, observed}`. This is Tycho's guarded execution, simplified, and a stop-on-surprise that
  catches wrong changes, not only the no-change his batch guard catches.

The tool timeout is raised to `time_limit + 15` s only for snippets that call `search(`. Kaggle's 48 vCPUs are idle
while the GPU decodes; the cost is one of 10 admission slots held during the search, roughly a 5-10% throughput dip
for that minute.

**Cost.** ~250 static system-prompt tokens (cached). Saves ~1-3k generated tokens per avoided planner rewrite and
the turns lost to timeouts. Wall time up to ~75 s per search call.
**Expected effect.** Reach: levels where the model has the mechanic but no working plan (r11l L3 was exactly that)
convert; fewer tokens per planned level. Efficiency: searched plans are short, and guarded execution stops wrong
batches early.
**Risk.** Tufa reported that hand-built tools hurt; Franzen saw no gain from a prescriptive step-verification hint.
Keep it a generic, optional library with descriptive documentation and no nudges. Never call it a solver of the
game. It is useless when the abstraction is wrong (that is what M5 is for). Long searches hold a slot: cap
`time_limit` at 60 s and nodes at 300k.
**Attachment points.** `inference/agent/python_tool_sandbox.py`: `_SANDBOX_BOOTSTRAP` / `main()` adds `search` and
`run_plan` to `runtime_globals` beside `frame_diff`; `run_plan` is built on the existing `action()` closure.
`inference/agent/tool_agent.py`: `_run_python_tool` gets the per-snippet timeout; `_build_system_prompt` gets a
`SEARCH_HELPER_ADDENDUM` behind a new `ARC3_SEARCH_HELPER` flag.
**Cheapest test.**
1. CPU: unit tests (shortest paths on toy grids, dedupe, timeout returns best partial, `run_plan` stops at a planted
   mismatch). Re-run r11l L3's own model code (from the transcript: floor-mean chain, free-cell set) through
   `search` and confirm a plan within 60 s.
2. GPU: one snapshot pair on planning-heavy dev levels (~3 GPU-h with 2 seeds). Read minutes and tokens to solve,
   tool timeouts, actions per solved level. Kill if `search` is called on fewer than 10% of the levels where the
   control arm writes its own BFS.

### M4. An exact ledger, re-pinned at every context trim

**Loss mode and evidence.** Every game trimmed within 25 minutes. After vc33's trim the model spent three calls
finding the level-3 completion frame and got it wrong first. On the old base: levels2plus mode C (previous-level
knowledge lost, 4 primary / 7 any of 48) and mode A (goal grew, old recipe reused, 5/13). The stall analysis found
that all 10 re-derivation episodes began after the previous level left the context. Lesson 0016: the goal kind is
constant within a game, so how a level was won is the strongest prior for the next.

**Mechanism.** Harness-written exact facts only; the model is never asked to write anything.
- At each level-up, a short "how level k was won" record goes into the level-up prompt (~150-300 tokens):
  - actions used, with the action sequence run-length encoded;
  - the object-level net change from the level's first frame to the board before the winning action, from
    `frame_diff`, with edge-bar and background "resized" entries filtered;
  - the winning action's own effect on the terminal layer (lesson 0017: the win frame of an animated win is not the
    first layer).
- `level_wins` in the sandbox holds start frame, pre-win frame, win frame and actions per level.
- At each trim, `_trim_messages_for_context` re-inserts one ledger message as the first message after the system
  prompt, holding:
  - all win records;
  - the current level's attempts and their deaths (fatal action, budget versus non-budget, from M2);
  - the names of retained functions;
  - M6's falsified goals, if M6 is on.

A trim already invalidates the prefix after the system prompt, so the insertion costs no extra prefill.

**CPU check already done** (`proto/winrecord.py`). It produces the record from recorded boards. On vc33 L2 it states
"N 12 px moved [52,8]→[52,24] together with c 16 px", which is the fact the model went looking for. It needs the
bar/background filter before use.
**Cost.** ~150-300 tokens per level-up and ~300-600 per trim (3-4 trims per hidden game); no actions.
**Expected effect.** Reach: fewer re-derivation calls on levels 2+ after trims. The goal and mechanics carry over
correctly (the right "kind", with the count or scope re-checked). Efficiency: fewer re-probing actions on later
levels.
**Risk.** Franzen's model-written summaries gave no gain and the structured world-model notes hurt. The difference
here is harness-computed facts, inserted only where the prefix is already broken. It must never state a wrong
fact; last-action lists invited a false rule in r11l B, so print net object changes, not commentary.
**Attachment points.** `inference/agent/tool_agent.py`: `_trim_messages_for_context` / `_drop_oldest_history_block`
(insert after a trim, tagged with `_mark_control_message(kind="ledger")` so his pruning code recognises it);
`_build_user_prompt` (level-up branch); reuse `_level_transition_diff_lines` and `compute_frame_diff`; `level_wins`
through `_serialized_runtime_state` and the sandbox `_refresh_state`.
**Cheapest test.**
1. CPU: build the ledger at each of the run's 12 trim points and check every line against the transcripts (no false
   facts).
2. GPU: a snapshot pair that starts each game at level k with no context, which is a fair proxy for the post-trim
   state: ledger versus none (~3 GPU-h). Confirm in the next full public-25 pair, which the base calibration needs
   anyway.

### M5. Check hypotheses against history, and let the harness pick the discriminating probe

**Loss mode and evidence.** Competing mechanic models are tested one LLM turn per probe (tu93 L4: three chaser
models, four probe turns of 1-2 actions each, two deaths). Effects that differ between states are re-derived by hand: vc33 "W3-left is a NO-OP
now! … But earlier, W3-left worked when A2 was 54 → 51"; sc25 "what triggered the clearing in the earlier
attempt?". Models are rarely checked against the recorded transitions: 0-10 history-reading calls per game. At the
frontier, Tycho's executable world model checked against transitions added about +6 RHAE (79 → 85, lesson 0013).

**Mechanism.** Library functions that reuse the `step` and `observe` the model already writes for its BFS.
- `check(step, observe, level=None)` replays every recorded transition of the level and returns the number
  matched out of the total, plus the first mismatch (index, action, predicted, observed).
- `probe(hyps={name: step}, observe, candidates=None)` predicts, from the current state, each candidate action's
  outcome under each surviving hypothesis. Candidates default to the valid keys plus one click per segmented object.
  It returns actions ranked by how many hypotheses they separate, and drops actions any hypothesis predicts fatal.
- An automatic variant, with no model code: when the same action (a key, or a click on the same object) had two
  outcome classes on this level, report the object differences between the two before-boards near the acted
  object. In this run that happens 19 times across 374 action signatures, mostly blocked moves, so it is worth less
  than the model-driven version.

**Cost.** ~150 static system-prompt tokens; ~200 tokens of tool output per use; CPU only.
**Expected effect.** Reach: fewer turns to settle a mechanic (tu93 L4: one discriminating probe instead of four
turns). Efficiency: fewer probing actions, and no probes predicted fatal.
**Risk.** This is the arm closest to Franzen's negatives: structured world-model notes hurt, and a stronger
verification hint did not help. Lesson 0010: a verifier without a stop rule turns plans into one action per call.
Keep it a tool the model may call, never a required step, and never a nudge. Uptake by a 6B-active model is the
open question.
**Attachment points.** `python_tool_sandbox.py`: `check` and `probe` in `runtime_globals`, using `transitions`,
`valid_actions` and frame `.segmentation`. The automatic variant goes in `ToolAgent._summarize_step_sequence` →
`_build_user_prompt`.
**Cheapest test.** CPU unit tests on planted step functions (tu93-style chaser variants over recorded boards). Then a
GPU snapshot pair on discovery-heavy dev levels (~3 GPU-h). Read uptake after 30 minutes and kill if it is used on
fewer than 10% of levels.

### M6. Goal monitor: goal predicates the model registers and the harness evaluates on every frame

**Loss mode and evidence.**
- Goal misreads were the largest hard-L1 family (11 of 31 primary).
- re86 L4 executed a whole plan before finding that its goal was wrong ("All 6 dots should be covered — but the
  level did NOT complete").
- ft09 L5 spent 36 clicks on two goal polarities.
- sc25 established "Ring target proved NOT to be the goal (cleared without completing)" by accident.
- After trims, falsified goals can be retried because nothing records them.

**Mechanism.** Retained functions named `goal_*(frame) -> bool` are evaluated by the sandbox's `action()` wrapper
after each executed action. Each result carries `goal_flags`. Two events get one line each in the next prompt and an
entry in the M4 ledger:
- "goal_X became true at step s without completing the level: falsified, or the win is checked only after another
  action kind (true in 5 of 19 dev games: try the other action kinds once)".
- "goal_X held / did not hold on level k's win frame".

**Cost.** ~100 static system-prompt tokens; a few tokens per action result; ~50 tokens per event.
**Expected effect.** Reach: faster goal falsification and kept-goal transfer across levels (lesson 0016).
Efficiency: a plan stops as soon as its goal predicate is true without completion.
**Risk.** The same uptake risk as M5. Win checks gated by action kind make "true without completion" ambiguous, so
the line must say so (wording above).
**Attachment points.** `python_tool_sandbox.py` `action()` (evaluate after `_refresh_state`); `ToolAgent._compact_action_result`
and `_build_user_prompt`; the M4 ledger.
**Cheapest test.** CPU tests (flags fire on planted predicates over recorded boards). Then a GPU snapshot pair on
goal-misread dev levels (~3 GPU-h), with the same uptake kill rule.

## 4. The test bed that makes these cheap: level-snapshot A/B

A full public-25 run at hidden-set compute is ~2.5-3 GPU-h. Its mean moves ~2 points between identical runs, so a +1
needs several pairs. Most methods above act inside one level. **Start games at a chosen level instead.** Replay a
recorded action prefix before the first analyzer turn, give each snapshot a fixed time cap (e.g. 15 minutes), and
score that level only: solved, minutes, generated tokens, actions against baseline. About 24 snapshot levels × 2
arms × 15 minutes ÷ 10 slots is ~1.2 h plus startup, so ~1.5 GPU-h per seed and ~3 GPU-h for two seeds. Every
comparison is paired per level.

- Prefix sources: this run's per-action boards and actions (the 10 stuck levels); exp-054's recorded actions for the
  other public games; exp-071 once it runs.
- Use dev games only. r11l and sc25 are in our validation split: reading their transcripts for loss modes is fine,
  but do not tune on their snapshots.
- Attachment: `framework/solver.py::_HarnessGameSession._play_inner`, which replays the prefix with the
  `_execute_action` path that `_execute_auto_reset` uses, before the analyzer loop, and records the snapshot's start
  step. Add a `--snapshot-file` option in `framework/run.py`.
- Limits: it cannot measure scheduler effects or cross-level memory, apart from M4's "no context at level k" proxy.
  Confirm any keeper in a full public-25 pair before it reaches the LB.

## 5. Examined and not recommended (with the evidence)

| Idea from the brief | Why not now |
|---|---|
| Detect "plan known, execution slow" and switch to batch execution | Turns that execute one planned action with low reasoning cost 1.8 of 238 game-minutes; the model already batches (median 2 actions per acting turn, up to 34). |
| Reasoning-length caps per call type | 38% of tokens are in requests over 4k, but those are mostly level-start analysis that preceded solves (sb26 L8). Cutting effort on the old base gave +35% calls and no levels (exp-037), and past reasoning is load-bearing (lesson 0022). At most, turn on his existing `ARC3_REASONING_EFFORT_LADDER` for length finishes (1 of 531 requests here). |
| Branching with UNDO to test hypotheses | UNDO was valid in 2 of 10 games and used 0 times; an undo is itself a counted action, so a probe-and-undo costs 2. Franzen saw use only in su15. Revisit if exp-071 shows UNDO-heavy games. |
| More budget-aware stopping or allocation | His P = (A+B)·C scheduler already decays stalled levels; his more elaborate priority rules gave no clear gain; our allocation levers measured near zero on a saturated server. |
| Better animation use (digests, composite images) | Franzen: composites hurt and the automatic timeline did not help. Here the model reads `last_animation_timeline` competently (sb26: 16 calls, sc25: 18) and solved sb26 8/8. |
| Shorter or deduplicated prompts | Franzen: shorter prompts and fewer reinsertions were not consistently better. |

## 6. Longer shots (honest feasibility)

1. **Fine-tuning Flash-Next on its own solved-level traces (rejection-sampling SFT / STaR).** Data would come from
   our runs on the public games, using solved levels at or below baseline, deduplicated by (game, level), with tool
   tokens weighted over reasoning.
   - Feasibility is low before Nov 2. LoRA on the MoE needs BF16 weights on rented multi-GPU (8×H100 at about $25/h),
     then a merge and a new W4A16 AutoRound quantisation, with calibration risk. Our SGLang path does not serve a
     LoRA on the quantised experts.
   - Evidence of transfer is weak: STaR on the 27B moved the LB 1.25 → 1.94 (n=1); "naively scaling the data hurt";
     one GRPO attempt collapsed.
   - Cost $1.5-3k all-in, roughly a 40% chance of no gain. It needs your approval and the external-data clause
     checked.
2. **Hand a stuck level 1 to the CPU novelty explorer, then back to the model.** Our explorer (`arc3/agents/explorer.py`,
   frame-only, budget bar masked) cleared level 1 on 9 of 19 dev games in at most 90 s of CPU. When the scheduler's C
   factor says a game is stuck on level 1, the explorer plays it out; level 1 then scores about 0 (it was going to
   score 0) and the model starts level 2 with the explorer's winning trajectory.
   - On a level-1-heavy hidden set this is the only lever that rescues games now at zero.
   - Against it: the model meets level 2 without understanding level 1, the explorer fails where the model fails
     most (deep or wide level 1s), and thousands of actions may meet a per-game action cap (VERIFY in the
     competition rules).
   - Expected +0.3-1 LB. CPU porting takes about 2 days; one GPU run.
3. **Fresh-eyes parallel goal hypotheses and same-run skill notes.** When a level has absorbed a set token amount
   without progress, sample 3 short goal-only completions on the cached prefix (~300 tokens each) and show their
   union, M6-checkable. Optionally, carry one-line notes from games won earlier in the same run (for example
   "programs show their execution in the animation") into new games' level-start prompts, never the system prompt,
   so the cache is unaffected. The prior is negative: the 3rd-place team's best run had its multi-candidate arbiter
   off, our P19 supervisor did not pay, and hidden games are novel. Run it only after M5/M6, as a snapshot arm.

A second small model (a perception or summariser VLM) is not feasible: the KV cache at 10 streams already uses the
card (static fraction 0.96), and Franzen found summaries did not help.

## 7. What this does and does not buy

These methods spend fewer tokens per correct decision; lesson 0025 (the LB moves with tokens) applies to them. My
estimate for all six together is +2 to +5 LB on 27.89. Tufa's 52.51 is about +25 and unexplained; nothing public
(model, serving, or harness) accounts for it. The plan that follows:
1. Ship M1 at once and M2 after its unit tests.
2. Build the snapshot bed and run M3 and M4 as paired snapshot arms.
3. Run M5/M6 only after reading exp-071's transcripts at hidden-set compute (48 minutes a game will show more trims,
   more budget pressure and longer stuck levels than this 25-minute demo).
4. Keep watching the LB and the forum for what moved Tufa and Yi-Chia Chen (lesson 0026).

## Scratch artifacts (`/tmp/claude-0/-home-user-Arc-Agi-3-Kaggle-comp/d342458e-03bd-545b-8a6d-06bca061963e/scratchpad/new-methods/`)

- `digest.py`, `digests/*.dig.txt`: per-turn digests of the 10 transcripts (reasoning, code, tool results).
- `timeline.py`, `timeline.txt`: per-turn time, level, reasoning characters, actions.
- `toolstats.py`: tool-call error, timeout, search and history counts.
- `proto/budget_bar.py`, `proto/budget_bar2.py`, `proto/bar2_*.txt`: the budget meter prototype and its readings on
  the recorded boards.
- `proto/winrecord.py`: the level win-record prototype (uses Franzen's `segment_layer` and `compute_frame_diff`).
