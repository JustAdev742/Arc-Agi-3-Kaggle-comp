Summary: patch `ours-02-budget-meter.patch` (method M2) reads each attempt's action budget from the edge bar in the
frames and adds one line to the user prompt, a "budget death" verdict after a game over, and `budget` in the python
sandbox, behind `OURS_BUDGET_METER=1` (default off; with it off the harness is byte-identical). Over 79,626 recorded
steps of 25 public games it never read a line that is not the bar, found the bar in 9 of Franzen's 10 demo games
(silent on sb26), read "0 left" right before all 3 demo budget deaths, and named 244 of 258 budget deaths with no
false verdict among 557 game overs. Near the end of an attempt 2% of readings were off by more than 2 actions, nearly
all because an action ahead cost more than usual (dc22 falls, su15 penalties). `EXPOSE_RESET=on` is safe with it.
Nothing has run on a GPU.

# M2 on Franzen's base: an exact budget meter (2026-10-08)

Design: [new-methods.md](new-methods.md) section M2. Pipeline and CPU bed: [engineering.md](engineering.md).
Evidence for the loss mode: [franzen-run-analysis.md](franzen-run-analysis.md) sections 3-4.

## 1. What it does

Files: `kaggle/franzen/patches/ours-02-budget-meter.patch` (applies after his patch and
`ours-sandbox-timeout-keeps-work.patch`); it adds `inference/utils/budget_bar.py` (the detector, stdlib only) and
touches `inference/agent/tool_agent.py` and `inference/agent/python_tool_sandbox.py`.

With `OURS_BUDGET_METER=1` (read at call time; "1/true/yes/on"):

1. **User prompt.** Once the detector has a reading for the current attempt, one line follows
   `Current state: step N, level L.`, e.g. (Franzen's demo, sc25 and tu93):

   ```
   Budget bar (columns 62-63, colour 'N'): about 43 more actions before it is empty (58 cells left, about 1.3 per action). `budget` in python has these numbers.
   Budget bar (row 63, colour 'M'): less than one action left before it is empty (3 cells left, about 3.2 per action). `budget` in python has these numbers.
   ```
   With free action kinds it ends "...; MOUSE did not lower it." An empty bar reads "Budget bar (...): empty."
   The colour is the letter the model sees in `.ascii`. Nothing tells the model what to do.
2. **Game-over prompt.** After a GAME OVER, right after the fatal-action lines and before the (optional) game-over
   diff: `Budget meter: the bar (row 63, colour 'M') ran out at the fatal action, so this was a budget death.` It is
   printed only when the reading on the last frame before the fatal action had at most one action left and the death
   frame shows the bar no fuller and below one action's worth. Otherwise nothing is said (his BAR RULE text stays).
   His `_game_over_diff_lines` is gated by `ARC3_GAMEOVER_DIFF`, which his notebook leaves off, so the verdict sits at
   that call site but outside the gate.
3. **Sandbox.** `budget` holds the same reading as a dict (`line`, `colour`, `cells_left`, `cells_at_attempt_start`,
   `cells_per_action`, `actions_left`, `drops_seen`, `attempt_start_step`, `free_actions`, `largest_drop`,
   `drops_every_action`, `refills`) or `None`, refreshed after every `action()` call.

### How the detector reads a bar

An attempt starts at the first frame, at every RESET (automatic after a death, or deliberate) and at every level
change. For each row and column within 4 cells of the border and each colour with at least 8 cells at the attempt
start, it follows the count over the attempt. A line counts as a bar only if all of these hold:

- it lost cells at least 3 times (MIN_DROPS);
- its usual drop is k cells, with at most a quarter of drops bigger than k+1, and k is small for the line
  (at most max(2, start/8));
- any rise is a refill of at least max(4, 2k+1) cells, at most one per 10 actions;
- it drains steadily: counted in charged actions, no gap between drops, wait before the first drop, or pause since
  the last drop exceeds twice the mean gap plus one;
- no more than 2 identical lines (a water column or segment 3-4 cells thick is an object, not a meter).

**Time is counted in charged actions.** A kind that did something at least 3 times without lowering a bar that
otherwise drops often is free (ar25's clicks, su15's UNDO). When the bar drops on at least 80% of the actions that
did something, an action that changed nothing inside the board and did not lower it (a blocked move, a click on
nothing) is not counted either (ls20, lp85).

**Rate and actions left.** Rate = cells lost after the first drop / charged actions between the first and last drop,
over the whole attempt (a recent window measured worse). Left = floor(cells / rate), minus the actions since the last
drop when the bar ticks every few actions instead of on every action.

**Which line, when several drain.** The (line, colour) read as the bar in an earlier attempt of the game is
remembered with the colour its cells turned into. If a remembered bar is drawn at the attempt start, only it may be
read, and only while it drains into the same colour as before; a different colour can mean a second layer of budget
below (ar25 from level 3 drains its top layer into the next layer's colour, 64 steps per layer). Otherwise a lone
candidate is read, and several only when their estimates agree within 2. When unsure: silence.

## 2. Evidence that it is worth having

From the analysis of Franzen's 10-game demo (2026-09-30): 3 of its 5 game overs were budget deaths (sc25 L4 twice, tu93
L4); bar reasoning appeared in 22% of calls (63 calls ran bar-reading code, 42 of them sc25); the model misread rates
("6 actions (if 1 row/action) or 3 (if 2 rows/action)" in sc25; "84 steps?!" in re86, whose bar ticks once every ~3
actions), misdiagnosed a budget death as a collision (tu93), and burned ~9 actions to force a reset (sc25), RESET not
being exposed. Lesson 0016: every dev game has a per-attempt budget, so hidden games will hit it too.

## 3. Accuracy on recorded games

Measured with `scripts/budget_meter_eval.py` (committed), which loads the detector from the patch and runs it at every
step of every recording on the history up to that step, as the harness does. Frames come from the per-action boards of
`*_events.jsonl`, or from replaying a run's action histories in the offline engine (every replay of recorded boards
matched them exactly, 1,337 of 1,337 frames for the demo). Ground truth:

- the true bar line of each of the 25 games, read off their frames (sb26 and sk48 draw no edge bar: sk48's 196-step
  counter is not drawn on any edge line; sb26's energy neither);
- the realized future of the true bar in the same recording: actions until it empties, or safe actions before a
  budget death (exact), or a lower bound when the attempt ended otherwise (a win, a hazard death, a RESET);
- the game's own `current_steps` counter, from the engine replay, in the 16 games that keep one.

"Near the end" = the reading or the truth is at most 10 actions. Validation games (cn04, g50t, lf52, r11l, sc25, sp80)
are reported apart; the prototype had seen sc25 and r11l, so they are not a clean holdout.

### Franzen's demo (2026-09-30, 10 games, 25 min each, 1,337 frames)

- **Bar found in 9 of 10 games**, one line each; **silent on sb26** (0 readings in 155 steps).
- **sc25 L4**: 4, 3, 2, 1, 0 actions left at actions 89-93, death at 94; again 4, 3, 2, 1, 0 at 124-128, death at 129.
  Both named "budget death".
- **tu93 L4**: 4, 3, 1, 0 at actions 102-105, death at 106, named. Its collision deaths at 52 and 85: not named.
- 847 readings; 0 on a wrong line; 36 near the end with a realized truth, 0 off by more than 2 (errors -1 to +1);
  25 against the game counters (tu93 21, lp85 4), 0 off. ar25 is read on levels 1-2 only (levels 3+ have the
  layered bar: silent).

### All recordings

| Data | Split | Steps | Readings | Coverage¹ | Wrong line | Near end: off by >2 / readings | vs counter: off by >2 (too high) / readings |
|---|---|---:|---:|---:|---:|---:|---:|
| Demo | dev | 1,112 | 706 | 85.0% | 0 | 0 / 11 | 0 (0) / 25 |
| Demo | val | 170 | 141 | 96.6% | 0 | 0 / 25 | - |
| Franzen-harness runs² | dev | 12,327 | 9,927 | 95.4% | 0 | 8 / 359 | 5 (0) / 236 |
| Franzen-harness runs² | val | 3,256 | 2,594 | 92.6% | 0 | 0 / 127 | 0 (0) / 7 |
| Older runs³ | dev | 52,318 | 40,618 | 94.7% | 0 | 66 / 2,677 | 77 (11) / 1,912 |
| Older runs³ | val | 10,443 | 8,576 | 92.2% | 0 | 3 / 790 | 0 (0) / 511 |
| **Total** | | **79,626** | **62,562** | | **0** | **77 / 3,989 (1.9%)** | **82 (11) / 2,691 (3.0%)** |

¹ Share of steps with a reading among steps where the true bar had already dropped 3 times in the attempt.
² exp-072a6/b/f/g and exp-073/073b (Franzen's harness, D'), action histories replayed: 118 recordings, 16,094 frames.
³ exp-032 to exp-057 and four public notebooks (25 runs of other harnesses): 625 recordings, 64,804 frames.

- **A bar where there is none: 0 readings in 79,626 steps**, including 0 in sb26 (2,955 steps) and sk48 (2,942).
- **Off by more than 2 near the end: 77 of 3,989 readings (1.9%).** 71 of them had an action ahead (or behind) that
  cost far more than the usual drop: dc22 falls (20 steps each), su15 penalties. The other 6: 3 in lp85 (a slow bar
  quantization, +3) and 3 in sc25 (+3/+4). Below a survived lower bound by more than 2: 0 of 1,176.
- **Against the game's counters: 82 of 2,691 off by more than 2 (3.0%), all in dc22 and su15**, 11 of them too high.
  Elsewhere the reading is the counter or one below it (the floor, and games that end the attempt on the action that
  empties the bar).
- **Coverage near the end** (truth within 10): 100% (demo dev/val, Franzen dev), 90.1% (older dev), 49.8-65.1% on
  validation games (lf52 and sc25 go silent near the end more often).

### Budget-death verdicts

Every GAME_OVER followed by the automatic RESET (truth: the true bar empty in the death frame, and the game's counter
at most 1 where it has one):

| | Game overs | Budget deaths | Named | Named wrongly | Missed |
|---|---:|---:|---:|---:|---:|
| dev | 453 | 192 | 187 | 0 | 5 |
| val | 104 | 66 | 57 | 0 | 9 |
| **all** | **557** | **258** | **244** | **0** | **14** |

The 299 other game overs (tu93's enemies alone: 211) were never called budget deaths. The 14 misses are silences:
ar25 2 (layered levels), dc22 2 (falls), lp85 1, sc25 9.

### What changed on the way (all measured on dev games)

The prototype's rule set (3 drops, rate since the last refill) read the bar in 9 of 10 demo games but also read
vc33's water columns when a level-5 valve was clicked repeatedly (5 wrong-line readings in exp-073b), ar25's top layer
as the whole budget (113 of 143 near-end readings 6+ actions low against its counter), lp85's free clicks as charged
(+3 to +5), and blocked moves as phase skips (ls20, -3). The remembered-bar rule, the thickness rule, the
drains-into-the-same-colour rule and the charged-action time removed all four. A recent-window rate was tried and
dropped (more near-end errors: 83 against 73).

## 4. The flag and EXPOSE_RESET

`OURS_BUDGET_METER=1` turns everything on; it is read at call time, so the arm sets it in cell 4 (`--env-add`).
`EXPOSE_RESET` is Franzen's existing flag (`action_names.reset_exposed()`, default off). What it changes:

1. RESET is listed among the valid actions (TAAF always offers it): in "Valid actions right now", the sandbox's
   `valid_actions` and every action result.
2. With `ARC3_ACTION_INFO=1` (on in his notebook) the system prompt gains `RESET_INFO_ADDENDUM` (428 characters,
   ~110-130 tokens, once per game): RESET restores the level and its bar, keeps completed levels, counts as one
   action, and must be the first game action of a snippet.
3. Snippet guard (`_handle_action`): a call with RESET anywhere but first, or after an action executed in the same
   snippet, is refused without executing anything (`reset_not_first`; the sandbox raises).
4. The known-no-op guard never blocks RESET and never learns it as a no-op.
5. After a deliberate RESET the next prompt says "You deliberately reset the current level..." instead of "You are
   still on the same level."
6. The RESET itself goes through the normal action path and is recorded as `RESET` with `automatic: False`. His
   notebook sets `ONLY_RESET_LEVELS=true`, so arcengine restarts only the current level, and in competition mode
   arc_agi never forwards a RESET at a level's action count 0 (it is billed, nothing resets): a double RESET cannot
   wipe completed levels.

**Safe with the meter: yes.** Every RESET entry starts a new attempt in the detector, so readings restart and stay
silent until 3 new drops; the budget-death verdict needs a game-over prompt and a history ending in the automatic
RESET, which a deliberate RESET never produces; and the meter's text gives no advice to reset. Checked: a unit test
(attempt restarts after RESET), the harness check (a deliberate RESET followed by 2 drops: "You deliberately reset..."
and no budget line), and the bed (10 deliberate RESETs, no errors, both budget deaths named). The cost of the flag is
behavioural: each RESET is a billed action and spent actions stay spent. The meter is meant to supply the number
that tells an unrecoverable attempt from a recoverable one, which is the use Franzen built RESET for
(CONFIGURATION.md: "recovery from an unrecoverable position without spending actions merely to exhaust the bar").

## 5. How it was tested

- **`tests/test_ours_budget_meter_patch.py`** (15 tests, 2.4 s): the detector loaded from the patch text (no repo or
  game files needed): synthetic bars (silence until 3 drops; slow bar with the phase correction; tu93-style rounded
  bar near the end; RESET and level change; a line that rises; a 3-thick block; two disagreeing bars; a remembered bar
  outranking a new line; drains into a new colour; free kinds and blocked moves; death verdicts; the incremental cache);
  the demo fixture `tests/fixtures/franzen/budget-bar/demo-edges.json` (the edge region of all 1,337 demo frames,
  121 kB; rebuilt frames give the same readings as the full boards on all 1,327 steps): 9 of 10 games, the sc25 and
  tu93 readings and verdicts above. In the bed venv on the tree the notebook builds (his patch, the sandbox patch, this
  one): **with the flag off, the system prompt, three user prompts (mid-attempt, game over, after a deliberate RESET),
  the sandbox's initial state and the tool result are byte-identical to the tree without this patch**; the flag alone
  on the unpatched tree changes nothing; with it on, exactly the meter line and the verdict are added and `budget`
  reaches the sandbox.
- **Full suite**: 360 passed, 2 skipped (no `vendor/`, no torch; both unrelated), slow bed test included; ruff clean
  on everything added. `franzen_tree.py check` applies his patch, the sandbox patch and this one. The slow bed test
  (`tests/test_franzen_bed.py -m slow`: the sample patch, not this one) is flaky on this box: it failed its "UNDO
  executed" check twice in this session, once with the committed `franzen_bed.py` and once with the changed one
  (sb26 never reached the undo slot in its 90 s), and passed in the final full run.
- **CPU bed** (`scripts/franzen_bed.py --games ls20,vc33,sb26 --seconds 150 --patch
  kaggle/franzen/patches/ours-sandbox-timeout-keeps-work.patch --patch kaggle/franzen/patches/ours-02-budget-meter.patch
  --env-add EXPOSE_RESET=on --env-add OURS_BUDGET_METER=1 --expect "Budget bar ("`, 155 s, final scripts): **14 of
  14 checks**; 337 requests; the meter line in a user message of 173 requests (transcripts: 43 lines in ls20, 59 in
  vc33, 0 in sb26); 2 game overs (one in ls20, one in vc33), both budget deaths by their frames and both named
  ("rows 61-62, colour 'Y'", "row 0, colour 'P'"); 12 deliberate RESETs (6 "You deliberately reset" prompts each
  in ls20 and vc33); ls20 and vc33 won level 1 by script; 65 retained-function reuses, 0 misses; 46 `frame_diff`
  calls; 45 batch no-op stops; 12 stale-state refusals; 6 UNDOs (sb26); 2 context overflows, both recovered; no
  tracebacks. `budget_meter_eval.py` on the run's own frames: 182 readings in 333 steps, 0 on a wrong line, 44 near
  the end with 0 off by more than 2, 12 against the counters with 0 off. (An earlier bed run in this session,
  before two whitespace/UNDO fixes to the bed script, gave the same picture: 14 of 14, 341 requests, 2 named budget
  deaths.)
- **Speed**: the incremental reading took at most 8.3 ms per step over 7,677 steps (demo and exp-073b), and gave the
  same readings as the from-scratch evaluation path on all of them.

Changes to shared scripts (small): `scripts/franzen_bed.py` lets `--expect` match a user prompt as well as the system
prompt (fact `expect_user_seen`), its scripted level-1 solve ignores RESET in the action set (with EXPOSE_RESET
on, `valid_actions` lists RESET and the solve never fired), and its undo slot RESETs where RESET is offered and UNDO is
not (fact `reset_executed`). Runs without EXPOSE_RESET play exactly as before.

## 6. Known limits

- **No edge bar, no reading**: sb26, sk48 (2 of 25 public games). Bars further than 4 cells from the edge are not seen.
- **Layered bars**: a bar that drains into the next layer's colour (ar25 from level 3) is silenced only once the game
  has shown an earlier level draining into another colour. A game whose first level is layered would be read as its
  top layer: too few actions, and "less than one action left" at a layer boundary. Not seen in the 25 public games.
  A death exactly at a layer boundary would be named a budget death; none occurred.
- **Variable costs**: the rate is the attempt's average. Falls (dc22: 20 steps), penalties (su15) and other costlier
  actions make the reading optimistic until they happen; they cause 71 of the 77 near-end errors and all 82 counter
  disagreements. `largest_drop` in `budget` shows when an action cost more than the usual drop.
- **Slow bars** (one cell every p actions) are exact only to about p/2 near the end; quantization cost lp85 3 readings.
- **The last action**: some games end the attempt on the action that empties the bar (tu93, su15, wa30), others on
  the one after (sc25). The floor makes the reading the count of actions that can still be taken in the demo's
  deaths, but readings can be one high or low; errors near the end are mostly 0 or +1 against the realized truth.
- **First attempt of a game**: no remembered bar yet, so a lone border object drained by repeated identical actions
  could be read; the 3-thick and steadiness rules caught every case in the recordings (0 wrong lines), hidden games
  may bring new ones. Validation games go silent near the end more often (50-65% coverage).
- **Unknown on a GPU**: whether the model reads the line, plans to it, resets instead of burning actions, or dies of
  the budget less. Bed results are about mechanics, not play.

## 7. Prompt-token cost

- Meter line: 77-157 characters, mean 152 over the 847 demo readings, so about 38-46 tokens (4.0-3.3 characters per
  token), in turns where the meter has a reading (64% of demo steps). At ~50 turns a game that is ~1.5-2.3k prompt
  tokens a game, in user turns that are cached afterwards; the uncached cost is the line once per turn.
- Budget-death verdict: 99 characters, ~25-30 tokens, once per named budget death.
- `budget` in the sandbox: no prompt tokens unless the model prints it (about 100 tokens of JSON).
- `EXPOSE_RESET=on` (the arm sets it too): RESET_INFO_ADDENDUM, ~110-130 system-prompt tokens once per game
  (cached), and ", RESET" in each valid-actions line.
- CPU: at most ~8 ms per reading; the harness reads once per prompt and once per sandbox state refresh.

## 8. An arm

The current full-length config (exp-075: D', REAP-448, 14 streams, MTP acceptance 0.5, the sandbox patch, input
fallback) plus the meter and RESET. These flags without the last patch and the two `--env-add` rebuild exp-075's
notebook cell for cell (checked; only the description cell differs):

```bash
.venv/bin/python scripts/build_franzen_nb.py --base dprime --full25 121 --input-fallback --wait-inputs 120 \
  --reap-kept kaggle/franzen/reap448_kept_experts.json \
  --cfg MAXREQ=14 --cfg CUDAGRAPH_MAXBS=14 --cfg MAMBA_CACHE=84 --cfg SPEC_ACCEPT_SINGLE=0.5 --cfg SPEC_ACCEPT_ACC=0.5 \
  --env ARC3_MAX_ACTIVE_STREAMS=14 \
  --patch kaggle/franzen/patches/ours-sandbox-timeout-keeps-work.patch \
  --patch kaggle/franzen/patches/ours-02-budget-meter.patch \
  --env-add EXPOSE_RESET=on --env-add OURS_BUDGET_METER=1 \
  --out "$SCRATCH/franzen/exp0NN" --slug arc3-dprime-r14a05-budget-meter-full \
  --note "M2 budget meter + EXPOSE_RESET on the exp-075 config (docs/research/beat-tufa/patch-budget-meter.md)"
```

Built once in this session (not pushed). To isolate the meter from RESET exposure, a third arm drops
`--env-add EXPOSE_RESET=on`. Read from the Save & Run, against exp-075: budget deaths per attempt (the verdict lines),
deliberate RESETs and what followed them, actions on levels that hit their budget, levels, and score; one run cannot
rank arms (identical runs differ by 1.5-2.3x on the LB, lesson 0018), the event counts can.

## 9. Reproduce

```bash
.venv/bin/python -m pytest tests/test_ours_budget_meter_patch.py
.venv/bin/python scripts/franzen_tree.py check --patch kaggle/franzen/patches/ours-sandbox-timeout-keeps-work.patch \
  --patch kaggle/franzen/patches/ours-02-budget-meter.patch
.venv/bin/python scripts/budget_meter_eval.py runs/exp073b-dprime-reap448-r14-accept05-full runs/exp054-fix-kv775-obj  # any run dirs
```

The evaluation needs the run outputs (`runs/*/kernel-output`, not in git) and `environment_files/`; Franzen's demo
output was a local download (`scratchpad/m2/franzen-output`).
