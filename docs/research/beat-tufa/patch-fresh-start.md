Summary: OURS_FRESH_START=1 clears the conversation (everything after the system prompt) at the start of a turn once the current level has cost 80,000 generated tokens and 20 minutes without a level-up, keeping the exact ledger (ours-03b), the retained functions, `mem`, `effects()`, the board and the state line, and adding one neutral line; at most 2 per level, the second after 1.5x the gap. The threshold is at the 91.7th percentile of 228 solved levels in exp-073b + exp-075. With the flag off the harness is byte-identical. It was checked on the CPU only (units, drives over real games, a bed run with all seven patches), so its effect on score is unknown.

# Patch ours-07: a fresh start of the conversation on a stagnating level (2026-10-08)

Patch: `kaggle/franzen/patches/ours-07-fresh-start.patch` (sha256
`a00f479180a795d11bc1c629b8558dcb0464f2eaeb5bbc3fe810d43e44656b72`; one file, `inference/agent/tool_agent.py`, 94
added lines, nothing removed). It applies with `git apply` after his patch and ours-01, 02, 04, 03b, 05, 06b, in that
order, with no offsets. `scripts/franzen_tree.py check` with all seven reported "ok: 7 patch(es) apply on top of his".
Flag `OURS_FRESH_START=1`, read at call time, default off.

Everything below was run on the CPU in this session. Nothing ran on Kaggle or a GPU.

## 1. Hypothesis

Whether a game is won seems to depend on which hypothesis the model forms early on a level. A run that settles on a
wrong one keeps defending it. Franzen's trims drop only the oldest history, so the model's own recent reasoning, the
part that holds the wrong hypothesis, survives every trim.

A deliberate fresh start works like a restart in a randomized search. It turns some of these stuck runs into new draws.
It keeps the exact facts the harness holds and drops the model's own interpretations.

Evidence:
- **exp-075 vs exp-073b.** These two runs used the same configuration, and per-game outcomes split into two groups.
  - tn36 won 7/7 in exp-073b but sat on level 2 for about 2 hours in exp-075. cn04 and sp80 behaved the same way
    (research log, exp-075).
- **exp-073b failure analysis.**
  - 8 of the 17 games it did not win stopped on a mechanic that was never decoded.
  - Stagnation used 61% of the tokens spent on final levels.
  - The model defended wrong conclusions for a long time:
    - m0r0 believed for 43 minutes that two rooms could not connect.
    - sp80 concluded "The level as I understand it is unsolvable".
    - ka59 concluded "the level appears unsolvable".
- **The same level in the two runs** (section 2): where one run spent 140-280k tokens on a level without solving it, the
  other run usually solved that level cheaply.

## 2. Threshold: the data

**Source.** Per-action records in two `kernel-output/benchmark.json` files:
- `runs/exp073b-dprime-reap448-r14-accept05-full`
- `runs/exp075-dprime-r14a05-sandbox-full`

Each record holds the generated tokens since the previous action and the wall clock since the game's session started.

**How a level's cost is counted.** Levels are split by `actions_per_level`.
- **Tokens:** the sum over the level's actions.
- **Wall time:** from the previous level's winning action to the level's own winning action. For level 1, the time is
  counted from the game's first action. That is a lower bound, because the first turn's time is missing.
- **Gate parking is included.** In these runs about 38% of game-minutes were parked (failure analysis, section 2).
- **Levels being played when a game stopped** also get the tokens generated after their last action and the time to the
  game's end.

The scripts were run with `python -I` on the recorded files. They are in the session scratchpad, not in the repo:
`scratchpad/fs07/tools/levelcost.py` (per-level costs), `dist.py` (quantiles), `hazard.py` (Kaplan-Meier and pairs),
`scan.py` and `fireplan.py`. The result: 265 levels, 228 solved and 37 unsolved.

| Quantile | p50 | p75 | p85 | p90 | p95 | p98 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Solved levels, generated tokens (n=228) | 20.7k | 49.3k | 65.2k | 75.1k | 98.7k | 116.0k | 161.5k |
| Solved levels 2+, wall minutes (n=179) | 7.6 | 18.6 | 27.7 | 33.9 | 56.7 | 68.6 | 83.8 |
| Unsolved last levels, tokens (n=37) | 107.3k | 142.7k | 164.0k | 176.4k | 205.5k | 262.8k | 279.7k |
| Unsolved last levels, wall minutes (n=37) | 48.4 | 77.0 | 98.0 | 110.6 | 116.6 | 118.4 | 122.0 |

**The chance of solving falls as tokens pile up.** This is a Kaplan-Meier estimate over all 265 levels, with unsolved
levels censored at what they had cost.

| Already spent on the level, unsolved | At risk | Solved within the next 40k tokens | Within the next 80k |
|---|---:|---:|---:|
| Fresh level (0 tokens) | 265 | 0.60 | 0.81 |
| 40k | 102 | 0.52 | 0.69 |
| 80k | 42 | 0.36 | 0.54 |
| 100k | 31 | 0.30 | 0.51 |

Part of this decline is selection: hard levels stay hard. Part of it is what the hypothesis says, that a run gets stuck.

**The same level in both runs.** On 10 levels, one run spent 140-280k tokens without solving. The other run solved
those levels with 27-157k tokens (median 70k; 7 of the 10 under 80k):

| Level | Solved in | Stuck in |
|---|---|---|
| cn04 L2 | 41k | 256k |
| dc22 L1 | 38k | 193k |
| ft09 L5 | 27k | 168k |
| su15 L4 | 33k | 158k |
| tn36 L2 | 70k | 185k |
| sp80 L2 | 77k | 280k |
| wa30 L2 | 70k | 140k |
| sk48 L2 | 104k | 159k |
| re86 L6 | 96k | 143k |
| bp35 L2 | 157k | 171k |

These pairs differ in more than the draw: earlier levels, gate timing and the sandbox patch all differ too. Still, they
are the most direct evidence that a new draw on the same level can succeed.

**The rule over the two runs.** The table varies the first token threshold, with 20 minutes, growth 1.5 and at most 2
fresh starts per level. A fire is checked at each recorded action.

| First threshold | Solved levels interrupted (threshold's percentile) | Their tokens still to go, median | Unsolved last levels with one | ... with 20+ min left |
|---:|---:|---:|---:|---:|
| 60k | 33 (81.1) | 7.9k | 28 | 16 |
| 70k | 28 (87.7) | 7.6k | 26 | 14 |
| **80k** | **19 (91.7)** | **10.1k** | **23** | **12** |
| 90k | 16 (93.0) | 6.8k | 22 | 12 |
| 100k | 11 (95.2) | 3.3k | 20 | 10 |
| 120k | 5 (97.8) | 8.9k | 14 | 8 |

**Choice: 80,000 generated tokens AND 20 minutes on the level; the second after 120,000 more tokens AND 30 more
minutes; at most 2 per level.**

- **80k tokens** sits just beyond the 90th percentile of solved-level costs, as the brief asked: 91.7% of solved levels
  cost less.
  - Over the two runs the rule would have made 44 fresh starts:
    - on 19 of 228 solved levels (11 in exp-073b, 8 in exp-075);
    - on 23 of 37 unsolved last levels, 2 of them twice.
  - After the first fresh start on an unsolved level, a median of 24 minutes remained (12 of the 23 had 20 minutes or
    more).
- **20 minutes** is about the 80th percentile of solved levels' wall time (levels 2+). At 80k it did not bind in this
  data: all 19 solved levels above 80k tokens also ran 20 minutes or more.
  - It guards against a level where tokens pile up fast in little time.
  - In a competition rerun, all games run at once under the gate with 532 minutes each. Parking stretches wall time
    there, so the token threshold is what decides.
- **The growth and the cap of 2** keep a level from being reset over and over. In the data, only cn04 and sp80 (both
  exp-075) would have reached a second fresh start.
- **Not settled by the data.**
  - 90k interrupts 3 fewer solved levels than 80k and gives the same 12 useful fresh starts.
  - The difference is within the noise of 50 game-runs.
  - All four values can be changed per arm without a rebuild (section 8).

The cost the table makes visible: the solved levels the rule would interrupt were close to their win. At the fire point
they were a median of 10k tokens and 4 minutes from it. Section 6 covers that risk.

## 3. Design

**When it fires.** Only where a turn starts, at the top of `ToolAgent.analyze()`:
- after the opener's text is built and before any request of that invocation;
- so no request or tool call is in flight, and every tool call in history has its result.

**What it measures.** Harness state only:
- this agent's generated tokens (`_session_generated_tokens`, the counter Franzen's per-game limits use);
- the monotonic clock;
- both counted from the first turn that saw the current `current_frame.level`, or from the level's previous fresh start.

**When it is due.** When both reach the threshold times `growth**count`, and fewer than `max` fresh starts have fired
on this level.

**What starts the count over.** A new level, or a new game session. So nothing fires right after a level-up, and what the
winning turn of the previous level generated does not count toward the next level. A game over or a RESET keeps the
level and keeps the count.

**What it does:**
1. It empties `self._history_messages` (everything after the system prompt).
2. It calls his `_note_history_evicted()`, exactly as a trim does.
3. It sets `_resume_after_yield = False`, so the turn opens with his full opener and the current board, not a resumption
   form.
4. With OURS_WIN_LEDGER on, it sets 03b's ledger head to a value no message equals. His `_trim_messages_for_context`
   then rebuilds the ledger from the recorded boards and pins it after the system prompt (`_ours_ledger_pin`), as at any
   trim.

**What the requests look like afterwards.** Every request is built by his code path as usual: the trim with its ledger
pin, `_persistent_history_messages` at commit, and his control-message tags. The opener is not tagged "resume", so
history pruning never drops it. The first request of the turn is exactly `[system prompt, ledger, opener]`.

**What survives a fresh start:**
- the system prompt, which holds the tool documentation of 02, 04, 05 and 06b;
- the ledger (03b): every level's win record, the current level's game overs and RESETs, and the retained function names;
- the opener: his state line ("Current state: step N, level L."), the budget line (02), valid actions, the retained
  functions' signatures, the board image and the diff image;
- the retained functions themselves, `mem` (05), `effects()` (06b) and `level_wins` (03b) in the sandbox. All of these
  live on the agent, not in the conversation.

**What goes:** all assistant reasoning, tool calls and results, earlier openers, nudges and stubs.

**The line.** It is the first line of the opener, a neutral and exact statement:

```
Conversation history was cleared at step 41 after 9,050 generated tokens on this level without a level-up; the facts stated by the harness are exact.
```

The step is the opener's own "Current state: step N". The tokens are those generated on this level.

**A turn rolled back after the reset.** A request error makes his `finally` block restore the history as it was before
the turn, which is the emptied history. The retry's opener then carries the line again, with the status
`ours_fresh_start_repeated`. Once a turn is committed, the line is not repeated.

**Errors.** A failure inside the check is logged ("ours fresh start check failed") and the turn goes on without a fresh
start.

**Logging, for a run report:**
- **Notebook log** (WARNING): `ours fresh start 1/2: ls20-9607627b_p0 level 2 step 41 after 9050 generated tokens and
  0.3 min on the level; 8 history messages dropped`.
- **Transcript:** `[ANALYZER STATUS] ours_fresh_start: conversation history cleared at step 41 (fresh start 1 of at most
  2 on level 2) after 9050 generated tokens and 0.3 min on the level without a level-up; 8 history messages dropped;
  ledger kept.`
- `scripts/franzen_report.py` counts these under the transcripts' statuses as `ours_fresh_start`. The bed run printed
  `statuses {'ours_fresh_start': 6, ...}`.

## 4. The priority scheduler (D')

The fresh start records the clear with `_note_history_evicted()`, the same call his trimmer makes. Here is what follows:

- **One handover.** At the turn's first request, his `_maybe_handover()` consumes the flag. It re-prices the game and
  calls `gate.handover()`: the game gives its slot back and competes for it again, as at a trim.
- **The game is priced as at any trim.** The fresh start does not touch the counts D' reads: actions and generated tokens
  on the level, the pace tracker, and the priority bands. So D' prices the game with the level's true cost. Because C
  decays with tokens on the level, a stagnating game ranks low. If games are waiting, it is parked until a slot frees, as
  it would be at its next trim.
- **The trim schedule shifts.** Without the fresh start, the game would have kept its slot until its next trim (his
  58k-token drain). After it, the context is small, so the next trim and its handover come much later. The extra handover
  moves the game's next yield earlier by at most one trim interval.
- **The re-prefill is small.** In the bed, the first request after a fresh start had about 8.7k prompt tokens, about 6.9k
  of them cached (system prompt and tools). After a trim his runs re-prefill about 55k.
- **Nothing in the gate itself changes.** The gate's state (heap, snapshots, admitted set) is touched only through his
  `handover`.
- **Checked:**
  - units: one handover, then none, and the gate's level counts are unchanged;
  - drive: each fresh start's first request came right after a handover, and every handover came before a request
    that broke the prefix;
  - the flag-on drive ends with the same gate inputs (tokens, level-start tokens, level-start actions, last completed
    level) as the flag-off drive of the same game.

## 5. How it was tested

`tests/test_ours_fresh_start_patch.py` has 8 tests plus 1 slow test. All passed in this session.

**What the patch is.**
- It is a harness-only git diff that only adds lines.
- It applies in the bundle order 01, 02, 04, 03b, 05, 06b, with all five flags' code underneath.
- No key is added to cell 4.

**Units.** Run in the bed venv on the patched tree (`tests/franzen_fresh_start_checks.py units`).
- *The rule on synthetic token/clock histories:*
  - each threshold alone is not enough (tokens fast and clock slow, and the reverse);
  - the second fires exactly at 1.5x the gap, and never a third;
  - a level-up starts over, with nothing on the turn of the level-up;
  - the tokens of the winning turn do not count toward the next level;
  - a new session starts over;
  - settings are read at call time, and `MAX=0` disables;
  - the defaults are 80k / 20 min / 1.5 / 2.
- *The reset, on an agent with a pinned ledger and 8 trimmed turns:*
  - the history is emptied, `_context_was_trimmed` is set, and the opener is not a resumption;
  - the next request is `[system, rebuilt ledger, opener]`, and the ledger holds the exact level-1 record, the current
    level and the retained function;
  - the opener starts with the line and lists the retained functions;
  - the same ledger object stays through the turn and the commit;
  - a rolled-back turn repeats the line, and a committed one does not;
  - at most 2 per level; a new level gets a new count;
  - with the ledger off: `[system, opener]`.

**Flag-off identity.** `tests/franzen_ledger_checks.py drive`:
- Setup:
  - his real `ToolAgent.analyze()` over 22 turns of a real game in the offline engine (sandbox, retained functions,
    trims, a level-up);
  - a scripted model that replies by request count.
- Compared: the tree without the patch against the tree with it, with the flag unset, `0`, and `1` at the default
  threshold, which these drives never reach.
- Covered: ls20 with the other five flags on, ls20 with them off, and vc33 with them on.
- Result: the same SHA-256 for every request on the wire, the same stored history and the same transcript (turn clock
  masked).

**Flag on, low threshold.** `tests/franzen_fresh_start_checks.py drive`: ls20, 30 turns, every bundle flag on, a
1-slot gate, 1,000 tokens and 0 minutes.
- The game is the same as with the flag off: same actions, level-ups, game overs and request count.
- Exactly two fresh starts fired, both on level 2, and none on level 1, which is won by the second request.
- Both were the first request of their turn, with 3 messages (system, ledger, opener).
  - The ledger held "Level 1: won in 13 actions (steps 1-13), no game over." and "Retained functions: drive_probe."
  - Each came right after a handover.
- No request had a tool call without its result.
- Every request that carried the line had the ledger at index 1.

**Bed.** The command (shell variables written out as `P` = `kaggle/franzen/patches/`):

```bash
.venv/bin/python scripts/franzen_bed.py --games ls20,vc33,sb26 --seconds 120 \
  --patch $P/ours-sandbox-timeout-keeps-work.patch --patch $P/ours-02-budget-meter.patch \
  --patch $P/ours-04-search-helper.patch --patch $P/ours-03b-win-ledger-on-02-04.patch \
  --patch $P/ours-05-level-mem.patch --patch $P/ours-06b-effect-table-on-02-04-03b-05.patch \
  --patch $P/ours-07-fresh-start.patch \
  --env-add OURS_BUDGET_METER=1 --env-add OURS_SEARCH_HELPER=1 --env-add OURS_WIN_LEDGER=1 \
  --env-add OURS_LEVEL_MEM=1 --env-add OURS_EFFECT_TABLE=1 --env-add EXPOSE_RESET=on --env-add OURS_FRESH_START=1 \
  --set OURS_FRESH_START_TOKENS=4000 --set OURS_FRESH_START_MINUTES=0.25 \
  --program search --program mem --program effects \
  --expect "Conversation history was cleared at step" --expect-in any \
  --expect-pinned "Ledger of exact facts from the recorded boards"
```

Result with the final patch: 26 of 26 checks passed and the child exited 0 after 128 s.
- **Run size:** 273 requests and 59 prefix breaks; the gate made 62 admissions with 31 waits.
- **Other mechanisms still worked:**
  - 4 context overflows, all recovered;
  - search, mem and effects checks all passed;
  - UNDO executed;
  - no tracebacks.
- **Fresh starts:** 6, two per level in each game (ls20 L2, vc33 L2, sb26 L1), so the cap held.
  - **The first request after each** was the first of its turn and held `[system, ledger, opener]` (6 of 6).
  - **Requests carrying the line:** 32, all of them with the ledger at index 1 (32 of 32).
  - **No tool call ever lacked its result** in any of the 273 requests (checked in the `*_requests.jsonl` logs).
  - **The ledgers were exact,** for example ls20's "Level 1: won in 14 actions (steps 1-14), no game over; the winning
    attempt began after the RESET at step 1". That is the bed's warm-up RESET, a quirk noted in patch-win-ledger.md.
- **An earlier run with the same command** used the patch before its last edit, the error guard around the check. It
  gave the same picture: 26/26 checks, 6 fresh starts, 33 of 33 requests with the ledger at index 1.
  - In that run one fresh start (vc33) fired on a resumed turn. Its opener kept his stale-summary lines ("Your previous
    exchange this turn produced no tool call ..."), which are true but refer to a turn whose messages are gone.

`scripts/franzen_bed.py` gained `--expect-pinned TEXT`, an additive option off by default:
- **Check 1:** TEXT sits at index 1 of every request that carries the `--expect` text.
- **Check 2:** at least one request consists of only the system prompt, TEXT and the message that first carries it.

**Suite and lint.** The whole suite (`pytest tests/`, slow tests deselected) gave 424 passed and 1 skipped, exit 0,
including this file's 8. After the last edit to the checks script (the scripted model's name), this file was run again
and its 8 passed. `ruff` is clean on the files added or changed.

**An arm builds.** Section 8's command built a notebook, and the builder's apply check passed for all seven patches. The
notebook is 991,989 bytes. That is above the ~0.9 MB guideline of lesson 0033 and above the largest notebook Kaggle has
accepted (927 KB), but below the 1.196 MB that was refused. ours-06b adds 44 KB and this patch 8 KB. It was not pushed.

## 6. Risks

- **Losing partial understanding.**
  - **The levels at risk were close to their win.** The 19 solved levels the rule would have interrupted were a median
    of 10k tokens and 4 minutes from winning at the fire point.
  - **Why the loss could be real.** Lesson 0022: stripping past reasoning at every call halved the score (exp-035).
    Here it happens at most twice per level, but a plan in progress is lost all the same.
  - **What softens it.** The exact ledger, `mem`, the retained functions and `effects()` survive, so the model keeps its
    tools and recorded facts. It loses only its prose.
  - **What to watch.** Levels where the model had a plan running when the history was cleared. ls20 and m0r0, for
    example, wrote step-by-step executors.
- **Repeating the same wrong hypothesis.**
  - **Why it can happen.** The new draw starts from the same board, prompt and recorded facts. A model that forms the
    same belief from the same evidence gains nothing, and the fresh start then costs a re-derivation and a slot handover.
  - **What the pairs show.** They show that other draws exist (10 levels). They do not show how often a fresh start
    within the same run lands on a different one.
  - **Retained functions can carry the old hypothesis back,** because their code encodes it. The opener lists them with
    his standard caveat ("Their presence does not mean they are correct"); this patch does not change that.
- **Little time left.**
  - 11 of the 23 unsolved levels with a fresh start had under 20 minutes left after it.
  - In the competition rerun's 532-minute game budgets, time is less of a problem, but the gate's parking stretches
    every level.
- **The scheduler.** One extra handover per fresh start: a stagnating game on its last level (D' B=0) will usually be
  parked right after a fresh start. See section 4.
- **Wrinkles in the text.**
  - **A fresh start on a resumed turn** keeps his stale-summary wording ("earlier this turn").
  - **After a game over within the same turn,** his reminder points to "an earlier message this turn" that is gone. The
    same reminder still names the sequence and the fatal action.
  - **The ledger's header** says older messages "were trimmed".
  - **The ledger's step convention differs from the line's.** The ledger's "as of step N" is the last recorded step; the
    line uses the opener's next step.
- **Settings outside his configuration.**
  - With ARC3_MEMORY_SECTIONS on, the opener's world-model block (the model's own notes) would survive a fresh start.
  - With rolling summaries on, a summary could be requested over an emptied history.
  - His notebook has both off.

## 7. What to read in a run's logs

- **Count.** Count the `ours fresh start k/N: GAME level L step S after T generated tokens and M min on the level` lines
  in the notebook log. The same events appear as `ours_fresh_start:` statuses in `transcripts/*.txt` (franzen_report's
  statuses). Expect roughly 20 per 25-game run: the rule fires 44 times over exp-073b + exp-075.
- **Outcome per fresh start.** Use the step S and benchmark.json's per-action records:
  - Was the level won afterwards?
  - What did the win cost (tokens and minutes) after the fresh start?
  - Compare with the Kaplan-Meier baseline above: 0.36 of levels unsolved at 80k are solved within the next 40k tokens,
    and 0.54 within 80k.
  - Count separately the levels that were close to a win (a plan executing, many actions just before) and the stuck ones.
- **Same belief again?** In the transcript after each `ours_fresh_start:` status, read the first few reasoning blocks. Does
  the model reach the same conclusion as before (same claimed rule, same "unsolvable")? Or does it test something new?
- **Gate.** Admissions rise by about the number of fresh starts (1 + prefix breaks per game). Look at the parked time
  after each fresh start on a last level.
- **Repeats.** `ours_fresh_start_repeated:` statuses (a turn rolled back after a reset) should be rare. Any
  `ours fresh start check failed` warning is a bug.

## 8. Builder flags for an arm

The bundle arm (exp-077's configuration) with this patch, at full length:
- D', REAP-448, 14 streams, MTP acceptance 0.5, the ARC FR-Spec map;
- --fail-fast and --full25 121;
- all seven patches with all flags on.

This command was built locally and its apply check passed. It was not pushed.

```bash
.venv/bin/python scripts/build_franzen_nb.py --base dprime --out build/fresh --slug arc3-dprime-r14a05-harness7-fresh-full \
  --input-fallback --wait-inputs 120 --reap-kept kaggle/franzen/reap448_kept_experts.json \
  --cfg MAXREQ=14 --cfg CUDAGRAPH_MAXBS=14 --cfg MAMBA_CACHE=84 --cfg SPEC_ACCEPT_SINGLE=0.5 --cfg SPEC_ACCEPT_ACC=0.5 \
  --env ARC3_MAX_ACTIVE_STREAMS=14 --hot-tokens kaggle/franzen/hot_tokens_64k_arc.pt --fail-fast --full25 121 \
  --patch kaggle/franzen/patches/ours-sandbox-timeout-keeps-work.patch \
  --patch kaggle/franzen/patches/ours-02-budget-meter.patch \
  --patch kaggle/franzen/patches/ours-04-search-helper.patch \
  --patch kaggle/franzen/patches/ours-03b-win-ledger-on-02-04.patch \
  --patch kaggle/franzen/patches/ours-05-level-mem.patch \
  --patch kaggle/franzen/patches/ours-06b-effect-table-on-02-04-03b-05.patch \
  --patch kaggle/franzen/patches/ours-07-fresh-start.patch \
  --env-add OURS_BUDGET_METER=1 --env-add OURS_SEARCH_HELPER=1 --env-add OURS_WIN_LEDGER=1 --env-add OURS_LEVEL_MEM=1 \
  --env-add OURS_EFFECT_TABLE=1 --env-add EXPOSE_RESET=on --env-add OURS_FRESH_START=1
```

**The control arm** is the same command without `--env-add OURS_FRESH_START=1`. With the flag off the patch is inert, so
the pair differs in the flag alone.

**Other thresholds** need no rebuild of the patch. Add any of these with `--env-add`; none occurs in cell 4:
- `OURS_FRESH_START_TOKENS` (default 80000);
- `OURS_FRESH_START_MINUTES` (20);
- `OURS_FRESH_START_GROWTH` (1.5);
- `OURS_FRESH_START_MAX` (2).

**Size.** Mind the notebook size (section 5).

**Noise.** One run's SD is about 4.5 points (exp-075), so a single pair detects only large effects. The log readouts in
section 7 carry most of the weight.

## 9. Not verified

- No GPU or Kaggle run, so the effect on levels, score and time is unknown. The bed's model is a script that does not
  read the prompt.
- Whether a fresh start within one run gives a different hypothesis. The paired evidence comes from separate runs.
- The threshold rests on two runs (50 game-runs). The scan cannot tell 80k from 90k.
- Wall time in the data includes gate parking, and level 1's time is a lower bound. A competition rerun parks far more,
  so there the token threshold decides alone.
- The D' module itself is not in the bed or the drives: the bed runs his cells and his priority function. The handover
  path the fresh start uses is his and the same under D', which replaces only the priority value and fresh-game
  admission. That was checked by reading the D' cell, not by running it.
