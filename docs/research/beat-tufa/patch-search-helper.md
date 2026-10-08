Summary: method M3 is built as `kaggle/franzen/patches/ours-04-search-helper.patch`. Behind `OURS_SEARCH_HELPER=1` the sandbox provides `search()` (BFS / A* / beam over a model the agent writes) and `run_plan()` (one real action at a time, stopping at the first surprise), described in 10 prompt lines (+461 tokens), and a call that calls `search(` gets its time_limit + 15 s. With the flag off the harness is byte-identical (tested). On recorded model code, `search()` finds tu93's 31-move plan, settles re86 (no plan exists in that model) in 0.1-2 s instead of a 30-s timeout, and finds a 7-click plan for r11l L3 in 0.12 s. In the real engine that plan fails at its first click, because the model's own map of free cells was wrong, and `run_plan` would have stopped right there. No GPU run yet.

# M3 on Franzen's base: a harness search service with guarded plan execution (2026-10-08)

CPU only: unit tests, the CPU bed, recorded model code from two real runs. Design: new-methods.md section M3. Evidence:
franzen-run-analysis.md and new-methods.md section 1. The patch applies after his patch and
`ours-sandbox-timeout-keeps-work.patch`, in that order, like every patch of ours (engineering.md section 1).

## 1. What it does

**The flag.** `OURS_SEARCH_HELPER`, read at call time with his `_get_env_bool` (`1`/`true`/`yes`/`on`), off by
default. An arm sets it with `--env-add OURS_SEARCH_HELPER=1` (section 7); the bed takes the same option.

**`search(start, step, is_goal, actions, key=None, heuristic=None, beam=None, max_nodes=300_000, time_limit=60)`**
runs in the sandbox process on the agent's own functions. Nothing is serialised, and functions that call it are
retained like any other function.
- `step(state, action)` returns the next state, `None` (illegal) or `'dead'` (fatal); both are skipped. `actions`
  is a list, or a function of the state.
- Without a heuristic it is BFS: the first goal generated is a shortest plan. With `heuristic` it is A*, with the
  goal test when a state leaves the queue and re-opening when a state is reached more cheaply. With `beam=N` it is a
  beam search that keeps the N best new states of each depth.
- Repeated states are dropped by `key(state)`. The default key is the state itself, with lists, sets and dicts frozen
  when the state is not hashable. Each state keeps one parent link, so the plan is rebuilt from links.
- It returns `status` ('found', 'exhausted', 'timeout', 'max_nodes'), `plan`, `expanded` and `best_partial`
  (the plan to and the state of the generated state with the lowest heuristic, or the deepest one without a
  heuristic, plus its depth and h). It also returns `seen`, `seconds` and `time_limit` (the seconds the search was
  allowed after clamping).
- Caps: time_limit at most 60 s; max_nodes (distinct states kept) at most 300,000.
- **Two clocks keep a search from losing the call.** First, its deadline is the earlier of time_limit and the call's
  own hard limit minus 5 s, counted from when the sandbox process started. A search therefore ends on its own and
  returns `best_partial`, and the snippet keeps running and keeps its stdout. Second, a clock check every 16 `step`
  calls is backed by a SIGALRM 1 s after the deadline, for a `step` that is slow on its own. `action()` raises inside
  a search, so a real action can never run from a hypothetical state, and it is restored afterwards.

**`run_plan(plan, observe, predicted, key=None)`** executes the plan one real action at a time through the
snippet's own `action()`, so every harness guard applies. After each action it compares `observe(current_frame)`
with the prediction (through `key` if given). `predicted` is a list with one expected state per action, or the
`step` function; with `step`, each prediction is made from the observed state, so one surprise does not cascade. It
stops:
- at the first mismatch;
- before an action that `step` calls illegal or fatal from the observed state;
- when an action is not executed (`stop` = the harness's stop_reason; the catchable stale-state refusal becomes
  `stop='stale_state'`, and the harness's other refusals still end the snippet, as the harness intends);
- at level_completed, game_over or run_complete;
- when less than 3 s of the call are left.

It returns `executed`, `mismatch_at`, `predicted`, `observed` and `stop`.

**The tool time limit.** His `_python_timeout` (30 s in the notebook) is unchanged except for a call whose code
contains a bare call `search(...)`, or calls a retained function whose source does (transitively). Such a call gets
`max(30, ceil(largest time_limit) + 15)`, where a time_limit that is not a number literal counts as 60, so at most
75 s. `re.search(`, `bfs_search(`, the text `'search('` and aliases (`f = search; f(...)`) do not count. An alias
call keeps 30 s, and the search then clamps itself to 25 s.

**The prompt.** Ten lines, `SEARCH_HELPER_ADDENDUM` in tool_agent.py, placed after the frame_diff lines and before
"Tool session rules". They describe signatures, semantics and limits. They do not tell the model when to use either
function, and they call neither function a solver. The model's own "IMPORTANT ... write an explicit search algorithm
such as BFS" line is his and stays.

**Code.**

| File | Change |
|---|---|
| `inference/utils/search_helper.py` (new, 258 lines) | stdlib only; spliced into the sandbox bootstrap like segmentation.py |
| `inference/agent/python_tool_sandbox.py` (+17, -1) | Two placeholder lines in the bootstrap. Flag off: both are dropped, so `_SANDBOX_BOOTSTRAP` is byte for byte the old one (same sha256, tested). Flag on: `_SANDBOX_BOOTSTRAP_SEARCH` holds the helper and adds `search`/`run_plan` to the provided globals before the retention bookkeeping. `run_sandboxed_python(search_helper=False)` picks the bootstrap. |
| `inference/agent/tool_agent.py` (+91, -1) | the prompt lines, `_search_call_timeout`, two arguments in `_run_python_tool` |

## 2. Evidence it targets

From one real run of his agent (the 10-game, 25-minute demo of 2026-09-30) and our exp-073b (25 games, 121 min):
- **r11l L3 (demo).** The mechanic was decoded, but four planner rewrites and a 30-s tool timeout followed. The
  timeout dropped all 13 retained functions, and the level got 1 action in 10.1 min. The model's own diagnosis:
  "The A* with 4 anchors x 400 candidates is too slow".
- **Timeouts and buggy planners.** ft09 L5: "Timed out (2^26 brute force is too slow)". tu93 L4: "I used `c=st`
  instead of the current state" (a reconstruction bug). In the demo, 28 of 531 tool calls wrote their own search.
- **re86 L4 (demo).** A 23-action plan ran open-loop on a wrong model: "All 6 dots should be covered - but the level
  did NOT complete". The hard-games study saw the same in sk48, ls20 and tn36 (open-loop batches of 15-41 actions on
  wrong rules).
- **exp-073b re86.** The model's own BFS on an unbounded state space timed out at 30 s. Its next two calls bounded
  the space, and both found no path (section 4).

## 3. Tests

`tests/test_ours_search_helper_patch.py`, 21 tests, about 30 s. 17 of them run on the copy of search_helper.py
inside the patch file, so they need neither his repo nor the bed venv; the other 4 run the patched tree.

- **Shortest paths.** On three toy grids (maze, a hazard on the direct route, a u-turn), BFS and A* return plans of
  the length an independent BFS computes, and beam plans replay to the goal. A start that is the goal gives `[]`,
  and `actions` may be a function of the state.
- **Deduplication and parents.**
  - A 6x6 grid with no goal: 36 states, 36 expansions, exactly 144 `step` calls; the deepest partial replays to
    (5,5).
  - List states are found through the default key.
  - A custom key merges states that differ in a counter.
  - An inconsistent but admissible heuristic makes A* re-open a state; the plan keeps the better parent.
- **Statuses.** Illegal and fatal moves are skipped. An unreachable goal is 'exhausted', with the closest partial by
  heuristic.
- **Timeout returns best_partial.** An endless space with time_limit 0.5 is cut by the clock check (under 1.5 s), and
  its partial replays to its state. A `step` that sleeps 0.4 s is cut by the alarm (under 2.5 s).
- **Caps.** time_limit 1000 becomes 60. max_nodes 10^9 stops at 300,000. A call with 8 s left gets at most 3 s, and
  a call with 4 s left returns 'timeout' at once.
- **No real actions inside search.** `action()` called inside `step` raises, and the real `action` is restored.
- **run_plan.**
  - A correct plan runs to level_completed (5 actions).
  - A planted mismatch, a wall the model does not know about, stops at index 1 after 2 actions, both with `step`
    predictions and with a list.
  - It does not act when its `step` says illegal or 'dead'.
  - It reports a refusal (`known_noop`), game_over and key-based comparison, and stops with 'time' when under 3 s are
    left.
  - The stale-state refusal becomes `stop='stale_state'`; a BaseException refusal still propagates.
- **On the notebook's tree.** The patch applies after his and ours-01 (3 of 3 files), and search_helper.py equals
  the patch's copy.
- **Flag off is byte-identical** (bed venv, the notebook's own cell-4 environment). Both trees, with only ours-01 and
  with ours-01 + ours-04 (flag unset, and flag `0`), give identical results for:
  - the bootstrap's sha256;
  - `ToolAgent._system_prompt`;
  - the time limit passed to each of 6 calls;
  - the 6 rendered tool results from real `_run_python_tool` calls against a fake game: print, `print(search)` →
    NameError, a model `def search` that is retained, an action, and a retained helper used later.
- **Flag on** (bed venv, base limit set to 10 s to make the limits visible):
  - The prompt is the old one with exactly the 10 new lines inserted before "Tool session rules"; "solver" is
    absent.
  - Call limits are [75, 35, 35, 16, 10, 15, 35] for: search with no time_limit; a def that calls
    search(time_limit=20); a call of that retained function; time_limit=1; an alias call; time_limit=0; the
    retained function again.
  - search + run_plan run against the fake game: plan found, 2 actions matched, a planted mismatch at index 0.
  - A retained function that calls search works in a later call.
  - A search that hits its own 1-s limit returns 'timeout' with best_partial and keeps the stdout. The alias call is
    cut to 4-5 s by the call budget and keeps its stdout.
  - A search call that then sleeps is killed by the host at 15 s ("Tool timed out after 15s"). The function
    retention note says the previous functions are still available (ours-01), and the next call uses them.
- **The time-limit rule on 17 code samples:** plain code; default, literal, positional, variable, `*args`, `**kw`,
  1e9 and 2.5 time_limits; `re.search`; `bfs_search`; the text `'search(1)'`; a retained `plan()`; a name that is not
  called; a transitive retained chain; a def in the snippet.

The rest of tests/ still passes (default selection), and so does `tests/test_franzen_bed.py -m slow` (94.9 s).
`ruff check` is clean on the test, on `scripts/franzen_bed.py`, and on the new module under the repo's rule set.

## 4. Recorded model code through search()

Script: `scratchpad/m3/tools/real_steps.py`, with code extracted from the `prompts/*.log` snapshots by
`scratchpad/m3/tools/extract_code.py`. The model's code runs unchanged in the patched sandbox
(`run_sandboxed_python(search_helper=True)`, call limit from `_search_call_timeout`); only the search call is new.

| Model code | What the model's own code did | search() |
|---|---|---|
| exp-073b re86, call 1: its literal `step(st, d)`, unbounded plane | BFS, "Tool timed out after 30s"; call lost | BFS: 'max_nodes' (300,000 states) in 1.65 s, deepest partial at depth 178. A* with the heuristic of its next call: 'max_nodes' in 2.25 s, best partial 21 actions at h=3. The whole call took 4.1 s and kept its stdout. |
| exp-073b re86, call 2: its `stp` with its own ±90 bound | A*: "expansions 18469 path None" | A*: 'exhausted' after 18,364 expansions (18,101 states, 0.12 s); BFS 'exhausted' in 0.10 s. In the model's model the goal is unreachable: no plan exists to find. |
| exp-073b tu93, call 1: its inline BFS, as step/actions (the chaser's options are part of the action, as in its loop) | 165 states, a 31-move plan | 'found': the identical 31-move sequence, 144 states, under 0.01 s |
| demo r11l L3, call #016: FREE built by the model's own code on the recorded board after action 20; plan_chain's transition as unit-cost clicks (select an anchor = 1 click, drag = 1 click), 3 + 441 actions per state | A*: "Tool timed out after 30s"; 13 retained functions lost | A* with its own heuristic (Chebyshev/14): 'max_nodes' at 300,000 states (45,043 expansions) in 33.3 s, no plan. Weighted A* (4 x its heuristic): a 7-click plan in 0.12 s. Beam 20: an 11-click plan in 0.13 s. Peak sandbox memory 147 MB (re86: 118 MB). |

**Does a found plan work?** `scratchpad/m3/tools/r11l_validate.py` replays r11l's 20 recorded actions in the offline
engine. This reproduces the recorded level-3 board exactly. It then clicks the 7-click plan.
- The first click is refused: the white anchor stays at (26,40) and the green blob at (18,34), where the model's
  model predicted the anchor at (52,61) and the blob at (24,39). Every later prediction is off.
- The cause is in the model's perception, not in search. The white anchor is drawn on top of the wall (row 26,
  cols 38-42: `SSSSS` before the click, `WWNWW` after). So the model's FREE set, "cells not S or g", counted wall
  under the anchor as free, and blob (24,39) looked legal.
- `run_plan(plan, observe, step)` would have stopped after that one click with `mismatch_at=0`. The model's own
  `layout2()` reads the anchor back as (26,40) against a predicted (52,61).

**Answer to "does search() find a plan within 60 s":**
- tu93: yes, in under 0.01 s.
- r11l L3: yes, in 0.12-0.13 s, with weighted A* or a beam. Plain A* with the model's own weak heuristic hits the
  node cap at 33 s.
- re86: no, and correctly so: its own bounded model has no path, which search reports in 0.1 s ('exhausted').

Where a plan was found, it is only as good as the model, and in r11l the model was wrong. That is the case
`run_plan` exists for.

## 5. The CPU bed

The bed gained a small additive option, `--program search` (scripts/franzen_bed.py). It appends one program to the
mock's cycle only when asked, so a plain bed run is unchanged. The program has two snippets:
- The first acts not at all. It runs a search over the step counter, which must get the longer call limit, and a
  search over an endless space, which must end on its own 2-s limit with a best_partial.
- The second runs `run_plan` over two actions whose second prediction is planted wrong.

Four coverage checks are added.

```bash
.venv/bin/python scripts/franzen_bed.py --games ls20,vc33,sb26 --seconds 120 \
  --patch kaggle/franzen/patches/ours-sandbox-timeout-keeps-work.patch \
  --patch kaggle/franzen/patches/ours-04-search-helper.patch \
  --env-add OURS_SEARCH_HELPER=1 --expect "searches a model of the game that you write; it never acts" --program search
```

Measured on the final patch (126 s wall):
- All 18 checks passed: the 14 standard ones (including retained-function reuse 45 / misses 0, stale-state
  refusals, UNDO, overflow recovery, no tracebacks) plus the 4 new ones.
- 271 requests, 269 answered; 2 context-overflow 400s, both recovered. The new prompt text was in the system prompt
  of all 271 requests.
- Gate: 50 admissions, 30 waits, 47 prefix breaks. 258 actions; ls20 and vc33 won level 1 by script.
- The search program ran 9 times:
  - The first search found its 2-step plan with the raised limit 9 of 9 times (time_limit 60.0, so the call had
    75 s).
  - The endless search ended at 2.0 s with a best_partial 9 of 9 times.
  - `run_plan` stopped at the planted mismatch 3 times (2 executed, `mismatch_at=1`). It stopped with
    `stop='stale_state'` after 1 action 6 times: the first action changed nothing on level 2, so the harness's
    stale-state guard refused the second.

An earlier bed run, before `run_plan` caught that refusal, showed it propagating out of `run_plan` in 11 of 16
calls. That is why it now returns `stale_state`.

## 6. Known limits

- **A held server slot.** A game that searches keeps its admission slot and decodes nothing for up to 75 s. At 14
  slots that is about 7% of decode capacity per searching game while it lasts. The gate hands slots over only at
  context trims, so nobody else gets that slot meanwhile. This is the price of the longer limit; the 30-s limit
  stays for every other call.
- **Wrong abstractions.** search() is exactly as good as the agent's `step` and `is_goal` (r11l above). `run_plan`
  bounds the damage to one action per surprise, but it cannot repair the model; that is M5's job.
- **Node cap.**
  - Cheap `step` functions reach 300,000 states in about 1.5-2 s (re86); a branching factor of about 440 reaches it
    in about 33 s (r11l). Plain BFS or A* with a weak heuristic then returns 'max_nodes' rather than a plan.
  - Each 300k-state search holds about 120-150 MB in its sandbox process.
- **The longer limit is syntactic.** It needs a bare `search(` in the call's code or in a retained function the code
  calls. An alias keeps the normal limit, and search then shortens itself to the call's remaining time minus 5 s.
- **A host kill still loses stdout.** If the snippet's own non-search code overruns, or a C-level operation blocks
  the alarm, the host kills the process as before: no stdout, and ours-01 keeps the retained functions. A model
  `step` with a bare `except:` can swallow one alarm; the clock check still ends the search within 16 calls.
- **run_plan's costs.**
  - Each action is its own `action()` round trip, which re-sends the whole history as a model-written loop does.
  - The harness's guards apply. A plan step that changes nothing makes the next action a stale-state stop. A known
    no-op, known death or repeat refusal ends the snippet with the harness's own message.
  - Plans from `search` do not contain no-ops, because repeated states are dropped, unless the agent's state holds
    something the board does not show.
- **Name conflicts.** With the flag on, an agent function named `search` or `run_plan` is not retained: the name
  conflicts with a provided global, and the tool result says so. Flag off, nothing changes; this is tested.
- **No GPU evidence.** The bed's model is a script. Uptake by the real model, and the effect on levels, tokens and
  actions, is unmeasured. Kill rule from new-methods.md: drop the arm if `search` is used on fewer than 10% of the
  levels where the control writes its own BFS.

## 7. System-prompt cost

The 10 lines are 1,711 characters and **461 tokens**, measured with the served model family's 248,077-entry
tokenizer (`scratchpad/frspec/tokenizer.json`). The system prompt under the notebook's cell-4 environment goes from
4,134 to 4,595 tokens (+11.2%). It is identical for every game, so it stays in the prefix cache. The costs are one
prefill of 461 tokens per server cache miss, and 0.35% of the 128 Ki window that history can no longer use.
new-methods.md had estimated about 250 tokens; the first 6-line draft measured 488.

## 8. An arm

The submission candidate's configuration (exp-074t: D' + REAP-448 at load + 14 streams + MTP acceptance 0.5 + input
fallback + sandbox fix) plus this patch and the flag, at full length on the public 25. This command was built
locally with the builder's apply check; nothing was pushed.

```bash
.venv/bin/python scripts/build_franzen_nb.py --base dprime --input-fallback --wait-inputs 120 --full25 121 \
  --env ARC3_MAX_ACTIVE_STREAMS=14 --cfg MAXREQ=14 --cfg CUDAGRAPH_MAXBS=14 --cfg MAMBA_CACHE=84 \
  --cfg SPEC_ACCEPT_SINGLE=0.5 --cfg SPEC_ACCEPT_ACC=0.5 --reap-kept kaggle/franzen/reap448_kept_experts.json \
  --patch kaggle/franzen/patches/ours-sandbox-timeout-keeps-work.patch \
  --patch kaggle/franzen/patches/ours-04-search-helper.patch --env-add OURS_SEARCH_HELPER=1 \
  --out build/m3 --slug arc3-dprime-r14a05-search-full --note "M3 search helper (OURS_SEARCH_HELPER=1) on exp-074t"
```

- **Control.** The same command without the last `--patch` and `--env-add` is exp-075's configuration. Applying
  the patch with the flag off is byte-identical to the control, so either can serve.
- **Submission version.** Drop `--full25 121`; the competition rerun gets the flag through `setup_env` like every
  `--env-add` key.
- **What to read.** new-methods.md suggests a level-snapshot pair (about 3 GPU-h) before a full-length run. Read:
  - search and run_plan calls per level, and status counts;
  - tool timeouts;
  - `stop` reasons;
  - minutes and generated tokens per solved level, and actions per solved level.
