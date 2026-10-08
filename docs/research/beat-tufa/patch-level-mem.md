Summary: OURS_LEVEL_MEM=1 gives the python sandbox a dict `mem` that persists across calls on one level (plain JSON only, up to 200 KB, emptied at a level change, a timeout keeps it, refused writes named in the tool result), for 3 system-prompt lines and a ~46-character field per tool result; with the flag off the harness is byte-identical; checked on the CPU only, so its effect on score is not measured yet.

# Patch ours-05: `mem`, a per-level data store for the python sandbox (M1c)

Built 2026-10-08 on the CPU (no GPU, no Kaggle). Patch: `kaggle/franzen/patches/ours-05-level-mem.patch` (sha256
`90068b01cd2c3ef14ea543cc581aa243bb24872e422327d2fa92f1c7441f30ea`, +215 lines in 3 files). It applies after his patch
and ours-01, and also as the last patch after the bundle order 01, 02, 04, 03b. Flag: `OURS_LEVEL_MEM=1`, read at call
time, off by default. Design source: `new-methods.md` M1 part (c).

## What it does

- Every python call sees a global `mem`, a dict. What a call leaves in it is there in the next call, as long as the game
  and the level are the same. Other variables still reset, as in Franzen's design.
- It is emptied when the level changes. If the level changes during a call, nothing that call wrote is kept: it may
  describe the old board. A game over or RESET keeps the level, so it keeps `mem`.
- It holds plain JSON only: dict with str keys, list, str, int, float, bool and None. A value must come back from JSON
  unchanged, so tuples, sets, int keys, dict subclasses (Counter, defaultdict) and objects are refused. The serialised
  size is capped at 200 KB.
- Writes are checked once, at the end of the call. A call that leaves anything else in `mem` (or deletes it, or replaces
  it with a non-dict) is refused as a whole: `mem` keeps exactly what it held before that call, and the tool result says
  why, naming the value. A call that times out or whose sandbox dies keeps the previous `mem` too. Writes made before an
  exception are kept, as in Python.
- Each tool result lists mem's keys with the `current_frame.step` at which each was last written, plus the step the call
  ended at and the size, so the age of every value is visible. An empty `mem` adds nothing.

What the model sees (verbatim). The system prompt gains three lines at the end of its tool-session rules:

```
- `mem` is a dict kept across `python` calls on the current level; it is emptied when the level changes. Other variables still reset.
- It holds only JSON data (str keys; lists, not tuples or sets) up to 200 KB. A call that breaks this has its changes to `mem` refused.
- Tool results list mem's keys with the `current_frame.step` at which each was last written.
```

Tool results gain a `mem` field (examples from the test scenario):

```
"mem": "walls (step 2), door (step 0), before (step 0), after (step 1); now step 2, 71 bytes of 200 KB"
"mem": "NOT SAVED: mem['walls'][0] is a tuple; JSON gives back only dict, list, str, int, float, bool and None. mem keeps its contents from before this call. Keys: walls (step 2), ..."
"mem": "NOT SAVED: mem would be 293.0 KB, over its 200 KB limit. mem keeps its contents from before this call. Keys: ..."
"mem": "This call did not finish, so mem keeps its contents from before it. Keys: walls (step 2), ..."
"mem": "mem was emptied because the level changed during this call (level 1 -> 2); nothing this call wrote to it was kept."
"mem": "mem was emptied because the level changed (level 2 -> 3)."
```

## Why: the evidence

- exp-073b's last-call logs (`exp073b-failure-analysis.md`, P1) hold 19 NameErrors on names the model had defined in an
  earlier call, in 10 games (3.5% of 545 calls), clustered on final levels. cn04 lost 4 calls in its last 21 minutes;
  m0r0 lost 2 calls and a rebuild in its last 2.6 minutes, when its plan was 4 moves short; sk48 lost 3 calls in 10.7
  minutes; ar25 rebuilt its masks on every call for 20 minutes.
- The models say so: "Variables reset between calls (only functions are retained). I need to rebuild everything in one
  call." (ar25, which also tried to cache masks in a mutable default argument); "W0, K, DOOR are not retained (they're
  data, not functions)." (m0r0); "I need to rebuild FREE/CORR each call (they're local)" (r11l, Franzen's demo).
- Franzen's demo run had 10 NameErrors in 531 calls (`new-methods.md` section 1).
- Franzen drops variables on purpose (stale board data). The safeguards here are the level scope, the emptying of
  anything written in a call that changed the level, and the visible write step of every key.

Expected effect (estimate from the failure analysis, not measured): +0.5 to +1.5 levels per 25-game run, by fewer lost
calls on final levels; no effect on actions per level.

## How it works: where state crosses the process boundary

| Step | Where | What |
|---|---|---|
| store | `ToolAgent._ours_mem` (host) | `{"scope": [game session dir, level], "values", "steps", "bytes", "note"}`; `inference/utils/level_mem.py` (new file) opens, closes and describes it |
| open | `_run_python_tool`, before the sandbox starts | the store is reused if the session and `current_frame.level` match, else replaced by an empty one (with a note if it held data and only the level differs) |
| to the sandbox | `_serialized_runtime_state`, wrapped in `_run_python_tool` | the first state built for the call (the initial state) carries `ours_mem` = values, steps, cap; action replies do not. The wrapper also notes the frame of every state it builds, so the host knows the call's last level and step without reading the state file again |
| in the sandbox | `main()` of the bootstrap, before `_available_names` | `mem` is a dict subclass built from the values; a top-level write (`[]=`, `update`, `setdefault`) to that object records `current_frame.step`; each key's JSON is snapshotted at start. Because `mem` is a provided global, retained functions may read it |
| back to the host | the `final` and `error` messages (`**_ours_mem_result()`) | values, steps and size if every value is exact JSON within the cap, else `{"refused": reason}`. A key changed in place (or by a `mem` that was replaced) gets step None, which the host fills with the call's last step. The check never raises: a failure reads as a refusal |
| pass-through | `run_sandboxed_python`, final/error branch | `ours_mem` is copied into the result only when the message has it; the timeout and crash returns never have it |
| close | `_run_python_tool`, right after `_record_retained_functions` | level changed during the call: empty store and a note; no `ours_mem` (timeout, dead sandbox): keep the store; refused: keep the store and name the reason; else take the new values. Then `payload["mem"]` |

With the flag off, the host never builds a store, so no `ours_mem` key reaches the sandbox; the sandbox then defines no
`mem` and adds nothing to its messages; `run_sandboxed_python` copies nothing; the prompt has no new lines. The sandbox
code sits inline in the bootstrap string (not injected from a module), so a bootstrap variant derived from it (ours-04's
search bootstrap) carries it too.

## How it was tested

All on the CPU, this session.

- `tests/test_ours_level_mem_patch.py` (8 tests, 9 s): a scenario of 23 real `ToolAgent._run_python_tool` calls (each a
  real sandbox subprocess) against a small fake game, run in the bed venv on the tree the notebook builds (his patch,
  ours-01, this patch), under the notebook's harness environment:
  - mem survives calls: written in one call, read in the next; write steps exact for top-level writes (0 and 1 around
    an action), the call's last step for an in-place change;
  - mem is empty after a level change during a call, and after one between calls; a game over keeps it;
  - refused: a set, a tuple inside a list, a tuple key, 293 KB over the cap, `mem` replaced by a list, `mem` deleted;
    each names the value, and the valid write made in the same call is gone too (never half-written);
  - a timeout (2 s limit) keeps the previous mem and, with ours-01, the retained functions; a retained function that
    reads `mem` works in later calls;
  - flag-off identity: the system prompt and all 23 tool results are byte-identical to the tree without this patch,
    with the flag unset and with `OURS_LEVEL_MEM=0`, under the notebook's environment and under the harness defaults;
    with the flag on the prompt is the old prompt plus exactly the three lines;
  - the host store's listing (40 keys at most, long keys cut) and the bed's `mem` program walking write, then read.
- The whole suite on the branch: 400 passed, 2 skipped, 2 deselected (slow) in 136 s; the new file alone 8 passed in
  8.8 s. `ruff` is clean on the files added or changed.
- Composition (one-off, not a test, since it depends on the other patches): the patch applies after 01, 02, 04, 03b;
  on that bundle with OURS_LEVEL_MEM, OURS_BUDGET_METER, OURS_SEARCH_HELPER, OURS_WIN_LEDGER and EXPOSE_RESET all on,
  the 23 scenario calls give the same `mem` field and stdout as on 01 + 05 alone.
- Bed run: `scripts/franzen_bed.py --patch .../ours-sandbox-timeout-keeps-work.patch --patch .../ours-05-level-mem.patch
  --env-add OURS_LEVEL_MEM=1 --program mem --expect '`mem` is a dict kept across `python` calls on the current level'`
  (ls20, vc33, sb26; 120 s each; 125 s wall): **19 of 19 checks**, child exit 0, no tracebacks. 299 requests, the
  prompt text in all 299 system prompts; ls20 and vc33 won level 1 by script; 3 context overflows, 3 recovered; 53
  prefix breaks; 51 retained-function reuses, 0 misses. The `mem` program (new, `--program mem`, additive): 12 writes,
  12 read back in the next call, 0 lost, 0 showing another level's value, 9 values found again in a later turn on the
  same level, 12 sets refused by name; 149 of 243 tool results carried the `mem` field.
- An arm builds: the builder command below produced a notebook listing "env added OURS_LEVEL_MEM=1" and both patch
  cells, with the builder's own apply check passing. It was not pushed.

## Cost

- Prompt: 3 lines, 363 characters (about 90 tokens), static, so prefix-cached after the first request of a game.
- Tool results: one `mem` field while `mem` is not empty. In the bed (one key) its median length was 46 characters
  (about 12 tokens; max 201 for a refusal). With 10 keys it is about 200 characters (about 50 tokens). It is input,
  not generated text, and it only sits in the context while that tool result does.
- Time: the store crosses the pipe twice per call. Measured on 15 calls each: 155.8 ms per call without `mem`, 154.6 ms
  with an empty `mem`, 185.9 ms with 181 KB of small lists (+30 ms; the check is pure Python in the sandbox).
- Actions: none.

## Known limits

- Exactness costs friction: models often hold coordinates as tuples and sets. Those are refused (the prompt says so in
  advance), and a refused call loses all its changes to `mem`, so the model has to convert and store again.
- A write step says when the value was stored, not when it was computed: data computed before an action and stored
  after it looks newer than the board it came from. Steps are `current_frame.step`; the user prompt's "step N" is one
  higher (it counts the next action).
- A call that changes the level drops everything it wrote, including data about the new level written after the
  level-up. That is deliberate (the old board is the common case).
- A game over keeps `mem` (same level), including values about the failed attempt's moving parts; the write steps show
  their age, but the model has to judge them.
- `mem` lives on the agent object: it survives context trims (that is its purpose), but nothing summarises it, and the
  listing shows keys, not values (at most 40 keys, keys cut at 40 characters).
- The bed cannot show a level change while `mem` holds data (its scripted level-1 wins happen in a game's first turn,
  before the `mem` program runs); the unit scenario covers that with a real sandbox.
- No score is measured. The cheapest readout is the NameError count in the last-call logs of a full-length arm against
  exp-075 (`exp073b-failure-analysis.md`, P1, cheapest offline test), then levels and score against the same baseline.

## Builder flags for an arm

The candidate's serving configuration (exp-074t / exp-075) plus this patch; `--full25 121` for a full-length test run:

```bash
.venv/bin/python scripts/build_franzen_nb.py --base dprime --out build/mem --slug arc3-dprime-reap448-r14-accept05-mem \
  --input-fallback --wait-inputs 120 --reap-kept kaggle/franzen/reap448_kept_experts.json \
  --cfg MAXREQ=14 --cfg CUDAGRAPH_MAXBS=14 --cfg MAMBA_CACHE=84 --cfg SPEC_ACCEPT_SINGLE=0.5 --cfg SPEC_ACCEPT_ACC=0.5 \
  --env ARC3_MAX_ACTIVE_STREAMS=14 \
  --patch kaggle/franzen/patches/ours-sandbox-timeout-keeps-work.patch \
  --patch kaggle/franzen/patches/ours-05-level-mem.patch --env-add OURS_LEVEL_MEM=1 --full25 121
```

In a bundle arm, add `--patch kaggle/franzen/patches/ours-05-level-mem.patch` after ours-03b and `--env-add
OURS_LEVEL_MEM=1` next to the other flags. The CPU bed for it: the bed command above, with `--program mem` added to any
other `--program` options.
