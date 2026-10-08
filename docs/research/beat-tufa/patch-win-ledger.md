Summary: patch ours-03 (method M4) gives the model exact, harness-computed facts about how each level was won: a short record in the level-up opener (about 280 tokens), and a ledger of every win record, the current level's game overs and the retained function names right after the system prompt at each context trim (about 400 tokens), where the trim has already broken the prefix cache. Flag OURS_WIN_LEDGER=1. On 77 recorded game-passes (329 level-ups) every one of 11,538 checkable clauses was true; with the flag off the harness is byte-identical; nothing has run on a GPU, so its effect on score is unknown.

# M4 win ledger: exact level records and a ledger re-pinned at every trim (2026-10-08)

Patch: `kaggle/franzen/patches/ours-03-win-ledger.patch` (2 files, +495/-2 lines, 10 hunks). It applies with `git apply`
after his patch and `ours-sandbox-timeout-keeps-work.patch`, in that order (`scripts/franzen_tree.py check --patch
.../ours-sandbox-timeout-keeps-work.patch --patch .../ours-03-win-ledger.patch`: "ok: 2 patch(es) apply on top of
his"). Everything below was run on CPU in this session; nothing ran on Kaggle or a GPU.

## 1. What it does

All text is computed by the harness from the recorded boards and history; the model writes nothing, and nothing in it
is an interpretation (no rule, goal or advice, only positions, counts, steps and actions).

**At each level-up**, the opener (after his "You have completed the previous level ..." text) gets a record. Real
example, produced by the patched code on Franzen's recorded ft09 run, replayed through the engine:

```
Level 2 record (exact facts from the recorded boards): won in 9 actions (steps 21-29), no game over.
Actions: MOUSE(row=17, col=39), MOUSE(row=26, col=48), MOUSE(row=35, col=48), MOUSE(row=17, col=39) x3, MOUSE(row=35, col=48) x3.
Before the winning action, changes since the level's first board include: recolored S->g (4px) at [35, 20], [35, 44], [44, 23]; recolored b->S (4px) at [26, 23], [35, 23], [35, 29]; recolored p->w (4px) at [20, 23], [35, 35], [44, 29]; recolored w->g (4px) at [26, 17], [29, 23], [38, 35]; more not listed.
The winning action itself, before the level ended, made changes including: recolored S->g (4px) at [35, 23]; recolored S->w (4px) at [35, 29]; recolored Y->S (4px) at [35, 32]; more not listed.
(4-cell edge band not compared; `level_wins` in python has the boards.)
```

- Header: actions spent on the level, the step range, every game over (step and fatal action), and the RESET that began
  the winning attempt when there was one ("Actions after that RESET: ...").
- Actions of the winning attempt, run-length encoded; a long list keeps its head and last run and says how many
  actions it left out (`..., ... 11 more actions ..., X`).
- Object changes from the attempt's first board to the board before the winning action, and the winning action's own
  changes from that board to the finished board (the engine's last layer before the next level; lesson 0017). Each
  clause is a plain statement about two boards, in his frame-diff vocabulary:
  - `moved` / `rotated`: the only object of that shape and colour on both boards is at a new place / pose
    (`rotated_by` is clockwise, as in his `frame_diff`);
  - `recolored`: the same cells hold the same shape in another colour;
  - `disappeared` / `appeared`: objects of that shape and colour at places where the other board has none; alike
    objects are listed by place with their counts on both boards, never paired into moves.
  - Not compared: objects whose box lies inside the 4-cell edge band (HUD and budget bars, his
    `ARC3_NOOP_GUARD_BORDER`). Not listed: parts carried inside a moved object, cells a moved object covers or
    uncovered, same-colour regions that only changed shape. Hence "include", and "more not listed" after 4 clauses
    (3 for the winning action).

**At each context trim**, one user message goes right after the system prompt. Real example (tu93, demo run, at the
step of its second game over on level 4):

```
Ledger of exact facts from the recorded boards, as of step 106 (placed here because older messages of this conversation were trimmed):
- Level 1: won in 48 actions (steps 1-48), no game over.
- Level 2: won in 15 actions (steps 49-63), 1 game over: step 52 (after RIGHT); the winning attempt began after the RESET at step 53.
- Level 3: won in 19 actions (steps 64-82), no game over. Actions: UP x2, RIGHT, UP, LEFT x2, ... Before the winning action, changes since the level's first board include: moved: color b 8px [42, 42]->[42, 18] rotated_by 180; ... The winning action itself, before the level ended, made changes including: moved: color b 8px [42, 18]->[42, 24] rotated_by 270; moved: color c 1px [44, 19]->[43, 26].
- Level 4 (current, since step 82): 24 actions so far, 2 game overs: step 85 (after RIGHT), step 106 (after UP).
- Retained functions: probe.
`level_wins` in python holds the win records with their boards.
```

The newest records stay in full; older ones are shortened to their header, oldest first, to keep the message under
1,500 characters. Every won level keeps at least its header line.

**In the sandbox**, `level_wins` is a list with one dict per won level: `level`, `first_step`, `win_step`, `actions`,
`game_overs`, `attempt_actions`, `record`, and three frames with `.ascii`/`.segmentation`: `start_frame` (the
attempt's first board), `prewin_frame` and `win_frame` (the finished board, or None when the animation did not show
it). It exists only when the flag is on.

## 2. Why (the evidence)

- **Earlier levels leave the context.** In exp-073b (our current serving config, 25 public games at the hidden set's
  121 min per game), the last request of each game held turns from only its last 1-3 levels: 95 of 124 won levels
  had no turn left in the final context, and in 14 of 25 games the context began on the level being played (the
  levels named by the `Current state: step N, level L` lines in each game's last request;
  `scratchpad/m4/tools/context_levels.py` on `runs/exp073b-.../kernel-output/prompts`). In his 25-minute demo every
  game trimmed at least once (12 trims, first at 13.6-20.8 min; franzen-run-analysis.md); the hidden set's ~48
  slot-minutes per game imply 3-4 trims.
- **What a trim cost.** vc33 L4 in the demo: after the trim at 16.4 min the model pressed a column "from level 3's
  layout", read the no-op as a false "cap rule" (about 3 calls), then spent 6 calls failing to rebuild level 3's win
  from `transitions` (franzen-run-analysis.md section 3; new-methods.md M4). On the old base, the stall analysis found
  all 10 re-derivation episodes began after the previous level left the context.
- **How a level was won is the strongest prior for the next:** the goal kind is constant within a game in 19 of 19
  dev games (lesson 0016).
- **Exact facts help, wrong ones hurt.** Exact object-change reports raised hard-game level-1 solves from 10/16 to
  15/16 over two paired runs (lesson 0024), and the review that made them work was about removing false statements.
  Franzen saw no gain from model-written summaries and a loss from structured world-model notes; this differs in being
  harness-computed and placed only where the prefix is already broken.

## 3. How it is wired into his harness

| Where | Change (flag on only, except the two marked "always") |
|---|---|
| `_build_user_prompt` | observes the history each turn (records new wins, the current level's game overs and RESETs); on the level-up opener appends the record after his level-start text |
| `_trim_messages_for_context` | removes any ledger before trimming, reserves its tokens in the budget, then pins the ledger after the system prompt (`_ours_ledger_pin`) |
| `_ours_ledger_pin` | rebuilds the ledger only when the first history message differs from the one it was last pinned above; otherwise puts the same message object back in the same place |
| `_force_reduce_messages` (overflow retry) | drops a real history block, never only the ledger (which the next trim would restore, looping on the same rejected request) |
| `_persistent_history_messages` (always) | counts history messages without the ledger when deciding whether the trim dropped any (identical when there is no ledger) |
| `_prune_control_kind_enabled` (always) | the `ledger` control kind is never pruned (no ledger exists with the flag off) |
| `_run_python_tool`, sandbox `_refresh_state` | `level_wins` added to the sandbox state, and exposed only when the state carries it |

The ledger is a user message tagged `_arc3_control: "ledger"`; `_strip_control_keys` removes the tag on the wire.

**No extra prefix break.** Without the patch a trim drops history from the front, so the next request differs from
the previous one right after the system prompt. The ledger is rebuilt only when that first history message changes,
so its change coincides with a break that already happens at the same place; between trims the identical message
object stays at index 1 and every request still extends the previous one. Within a turn no action executes before the
turn's last tool call, so the ledger's facts are current; a trim at commit can leave it one batch behind, which its
"as of step N" keeps exact.

## 4. How it was tested

**Fact-check on recorded runs** (`scratchpad/m4/tools/factcheck.py` and `replay.py`, run on the bundle built from the
patch file). Franzen's recorded runs with per-action boards: the 10-game demo (2026-09-30) and his v3 run (17 games x 4
passes; one empty pass skipped): 77 game-passes, 16,347 actions, 329 level-ups. Each pass's actions were replayed
through the offline engine (every replayed board equalled the recording) to get the animation chain the harness holds
for each winning action, and the engine's exact finished board (its `next_level()` wrapped: instrumentation the agent
never sees). The patched code then built the record at every level-up and the ledger at 432 check points (the 26
recorded context trims from the request logs, every game over, every level-up, each game's end), and a checker that
shares no code with the patch checked every clause: boundaries, counts, steps, game overs and RESETs against the
recording; action sequences by expanding the run-length encoding; object clauses with its own connected-component
analysis (scipy.ndimage) of the two boards, the winning action's clauses against the engine's exact finished board.

| | demo | v3 | total |
|---|---:|---:|---:|
| records / ledgers built | 40 / 63 | 289 / 369 | 329 / 432 |
| lines checked (record + ledger) | | | 3,532 (1,286 + 2,246) |
| clauses checked | | | 11,538 |
| errors | 0 | 0 | **0** |

The 11,538 clauses: 1,734 level boundary and action-count, 1,734 game-over, 45 RESET, 843 action-sequence, 4,733
object-change, 833 "more not listed", 775 winning-action, 409 current-level, 432 ledger headers. Not checkable on
recordings: the 432 "Retained functions" lines (taken from the same dict his opener lists) and the 429 `level_wins`
pointer lines. Also checked: all 306 recorded level switches happened inside the recorded winning step, which is what
makes "the level's first board" exact. 299 of the 329 records carry the winning action's own changes; the rest had no
animation chain showing the jump to the next level, so that clause is left out.

**The checker catches wrong facts.** Every single-fact corruption of the demo's 40 records and 10 end-of-game ledgers
(each number +/-1, each colour letter, each direction, appeared/disappeared swapped, "no game over" replaced): 5,392
corruptions, 5,392 flagged.

**Flag off is byte-identical; flag on breaks the prefix only after the system prompt.** `tests/franzen_ledger_checks.py
drive` runs his real `ToolAgent.analyze()` turn after turn (openers, sandbox, retained functions, trimming) on a real
game in the offline engine with a scripted model chosen by request count and an engine-backed `step_env`, in his
notebook's environment with the bed's 24k context. On ls20 and vc33 (22 turns, about 54 requests each, level-ups, a
game over on ls20, trims), the tree with this patch, flag unset or `0`, gives the same SHA-256 for every request on the wire,
the same stored history and the same transcript (turn-header clock times masked) as the tree without it. With the flag
on, the same game is played; every request either extends the previous one or diverges at index 1, where the ledger
then sits (one copy); between those breaks the ledger is unchanged; the opener after each level-up carries its record;
`level_wins` reaches the sandbox.

**Bed run** (`scripts/franzen_bed.py`: his real harness and solver with both patches, ls20 + vc33 + sb26 for 90 s each
against the mock model, the bed's 24k context so history is trimmed within minutes; `--seconds 90 --set
OURS_WIN_LEDGER=1 --expect "Ledger of exact facts from the recorded boards" --expect-in any`):

- 254 requests, 44 prefix breaks (trims). All 44 diverge at index 1, right after the system prompt, and carry the ledger
  there. 236 requests carry the ledger, always at index 1, never twice; one distinct ledger per break.
- 10 requests carry a level record (ls20 and vc33 win level 1 by script; the record stays until trimmed).
- 13 of 14 checks pass, including the `--expect` check. The failing one, "UNDO executed" on sb26, also failed in a 150-s
  pair with the flag off (on/off: 354/351 requests, 63/63 prefix breaks, cache share 82.3%/82.2%), so it is not the
  ledger. The mock picks its next program from markers in the conversation, and in sb26 every trim removed them, so
  its cycle restarted at `solve` and never reached `undo`. The repo's slow bed test (sample patch, 90 s) passed in this
  session. The flag-off run is also the negative control for `--expect`: no request held the ledger text.

**Tests.** `tests/test_ours_win_ledger_patch.py` (6 tests, about 1 min; one more marked slow runs the bed): the patch is
a harness-only git diff; it applies after his and the sandbox patch; unit checks in the bed venv (exact clauses on
synthetic boards: a unique move with rotation, alike objects by place, a recolour, a HUD bar and the reshaped
background not compared, a carried part and a covered marker left out, the 4-clause cut, run-length encoding, the
finished-board rule, a level with a game over and RESET; the trim re-insertion: no ledger before the first trim, one at
index 1 after, the same object between trims, never pruned, a forced reduction drops a real block; `level_wins` in the
sandbox and absent without the key); the flag-off identity and flag-on prefix drives on ls20 and vc33; the level-1
record headers on real boards. Whole suite: 350 passed, 2 skipped (no `vendor/`, no torch); the slow bed test of this
patch passed; `make lint`'s ruff scope is clean.
`scripts/franzen_bed.py` gained `--expect-in any` (default unchanged) and records where the text was found.

## 5. Cost

Measured on the 329 records and 432 ledgers above with `scratchpad/frspec/tokenizer.json`, which has the same
248,044-entry vocabulary and the same 33 special tokens (ids 248,044-248,076) as the served model's tokenizer config
(`scratchpad/fn_tokenizer_config.json`): about 2.7 characters per token (coordinates tokenize densely).

| | tokens: min | median | p90 | max |
|---|---:|---:|---:|---:|
| level-up record (once per level, in the opener) | 154 | 284 | 339 | 382 |
| ledger (once per trim, then cached until the next) | 60 | 402 | 516 | 574 |

- Per trim the ledger is prefilled once together with the rest of the re-prefill (his trims re-prefill about 55k
  tokens; franzen-run-analysis.md), so under 1% more prefill on that request; afterwards it is cached and takes about
  0.35% of his 116k window. The trimmer reserves its size, so trims start that much earlier.
- A hidden-set game with 5 level-ups and 4 trims adds about 1.4k + 1.6k prompt tokens prefilled once each, against a
  mean of 63k prompt tokens per request in his demo. No actions are spent.
- CPU: 0.1-0.3 s per level-up (three segmentations); the ledger text per trim call is built from cached records.

## 6. Known limits

- No GPU or Kaggle run: its effect on decisions and score is unmeasured. The bed's model is a script that never reads
  the ledger. new-methods.md estimated +0.3 to +1.5 LB and proposed a snapshot A/B (start at level k with no context)
  as the cheapest real test.
- The fact-check covers Franzen-harness runs on public games only (77 game-passes). It cannot check the retained
  function names against recordings.
- The winning action's own changes need `ARC3_ANIMATION` on (his notebook sets it) and an animation chain whose last
  change is the jump to the next level (299 of 329 here); otherwise that clause is omitted. If a game ever switched
  levels on a later action (not seen in 306 switches), "the level's first board" would be the finished board.
- Omissions are deliberate: the 4-cell edge band, carried parts, covered cells, reshaped regions, more than 4 (3)
  clauses. "moved" assumes identity only for an object unique on both boards; alike objects are never paired.
- A trim at commit time can leave the ledger one batch behind until the next trim; "as of step N" keeps it exact.
- No budget-death classification (M2) or falsified goals (M6) yet; they would be further ledger lines.
- Not checked against ours-02 (budget meter) or ours-04 (search helper), built in parallel: their hunks may touch the
  same functions (`_build_user_prompt`, `_serialized_runtime_state`, the sandbox's `_refresh_state`), so the order
  of application has to be checked when they are combined.
- Bed: with both patches, the bed's "UNDO executed" check failed on sb26 with the flag on and off alike (the mock's
  stateless program cycle restarts after each trim). Follow-up for the bed, not for this patch.
- The harness's own warm-up RESET (step 1 of the first game dispatched) counts as a RESET, so that game's level-1
  record says its winning attempt began after the RESET at step 1. True, but it reads oddly.

## 7. Building an arm

On top of the current submission configuration (exp-074t: D', REAP-448, 14 streams, MTP acceptance 0.5, input
fallback, the sandbox fix), checked to build with the apply check passing:

```bash
.venv/bin/python scripts/build_franzen_nb.py --base dprime --input-fallback --wait-inputs 120 \
  --env ARC3_MAX_ACTIVE_STREAMS=14 --cfg MAXREQ=14 --cfg CUDAGRAPH_MAXBS=14 --cfg MAMBA_CACHE=84 \
  --cfg SPEC_ACCEPT_SINGLE=0.5 --cfg SPEC_ACCEPT_ACC=0.5 \
  --reap-kept kaggle/franzen/reap448_kept_experts.json \
  --patch kaggle/franzen/patches/ours-sandbox-timeout-keeps-work.patch \
  --patch kaggle/franzen/patches/ours-03-win-ledger.patch \
  --env-add OURS_WIN_LEDGER=1 \
  --out build/exp-ledger --slug arc3-dprime-reap448-r14-accept05-ledger --note "M4 win ledger"
```

Add `--full25 121` for a full-length public-25 run. The control arm is the same command without `--env-add
OURS_WIN_LEDGER=1` (the patch is inert with the flag off), so the pair differs in the flag alone.

Re-run the fact-check on a new run's boards: build the bundle (`scripts/franzen_tree.py bundle DIR --patch ... --patch
...`), then `~/.cache/arc3-franzen-bed/venv/bin/python -I scratchpad/m4/tools/factcheck.py DIR/src/ARC3-Inference
environment_files OUT.json RUN/artifacts/*_events.jsonl` (`FACTCHECK_MUTATE=1` for the corruption test).
