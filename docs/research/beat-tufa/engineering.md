# Engineering on the Franzen base: our patches, a CPU bed, a run report (2026-10-02)

Three tools let a research idea go from an edit to a checked notebook arm without a GPU or a Kaggle run:

1. **Our harness patches on top of his** (`scripts/franzen_tree.py` + `scripts/build_franzen_nb.py --patch`): edit
   the exact tree his notebook runs, turn the edit into a patch, build an arm that applies it after his patch. The
   builder refuses a patch that does not apply, and the notebook stops at cell 4 if one ever fails on Kaggle.
2. **A CPU test bed** (`scripts/franzen_bed.py`): his real harness, with our patches, plays real public games for a
   few minutes against a scripted mock model and reports which mechanisms ran (priority gate, guards, retained
   functions, history trimming, overflow recovery, UNDO, our patch's code).
3. **A run report** (`scripts/franzen_report.py`): per-game levels, actions against baselines, our scorer next to
   TAAF's, requests, tokens, prefix-cache hit share, gate admissions, and SGLang decode throughput, from his
   output format (benchmark.json, *_requests.jsonl, transcripts/, serve.log, summary.txt, the notebook log).

Everything below was run in this session; commands and results are quoted. Base: Daniel Franzen's notebook
(kaggle/franzen/, Apache-2.0), Tufa Labs' Duck harness and TAAF (MIT).

## 0. The tree his notebook runs is not his GitHub repo

His notebook copies Tufa's Kaggle source bundle (dataset `dfranzen/taaf-kaggle-source-bundle-copy`, "an unchanged
copy of Tufa Labs' original bundle") and runs `git apply --include=ARC3-Inference/* -v /kaggle/harness-changes.patch`
in its `src/` (cell 4; the patch is cell 2). His repo (github.com/da-fr/arc-agi-3-solution @ 10882e3, local copy
/home/user/da-fr/arc-agi-3-solution) is built on Tufa's GitHub release instead, and the two differ:

- Tufa's GitHub `ARC3-Inference/` + his patch == his repo, for every file under `inference/` (checked by applying
  his patch to a clone of Tufa's duck-harness @ 7652836 and diffing). Only Makefile, README, viewer/ and the added
  CONFIGURATION.md differ there; his patch does not touch them.
- The Kaggle bundle (downloaded 2026-10-02, 73 files, 439 kB) differs from Tufa's GitHub release in
  `inference/framework/run.py` (48 more lines, so his patch's two run.py hunks apply "with offset 48 lines", exactly
  as in his Kaggle log of 2026-09-30), `inference/tools/{eval,significance,traces}.py`, two extra
  `inference/utils/rearc_*.py`, seven TAAF files (`game_api.py`, `competition_arcade.py`, `deploy*.py`,
  `standard_benchmarks.py`) and pyproject/uv.lock/README/Makefile/configs/viewer. Everything else under
  `inference/agent` and `inference/utils` is byte-identical.

So a patch made against his repo could apply there and fail on Kaggle (or the reverse) in any of those 21 files.
`scripts/franzen_tree.py` therefore rebuilds the bundle byte for byte from his repo: reverse his patch, drop
CONFIGURATION.md, apply the vendored delta `kaggle/franzen/bundle/his-repo-to-bundle.diff` (100 kB), add the bundle's
root files (`kaggle/franzen/bundle/root/`: the pickled TAAF benchmark and deployment target, git status, preamble),
and compare every file with the sha256 manifest of the downloaded dataset (`kaggle/franzen/bundle/MANIFEST.json`).
Then it applies his patch with the notebook's own command and checks the result against the manifest again. A
download of the dataset can replace his repo as the source (`--bundle DIR`); both are checked against the same
manifest. tests/test_franzen_tree.py pins all of this, including the offset-48 hunks and the exact list of files
in which his repo differs from the notebook's tree.

## 1. Our harness changes as patches on top of his

```bash
# a git repo of the notebook's src/ (ARC3-Inference/, tufa-arc-agi-framework/), tags: bundle, franzen
.venv/bin/python scripts/franzen_tree.py build /tmp/fz
# ... edit /tmp/fz/ARC3-Inference/inference/... (new files are fine) ...
.venv/bin/python scripts/franzen_tree.py diff /tmp/fz > patches/ours-x.patch   # git diff HEAD, a/ b/ prefixes
.venv/bin/python scripts/franzen_tree.py check --patch patches/ours-x.patch      # his patch, then ours, as cell 4
.venv/bin/python scripts/build_franzen_nb.py --out build/x --slug arc3-franzen-x --patch patches/ours-x.patch
```

- `build DIR [--patch P ...]` applies earlier patches of ours too (tag `ours`), so the next patch is made on top of
  them; `diff DIR --base franzen` gives everything of ours as one patch instead.
- His repo is found at `--his-repo`, `$FRANZEN_REPO`, or /home/user/da-fr/arc-agi-3-solution. `bundle DIR` writes
  the whole `/kaggle/taaf-kaggle-source-share` as it is after cell 4 (pickles + patched src). `vendor --bundle DL`
  regenerates kaggle/franzen/bundle/ (maintainer only; needs a download of the dataset).
- In the arm, each `--patch` becomes a `%%writefile /kaggle/ours-NN-NAME.patch` cell right after his patch cell, and
  cell 4 applies them in order right after his `git apply`, in the same directory, with `git apply -v`. The cell
  raises (the notebook stops) unless git exits 0 and reports "Applied patch" for every file the patch names. The
  builder first runs his patch and ours on the rebuilt tree and refuses to build when that fails
  (`--no-apply-check` only when neither his repo nor the bundle is available). Patches must be `git diff` output
  confined to `ARC3-Inference/` or `tufa-arc-agi-framework/`, not binary.
- tests/test_build_franzen_nb.py executes the built arm's cells 2-4 (patch files written with IPython's
  `%%writefile` semantics, then the cell's own apply lines) against the rebuilt bundle and checks the result is
  byte-identical to what the builder checked; a patch that does not apply is refused at build time and, built
  with `--no-apply-check`, stops the cell with `did not apply: exit 1, 0 of 1 files`.

A pitfall the tools guard against: run inside a directory of another git repository, `git apply` resolves paths
from that repository's top, prints "Skipped patch" and exits 0 having changed nothing (reproduced in
tests/test_franzen_tree.py). `franzen_tree` and our notebook cell set `GIT_CEILING_DIRECTORIES` and count the applied
files. On Kaggle `/kaggle/taaf-kaggle-source-share` is not inside a repository, so his own line is safe there.

Other builder options (each anchored on text that must occur exactly once, listed in the arm's first cell, tested):

| Option | Changes | Notes |
|---|---|---|
| `--full25 MIN` | Save & Run plays all 25 public games, MIN min per game | 121 = the hidden set's compute per game; 25 = 25 games contending for his 10 slots. The rerun branch is his. |
| `--env K=V` | an existing key of `setup_env` / the priority dict (cell 4) | |
| `--env-add K=V` | a key cell 4 does not set at all, added to `setup_env` | refused if the key occurs anywhere in cell 4 (then `--env`) |
| `--cfg K=V` | one entry of `CFG = dict(...)` in the SGLang launcher (cell 12) | keeps the entry's type (int, float, bool, string; ints may be arithmetic like `(116+12+16)*1024`); `SERVED_NAME` refused. `MAXREQ` should move with `ARC3_MAX_ACTIVE_STREAMS`. |
| `--server-env K=V` | a string in cell 12's `env.update({...})` for the server | the key must occur once in cell 12, with a plain string value. Effective: cell 12 starts `sglang serve` via `subprocess.Popen(args, env=env)` and never sources the wheelhouse's `runtime_env.sh` (which also exports `SGLANG_SM120_ONLINE_MXFP8=0`, but only for a shell that sources it); nothing sets the key after `env.update`. |

Example: `--full25 25 --cfg MAXREQ=12 --cfg CUDAGRAPH_MAXBS=12 --cfg MAMBA_CACHE=72 --cfg MEMFRAC=0.975
--env ARC3_MAX_ACTIVE_STREAMS=12 --server-env SGLANG_SM120_ONLINE_MXFP8=true --env-add EXPOSE_RESET=on --patch P`.

## 2. The CPU bed

```bash
.venv/bin/python scripts/franzen_bed.py                       # his notebook, ls20 + vc33 + sb26, 150 s per game
.venv/bin/python scripts/franzen_bed.py --patch patches/ours-x.patch --set OURS_X=1 --expect "text our patch adds"
.venv/bin/python scripts/franzen_bed.py --notebook build/x/arc3-franzen-x.ipynb --games ls20,ft09 --seconds 300 -v
```

**What is real.** The arm notebook (his; or built on the fly from `--env/--env-add/--patch`; or `--notebook`) gives
his patch (cell 2), our patch cells, the harness environment (cell 4's `setup_env`, evaluated from the cell's own
statements with `TRUE_SUBMISSION = False`) and the solver settings (cell 16, executed as in a Save & Run).
`franzen_tree` builds the bundle and applies the patches as cell 4 does. A child process does cells 10, 14 and 20:
the bundle's source roots on `sys.path`, the pickled TAAF benchmark and deployment target, the offline arcade on
`environment_files/`, `asyncio.run(bm.run(...))` with full diagnostics. The child runs in a venv with what the
Kaggle image provides the harness (arc-agi 0.9.9, arcengine 0.9.3, numpy, scipy, matplotlib, imageio,
imageio-ffmpeg, pillow, requests, python-dotenv), created once with uv under `~/.cache/arc3-franzen-bed/venv`
(`$FRANZEN_BED_VENV`, or `--python`).

**What is mocked.** The model: an OpenAI-compatible server in the bed process. Each reply is reasoning plus one
`python` tool call (or deliberately none) from a fixed cycle of small programs, chosen statelessly from the
request (a marker comment in the last call it emitted names the next snippet):

| Program | What it does | Exercises |
|---|---|---|
| `solve` | first turn of a game: ls20's level 1 in 13 moves, vc33's in 3 clicks (found by BFS in the engine) | level-up path; level 2 arms the per-action guards (`ARC3_GUARDS_FROM_LEVEL=2`) |
| `define` | defines `bed_pick`, calls `frame_diff()`; the next call uses `bed_pick` without defining it | retained functions (`ARC3_PERSISTENT_FUNCTIONS`, scope game), `frame_diff` |
| `noop` | a batch walking into a wall / clicking a border cell | `ARC3_BATCH_NOOP_BLOCK` |
| `print` | dumps the board 12 times, then acts | tool-output middle truncation; fills the context |
| `stale` | repeats one action until it stops changing the board | `ARC3_STALE_STATE_BLOCK` (the harness raises `StaleStateActionError`) |
| `undo` | UNDO where offered (sb26), else a plain action | `EXPOSE_UNDO=on` |
| `chat` | text, no tool call | the "You have not acted yet" nudge |
| `think` | ~3k tokens of reasoning, no tool call | `LOCAL_ANALYZER_YIELD_TOKENS=2048`: the turn yields and resumes |
| (every 41st request, if long) | HTTP 400 with SGLang's "longer than the model's context length" | force-drain and retry (`context_overflow_recovered`) |

Usage: prompt tokens estimated from text (3.3 chars/token) and images (402 each), cached tokens = the longest
message prefix shared with one of the last 64 requests (page-rounded), completion tokens from the reply. The mock
writes an SGLang-style `serve.log`, so `franzen_report` reads a bed run like a real one; its server numbers are the
mock's, not a GPU's.

**Bed overrides** (printed at start, saved in `bed_config.json`): the model URLs; `ARC3_MAX_ACTIVE_STREAMS` = games
- 1 (`--slots`), so games wait at the priority gate; `ARC3_DIAG_CONCURRENCY=1` (gate counters in the log); and,
unless `--keep-context`, `LOCAL_ANALYZER_CONTEXT_WINDOW=24576`, `LOCAL_ANALYZER_MAX_OUTPUT=4096`,
`ARC3_CONTEXT_DRAIN_TOKENS=8192`, so history is trimmed (and the gate hands slots over) within minutes. `--set K=V`
adds environment for the harness process only (for example a knob our patch reads).

**Output** (`--out`, default a temp dir), laid out like `/kaggle/working`: benchmark.json, summary.txt,
transcripts/, prompts/, artifacts/, `*_requests.jsonl`, diagnostics.html, movies/, serve.log (mock), plus
`bed.log` (the child's output), `mock.jsonl` (one line per request: program, tokens, tool-result tags),
`bed_inner.json` (per-game state and the gate's own counters) and `bed_report.json`. About 65 MB for 3 games x 120 s.
The script prints the checks, the facts and the report table; it exits 1 if a check failed.

Measured (this session; `--games ls20,vc33,sb26 --seconds 120 --patch tests/fixtures/franzen/sample-ours.patch
--set OURS_SYSTEM_PROMPT_SUFFIX=[ours-bed-marker] --expect [ours-bed-marker]`, 127 s wall): 14/14 checks; 288
requests; ls20 and vc33 won level 1 by script; gate 56 admissions, 33 waits (120 s blocked in total), tail fade
logged; 53 prefix breaks (history trims); 55 retained-function reuses, 0 misses; 42 `frame_diff` calls; 40 batch
no-op stops; 8 stale-state refusals; 4 UNDOs on sb26; 12 nudges; 10 yields after `think`; 1 overflow error, 1
recovery; our patch's marker in all 288 system prompts; no tracebacks.

**What the bed cannot tell you.** Anything about score or play quality (the model is a script), timing or
throughput (mock latency 0.15 s + 3000 tok/s; the GPU's prefill and decode are absent), or the serving stack (no
SGLang). Its venv is close to, not identical with, Kaggle's image (package versions float except arc-agi/arcengine).
It tests that a patch applies, imports, runs on the paths it touches and does not break the mechanisms above; read
the effect of a patch on Kaggle runs.

## 3. The run report

```bash
kaggle kernels output OWNER/SLUG -p runs/franzen/x        # the run's /kaggle/working
.venv/bin/python scripts/franzen_report.py runs/franzen/x [--log runs/franzen/x/SLUG.log] [--json report.json]
```

Per game: state, levels completed of total, actions, actions on completed levels against their baselines, our
scorer (`arc3/scoring.py game_score` on the completed levels' action counts) next to TAAF's `final_score` (a
mismatch is printed as a warning), requests, prompt and generated tokens, prefix-cache hit share
(cached / prompt tokens from the server's usage), gate admissions, UNDOs. "Prefix breaks" are requests whose
messages do not extend the game's previous request: history trims. The priority gate hands a game's slot over
exactly there, so admissions = 1 + prefix breaks. Requests in `requests.jsonl` (made while only one game had a
runtime-state file, a harness quirk at the start and end of a run) are shown as "(unattributed)". From serve.log:
decode tok/s (mean, median, p90 of `gen throughput` over Decode batch lines with running requests), mean running
requests, max queue, MTP accept length, server-side cache share and output tokens from `ReqTimeStats`, prefill share
of forward time, HTTP statuses. From the notebook log: gate slots, the gate's DIAG counters when present, the tail
fade line, read timeouts, warmup resets, `[finished]` lines. Request logs exist only when `save_request_logs` is on
(Save & Run: yes; competition rerun: no).

On his real Save & Run of 2026-09-30 (10 demo games, 25 min each; downloaded output, 2.7 s to read 280 MB):

| game | state | levels | actions | completed / baseline | score (ours = TAAF) | requests | cache | admissions |
|---|---|---|---|---|---|---|---|---|
| ar25 | gave_up | 5/8 | 182 | 150/283 | 41.67 | 81 | 95.4% | 3 |
| ft09 | gave_up | 4/6 | 93 | 51/106 | 47.62 | 35 | 90.2% | 2 |
| lp85 | gave_up | 5/8 | 112 | 74/143 | 41.67 | 51 | 94.7% | 2 |
| r11l | gave_up | 2/6 | 20 | 19/55 | 14.29 | 45 | 93.9% | 2 |
| re86 | gave_up | 3/8 | 227 | 134/154 | 16.67 | 64 | 95.4% | 2 |
| sb26 | won | 8/8 | 162 | 162/213 | 93.34 | 49 | 93.4% | 2 |
| sc25 | gave_up | 3/6 | 157 | 51/74 | 28.57 | 59 | 93.9% | 3 |
| tr87 | gave_up | 4/6 | 184 | 166/197 | 47.62 | 56 | 94.5% | 2 |
| tu93 | gave_up | 3/9 | 107 | 82/69 | 13.07 | 45 | 92.9% | 2 |
| vc33 | gave_up | 3/7 | 83 | 57/69 | 21.08 | 52 | 94.2% | 2 |

Totals: mean 36.56 (= his summary.txt), 40 levels, 1327 actions; 537 requests (532 answered, 5 read timeouts),
33.7M prompt / 0.93M generated tokens, prefix-cache hit share 94.2%, 12 prefix breaks. Server: decode 671 tok/s mean
(median 723, p90 833), 8.9 running requests on average, max queue 1, MTP accept length 2.67, 617 output tok/s over
the 1516 s of requests, prefill 2.1% of forward time; gate 10 slots, tail fade at 20.5 min. His write-up's earlier
demo run reported 93.43% cache reuse and a p90 aggregate decode of 836.6 tok/s, consistent with these.

## 4. Tests

| File | What | Time |
|---|---|---|
| tests/test_franzen_tree.py | bundle rebuilt byte for byte; his patch applies as on Kaggle; where his repo differs; build/edit/diff round trip; drifted repo refused; the enclosing-repo pitfall | ~1 s |
| tests/test_build_franzen_nb.py | all builder options; the built arm's cells 2-4 executed against the rebuilt bundle; a bad patch refused and stopping the cell | ~1 s |
| tests/test_franzen_report.py | the report on tests/fixtures/franzen/run-extract (an extract of his real output; per-game numbers equal the full output's) and synthetic logs | <1 s |
| tests/test_franzen_bed.py | the mock and the cell-4 environment (fast); the whole bed with our sample patch (`-m slow`) | fast <1 s; slow ~95 s |

`pytest` deselects `slow` (registered in pyproject.toml); run it with `.venv/bin/python -m pytest -m slow`. Tests
that need his repo or the game files skip without them. tests/fixtures/franzen/sample-ours.patch is the sample
patch (adds `OURS_SYSTEM_PROMPT_SUFFIX`); its NOTICE.md gives attribution for the run extract.

## 5. Not done / open

- No Kaggle run was made with a `--patch` arm (none was asked for). The first one should be a no-op-ish patch
  (like the sample) to confirm the cell-4 apply step on Kaggle's git before a real change rides on it.
- The bed's mock never exercises rolling summaries, the death ledger/guard, RESET exposure or animation images
  (his competition settings leave those off or they need specific games); add programs when a patch touches them.
- `franzen_report` reads gate waits only from DIAG lines, which his notebook does not enable; on a real run the
  admissions column (1 + prefix breaks) is all there is.
