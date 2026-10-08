# Fixtures for Daniel Franzen's harness

Used by tests/test_build_franzen_nb.py, test_franzen_tree.py, test_franzen_bed.py, test_franzen_report.py and
test_ours_budget_meter_patch.py.

- `sample-ours.patch`: our own sample harness patch, made with `scripts/franzen_tree.py build` + `diff` against the tree
  Franzen's notebook builds. It changes `ARC3-Inference/inference/agent/tool_agent.py` (Tufa Labs' Duck harness, MIT,
  as patched by Franzen, Apache-2.0): `OURS_SYSTEM_PROMPT_SUFFIX`, when set, is appended to the system prompt.
- `run-extract/`: an extract of the real output of Franzen's notebook
  (https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution, Apache-2.0; his Kaggle run of 2026-09-30,
  10 demo games, 25 minutes each), downloaded 2026-10-02 and cut down as follows: `benchmark.json` with each action
  history entry reduced to its action id; `summary.txt` verbatim; `serve.log`: the first 240 batch, request-stats and
  HTTP lines among its first 2000 lines; `ft09-0d8bbf25_p0_requests.jsonl` with every message replaced by a stub
  derived from its sha1 (so counts, usage and prefix breaks are unchanged) and the tool schema removed;
  `transcripts/ft09-0d8bbf25_p0.txt`: only the turn headers and analyzer status lines; the notebook log: only the
  rows the report reads (gate, tail fade, timeouts, warmup, finished games, summaries). The games described belong to
  ARC Prize; the harness is Tufa Labs' (MIT) as modified by Franzen (Apache-2.0).
- `budget-bar/demo-edges.json`: from the same run's per-action boards (`artifacts/*_events.jsonl`, all 10 games,
  1,337 frames): per frame only the action, level, step, game-over flag, an interior id (equal interiors share it)
  and the edge region the budget meter reads (rows 0-3 and 60-63, columns 0-3 and 60-63 for rows 4-59), run-length
  encoded and stored only when a line changed. Rebuilding frames from it gives the meter the same readings as the
  full boards (checked on all 1,327 steps). The games belong to ARC Prize.
