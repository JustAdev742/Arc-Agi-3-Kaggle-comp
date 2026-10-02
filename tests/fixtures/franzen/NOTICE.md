# Fixtures for Daniel Franzen's harness

Used by tests/test_build_franzen_nb.py, test_franzen_tree.py, test_franzen_bed.py and test_franzen_report.py.

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
