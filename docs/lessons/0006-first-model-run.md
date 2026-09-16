On real hardware the loop's cost is prompt size and turns without actions, not model quality; and a cheap fallback that spends actions is worse than doing nothing.

- 15k prompt tokens per call at ~1 s per 1k tokens prefill means 10+ s per call before any thinking. Everything in
  the per-turn observation must earn its tokens: no ASCII board when an image is attached, summaries not dumps,
  hard caps on tool output.
- A turn that ends without act() costs 1-2 minutes of the game's budget. Force a decision after a few inspection
  steps; let the model batch several actions per act().
- The explorer fallback took 146 actions in seconds when the model ran out of time. Under RHAE an unfinished level
  scores the same as one buried under random actions, so the fallback must be rare, small and capped.
Source: diag v5 REPL smoke, `runs/kaggle-diag-v5-repl-smoke/summary.json`.
