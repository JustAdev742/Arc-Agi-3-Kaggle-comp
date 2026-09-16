A game gets about 35 model calls in its 20 minutes; design every helper and prompt line to save calls, not tokens.

- Measured (exp-003c, exp-007, Qwen3.8-27B-FP8 with 8 concurrent games on one RTX PRO 6000): 23-49 s per call,
  24-51 calls per game, 2.5 inspection-only calls per turn. RHAE counts only environment actions, so a call that
  does not end in `act(...)` is pure loss unless it changes the next action.
- Corollaries: put everything the model would otherwise print into the observation (entity list, events, fitted
  rules, coverage, untested actions, goal hints); make helpers return a plan the model can execute in the same
  call; verified execution of a long batch is worth more than any single probe; tell the model the real cost per
  call and the calls left.
- The per-call cost is a serving question too: fewer concurrent games or shorter outputs cut latency per call but
  the total throughput stays the same; the trade is measured by levels per game, not tokens/s (exp-004 pending).
Source: `docs/postmortems/exp007-inspection-budget-2026-09-16.md`, `runs/kaggle-repl-dev-007/*.transcript.jsonl`.
