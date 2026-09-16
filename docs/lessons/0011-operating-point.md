The submission runs every hidden game in parallel for the whole 9 hours; a 20-minute, 8-game dev evaluation samples about one sixth of the calls a game will really get, so measure the long regime before tuning for the short one.
- arc3/kaggle.py: one global deadline (9 h minus the reserve) shared by all games, each in its own framework thread; the
  model server sees N concurrent games (N = hidden set size, unknown, 20-40?) for the full run. Per-call latency rises
  with N but the total number of calls in 9 h is set by server throughput: roughly 6x the calls per game of a
  1200 s / 8-worker dev run.
- Consequences: (1) the agent's long-horizon behaviour (context eviction, stagnation handling, memory across levels,
  hundreds of calls) is what scores, and it was never measured before exp-017/018 (2026-09-16 evening); (2) action
  efficiency (RHAE) matters more, not less, when time is plentiful: the model can afford to think and verify;
  (3) vLLM must hold N concurrent 32k contexts (max_num_seqs 32, ~1.0 M tokens of FP8 KV cache: about 30 games).
- Dev runs: exp-017 (3600 s/game, 8 workers) and exp-018 (all 25 games concurrently, 10800 s each) are the proxies.
Source: arc3/kaggle.py global_deadline, docs/status.md Kaggle runs, diag v2 KV-cache figure.
