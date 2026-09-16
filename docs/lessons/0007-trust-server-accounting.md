Never trust a client-side token estimate; calibrate against the server's usage counts, and treat a context-length error as "evict and retry", never as an outage.

- Qwen3.8 via vLLM: 576 tokens per 384 px image; code/JSON at ~chars/2.5. The agent now keeps an EMA of
  prompt_tokens/estimate and evicts against 32k minus reply reserve minus a 2k margin.
- A burst of identical errors is a hypothesis, not a diagnosis: check `/models` before declaring the server dead.
- Any fallback that spends actions must be capped by what the metric can tolerate (40 per game when the server
  is gone), because under RHAE 400 random actions and an unfinished level score the same.
Source: `docs/postmortems/exp003-context-overflow-2026-09-16.md`, `runs/kaggle-repl-dev-003/`.
