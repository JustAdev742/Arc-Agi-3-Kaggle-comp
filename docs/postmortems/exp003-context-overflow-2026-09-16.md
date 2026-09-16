# exp-003: four games (sb26, vc33, cd82, wa30) lost to a context-overflow cascade

**What we believed.** Prompt size was under control: the agent estimates tokens as chars/3 plus a fixed 300 per
image and evicts old turns when the estimate exceeds 32k minus the reply reserve and a 1k margin. Five consecutive
model errors meant the server had died, so the explorer should take over.

**What happened.** 40 of 48 model errors were HTTP 400 "maximum context length" from vLLM. Prompts estimated
under budget were over it: Qwen3.8 charges 576 tokens for a 384 px image (the diag measured it) and Python code and
JSON tokenize at closer to chars/2.5. After five overflows in a row the agent declared the server dead and switched to
fallback-only mode, whose cap was 400 actions. Four games spent 400 explorer actions in seconds and ended with 0.02
or 0 points; two of them (sb26, vc33) had already completed level 1 with the model.

**Which assumption was wrong, and what showed it.** "Our estimate is close enough" and "repeated errors mean the
server is dead". The kernel log's error histogram (`model call failed ... 400 ... maximum context length` x40) showed
both at once; `vllm.log` showed the server healthy with 8 running requests throughout.

**The cheaper test that would have caught it.** Comparing the server's `usage.prompt_tokens` with our estimate on
the very first smoke run (the field was already in every response) and asserting the ratio; and a unit test that an
error whose message names the context length is handled by eviction, not counted as an outage.

**Fix.** Calibrate the estimate with an EMA of prompt_tokens/estimate (start 1.3), 2k margin, 600 tokens per image;
on a context error evict the oldest block and retry immediately; timeouts widen the timeout instead of counting;
"dead" requires a failed `/models` liveness check; dead-server fallback capped at 40 actions. Tests in
`tests/test_repl_agent.py`. Re-run as exp-003c.

**Is the lesson general?** Yes: every number the harness estimates about the server must be checked against the
server's own accounting, and every fallback that spends actions needs a cap sized by the metric, not by time.
Promoted to `docs/lessons/0007-trust-server-accounting.md`.
