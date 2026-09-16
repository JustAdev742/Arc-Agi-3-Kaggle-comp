# Prompt limits other than tokens bite only at the submission's horizon: cap images, and never retry a 400 unchanged

Found 2026-09-16 in exp-018 (all 25 games concurrently, 3 h each; kernel `arc3-eval-all-long-a` v2, live log via
`GetKernelSessionLogsStream`). From 22:51, 1 h 23 min into the run, every model call for ft09 and later s5i5
failed with vLLM's `400 At most 16 image(s) may be provided in one prompt`; the agent retried the same prompt
554 and 340 times (the "model server dead?" path never triggered because the server answered). No 1200 s run and
no 3600 s run (exp-017, 8 workers) ever hit it.

Why: the REPL agent attaches one 64x64 board image to every observation and evicts history by an estimated
token budget only. Terse turns (one act() per call, short outputs) let 17+ observations fit under 32k tokens.
The limit is a count, not a token cost, so it appears only after enough turns, which only the 9-hour operating
point (lesson 0011) reaches.

Rules:
- Eviction must honour every server-side limit separately: tokens, images per prompt (`max_images`, default 12
  against `--limit-mm-per-prompt 16`), and anything else the serving flags name. Old observations keep their
  text and lose their image.
- A deterministic 400 is never retried unchanged. The error text says what to shrink; shrink it and retry at
  once (image limit: halve the cap; context length: force-evict). Only transport errors count toward "server
  dead".
- Long-horizon failure modes are found by long-horizon runs or by reading a live log; a 20-minute dev run
  cannot see them. Before a submission, run at least one game for the full per-game budget and read its log
  for repeated warnings.

Fix: `arc3/agents/repl_agent.py` (`_cap_images`, `_evict`, the image-limit branch of the error handler), tests
`test_image_cap_keeps_prompt_under_the_server_limit` and
`test_image_limit_error_strips_images_and_retries_without_counting_an_error`; commit d8e3ce4.
