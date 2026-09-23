Summary: on the Flash-Next server the KV cache, filled by 20k-token prompts, caps concurrency at about 3 requests; shrink prompts before anything else.

# Flash-Next throughput is prompt-bound (2026-09-23)

The Qwen3.8-Flash-Next NVFP4 weights take 81.8 GiB of the 96 GB card; the public profile gives the KV cache 5 GiB =
105,202 tokens. The model's QSA attention requires a BF16 KV cache (exp-033: FP8 fails at startup). With the Duck's
prompts at about 20.6k tokens, only about 3 of 25-28 games' requests run at once; each call waits about 127 s in the
queue for 19 s of work, and a game gets about 54 calls in 132 minutes. Scores were still rising at the time cap.

What follows:
1. Tokens in the prompt, not generated tokens, are the scarce resource. Every prompt token that carries nothing new
   (repeated instructions, old images, old reasoning, duplicate re-prompts) costs concurrency for all games.
2. The harness estimates tokens as JSON length / 3, counting base64 images and past reasoning that the server never
   sees; its budget is not the server's prompt size. Check the server metrics (vllm:request_prompt_tokens) after any
   change to context handling.
3. A larger `kv_cache_memory_bytes` is the other lever; about 12.6 GiB is free after the weights, so test it under
   load (scripts/build_kv_stress_nb.py) before a full run relies on it.
