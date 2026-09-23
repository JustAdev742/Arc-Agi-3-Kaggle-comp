Summary: on the Flash-Next server the KV cache, filled by 20k-token prompts, caps concurrency at about 3 requests; raise capacity or allocate calls better, but do not shrink the prompt by dropping history (exp-035).

# Flash-Next throughput is prompt-bound (2026-09-23, amended the same day after exp-035)

The Qwen3.8-Flash-Next NVFP4 weights take 81.8 GiB of the 96 GB card; the public profile gives the KV cache 5 GiB =
105,202 tokens. The model's QSA attention requires a BF16 KV cache (exp-033: FP8 fails at startup). With the Duck's
prompts at about 20.6k tokens, only about 3 of 25-28 games' requests run at once; each call waits about 127 s in the
queue for 19 s of work, and a game gets about 54 calls in 132 minutes. Scores were still rising at the time cap.

What follows:
1. Prompt tokens are the scarce resource, but the obvious cut is wrong: past-turn reasoning (35% of prompt tokens)
   and the 32,768-token history window carry the model's working state. exp-035 dropped old reasoning, instructions
   and images and lowered the window to 22,528: 4.2 requests ran instead of 3.1, yet requests per run did not rise
   (replies grew 38%, preemptions 21 -> 615) and the score fell from 7.86 to 3.58 (lesson 0022).
2. The levers that keep the prompt intact: more KV memory (8 GiB fails at startup; 6.5 GiB with 4,096-token prefill
   chunks is under test), prefix caching (the system prompt is shared; the public 27B fork saw 65% hits), faster
   decode (the launcher's flashinfer_b12x MoE backend, under test), and spending the fixed throughput where it is
   worth most (P21: calls on later levels first). Check vllm:request_prompt_tokens, preemptions and the acting share
   after any change to context handling.
3. Test any memory change under load (scripts/build_kv_stress_nb.py) before a full run relies on it.
