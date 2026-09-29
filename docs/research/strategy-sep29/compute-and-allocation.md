# Compute and allocation: where the 9 hours go, and what separates 5 from 45

Sources: `runs/exp054-fix-kv775-obj/` (summary, benchmark, vLLM log and metrics), exp-054r/050/050r/032 summaries, `docs/research/levels2plus-exp054.md`, research log Sep 23-29, `docs/research/public-code-sep27*`. Numbers are ours unless a team is named.

## 1. Where a 132-minute game goes (exp-054, public 25, LB 4.70)

**Setup.** Server ready 1,260 s after kernel start (823 s weight load from NFS). 3.9% of the 540-min rerun; 4 x 132 + 21 = 549 min, so wave-fit must trim the fourth wave (it does).

**Per request:** 21.2k prompt tokens, 1,526 generated; 96.4 s queued, 2.0 s prefill, 23.3 s decode. Running 5.1 (cap `max_num_seqs 8`; 7.75 GiB BF16 KV = 163k tokens = 7.7 prompts), waiting 19.6, KV 86% full, flat across all four quarters (318/314/309/298 gen tok/s). **79% of every game's clock is queueing**; a game gets 0.49 calls/min, 64 calls per 132 min, median 153 s between actions.

**GPU steps** (176,124 in 7,948 s = 45 ms each): 34.3M prompt tokens in 2,048-token chunks = ~16.7k prefill steps; `request_prefill_time_sum` 3,250 s, so ~195 ms per chunk and **41% of GPU time is prefill**. The other ~159k steps decode at ~30 ms, 15.4 tokens each (5.1 seqs x 2.8; MTP-3 acceptance 1.79). Prefix caching is off and the Duck drops the oldest block on every over-budget call, so consecutive prompts share nothing past the system prompt: **the 41% re-prefills text prefilled one call earlier.** Net 310 gen tok/s = 1.1M tokens/hour from a 6B-active model, because the batch is 5 and 41% of the time is redundant.

**Where calls go.** Solved L1: median 26.6 min / 18 calls; solved L2+: 25 min / 11 calls; the level a game ends on eats 54% of minutes and 50% of calls. Score with every game stopped at T: 1.2 / 2.7 / 6.1 / 7.8 / 11.5 / 12.9 at 15/30/60/90/120/132 min; the last 12 min added 1.4 (0.12/min, the steepest segment, since late levels weigh more); 14 of 49 levels fell in the last hour. Son Pham: 264 min/game 25.1 vs 13.6-15.7 at 132. **Calls per game is binding and our curve is not yet concave.**

**Efficiency vs discovery.** Had every solved level scored 1.0, exp-054's mean would be 13.84 not 12.87: **solved-level inefficiency 0.98 points; unsolved levels 86.2** (exp-054r 1.56 vs 87.7; exp-050 1.9 vs 87.1; base 0.95 vs 91.2). Solved levels run at median 0.79x the human count (L1 0.81, L2+ 0.74). The squared metric bites only inside stuck levels (17 of 50 already exceed the baseline count).

**Allocation levers already measured, all near zero:** stopping no-level games at 60/90 min loses 7.6%/3.2% of score to free 17%/8% of time (43% of games with no level at 60 min solve one later); the P21 gate starved hard L1 (6/16 vs 16/24) and drew 4.23 vs 3.81; 27,000 s/game locally 13.3 -> LB 2.58 (another team). On a saturated server, moving time between games is zero-sum; **only more useful tokens is not.**

## 2. What separates 5 from 45 in this dimension

Tufa: 4.71 -> 11.04 -> 18.81 -> 27.29 -> 45.33 in four weeks, every run ~540 min, same card, same base. Forum attribution (participants, not Tufa): "squeezing more tokens out of the hardware and allocating them properly"; "correct vllm installation and nvidia/qwen3.8-flash-next-nvfp4"; "compaction rather than truncation". My decomposition:

1. **Tokens per hour, ~2-2.5x.** Batch 5 -> 24-28 (MTP off frees 7.5 GiB -> ~15 GiB KV = 14 prompts; FP8 KV doubles that; Thuitanium 722 tok/s at MTP-0/28 seqs vs 359 stock; Pennyroyal 632 at 8) plus ~90% prefix-cache hits (Son Pham) to remove the 41% prefill tax. At 28 running without caching, prefill alone would need ~6,000 of 7,920 s, so **more sequences and caching only work together.** Estimate: ~700 gen tok/s -> ~3,300 requests per 132-min run vs 1,609. By our own curve, 2x calls ~ +70-100% public-25 -> **+3-5 LB** at our 0.35-0.40 public->LB conversion (which rises as the agent strengthens).
2. **Context that keeps state.** Lesson 0022 (stripping history halved the score) and the levels-2+ study (9 of 48 stuck levels from lost previous-level facts, 5 from a goal that grew) show the memory is load-bearing; our window truncates it. Compaction (fold the old half into a digest at a fixed boundary) preserves it and makes the prefix cacheable. **+1-2 LB.**
3. **Model build / engine.** Our runtime is a vLLM dev fork: a custom QSA attention backend that cannot fuse MTP drafts ("rebuilding attention metadata between draft steps"), a separate-process PLE-offload worker on every step, RadixArk's quantization. NVIDIA's ModelOpt checkpoint on an official build may be faster per step and numerically better; unmeasured. **0 to +3.**
4. **The remainder (45 vs the ~12-15 above) is harness and model behaviour we cannot see.** Tufa's 27 -> 45 in two days at constant runtime is not a serving change.

## 3. Plan, 5 weeks at 30 GPU-h/week (~12 public-25 runs, or one 9-h run + 8)

Every arm is read against the exp-054 pair (z-sum +12.45, 95 levels, hard-L1 15/16, 1,609/1,627 requests). Daily LB slots: exp-054 draws 2-3 first (the selection rule needs 3), then the passing candidate.

| # | Week | Change | Cost | Expected | Pre-registered pass | Kill |
|---|---|---|---|---|---|---|
| 1 | Oct 3 | MTP-0 + 14 GiB KV stress, s16/s28 (built) | 1 h | 200-300 tok/s in the bench (vs 152.9) | >= 165 tok/s, 0 errors | below -> retry once at 12 GiB, then stop |
| 2 | Oct 3 | exp-059 pair: exp-054 patches on the MTP-0 profile | 5 h | requests/run 2,000-2,400 | requests >= 2,000 AND z-sum >= 9.45 AND hard-L1 >= 14/16 AND acting share >= 0.60 | requests < 1,850 (profile does not convert) |
| 3 | Oct 3-9 | exp-060: + prefix caching + P28 stable-prefix trim + lanes = running capacity | 5 h (pair) | hit rate 0.6-0.9; requests +25% over exp-059 | `prefix_cache_hits/queries` >= 0.6 AND requests >= 1.25x exp-059 | hit rate < 0.4 (trim still breaks prefixes: fix P28 before another run) |
| 4 | Oct 3-9 | Diag: NVIDIA NVFP4 checkpoint (Kaggle `xiaoz259/...`) on an official vLLM build; step time at 1/8/16/28 seqs, PLE placement | 1 h | unknown; xz claims it is the lead | starts offline; tok/s >= MTP-0 profile | no installable offline build by Oct 9 |
| 5 | Oct 10-16 | FP8 KV for QSA (reimplement the 4-file overlay idea) | 2 days CPU + 1 h stress | 28 seqs at 32k without preemption; +10-20% requests | starts, 0 errors, >= 1.6x KV tokens, pair z-sum within 3 of arm 2/3 | BF16 requirement not lifted in 2 days |
| 6 | Oct 10-16 | P30 compaction: replace truncation with a harness-built digest (note + P23 reports + win facts) at a fixed boundary | 5 h (pair) | levels 2+ over pair > 47; hit rate up | acting share >= 0.65, reply length <= +15%, preemptions flat, z-sum >= 9.45 | any gate fails (lesson 0022 failure mode) |
| 7 | Oct 10-16 | exp-062: best profile on the NVIDIA/official build (if 4 passed) | 5 h | 0 to +3 | z-sum >= 9.45 AND hard-L1 >= 14/16 AND tok/s >= arm 2 | either fails |
| 8 | Oct 17-23 | One 540-min run at the real shape: 110 games (25 x 4 + 10), 28 lanes, wave-fit | 9 h | all 4 waves finish; calls/game within 15% of the 132-min run | no OOM/preemption storm, no cut wave, cache holds at 28 lanes | a cut wave -> fix reserve before the next submission |
| 9 | Oct 17-30 | Repeats to n=2-3 on the two leading configs; daily LB draws; one 5-h contingency slot | ~25 h | LB spread measured | selection rule: mean of >= 3 draws | - |

Nov 1-2: select two by the fixed rule (highest mean over >= 3 draws; tie under 0.5 -> hard-L1, then validation; second slot diversified).

Not in the plan, by evidence: early stopping and per-game reallocation (zero-sum on a saturated server); effort < xhigh (exp-037 +35% calls, no levels); dropping history (exp-035); SGLang/Pennyroyal (text-only profile; images moved the LB for Le Grand and Tufa); 27B LoRA (wrong base).

## 4. Honest ceiling

Additive expectation: serving stack (arms 1-3, 5) +3-6 LB, compaction +0.5-2, model build 0 to +3, from a single draw of 4.70 with a same-code spread of 1.5-2.3x. **Most likely 9-13 on the public LB by Nov 2, ~15 if the NVIDIA build is a real step, private LB similar +/-2.** That is a top-10 to top-15 finish if the Duck family stays where it is. 20+ needs a published leader notebook or a harness change we have not found; 45 is the Duck's authors' private work over four weeks; 60-100 is not reachable by anyone on this hardware (the best hidden-set score with a frontier model and thousands of calls per game is 62.7). Compute explains about a 2-3x on our score, not a 10x.
