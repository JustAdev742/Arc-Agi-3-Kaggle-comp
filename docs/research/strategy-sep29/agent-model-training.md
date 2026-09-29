# The agent, the model and training: what separates our 4.7 from 45, and the 5-week plan (2026-09-29)

Sources: the repo docs named in the brief, the 25 exp-054 transcripts and server metrics, the leaderboard monitor
JSON (pulled today), arXiv 2605.25931, a web check on Qwen releases. No GPU, no Kaggle, nothing committed.

## 0. Outcome first

Our agent does not lose on efficiency; it loses on reach. Solved levels cost about human-level actions (median 0.79x
baseline, 75% at or under 1.0x, exp-054); the score is lost to levels never reached because each game gets only ~42
decisions (1,059 turns / 25 games, 66% acting) in 132 minutes, 96 s of every 122 s call spent queued behind a
7.75 GiB KV cache filled by 21k-token prompts. Other teams' 5 -> 20 is serving, context compaction and
allocation on the same Duck + Flash-Next design; Tufa's 27 -> 45 in two days is unexplained (a 595-minute run, longer
than any other top run). A realistic target for us by Nov 2 is 8-15 on the LB without paid compute; 45 is not reachable
on any evidence we have.

## 1. How the agent actually loses (exp-054 pair, transcripts read)

- Reach, not efficiency: 49/183 levels; on the level a game got stuck 54% of minutes and 50% of calls were spent.
  Score vs time is near-linear where it matters (stopped at 60/90/120/132 min: 6.1/7.8/11.5/12.9), so more decisions
  per game convert into levels.
- Stuck-level modes (48 L2+ levels): mechanic never found/decoded 15, out of time while progressing 11, rule misread 7,
  started too late 6, goal grew and old recipe reused 5, previous-level knowledge lost 4. Hard level 1s: goal misread
  11/31, key action's effect misunderstood 9/31 (g50t's SPACE rewind, tn36's mid-animation move).
- The model (Flash-Next, 6B active) reasons competently when the mechanic is decodable in code (tr87: 5 levels via
  cycle detection and BFS) and is sloppy in execution: 55 of 1,059 turns (5%) ended in a tool error (g50t lost a
  20-minute turn sending 'D' for 'DOWN'); it re-diffs frames by hand in ~24% of hard-L1 calls; it defends an early
  wrong goal (sb26 misread the harness's exact report for 60 minutes).
- Server accounting: 1,609 requests of 21.2k prompt / 1.5k generated tokens, prefix cache off (0 queries), 5.1
  requests running on average, and MTP-3 caps scheduled tokens at 2,048 (the log warns "suboptimal"). Decode 23 s vs
  prefill 2 s per request: we are concurrency-bound by KV memory, not compute-bound.

## 2. What separates 5 from 45 (estimate, evidence, confidence)

1. **Tokens per hour and their allocation (5 -> ~10). High confidence.** Scott Le Grand: 16 sequences alone took the
   stock Duck to the top-10% edge; Thuitanium's bench: MTP off + 7 GiB KV + 28 seqs = 722 tok/s vs 359 stock; Son Pham:
   89% prefix-cache hits with 50%-block trimming, 7.36 -> 12.21 after SGLang serving work; xz (Sep 28): "correct vllm
   installation and nvidia/qwen3.8-flash-next-nvfp4". Our own draws follow the same axis (base 2.72 -> +KV 4.23 ->
   +KV 4.70). We serve the RadixArk quant, MTP-3, no prefix cache: exactly the un-tuned profile.
2. **Compaction, not truncation (10 -> ~15-20). Medium confidence.** xz names it; Tufa's write-up admits poor prefix
   caching; Lord Han Solo, Franzen and Tong Hui Kang climbed 1-4 points per step over 75-87 submissions each (iterated
   engineering, no leak per CPMP). Our exp-035 halved the score by silently stripping reasoning (lesson 0022); a
   model-written state summary at the context limit is untried and shortens prompts without losing working state.
3. **20 -> 45: unexplained. Low confidence on any single cause.** Tufa (27.29 @540 min Sep 26 -> 45.33 @595 min Sep 28)
   and Yi-Chia Chen (18.8 -> 28.3 -> 36.7 in three full-length runs, 15 submissions total) moved together on Sep 27-28,
   which hints at a shared ingredient (vLLM/SGLang build, NVIDIA checkpoint) rather than private training, though Tufa
   has GPUs and credited stronger base models for its M1 gains. No open model release explains it: Qwen3.8-Max's open
   weights are 2.4T/A95B text-only. 45 on hidden games exceeds Claude Opus 5's 30.2 on ARC's semi-private set, so it is
   a strong harness x model interaction, training, or a rerun condition we do not share (595 min: verify that the
   9-hour limit is what binds).
4. **Training: feasible but weak expected value for us.** Evidence: STaR LoRA on the 27B's own wins 1.25 -> 1.94 LB
   (n=1); "naively scaling the data hurt"; U4AR's GRPO collapsed; Tong Hui Kang's action-prediction models were no
   better than random. Our corpus is 924 solved-level transcripts over ~95 distinct (game, level) pairs, dominated by
   games we already solve, so SFT mostly teaches what we can do. Flash-Next cannot be tuned on one 96 GB card; LoRA on
   8xH100 (~$25/h) trains for ~$300, but our vLLM cannot serve a LoRA on NVFP4 MoE experts, so it means merge +
   requantize (calibration risk) plus ~$400-800 of rollouts and evals. Frontier-distilled data needs the Rules tab's
   external-data clause and the providers' terms checked (unrecorded in docs/status.md; VERIFY first).
5. Hidden vs public: ARC says the private set is out of distribution and every Duck fork drops 3-4x, but our LB order
   matched our public order (2.72 < 3.81 < 4.23 < 4.70) and hard-game level-1 solves predicted the LB best. Build for
   a level-1-heavy hidden set; judge arms on hard-L1 and throughput, not the public mean.

## 3. Plan, ranked (5 weeks: ~150 GPU-h = ~50 public-25 runs; ~34 LB draws)

Each item: expected LB gain, cost, pre-registered test, kill rule. Public-25 runs are always pairs (single-run
difference sd ~2.3).

1. **Serving profile: MTP off, 14 GiB KV, 16 sequences, NVIDIA NVFP4 checkpoint, chunk cap lifted** (exp-059 plus a
   checkpoint-swap arm). Gain +2-4 LB. Cost 2 stress tests + 2 pairs (11 GPU-h), 1 day. Test: >= 8% more generated
   tok/s (>= 165 vs 152.9); requests per run >= 1,900 (vs 1,609) with acting share >= 0.65; hard L1 >= 14/16. Kill:
   fewer requests than exp-054, or z-sum below its pair by > 3.
2. **Model-written compaction (replaces P28's trim).** At 70% of budget, one no-action consolidation turn writes world/
   action/goal model, open questions, plan and helper names; keep the last 6 turns verbatim and P6's persisted
   helpers; drop the rest; prefix caching on. Gain +3-6 LB (prompts ~10k -> ~2x concurrency and cache hits). Cost 3
   engineering days, 2 pairs (10 GPU-h). Test: prompt tokens/request <= 12k, cache hit rate >= 60%, requests >= 2,500,
   acting share >= 0.65, reply length within +15% of exp-054, z-sum not below exp-054's pair. Kill: the exp-035
   signature (replies +30%, acting share < 0.55) in the first run.
3. **Execution hygiene for a 6B-active model** (from the transcripts): validate action names before the sandbox sends
   them; a one-line "untried actions on this level" after 8 idle calls. Gain +0.5-1. Cost 1 day, 1 pair. Test:
   tool-error turns <= 2% (from 5%), acting share up; kill if z-sum falls.
4. **Allocation on a level-1-heavy hidden set.** Give every game a 60-minute floor, then move lanes from games with no
   board change in 40 minutes to games whose level count rose in the last 30. Gain +1-2. Cost 1 day, 1 pair; must be
   read on the validation split and hard L1 (P21 starved L1: 6/16 vs 16/24). Kill: hard L1 below 14/16.
5. **Text-only MoE A/B (gpt-oss-120b MXFP4, ~63 GB weights, 30 GB KV)** on the same harness minus the image. Gain 0 to
   +5, bimodal. Cost 15-min serving check + 1 pair (5 GPU-h). Test: aggregate tok/s >= 2x Flash-Next and hard L1
   >= 12/16. Kill at the serving check if the SM120 MXFP4 path needs FlashInfer JIT offline.
6. **Fine-tuning, conditional (only with owner approval and after 1-2 land):** curate <= 300 trajectories from
   distinct (game, level) pairs solved at <= 1.0x baseline, weight tool tokens 1.0 / reasoning 0.3 (forum 743319),
   LoRA the 27B locally or Flash-Next on 8xH100 (~$1.5-3k all-in including rollouts). Gain +0-4 with ~40% chance of
   none. Test: it must beat untuned Flash-Next on hard L1 and validation in a pair; kill after one pair.
7. Not now: a procedural game generator (only as data for 6; 1-2 weeks, unknown transfer); human replays (no
   reasoning traces, download blocked); AVO/supervisor arms and higher-res images (ablated at or below baseline).

LB slots: alternate the best two configurations so each has >= 3 draws by Oct 25; final two by mean over draws (rule
already pre-registered in status.md). Submission requires the owner (the permission check blocks the script).

## 4. Honest ceiling

- Without paid compute: 8-15 on the public LB (median ~11) from items 1-4; top-15 on today's board, which will move
  up. The public 25 would read 20-30 for the same configuration.
- With ~$2-4k of rented training compute: 12-20 if the SFT transfers, unchanged if not (~40% odds of no transfer).
  Neither path reaches 36-45; 60-100 is not available to any open model on one GPU in 9 h.
- Decisions needed from the owner: (a) allow the daily submission command or submit by hand; (b) yes/no on the paid
  fine-tuning route and its budget; (c) read the Rules tab's external-data clause before any frontier-distilled data.
