# Hidden-LB evidence from four public repos (read 2026-09-27)

**Bottom line.** None of the four repos has a controlled, replicated measurement of a harness change that raised the hidden LB. They do show two things:
- **Same-code variance is large.**
  - Son Pham: one kernel version scored 5.54 and 7.36. Their notes say "same-code spread on Kaggle is ±0.9".
  - Hisernberg: the stock base scored mean 3.29, sd 0.51 over 12 submissions.
- **The big LB jumps track the model, serving and time allocation.** Prompt and harness tweaks do not stand out from that noise.

## 1. sonpham-org/arc-3 (Son Pham & Mark Barney; rank 14, LB 7.36 on 21-Sep)
**License:** there is no top-level LICENSE file. The README says the harness is a fork of Tufa's MIT-licensed Duck.

**Hidden-LB numbers.** The kernel map is on branch `docs/deepseek-v41-ceiling-run`, `HARNESS-NOTES.md` ("verified 21-Sep via API"). The other numbers are in `CONVERSATION_LOG.md` on branch `research/flashnext-harness-20260911`.

| LB | Config |
|---|---|
| 2.03 | Sub 55551321: Qwen3.8-27B-FP8, xhigh, TAAF + 8-action checkpoint, concurrency 28, ctx 65536/32768 (`harnesses/taaf-kaggle203-nocap-control/MANIFEST.md`) |
| 4.84 | Sub 55991804; config not identified in the files |
| 4.28, 5.66 | The same "MTR" notebook v347757078, submitted twice unchanged |
| 5.66 | W7 symbolic+MTR |
| 5.44 | W7 transition-sentence removal + CPU PLE prefetch |
| 5.02, 5.94, 6.49 | One version of `search-scorer-swap50-ready` |
| 4.71 | `no-cap-swap50` |
| 5.54, 7.36 | One version (351076929) of `clean-return-swap50`: `ARC3_ACTION_CAP=14` mode `return`, search/scorer prompt lines removed, 50% context swap |

"MTR" is probably metadata+toolkit+reminder. That is my inference from the kernel name.

**Which changes moved the LB, and which did not transfer.**
- Their own verdict: "a single submission does not separate candidates", and the offline-vs-online gap "is the model, not the harness arm".
- The 7.36 code scores 13.6–16.0 on the public 25. The streamer-prompt study (`docs/trace-findings/2026-09-25-streamer-prompt-trace-study.md`) attributes all of the "7.36 vs 15" gap to the change of game set.
- Everything else is public-25 only (GCP, never submitted):
  - Loop-A prompt deletions: 18.0 and 19.2, against controls of about 14.
  - Lowering thinking effort always lost: medium about −6, high −1 to −8.
  - 264 min per game: 25.07, against 13.6–15.7 at 132 min.
  - Lane sweep (264 min, n=1 each): 7 lanes × 103k ctx 26.1 > 11 × 65k 23.5 > 5 × 144k 18.7.

**Serving** (`ACTIVE_RUNS.json`, `research/astra-audit-20260905/context-memory-options.md`):
- `RadixArk/Qwen3.8-Flash-Next-NVFP4`, revision 7b71922, gpu-util 0.965, KV `fp8_e4m3`.
- 7 lanes, 102,985 ctx and 94,281 input tokens per lane.
- Sampling: T 1.0, top-p 0.95, top-k 20, thinking on, 4× image.
- An older log: BF16 KV of 13.74 GiB = 476k tokens, max_num_seqs 22, max_model_len 32,768.
- Throughput about 291 tok/s aggregate (about 46 per lane). "We cache 89% of prefill."
- The Kaggle bundle uses a BF16 CPU PLE; GCP uses native FP8.
- The MTP setting is not stated.

**Time.**
- 110 games on 7 lanes is 16 waves. The code reserves 10 min, divides the rest by 16, and caps at 1938 s per game. The 7.36 run played 1938 s per game.
- Kaggle setup took 21–41 min, mostly the CPU PLE load. PLE prefetch cut it to 22m40.
- The log notes that stock Duck (28 lanes) gets 4 waves at 132 min.

**Why the public 25 mislead** (`docs/how-this-feeds-kaggle.md`, `HARNESS-NOTES.md`):
- ARC says the public set "deliberately [does] not represent the mechanics of the private set".
- One game (sb26) swings the all-25 mean by ±23.

## 2. sonpham-org/arc-agi-3
- No LICENSE file.
- A Flask web play and arena platform.
- No Kaggle, vLLM or LB content on either branch. **No LB numbers.**

## 3. Hisernberg/arc-agi-3 (best LB 3.85)
**License:** no LICENSE file. PRs #1–#4 are merged into `main`.

**Own LB** (`plans/SUBMISSION_LOG.md`): 17 submissions between 1.79 and 3.85.
- The best (3.85) is an AGENTFIX Duck fork on Flash-Next.
- Their own additions scored lower: governor 2.77, nooa-lite 2.61.
- `plans/MISTAKES.md` #2: memory was dead in every production run because all games shared one "unknown" state.

**KV-bound analysis** (`analysis/01_run_2026-09-22_taaf-flashnext.md`, `research/10_kaggle_notebooks.md` §5.3):
- MTP3 weights take 81.8 GiB, leaving 5 GiB of KV = 105k tokens.
- That allows about 3 running requests, with 21.6 waiting.
- 87% of each request's latency is queueing; 342 preemptions; about 42 LLM calls per game in 2 h; 236 tok/s aggregate; prefix cache off.
- MTP off: weights 74.3 GiB, and the same 5 GiB holds 188k tokens.
- Thuitanium's replay bench (no games) measured 722 tok/s for MTP0 + KV 7 GiB + 28 seqs, against 359 stock.
- `serving/README.md`: the keithtyser runtime raises "QSA requires a BF16 main KV cache" for FP8 KV. **This conflicts with Son's `fp8_e4m3`.**
- Their "ctx16k" setting never took effect: setup re-persisted 32768.

**Time.** Stock Duck runs 28 slots at 7,920 s per game, so 4 waves need about 31.7k s. Only about 28k s is available, so the last wave is cut.

**Other teams' LB deltas** (relayed from forum posts and notebooks, unverified; `research/10`, `research/11`):
- Local-vs-LB correlation across 41 notebooks: r = 0.16.
- LB median / best by model: Qwen3.6-27B 0.92 / 1.61, Qwen3.8-27B 1.56 / 3.11, Flash-Next 3.21 / 5.19.
- Scott Le Grand (5.19):
  - Exposing animation frames in the sandbox was "the only change that has ever moved the hidden leaderboard (3.20 → 3.71)".
  - Text narration of those frames scored 2.57.
  - "Increase the sequence concurrency to 16 … brink of the top 10%. All of my harness engineering beyond that has been utterly useless."
- gedouluhui (27B): scoring-formula and time_remaining prompt lines 1.12 → 1.43. 64k ctx 1.20, T 0.3 0.95.
- LoRA on the model's own wins: 1.25 → 1.94.
- Removing the soft deadline: 1.33 vs 1.66.
- Single samples: wave-fit 3.64, MTP off 3.49, prefix cache 4.08, analyzer_timeout 1200 mean 3.80 (n=3).
- Did not transfer:
  - 27,000 s per game: local 13.31, LB 2.58, because it starves the hidden games.
  - Giving extra time to games already at L2+ helped the public set only.
- Local → LB pairs: 17.3 → 5.2/6.9, 22.3 → 5.4, 15.7 → 7.4, 11.0 → 2.7.
- Top teams run about 540 min; Yi-Chia Chen runs 90–500. Tufa went 4.71 → 11.04 → 18.81 → 27.29 (08-30 to 09-26).
- A forum comment on how Tufa got there: "squeezing more tokens out of the hardware and allocating them properly".

## 4. leejianrong/solve-arc-agi-3
- **License:** Apache-2.0. It is a clean-room re-implementation of Duck.
- **No LB numbers, and no submission is recorded.**
- Stack: Qwen3.6-27B-FP8, vLLM 0.19, max_model_len 65,536.
- Concurrency sweep on an **A100**, not the RTX PRO 6000, with a one-step fixture game: 30 / 108 / 140 / 354 tok/s at concurrency 1 / 4 / 8 / 16.
- `docs/PLAN.md` envelope: 8h30 total = 15 min startup, 7h20 of games, 25 min output, 30 min reserve.
