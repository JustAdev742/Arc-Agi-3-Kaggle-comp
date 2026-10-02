# What Tufa Labs and Yi-Chia Chen do beyond Franzen: evidence and hypotheses (2026-10-02)

One-line summary: neither leader has published anything usable since Sep 1. The leaderboard suggests Tufa made one
large step between Sep 27 and Sep 28 on top of a system already at Franzen's level, and that Yi-Chia improved every
day from Sep 27. The best-supported explanations are more decode capacity at long context, a post-trained Flash-Next
policy, and fewer tokens per action. Each has a cheap test on Franzen's base.

Research run 2026-10-02 22:35-23:50 UTC. Downloads are in the session scratchpad
`/tmp/claude-0/-home-user-Arc-Agi-3-Kaggle-comp/d342458e-03bd-545b-8a6d-06bca061963e/scratchpad/intel/`:
- 50 forum threads, as `threads/*.json` and `*.txt`
- the leaderboard-monitor JSON (`lbhist.json`)
- Tufa's Sep 1 bundle (`taaf-sep01/`)
- the Tufa repos and forks (`duck-harness/`, `website/`, `forks/`)
- other teams' notebooks (`k_*/`)

Commands (CLI = `.venv/bin/kaggle`, rate-limited, retried with back-off by `fetch.py`):
- `competitions topics list arc-prize-2026-arc-agi-3 -s new -p N`
- `competitions topics show arc-prize-2026-arc-agi-3 <id> --format json` (the text form truncates comments)
- `datasets list --user <u>`, `kernels list --user <u>`, `models list --owner <u>`, `models get <ref>`
- `kernels pull <ref> -m`, `competitions team-submissions <team_id>`
- `git ls-remote` and shallow clones of the GitHub repos
- the Hugging Face API, `curl https://tonghuikang--arc3-leaderboard-monitor-get-history.modal.run`, and WebSearch/WebFetch

"Verified" below means I read it this session in a source named next to it. "Inference" is my reading, and it is
labelled as such.

---

## 1. Leaderboard facts

### Verified

- **Submission limit: 1 per day.** Kaggle restored it on Jun 8 after a 13-day window of 5 per day (thread 705405,
  María Cruz). This settles the `DAILY_SUBMISSIONS` VERIFY item in CLAUDE.md. So every daily step below is one draw.
- **Daily best score and runtime in minutes** (monitor JSON pulled 2026-10-02 22:42 UTC). The runtime includes
  queueing (CPMP, thread 744365), and a cell is missing on days with no record.

| Day | Tufa Labs (153 subs) | Yi-Chia Chen (18 subs) | Daniel Franzen |
|---|---|---|---|
| 09-13 | 18.81 / 543 | 3.24 | 7.63 (09-15) |
| 09-19 | 18.81 | 12.88 / 503 | 13.12 |
| 09-21 | 18.81 | 15.98 / 494 | 13.12 |
| 09-24 | – | 18.63 / 544 | 16.68 |
| 09-26 | **27.29** / 540 | 18.80 / 298 | 21.01 |
| 09-27 | 27.29 / 544 | **28.34** / 539 | 21.01 |
| 09-28 | **45.33** / 595 | **36.73** / 539 | 26.55 |
| 09-29 | 45.33 / 546 | **40.80** / 549 | 27.24 |
| 09-30 | **50.65** / 538 | 40.97 / 533 | 27.89 (published 09-29 22:50 UTC) |
| 10-01 | **52.51** / 542 | **48.07** / 541 | – |
| 10-02 | 52.51 / 212 | – | – |

- **Tufa's longer history.** 1.21 (June) → 1.45 (07-18) → 1.62 (08-10) → 2.07–4.71 (08-19 to 08-30, after the
  Qwen3.8-27B release) → 11.04 (09-06, after the Flash-Next release on 08-24) → 18.81 (09-13) → flat for 13 days →
  27.29 → 45.33.
- **Current best submissions** (`team-submissions`). Tufa's 52.51 is submission 56751075, submitted
  2026-10-01 12:36 UTC. Yi-Chia's 48.07 is 56757745, submitted 10-01 19:19 UTC. Both came after Franzen's notebook
  (09-29 22:50 UTC) and after Tong Hui Kang's comparison table (10-01 00:36 UTC).
- **The noise of one notebook.** Son Pham measured single draws of Franzen copies at mean 25.77, SD 3.93, median 26.19
  (thread 745062, 10-02). My own count from the monitor:
  - 444 teams that were below 15 before 09-30 and are now at 15 or more average 26.6 (SD 3.1, maximum 34.77).
  - Single-draw teams reached 32.28 (Haraguchi-T, 1 sub) and 32.73 (Mahmoud Nasser, 2 subs).
- **A Kaggle outage on Oct 1–2** ended several runs early: Son Pham's ran 4 h and 2 h, rellik13's 2 h (thread 744995).
  Tufa's Oct 2 record of 212 min is probably one of these.

### Inference

- **Tufa's step happened between Sep 27 and Sep 28.** The Sep 27 draw did not beat 27.29. The run recorded on Sep 28
  scored 45.33, which is +66% in one submission.
  - The Sep 26 jump (18.81 → 27.29) brought Tufa level with a typical Franzen draw, so by Sep 26 Tufa had something
    roughly equal to Franzen's whole system.
  - The later draws (≤45.33, 50.65, 52.51) fit one configuration with **a true mean of about 46–50**, given CV 12–15%.
    52.51 is the best of about four draws.
- **To finish a few points ahead of Tufa on the private half, we need a mean of about 55 or more.** That is roughly 2.1
  times Franzen's mean, and Tufa has a month left to improve.
- **Yi-Chia beat their own best on five days running (Sep 27 – Oct 1)**, one draw a day. For draws from a single
  fixed configuration, the chance of that is about 1/120. So Yi-Chia almost certainly changed the system every day:
  new checkpoints or harness changes, not luck.
  - Their short runs before Sep 24 (93–300 min) look like failed or test runs, not a faster solver. They have only 18
    submissions in total.

---

## 2. Tufa Labs

### Verified: public artifacts. Nothing new since Sep 1–3.

- **GitHub, `Tufalabs/duck-harness`.** HEAD is still 7652836 (2026-07-01). The only other refs are four PR heads.
  - The repo has no LICENSE file. Issue #8 (Oct 1) asks for MIT-0 or CC0 and has no answer.
  - Jeroen Cottaar (thread 717133, Aug 20): the "code dataset [is] MIT license", the notebook Apache-2.0.
- **Kaggle datasets** (`datasets list --user`), all already known:
  - `jakobbrggen/taaf-kaggle-source`, last updated 2026-09-01 16:58 (CC0-1.0 dataset; code MIT)
  - `jakobbrggen/qwen3-8-27b-fp8-hf-snapshot` (08-15)
  - `jakobbrggen/taaf-kaggle-source-anim-20260807-anim` (08-07)
  - `driessmit1/qwen3-8-27b-fp8-hf-017b9c7a` (08-18; Dries Smit is on the Tufa team)
  - `jeroencottaar/taaf-kaggle-source(-share)` (June)
  - The wheelhouse `driessmit1/arc3-vllm-h100-wheelhouse-v3` that the Sep 1 bundle needs is not public. Tufa uses
    private datasets.
- **Kaggle notebooks.** Nothing ARC-related since Aug 15. `jakobbrggen/eda-base` (Sep 3) is a Playground S6E9
  notebook about electric-vehicle purchases, unrelated.
- **Kaggle models.** None for `jeroencottaar`, `jakobbrggen` or `driessmit1`. `tufalabs` is not a Kaggle user.
- **Hugging Face.** No models or datasets under tufalabs, Tufalabs, TufaLabs or the members' handles I tried.
- **Tufa website repo** (`Tufalabs/website`, latest commit 09-18). The only ARC-AGI-3 news is the Milestone 1 post
  (published 09-07), and the Duck post is unchanged since 07-01.
  - That post: "The harness can still benefit from many improvements, especially in context management and
    perception."
- **`Tufalabs/tufazip`** (updated 09-30, MIT) is an enwik9 compressor by Tommy He, run on 2× B200. It is unrelated to
  ARC, but it shows the lab's compute.
- **There is no newer source bundle, so no diff against Franzen or the Sep 1 bundle was possible.** The harness file
  sizes, for scale: Duck M1 9,740 lines of Python; Sep 1 bundle 11,774 (adds avo/, noop_guard, animation); Franzen
  17,932 (adds priority_scheduler, frame_diff, retained_functions, noop_repeat_guard).

### Verified: indirect signals

- **Tufa forked a full RL post-training stack in early September.** The forks under `github.com/Tufalabs` (shallow
  clones; every fork's `main` equals the upstream commit at fork time, so the work is in private repos or branches):

  | Fork | Upstream | Fork date (HEAD commit) |
  |---|---|---|
  | slime | THUDM | 09-03 |
  | miles | radixark (RL framework "for LLM and VLM post-training") | 09-09 |
  | verl | – | 09-09/10 |
  | sglang, vllm, flashinfer, transformer-engine, flash-attention, transformers | – | 09-09/10 |

  - This set is miles' dependency stack: Megatron and TransformerEngine for training, SGLang for rollouts.
  - Another competitor reports miles as the working route to RL on Flash-Next: AAAAAtjc, thread 742835 (09-24): "I
    successfully built and validated rl training pipeline of Qwen 3.8 Next Flash based miles, but I did not gain any
    performance in arc-agi-3 yet ... I used a full H200 node."
- **Tufa runs evaluations on a cluster.** The Sep 1 bundle's `configs/inference.json` has `deployment.slurm` with
  B200 × 2 per job, `n_passes: 20` and `concurrent_jobs: 32`. Their M1 example run is 20 passes × 25 games.
  - The `preamble.txt` path shows Jakob working from Claude Code worktrees.
- **Tufa's framework could serve LoRA adapters from June.** `inference/tools/vllm_runtime_lora_guard.py` handles vLLM
  runtime-LoRA warnings, with module aliases for the GDN layers (`in_proj_ba`). It is in both the M1 repo and the
  Sep 1 bundle.
- **The team's skills are RL and training** (`site/_data/team.yml`):
  - Dries Smit: "large-scale training and reinforcement learning"
  - Isaiah Pressman: "self-play reinforcement learning ... techniques needed to train large-scale deep learning
    models"
  - Stefano Viel: "unsupervised environment design, meta learning, synthetic data generation"
  - Jerome Sieber: "test-time RL"
- **Their own method paper fits interactive games.** Tufa's paper "A Predictive Law for On-Policy Self-Distillation
  From World Feedback" (He, Sieber, Saponati; arXiv 2605.30070, 2026-05-28, ICML RLxF) is about on-policy
  self-distillation that "uses arbitrary feedback as learning signal". Frame-by-frame game feedback is that kind of
  signal.
- **What Tufa has said on the forum:**
  - Jeroen Cottaar, 742801 (09-23): "We, Tufa Labs, will not be sharing our solution on September 30 ... We still
    plan to release our final solution when the competition ends." Also: "it is not so obvious that big improvements
    are still possible in the limited time that remains."
  - Jeroen, 742801 (09-28), asked whether they would open-source if first place did: "No, we'd just hope their
    solution is very different from ours, and that we can learn from it…"
  - Jeroen, 744792 (10-01), on Tong Hui Kang's table of the three winners: "Great overview ... Saves us all a bunch
    of tokens…"
  - Jakob Brüggen, 737617 (08-26), on other teams' jumps: suspects "a public new harness or idea", and points to
    Polyphony.
- **What others guess about Tufa** (speculation, not evidence):
  - Scott Le Grand (742801, 09-23): "My guess is part of that is some fine tuning". On 743952 (09-29): "throwing
    more HW at distilling solutions into a tractable model like flash-nvp4 ... poor person's RSI by running the
    solver multiple times and then distilling in the best behavorial solutions".
  - cm391 (743952): "correct vllm installation and using nvidia/qwen3.8-flash-next-nvfp4 ... compaction rather than
    truncation".
  - Penguin (742801): "10s of TB of gpu vram, allowing continuous testing".

### Inference

- **The step from 27 to 45 is not a plain serving fix.** By Sep 26 Tufa already scored like a Franzen draw, and
  Franzen's notebook already has FP8 KV, 128 Ki context, prefix-cache-friendly trims and a scheduler. So the +66% came
  from something else: more capacity than Franzen's 10 streams, a different policy (weights), fewer tokens per action,
  or several harness wins measured with their low-noise evaluation.
- **The fork dates fit a post-training pipeline.** Forks on Sep 3–9, a 13-day plateau from Sep 13 to 26, then two
  jumps: that is the shape of a training pipeline coming online. It is only consistent with the evidence, not shown by
  it.
- **If Tufa ships trained weights, they must later publish them** to be prize-eligible.

---

## 3. Yi-Chia Chen (Kaggle `threerabbits`, 18 submissions)

### Verified

- **Earlier work on the same GPU.** Notebook `threerabbits/submission-32b-fix4` (2026-06-25, AI Mathematical Olympiad
  Proof Pilot competition, RTX PRO 6000, offline; a Kaggle notebook, so Apache-2.0) has:
  - **Their own SGLang build**: triton attention tuned for sm120, KV fp8, hybrid sliding-window attention,
    48 running requests, 200k context.
  - **DFlash speculative decoding with a self-trained draft.** `threerabbits/dflash-32b-draft-v2test-phasel` and its
    `-int4mlp` variant: block 8, 8 draft tokens, the draft's MLP in int4.
  - **An on-policy-distilled target model**, `threerabbits/opd-32b-v33-s200-gptq-w4a16` (18.8 GB). The name reads as
    OPD, version 33, step 200, quantized to GPTQ W4A16.
  - **The environment shipped as a venv archive with warm kernel caches**, for a fast start. Their older notebooks
    include `launch-gpt-oss-120b-in-6mins` and `crystal_math_gpt_oss_20b_grpo`.
  - The two models I checked are dated 2026-06-24, licence "Other".
  - Nothing ARC-related is public under `threerabbits`: datasets, notebooks and models list only the items above.
- **Forum statements:**
  - 742935 (09-28): "Given the complexity of the process, I've decided not to open-source my solution for the second
    milestone."
  - 744792 (10-01): "I'm excited to see how high a score people can achieve by combining these three winning
    notebooks!"
  - 744545 (10-01): "I regret it so much 🫠"
- **Tong Hui Kang's view** (744545): "I thought your advantage has been a significantly better finetuned model,
  given your experience in dataset construction and finetuning. Maybe also did some inference engineering."

### Inference

- **Yi-Chia probably runs their own trained artifacts.** The skills shown in June (OPD-trained target, self-trained
  DFlash drafter, their own SGLang) and the daily improvements point to their own trained model, drafter, or both,
  and "the complexity of the process" reads like a training pipeline.
- **Their notebook is probably structurally different from the three winners'**, given the "combining these three"
  remark.

---

## 4. Other teams since Sep 25: what moved their scores

### The three tentative Milestone 2 winners

Source: Tong Hui Kang's comparison, thread 744792 (10-01).

| | dfranzen (27.89) | lordhansolo (23.84) | sirikilohit (22.53) |
|---|---|---|---|
| Quantization | Intel W4A16 + BF16 PLE | primitive-ai mixed NVFP4-FP8 | W4A16 + FP8 PLE |
| Speculative decoding | Albucino INT4 MTP draft, 3 draft tokens; FR-Spec 64k hot tokens | built-in MTP, 32k draft vocab | NVFP4 MTP draft |
| Server | Pennyroyal SGLang v2.5.3 | vLLM 0.29.1rc1 nightly | Pennyroyal v2.5.0 |
| KV pool | 1.01M tokens | 1.42M tokens | 1.00M + 48 GB host tier |
| Streams | 10 | 14 | 16 of 28 games live |
| Context | 128 Ki | 127 Ki | 69.6 Ki |
| Peak decode | 946 tok/s | 1,135 tok/s | 1,159 tok/s |
| Temperature | 0.7 | 0.6 | 0.6 |

- **Lord Han Solo** also uses a 3,918 s cap per game, a rewritten system prompt, and a strategy-audit prompt once a
  level has used 25% of the game's time.
- **sirikilohit** also uses a UCB scheduler (a 2,400 s slice per game, then extra slices to the games clearing the
  most levels per token) and pins each solved level's rule and last 30 winning actions in the system prompt.

### Per-team reports

- **sirikilohit (Kaggle; posts as rellik13, 744792, 10-01).** These describe the same change (his notebook's version
  notes and his forum post).
  - FP8 KV, spent on history (trims of 37k→27k became 57k→45k): "I guess this alone took the score from 14.5 to
    22.5."
  - "Some harness changes that helped were simply to not mislead the agent." His examples: "UNDO instead of
    ACTION7, 'level restarted' instead of 'game over', removing lines that tell the agent to ignore the budget bar."
  - His cleared levels used a median of about 0.8x the human baseline actions.
- **Son Pham & Mark Barney.** Leaderboard 8.23 (09-27) → 12.21 → 15.68 → 18.99 (09-30). Their repo `sonpham-org/arc-3`
  has no top-level LICENSE, so read it for ideas only.
  - **Report of 10-01** (`docs/trace-findings/2026-10-01-before-after-new-harness.md`), on their 132-minute,
    all-25-games cloud suite:
    - Median scores: old harness 25.3 (09-27/28) → "halfway" 39.7 (09-29) → new harness 44.2 → with Franzen's tool
      bundle 49.1 (one run).
    - The jump from 25 to 40 came with: retraining the MTP draft head for their pruned model, gameplay temperature
      0.6–0.8, and an "honest restart" that cuts off a stalled turn after 60 s.
    - Matched pair, one run each: the 60 s restart took 101 to 109 levels and 39.1 to 44.2.
  - **Pruning review of 09-27** (`2026-09-27-moe-expert-pruning-review.md`). Pruning the experts (keeping 384 or 336
    of 512 per layer) frees weight memory, so the KV pool grows from 736k to 1.97M tokens and 16 games run at ~103k
    context instead of 7.
    - Their public-25 score went from 21.4 to 33.9 (one run). With 7 lanes, the pruned model scored 19.7 against
      21.4 unpruned: "a memory lever, not a quality lever."
- **暗黑AGI** (LB 31.81) published `boristown/qwen3-8-flash-next-nvfp4-reap-448e`, NVFP4 pruned from 512 to 448
  experts, "~20 GiB lighter ... to grow KV for 128K serving". Licence: NVIDIA Open Model License + Qwen Community
  License. A NVFP4 REAP-k448 build also exists on Hugging Face (`lee-chang-93`). The sh0wie REAP-384 builds there are
  MLX and BF16.
- **xz** (LB 28.16, 5 submissions): ablated NVFP4 KV and "found very little NLL penalty". In a token-matched
  comparison, xz found his agent less action-efficient than the milestone solutions.
- **Tong Hui Kang** (LB 20.53, unchanged since 09-25) uploaded a LoRA fine-tune of Flash-Next,
  `huikang/qwen-38-flash-next-finetune` (checkpoint 09-29, 15 versions, Apache-2.0 on Kaggle; merged into the
  RadixArk NVFP4 base). There is no leaderboard gain from it yet.
- **Swift 1.5 Qwen3.8-Flash-Next** (UkisAI, released 09-22, gated): "63.4% fewer thinking tokens ... 1.8x speed up
  ... accuracy loss <1%". Teams were mirroring it:
  - `lordhansolo/swift-1-5-qwen3-8-flash-next-nvfp4` (09-28)
  - `cihanatak/swift15-flashnext-nvfp4-nonple` (09-28)
  - `michaelpoluektov/qwen3-8-swift-nvfp4` (09-29)

  None of them has a leaderboard result attributed to it. Licence: Swift Open License v1.0 + Qwen Community License
  1.0, so prize eligibility is unclear.
- **Fine-tuning reports** (thread 742835):
  - blakewest: supervised fine-tuning on H200s with Qwen Max or Astra as teacher, on synthetic variants of ls20 and
    cd82, at stall points; "No permutation of these has been able to make meaningful improvements ... finetuning
    with Qwen Flash Next is very difficult because it's an MoE".
  - Scott Le Grand: behaviour cloning on Qwen3.6-27B gave "+2 on the public games but it didn't translate [to] the
    hidden data".
  - So no public team has a leaderboard gain from fine-tuning Flash-Next.
- **Public-25 vs the leaderboard:**
  - skarin, notebook `arc-agi-3-27-80-lb-why-one-run-lies`: Franzen's 10-game demo scored 47.14 for skarin and 36.56
    for the author, and the same kernel drew 27.80 on the leaderboard.
  - Son Pham: public-25 of 40–49 against a leaderboard of 18.99.
  - Nhan Duc Nguyen: "+10 and more points diff between 2 runs of the same version" on the public games.
- **Licences of the models and harnesses** (thread 745079, 10-02, unanswered): it asks whether Flash-Next (Qwen
  Community License 1.0) and NVIDIA's NVFP4 build (NVIDIA Open Model License) can be used in a prize-eligible
  solution. Both licences are non-OSI. Kaggle notebooks are Apache-2.0 (Jeroen: "the only one possible on
  Kaggle").

---

## 5. Web: ARC Prize and papers

- **ARC Prize.** I found no Milestone 2 blog post. X posts return HTTP 402, so I could not read ARC Prize's own
  announcements. The competition page lists the Milestone 2 prizes as $25K, $10K and $2.5K.
- **No public small-model ARC-AGI-3 work** from Sep–Oct 2026 reaches high scores. The high scores I found all use
  frontier models:
  - "Twin" (arXiv 2608.14490): 93.3% public with a frontier coding agent.
  - "Schema Harness" (Hacker News 48935905): about 99% public with Opus 4.8, Fable 5 and GPT-5.6 Sol.
  - Tycho (2607.28287) and the AVO, Retrodict and Polyphony results were already in our notes.
- **Fast drafters for Flash-Next exist but help little on generic data.** PixelML's DFlash drafter (09-08) was
  trained on 98.5k generic chats with thinking off: "beats the target's own tuned speculative head by 3.87%".
  - DFlash2 drafters exist for Qwen3.8-27B (z-lab, 08-15), and Pennyroyal has a `qwen38-dflash2-pro6000-20260824`
    tag.
- **No new model that fits the card** is stronger than Flash-Next. Every model above it is too large: GLM-5.3-Flash
  (321B), DeepSeek-V4.1-Flash (763B), MiMo-V2.6-Pro-RL (1T). The serving parts teams rely on were all out by Sep 6
  (RadixArk 08-25, Intel W4A16 08-28, Albucino 08-30, NVIDIA NVFP4 09-02). Pennyroyal v2.5.3 came on 09-27.
  - So the Sep 26–28 jumps were not set off by a new public model, apart from Swift 1.5 on 09-22.

---

## 6. Ranked hypotheses: what Tufa does beyond Franzen

The ranking weighs how likely each one is to be part of Tufa's edge against how much it could add. "Gain" is my
estimate for our Franzen base.

**1. More decode capacity at long context: 14–20 live streams at ≥100k instead of 10.** This would come from freeing
VRAM (expert pruning calibrated on ARC traffic, or a lower-bit KV or weights) and spending it on KV and Mamba state.
Likely; gain ×1.2–1.5.
- **For:**
  - Son Pham's one run: 7 lanes → 16 lanes (×1.58, 21.4 → 33.9) from pruning alone.
  - sirikilohit's +55% from KV memory spent on history.
  - Lord Han Solo runs 14 streams at 127k on a 1.42M pool. 暗黑AGI and Son prune to the same end.
  - Franzen's pool is 1.01M tokens for 10 streams.
  - Tufa has the engineers and the cluster to calibrate pruning on their own traces. Their Sep 9 forks include
    sglang and flashinfer.
- **Against:** Lord Han Solo has 14 streams and scores below Franzen, so streams alone are not enough.
- **Cheap test:**
  - (a) In exp-070's server log, check whether queueing or the Mamba-slot or KV pool limits the 10 streams.
  - (b) Teacher-forced NLL of the REAP-448 NVFP4 checkpoint against the base on 20 of our Flash-Next transcripts.
    About 1 GPU-hour on the local box.
  - (c) Franzen's notebook with REAP-448, MAXREQ 14 and a matching Mamba cache, against unchanged, ×2 public-25 runs
    each at 121 min per game.

**2. A post-trained Flash-Next policy: RL or on-policy self-distillation on ARC-like games, possibly synthetic
ones.** Plausible for Tufa; for Yi-Chia, likely for their own system. Gain unknown; it could be the whole gap.
- **For:**
  - Tufa forked slime on Sep 3, then miles, verl, TE and flash-attn on Sep 9 (miles is the known route for RL on
    Flash-Next).
  - Team bios: RL, self-play, environment design, synthetic data.
  - Their May paper on self-distillation from world feedback.
  - A B200 Slurm cluster and LoRA serving in their framework.
  - A 13-day plateau, then a +66% step that no public serving change explains.
  - Jeroen's M1 claim that gains came "from stronger base models".
- **Against:**
  - No other team has a leaderboard gain from training Flash-Next (blakewest, AAAAAtjc, Scott Le Grand, Tong Hui Kang
    so far).
  - It is costly, and no weights are visible.
- **Cheap test:**
  - (a) Label why levels failed in Franzen-run transcripts: stalls, a wrong goal defended, ignoring the budget bar,
    tool misuse, against capacity. If behaviour dominates, training has room.
  - (b) Run Tong Hui Kang's public LoRA fine-tune (Apache-2.0 on Kaggle) through Franzen's harness once, as a
    sensitivity probe.
  - (c) Real training needs a rented H200 node, which means money. That is the owner's call.

**3. Fewer tokens per action, so more decisions per hour.** This could come from a reasoning-efficient model (Swift
1.5, released 09-22, "63% fewer thinking tokens"), a tighter turn budget, or a stall cut-off. Plausible; gain
×1.1–1.4.
- **For:**
  - The timing: 4–6 days before both leaders jumped.
  - Three teams mirrored Swift on 09-28/29.
  - Son Pham's 60 s honest restart: +5 points public-25, one matched pair.
  - Franzen yields at 2,048 generated tokens per turn, but his reasoning length per turn is otherwise untouched.
- **Against:** No leaderboard result is attributed to Swift. Swift's licence is custom and gated.
- **Cheap test:** On Franzen's base, run two arms against unchanged, ×2 runs each, reading thinking tokens per action
  and levels per hour:
  - Swift-1.5 W4A16-AutoRound (`ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AutoRound`).
  - A 1,024-token yield.

  Check the licence before any submission.

**4. A speculative drafter trained on ARC harness traffic: an MTP head or DFlash, and an FR-Spec hot-token map
built from our own traces.** Plausible, and very likely part of Yi-Chia's system. Gain ×1.1–1.3 on decode.
- **For:**
  - Yi-Chia trained DFlash drafters in June.
  - Son Pham's retrained draft head was part of his 25 → 40 step.
  - Generic-data drafters help only 3.9% (PixelML), but harness output (Python, tool calls, grid talk) is narrow
    and repetitive.
  - Franzen's FR-Spec map is the generic `flash-next-64k.pt` from Pennyroyal.
- **Cheap test:**
  - (a) Read the acceptance length from exp-070's SGLang log.
  - (b) Build a 64k hot-token map from the token frequencies in our own Flash-Next transcripts and compare acceptance
    and decode tok/s on a replayed request set. About 1 GPU-hour, no gameplay.
  - (c) If acceptance is below about 2.3 of 3, fine-tune the Albucino draft on dumped hidden states (1–2 days).

**5. Harness behaviour fixes beyond Franzen.** Plausible; gain ×1.05–1.15.
- **What they would be:**
  - Pinned rules for solved levels plus the winning actions (sirikilohit).
  - The budget bar treated as a hard budget, labelled goal and action notes, test discipline (Son Pham).
  - A strategy audit at 25% of the game's time (Lord Han Solo).
  - Removing misleading prompt text (rellik13).
- **For:** Son Pham's +4.5 public-25 from his prompt fixes. Tufa's own M1 note names "context management and
  perception" as the open areas.
- **Cheap test:** Port the four fixes onto Franzen's base as one arm and run it ×2 against unchanged. Keep each part
  only if a per-game paired comparison favours it.

**6. A low-noise evaluation process: 20 passes × 25 games on a B200 cluster, possibly with synthetic held-out games.**
This lets them stack many +5% changes that a team with one run per arm cannot see. Likely; it is not a mechanism, but
it multiplies the others.
- **For:** The Sep 1 bundle config, and their M1 run of 20 passes × 25 games with 29.6M tokens.
- **Cheap test (adopt it ourselves):**
  - Use the rental runner (about $4–5 per public-25 run) to get 3 or more runs per arm.
  - Compare arms with skarin's paired per-game bootstrap. Check the notebook's licence first; Kaggle notebooks are
    Apache-2.0.

**7. Better allocation across games than Franzen's priority gate.** Less likely to be a large part; gain ×1.0–1.05.
- **For:** sirikilohit's UCB slices; Tufa's early wave-fit work.
- **Against:**
  - Scott Le Grand: dropping level-0 games "no dice".
  - Our own allocation levers measured near zero on a saturated server.
- **Test:** Only after 1–3 are measured.

---

## 7. What this means for us (inference)

- **Gaps 1, 3, 4 and 5 can be tested on one card**, on Franzen's base. If they compound to about ×1.8–2.1, we land
  near Tufa's current mean, not above it.
- **Hypothesis 2 is the one that could make Tufa unreachable** for a team without training compute.
- **Watch for new public artifacts:** Tufa (any Kaggle model or dataset under jakobbrggen, driessmit1 or
  jeroencottaar), Yi-Chia (`threerabbits` models), and new Flash-Next post-trains on Hugging Face. A public
  ARC-trained checkpoint would change the plan overnight.
- **Licences of the artifacts named here:**

  | Artifact | Licence | How we may use it |
  |---|---|---|
  | Franzen (repo and notebook) | Apache-2.0 | Base |
  | Tufa code | MIT, per Jeroen (no LICENSE file in the repo) | – |
  | Tufa Sep 1 dataset | CC0 | – |
  | Kaggle notebooks (lordhansolo, sirikilohit, skarin) | Apache-2.0 | Free to read and reuse |
  | Son Pham repo | No licence | Ideas only |
  | Flash-Next | Qwen Community License 1.0 | Not OSI; prize use open (thread 745079) |
  | NVIDIA NVFP4 and REAP-448 builds | NVIDIA Open Model License + Qwen Community | – |
  | Swift 1.5 | Swift Open License v1.0 + Qwen Community; gated | Check before any submission |
  | Tong Hui Kang's fine-tune | Apache-2.0 on Kaggle; base licence included | – |
  | Yi-Chia's models | "Other" | – |
