# Intel refresh: what is new since Oct 2 (2026-10-08)

One-line summary: the leaders published nothing new. On the public 25 games, other teams' 53-57 maps to an LB score of 30-34,
so the leaders' 55.9 points to a much better policy, not better serving. Artificial Agency Lab (#6, 38.62) released a
runtime that runs 76 streams with heavy compression. No one has published a working Swift-1.5 setup with relaxed MTP
acceptance, and UkisAI's W4A16 build stores the linear-attention projections in INT4, which is the likeliest cause of our
repeat loops.

Research run 2026-10-08 16:25-19:10 UTC. Downloads are in the session scratchpad
`/tmp/claude-0/-home-user-Arc-Agi-3-Kaggle-comp/d342458e-03bd-545b-8a6d-06bca061963e/scratchpad/intel-oct8/data/`:
- forum threads: `threads/`
- leaderboard CSV: `lbcsv/`
- monitor JSON: `lbhist/`
- user and search listings: `users/`, `search/`
- Hugging Face cards and discussions: `hfcards/`, `hfdisc/`
- competition pages: `pages/now/`
- the Artificial Agency Lab runtime: `dl_csaky_p50/`, with the files I read in `csaky_files/`

"Verified" means I read it in this session, in the source named next to it. "Inference" is my reading. X/Twitter was
unreadable: x.com returns HTTP 402 to WebFetch and a JavaScript shell to curl, so the only X content below comes from
search-engine snippets.

---

## 1. Tufa Labs, Yi-Chia Chen, Majkel1337 and the 37-43 band

### 1.1 Leaderboard (verified)

Sources:
- Kaggle leaderboard CSV, downloaded through the CLI at 2026-10-08 16:33 UTC (it includes team members).
- `competitions team-submissions <team>` for each team's best submission.
- The monitor history at `tonghuikang--arc3-leaderboard-monitor-get-history.modal.run`, pulled 16:35 UTC. It gives the
  daily best score.

| Team (Kaggle members) | Subs | Best (submission time, UTC) | Daily best since Oct 1 |
|---|---|---|---|
| Tufa Labs (dlorah, driessmit1, infinitecreativity, jeroencottaar, pressman1, stefano1283) | 157 | **55.89** (56792353, Oct 3 07:17) | 52.51 → 55.89 (10-03); no improvement on 10-05, 10-06, 10-07 |
| Yi-Chia Chen (threerabbits) | 24 | **55.77** (56920961, Oct 7 20:11) | 48.07 → 48.59 (10-03); flat; → 55.77 (10-07) |
| Majkel1337 (majkel1337) | 11 | **42.66** (56888365, Oct 6 18:32) | 9.07 (09-30) → 28.69 → 29.61 → 32.97 → 34.59 → 42.66 (10-06) |
| the last dance (dwellement0baser, fses91, gklambauer, lukasaichberger) | 75 | 39.30 (Oct 4 19:30) | 24.54 → 27.91 → 32.82 → 39.30 (10-04) |
| dreach.ai (denialguo, jaydenszeto123, snoopydoo); this team was "gng" on Oct 2 | 32 | 39.11 (Oct 6 08:30) | 34.77 → 36.78 → 39.11 |
| artificialagencylab.com (cmechevalier, richardcsaky) | 47 | 38.62 (Oct 7 22:01) | 20.17 (10-02) → 28.12 (10-03) → 37.94 (10-04) → 38.62 |
| mtg (michaeltgao) | 63 | 38.33 (Oct 5 06:11) | 22.37 → 30.64 → 37.54 → 38.33 |
| 復活の混テキスト (six members) | 75 | 38.15 (Oct 7 15:41) | 27.00 → 30.51 → 38.15 |
| NVARC3 (cpmpml, darraghdog, sorokin and three others) | 31 | 37.51 (Oct 5 20:18) | 16.07 → 21.79 → 37.51 |
| _hans (deepdreamer) | 6 | 37.07 (Oct 8 00:07) | 28.70 → 33.00 → 35.69 → 37.07 |

Other placings:
- Daniel Franzen is still at 27.89 (rank 596, 89 submissions), and he submitted again on Oct 8.
- We are "Jovian Game Studios" at 28.87, rank 440. That draw is exp-070d (a D′ copy of Franzen, 10 streams).
- 21 teams are at 34.30 or above, and 288 teams at 30 or above.

### 1.2 What they said or published since Oct 2 (verified)

**Kaggle forum.** I read every thread with activity since Oct 2, including the pinned ones; the full list is in
`threads/`.
- No post by Tufa, Yi-Chia, Majkel1337, the last dance, dreach.ai, mtg or artificialagencylab.com.
- The only top-10 voice is CPMP (NVARC3), in [746010](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/746010)
  on Oct 6: "I can't disclose anything about what we do before end of competition."
- Teams' scores on the public 25 games, all in
  [732854](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/732854):
  - Nick2187, Oct 4: "My current best is 56.76. 124 levels cleared and 11 games fully won. Just submitted that setup".
    Their LB best since then is 33.70.
  - Scott Le Grand, Oct 5: "53.74 and I haven't been able to repeat it". His LB best is 30.34.
  - Nick Pellegrin, Oct 4: about 105 levels, or about 40%, on the public 25 games, which "scores ~23% on the private
    set".
  - Mark Barney, Oct 4: about seven public games ("the Slippery Seven") barely move for any current solution.
- Shehab Anwer, [746295](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/746295), Oct 6: a
  "lexicon view" perception tool had no visible score effect on Franzen's base (24.95, one draw).
  - "On levels it cleared, the agent used a median of 0.71× the human's moves. Yet 62% of all its moves went into the
    one level per run it never understood."
- **Previously missed** (intel.md, Sep 28): Scott Le Grand,
  [743952](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743952): "I plan to give away
  whatever fine-tuned model I have mid-month".
- Queue complaints, [745951](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745951), Oct 5-7:
  waits of 6-17 h. This matches our lesson 0031.

**GitHub** (git ls-remote and blobless fetches; the GitHub search API returned 502 or 403 in this session).
- `Tufalabs/duck-harness` is still at 7652836 (Jul 1).
- The `website` repo is unchanged since Sep 18, and the Tufa research page's newest post is from Jul 18.
- The RL-stack forks (miles, slime, sglang, vllm, verl and the others) are unchanged since Sep 9, with no new branches.
- I found nothing public from Yi-Chia, Majkel1337, mtg, dreach.ai or the last dance.

**Kaggle user content** (datasets, kernels and models of every listed member).
- Nothing new from Tufa, apart from driessmit1 re-creating two deleted Milestone 1 datasets on Oct 5: the vLLM 0.19
  wheelhouse and the Qwen3.6-27B-FP8 snapshot. Both are reproducibility items, not a new method.
- Yi-Chia's (threerabbits) models are unchanged since June.
- Majkel1337, michaeltgao and the last-dance members have no public content.

**Hugging Face.** No models from any of these handles (HF API `models?author=`).

**arXiv.** Nothing by these teams. The new ARC-AGI-3 papers all use frontier models:
- [VISTA 2610.02200](https://arxiv.org/abs/2610.02200), Oct 1: a lossless visual memory the model can retrieve from.
  Opus 5.0 goes from 40.68 to 100 on the public set.
- [Kepler 2610.00834](https://arxiv.org/abs/2610.00834), Sep 30: auditable executable world models. Animation frames
  "contained task-relevant information absent from settled text grids".
- [Schema 2609.39140](https://arxiv.org/abs/2609.39140), Sep 30.
- [FreeEvolve 2610.09197](https://arxiv.org/abs/2610.09197), Oct 6.

**Blogs and press.** No ARC Prize blog post since Sep 3. TechTimes (Oct 6) and Metir AI (Oct 5) only restate the
leaderboard.

### 1.3 Artificial Agency Lab's public runtime (verified: I read the configs; I did not run anything)

Richard Csaky (artificialagencylab.com, #6) published these as public Kaggle datasets on Sep 29-30, before they were
added to intel.md:
- [`arc-expert-gptq-arc-v1`](https://www.kaggle.com/datasets/richardcsaky/arc-expert-gptq-arc-v1): "ARC GPTQ-ARC
  routed-expert overlay", 64 GB, 589 downloads.
- [`arc-expert-int3-g16-v1`](https://www.kaggle.com/datasets/richardcsaky/arc-expert-int3-g16-v1): 49 GB.
- Three 257 MB "arena" runtimes, for example
  [`arc-arena-hero-v2-blocks2-p50-int3-runtime`](https://www.kaggle.com/datasets/richardcsaky/arc-arena-hero-v2-blocks2-p50-int3-runtime).

The licence field is "other", so these are for ideas only. The p50 runtime's `configs/comparison.json` (variant
"hero-v2", dated "user, 29 September") specifies:

- **Capacity:**
  - 76 concurrent games (`max_num_seqs` 80) at a 98,304-token window.
  - An **int4 KV cache**.
  - **50% routed-expert pruning** (`prune50-layer`).
  - **int3 g16 experts "re-rounded from GPTQ-ARC"**, meaning GPTQ calibrated on ARC traffic.
  - **MTP off**.
  - The per-layer n-gram embedding (PLE) and the input embedding in host memory (UVA).
  - int8 dense kernels for `lm_head`, the GDN input projections, QSA qkv and attention out.
- **The change from their previous base:** that base ran 14 games at 128K with int8 KV and 3 MTP tokens, close to our
  operating point.
- **Schedule for 110 games:**
  - Two wall-clock blocks of 15,250 s, with 76 games in the first and 34 in the second.
  - A game with no level cleared retires at 90,000 completion tokens.
  - Each level gets an allowance of 90,000 × (levels cleared + 1) tokens.
  - An exact KV gate means no request is ever preempted.
- **Harness:**
  - A Duck port ("duck-transfer"), with reasoning history kept verbatim.
  - One model-written summary when the context cap is reached.
  - Nine "harness fixes": stuck notices, an inventory, a sandbox, timeout interrupts and others.
  - A `run_plan` helper.
- **Tests visible in the code:**
  - Gittins, hazard-index and Thompson schedulers.
  - Adaptive thinking.
  - A trained draft head (`draft_head_train.py`).
  - A Holo4-27B candidate. Their smoke test measured a 2.10 M-token KV pool for it.

**Inference:** their LB went 20.17 → 38.62 in the four days after they published this. That makes it one documented
route to about 38-39: a very wide, heavily compressed operating point on a stock policy. It is still 17 points below
the leaders.

### 1.4 Public-25 vs leaderboard calibration (data verified; the reading is inference)

| Team | Public 25 | LB |
|---|---|---|
| Franzen v3 | 46.5 (4 passes) | copies' mean 25.8 (ratio 0.55) |
| Our D′ copy | similar config | 28.87 (one draw) |
| Nick2187 | 56.76 | best since then 33.70 (≤ 0.59) |
| Scott Le Grand | 53.74 | ≤ 30.34 (≤ 0.57) |
| Nick Pellegrin | about 40 | about 23 (about 0.58) |

**Inference:**
- If our ratio is similar, exp-073b's 56.00 on the public 25 corresponds to an LB score of about **31-35**, not 56.
- At the same ratio, Tufa's and Yi-Chia's 55.8-55.9 would correspond to roughly 90+ on the public 25, or else to a
  system that transfers to unseen games much better.
- Either way, their edge is in the policy (how the model plays), not in serving capacity we can still add.

### 1.5 How the leaders got from about 28 to about 56 (inference; confidence low to medium)

- **No direct statement exists.** The only evidence is indirect:
  - Tufa's score has stopped rising: 55.89 on Oct 3, then three draws that did not beat it.
  - Yi-Chia made another single-step jump (+7.2 on Oct 7). That again fits a new trained artifact, given their June
    record of an on-policy-distilled (OPD) target model and DFlash drafters.
  - The public-to-LB ratio above means the leaders' policy is about 1.6-1.8 times ours on unseen games. The other routes
    seen on the forum (harness text, a capacity of about 38 via Artificial Agency Lab) have not reached that.
- **The post-training hypothesis (intel.md hypothesis 2) gained weight.** Serving-only routes now have a documented
  ceiling near 39.
- **Majkel1337's climb** (9 → 42.7 in six days, 11 submissions) is the only newcomer near the top.
  - The jump to 28.7 on Oct 1, two days after Franzen's notebook was published on Sep 29, fits adopting his base.
  - The +8 on Oct 6 is unexplained.

---

## 2. New public notebooks, datasets and models since Oct 2

### 2.1 Notebooks (verified; `scripts/kaggle_nb_scores.py`, 294 competition notebooks, 196 with a score)

- **No public notebook beats Franzen's 34.30.** That 34.30 is a copy's draw shown on his notebook page, per
  [746010](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/746010); his own LB is 27.89.
- The next scores, each one draw:
  - sujanmajhisuzan 31.93 (Oct 8)
  - leoprovorov "Fable & Astra Play ARC-3: Handbook + Harness" 31.73 (Oct 7)
  - skarin 31.66
  - shiiin9 31.54
  - sigeward 31.27
- All of them except leoprovorov's mount exactly Franzen's inputs (pennyroyal-v253, Intel W4A16, Albucino drafter), so
  they sit inside his copy noise (mean 25.8, SD 3.9).
- No public notebook reports a Swift or Tinfield result. A search for "swift", "swift 1.5" and "tinfield" finds no
  scored notebook that uses them.
- wkdrbwnd1 ran notebooks named "mtp-featdump-full", "mtp-gtcapture" and "eval30-newbase-ab-promptrule" on Oct 6, on
  Lord Han Solo's base. **Inference:** someone is collecting features to train an ARC MTP drafter.

### 2.2 Fine-tuned Flash-Next checkpoints, drafters and pruned or low-bit builds

- **Fine-tunes:**
  - Nothing new on Kaggle. Tong Hui Kang's LoRA is unchanged since Sep 30 (v15).
  - On Hugging Face, [Tinfield-1](https://huggingface.co/badtheorylabs/Tinfield-1) is an agentic post-train of
    Flash-Next released Sep 21 (see §3). A Kaggle "ARC research conversion" to NVFP4 exists
    ([natnitarach/tinfield-1-nvfp4-probe](https://www.kaggle.com/models/natnitarach/tinfield-1-nvfp4-probe), Sep 27),
    with no score attached.
  - `antoine1anthony/qwen3.8-flash-next-fullft` (Oct 5) has no model card.
- **Trained drafters:** none published for Flash-Next since Oct 2.
  [tcclaviger/Qwen3.8-Flash-Next-Dflash2](https://huggingface.co/tcclaviger/Qwen3.8-Flash-Next-Dflash2) (Oct 7) is an
  empty placeholder ("Soon™").
- **Pruned or low-bit builds (Hugging Face):**
  - RAZOR expert pruning ([arXiv 2609.30465](https://arxiv.org/abs/2609.30465)), keeping 384 or 256 of 512 experts,
    calibrated on a generic corpus. BF16 builds came Sep 30 and NVFP4 builds by cleyesode on Oct 5-6.
  - ISTA-DASLab's MoESQ
    [P48NVFP4](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-P48NVFP4-MoESQ) (Oct 2): 2.75 bits per weight, but
    SWE-bench 67.0 against 79.8 for BF16. It needs a patched vLLM 0.30. Too lossy.
  - Local Inference Lab's NVFP4 build trained with quantization-aware distillation (QAD, step 5500), on its own vLLM
    fork with b12x kernels. The authors could not distinguish its quality from their earlier build. Their licence is a
    custom LIL licence.
  - Intel MXFP4 AutoRound variants (Oct 4-8).
  - A 2-bit GSQ-RCO Flash-Next with an MTP runtime, cached on Kaggle by clothespin ("Strata QFN", Oct 5-6, llama.cpp
    style).
- **Pennyroyal (our SGLang build):** no release after v2.5.3. A 3.0 release candidate is in active development
  (`rc/3.0rc2`, merged Oct 7). Its branches add adaptive speculative width, ported draft-MoE GEMV, router and fused FP8
  KV store kernels, and low-row W8A16 GEMV for dense MXFP8 linears.

### 2.3 Swift-1.5: mirrors, leaderboard effect, and repeat loops

- **Mirrors and LB effect:** the Kaggle mirrors are unchanged (phuongncn, lordhansolo, cihanatak, michaelpoluektov). No
  team attributes a leaderboard result to Swift. Artificial Agency Lab kept a Swift profile in their runtime but serves
  the base NVFP4 model.
- **Licence:** the Hugging Face repo is no longer gated (HF API `gated: false`). Its licence is still the Swift Open
  License v1.0 on top of the Qwen Community License 1.0.
- **No public report of exact repeat loops.** The closest reports are runaways:
  - HelixML, [Sep 27](https://helix.ml/blog/swift-flash-next-four-gpu-evaluation): "an additional Swift pilot reached a
    16,000-token cap without an answer".
  - A "max tokens issue" on the
    [NVIDIA forum](https://forums.developer.nvidia.com/t/swift-1-5-qwen3-8-flash-next-63-4-fewer-thinking-tokens/384473)
    (Sep 28).
  - One runaway in apollo-mg's test
    ([HF discussion #3](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next/discussions/3), Sep 29).
- **No one reports running Swift with relaxed (lossy) speculative acceptance.**
- **Working setups that have been reported:**
  - todiadiyatmo, [HF discussion](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AutoRound/discussions/1),
    Sep 25: vLLM on 4× RTX 3090 with UkisAI's W4A16-AutoRound, MTP on and FP8 KV. MTP acceptance "80%"; tool-call
    quality 68/75 against 65/75 for Intel's AutoRound (temperature 0, medium effort).
  - The same author's
    [W4A16-Attn8-FP8PLE build](https://huggingface.co/todiadiyatmo/Swift-1.5-Qwen3.8-Flash-Next-W4A16-Attn8-FP8PLE):
    MTP acceptance 87% with 2 draft tokens, quality 67-68/75.
  - apollo-mg: llama.cpp with the base MTP head; acceptance 0.729 on Swift against 0.730 on base.
  - The only contrary report is auggie246
    ([discussion #1](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next/discussions/1)): acceptance "15-20%
    lower than on the base model".
  - UkisAI's own benchmarks use BF16, MTP disabled, temperature 1.0, top_p 0.95 and top_k 20 (model card).
- **Facts that bear on our loop:**
  - **Swift's MTP head and PLE table are byte-identical to the base model's** (sha256, per the todiadiyatmo card; the
    UkisAI card says "includes the base model's one-layer MTP head"). So a draft "made for the original model" is the
    same head Swift ships.
  - **UkisAI's W4A16-AutoRound stores the GDN and QSA attention projections in INT4.** This is stated on the
    todiadiyatmo card and matches our exp-076 log entry. Intel's base build keeps them in BF16.
  - **UkisAI's own NVFP4 build quantizes only the routed experts** (`hf_quant_config.json`, 323 excluded modules). All 36
    `linear_attn` layers and all 12 QSA `self_attn` layers stay unquantized, as do the shared experts, routers,
    hyper-connections, embeddings and `lm_head`.
- **Hypotheses for our loops, most likely first (inference):**
  - **H1. INT4 linear-attention projections.** Quantization error compounds in the recurrent state over long multi-turn
    contexts. Of the reported working setups, only todiadiyatmo's vLLM run uses this INT4 layout, and it was a
    single-turn quality test, not long multi-turn play.
  - **H2. Lossy acceptance at 0.5** against Swift's sharper, RL-trained distribution.
  - **H3. Temperature 0.7**, below Swift's validated 1.0.
  - The test order is in action 2 below.

Confidence: high on the facts above (read at source); low on which hypothesis is the cause.

---

## 3. New open-weight models released Sep 20 - Oct 8 that might fit one 96 GB card

| Model (date) | Size | Licence | Quantized builds | SGLang / vLLM | Verdict |
|---|---|---|---|---|---|
| [Tinfield-1](https://huggingface.co/badtheorylabs/Tinfield-1) (badtheorylabs, Sep 21) | Flash-Next post-train: 177B total incl. n-gram tables, 6.6B active | Qwen Community 1.0, the same as the base | Vendor: BF16 and GGUF (61-111 GB). Community: NVFP4 probe (HF and Kaggle), EXL3. No W4A16 | Same `qwen4_exp` architecture as Flash-Next, so it should load where Flash-Next does (unverified) | **The only plausible candidate.** Vendor claims Terminal-Bench 4.0 33.0 vs 29.0 for the base and DeepSWE 62 vs 58.7. Needs a W4A16 build and a hash check against the base |
| Swift-1.5 Flash-Next (UkisAI, Sep 22) | Same as the base | Swift Open v1.0 + Qwen Community | W4A16-AutoRound, AWQ, NVFP4 | Yes | See §2.3 |
| [Holo4-27B](https://huggingface.co/Hcompany/Holo4-27B) (H Company, Sep 24-28) | 27B dense vision-language model on Qwen3.8-27B | **CC BY-NC 4.0** (non-commercial) | FP8, NVFP4, AutoRound W4A16, GGUF | vLLM (Artificial Agency Lab smoke-tested it) | OSWorld 2.0 61.7%. The licence fails the rules' open-source test. Dense 27B decodes about 4× the active parameters of Flash-Next |
| Holo4-35B-A3B (Sep 24-28) | 35B total, 3B active, on Qwen3.6-35B-A3B | Apache-2.0 | FP8, NVFP4 | Yes | OSWorld 2.0 30.9%. Weaker than Flash-Next |
| [JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL) (autotrust, Sep 30) | Qwen3.8-27B LoRA | Apache-2.0 | FP8, NVFP4 | vLLM with its own `/v1/decide` endpoint | A "decision model" that outputs typed choices with probabilities. Not a replacement for the agent |
| Reflection Beam (Oct 5) | 501B total, 23B active | Apache-2.0 (weights pending) | — | — | Too large; weights not out |
| Mistral Large 4 preview (Oct 6) | About 1.05T | Weights pending | — | — | Too large |
| MiMo-V2.6-Flash (Sep 21); NaiveAI (Sep 27) | About 309B total, 15B active | MIT | GGUF and REAP-50 community builds | — | About 155 GB at 4 bit. Does not fit |

Not released in the window: Qwen 4 (announced at Apsara on Sep 22, still training), Gemma 5, and any new Nemotron
(3.5 Lightning came out Aug 11 and is text-only). Ling-3.0-flash-VL is from Sep 4-10, and Ling-3.1-flash has only an
OpenRouter listing.

Sources: HF API and model cards; [Reflection](https://reflection.ai/blog/introducing-beam);
[The Register on Mistral, Oct 6](https://www.theregister.com/ai-and-ml/2026/10/06/european-ai-flag-bearer-mistrals-new-open-weights-model-is-le-chonk/5301443);
[AI Weekly on Holo4](https://aiweekly.co/alerts/h-company-ships-holo4-agents-27b-hits-617-on-osworld-20); the
[digitalapplied October tracker](https://www.digitalapplied.com/blog/ai-model-releases-october-2026-tracker).

Confidence: high that no stronger new model fits the card; medium on Tinfield-1's claims, which are the vendor's own.

---

## 4. ARC Prize and Kaggle staff statements since Oct 2 (verified; confidence high)

- **No staff post** in any forum thread with activity since Oct 2, pinned threads included.
- **Licence questions are still unanswered:**
  - [745079](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745079): the Qwen Community
    License and NVIDIA's NVFP4 build. Follow-ups on Oct 2 and Oct 5 tagged @gregkamradt and @macruzbar.
  - [745837](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745837), Oct 4: whether the
    pretrained-model exemption (§2.5.a.3) covers the Qwen licence. The only reply, Oct 6, is "no response from the
    organization".
- **Docker image for the scored rerun**
  ([745654](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745654)): the only answer comes from
  a non-staff user, Kriztian Rubin. It says the rerun uses the submitted version's pinned image. Unverified; our lesson
  0029 pins the image anyway.
- **Rules and pages are unchanged.** I fetched them through the Kaggle API today (`pages/now/`); the Prizes, Timeline,
  Code Requirements and Evaluation pages are identical to the Sep 23 copies.
  - The rules still require an "open source system, open source model, and open source weights" under the OSI
    checklist (§2.5.a).
  - §2.5.a.3 still exempts "input data or pretrained models with an incompatible license".
  - The winner licence is CC-BY 4.0.
  - The data description still says 110 private games, half for the public LB and half for the private LB.
- **arcprize.org:**
  - No blog post after Sep 3.
  - The competition page says "All code and methods must be open sourced to be eligible for prizes".
  - It lists the Milestone 2 prizes as $25K / $10K / $2.5K, against Kaggle's $25K / $7.5K / $5K.
  - Per search snippets, the X post of Oct 1 named the winners as Franzen, Lord Han Solo and Lohit Siriki.
- **Press:** TechTimes (Oct 6) says solutions must be CC0 or MIT-0. That is secondary and conflicts with Kaggle's
  CC-BY 4.0 rule, so treat it as unverified.

---

## What this changes for us

The actions are ranked by expected leaderboard gain per GPU-hour. GPU-hour figures are Kaggle RTX hours, and every gain
is an estimate.

1. **Get LB draws of exp-074t before tuning further on the public 25 (0 GPU-h; the owner submits).**
   - Five data points put LB at 0.55-0.62 times the public-25 score. If that holds, exp-073b's 56.00 is about 31-35 on
     the LB.
   - That draw decides between two paths. If it lands near 33, serving work (×1.1 at best) cannot close a gap of about
     1.7 times, and the remaining options are a better policy (a model or harness that transfers) and spending each
     draw on our best configuration. If it lands above 40, our ratio is better than other teams' and the plan stands.
2. **Swift-1.5 loop triage: two 25-minute gate runs, about 1 GPU-h.**
   - Gate A: the same UkisAI W4A16 build with **lossless acceptance** and Swift's sampling (temperature 1.0, top_p 0.95,
     top_k 20).
   - Gate B, if A still loops: a build whose **GDN and QSA projections are not INT4**. Assemble it CPU-only in Intel's
     layout (UkisAI's INT4 experts plus Swift's BF16 attention projections), or try UkisAI's NVFP4 build, which keeps all
     attention unquantized.
   - The base MTP head is byte-identical, so keep the draft.
   - If Swift then plays cleanly, the gain is somewhere between −5% and +25% score at our measured elasticity.
     Thinking tokens fall 56% on single-turn benchmarks, but mean output rose 12% on Terminal-Bench (vendor numbers).
   - The Swift licence is the owner's call before any submission.
3. **REAP recalibrated on our own ARC traffic, then more streams (about 2-3 GPU-h: one gate plus one full run against
   exp-073b).**
   - Re-score the experts on our logged requests, image turns included. That targets the fidelity probe's finding that
     REAP-448's error sits on image turns.
   - Then try keeping 384 experts with 18-20 streams.
   - Artificial Agency Lab's runtime (§1.3) shows the far end of this road: 76 streams with int3 and int4 compression.
     It reached about 38.6, so expect a modest gain and not parity with the leaders. Their overlays carry an "other"
     licence: use them for ideas only.
4. **A Tinfield-1 arm (CPU-only build, then about 2.5 GPU-h).**
   - It is the only new drop-in Flash-Next post-train with an agentic focus, and it has no licence beyond Qwen's.
   - First compare the tokenizer, template, MTP and PLE hashes with the base, as we did for Swift.
   - Then build a W4A16 version: RTN at g128 on the routed experts, with attention kept in BF16.
   - Run one gate, then one full run.
   - The gain is unknown, so this ranks below the Swift fix.
5. **Daily zero-GPU watch list:**
   - A Pennyroyal 3.0 tag. When it appears, run a 25-minute gate, since adaptive speculative width targets our decode
     bottleneck.
   - Scott Le Grand's promised fine-tuned model ("mid-month").
   - A staff answer on threads 745079 and 745837.
   - New Flash-Next drafters on Hugging Face (the DFlash2 placeholder).
   - Any Tufa, Yi-Chia or Majkel1337 artifact.
