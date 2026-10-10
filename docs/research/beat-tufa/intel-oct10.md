# Intel refresh: what is new since Oct 8 (2026-10-10)

One-line summary: the two leaders climbed again (Yi-Chia 59.17, Tufa 56.52) and mtg jumped to 44.32, but nobody
published how. No public notebook beats Franzen's 34.30. The only new technique with measurements behind it is a
static reasoning effort of "medium" (one competitor's runs on Franzen's stack). In short runs it almost doubled the
levels on the 15 hard public games, and it cost about 6 levels on the 10 easy ones. It is the one change from this
refresh worth GPU time: one hard-15 run at the hidden set's time share, about 1.6 GPU-h.

Research run 2026-10-10 01:03-01:35 UTC. Downloads are in the session scratchpad
`/tmp/claude-0/-home-user-Arc-Agi-3-Kaggle-comp/d342458e-03bd-545b-8a6d-06bca061963e/scratchpad/intel-oct10/data/`:
- leaderboard CSV `lbcsv/`, monitor JSON `lbhist/`, best submissions `teamsub/`
- notebook scores `nbscores/`, notebook listings `kernels/`, pulled notebooks `kpull/` (one new directory each),
  extracted cell text `ext/`, kernel logs `klogs/`, two runs' request logs `kout/`
- forum `topics/`, `threads/`; competition pages `pages/now/`
- a CC0 measurement study `dl_vinicius/`; Hugging Face listings and cards `hf/`; Kaggle searches `search/`

The helper scripts are in `intel-oct10/bin/`. Every downloaded file was treated as data: I read it and executed
none of it. Base64 and base85 blobs inside notebooks were decoded to text and read, never run. "Verified" means I
read it this session in the source named; "inference" is my reading.

## Bottom line

1. **Leaders (verified).** Yi-Chia Chen went 55.77 → 59.17 (submitted Oct 8 18:49 UTC) and Tufa Labs 55.89 → 56.52
   (Oct 8 17:53). mtg jumped 38.33 → 44.32 (Oct 9) and is now #3. SparseTech jumped 29.33 → 37.98 in one
   submission (rank 371 → 11). None of these teams posted, published a notebook, or released a model or dataset
   since Oct 8. We are unchanged at 28.87 and fell from rank 440 to 504; exp-074t drew 27.97.
2. **Public notebooks (verified).** Nothing beats 34.30. The new scores of 30 or more are copy draws: a
   byte-identical Franzen copy drew 30.75 and a D′ fork drew 30.82. The new techniques (prompt injections, a world-model
   simulator, NVFP4 on SGLang, temperature 0.4, a no-LLM explorer) have no score evidence above copy noise. The
   NVFP4 build is clearly worse: 145 tok/s and a 9.80 draw.
3. **Reasoning effort "medium" (verified data; the reading is inference).** juliancamilovilla ran clean single-knob
   arms on Franzen's notebook on Oct 8-9:
   - On the 15 hard games (37.5 min each), levels went 17 → 32 (score 5.37 → 16.66). Nine games went up and one
     went down.
   - On the 10 easy games (25 min each), levels went 47.5 ± 1.0 (four base runs from two authors) → 41.
   - Mechanism, read in the request logs: medium cuts the long-deliberation tail (p90 completion 5.4k → 3.4k
     tokens), not the typical turn.
   - Our Sep 23 test (exp-037) on the older harness found no gain at full length. The open question is whether the
     hard-game gain survives at the hidden set's time per game.
4. **Forum (verified).** Quiet. No staff post, no answer on the licence or docker questions, and the rules pages
   are byte-identical to Oct 8. The one technical thread is CPMP's (NVARC3) failed dataset mount. Two other
   competitors logged the same failure class in their notebooks on Oct 7-9, which fits our lessons 0030 and 0038.

---

## 1. Leaderboard now vs Oct 8 (verified)

Sources:
- the Kaggle leaderboard CSV, downloaded through the CLI at 2026-10-10 01:04 UTC, compared with the Oct 8 16:33
  CSV
- `competitions team-submissions <team_id>` for each team's best submission
- the leaderboard-monitor history (`tonghuikang--arc3-leaderboard-monitor-get-history.modal.run`), pulled 01:04 UTC

"Subs" is the total submission count, with the increase since Oct 8 16:33 in brackets.

| Rank (Oct 8) | Team | Best now | Change | Subs | Best submission (UTC) |
|---|---|---|---|---|---|
| 1 (2) | Yi-Chia Chen | **59.17** | +3.40 | 25 (+1) | 56967356, Oct 8 18:49 |
| 2 (1) | Tufa Labs | **56.52** | +0.63 | 159 (+2) | 56965506, Oct 8 17:53 |
| 3 (7) | mtg (michaeltgao) | **44.32** | **+5.99** | 64 (+1) | 56999892, Oct 9 07:52 |
| 4 (3) | Majkel1337 | 42.66 | 0 | 12 (+1) | Oct 6 18:32 |
| 5 (9) | NVARC3 | 40.97 | +3.46 | 32 (+1) | 56970800, Oct 8 20:03 |
| 6 (4) | the last dance | 39.30 | 0 | 76 (+1) | Oct 4 19:30 |
| 7 (5) | dreach.ai | 39.11 | 0 | 34 (+2) | Oct 6 08:30 |
| 8 (18) | Nhan Duc Nguyen (niyanng) | 38.84 | +4.10 | 15 (+2) | 57003830, Oct 9 09:29 |
| 9 (6) | artificialagencylab.com | 38.62 | 0 | 48 (+1) | Oct 7 22:01 |
| 10 (8) | 復活の混テキスト | 38.15 | 0 | 79 (+4) | Oct 7 15:41 |
| 11 (371) | SparseTech (aaronflouro and two others) | 37.98 | **+8.65** | 33 (+1) | 56977974, Oct 8 23:33 |
| 12 (10) | _hans | 37.07 | 0 | 7 (+1) | Oct 8 00:07 |
| 13 (11) | Lord Han Solo | 36.61 | 0 | 87 (+1) | — |
| 14 (12) | AI_hwlee | 36.54 | 0 | 15 (+2) | — |
| 15 (13) | fshindo | 36.16 | 0 | 9 (+1) | — |

Others:
- Son & Mark & Ronen 33.74 → 35.45 (#19).
- AFF AI CLUB, the D′ authors' team (shiiin9), went 31.54 → 33.92 on Oct 9. They have published nothing since Oct 4.
- New teams posted high first draws: Ryo Takaki 33.31 (2 submissions), SK #2 31.92 (1) and Szymon Kamiński
  31.04 (1).
- Daniel Franzen: 27.89, rank 673, 90 submissions. Scott Le Grand: 30.34.
- Us (Jovian Game Studios): 28.87, rank 504, 10 submissions. The leaderboard shows only our best, so exp-074t's
  27.97 (submission 56980485) does not appear.
- Teams at 40 or more: 3 → 5. At 38 or more: 8 → 10. At 34.30 or more: 21 → 23. At 30 or more: 289 → 333.
  4,076 teams in all.

What the jumps suggest (inference):
- **The leaders are still improving in small steps.** Yi-Chia beat their own best again with one submission a
  day; that fits a run of trained artifacts (intel-oct8 §1.5). Tufa's +0.63 is within draw noise of their Oct 3
  system.
- **The gap from us to #1 is now 30 points.** The gap to #3 (mtg) is 15.
- **SparseTech's one-step +8.65 is unexplained.** The only public trace of them is Aug-Sep datasets from
  aaronflouro: an SM120 inference binary ("sparson-ds4-bin-sm120") and a 150B REAP-pruned MXFP4 build
  ("sparson-reap150b-mxfp4"). That points to a custom runtime with pruning, a route like Artificial Agency Lab's
  (about 38-39), but it is a guess.
- **mtg (+6) and NVARC3 (+3.5) left no public trace.** CPMP repeated in thread 747438 that NVARC3 discloses
  nothing before the end.

**D′ family on the leaderboard.**

| Configuration | Leaderboard draws |
|---|---|
| D′ or close variants | 31.54 (shiiin9), 30.82 and 29.27 (amatlas fork), 29.15 (lwq255 "guarded"), 28.87 (our D′ copy), 27.97 (our exp-074t) |
| Franzen copies | Son Pham's 25.8 ± 3.9; four newer draws, 24.25-30.75 |

The four public D′ draws are each that notebook's best score, so they are biased upward.

**Inference:**
- D′ may be worth about +2-3 over Franzen on the leaderboard, or the gap may be selection; these draws cannot
  separate the two.
- exp-074t's 27.97 sits inside the D′ spread.
- No leaderboard evidence yet shows that REAP-448 with 14 streams helps.

## 2. Public notebooks (verified unless marked)

### 2.1 Scores

`scripts/kaggle_nb_scores.py --top 120` (289 competition notebooks, 195 with a score), diffed against the Oct 8 output:

- Still no public notebook above Franzen's 34.30.
- **New notebooks with a score:**
  - spark328/arc-prize-2026-arc-agi-3-franzen-v1: 30.75. Its code is identical to Franzen v3.
  - feili6458: 29.31.
  - viniciussignorelli/arc-agi-3-duck-franzen-base: 28.85. Identical to Franzen v3.
  - leoprovorov F001/F002 "Fable layer": 25.98 and 18.95.
  - hitarthjain0: 25.10.
  - juliancamilovilla/arc-agi3-franzen-h1: 24.17.
  - hknight888/hknight3-opt: 22.17.
  - superstringdev/arc-agi-3-nvfp4-sglang-public: 9.80.
- **New draws on known notebooks:**
  - amatlas D′ fork: 29.27 → 30.82.
  - prvsiyan (a Franzen V1 rerun with input checks): 2.95 → 27.83.
- **Updated, but no new best score:**
  - sujanmajhisuzan: still 31.93.
  - vladimiryakunin DF-WM: still 29.30.
  - bang1850: still 29.08.

### 2.2 What each new or updated notebook changes

Method: `kaggle kernels pull <ref> -m` into a new empty directory per notebook. I extracted the cell text and diffed
it against Franzen's notebook (`kaggle/franzen/arc-agi-3-milestone-2-solution.ipynb`) and D′
(`kaggle/dprime/affectify-arc-31-54-in-a-single-sub.ipynb`). The run results come from `kaggle kernels logs`.
Leaderboard scores are one draw each.

| Notebook (last run) | Base | What it changes | Score evidence | Verdict |
|---|---|---|---|---|
| juliancamilovilla arc-agi3-e1-med, -e1-med-g15 (Oct 9) | Franzen | **reasoning_effort = medium on every request** (5-line patch to the harness's chat-template kwargs, env `ARC3_STATIC_REASONING_EFFORT`) | hard 15: 17 → 32 levels; easy 10: 47.5 → 41 (§2.3); no leaderboard draw yet | **Test it** (§4) |
| juliancamilovilla arc-agi3-e1-low-g15 (Oct 9) | Franzen | same patch at "low" | crashed in cell 2: `cp: cannot stat '/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy'` (input not mounted at that path) | no data |
| juliancamilovilla arc-agi3-franzen-h1 (Oct 8) | Franzen | retained functions survive a sandbox timeout; huge result lists/dicts truncated | easy 10: 46.63, 45 levels (base about 47.5); leaderboard 24.17 | We already have this (ours-sandbox-timeout-keeps-work; exp-075). Noise |
| juliancamilovilla arc-agi3-franzen-d1 (Oct 8) | Franzen | `ARC3_CONTEXT_DRAIN_TOKENS` 58K → 24K, `ARC3_MAX_ACTIVE_STREAMS` 10 → 8 | easy 10: 29.76, 34 levels, 486 tok/s | Worse. Skip |
| juliancamilovilla arc-agi3-franzen-g15 (Oct 9) | Franzen | demo game list only (hard 15 at 37.5 min) | the baseline arm for E1 | reference |
| sujanmajhisuzan arc-agi-3-m2-top-submission (Oct 8 19:16) | Franzen | "V20 Superstack": monkey-patches `ToolAgent._build_user_prompt` to append (a) the last 250 characters of reasoning as "invariants" per cleared level, (b) a death-loop warning after 2 game-overs on a level, (c) a "strategy audit" block on every prompt after 12 prompts on a level, telling the model to UNDO/RESET; skips the demo run | best 31.93 (an earlier version); no newer draw | Prompt text; inside copy noise (lesson 0025). Skip |
| hitarthjain0 apex-cognitive-superstack, arc-agi-3-milestone-2-solution (Oct 9) | copies of sujanmajhisuzan and of D′ | none beyond the copies | 25.10 (its page credits that to a Franzen version) | noise |
| yukailiu arc-agi-3-champion-solution (Oct 9) | D′ | infrastructure only: input-path fallbacks with a `/kaggle/input` scan, libcuda stub handling, always writes submission.parquet; image `kaggle-images/python@sha256:e5452ce6…` (lesson 0029: that is the CPU image) | none | noise |
| bang1850 arc-prize-2026-sovereign-agent (Oct 9) | Franzen | deletes the customization hook (so solver defaults apply) and re-sets six env flags to the values Franzen already uses | demo: 25 games, 1 pass, 2 h 13 min, 43.09; leaderboard best 29.08 (old) | noise |
| vladimiryakunin DF-WM (Oct 9) | Franzen | adds a 472 KB "und" patch (a world-model simulator `SimFrame`/`check_sim`, BFS `search`, `act_checked`, a cycle detector, a claim graph and others) behind `ARC3_UND_*` and `ARC3_WORLD_MODEL_SIM`; 4-pass demo setting; demo skipped in this version | leaderboard best 29.30 (Oct 4 version); no new draw | no evidence. Skip |
| superstringdev arc-agi-3-nvfp4-sglang-public (Oct 9) | Franzen harness | ModelOpt NVFP4 W4A16 (block 16) with Marlin MoE, external BF16 PLE, its own SGLang runtime, a counted RESET every 600 s during server start | server ready after about 33 min; 145 tok/s (Franzen's W4A16: about 570); demo 27.84 on 10 games over 59 min; leaderboard **9.80** | Negative for NVFP4 on this path |
| hknight888 hknight3-opt (Oct 9) | Franzen v3 | `LOCAL_ANALYZER_TEMPERATURE` 0.7 → 0.4 | leaderboard 22.17 | one low draw. Skip |
| leoprovorov F001/F002 "Fable layer" (Oct 8) | Franzen | harness-computed indicator notes, a budget alert, a pinned level log (F002: plus a pinned digest of cleared levels) | leaderboard 25.98 and 18.95 | close to our patches 03/05/06; no gain. Skip |
| luisignaciomoreno Prometheus (Oct 10) | none (CPU-only, MIT-0) | no LLM: masked state memory, DSL world model, click ranker; its own budget sweep reports 4.0 levels at 400 actions and 8.5 at 1,600 on the public 25 | no draw | irrelevant to us |
| spark328, viniciussignorelli, zhangyues0000 forks | Franzen v3 | none (a copy, or a 1-pass demo) | 30.75 and 28.85 | copy noise |

Not ARC solvers, or too far from our stack: kai57sh (an offline non-LLM agent), iancblenke Carnot, poco34 BAYES,
manderson240 Cohezion, omn1v3r53 Omniverse, huyle345 VoT-DAR, hieunnguyen (unrelated), dhiaalhemdani (its own source
bundle), fabioolival (the stock Duck on the RadixArk NVFP4 build in vLLM), skarin (a token-accounting parser, no
score).

### 2.3 The reasoning-effort experiment in detail

**What the knob does (verified).**
- The Flash-Next chat template defaults to `reasoning_effort` "xhigh". That default adds "think carefully,
  validate key assumptions, consider alternatives" to the system prompt (lesson 0021).
- Per juliancamilovilla's patch docstring, "medium" injects nothing and "low" injects "think briefly".
- Franzen's harness only lowers the effort on a truncation ladder (`ARC3_REASONING_EFFORT_LADDER`), and that is off.
  So every request runs at xhigh.
- E1 adds:
  - an env `ARC3_STATIC_REASONING_EFFORT`, read in the method that builds `chat_template_kwargs`, before the ladder
    lines;
  - an `os.environ` line setting it to "medium".
- The logs confirm the patch applied. Every response record in the E1 request logs carries
  `'reasoning_effort': 'medium'`.

**Runs.**
- All are on Franzen's unchanged serving stack: Intel W4A16, the Albucino drafter, 10 streams, image 57e612b4.
- One run per arm, on Kaggle, Oct 3-9.
- The easy-10 baselines include vinicius's three runs from his CC0 measurement study (§2.4).

| Games, budget | Arm | Levels | Mean score | Actions | Tokens per action | Gen tok/s |
|---|---|---|---|---|---|---|
| hard 15, 37.5 min, 15 games on 10 slots | base (franzen-g15) | 17 | 5.37 | 1,097 | 1,154 | 541 |
| same | **medium** (e1-med-g15) | **32** | **16.66** | 1,583 | 876 | 593 |
| hard-10 subset of the above | base: julian 13; vinicius 11, 18, 15 | 14.3 ± 3.0 | | | | |
| same | **medium** | **23** | | | | |
| easy 10, 25 min, 10 slots | base: julian 47 (46.31); vinicius 47, 49, 47 (46.6, 53.2, 50.4) | 47.5 ± 1.0 | 49.1 | | 650 (julian) | 585 |
| same | **medium** (e1-med) | **41** | **38.01** | 1,337 | 639 | 567 |

Hard-15 levels per game, base → medium:
- up (9 games): cn04 1→2, dc22 1→2, g50t 0→1, ka59 2→5, s5i5 1→3, sk48 0→1, sp80 1→3, tn36 2→6, wa30 1→2
- down (1): lf52 2→1
- tied (5): bp35, cd82, ls20, m0r0, su15

That is a two-sided sign-test p of about 0.02.

Easy-10, base (julian, Oct 3) → medium:
- down (5 games): ar25, lp85, tu93, vc33, and tr87 4→1 (its level 1 took 124 actions against a human 54)
- up (2): ft09, re86
- tied (3): r11l, sb26, sc25

**Mechanism.** I downloaded the request logs of ka59 and tn36 from both hard-15 arms; my parser is
`bin/reqstats.py`.

| Run, game | Responses | Mean completion | p50 | p90 | Max | Token share in completions ≥ 6k |
|---|---|---|---|---|---|---|
| base ka59 | 73 | 1,864 | 647 | 5,352 | 12,288 (1 truncated) | 28% |
| medium ka59 | 79 | 1,399 | 639 | 3,401 | 9,827 | 17% |
| base tn36 | 61 | 2,092 | 1,140 | 4,815 | 12,288 (1 truncated) | 41% |
| medium tn36 | 79 | 1,426 | 1,040 | 3,357 | 5,876 | 0% |

Medium removes the long-deliberation tail and leaves the median turn unchanged.

**Caveats.**
- Each arm is one run, and the hard-15 baseline is a single run. Its 17 levels does match vinicius's hard-15 run
  (17 levels, 4.53).
- Both budgets are short. Each game got about 25 slot-minutes; in our full-length public-25 runs and on the hidden
  set it gets about 70.
- The two hard-15 runs were 6.7 h apart on different Kaggle sessions. Serving matched within 10% (541 vs 593 tok/s).
- **Contrary evidence: our exp-037 (Sep 23).**
  - Setup: medium on the older TAAF anim harness with Flash-Next, at full length.
  - Result: 6.70 and 37 levels against 7.86 and 38.
  - Mechanism: +35% requests and −35% tokens per request.
  - Its log reads: "per-call quality fell about as much as the call count rose".
  - Two differences from now: that harness was much weaker, and the effect was not read separately on hard and
    easy games.

**Inference.**
- Medium makes the agent act more and deliberate less.
- That helps where the model is stuck exploring. vinicius's transcript study puts 9 of the hard 15 in the
  "hypothesis/exploration" failure class.
- It hurts on games that need long planning on later levels.
- The hidden set behaves more like the hard games. Our leaderboard runs at about 0.55 of our full-length public-25
  score, and most hidden games are expected to stay near level 1.
- So the trade could favour medium on the leaderboard if any of the hard-game gain survives at the full time
  share. Nothing measured so far shows that it does.

### 2.4 A public measurement study worth knowing (verified; dataset viniciussignorelli/arc-agi-3-duck-measurement-study, CC0, Oct 9)

vinicius's `eval/FINDINGS.md` logs his experiments on Franzen's stack. The ones that bear on us:

- **Run-to-run noise in the short diagnostics.**
  - Easy 10 at 25 min: 47, 49, 47 levels (50.0 ± 2.7).
  - Hard 10: 11, 18, 15 levels. One game, cd82, alone swung 1, 6 and 2.
  - His conclusion: "A 2-run design cannot detect anything below about +4 levels per run here." This is the
    yardstick I used in §2.3.
- **12 streams on stock memory (Oct 2).**
  - Prefix-cache hits fell 94% → 70% and uncached prefill tripled.
  - Decode stayed flat at about 780 tok/s.
  - Levels fell 47 → 37.
  - This fits our finding that more streams need memory freed first (REAP-448).
- **Speculative steps 4 cannot run (Oct 3).** The run raised `NotImplementedError: Qwen QSA requires
  speculative_num_draft_tokens <= the QSA compress ratio (4)`. Our exp-072j (steps 4) also produced 0 levels.
- **Three prompt-level interventions were neutral or slightly negative:**
  - a preloaded `tools` namespace: 45.3 vs 47.7 levels, p 0.05 over 3 × 3 runs;
  - `ARC3_LEVEL_INVENTORY`: 15.0 vs 15.0 on hard 10;
  - a probe ledger: 13 vs 15.0, with the ledger used in 6.6% of turns.

  This agrees with our lesson 0025.
- **Two input-mount failures (Oct 7-9).** A kernel failed twice because `dfranzen/taaf-kaggle-source-bundle-copy` was
  not mounted under `/kaggle/input/datasets/`. The same notebook ran under a new slug.

## 3. Forum and other channels (verified)

**Kaggle forum.**
- Method: `competitions topics list -s new|recent|active|hot` (pages 1-4 and a 50-per-page check), then
  `topics show <id>` for every thread created or commented since Oct 8 16:30 UTC.
- No new thread on Oct 9. Activity since Oct 8:
  - [747438](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/747438) "Dataset error"
    (CPMP, NVARC3, Oct 8 18:44).
    - Saving a notebook that uses a model from their last two submissions failed with `ERRORED_MOUNTING_DATASET …
      retry budget exhausted (30 attempts): rpc error: code = Internal desc = dataset loading failed`.
    - Natapong Nitarach replied "Same here!". No staff reply.
    - CPMP again declined to share methods.
  - 746295 (Shehab Anwer's lexicon view): a toy example, no score.
  - 745951 (queue): "even submitted notebooks are taking longer now" (Oct 8 16:57).
  - 744545, 747216, 747304: no technical content.
- **No staff post anywhere since Oct 2.** No new answers on:
  - the Qwen licence and prize eligibility (745079, 745837);
  - which Docker image a scored rerun uses (745654).
- The comment thread on Franzen's notebook (kernel topic 744763) has nothing new since Oct 4.

**Competition pages.** I fetched them through the Kaggle API (`pages/now/`). All nine pages are byte-identical to the
Oct 8 copies: rules, timeline, code requirements, evaluation, data description, prizes and the rest.

**Docker image (inference from notebook metadata).**
- Notebooks run on Oct 9-10 carry a new GPU-image digest, `kaggle-private-byod/python@sha256:37c64f7d…`. bang1850 has
  it, among others.
- bang1850's log shows Python 3.12 site-packages and a complete 2 h 13 min Franzen run.
- So the image Kaggle gives unpinned notebooks seems to have moved again, apparently back to 3.12; that is not
  verified.
- Most M2 forks, julian's runs included, still run on Franzen's pinned 57e612b4, as ours do. No action.

**Kaggle input mounts (verified, three sources).** Missing or failed input mounts on Oct 7-10:
- CPMP's thread;
- vinicius's FINDINGS;
- julian's e1-low-g15 crash.

Our session A also saw 4-5× slower input reads at 00:14 Oct 10 (lesson 0038). Our builder's `--input-fallback` and
`--wait-inputs` cover the path cases (lesson 0030). A mount that never arrives still fails the run.

**Other channels.**
- **Kaggle user content.**
  - Checked: datasets, notebooks and models of every member of the top 11 teams, plus Franzen, Scott Le Grand,
    shiiin9, Artificial Agency Lab, Lord Han Solo and the new high-scoring teams.
  - Nothing new since Oct 8. Scott Le Grand's promised fine-tuned model ("mid-month") is not out.
  - New Kaggle mirrors of the Intel W4A16 build and the Albucino drafter appeared (mirzamilanfarabi). They are
    copies only.
- **Hugging Face.**
  - Since Oct 8: about 35 community quantizations of Flash-Next and Swift-1.5 (GGUF, EXL3, NVFP4 variants, a W4A4
    "NVFP4 dense" build of 171 GB).
  - No Flash-Next drafter, no ARC fine-tune, no Tinfield-1 update.
  - The DFlash2 drafters released are for the dense 27B, not Flash-Next.
- **GitHub.**
  - `Tufalabs/duck-harness` main is still 7652836.
  - New PR #9 (Oct 9) is from an outside contributor: "Fix context retention and add DeepSeek thinking support".
  - The Tufa website is unchanged since Sep 18.
- **Web.** No ARC Prize blog post or press item since Oct 8.

## 4. Implications for our config

### Candidate 1: static reasoning effort "medium" (the only one worth GPU time)

- **The change.**
  - A 5-line patch: `kaggle/franzen/patches/ours-09-static-effort.patch`, our own text and not a copy of E1's.
  - Where it goes: in `tool_agent.py`'s chat-template-kwargs builder, before the ladder lines:

    ```python
    static_effort = os.environ.get("ARC3_STATIC_REASONING_EFFORT", "").strip()
    if static_effort:
        kwargs["reasoning_effort"] = static_effort
    ```
  - The env flag: `--env-add ARC3_STATIC_REASONING_EFFORT=medium` in `scripts/build_franzen_nb.py`.
  - Everything else stays as exp-074t: D′, REAP-448, 14 streams, acceptance 0.5.
  - It cannot be done with flags alone. The ladder only acts after a truncation. A server-side
    `--default-chat-template-kwargs` is unreliable, because the harness sends `chat_template_kwargs` wholesale
    (Franzen's docstring on `_preserve_thinking_kwarg`).
- **The evidence.** §2.3:
  - +15 levels on the hard 15 (9 up, 1 down);
  - −6.5 levels on the easy 10 (about 6 SD of the base runs);
  - both at about 25 slot-minutes per game, one run per arm;
  - against it, our full-length null from Sep 23 on the older harness.
- **Expected size (inference, wide).**
  - In the short regime, the public-25 trade nets about +8 levels, or roughly +2-3 points on a mean of 22-25.
  - At the hidden set's time share, anything from −1 to +3 leaderboard points.
  - The leaderboard leans toward hard games, which is why the sign could be positive. Our own null at full length
    is why it may be zero.
- **CPU check first (0 GPU-h).** Render the served chat template for xhigh and medium (lesson 0021) to confirm
  exactly which text medium removes. Then apply the patch to Franzen's tree with the builder's apply check.
- **The GPU test (about 1.6 GPU-h, one run).**
  - Run the hard 15 only, with the exclusion list set to the easy 10 as in julian's g15 variant.
  - Use 14 slots and about 80 min per game. That roughly matches the slot-minutes the hard games get in our 121-min
    public-25 runs, where easy games finish early and free slots. Add 10-20 min of server start.
  - Compare hard-15 levels with our eight full-length runs. They range 39-58 (exp-073, 073b, 075, 077, 078, 079,
    080, 081). The three base-configuration runs scored 47, 58 and 43.
  - Also read tokens per action and actions per solved level, since RHAE pays for efficiency.
  - Decision rule, fixed now:
    - **60 or more levels** (above every run so far): do one full public-25 run to measure the easy-game cost at
      full length, then put it into the leaderboard rotation.
    - **50 or fewer**: drop it.
    - Between 51 and 59: one repeat before deciding.
- **What a positive result would not settle.** The easy-game loss at full length still needs measuring. If the
  hard-15 run is clearly positive, the next design to consider is effort by phase: medium until a level's first
  win, xhigh after it. tr87 lost its level 1 at medium, so that is untested.

### Watch, no GPU

- A leaderboard draw of juliancamilovilla's E1 notebook. Its best-score field will show it.
- A rerun of julian's "low" arm, which crashed on the input mount.
- Any notebook or post from mtg, SparseTech or Nhan Duc Nguyen (the three unexplained jumps).
- Scott Le Grand's fine-tuned model.
- A staff answer on 745079 and 745837.
- The image Kaggle gives unpinned GPU notebooks (37c64f7d…).

### Noise (do not spend GPU on these)

- **Prompt and harness-text additions:** sujanmajhisuzan's V20 Superstack and its hitarthjain0 copy,
  leoprovorov's Fable layer, DF-WM's world-model simulator, and vinicius's tools, inventory and probe ledger. Each
  is inside copy noise or measured neutral-to-negative, which matches lesson 0025.
- **Serving variants:**
  - NVFP4 W4A16 with Marlin on SGLang: 145 tok/s, a 9.80 draw.
  - Shorter drain with 8 streams (D1): −13 levels on the easy 10.
  - Temperature 0.4: one draw at 22.17.
  - 12 streams without freed memory: −10 levels on the easy 10.
- **Robustness forks:** yukailiu's "Champion Solution", prvsiyan's, and bang1850's hook removal. Their scores are
  copy draws.
- **No-LLM agents:** Prometheus and the others. They score far below an LLM harness.
