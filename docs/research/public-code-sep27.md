# What other teams' public code and notes say (read 2026-09-27)

One-line summary: the public Duck forks all draw 3-7 on the hidden leaderboard, and the evidence from other teams points
at tokens per hour (serving) and their allocation, not harness text, as what moves it; three serving changes follow.

Source reports (subagents, 2026-09-27): public-code-sep27/web-and-forum.md (web, arXiv, 225 forum threads read with
`kaggle competitions topics show` / `topic-messages`), public-code-sep27/duck-lab-repos.md (four public repos),
public-code-sep27/qwen38-rl-repo.md. Sources were read, not copied: code without a license is used for ideas only, and anything we adapted is rewritten and
credited in `scripts/taaf_ours_patch.py`. Numbers below are other teams' unless marked "ours".

## Leaderboard context

- Top of the public LB (Sep 26): Tufa Labs 27.29, Daniel Franzen 21.01, Lord Han Solo 20.80, Tong Hui Kang 20.53, Yi-Chia
  Chen 18.80, NVARC3 16.07. Ours: 4.70 (exp-054, one draw).
- Tufa's trajectory (Tong Hui Kang's tracker, relayed): 4.71 (08-30) -> 11.04 (09-06) -> 18.81 (09-13) -> 27.29 (09-26),
  all about 540 min runs. A participant's comment: "a lot of the improvements simply came from squeezing more tokens out of
  the hardware and allocating them properly".
- Milestone 2 publications (forum): Tufa will not open-source until the end; NVARC3 (NVIDIA) will not; Tong Hui Kang only
  if 1st; Lord Han Solo if in the top 3 of teams willing to share. So a ~20-point notebook may appear by Sep 30 (Lord Han
  Solo), but the leaders' code will not.
- Public-25 -> LB: teams report a 3-4x drop (e.g. 17.3 -> 5.2/6.9, 22.3 -> 5.4, 15.7 -> 7.4); identical notebooks vary
  1.5-2.3x between submissions; a stock-base resubmission series: mean 3.29, sd 0.51 over 12. Our own pairs fit this.

## Tufa Labs' newer source (jakobbrggen/taaf-kaggle-source, 2026-09-01; MIT code in a CC0 bundle)

- Experiment 4: the `animation()` frame-retrieval tool is now off by default. In their notes it "bought no score" across
  two experiments (called unprompted in 64% of calls; 2.1% hit rate on informative animations). The per-action animation
  summary stays (+3.3% tokens, no harm) and is shown only when `worth_inspecting` (> 200 transient px or bbox > 25% of the
  board) or `board_unchanged`.
- Experiment 5, "AVO" (after NVIDIA's AVO): durable per-game memory, a rotating inspect/plan/implement/evaluate directive,
  a text-only stagnation supervisor, an exploit directive after 60% of the game's budget. Opt-in; no result published.
  Our P19 (supervisor) already tested the supervisor idea and cost throughput through its extra calls; AVO's is text only.
  Deployed to Kaggle on Sep 1, between Tufa's LB draws of 4.71 and 11.04. Its memory file is shared by every game that
  runs at once (all runtime states sit in one directory). -> **P29 / exp-061**, with the memory made per game.
- Their control run (model-20260816-q38-anim-on, Qwen3.8-27B FP8) renders the board at upscale 8 with 1-px grid lines;
  the public Flash-Next notebooks use upscale 4 without lines. -> **P27 / exp-058**.

## Serving (the lever other teams' LB moves line up with)

- Flash-Next weights: 81.8 GiB with the MTP head, 74.3 GiB without. Thuitanium's replay bench (serving only): MTP off +
  7 GiB KV + 28 sequences 722 generated tokens/s vs 359 for the stock profile. Son Pham's Flash-Next lock (his Flash-Next LB
  draws 4.3-7.4): official vLLM nightly container, MTP off, 22 sequences, 6,144-token batches. Scott Le Grand (forum): 16 sequences
  brought the stock Duck to "the brink of the top 10%"; "all of my harness engineering beyond that has been utterly
  useless" (his notebook credits one change, animation frames in the sandbox, with 3.20 -> 3.71).
- Ours: our MTP 2/4 tests kept the head loaded, so they never gave the freed memory to KV. -> **stress tests
  mtp0-14g-s16/-s28, exp-059**.
- Ours (exp-054's server metrics): 1,609 requests, 21.2k prompt tokens each (79% above 20k), 2.46M generated, no prefix
  cache; per request 2.0 s prefill, 23.3 s decode, 96 s queued. The Duck trims one history block per call at its budget,
  so consecutive prompts never share a prefix. Son Pham's best config trims 50% at a time ("swap50") and reports caching
  89% of prefill. -> **P28 (hysteresis trim) + prefix caching + 16 lanes, exp-060**.
- Pennyroyal (SGLang fork for this GPU, Apache-2.0 launcher at keplerzip/Qwen3.8-Flash-Next-NVFP4-RTX-PRO-6000-Single):
  Flash-Next 147 / 253 / 416 / 632 generated tokens/s at 1 / 2 / 4 / 8 concurrent (short synthetic prompts), FP8 KV,
  radix prefix cache. Its qualified profile is text-only (no vision tower) and it needs a native build; not attempted.
- FP8 KV for Flash-Next: keith's runtime rejects it (our exp-033); Son Pham's overlay of four vLLM files (QSA attention and
  its Triton op) makes it work and ran on Kaggle (7 lanes x 103k context). His repo has no license; reimplementing it is
  a kernel-level change we defer until the MTP-0 and caching results are in.

## Evidence against two of our arms (from the web and forum report)

- AVO: NVARC3 (NVIDIA, 16.07): "We tried AVO style ideas and haven't seen them beat our current harness yet" (forum
  thread 737617). exp-061 moves to the end of the queue.
- Higher-resolution images: Makarov's one-change ablations on a Duck harness (thread 743723) put higher-resolution
  board images at or below baseline; Tufa's control run uses 8x anyway. exp-058 runs once before any repeat.
- Makarov's other finding: failures come from an early wrong goal that the model then defends; world-model scaffolds
  were largely ignored. The Polyphony agent and the arXiv world-model papers reach high public scores only with far
  more compute or frontier models; 2607.15439 concludes "capability and reasoning effort dominate".

## Other findings, not acted on

- Scoring (ours, from the arc_agi source): a game scores the best of its runs, but in competition mode the gateway refuses
  a second play of a game and turns a RESET at a level's first action into a no-op, so there is one run per game and a
  level's actions include every failed attempt.
- U4AR/qwen38-arc3-rl: LoRA RL of Qwen3.8-27B on 5 public games; no measured gain; adapters fit the 27B only. Skip.
- Fine-tuning Flash-Next is out of reach on one card (forum: "nearly impossible with 96 GB"); behaviour cloning on the
  model's own winning traces gave another team 11.04 -> 12.97 on the public set (27B-class, not transferable to us).
- Forum claims worth a probe later: the scoring formula and time left in the prompt (+28% for a 27B team); upscale-8
  forks as "the arm that moved the score" (no clean A/B); `environment_info.baseline_actions` may be visible in
  competition mode.
