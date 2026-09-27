# What the 20+ teams have published (web and forum sweep, 2026-09-27)

## Bottom line
None of the six teams above 16 has published its current method. Tufa Labs and NVARC3 have both said they will not open-source for Milestone 2. Tong Hui Kang says he will publish only if he is 1st, and Lord Han Solo only if he is in the top 3 among teams willing to share. Daniel Franzen has an empty public GitHub repo `da-fr/arc-agi-3-solution`: it was updated 2026-09-25 and my clone found no commits. So the only direct evidence is the leaderboard history and forum posts, most of them from teams scoring under 10. Everything below that goes past those sources is labelled as interpretation.

## Leaderboard history (fact)
Source: Tong Hui Kang's minute-by-minute monitor, https://arc3.huikang.dev/leaderboard (JSON at tonghuikang--arc3-leaderboard-monitor-get-history.modal.run), pulled 2026-09-27. It records each team's best public score and the runtime of each submission.

| Team (subs) | Score steps |
|---|---|
| Tufa Labs (147) | 4.71 on Aug 30 → 11.04 Sep 6 → 18.81 Sep 13 → 27.29 Sep 26 |
| Daniel Franzen (84) | 7.63 Sep 5 → 11.59 Sep 16 → 13.12 → 16.68 Sep 24 → 21.01 Sep 26 |
| Lord Han Solo (75) | 4.99 Aug 24 → 8.44 Sep 13 → 15.60 Sep 19 → 19.40 Sep 22 → 20.80 |
| Tong Hui Kang (85) | 4.45 Aug 31 → 8.72 Sep 16 → 10.78 → 15.02 Sep 24 → 20.53 Sep 25 |
| Yi-Chia Chen (13) | 4.79 Sep 17 → 12.88 Sep 19 → 15.98 → 18.80 Sep 25 |
| NVARC3 (22) | 3.32 Sep 6 → 8.40 Sep 11 → 11.04 → 16.07 Sep 19 |

- The top teams' submissions run 530–555 min, so they use essentially the full 9 h. Yi-Chia Chen has several runs of 93–300 min.
- Of 3,374 teams, 9 are at 10 or above and 4 are at 20 or above.
- Interpretation: six teams, climbing on their own schedules, went from about 5 to 16–27 between Sep 6 and Sep 26. That points to many small, iterated gains rather than one shared leak. CPMP says "we did not see any leaked solution" (discussion 739186). Each public score is also the best of 13–147 noisy draws; identical code has produced a 2.3x spread on the leaderboard (731522).

## Source by source

**Tufa Labs.** Milestone 1 write-up: kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133. Blog: tufalabs.ai/research/duck-harness.
- Setup: Qwen3.6-27B FP8 served by vLLM. The model uses a REPL tool that is reset on every call, with a 30 s time limit and 4,096 characters of output.
- Context: a 4x-upscaled image goes in every turn. A world-model note is carried from turn to turn. The UNDO action is hidden from the model.
- Eviction: the oldest turns are dropped to keep the input near 32k tokens, with a 64k cap. They say they "do not optimally use prefix caching."
- Scores: 1.21 on the leaderboard, 1.60 ± 0.45 on the public 25 (20 tries per game). Code is MIT; the Kaggle notebook is Apache-2.0 (Cottaar, 717133).
- Milestone 2 (742801): not publishing on Sep 30, "worry it may not be possible to beat the best of 4000 copies"; the final solution comes after the competition. Nothing newer on their research page. X returned HTTP 402. The MLST interview is video only; I could not read its content.
- Interpretation: their jump from 4.71 to 11.04 on Sep 6 fits a switch to Qwen3.8-Flash-Next (125B MoE, 6B active, weights out Aug 26–27). xz: "a lot of the improvements simply came from squeezing more tokens out of the hardware and allocating them properly." That is a guess by another team, not Tufa's statement.

**Daniel Franzen.** No ARC-AGI-3 write-up or Kaggle kernel; the repo above is empty.

**Tong Hui Kang.** No ARC-AGI-3 code. He won the Nemotron progress prize with LoRA SFT (github.com/tonghuikang/nemotron). At a May hackathon, action-prediction models trained on ARC-AGI-3 play showed "no evidence of outperforming random play" (blog.huikang.dev/2026/05/31/autoresearch-hackathon.html).

**Lord Han Solo, Yi-Chia Chen.** Nothing published beyond the sharing statements.

**NVARC3** (CPMP, Darragh, rfbr; NVIDIA Kaggle grandmasters). "We tried AVO style ideas and haven't seen them beat our current harness yet" (737617). They will open-source at the end if they place (742801).

**NVIDIA AVO** (developer.nvidia.com blog, Aug 21). Claude Opus 5 inside a long-horizon agent with persistent memory and a stagnation supervisor: 100 on the public 25 with 6,624 actions. No hidden-set score, no code.

**ARC Prize Milestone 1 blog** (arcprize.org/blog/arc-prize-2026-milestone-1).
- 2nd, Reki: Gemma-4-31B returning JSON of 1–4 actions per step, reflection memory every ~10 steps, a click heuristic, and a "dead-signature" rule for objects that never react.
- 3rd, forge: the same design with a candidate generator and arbiter, but the best run switched that machinery off.

**ARC Prize, GPT-6 Astra** (arcprize.org/blog/astra). 62.7 semi-private with the standard harness. A "Provider Adapter" harness that "preserves opaque reasoning state between requests" scored 99.9.

**Polyphony Agent** (github.com/Mininglamp-AI/polyphony-arc-3, MIT).
- Qwen3.6-27B at BF16 across 8 GPUs, 262k context, 5 games in parallel, 4 h per game.
- The model writes a per-game Python policy; a policy is accepted only if it reproduces the observed frames, then the agent plans through it.
- 19.8 on the public 25. That compute is far beyond Kaggle's.

**Retrodict** (github.com/ryanbbrown/retrodict, no LICENSE file). gpt-5.6-sol at max effort. Hypotheses are checked against recorded history, and planned actions carry predicted frames. 99.86 public for $654.

**Papers**
- arXiv 2605.05138 (Rodionov, MIT code): executable world models; GPT-5.5 scores 58.12 public.
- arXiv 2607.15439: public RHAE by variant.

  | Model, effort | Textual | Executable | Simplification | Verification |
  |---|---|---|---|---|
  | GPT-5.4 high | 34.2 | 33.5 | 30.6 | 39.2 |
  | GPT-5.5 xhigh | 72.5 | 69.7 | 73.1 | 74.8 |

  "Capability and reasoning effort dominate." Simplification hurt the weakest setting. No open-weight models were tested.
- arXiv 2605.25931 (AERA): low value.
- arXiv 2607.20709 (NVIDIA NOOA, Apache-2.0): 85.1 public on the community leaderboard.

**Forum, from teams below the top six**
- Makarov (743723), controlled single-layer ablations on a Duck harness:
  - At or below baseline: consensus sampling, a reasoning-discipline block, higher-resolution images, and an executable world model with replay and search. The model "largely ignored" the world model.
  - Stuck and successful runs split within the first few moves of a level, usually after the decisive action had already been tried. The failure is an early goal judgement that the model then defends.
- gedouluhui (743060), hidden-set scores:
  - Qwen3.8-27B alone: 1.12.
  - Plus scoring formula, time-remaining line and click candidates from connected components: 1.43.
  - Rejected: 64k context scored 1.20, with throughput dropping from 283 to 195 tok/s. Temperature 0.3 scored 0.95; games with progress fell from 12 to 8.
- STaR LoRA on the Duck's own winning trajectories (739047): 1.25 → 1.94 on the leaderboard with Qwen3.6. "Naively scaling the data hurt."
- 743319: an adapter that down-weighted reasoning tokens to 0.1 collapsed into turns with no tool call (1,474 of 4,767). Weighting tool tokens at 1.0 and reasoning at 0.3 avoided it.
- Serving and control:
  - Concurrency above 16 saturates (739938).
  - Flash-Next cannot be trained on the RTX 6000; it needs an H200 node (742835).
  - Animation access adds 17% tokens per action, and every run hits the wall-clock cap (734369).
- Public-25 is a poor proxy: 22.26 → 5.37 and 17.34 → 6.91 on the leaderboard (732854).

## Ideas that transfer to one offline GPU
Ranked by evidence ÷ cost. All are hypotheses to A/B.
1. **Tokens per wall-clock hour.** Every top run uses the full 9 h and every Duck run hits its cap. Candidates:
   - Evict in large blocks, or compact, so the vLLM prefix cache survives; Tufa admits it is not using prefix caching well.
   - Sweep concurrency around 16.
   - Measure calls per game at 110-game concurrency.
2. **Move time to games that are progressing, with a stop rule for stuck ones.** One team's version was worse on the hidden set (740812), so check it on the harder games too.
3. **Keep past reasoning in context** (the Astra adapter result; our lesson 0022).
4. **STaR LoRA on the 27B from our own winning trajectories.** Use tool tokens at 1.0 and reasoning at 0.3, and keep the training set small and curated. Only this route is trainable on 96 GB, and it has one +55% leaderboard datapoint. It must beat untuned Flash-Next to matter.
5. **Two cheap prompt lines:** the true scoring formula and time remaining.
6. **Avoid:** temperature below 0.6, 64k context on the 27B, archetype playbooks, always-on multi-role pipelines, and world-model scaffolds the model is not seen using in its tool calls.
7. **Predict-and-verify plans (Retrodict, Polyphony).** Strong with frontier models, and one of our own ranked hypotheses; so far no evidence with a 27B.

## Decisions for you
1. Lord Han Solo, Franzen or Tong Hui Kang may publish a 20+ notebook around Sep 30. Do we hold GPU quota and a slot to reproduce one on Oct 1?
2. Invest the remaining month in LoRA on the 27B, or stay on untuned Flash-Next and work on throughput and time allocation?
