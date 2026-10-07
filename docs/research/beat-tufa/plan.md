# Plan to beat Tufa Labs (synthesis of the four research reports, 2026-10-02 23:1x UTC)

One-line summary: to finish a few points above Tufa on the private half we need a true mean near 55, about 2.1x a
Franzen draw (25.8 +/- 3.9); capacity (more decode at long context), fewer tokens per action and harness fixes can
plausibly compound to x1.4-1.9 on one card (about 36-49), and the rest of the gap most likely needs what Tufa
appears to have: a post-trained model.

Sources: intel.md (what is known about Tufa, Yi-Chia Chen and others), franzen-run-analysis.md (where Franzen's run
loses, measured), serving.md (decode capacity on his stack, measured from his serve.log), new-methods.md (harness
methods). Gains below are estimates unless marked measured; score is assumed to grow with useful tokens to the
power ~0.75 (franzen-run-analysis.md, from his score-over-time curve and our exp-054).

## Where we stand

- Franzen's system is decode-bound (measured): 86% of game time is generation at ~74 tok/s per stream, 10 streams,
  ~765 tok/s with all 10 running, 94% of prompt tokens from the prefix cache, MTP accept length 2.66 of 4. Prefill
  and queueing are negligible. 86% of generated tokens are reasoning; 47% of calls take an action.
- Its losses are reach, not efficiency (measured): human-level efficiency on every solved level adds 0.7 points of
  36.6 on his demo; unreached levels cost 62.7.
- Tufa: 18.8 for 13 days, 27.3 on Sep 26 (one Franzen-level draw), then one step to 45.3 on Sep 28, then 50.7 and
  52.5; true mean ~46-50. Public signals: an RL stack forked on GitHub in early September, a B200 evaluation cluster,
  LoRA serving in their framework since June, a team of RL / self-distillation people; they publish only at the end.
- Yi-Chia Chen: improved every day (28.3 -> 48.1); earlier work used a self-trained DFlash drafter and an
  on-policy-distilled model on this GPU.

## Levers, ranked by expected gain per GPU-hour

| # | Lever | Mechanism | Est. score factor | Cost to test | Source |
|---|---|---|---|---|---|
| 1 | **Online MXFP8 + 12-13 streams** (flags) | Halves BF16 weight reads; freed VRAM pays for streams | x1.11-1.18 | ~0.7 GPU-h gate run + 2 runs | serving.md arm 1 |
| 2 | **Sandbox fixes** (timeout keeps retained functions, tool timeout 90 s, `time` module, partial output) | r11l lost 13 helpers to one timeout | +0.2-0.5 pt | CPU tests + any run | both |
| 3 | **Budget meter + EXPOSE_RESET** | 3 of 5 deaths were budget deaths; sc25 burned ~9 actions to force a reset | +0.3-1.0 pt | 1 run | new-methods M2, analysis lever 3 |
| 4 | **REAP-448 checkpoint + 14-16 streams** | ~7-20 GiB more KV; Son Pham: 21.4 -> 33.9 on his harness (7 -> 16 lanes) | x1.15-1.35 | gate run + 2 runs; checkpoint work | serving arm 3, intel hyp. 1 |
| 5 | **Fewer tokens per action** (Swift-1.5's shorter thinking, or a turn cap / stall cut-off) | 86% of tokens are reasoning; Swift claims -63% thinking at <1% loss | x1.0-1.3 (quality risk) | 2 runs each; Swift licence check first | intel hyp. 3 |
| 6 | **Harness search service + guarded plan execution** | self-written planners time out or run blind | +0.5-1.5 pt | 1-2 runs + CPU bed | new-methods M3 |
| 7 | **Relaxed MTP acceptance / ARC-trained draft or hot-token map** | accept 2.66 -> ~3 | x1.05-1.12 | gate run; lossy, needs score runs | serving arm 4, intel hyp. 4 |
| 8 | **Exact ledger re-pinned at trims; end-of-run push; guards from level 1** | vc33 re-derived a won level after a trim; 3 of 9 cut-off levels had a verified plan | +0.3-0.8 pt | 1 run | both |
| 9 | **Post-trained policy** (SFT/RL on our own and public traces) | Tufa's most likely edge | x1.3-1.8 if it works | multi-GPU training compute (owner's money) | intel hyp. 2 |

Compounding 1-8 (if each holds and they stack): roughly x1.4-1.9 over Franzen, i.e. ~36-49 mean. That is a
competitive score but below Tufa's mean; lever 9 is where the remaining gap most plausibly lives.

## This week (Kaggle quota 30 GPU-h from Oct 3 00:00 UTC; the owner submits once a day)

1. **exp-070**: Franzen unchanged (~0.75 h) -> the owner submits it (~26 expected).
2. **exp-071**: 25 games x 121 min (~2.3 h): calibration; its serve.log gives our own tok/s, accept length and KV
   pool pressure under contested slots.
3. **Serving gate runs** (25 games x 25 min, ~0.7 h each, read tok/s, pool, errors, not score): unchanged, then
   lever 1 (MXFP8 + 12 streams), then lever 4 with the public REAP-448 NVFP4 build (needs the NVFP4 serving
   settings Franzen used before his W4A16 switch).
4. **Harness arm** (levers 2+3, then 6/8 as built on the CPU bed): 25 x 121 min, against exp-071.
5. The best combination gets the daily submission; then repeats of the two best configurations (rentals when the
   owner's account is ready) to choose the final two submissions.

## Decisions only the owner can make

- **Rentals** (when your vast.ai account is ready): repeats are what make the 1.5-2.3x draw noise readable; ~$4-5 per
  run (rental-runner.md).
- **Training (lever 9)**: a post-train of Flash-Next needs a multi-GPU node (8x H200/B200 class, typically $20-40 per
  hour, for one to several days, i.e. several hundred to a few thousand dollars) plus engineering time; before any
  of that, a cheap probe: run Tong Hui Kang's public LoRA fine-tune (Apache-2.0) through Franzen's harness once to
  see whether a fine-tune moves our numbers at all. I will not spend on training without your explicit OK and
  budget.
- **Swift-1.5**: its custom licence must allow prize use before it can be submitted.

Verified on the way (intel.md): 1 submission per day (forum thread 705405) - the CLAUDE.md "DAILY_SUBMISSIONS" item.

## Measured 2026-10-07: what extra tokens are worth at the full budget (scripts/score_over_time.py)

Franzen's v3 run (100 game runs = 4 passes of the public 25, 484 min, the hidden set's compute per game; our scorer
reproduces his 46.49) cut at earlier common wall-clock times: 0.5 of the run 27.78, 0.7 35.37, 0.8 39.01, 0.9 43.70,
0.95 45.72, 1.0 46.49. Local elasticity d ln(score)/d ln(time) over the last 10-30% of the run: **0.59-0.79**. Extra
decode throughput is (approximately) extra time, so +20% tok/s is worth roughly +12-16% score, i.e. about +3.5-4.5
LB points on a ~28.5 mean. (Caveats: the left derivative of one run's curve, shaped near the end by the tail fade;
the right derivative is probably somewhat smaller.) Truncating each game at a fraction of its *own* tokens instead
gives a spurious jump at 1.0 (won games always lose their last, heaviest level), so it is not used.
This ranks serving capacity (levers 1, 4, 7) first among the levers we can pull without training.

## Status 2026-10-07 23:41 UTC (after the first serving gates)

- **Lever 1 (online MXFP8 + more streams): rejected.** It frees memory (pool 1.31 M) but decodes ~24% slower per step
  than Franzen's BF16 path at equal streams (596 vs 783 tok/s at 10 running); 12 streams lose 17% output overall.
- **Lever 4 (REAP-448 + more streams): adopted for full-length testing.** Built without a new checkpoint
  (docs/research/beat-tufa/reap-at-load.md): +14% output tok/s at 14 streams in the same 25-min gate, accept length
  unchanged. By the measured elasticity that is about +8-11% score. exp-073 (121 min/game) checks it at full length.
- In flight: acceptance 0.5 on top (exp-072g), 16 games on 14 server slots (exp-072h), 16 streams at mem fraction
  0.975 (exp-072i). Submission candidate built: exp-074s (D' + REAP-448 + 14 streams + input-path fallback).
- Kaggle infrastructure found on the way: the latest image moved to Python 3.13 (pin the image, lesson 0029) and
  inputs are mounted in two layouts (resolve both, lesson 0030); both would have zeroed a submission.
