# Milestone 2 notebooks (read 2026-10-02): Franzen's solution becomes our base

One-line summary: Daniel Franzen open-sourced his Milestone 2 solution (public LB 27.89, Apache-2.0), hundreds of teams
now submit it, and it is about six times our best draw; we adopt it unchanged first (exp-070) and build on it.

## Leaderboard, 2026-10-02 22:2x UTC (downloaded with `kaggle competitions leaderboard --download`)

| Rank | Team | Score |
|---:|---|---:|
| 1 | Tufa Labs | 52.51 |
| 2 | Yi-Chia Chen | 48.07 |
| 3-61 | 59 teams | 30.0-34.8 |
| 158 | Daniel Franzen | 27.89 |
| 226 | keithtyser | 26.71 |
| 379 | Lord Han Solo | 23.84 |
| 429 | Tong Hui Kang | 20.53 |
| 441 | Son Pham & Mark Barney | 18.99 |
| **509** | **ours (Jovian Game Studios)** | **4.70** |

3,605 teams; 438 at 20 or more. The 20-35 band is almost certainly Franzen's notebook and its forks submitted by
many teams: identical notebooks spread 1.5-2.3x between draws (our exp-054: 4.70 / 3.36), so hundreds of draws of
a ~27 notebook produce a tail into the mid-30s. Only Tufa Labs and Yi-Chia Chen are clearly beyond it; neither
has published.

## Franzen's solution (notebook `dfranzen/arc-agi-3-milestone-2-solution`, repo `da-fr/arc-agi-3-solution` @ 10882e3)

Licence: Apache-2.0 (repository LICENSE); the notebook builds on Tufa's Duck harness (MIT); third-party weights keep
their licences. Vendored unmodified at `kaggle/franzen/` with a NOTICE. His WRITEUP.md, summarised:

- **Serving:** Qwen3.8-Flash-Next as Intel's W4A16 AutoRound (more VRAM left for KV than NVFP4) with Albucino's
  quantized MTP draft, on Pennyroyal (John Pezzulli's SGLang fork v2.5.3) plus his patches: low-M BF16 GEMM, a
  speculative-state memory fix, a Mamba prefix-cache patch that keeps only the end-of-prefill checkpoint and refreshes
  it, and checkpoint prefetching. 10 parallel streams, FP8 KV, 136 Ki server context.
- **Context:** 128 Ki harness window (12 Ki output), calibrated token estimates (images by size), and trimming in
  large blocks down to ~58 Ki, so the prefix stays reusable: 93% of prompt tokens served from cache in a demo run.
- **Scheduling:** all games start at once; a priority gate admits 10 (= the server's streams) by
  P = (A + B) * C (immediate level value with an action penalty, a continuation value by levels remaining, a decay
  in actions and tokens spent on the level, calibrated on 725 level attempts); a game keeps its slot across tool calls
  and re-queues only at a context trim. Finished games' time flows to the others automatically.
- **Perception:** animation frames and a timeline as REPL variables, `frame_diff()` object changes, 10x board images
  (his largest image gain), difference images, the game-over frame.
- **Prompts and tools:** explicit game-over diagnosis and level-transfer guidance, UNDO exposed, Python functions
  retained for the whole game, faithful per-action results, 3 Ki tool output with middle truncation.
- **Guards:** stop a batch after an action that leaves the interior board unchanged; block a further action after an
  ineffective one in the same snippet (from level 2); stop after game over or level completion.
- **Turned off:** the structured world-model notes (best result with them off at 128 Ki context); summarisation did
  not help; yielding by 2,048 generated tokens.
- He rented about 150 GPU-hours of single RTX PRO 6000s for experiments in the final phase.

## How it compares with our fork (exp-054, LB 4.70 / 3.36)

| | Ours (exp-054) | Franzen |
|---|---|---|
| Serving | Keith's vLLM, NVFP4, 32 Ki context, 7.75 GiB KV, no prefix cache | Pennyroyal SGLang, W4A16 + MTP draft, 128 Ki, FP8 KV, 93% prefix reuse |
| Concurrency | 28 games in flight, ~5 running | 10 admitted streams, priority gate |
| Allocation | equal time per game (wave-fit) | priority by expected score, decays on stalls |
| Images | 4x | 10x + difference + game-over frames |
| Object changes | P23/P24 report | `frame_diff()` helper |
| History | trim one block per call (P28/P31 built, not run) | blockwise trim to 58 Ki |
| World-model notes | on (P1/P1b fixes) | off |

Our research had found the same direction (lesson 0025: the LB moves with tokens per hour; P28 = hysteresis trim;
the SGLang plan), but he had shipped all of it, tested, plus a scheduler and perception work we did not have.

## Decision

1. **exp-070:** his notebook unchanged as our private kernel (`scripts/build_franzen_nb.py`), pushed right after the
   Oct 3 quota reset. Its Save & Run plays his 10-game demo (25 min each, ~45 min of GPU) and makes the notebook
   submittable; **the owner submits it** (our permission rule), which should move us from 4.70 to the ~27 band.
2. **exp-071:** the same with all 25 public games at 121 min per game (the hidden set's compute per game), as the
   calibration every later arm on this base is read against.
3. Our vLLM-fork arms (exp-059/060/062/063/064, the SGLang launcher) are superseded and leave the queue; the rental
   runner works for any notebook and becomes the way to run his knobs often enough to read them through the noise.
4. Next, on his base: knobs he did not settle (admission slots 10 vs 12, the drain target, guards from level 1,
   image scale, the continuation values), each as repeated runs; and what Tufa Labs and Yi-Chia Chen (48-52) do,
   which nobody has published.
