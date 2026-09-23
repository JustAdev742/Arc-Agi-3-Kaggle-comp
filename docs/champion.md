# Champion record (immutable comparison point; written 2026-09-17)

The measured best system: one Qwen3.8-27B-FP8 REPL agent on the exp-011 harness. Every experiment compares against
this record; a challenger replaces it only with a fixed dev evaluation above the noise band and a validation score
that is not worse. `arc3/presets.py:CHAMPION` reproduces this configuration at any later commit (every knob added
since is off in it) and is what the submission notebook runs by default.

## Identity

| Item | Value |
|---|---|
| Runs | `runs/kaggle-repl-dev-011` (1.229, 9 levels), `runs/kaggle-repl-dev-011b` (1.372, 9), `runs/kaggle-repl-dev-011c` (0.775, 6): the same notebook three times; `runs/kaggle-repl-val-011` (val 0.794, 1 level) |
| Harness commit | 756a87e (exp-011 bundle; the notebooks were built from it; the run records say `harness_commit: unknown` because the kernel has no git) |
| Model / checkpoint | Qwen3.8-27B-FP8, official `Qwen/Qwen3.8-27B-FP8` @ 017b9c7a, Kaggle dataset `saltb0x/qwen3-8-27b-fp8` (byte-for-byte verified 2026-09-16) |
| Quantization | FP8 weights (vendor), FP8 KV cache |
| Serving | vLLM 0.27.1 (wheelhouse `saltb0x/arc3-vllm-wheelhouse-v0271-cu129`), `arc3.serve.build_vllm_command`: `--max-model-len 32768 --gpu-memory-utilization 0.9 --max-num-seqs 32 --limit-mm-per-prompt {"image": 16} --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_coder --attention-backend TRITON_ATTN --reasoning-parser qwen3 --kv-cache-dtype fp8 --speculative-config {"method": "mtp", "num_speculative_tokens": 2, "attention_backend": "TRITON_ATTN"}`; env `VLLM_USE_FLASHINFER_SAMPLER=0`. The log shows TRITON_ATTN for the language model and FLASH_ATTN for the vision encoder only; MTP draft on TRITON_ATTN |
| Reasoning | thinking on, `reasoning_effort` low (chat-template kwarg), adaptive policy (medium for one turn when stagnant), `max_output_tokens` 3072, `context_tokens` 32768, temperature 0.6, top_p 0.95, one board image per turn (scale 4) |
| Agent config (as recorded) | `{"context_tokens": 32768, "reasoning_effort": "low", "max_output_tokens": 3072}` plus the harness defaults of 756a87e = `arc3/presets.py:CHAMPION` |
| Evaluation | dev split (19 games), seed 0, 1200 s per game, 2000 actions per game, 8 concurrent games, one shared vLLM server on the Kaggle RTX PRO 6000 |
| Environment | `arc-agi` 0.9.9 / arcengine 0.9.3, competition `environment_files` (game versions in the table below), Python 3.12.13, Kaggle image driver 580.159.04, torch 2.13 after the wheelhouse install |
| Scorer | `arc_agi/scorecard.py` (arc-agi 0.9.9) through `arc3.scoring` (parity-tested in `tests/test_scoring.py`); score = percent of human-level RHAE, level-index weighted, unsolved = 0 |

## Scores

| Measure | Value |
|---|---|
| dev (three runs of the identical notebook) | 1.229 / 1.372 / 0.775; pair mean 1.30, three-run mean 1.13; run-to-run band about ±0.3 and ±2 levels (exp-009/009b, exp-011 family) |
| val (six held-out games, never tuned on) | 0.794, 1 of 41 levels (sp80 L1), `runs/kaggle-repl-val-011` |
| levels | 9 / 142 (011), 9 / 142 (011b), 6 / 142 (011c); games with a level: 8, 9, 6 |
| actions | 1180 / 938 / 1012 per run; median 27 s per action (011b), model p50 latency 27 s per call |
| wall-clock | 3557-3606 s per run for 19 games at 8 concurrent (every game uses its full 1200 s: the failure tag is always `timeout`); kernel setup 461 s (vLLM install + start) |
| tokens (011b) | 11.7 M prompt + completion over 19 games; about 40 model calls per game |
| VRAM | vLLM reserves 0.9 of 97,887 MiB (about 86 GiB): 28.95 GiB weights, 53.22 GiB KV cache (32 sequences at 32k, "maximum concurrency 32"), the rest activations and CUDA graphs (kernel `arc3-eval-dev-f` vllm.log; diag v5) |

## Per-game results (levels completed; score)

| game | version | 011 | 011b | 011c | val-011 |
|---|---|---|---|---|---|
| ar25 | ar25-0c556536 | L1 2.78 | L2 8.33 | L1 2.78 | |
| bp35 | bp35-0a0ad940 | 0 | 0 | 0 | |
| cd82 | cd82-fb555c5d | 0 | 0 | 0 | |
| dc22 | dc22-fdcac232 | 0 | 0 | 0 | |
| ft09 | ft09-0d8bbf25 | 0 | 0 | 0 | |
| ka59 | ka59-38d34dbb | 0 | 0 | 0 | |
| lp85 | lp85-305b61c3 | L1 2.78 | L1 2.78 | L1 2.78 | |
| ls20 | ls20-9607627b | 0 | 0 | 0 | |
| m0r0 | m0r0-492f87ba | L1 4.76 | L1 4.76 | 0 | |
| re86 | re86-8af5384d | 0 | 0 | 0 | |
| s5i5 | s5i5-18d95033 | L1 1.42 | L1 2.78 | 0 | |
| sb26 | sb26-7fbdac44 | L2 6.55 | L1 0.37 | L1 2.78 | |
| sk48 | sk48-d8078629 | 0 | 0 | 0 | |
| su15 | su15-1944f8ab | L1 1.59 | L1 2.03 | L1 2.22 | |
| tn36 | tn36-ef4dde99 | 0 | L1 3.57 | L1 3.57 | |
| tr87 | tr87-cd924810 | 0 | 0 | 0 | |
| tu93 | tu93-0768757b | L1 0.74 | 0 | 0 | |
| vc33 | vc33-5430563c | L1 2.73 | L1 1.45 | L1 0.61 | |
| wa30 | wa30-ee6fef47 | 0 | 0 | 0 | |
| cn04, g50t, lf52, r11l, sc25 | val | | | | 0 |
| sp80 | sp80-589a99af | | | | L1 4.76 |

## Known weaknesses (from the 18-run matrix, `scripts/research_status.py --matrix`, and the post-mortems)

- Five dev games never solved in 18 runs: dc22, ft09, sk48, tr87, wa30 (goal or mechanics never inferred;
  post-mortems for dc22, ka59). Level 2 reached in only a handful of game-runs; level 3 never.
- Every game runs to its time limit: 1200 s buys about 40 calls of 27 s. exp-017 (3600 s per game) reached 10
  levels with 4634 actions, so time is not the first limit; actions per level are (tn36 796, wa30 498, tr87 355).
- 293 of exp-017's 4634 actions re-sent a (frame, action) pair already known to change nothing (no no-op memory).
- The goal-predicate library misses whole classes of win conditions (ls20 from level 3 on: shape matching).
- Generalization: val 0.794 from one level; the six val games are as hard as the unsolved dev games.

## How to reproduce the control

Build with the preset (the harness at any commit, every later knob off) and the fixed evaluation:

```bash
.venv/bin/python scripts/build_eval_notebook.py --agent repl --split dev --time-per-game 1200 --workers 8 \
    --preset champion --config '{}' --slug arc3-eval-dev-p --run-name kaggle-repl-dev-024 --note "exp-024 control" --out <folder>
.venv/bin/python scripts/push_eval.py <folder>
```

A new champion needs three runs a side or a difference well outside ±0.3 RHAE / ±2 levels, plus a val run.

## Changes on the champion's code path since the record

- 2026-09-23: the REPL agent archives a completed level's winning frame with `perception.terminal_layer` instead of
  `layers[0]` (lesson 0017). A bug fix, not a knob: on animated wins (12 of 43 collected dev levels: cd82, tu93,
  sk48, su15) the level-completion notice's "win conditions consistent with every completed level" were computed
  from the board before the winning move. The submitted notebook (v3, harness 8f3af9e) predates it; the exp-024
  control at HEAD is the first model run that includes it.
