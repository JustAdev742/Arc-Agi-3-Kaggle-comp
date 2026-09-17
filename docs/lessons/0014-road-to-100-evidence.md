# The gap to 100 is first the model, then calls per game, then harness structure; measure candidates by RHAE per GPU hour at the real concurrency

From the 2026-09-17 evidence survey (`docs/research/road-to-100.md`); numbers are from the sources named there.

- **Nobody is at 100 on hidden games.** ARC Prize's verified leaderboard (2026-09-14): 62.7 for the best frontier
  system with its own harness and thousands of calls per game; the Kaggle public leader is 18.81. The systems that
  report 100.00 (Tycho with Opus 5 / GPT-5.6 Sol) do so on the public set, at 600-1,800 calls per game.
- **Harness structure is worth about +9 at the frontier** (Tycho: 79 direct -> 88 orchestrator with one model). A
  27B model with the same ingredients in reduced form scores about 1. So the model and the operating point (calls
  per game, output tokens per call, reasoning effort) come before more harness; Rodionov (2607.15439) found the same.
- **The budget on Kaggle is the binding constraint.** 9 hours on one GPU for all hidden games concurrently: at the
  measured 308 tok/s aggregate, 30 games get about 160 calls of 2k tokens each; Tycho's regime is an order of
  magnitude above. Throughput at the real concurrency, not single-stream tok/s, is the number to optimise; an MoE
  model with few active parameters is the largest lever on it.
- **Exploration is cheap and general** (2605.25931: 24 of 25 public games solvable by exploration alone) and
  **objective inference is the residual failure** everywhere (Tycho, 2605.05138's "premature commitment", our
  post-mortems). Harness-owned probing and goal-hypothesis falsification (a goal that comes true without completing
  the level is not the goal) address both without a model call.
- **SM120 serving facts (vLLM 0.27.1, offline):** FlashInfer paths that JIT or download cubins fail; Triton attention
  works; MXFP4 MoE falls back to Marlin unless `--moe-backend` says otherwise; NVFP4 GEMM has a Marlin env; parsers
  are `openai` + `openai_gptoss` for gpt-oss, `qwen3_coder` + the shipped `super_v3` plugin for Nemotron 3 Super;
  harmony models read `reasoning_effort` from the request, Qwen from chat-template kwargs.
- **Licenses matter for the prize:** Nemotron's Open Model License is not an OSI license; the owner decides before
  it is used in a submission.
