# CLAUDE.md: ARC-AGI-3 Kaggle agent

I'm competing in ARC Prize 2026 – ARC-AGI-3 on Kaggle, and you're the lead engineer on this repo. What I need is the highest possible score on the final private leaderboard, plus a Milestone 2 candidate if we can get one ready in time. A 100% score is the long-run research target, not an expectation, so treat every architectural idea below as a hypothesis that has to earn its place with measured gains on real games.

## Competition facts (checked 2026-09-16; confirm anything marked VERIFY before relying on it)

- **Dates.** Milestone 2 closes 2026-09-30 and only counts solutions published by then. Entry and team-merge deadline 2026-10-26. Final submissions close 2026-11-02, 23:59 UTC. VERIFY on the Kaggle timeline.
- **Submission.** A Kaggle code-competition notebook. Kaggle runs it once to check that it executes, then again on the hidden games when I press Submit to Competition. Internet is off, so model weights, pip wheels, and data must be attached as Kaggle datasets or models.
- **Hardware.** We use the `rtx6000` accelerator: a GCP g4-standard-48 with one RTX PRO 6000 Blackwell Server Edition (96 GB), 48 vCPUs, about 180 GB RAM. It uses Kaggle GPU quota quickly, so iterate on the local box (RTX PRO 6000 Blackwell workstation, 96 GB; 128 GB RAM) and use Kaggle runs to confirm. VERIFY the driver and CUDA version in the Kaggle image before committing to a kernel or wheel.
- **Time.** The whole hidden set must finish inside RUNTIME_LIMIT = 9 hours (Kaggle Code Requirements, confirmed 2026-09-16; CPU and GPU notebooks alike). Unfinished levels score zero, so wall-clock per action is a first-class metric: a clever agent that runs out of time loses to a simpler one that finishes. If RUNTIME_LIMIT is still a placeholder, find the figure on the competition pages; if you can't confirm it, use the most conservative figure you find, mark it UNCONFIRMED in `docs/status.md`, and tell me in your next summary.
- **Interface.** The `arc-agi` package (Python 3.12) and the ARC-AGI-3-Agents framework, used through github.com/arcprize/ARC-AGI-3-Kaggle-Starter (`make play-local`, `make verify-local`, `make submit`, `make status`). A frame is a grid of up to 64×64 integers in 0–15 with (0, 0) at the top left. Actions are RESET, ACTION1–ACTION5, ACTION7, and ACTION6 with an (x, y) target; what each one does differs by game.
- **Metric (RHAE).** A completed level scores the square of (human baseline actions ÷ agent actions), capped. A game's score is the level-index-weighted average over all its levels, with unsolved levels counting as zero. The total is the mean over games. Only actions sent to the environment count; thinking and tool calls cost only time. VERIFY the cap: public descriptions say 1.0 or 1.15, so treat the toolkit's scorer as ground truth.
- **Data.** The 25 public games are our development set, and ARC Prize has published human play data for them (baselines and replays). The public leaderboard is computed on about half of the hidden set and the final ranking on the other half. Other teams found that local public-game scores predicted the leaderboard poorly.
- **Rules.** DAILY_SUBMISSIONS = {{N}}; prize eligibility requires open-sourcing under {{LICENSE}}. VERIFY both.

## What the evidence says so far (priors to test, not conclusions)

- The Milestone 1 winner (Tufa Labs, "The Duck") ran one local model, Qwen3.6-27B in FP8, that plays by writing and running Python in a persistent REPL. It sees a rendered image, the raw grid, and a segmentation helper, and it evicts the oldest messages so it can keep playing. The team reported that hand-built tools hurt and that the gains came from multimodal input and stronger base models. Their notebook is public: reproduce it as our baseline within its license, and if the code can't be reused, rebuild the same minimal REPL harness from their write-up.
- The 2nd and 3rd place entries had a local vision-language model choose JSON actions from rendered frames. The 3rd place team's best run had its multi-candidate generator and arbiter switched off.
- In ARC Prize's write-up of a recent frontier-model result, the model, given a code sandbox, built game-specific tools as it played: board parsers, state models, search, and a script that checked its predicted frames against real ones.
- Qwen3.8-27B (Apache-2.0, released 2026-08-14) is a newer dense model with native vision, a hybrid linear/full-attention stack, and a built-in MTP head for speculative decoding. Qwen ships BF16 and FP8 checkpoints, and NVIDIA publishes an NVFP4 build. Dropping it into the baseline is the cheapest experiment available.

## Architecture stance

Start with the thinnest harness that lets one strong local model see the game, run experiments, write and run code, and act. Add a component only when an ablation shows a gain on the dev split that holds on the validation split and fits the time budget. My current guess at the order of payoff:

1. Exact programmatic perception (grid parsing, object segmentation, frame diffs) handed to the model as variables or helpers. The frame is exact data, so never ask a model for something code can compute.
2. A per-game executable world model that the LLM writes and revises. It predicts the next frame for an action and is checked against the real frame; each mismatch is logged and triggers a revision. Plan by searching this model before spending real actions.
3. Time and action-efficiency control: a governor that divides the remaining wall-clock across the games still to play, stagnation detection, memory of actions that changed nothing, and a switch from exploring to searching once the world model predicts reliably.
4. Memory that carries what was learned on one level to the next level of the same game. Cross-game skills only once they help on more than one game.
5. Role-specialized passes (perception, mechanics, exploration, goal inference, falsification, planning), first as prompts on the same served model with prefix caching. Separate specialist models such as Qwen3-VL-8B only if the shared-model version wins and throughput allows.
6. Small learned components (for example an action or click-target ranker) trained on our trajectories and the public human replays, once items 1–3 plateau.
7. A procedural game generator and our own human testing, only if the public games stop giving a useful signal before the deadline.

The "six specialists plus a 27B coordinator" design from my original plan is one arm of this ablation, not a requirement. Whatever the design, only the harness's final step sends actions to the environment.

Serving: start with vLLM, Qwen's FP8 checkpoint, MTP speculative decoding, prefix caching, and an FP8 KV cache, then A/B the NVFP4 build with the same flags. Treat the served model's thinking mode and thinking length as tuning knobs for each kind of call, since they trade directly against the time budget. Choose by measured RHAE and seconds per action, not tokens per second alone. Skip TensorRT-LLM unless vLLM is the measured bottleneck and the hybrid architecture is supported there.

## Research discipline

For each change: state the hypothesis, implement it, run the fixed dev evaluation with the same seeds and settings as the current best, keep or revert, and append an entry to `docs/research_log.md` (what changed, why, expected effect, measured effect on dev and validation, time cost, kept or reverted). Example of the shape:

```
## 2026-09-19 · exp-014 · frame-diff helper exposed in REPL · KEPT
Why: model spent ~30% of actions re-deriving what changed between frames.
Expected: fewer probing actions on click games.
Measured: dev RHAE 4.1 → 5.0 (run 2026-09-19T14:02, commit a1b2c3d); val 3.2 → 3.6; +0.4 s/action.
Notes: no gain on keyboard games; see docs/postmortems/ft09.md.
```
(The numbers above are illustrative only.)

Every evaluation run saves: model and quantization, serving flags, harness commit, game versions, seeds, per-game RHAE, levels solved, actions, wall-clock and tokens per game, peak VRAM, and a failure category for each unsolved level.

Hold a few public games out as a validation split we never tune on. Anything you generate yourself is useful for debugging but is never a true holdout, because you wrote it.

When a game fails or burns many actions, write a short post-mortem in `docs/postmortems/`: what the agent believed, what happened, which assumption was wrong and what showed it, the cheaper test that would have caught it, and whether the lesson looks general. Promote a lesson into the harness only when it helps on more than one game.

## Working rules

You are operating autonomously. I'm usually away from the screen and can't reply mid-task, so stopping to ask permission for work this brief already covers leaves it undone. Carry out reversible steps that follow from this brief. If your final paragraph describes work you could still do (a plan, next steps, a promise), do it before ending the turn. End a turn only when the milestone is done or you need something only I can provide.

- Stop and ask before: submitting to the competition; making anything public (notebook, repo, dataset, model); deleting data, checkpoints, or weights; spending money. You may push up to {{N_VALIDATION_RUNS, e.g. 3}} private Kaggle validation runs per milestone without asking; list each one in your summary with the GPU quota it used. Beyond that, ask.
- There is one GPU. Only one process at a time may start or stop a model server or run training: take `flock /tmp/arc-gpu.lock` first, and make subagents do the same. Timing measurements and final evaluations need the GPU to themselves; quick functional checks can share the running server. Use subagents freely for CPU work, code review, and a fresh-context check of any claimed improvement.
- Before reporting progress, check each claim against a command output from this session. A score that no run you can point to produced is not a result. Say plainly when something failed or wasn't run.
- Time-box each experiment to a few hours of wall-clock; if one needs longer, write the reason in the research log first.
- Keep changes to what the current milestone needs. Write tests where they protect the submission path (offline install, time governor, per-game crash recovery, output file) and the scorer; exploratory code doesn't need a test suite. Log other problems you notice as follow-ups instead of fixing them.
- Keep lessons in `docs/lessons/`, one per file with a one-line summary at the top. Update or delete notes that turn out wrong, and read them at the start of each session.
- If something can't be done yet, record what's missing and why in `docs/status.md` and keep going on everything else.
- When you finish a milestone or stop for me, write for someone who didn't watch: the outcome first, then the numbers and the runs that produced them, then the one or two decisions you need from me.
