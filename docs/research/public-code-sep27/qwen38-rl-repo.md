# U4AR/qwen38-arc3-rl: is it worth an experiment?

**Verdict: no GPU experiment.** None of their ARC-trained adapters has shown a gain, the one they published was never evaluated, and every adapter fits only dense Qwen3.8-27B, not our Flash-Next base.

I cloned the repo to /home/user/u4ar/qwen38-arc3-rl (master 655309a from 2026-09-24, 19 commits, plus the `adapters` branch at 08704e6).

## License
- No LICENSE file, and the README grants none for U4AR's own code (`arc3rl/`, `scripts/`). That means all rights reserved: we can borrow ideas but not copy their code.
- The vendored Duck harness is MIT. The starting Terse LoRA is Apache-2.0.

## Method and data
- **Method:** GRPO-style LoRA RL with a clipped ratio against the vLLM logprobs. BF16 Qwen3.8-27B on 2x H100, vLLM 0.17.2rc1 with MTP-2. They start from Shockem's Terse-Coder LoRA (r=16 on attention, MLP and GatedDeltaNet projections).
- **Reward per level** (`arc3rl/reward.py`): `k*(0.5+min(1.15,(h/a)^2)+0.25*max(0,1-tok/60k))`.
- **Data: yes, public games.** They trained on 5 of the 25 public games: ar25, cd82, ft09, lp85 and ls20 (`configs/split.json`). They picked these after the baselines, as the easiest 5, so any gain on them is contaminated. The other 20 were held out, but only 7 were ever played, at 2 episodes each.

## Every measured result
Scores are final_score on a 0–100 scale. There is no Kaggle leaderboard result anywhere, and no held-out evaluation of any trained adapter.

**Baselines** (`results/REPORT.md`, `results/eval_baseline*.json`). Settings: 60k generated tokens per episode, no per-turn images, stopped at 24 of 100 episodes (12 games x 2 episodes).

| Policy | All games | Train-5 | Held-out-7 |
|---|---|---|---|
| Base | 1.62 (0.42 levels/episode) | 3.22 | 0.47 |
| Base + Terse | 1.21 (0.33 levels/episode) | 2.92 | **0.00** |

**Run 1** (same settings): one update. The Terse rollouts on train-5 reached 0.75 levels/episode with a score of 2.78. The resulting adapter, `run1-noimage-iter001`, was **never evaluated**.

**Run 3** (`results/run3-collapsed/train_history.jsonl`). Settings: an image every turn, 150k tokens per episode, lr 1e-5, 40 episodes per iteration on train-5.

| Iteration | Levels/episode | Score |
|---|---|---|
| iter1 (Terse) | 1.725 | 8.99 |
| iter2 | 1.575 | 6.75 |
| iter3 | **0** (collapsed: KL 6.43) | 0 |

**After their collapse fix:** the only record is commit message 655309a, "no significant difference, p=0.71" against Terse on the training games. The numbers were not committed.

**Cost:** each iteration took 8,760–14,415 s of rollout plus about 2,700 s of training on 2x H100.

## Published weights
- **GitHub LFS, `adapters` branch:**
  - `terse-hf/` (233.6 MB): the Terse LoRA with keys renamed.
  - `run1-noimage-iter001/` (467.1 MB, probably fp32). No license stated.
- **No run-3 adapter was published.** `scripts/publish_if_improved.sh` only publishes one that beats Terse, and none did.
- **Nothing on Hugging Face or Kaggle.** huggingface.co/U4AR returns 404 and the HF API search is empty. Shockem's own HF page for the Terse LoRA is Apache-2.0 and doesn't mention ARC.

## Could we serve it?
- **Flash-Next NVFP4: no.** It is a different MoE/QSA model, so the adapter's shapes don't match. Moving to 27B would cost more than any plausible gain: on our public-25 harvest, 27B-FP8 (anim) scored 3.79 against 5.16–11.02 for Flash-Next.
- **27B NVFP4: plausible.** Shockem ran vLLM 0.28 with NVFP4 and MTP-3 on a 5060 Ti:
  - About 10% slower single-stream (54.1 to 48.9 tok/s).
  - MTP acceptance 0.43 with or without the adapter.
  - Untested batched, and untested on our vLLM 0.27.1.
- **27B FP8:** untested.
- **Hybrid linear attention:** U4AR verified that vLLM applies the linear_attn LoRA. Mismatched key names silently attach nothing, so a logprob check with and without the adapter is required.
- **Merging instead:** only 31–61% of the delta survives a bf16 merge, so it would have to be served at runtime.

## Assessment
Skip it:
1. The measured effect is zero or negative, and Terse scored 0 on the held-out games.
2. It targets the wrong base for our best model.
3. Reproducing it needs about 4 h per iteration on twice our GPU. That's impossible before M2 (2026-09-30) and a poor bet before the 11-02 deadline.

Three ideas are worth borrowing, without their code:
- The load-independent `EPISODE_MAX_GENERATED_TOKENS` budget.
- The context-edge margin in `scripts/patch_vllm.py`: MTP drafts past max_model_len caused device-side asserts, so check our logs for the same crash.
- Replaying reasoning verbatim to get prefix-cache hits. Their own run 3 reverted this to match the Kaggle Duck.

Sources: https://github.com/U4AR/qwen38-arc3-rl ; https://huggingface.co/Shockem/Qwen3.8-27b-Terse-Coder-LoRA
