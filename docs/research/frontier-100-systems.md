# The systems that report 100 on ARC-AGI-3, and what we can take from them (2026-09-23)

The owner asked for a way to 100 percent, pointing at NVIDIA and Tycho. This note records what those systems are,
why they do not run inside the Kaggle competition, and which of their structures we port to our offline Duck fork.

## What reports 100, and on what

| System | Model | Set | Score | What it costs / needs |
|---|---|---|---|---|
| NVIDIA AVO (NVIDIA blog, Aug 2026) | Claude Opus 5 (API) | 25 public games, 183 levels | 100.00 RHAE, 6,624 actions | not disclosed; no Kaggle/offline claim |
| VISTA (MIT; vista-research.github.io, code MIT) | Claude Opus 5 / GPT-5.6 Sol (API) | 25 public games | 100.00 / 98.27; 7,542 actions (56% fewer than humans) | not disclosed; effort max/xhigh |
| Tycho (NIMI Research, arXiv 2607.28287, Apache-2.0) | Opus 5 / GPT-5.6 Sol (API) | 25 public games | 100.00 | 600-1,800 calls per game, 24k output tokens per call, 2.99k-5.78k USD per run (lesson 0013) |
| Frontier models on ARC Prize's own leaderboard (semi-private set) | GPT-6 Astra / Claude Opus 5 | semi-private | 62.7 / 30.2 | the official evaluation, not the public games |
| Open weights, best documented | Qwen 3.6 (Polyphony, self-reported) / our Duck base | public games | 19.8 / about 7 | the only class allowed on Kaggle |
| Kaggle ARC-AGI-3 leaderboard today | offline, one RTX PRO 6000, 9 h for 110 hidden games | hidden set | top 19.40 | this competition |

Every 100 is on the public games with a frontier model called over the internet. The Kaggle rerun has no internet,
one GPU and nine hours for 110 unseen games; those models cannot run in it, and on unseen games they score 30-63.
No published system reaches 100 under the competition's constraints; the target we can move is our own score.

## The structures they share, and our status

| Structure (who) | Our status |
|---|---|
| Persistent memory across the game: notes, prior code, results (AVO, VISTA GUIDE.md, Tycho workspace) | P1/P1B carried note from reasoning, P3 goal/action models across levels, P6 persisted helpers, P8 cross-level note, P16 harness record of each completed level |
| Level-boundary handling: summary, what is new (Tycho consolidation) | P14 level-start comparison and untried actions, P3, P16 |
| Exact perception handed to the model (AVO: exact 64x64 text grid; Tycho: typed diffs; VISTA read_pixels) | Duck's segmentation and ascii, P9 board diff |
| Stagnation supervisor that redirects (AVO) | P15 idle-probe nudge (prompt only); **P19 supervisor call: new** |
| State the expected result before acting, compare after (VISTA; arc3cb's plan queue with expectations) | **P18: new** |
| Test hypotheses against recorded history before spending actions (arc3cb retrodiction; Tycho replay verification) | in the Duck's python tool (history, transitions); P18's wording asks for it |
| Executable world model verified against terminals, guarded plans (Tycho) | not ported: needs far more calls than our ~55 per game |
| One action per turn (VISTA, Tycho) | not ported: throughput; the Duck batches |

## Sources
- NVIDIA AVO: https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/
- VISTA: https://vista-research.github.io/ and https://github.com/joshhhhhan/VISTA
- Tycho: https://github.com/NIMI-research/Tycho, arXiv 2607.28287 (lesson 0013)
- arc3cb / avo-qwen-arcagi3 (open weights via Cerebras, results pending): https://github.com/criticaldata/avo-qwen-arcagi3
- Frontier and open-weight reference points: https://benchlm.ai/benchmarks/arcagi3, https://arcprize.org/leaderboard/community
- NVARC (NVIDIA's ARC-AGI-2 winner, fine-tuned 4B + synthetic data + test-time training): https://developer.nvidia.com/blog/nvidia-kaggle-grandmasters-win-artificial-general-intelligence-competition/
