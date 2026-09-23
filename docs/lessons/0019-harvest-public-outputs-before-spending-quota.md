Summary: a public Kaggle notebook's own run output is downloadable; harvest it before spending GPU quota on the same configuration.

# Harvest public outputs before spending quota (2026-09-23)

`kaggle kernels output <owner/slug>` downloads the files a public notebook wrote in its last "Save & Run All". For the
Duck-family notebooks that is a full public-25 run: benchmark.json (every action with tokens and wall-clock, levels,
baselines), transcripts, prompt logs and the vLLM server log and metrics. `scripts/harvest_public_runs.py` rescored 20+
such runs with our scorer in about an hour with no GPU quota, which gave the configuration distributions (lesson 0018),
the time-vs-score curves, the serving bottleneck and the transcript analysis (research log, 2026-09-23). Two of the
four GPU runs queued before the harvest duplicated configurations already sampled several times.

Rules that follow:
1. Before running a public configuration under our account, harvest its public outputs and those of its copies.
2. Our own GPU runs are for our changes and for creating a submittable version, not for re-measuring public notebooks.
3. Other teams' numbers are for ranking configurations; never report them as our results.
