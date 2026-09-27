Summary: on this competition the hidden leaderboard has tracked tokens per hour and their allocation across games, not harness prompt text; judge a change by what it does to calls per game at the real load, and read other teams' evidence (the Kaggle CLI reads the forum) before building.

# The LB moves with tokens, not harness text (2026-09-27, from other teams' evidence plus our own draws)

Every public Duck fork, ours included, draws 3-7 on the hidden LB while scoring 8-22 on the public 25; teams report a
3-4x drop and 1.5-2.3x spreads between identical submissions. The changes other teams could tie to LB gains were
serving and allocation (16 vLLM sequences, MTP off with the memory moved to KV, prefix-cache-friendly history,
per-game time sized to the 110-game waves); careful one-change ablations of harness ideas (world models, higher-res
images, consensus, AVO-style supervisors at NVARC3) came out at or below baseline. Our own LB draws fit: the base 2.72,
our fixes 3.81, fixes + KV 4.23, + P23/P24 + more KV 4.70.

How to apply: for a harness idea, estimate its token cost per call and its effect on calls per game before its effect
on reasoning; prefer changes that make the server do less redundant work (prefill of repeated history, idle queueing).
Before building, search what others measured: `kaggle competitions topics list arc-prize-2026-arc-agi-3 -s top`,
`kaggle competitions topics show <id>`, `kaggle competitions topic-messages <id>` return forum threads in full.
Details and sources: docs/research/public-code-sep27.md.
