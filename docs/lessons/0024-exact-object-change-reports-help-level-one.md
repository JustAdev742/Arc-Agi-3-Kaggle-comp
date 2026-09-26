Summary: telling the model exactly which objects each action moved, turned, created or removed (including mid-animation) raised hard-game level-1 solves from 10/16 to 15/16 over two paired runs; state computed facts, don't leave them to the model's own diffing. Leaderboard confirmation pending.

# Exact object-change reports help level 1 (2026-09-26, public 25 only)

P23 (scripts/taaf_ours_patch.py) appends, after each executed action sequence, a short exact report: objects that
moved (with the offset), turned, appeared, vanished, changed colour or size, and what happened only during the
animation. P24 lists objects equal up to rotation/reflection/colour at each level start. Two paired runs on the fixes +
6.5 GiB KV profile: with them 10.98 and 9.64 (hard-game level 1 8/8 and 7/8), without 9.60 and 6.30 (6/8 and 4/8);
z-sum +14.3 against +2.8. The report costs about 240 prompt tokens per request.

What made it work, from the review and replays (docs/research_log.md, 2026-09-25): the report must not say false
things. The first version paired background panels with pockets, called in-between animation frames "gone", paired
identical objects with wrong offsets and read a moved sprite as a colour change; each of those was a unit test before
the GPU run. Replaying real runs' recorded actions through the local engine (docs/research/hard-games-level1/) was the
cheapest way to see the report on real batches before spending GPU time.

Open: whether the gain carries to the hidden set (earlier forks scored 10-15 on the public 25 and about 4 on the LB).
