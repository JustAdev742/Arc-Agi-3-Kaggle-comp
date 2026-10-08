Summary: clearing the conversation on a level that has run 20 minutes without a level-up (ours-07 fresh start) roughly halved that level's chance of being solved (14 of 44 vs ~61% expected), because the context held working understanding the pinned ledger does not carry.

# A mid-level context reset loses more than it frees (2026-10-08)

What happened: ours-07 cleared the history (keeping the system prompt and the exact win ledger) once a level had taken
80k generated tokens and 20 minutes, at most twice per level. In exp-078 and exp-079 it fired on 22 levels each; 6 and
8 of them were solved later (32%). Levels still unsolved at 20 active minutes are solved by minute 60 about 61% of the
time in the base runs (docs/research/beat-tufa/time-allocation.md §3). Typical failure: tr87 level 3, fresh-started
at 21.8 min and again at 55.6 min at the same step, 34 minutes and ~120k tokens without an action, 2 of 6 levels where
every other run got 5-6. The bundle with it scored 39.73 and 45.02 against a base mean of 49.4.

Why: at 20-30 minutes on a level the hazard is still 0.04-0.05 per minute, so the level is usually about to fall; a
reset restarts the analysis from the board without the hypotheses already tested, the probe results and the partial
plans the conversation held. The finding that 35 of 44 stuck final levels were solved in another run is about whole
runs (a fresh start of the game, with the same time ahead), not about a re-roll mid-level.

How to apply:
- Do not clear or summarise away a level's own history while the level is live; trim old levels first.
- A "fresh eyes" mechanism, if any, must add a view (a second opinion on the cached prefix) rather than replace the
  context, and must be judged on whether the levels it touches get solved, against the hazard curve.
- Count mechanism outcomes per touched level (scripts/mech_wholegame.py, scripts/fresh_start_outcomes.py over the
  solver_analysis HTML; scripts/run_pergame.py for the per-game table) before reading the run's mean score.
