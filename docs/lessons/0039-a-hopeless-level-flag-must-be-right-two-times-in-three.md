Summary: under D′ a flag that ends or parks a "hopeless" level pays only if it is right about 61-66% of the time; the best early signal we have (the previous level's time, AUC 0.68 at entry) is right 8-27%, so no scheduling patch.

# A hopeless-level flag must be right two times in three (2026-10-10)

What happened: time-allocation.md §4.6 priced a perfect detector of levels needing more than 60 active minutes at
+8.07 public-25 points (rerun model). docs/research/beat-tufa/hopeless-signal.md looked for a real one in six
full-length runs: at level entry the game's own history (mostly how long the previous level took) gives AUC 0.68-0.70
(leave-one-game-out; null 0.45); what the agent does in its first 5-15 minutes on the level adds nothing. Every one of
48 end/park policies built on it lost to D′ in the rerun model (best -0.22 [-1.81, +1.13]).

Why: a correct flag frees about 0.17 points of slot-time for other games; a wrong one throws away a level that would
have been solved (and the levels after it), about 0.27-0.33. Hopeless levels are only 9-28% of attempts at the
checkpoints, so a ranking with AUC 0.7 still flags mostly solvable levels. D′ already starves stuck games, which shrinks
the gain side further.

How to apply:
- Do not build a give-up, park or demotion rule on time-on-level or on the previous level's time.
- A new signal is worth simulating only if, out of fold and across games, it is right about two times in three at a
  useful firing rate (or reaches AUC ~0.9 with <= 1% false positives at entry). Check precision at the firing rate
  first; AUC alone misleads at these base rates.
- Judge any such rule in the rerun model (lesson 0036), never on public-25 runs.
