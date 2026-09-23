Summary: one run ranks nothing; identical notebooks vary 1.5-2.3x on the leaderboard and by 2 points on the public games, and Duck-style public-game scores of 10-22 become 5-7 on the hidden set.

# Leaderboard noise and the public-to-hidden gap (2026-09-23)

Source: the competition forum, read with `kaggle competitions topics show` (threads 732854, 736578, 739801, 731522,
738762, 740812, 717133). These are other teams' numbers, not our runs.

- Same notebook, same settings, repeated official submissions: 1.29, 1.05, 0.71, 0.73, 0.68, 0.75, 0.55, 1.11, 1.17,
  0.90, 0.94 (a 2.3x spread, thread 731522); another team: "2.5-5.5 same code"; Tufa Labs: their best notebook scored
  between 0.77 and 1.30 on reruns.
- Same notebook on the 25 public games twice in one job: 2.85 and 4.75 (thread 739801). Tufa Labs: 1.60 +- 0.45 over
  20 tries of 25 games.
- Public-25 to leaderboard for Duck-style agents on Qwen3.8 (thread 732854): 17.34 -> 5.19, 22.26 -> 5.37,
  17.34 -> 6.91, 9.91 -> 6.23, 15.7 -> 7.37, 11.04 -> 2.71, 10.2 -> 3.05, 7.6 -> 3.35. The hidden games are harder by
  design (ARC-AGI-3 technical report: the private set is out of distribution and more difficult).
- A harness change that gave more time to games at level 2 or higher raised one team's public score from 7.6 to about
  13 and slightly lowered its hidden score (thread 740812): on the hidden set a weak agent rarely reaches level 2, so
  public-only gains can be artefacts of the easier games.

What follows for us:
1. Never rank two arms on one public-25 run unless they differ by more than about 2 points; repeat runs or use paired
   per-game comparisons, and read level counts and actions per level, not the mean alone.
2. The final score is the better private score of the two selected submissions, each fixed at its own run. Submitting
   the strongest candidate every day and selecting the two best public scores at the end buys draws; a new idea should
   displace the candidate only on repeated evidence.
3. Prefer changes that plausibly help on harder, unseen games (throughput, context hygiene, bug fixes in what the model
   sees) over changes whose gain shows only on the public games.
