Summary: on Flash-Next at our full time budget, reasoning effort "medium" instead of the template's default "xhigh" loses badly (exp-082: 32.53 vs 39.7-56.0 in eight runs; exp-037 also lost); short-budget wins for medium do not transfer.

# Reasoning effort "medium" loses at full length (2026-10-10)

What happened: a competitor's runs on Franzen's stack at ~25 slot-minutes per game showed medium nearly doubling the
levels on the hard 15 games (17 -> 32). We tested it with a 5-line patch (ours-09, OURS_REASONING_EFFORT) on the
exp-074t candidate at full length (25 games x 121 min, ~70 slot-minutes per game, like the hidden set): 32.53 and 84
levels, the lowest of nine full-length runs (39.73-56.00, 95-124 levels); hard 15 29 levels (39-58), easy 10 55
(56-66); -16.4 per game against six comparable runs (SE 5.2). Our Sep 23 test on the older harness (exp-037) had also
found no gain.

Why (reading): with time to spare, the extra deliberation xhigh asks for ("validate key assumptions, consider
plausible alternatives") decides levels; medium mostly removes it rather than the long-tail turns (output tokens per
request were unchanged, 2,063 vs 1,989-2,202). At short budgets the extra turns matter more.

How to apply:
- Keep the template's default (xhigh) for the submission. Do not retest a static lower effort on short-budget
  evidence; any effort change must be judged at the full time share per game.
- Effort-by-phase (medium only before a level's first action, etc.) is untested; given two full-length losses, it is
  low priority.
- ours-09 stays in the repo (default off: no env, no change to the requests).
