RHAE facts that shape every design choice: exploration is quadratically expensive, resets cost one action, later levels weigh more.

- Per level `min((baseline/actions)^2, 1.15)`: 2x the human action count already scores 25%, 3x scores 11%.
- Level weight = level index, so level 8 of 8 is worth 8x level 1. Reaching deep levels matters more than
  polishing level 1, but the game cap `completed_weight / total_weight` means an unsolved last level
  forfeits its share entirely.
- RESET costs one action but restores the level start. There is no "replay for free" trick in competition
  mode (level resets only, one play per game).
- Actions taken while exploring a level all count toward that level. Thinking time does not.
Source: `arc_agi/scorecard.py`, Technical Report §4, `tests/test_scoring.py`.
