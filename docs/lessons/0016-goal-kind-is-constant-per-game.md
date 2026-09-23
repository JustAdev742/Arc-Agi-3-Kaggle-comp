# A game's kind of win condition never changes between levels, the target is drawn, and every level has a budget bar

Found 2026-09-23 by tracing the win and lose checks in the source of all 19 dev games (three read-only agents; the six
validation games were not opened). Full table: `docs/research/win-conditions-dev.md`.

- **The goal kind is constant within a game (19 of 19).** Later levels add mechanics (bridges, bombs, enemies,
  gravity flips, fog), never a new kind of goal. Infer the goal on level 1 and carry it; relearn mechanics per level.
- **The target is on screen from the first frame in 17 of 19** (outline, marker, reference picture or strip, clue
  tiles); bp35's gem starts off-screen, tr87's answer must be computed from a shown rule table, m0r0 has no marker.
- **Kinds seen:** place pieces in slots (11 games), make a region equal a shown reference (10), bring a sprite to a
  marker (7), align (1), count (1), all tiles satisfy a local rule (1). None was "collect everything".
- **Every game has a per-attempt action budget drawn as an edge bar**; running out loses the attempt (ls20 costs a
  life). That bar is what volatility masking removes from state keys (exp-027).
- **The win check can be gated by action kind** (dc22 arrows only, cd82 pour or stamp, tr87 UP/DOWN, m0r0, sk48): a
  board can satisfy the goal without ending the level. Contrastive goal induction must tolerate this.
- **Human baselines include failed attempts** on at least one level of 8 games (the baseline exceeds the per-attempt
  budget), so those levels are lenient.
- **Multi-target games need every target** (9 of 9); level 1 usually has one target, where "some" and "every" agree,
  so prefer the universal form until a level separates them.

This is public-set evidence: use it as priors for the grammar and the harness, never as per-game rules.
