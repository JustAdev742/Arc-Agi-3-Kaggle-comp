Connected components are not entities: separate terrain from sprites before fitting rules, or most "events" are artefacts.

What the code-only rule probe (`scripts/rule_coverage.py`) showed on the 25 public games, in order of impact:

- **Occlusion is not an event.** A sprite moving over the floor makes the floor component "resize" or "move"
  every step. Filter any change whose cells all lie under a mover before or after the step; keep a static layer
  (`Tracker.under`) of the last-seen terrain colour per cell so predictions can restore what a sprite uncovers.
- **Sprites have parts.** ar25's avatar is a 9x9 body with 1-cell eyes of another colour; ls20's is two colour
  bands. Parts get their own ids and 1-cell parts are matched to the wrong twin (spurious moves of 9 cells).
  Merge parts strictly inside a mover's box and co-moving adjacent groups into one compound entity
  (`Tracker.compound_frames`). ar25 went from 0.24 to 1.00 explained.
- **Blocking is per cell, not per box.** A wall ring has a bounding box covering the arena. Judge moves by the
  colours under the sprite's target cells: "only onto colours {floor}" (walkable) fits better and transfers to
  the next level, unlike an obstacle map.
- **Moves can require a resource.** ls20 stops moving when its energy bar reaches zero. A move rule with
  "while colour 11 exists" removes the last contradiction; the fitter tries a required colour whenever a free
  move failed.
- **HUD bars shift as they shrink.** A bar that empties from the left moves +1 per action; treat edge strips that
  only move along their own axis as HUD and leave them out of the coverage accounting.

Source: `runs/rule-coverage/`, `arc3/entities.py`, `arc3/dsl.py`, research log exp-006a.
