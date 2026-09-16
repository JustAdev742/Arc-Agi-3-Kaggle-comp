A world model that is wrong three times running is not being revised; retire it, refit from evidence automatically, and generalise obstacles by terrain colour rather than by visited cells.
- exp-009 ka59: a stale strict move model mismatched 14 consecutive correct moves (one action per call) because the floor
  was an entity and the predictor was a snapshot. Fix: live re-fitting predictor, walkable colours from the static layer,
  retirement after 3 misses with a message the model reads.
- The same stop rule applies to any set_model predictor and to rules_predictor(): a verifier without a stop rule turns
  a good plan into one action per call.
Source: docs/postmortems/exp009-ka59-floor-entity-2026-09-16.md, scratchpad replay of environment_files/ka59.
