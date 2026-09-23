Blind exploration (random or novelty search) completes a few early levels but at 10-100x the human action count, which RHAE rounds to zero.

- exp-000 (random, 5000 actions/game): 8/183 levels, total 0.19, all from one lucky 26-action level.
- exp-002 (novelty explorer, 3000 actions/game): 5/183 levels, total 0.06.
- Implication for the harness: probing must be hypothesis-driven and cheap (one action per hypothesis),
  and once mechanics are known the agent must plan offline (search its own model) instead of trying
  things in the environment. Budget rule of thumb: stay within ~2x the human count to keep 25% of a level.
- Where blind search still helps: as the crash fallback (never leave a game idle) and as a probe
  primitive the model can call for a bounded number of actions when it has no hypothesis.

Addendum 2026-09-23 (still true, with one exception): blind search is worthless for the levels it plays, but level 1
is nearly free when every later level beats the human count (4 levels: ≤ 0.95 of it; 7: ≤ 0.98; 9+: ≤ 0.99;
`docs/research/road-to-100-v2.md` section 1). So exploring level 1 blind costs little if, and only if, the agent
then wins the later levels. With budget-bar cells masked from its state keys (exp-027) the explorer clears level 1
of 9 of 19 dev games in at most 90 s of CPU; its levels 2+ still cost 700 to 19,000 actions.
