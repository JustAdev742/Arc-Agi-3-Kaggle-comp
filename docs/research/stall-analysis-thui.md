Why progress stalls after level 1 (run thui-animfast-b71-full25-r1, one seed)

Scope: the 17 games that solved some levels but not all (11 stuck on L2): 1,252 stuck minutes, 309 turns, 442 calls. Time shares split each turn across its calls at 145 s + 4 s per 1k generated characters (fit on 444 turns, R² 0.27). Root causes are my reading of the transcripts. The run has none of P1–P10 (fresh namespace, note resets, no board diff).

Per game (solved, minutes on the stuck level, dominant modes)

| game | solved | min | dominant |
|---|---|---|---|
| sb26 | 1/8 | 118 | a: flat slot order submitted 4×, connector-slot order never tried; 35% on animations |
| s5i5 | 1/8 | 117 | f/i: 62 min without an action, 4 length cut-offs; h: re-derived the L1 goal it stated at level start |
| ls20 | 1/7 | 109 | a+e: L1 note "icon = decorative", icon first examined 97 min into the level; d: cumulative score read as a pickup reward for 52 min |
| lf52 | 1/10 | 102 | a+e: kept "1 peg" goal; arrows first pressed 42 min in |
| r11l | 1/6 | 101 | h 30% (5 returns to L1 history); d: score misread |
| sp80 | 1/6 | 96 | b: rules visible only in animations (24%); g 17% |
| ka59 | 1/7 | 81 | b: movement rule revised 4×; e: wrong cross-level note |
| tn36 | 1/7 | 60 | a: goal panel spotted 28 min in, then dropped |
| wa30 | 1/9 | 50 | d: cell grid off by 2 for ~18 min; e: L1 goal re-derived |
| cd82 | 1/6 | 45 | b+e: assumed L1 controls; 24 actions vs human 8 |
| tr87 | 1/6 | 39 | d: misread reward, 49 presses on a false oracle |
| re86 | 3/8 | 70 | a: never tried a different-coloured frame; h 35% |
| dc22 | 2/6 | 70 | b: unlock never found |
| vc33 | 3/7 | 58 | b: untested conservation assumption; i 18%, h 10% |
| lp85 | 4/8 | 49 | b: wrong cycle model; g 17% |
| sc25 | 3/6 | 48 | a: wrong handle, GAME_OVER; h 14% |
| ar25 | 3/8 | 40 | b: plan premise wrong |

Aggregate
- By time: analysis without acting (f) 38%; calls that acted 32%; re-deriving earlier levels (h/e) 9%; animation decoding 9%; runaway reasoning (i) 6%; tool errors (g) 5%. 151 of 309 turns ended at the 180-s yield with no action; 107 re-sent prompts were identical. Median longest action-free stretch: 22 min. The median stuck level got 8 action batches.
- By count (primary cause): b 8 games, a 6, d 3. e contributed in 6, h in 10, score misread in 3.
- Knowledge loss is built in. The note was empty at the start of all 39 new levels, including 845–1,485-character notes, and only 3 of 39 first new-level turns wrote a Cross-level line. The previous level leaves the raw context a median 22 min after level-up (12–34), so 73% of stuck time ran without it. All 10 re-derivation episodes began after that point (25–86 min in). In 7 games the note stayed empty for ≥90% of the level: labels sat in reasoning or used non-standard forms.
- 13 of 17 stuck levels were still under the human action count at the cap: wall-clock, not actions, binds.

Proposals, ranked
1. Level-start protocol. Carry goal and action models marked "verify" (P3). Add a harness diff of objects and valid actions against the previous level, plus a rule to spend ≤1 action on each new element or unused action before committing. Evidence: lf52 arrows, ls20 icon, sb26 connector, re86 frames, tn36 panel, ka59/cd82 changed controls. P3 covers the carry; the diff and per-level probing are new (P10 is L1-only). Measure: minutes until every valid action and new object has been used on L2+, and L2+ solve rate.
2. Experiment governor. After 2 action-free calls on a level more than 15 min old, require a probe of ≤3 actions with its outcome predicted first. Evidence: 68% non-acting, s5i5's 62-min gap, 13/17 under the action baseline. Not covered. Measure: acting share, levels solved, actions per solved level (RHAE cost).
3. Persistence plumbing. P1, P1B, P8, P4, plus a pinned, harness-written record of how each level was won (last actions before level_completed, final-state summary). Evidence: 39/39 resets, 115 min re-deriving, r11l's cross-level note stuck in reasoning. The record is new. Measure: calls indexing earlier-level history fall to 0.
4. Rename score to levels_completed, stating it changes only on completion. Evidence: ls20, r11l, tr87. Not covered. Measure: regex count of "score went/+1" misreads.
5. Cap thinking (P11) and stop re-sending identical prompts after a yield. Evidence: 10 length cut-offs; history shrank to 6–7 messages after 26–36k-character reasoning (sb26, ka59). P11 covers the cap; dedup is not listed. Measure: cut-offs, calls per game, minutes until the previous level leaves the context.
6. P6/P7, a consistent animation() schema, and a nudge that fires once per new animated action. Evidence: 12 of 34 failed calls involved animation() (missing keys, error dicts); s5i5 re-read one stale animation 7× after 4 nudges. The animation fixes are not covered. Measure: error calls and animation time share.

Surprises: every note resets at level-up; the cumulative score is repeatedly read as within-level progress; ka59 kept a good note and still stalled, so persistence is necessary but not sufficient. Caveat: one seed and 17 games cannot reliably separate a from b.

Files: /tmp/claude-0/-home-user-Arc-Agi-3-Kaggle-comp/d342458e-03bd-545b-8a6d-06bca061963e/scratchpad/stall_analysis/ (notes.md has per-game evidence with timestamps; classify2.py/.json the time shares; metrics.json and summary.json the counts).
