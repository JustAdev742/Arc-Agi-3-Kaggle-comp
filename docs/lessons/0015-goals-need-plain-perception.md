# Goal predicates must see the board as it is: plain components, background-coloured holes, and the entity you control

Found 2026-09-17 by replaying the human ls20 recording (546 actions, 7 levels) through the agent
(`scripts/goal_probe.py --recordings data/human`). Three perception facts, each of which alone hid the win condition:

1. **The tracker's frames are an interpretation, not the board.** Occlusion handling keeps a canonical shape for an
   entity a mover covers and hides pieces that split off; the compound view merges multi-part sprites. Right for
   movement and rule fitting, wrong for a goal such as "the key sits in the socket": on level 7 the shrunken socket
   was absorbed into the key's compound and vanished from both views. `Tracker.plain_frames()` (plain connected
   components per observed grid, tracker ids lent by overlap) is the representation for goals; candidates are
   evaluated on both (`dsl.goal_candidates_dual`, entries tagged `rep`).
2. **Background-coloured islands are objects.** The tracker ignores every cell of the background colour; on level 7
   the socket is a 7x7 island of the background colour enclosed by the board. Plain frames keep background-coloured
   components that are small (<= 400 cells) and enclosed (not touching the border).
3. **Win conditions are often about the entity you control, not about colour pairs.** `inside(colour 9, colour 5)`
   fails because a HUD legend shows the same relation from the start; `avatar_inside(colour 5)` (the level's avatar
   ends strictly inside a colour-5 entity) is consistent on all 7 levels, ranked first before every win from level 2
   on, and the per-level falsification removed the spurious candidates (`count(colour 9) == 4`, a colour-9 vanish)
   on levels 2, 3 and 5. Avatar ids are archived per level with the frames.

Also: the final WIN frame arrives as a single layer that is the terminal itself (a level transition carries the
terminal first and the next level's start last); the harness now archives it as observed. The local engine
reproduced all 546 recorded frames exactly.
