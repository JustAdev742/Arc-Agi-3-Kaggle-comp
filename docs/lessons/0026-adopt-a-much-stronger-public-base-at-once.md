Summary: when a public, permissively licensed notebook scores several times our best, adopt it unchanged first (one run to make it submittable, then the owner submits) and move our work on top of it; watch the public notebook list around every milestone deadline, the day it passes.

# Adopt a much stronger public base at once (2026-10-02)

Daniel Franzen published his Milestone 2 solution on Sep 30 (LB 27.89, Apache-2.0). By Oct 2 hundreds of teams
had submitted it and 438 teams scored 20 or more; we sat at 4.70, rank 509. Our own research had pointed the same
way (serving throughput, long reusable prefixes, hysteresis trims, an SGLang path), but his notebook had all of it
tested and shipped, plus a priority scheduler and perception changes we never built. Two weeks of our serving arms
would at best have reached part of what one copy of his notebook gives.

Our scheduled check for Milestone 2 notebooks was set for Oct 1 00:30 but was only handled on Oct 2 22:2x. A
milestone deadline is the moment public code jumps; the check belongs on the deadline day itself.

How to apply: at each milestone, and whenever the LB moves by more than the draw noise, list public notebooks by
score (`kaggle kernels list --competition ... --sort-by scoreDescending`) and download the leaderboard. If one is
clearly stronger and its licence allows it, build an unchanged copy, run it once, have the owner submit it, and
only then branch our ideas off it, each read against repeated runs of the unchanged copy.
