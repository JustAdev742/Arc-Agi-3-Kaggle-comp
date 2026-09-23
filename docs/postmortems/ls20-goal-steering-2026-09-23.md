# ls20 level 2 lost by goal-steered exploration (exp-029, 2026-09-23)

Runs: `runs/exp029-explorer-control` (level 2 in 6,232 actions) and `runs/exp029-explorer-goal` (level 2 unsolved
after 19,404 actions); same seed, same 20,000-action cap, only `goal_directed` differs.

**WHAT DID WE BELIEVE?** Goal candidates induced by contrast on level 1 (true on the winning frame, false on every
visited state) would steer the explorer: among the 48 nearest unexplored states, prefer the one whose frame is closer
to the goal (path length + 0.25 x goal distance). On ls20 the candidates were `every_in(colour 9, colour 5)`,
`every_in(colour 12, colour 5)`, `inside(colour 12, colour 5)` and two `vanish(colour 9, shape ...)`: the key ends up
in the socket.

**WHAT ACTUALLY HAPPENED?** The frontier choice differed from the nearest state 282 times; level 2 was never solved
(control: 6,232 actions). Every other game played identically except ar25 (level 2 in 14,493 instead of 19,267 actions,
level 3 slower).

**WHICH ASSUMPTION WAS WRONG?** That a state nearer the goal is nearer a win. On ls20 a socket accepts the key only
after changer tiles have reshaped, recoloured or rotated it (census, `win-conditions-dev.md`); until then the socket
blocks like a wall. The goal distance pulls the search toward the socket and away from the changer tiles it must
visit first.

**WHAT EVIDENCE EXPOSED IT?** The paired runs above; the census's reading of ls20's win check (`ls20.py:2040-2044`:
the socket is used only when shape, colour and rotation all match).

**WHAT CHEAPER TEST WOULD HAVE CAUGHT IT?** One game with a known precondition (ls20) in a single-game smoke before
the dev run; it showed the regression in 20 s (the smoke on ls20 was run, after the dev run had been launched).

**WHAT SHOULD THE REVISED MODEL BE?** Use a goal distance only inside a model of the mechanics (planning, where the
changer tiles are part of the simulated state), never as a greedy bias on blind exploration. Blind exploration stays
breadth-first.

**IS THIS LESSON GAME-SPECIFIC OR GENERAL?** General: any goal with preconditions (keys, switches, bridges, colour
changers) turns greedy goal-seeking into a trap; the census lists such mechanics in ls20, dc22 (buttons, bridges),
m0r0 (plates and doors), ka59 (kicks), wa30 (carrying).

**SHOULD IT BECOME A REUSABLE SKILL?** No new skill; it is recorded in the research log (exp-029, reverted) and in
`docs/research/road-to-100-v2.md` section 4.5.

**CONFIDENCE:** high for ls20 (the mechanism is read from the source); medium for the generalisation (one paired run).
