import numpy as np

from arc3.entities import Tracker, tile_size


def scene(avatar_x, bar_len, extra=None):
    g = np.zeros((64, 64), dtype=np.int16)
    g[60:64, 0:64] = 3            # static floor
    g[20:24, avatar_x:avatar_x + 4] = 9   # avatar 4x4
    g[40:44, 30:34] = 12          # static block
    g[0:2, 0:bar_len] = 8         # HUD bar along the top edge, shrinking
    if extra:
        for (x, y, c) in extra:
            g[y:y + 4, x:x + 4] = c
    return g


def test_tracking_events_roles_and_keymap():
    t = Tracker()
    t.reset(scene(4, 40))
    ids = {e.color: e.id for e in t.current}
    r = t.update(scene(8, 38), action="RIGHT")
    assert (ids[9], 4, 0) in r["moved"]
    assert any(eid == ids[8] for eid, _, _ in r["reshaped"])
    r = t.update(scene(12, 36), action="RIGHT")
    r = t.update(scene(12, 34), action="UP")   # blocked: nothing moves, bar shrinks
    assert r["moved"] == []
    r = t.update(scene(8, 32, extra=[(50, 50, 14)]), action="LEFT")
    assert (ids[9], -4, 0) in r["moved"] and len(r["appeared"]) == 1
    av = t.avatar()
    assert av and av["id"] == ids[9] and av["keymap"]["RIGHT"] == (4, 0) and av["keymap"]["LEFT"] == (-4, 0)
    roles = t.roles()
    assert roles[ids[3]] == "static" and roles[ids[12]] == "static"
    assert roles[ids[8]] == "hud" and roles[ids[9]] == "avatar"
    line = Tracker.describe(r, t)
    assert line.startswith("LEFT:") and "moved (-4,+0)" in line and "appeared" in line


def test_groups_and_tile():
    t = Tracker()
    g = np.zeros((64, 64), dtype=np.int16)
    g[10:14, 10:14] = 5
    g[10:14, 14:18] = 6  # two-colour sprite moving together
    g[30:34, 30:34] = 7
    t.reset(g)
    for k in range(1, 3):
        g2 = np.zeros((64, 64), dtype=np.int16)
        g2[10:14, 10 + 4 * k:14 + 4 * k] = 5
        g2[10:14, 14 + 4 * k:18 + 4 * k] = 6
        g2[30:34, 30:34] = 7
        t.update(g2, action="RIGHT")
    groups = t.groups()
    assert len(groups) == 1 and len(groups[0]) == 2
    assert tile_size(g) == 4


def test_real_game_avatar_detected():
    from arc3.env import Action, LocalEnv, make_arcade

    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    t = Tracker()
    t.reset(env.frame.grid)
    for a in (4, 4, 1, 3, 2, 4, 1, 1):
        before = env.frame
        env.step(Action.simple(a))
        t.update(env.frame.grid, action=a)
        assert before is not env.frame
    av = t.avatar()
    env.close()
    assert av is not None and av["key_moves"] >= 3, av
    assert len(t.static_ids()) > 0


def test_turning_sprite_keeps_its_id():
    """wa30 (exp-009): the avatar's body is 3x4 when facing up/down and 4x3 when facing left/right; every turn used to
    create a new id, so avatar() reported a dead entity and the key map was split across ids."""
    import numpy as np

    from arc3.entities import Tracker

    def frame(x, y, horizontal):
        g = np.ones((64, 64), dtype=np.int16)
        if horizontal:
            g[y:y + 3, x:x + 4] = 14
        else:
            g[y:y + 4, x:x + 3] = 14
        g[10:14, 10:14] = 9  # a static block
        return g

    t = Tracker()
    t.reset(frame(32, 48, False))
    t.update(frame(32, 44, False), "UP")
    t.update(frame(32, 48, False), "DOWN")
    rec = t.update(frame(28, 48, True), "LEFT")  # turns and moves
    assert [m[0] for m in rec["moved"]] == [t.avatar()["id"]] and rec["reshaped"] and not rec["appeared"] and not rec["disappeared"]
    t.update(frame(32, 48, True), "RIGHT")
    av = t.avatar()
    assert av["alive"] and av["key_moves"] == 4
    assert av["keymap"] == {"UP": (0, -4), "DOWN": (0, 4), "LEFT": (-4, 0), "RIGHT": (4, 0)}


def test_marker_riding_on_a_sprite_edge_joins_its_compound():
    """wa30 (exp-009): a 4x1 'eyes' strip sits on whichever edge of the 4x3 body faces the move; the body's box shifts
    by 3 or 4 px, the union by exactly 4. compound_frames() must move the union."""
    import numpy as np

    from arc3.entities import Tracker

    def frame(x, y, facing):
        g = np.ones((64, 64), dtype=np.int16)
        if facing in ("UP", "DOWN"):
            g[y:y + 4, x:x + 4] = 14
            g[y if facing == "UP" else y + 3, x:x + 4] = 0
        else:
            g[y:y + 4, x:x + 4] = 14
            g[y:y + 4, x if facing == "LEFT" else x + 3] = 0
        g[10:14, 10:14] = 9
        return g

    t = Tracker()
    t.reset(frame(32, 48, "DOWN"))
    pos = (32, 48)
    for k, d in (("UP", (0, -4)), ("DOWN", (0, 4)), ("LEFT", (-4, 0)), ("RIGHT", (4, 0)), ("UP", (0, -4))):
        pos = (pos[0] + d[0], pos[1] + d[1])
        t.update(frame(pos[0], pos[1], k), k)
    frames = t.compound_frames()
    heads = [[e for e in f if e.color == 14] for f in frames]
    assert all(len(h) == 1 and h[0].w == 4 and h[0].h == 4 for h in heads), heads[-1]
    xy = [(h[0].x0, h[0].y0) for h in heads]
    assert xy == [(32, 48), (32, 44), (32, 48), (28, 48), (32, 48), (32, 44)]
    assert len({h[0].id for h in heads}) == 1


def test_plain_frames_show_the_board_as_it_is_and_keep_the_avatar_id():
    """A mover entering a static block: the symbolic frame keeps the block's canonical shape (occlusion handling),
    plain_frames() shows the block with the covered cells gone and lends the mover its tracker id."""
    import numpy as np

    from arc3.entities import Tracker
    g0 = np.zeros((64, 64), dtype=np.int16)
    g0[20:27, 20:27] = 5   # 7x7 block
    g0[2:5, 2:5] = 9       # 3x3 mover
    t = Tracker()
    t.reset(g0)
    mover = next(e for e in t.current if e.color == 9).id
    g1 = g0.copy()
    g1[2:5, 2:5] = 0
    g1[10:13, 10:13] = 9
    t.update(g1, "RIGHT")
    g2 = g1.copy()
    g2[10:13, 10:13] = 0
    g2[22:25, 22:25] = 9   # inside the block, covering 9 of its cells
    t.update(g2, "RIGHT")
    plain = t.plain_frames()
    assert len(plain) == 3
    last = plain[-1]
    block = next(e for e in last if e.color == 5)
    assert block.size == 49 - 9 and (block.w, block.h) == (7, 7)
    av = next(e for e in last if e.color == 9)
    assert av.id == mover and (av.x0, av.y0) == (22, 22)
    assert t.plain_frames() == plain  # cached, aligned with the grids


def test_plain_frames_keep_small_enclosed_background_holes():
    """ls20 level 7: the socket is an island of the background colour inside the board; the symbolic frames ignore
    every background-coloured cell, the plain frames keep small enclosed ones and drop the background itself."""
    import numpy as np

    from arc3.entities import Tracker
    g = np.full((64, 64), 5, dtype=np.int16)   # background colour 5
    g[10:40, 10:50] = 3                        # a board
    g[20:27, 20:27] = 5                        # a 7x7 hole in the board, background colour
    g[2:5, 2:5] = 9
    t = Tracker()
    t.reset(g)
    assert not any(e.color == 5 for e in t.frames[0])  # symbolic frames: background ignored entirely
    plain = t.plain_frames()[0]
    holes = [e for e in plain if e.color == 5]
    assert len(holes) == 1 and (holes[0].x0, holes[0].y0, holes[0].w, holes[0].h) == (20, 20, 7, 7)


def test_bg_holes_knob_makes_enclosed_background_islands_entities(monkeypatch):
    import numpy as np

    from arc3.entities import Tracker, default_bg_holes, segments
    g = np.full((64, 64), 5, dtype=np.int16)
    g[10:40, 10:50] = 3
    g[20:27, 20:27] = 5   # enclosed island of the background colour
    g[0:3, 30:33] = 5     # background touching the border (already part of it here, but a separate blob would be dropped too)
    assert not any(o.color == 5 for o in segments(g, 5))
    holes = [o for o in segments(g, 5, bg_holes=True) if o.color == 5]
    assert len(holes) == 1 and (holes[0].x0, holes[0].y0, holes[0].w, holes[0].h) == (20, 20, 7, 7)
    t = Tracker(bg_holes=True)
    t.reset(g)
    assert any(e.color == 5 and e.w == 7 for e in t.frames[0])
    t2 = Tracker()
    t2.reset(g)
    assert not any(e.color == 5 for e in t2.frames[0])
    monkeypatch.setenv("ARC3_BG_HOLES", "1")
    assert default_bg_holes() and Tracker().bg_holes
    t3 = Tracker(bg_holes=True)
    t3.reset(g)
    g2 = g.copy()
    g2[0:3, 0:3] = 9
    t3.update(g2, "UP")
    assert any(e.color == 5 and e.w == 7 for e in t3.frames[-1])  # the knob survives reset() and update()
