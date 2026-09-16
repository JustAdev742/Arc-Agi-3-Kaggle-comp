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
