import numpy as np

from arc3.entities import Tracker
from arc3.planner import MoveModel


class GridWorld:
    """Avatar 4x4 moves 4 px per key; walls (colour 3) block; target (colour 12) static."""

    def __init__(self):
        self.walls = np.zeros((64, 64), dtype=bool)
        self.walls[0:4, :] = True
        self.walls[60:64, :] = True
        self.walls[:, 0:4] = True
        self.walls[:, 60:64] = True
        self.walls[20:44, 28:32] = True  # a vertical wall segment in the middle
        self.pos = (8, 32)
        self.target = (48, 32)

    def grid(self):
        g = np.zeros((64, 64), dtype=np.int16)
        g[self.walls] = 3
        x, y = self.target
        g[y:y + 4, x:x + 4] = 12
        x, y = self.pos
        g[y:y + 4, x:x + 4] = 9
        return g

    def act(self, key):
        d = {"UP": (0, -4), "DOWN": (0, 4), "LEFT": (-4, 0), "RIGHT": (4, 0)}[key]
        nx, ny = self.pos[0] + d[0], self.pos[1] + d[1]
        if not self.walls[ny:ny + 4, nx:nx + 4].any():
            self.pos = (nx, ny)
        return self.grid()


def test_fit_and_plan_around_wall():
    w = GridWorld()
    t = Tracker()
    t.reset(w.grid())
    for k in ("RIGHT", "RIGHT", "UP", "DOWN", "LEFT", "RIGHT"):  # learn the key map (RIGHT at x=16 -> 20 -> blocked later)
        t.update(w.act(k), k)
    m = MoveModel(t)
    assert m.keymap == {"UP": (0, -4), "DOWN": (0, 4), "LEFT": (-4, 0), "RIGHT": (4, 0)}
    target = next(e for e in t.current if e.color == 12)
    path = m.plan_to_entity(target)
    assert path is not None and len(path) >= 10  # must go around the wall segment
    # Execute the plan in the world and confirm arrival; verify the predictor along the way.
    for k in path:
        before = w.grid()
        pred = m.predict(before, k)
        after = w.act(k)
        assert np.array_equal(pred, after), k
        t.update(after, k)
        m = MoveModel(t)
    ax, ay = w.pos
    assert abs(ax - w.target[0]) <= 4 and abs(ay - w.target[1]) <= 4


def test_bump_evidence_marks_invisible_wall():
    w = GridWorld()
    w.walls[32:36, 20:24] = True  # invisible: not drawn
    g0 = w.grid()
    g0[32:36, 20:24] = 0
    t = Tracker()
    t.reset(g0)
    seq = ("UP", "DOWN", "LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT")  # the last RIGHT (from x=16) bumps the invisible wall at x=20
    for k in seq:
        g = w.act(k)
        g[32:36, 20:24] = 0
        t.update(g, k)
    m = MoveModel(t)
    assert m.obstacles[32:36, 20:24].all()
    assert m.step((16, 32), "RIGHT") == (16, 32)


class FloorWorld:
    """A corridor drawn as one colour-1 entity on a colour-0 background (a floor that is an object, as in ka59);
    the avatar (colour 9, 4x4, 4 px per key) may only stand on the floor; a colour-3 wall block sits in the corridor."""

    def __init__(self):
        self.floor = np.zeros((64, 64), dtype=bool)
        self.floor[28:36, 4:44] = True
        self.wall = np.zeros((64, 64), dtype=bool)
        self.wall[32:36, 20:24] = True
        self.pos = (8, 32)
        self.target = (36, 32)

    def grid(self):
        g = np.zeros((64, 64), dtype=np.int16)
        g[self.floor] = 1
        g[self.wall] = 3
        x, y = self.target
        g[y:y + 4, x:x + 4] = 12
        x, y = self.pos
        g[y:y + 4, x:x + 4] = 9
        return g

    def act(self, key):
        d = {"UP": (0, -4), "DOWN": (0, 4), "LEFT": (-4, 0), "RIGHT": (4, 0)}[key]
        nx, ny = self.pos[0] + d[0], self.pos[1] + d[1]
        cells = np.zeros((64, 64), dtype=bool)
        cells[ny:ny + 4, nx:nx + 4] = True
        if (cells & self.floor).sum() == 16 and not (cells & self.wall).any():
            self.pos = (nx, ny)
        return self.grid()


def test_floor_entity_the_avatar_stood_on_is_walkable():
    w = FloorWorld()
    t = Tracker()
    t.reset(w.grid())
    for k in ("UP", "DOWN", "LEFT", "RIGHT"):
        t.update(w.act(k), k)
    m = MoveModel(t)
    assert m.walkable == {1}
    target = next(e for e in t.current if e.color == 12)
    path = m.plan_to_entity(target)
    assert path is not None and not m.optimistic, "strict plan must exist across the floor entity"
    for k in path:
        before = w.grid()
        pred = m.predict(before, k)
        after = w.act(k)
        assert np.array_equal(pred, after), k
        t.update(after, k)
        m = MoveModel(t)
    assert abs(w.pos[0] - w.target[0]) <= 4 and abs(w.pos[1] - w.target[1]) <= 4
    # the wall block (colour 3) is still an obstacle; the floor is not
    assert m.obstacles[33, 21] and not m.obstacles[29, 12]


def test_avatar_follows_control_after_merge():
    """ka59 (exp-009): the avatar vanished into another sprite of its colour and a different entity moved from then on."""
    t = Tracker()

    def frame(a_pos, b_pos, a_alive=True):
        g = np.zeros((64, 64), dtype=np.int16)
        if a_alive:
            g[a_pos[1]:a_pos[1] + 3, a_pos[0]:a_pos[0] + 3] = 14
        g[b_pos[1]:b_pos[1] + 3, b_pos[0]:b_pos[0] + 3] = 14
        g[60:64, :] = 4
        return g

    a, b = (10, 30), (40, 30)
    t.reset(frame(a, b))
    for k, d in (("UP", (0, -3)), ("DOWN", (0, 3)), ("LEFT", (-3, 0)), ("RIGHT", (3, 0)), ("RIGHT", (3, 0))):
        a = (a[0] + d[0], a[1] + d[1])
        t.update(frame(a, b), k)
    av = t.avatar()
    assert av["alive"] and av["key_moves"] == 5
    a_id = av["id"]
    t.update(frame(a, b, a_alive=False), "RIGHT")  # the avatar disappears
    assert t.avatar()["id"] == a_id and not t.avatar()["alive"]
    for k, d in (("LEFT", (-3, 0)), ("UP", (0, -3))):
        b = (b[0] + d[0], b[1] + d[1])
        t.update(frame(a, b, a_alive=False), k)
    av = t.avatar()
    assert av["alive"] and av["id"] != a_id and av["keymap"] == {"LEFT": (-3, 0), "UP": (0, -3)}
