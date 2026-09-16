import numpy as np, sys, logging
logging.disable(logging.INFO)
sys.path.insert(0, ".")
from arc3.env import make_arcade, LocalEnv, Action
from arc3.entities import Tracker
from arc3.planner import MoveModel

KEYS = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4}
arc = make_arcade("environment_files")
env = LocalEnv(arc, "ka59", seed=0)
f = env.reset()
g = np.asarray(f.grid, dtype=np.int16)
t = Tracker(); t.reset(g)
def do(a):
    global g
    act = Action.click(a[1], a[2]) if isinstance(a, tuple) else Action.simple(KEYS[a])
    fr = env.step(act)
    g = np.asarray(fr.grid, dtype=np.int16)
    rec = t.update(g, a if isinstance(a, str) else ("CLICK", a[1], a[2]))
    print(a, Tracker.describe(rec, t)[:160])
for a in ["UP", "DOWN", "LEFT", "RIGHT", ("CLICK", 36, 30)]:
    do(a)
av = t.avatar(); print("avatar", av["id"], av["keymap"], av["entity"]["x"], av["entity"]["y"])
m = MoveModel(t)
print("pos", m.pos, "sprite", m.w, m.h, "colour", m.color)
x0, y0 = m.pos
print("obstacle cells right of avatar:", m.obstacles[y0:y0+m.h, x0+3:x0+3+m.w].astype(int).tolist())
print("under colours right of avatar:", t.under[y0:y0+m.h, x0+3:x0+3+m.w].tolist())
print("under colours under avatar history:", sorted({int(t.under[y, x]) for (px, py) in t.pos_hist[m.avatar_id] for y in range(py, py+m.h) for x in range(px, px+m.w)}))
e7 = t.get(7); print("target #7", e7.x0, e7.y0, e7.color)
print("strict plan", m.plan_to_entity(e7), "relaxed plan", m.relax().plan_to_entity(e7))
pred = m.predict(g.copy(), "RIGHT")
do("RIGHT")
ys, xs = np.nonzero(pred != g); print("wrong cells", [(int(x), int(y), int(pred[y, x]), int(g[y, x])) for y, x in zip(ys, xs)]); print("groups", t.groups(), "hud", t.hud_ids())
m2 = MoveModel(t); print("refit pos", m2.pos, "strict plan now", m2.plan_to_entity(t.get(7)))
do("RIGHT"); do("RIGHT")
print("avatar after merge:", t.avatar())
print("current ids/colours:", [(e.id, e.color, e.x0, e.y0, e.w, e.h) for e in t.current if e.color in (14, 0)])
do("LEFT"); print("avatar after LEFT:", {k: v for k, v in t.avatar().items() if k != 'entity'})
