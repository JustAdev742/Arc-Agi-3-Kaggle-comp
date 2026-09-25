import sys, logging
sys.path.insert(0, '/home/user/Arc-Agi-3-Kaggle-comp')
logging.disable(logging.INFO)
import numpy as np
from arc3.env import LocalEnv, make_arcade, Action
arc = make_arcade('/home/user/Arc-Agi-3-Kaggle-comp/environment_files')
env = LocalEnv(arc, 'g50t'); f = env.reset()
def blocks(g):
    out = []
    for r in range(7, 58):
        for c in range(7, 58):
            w = g[r:r+5, c:c+5]
            if w.shape == (5, 5) and len(np.unique(w[[0,0,4,4],[0,4,0,4]])) == 1 and w[0,0] not in (5, 0, 1) and (w[0,:] == w[0,0]).all() and (w[:,0] == w[0,0]).all() and (w[4,:] == w[0,0]).all():
                out.append((int(w[0,0]), r, c))
    return out
seq = [('RIGHT',4),('RIGHT',4),('RIGHT',4),('RIGHT',4),('SPACE',5),('DOWN',2),('DOWN',2),('RIGHT',4),('RIGHT',4)]
for name, a in seq:
    f = env.step(Action.simple(a))
    print(name, len(f.layers), blocks(f.grid))
