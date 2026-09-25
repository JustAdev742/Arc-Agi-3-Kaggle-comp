import sys, logging
sys.path.insert(0, '/home/user/Arc-Agi-3-Kaggle-comp')
logging.disable(logging.INFO)
import numpy as np
from arc3.env import LocalEnv, make_arcade, Action
arc = make_arcade('/home/user/Arc-Agi-3-Kaggle-comp/environment_files')
env = LocalEnv(arc, 'tn36'); f = env.reset()
g0 = f.grid.copy()
print('colours', {int(k): int(v) for k, v in zip(*np.unique(g0, return_counts=True))})
def bbox(g, col):
    ys, xs = np.nonzero(g == col)
    return (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()), len(ys)) if len(ys) else None
f = env.step(Action.click(36, 55))
print('layers', len(f.layers), 'final==before', bool((f.grid == g0).all()) if False else int((f.grid != g0).sum()))
for i, L in enumerate(f.layers):
    print(i, {c: bbox(L, c) for c in (0, 1, 9, 11)}, int((L != g0).sum()))
