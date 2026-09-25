"""Offline check: what does an object-level diff show for g50t's SPACE (ACTION5) on level 1?"""
import sys, logging
sys.path.insert(0, '/home/user/Arc-Agi-3-Kaggle-comp')
logging.disable(logging.INFO)
import numpy as np
from arc3.env import LocalEnv, make_arcade, Action
arc = make_arcade('/home/user/Arc-Agi-3-Kaggle-comp/environment_files')
env = LocalEnv(arc, 'g50t')
f = env.reset()
def objs(g):
    out = {}
    for col in np.unique(g):
        ys, xs = np.nonzero(g == col)
        if len(ys) < 400:
            out[int(col)] = (len(ys), int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
    return out
def show(tag, before, after, fr):
    ch = np.argwhere(before != after)
    print(f'{tag}: frames={len(fr.layers)} changed={len(ch)}', 'bbox', (ch.min(0).tolist(), ch.max(0).tolist()) if len(ch) else None)
    ob, oa = objs(before), objs(after)
    for c in sorted(set(ob) | set(oa)):
        if ob.get(c) != oa.get(c): print('   colour', c, ob.get(c), '->', oa.get(c))
prev = f.grid.copy()
for name, a in [('RIGHT', 4), ('RIGHT', 4), ('SPACE', 5), ('RIGHT', 4), ('DOWN', 2)]:
    f = env.step(Action.simple(a))
    show(name, prev, f.grid, f)
    prev = f.grid.copy()
print('---- replay with crops')
env2 = LocalEnv(arc, 'g50t'); f = env2.reset()
def crop(g): return '\n'.join(''.join('%x' % v for v in row[13:32]) for row in g[7:20])
for name, a in [('RIGHT', 4), ('RIGHT', 4), ('SPACE', 5), ('RIGHT', 4), ('RIGHT', 4)]:
    f = env2.step(Action.simple(a))
    print(name, 'layers', len(f.layers)); print(crop(f.grid))
    if name == 'SPACE':
        for i in (0, len(f.layers)//2, len(f.layers)-1):
            print('  SPACE layer', i); print(crop(f.layers[i]))
