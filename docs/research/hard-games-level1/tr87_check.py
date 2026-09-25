"""Offline check: would a rotation-invariant shape key pair tr87's rotated glyphs on level 1?"""
import sys, logging
sys.path.insert(0, '/home/user/Arc-Agi-3-Kaggle-comp')
logging.disable(logging.INFO)
import numpy as np
from collections import defaultdict
from arc3.env import LocalEnv, make_arcade
arc = make_arcade('/home/user/Arc-Agi-3-Kaggle-comp/environment_files')
env = LocalEnv(arc, 'tr87')
g = env.reset().grid.copy()
bg_counts = np.unique(g, return_counts=True)
print('colours', dict(zip(*[x.tolist() for x in bg_counts])))

def comps(grid):
    H, W = grid.shape; seen = np.zeros_like(grid, bool); out = []
    for r in range(H):
        for c in range(W):
            if seen[r, c]: continue
            col = grid[r, c]; st = [(r, c)]; seen[r, c] = True; cells = []
            while st:
                y, x = st.pop(); cells.append((y, x))
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and grid[ny, nx] == col:
                        seen[ny, nx] = True; st.append((ny, nx))
            out.append((int(col), cells))
    return out

def bitmap(cells):
    ys = [y for y, _ in cells]; xs = [x for _, x in cells]
    b = np.zeros((max(ys)-min(ys)+1, max(xs)-min(xs)+1), np.uint8)
    for y, x in cells: b[y-min(ys), x-min(xs)] = 1
    return b

def key(b):
    return (b.shape, b.tobytes())
def rot_key(b):
    return min(key(np.rot90(b, k)) for k in range(4))
def d4_key(b):
    return min(min(key(np.rot90(b, k)), key(np.rot90(b.T, k))) for k in range(4))

cs = [(col, cells) for col, cells in comps(g) if 5 <= len(cells) <= 25]
rows = []
for col, cells in cs:
    b = bitmap(cells)
    ys = [y for y, _ in cells]; xs = [x for _, x in cells]
    rows.append(dict(col=col, top=min(ys), left=min(xs), n=len(cells), k=key(b), rk=rot_key(b), dk=d4_key(b)))
for name in ('k', 'rk', 'dk'):
    groups = defaultdict(list)
    for r in rows: groups[(r['col'], r[name])].append((r['top'], r['left']))
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print(f'{name}: {len(rows)} small components, {len(multi)} groups with >1 member')
    for (col, _), v in sorted(multi.items(), key=lambda kv: kv[1]):
        print('   colour', col, sorted(v))
