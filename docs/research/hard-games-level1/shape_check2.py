import sys, logging
sys.path.insert(0, '/home/user/Arc-Agi-3-Kaggle-comp')
logging.disable(logging.INFO)
import numpy as np
from collections import defaultdict
from arc3.env import LocalEnv, make_arcade
arc = make_arcade('/home/user/Arc-Agi-3-Kaggle-comp/environment_files')
game = sys.argv[1]
env = LocalEnv(arc, game)
g = env.reset().grid.copy()
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
def rk(b): return min((np.rot90(b,k).shape, np.rot90(b,k).tobytes()) for k in range(4))
groups = defaultdict(list)
for col, cells in comps(g):
    if len(cells) > 200: continue
    b = bitmap(cells); ys=[y for y,_ in cells]; xs=[x for _,x in cells]
    groups[rk(b)].append((col, min(ys), min(xs), len(cells)))
for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    cols = {c for c,_,_,_ in v}
    if len(v) > 1 and len(cols) > 1:
        print(k[0], 'n=%d' % len(v), sorted(v)[:12])
