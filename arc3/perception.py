"""Exact programmatic perception for ARC-AGI-3 frames.

Everything here is deterministic numpy. The frame is exact data, so nothing in this
module should ever be asked of a model: grid parsing, connected components, diffs,
click-target enumeration, hashing, and rendering.

Coordinate convention: grids are indexed ``grid[y, x]`` (row, column), matching the
engine's ``(x, y)`` for ACTION6 where ``x`` is the column and ``y`` the row, origin
top-left. ``Obj.center`` returns ``(x, y)`` ready to pass to ACTION6.
"""
from __future__ import annotations

import hashlib
import io
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

GRID = 64
NUM_COLORS = 16

# Letter code per color, used for compact ASCII views. Index = color value.
COLOR_CHARS = "0123456789ABCDEF"

# RGB palette matching the official ARC-AGI-3 viewer (Tufa Labs' duck uses the same).
PALETTE: dict[int, tuple[int, int, int]] = {
    0: (255, 255, 255), 1: (204, 204, 204), 2: (153, 153, 153), 3: (102, 102, 102),
    4: (51, 51, 51), 5: (0, 0, 0), 6: (229, 58, 163), 7: (255, 123, 204),
    8: (249, 60, 49), 9: (30, 147, 255), 10: (136, 216, 241), 11: (255, 220, 0),
    12: (255, 133, 27), 13: (146, 18, 49), 14: (79, 204, 48), 15: (163, 86, 214),
}


def to_grid(frame: Any) -> np.ndarray:
    """Return the *last* layer of a frame as an int16 2-D array.

    Accepts ``FrameData.frame`` (list of 2-D lists), ``FrameDataRaw.frame`` (list of
    ndarrays), a bare 2-D list/ndarray, or an object with a ``.frame`` attribute.
    """
    if hasattr(frame, "frame") and not isinstance(frame, (list, tuple, np.ndarray)):
        frame = frame.frame
    if isinstance(frame, np.ndarray):
        if frame.ndim == 3:
            frame = frame[-1]
        return np.asarray(frame, dtype=np.int16)
    if isinstance(frame, (list, tuple)):
        if len(frame) == 0:
            return np.zeros((GRID, GRID), dtype=np.int16)
        first = frame[0]
        if isinstance(first, np.ndarray) and first.ndim == 2:
            return np.asarray(frame[-1], dtype=np.int16)
        if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], (list, tuple)):
            return np.asarray(frame[-1], dtype=np.int16)  # list of layers
        return np.asarray(frame, dtype=np.int16)  # single 2-D list
    raise TypeError(f"cannot convert {type(frame)!r} to grid")


def grid_hash(grid: np.ndarray) -> str:
    g = np.ascontiguousarray(np.asarray(grid, dtype=np.int16))
    return hashlib.blake2b(g.tobytes() + bytes(g.shape), digest_size=12).hexdigest()


def background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def color_histogram(grid: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(grid, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


@dataclass(frozen=True)
class Obj:
    """One 4-connected same-color component."""

    id: int
    color: int
    y0: int
    x0: int
    y1: int  # inclusive
    x1: int  # inclusive
    size: int
    shape_hash: str  # color + mask, position-independent
    mask: np.ndarray  # bool array of shape (h, w) over the bbox

    @property
    def h(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def w(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def center(self) -> tuple[int, int]:
        """(x, y) of a pixel guaranteed to belong to the object, near its bbox centre."""
        cy, cx = (self.y0 + self.y1) // 2, (self.x0 + self.x1) // 2
        if self.mask[cy - self.y0, cx - self.x0]:
            return cx, cy
        ys, xs = np.nonzero(self.mask)
        d = (ys + self.y0 - cy) ** 2 + (xs + self.x0 - cx) ** 2
        k = int(np.argmin(d))
        return int(xs[k] + self.x0), int(ys[k] + self.y0)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1

    def is_rect(self) -> bool:
        return bool(self.mask.all())

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id, "color": self.color, "x": self.x0, "y": self.y0,
            "w": self.w, "h": self.h, "size": self.size, "rect": self.is_rect(),
            "shape": self.shape_hash[:8],
        }


def components(grid: np.ndarray, *, connectivity: int = 4, ignore: Iterable[int] = ()) -> list[Obj]:
    """Label same-color connected components. Ordered top-left first (reading order)."""
    g = np.asarray(grid)
    h, w = g.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    ignore_set = {int(c) for c in ignore}
    objs: list[Obj] = []
    if connectivity == 4:
        nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    next_id = 0
    for y in range(h):
        for x in range(w):
            if labels[y, x] != -1:
                continue
            c = int(g[y, x])
            if c in ignore_set:
                labels[y, x] = -2
                continue
            stack = [(y, x)]
            labels[y, x] = next_id
            pts = []
            while stack:
                cy, cx = stack.pop()
                pts.append((cy, cx))
                for dy, dx in nbrs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] == -1 and g[ny, nx] == c:
                        labels[ny, nx] = next_id
                        stack.append((ny, nx))
            ys = np.fromiter((p[0] for p in pts), dtype=np.int32, count=len(pts))
            xs = np.fromiter((p[1] for p in pts), dtype=np.int32, count=len(pts))
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
            mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
            mask[ys - y0, xs - x0] = True
            sh = hashlib.blake2b(bytes([c]) + np.packbits(mask).tobytes() + bytes(mask.shape), digest_size=8).hexdigest()
            objs.append(Obj(next_id, c, y0, x0, y1, x1, len(pts), sh, mask))
            next_id += 1
    return objs


@dataclass(frozen=True)
class Diff:
    changed: int
    bbox: tuple[int, int, int, int] | None  # x0, y0, x1, y1 inclusive
    transitions: dict[tuple[int, int], int]  # (from_color, to_color) -> count
    mask: np.ndarray

    @property
    def empty(self) -> bool:
        return self.changed == 0

    def summary(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "bbox": self.bbox,
            "transitions": {f"{a}->{b}": n for (a, b), n in sorted(self.transitions.items(), key=lambda kv: -kv[1])[:8]},
        }


def diff(a: np.ndarray, b: np.ndarray) -> Diff:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return Diff(int(max(a.size, b.size)), (0, 0, b.shape[1] - 1, b.shape[0] - 1), {}, np.ones(b.shape, dtype=bool))
    mask = a != b
    n = int(mask.sum())
    if n == 0:
        return Diff(0, None, {}, mask)
    ys, xs = np.nonzero(mask)
    trans = Counter(zip(a[mask].tolist(), b[mask].tolist()))
    return Diff(n, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), {(int(k[0]), int(k[1])): int(v) for k, v in trans.items()}, mask)


def detect_scale(grid: np.ndarray) -> int:
    """Detect the engine's integer upscale factor (1, 2, 4, ...) by checking uniform blocks."""
    g = np.asarray(grid)
    h, w = g.shape
    for s in (8, 4, 3, 2):
        if h % s or w % s:
            continue
        blocks = g.reshape(h // s, s, w // s, s)
        if (blocks == blocks[:, :1, :, :1]).all():
            return s
    return 1


def downscale(grid: np.ndarray, scale: int | None = None) -> tuple[np.ndarray, int]:
    s = detect_scale(grid) if scale is None else scale
    if s <= 1:
        return np.asarray(grid), 1
    return np.asarray(grid)[::s, ::s], s


def click_targets(grid: np.ndarray, *, max_targets: int = 96, ignore_background: bool = True) -> list[tuple[int, int, Obj]]:
    """Candidate ACTION6 targets: one (x, y) per non-background component, largest first.

    Big components (walls, frames) come first only if they are not the background;
    callers wanting small interactive things should sort by size ascending.
    """
    bg = background_color(grid) if ignore_background else None
    objs = components(grid, ignore=() if bg is None else (bg,))
    objs.sort(key=lambda o: (-o.size, o.y0, o.x0))
    out = []
    for o in objs[:max_targets]:
        x, y = o.center
        out.append((x, y, o))
    return out


def ascii(grid: np.ndarray, *, scale: int | None = None) -> str:
    g, _ = downscale(grid, scale)
    return "\n".join("".join(COLOR_CHARS[int(v) % NUM_COLORS] for v in row) for row in g)


def tile_map(grid: np.ndarray, tile: int) -> str:
    """Compact text view at the game's logical tile size: one character per tile (the most common colour in the
    block, hex 0-f), so a 64x64 board with 4-cell tiles reads as 16 lines of 16 characters."""
    g = np.asarray(grid)
    t = max(1, int(tile))
    h, w = g.shape
    rows = []
    for y in range(0, h, t):
        row = []
        for x in range(0, w, t):
            block = g[y:y + t, x:x + t].reshape(-1)
            row.append(COLOR_CHARS[int(np.bincount(block, minlength=NUM_COLORS).argmax()) % NUM_COLORS])
        rows.append("".join(row))
    return "\n".join(rows)


def render_png(grid: np.ndarray, *, scale: int = 8, gridlines: bool = False) -> bytes:
    """Render a grid to PNG bytes with the official palette (needs Pillow)."""
    from PIL import Image

    g = np.asarray(grid)
    h, w = g.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c, col in PALETTE.items():
        rgb[g == c] = col
    img = Image.fromarray(rgb, "RGB").resize((w * scale, h * scale), Image.NEAREST)
    if gridlines and scale >= 4:
        px = img.load()
        for yy in range(0, h * scale, scale):
            for xx in range(w * scale):
                px[xx, yy] = tuple(int(v * 0.85) for v in px[xx, yy])
        for xx in range(0, w * scale, scale):
            for yy in range(h * scale):
                px[xx, yy] = tuple(int(v * 0.85) for v in px[xx, yy])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def objects_summary(grid: np.ndarray, *, limit: int = 40) -> list[dict[str, Any]]:
    bg = background_color(grid)
    objs = components(grid, ignore=(bg,))
    objs.sort(key=lambda o: (-o.size, o.y0, o.x0))
    return [o.summary() for o in objs[:limit]]


def moved_objects(before: Sequence[Obj], after: Sequence[Obj]) -> list[tuple[Obj, Obj, int, int]]:
    """Pair objects with equal shape_hash that changed position: (before, after, dx, dy)."""
    by_hash: dict[str, list[Obj]] = {}
    for o in after:
        by_hash.setdefault(o.shape_hash, []).append(o)
    out = []
    used: set[int] = set()
    for o in before:
        cands = [c for c in by_hash.get(o.shape_hash, []) if c.id not in used]
        if not cands:
            continue
        c = min(cands, key=lambda c: abs(c.x0 - o.x0) + abs(c.y0 - o.y0))
        if (c.x0, c.y0) != (o.x0, o.y0):
            used.add(c.id)
            out.append((o, c, c.x0 - o.x0, c.y0 - o.y0))
    return out


def terminal_layer(layers: Sequence[np.ndarray], before: np.ndarray | None = None, *, jump: float = 2.0,
                   min_jump: int = 20) -> int:
    """Index of the completed level's final board among the layers of the step that completed the level.

    The engine renders one layer per internal step. When the game declares the level won it renders the finished
    board; a level switch inside the same action then renders the next level's start after it. An animated win puts
    the finished board late (cd82's pour, 2026-09-23: layers[14] of 16, while layers[0] is still the board before the
    pour), so the first layer is not the terminal frame. Rule: when the last consecutive change is a jump (at least
    ``jump`` times every earlier change in this step, the winning move's own change from ``before`` included, and at
    least ``min_jump`` cells), the last layer is the next level and the terminal is the one before it; otherwise the
    switch is still pending (the next action shows the new level) and the terminal is the last layer.
    Checked against the engine's exact winning board on 43 dev levels (exp-028): ``jump`` 1.5 to 2.5 are exact on all
    43, 3.0 misses su15 (a pull animation of 103 cells, then a 262-cell switch); ``layers[0]`` is exact on 31.
    """
    n = len(layers)
    if n <= 1:
        return 0
    changes = [int(np.count_nonzero(np.asarray(layers[i]) != np.asarray(layers[i - 1]))) for i in range(1, n)]
    if before is not None and np.shape(before) == np.shape(layers[0]):
        changes.insert(0, int(np.count_nonzero(np.asarray(layers[0]) != np.asarray(before))))
    last, earlier = changes[-1], changes[:-1]
    if last >= min_jump and last >= jump * max(earlier or [0]):
        return n - 2
    return n - 1
