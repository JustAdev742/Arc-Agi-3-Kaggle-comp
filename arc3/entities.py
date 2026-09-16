"""Entity tracking: persistent object ids across frames and symbolic transitions (plan-100 Part II, component A).

Everything here is exact code over the grid. The model receives sentences like
"UP: entity #3 (colour 9, 4x4) moved (0,-4); #7 disappeared" instead of "52 cells changed",
plus roles: static (never changes), HUD (edge strips that shrink/grow), avatar (moves with keys) and its key map.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .dsl import Ent, ent_from_tracker, register_shapes
from .perception import Obj, background_color, components, detect_scale

KEY_ACTIONS = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "ACT", 6: "CLICK", 7: "UNDO", 0: "RESET"}


@dataclass
class Entity:
    id: int
    color: int
    x0: int
    y0: int
    x1: int
    y1: int
    size: int
    shape_hash: str
    mask: np.ndarray = field(repr=False)

    @property
    def w(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def h(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def center(self) -> tuple[int, int]:
        return (self.x0 + self.x1) // 2, (self.y0 + self.y1) // 2

    def touches_edge(self, n: int = 64) -> bool:
        return self.x0 == 0 or self.y0 == 0 or self.x1 == n - 1 or self.y1 == n - 1

    def summary(self, tile: int = 1, role: Optional[str] = None) -> dict[str, Any]:
        d = {"id": self.id, "color": self.color, "x": self.x0, "y": self.y0, "w": self.w, "h": self.h, "size": self.size}
        if tile > 1:
            d["tile"] = (self.x0 // tile, self.y0 // tile)
        if role:
            d["role"] = role
        return d


def tile_size(grid: np.ndarray) -> int:
    """Logical cell size: the engine upscale times the most common small-sprite side, when consistent."""
    s = detect_scale(grid)
    bg = background_color(grid)
    sides = Counter()
    for o in components(grid, ignore=(bg,)):
        if 1 < o.w <= 16 and 1 < o.h <= 16 and o.w == o.h and o.size >= 4:
            sides[o.w] += 1
    if not sides:
        return max(1, s)
    side, n = sides.most_common(1)[0]
    if side % s == 0 and n >= 2:
        return side
    return max(1, s)


def _obj_to_entity(o: Obj, eid: int) -> Entity:
    return Entity(eid, o.color, o.x0, o.y0, o.x1, o.y1, o.size, o.shape_hash, o.mask)


def _bbox_iou(a: Entity, b: Entity) -> float:
    ix = max(0, min(a.x1, b.x1) - max(a.x0, b.x0) + 1)
    iy = max(0, min(a.y1, b.y1) - max(a.y0, b.y0) + 1)
    inter = ix * iy
    if inter == 0:
        return 0.0
    return inter / float(a.w * a.h + b.w * b.h - inter)


class Tracker:
    """Assigns persistent ids to entities across the frames of one level and records events per action."""

    def __init__(self, max_log: int = 5000):
        self.next_id = 1
        self.current: list[Entity] = []
        self.bg: Optional[int] = None
        self.tile = 1
        self.log: list[dict[str, Any]] = []
        self.moves: dict[int, list[tuple[Any, int, int]]] = defaultdict(list)  # id -> [(action, dx, dy)]
        self.changed_ids: Counter = Counter()  # id -> number of transitions in which it changed at all
        self.pos_hist: dict[int, list[tuple[int, int]]] = defaultdict(list)  # id -> [(x0, y0) per frame]
        self.blocked: list[tuple[int, Any, int, int]] = []  # (avatar-candidate id, action, x0, y0) key presses that moved nothing
        self.max_log = max_log
        self.n_transitions = 0
        self.frames: list[tuple[Ent, ...]] = []  # symbolic frame per observed grid (for arc3.dsl)
        self.actions: list[Any] = []  # actions[i] took frames[i] to frames[i+1]
        self.shapes: dict[str, np.ndarray] = {}  # shape hash -> mask, for rendering predicted frames
        self.under: Optional[np.ndarray] = None  # static layer: last seen colour of each cell when no mover covered it
        self.unders: list[np.ndarray] = []  # static layer snapshot per frame (aligned with frames)
        self.occluded_events = 0  # changes attributed to occlusion by movers and therefore not reported

    # ---------------------------------------------------------------- core
    def reset(self, grid: np.ndarray) -> list[Entity]:
        self.__init__(self.max_log)
        self.bg = background_color(grid)
        self.tile = tile_size(grid)
        self.current = [_obj_to_entity(o, self._new_id()) for o in components(grid, ignore=(self.bg,))]
        for e in self.current:
            self.pos_hist[e.id].append((e.x0, e.y0))
        self.under = np.asarray(grid).copy()
        self._snapshot()
        return self.current

    def _snapshot(self) -> None:
        self.frames.append(tuple(ent_from_tracker(e) for e in self.current))
        self.unders.append(self.under.copy() if self.under is not None else None)
        for e in self.current:
            if e.shape_hash not in self.shapes:
                self.shapes[e.shape_hash] = e.mask
                register_shapes({e.shape_hash: e.mask})
        if len(self.frames) > self.max_log:
            del self.frames[0]
            del self.actions[0]
            del self.unders[0]

    def _cells(self, e: Entity, shape: tuple[int, int]) -> np.ndarray:
        m = np.zeros(shape, dtype=bool)
        m[e.y0:e.y1 + 1, e.x0:e.x1 + 1] |= e.mask
        return m

    def _mover_cells(self, ents: list[Entity], shape: tuple[int, int], moved_ids: set[int]) -> np.ndarray:
        """Cells covered by entities that move (known movers plus those that moved in this transition)."""
        m = np.zeros(shape, dtype=bool)
        for e in ents:
            if e.id in moved_ids or self.moves.get(e.id):
                m[e.y0:e.y1 + 1, e.x0:e.x1 + 1] |= e.mask
        return m

    def _new_id(self) -> int:
        i = self.next_id
        self.next_id += 1
        return i

    def update(self, grid: np.ndarray, action: Any = None) -> dict[str, Any]:
        """Match the new frame's components to the tracked entities; return the event record."""
        if self.bg is None:
            self.reset(grid)
            return {"action": action, "moved": [], "appeared": [], "disappeared": [], "recolored": [], "reshaped": [], "same": len(self.current)}
        new_objs = components(grid, ignore=(self.bg,))
        new: list[Entity] = [_obj_to_entity(o, 0) for o in new_objs]
        prev = self.current
        matched_prev: set[int] = set()
        matched_new: set[int] = set()
        moved, recolored, reshaped = [], [], []
        # Pass 1: same shape hash, nearest position (greedy by distance).
        pairs = []
        by_hash: dict[str, list[int]] = defaultdict(list)
        for j, e in enumerate(new):
            by_hash[e.shape_hash].append(j)
        for i, p in enumerate(prev):
            for j in by_hash.get(p.shape_hash, []):
                d = abs(new[j].x0 - p.x0) + abs(new[j].y0 - p.y0)
                pairs.append((d, i, j))
        for d, i, j in sorted(pairs):
            if i in matched_prev or j in matched_new:
                continue
            matched_prev.add(i)
            matched_new.add(j)
            new[j].id = prev[i].id
            if d:
                dx, dy = new[j].x0 - prev[i].x0, new[j].y0 - prev[i].y0
                moved.append((prev[i].id, dx, dy))
                self.moves[prev[i].id].append((action, dx, dy))
        # Pass 2: overlapping bbox, different appearance (recoloured or reshaped in place).
        pairs = []
        for i, p in enumerate(prev):
            if i in matched_prev:
                continue
            for j, e in enumerate(new):
                if j in matched_new:
                    continue
                iou = _bbox_iou(p, e)
                if iou >= 0.3:
                    pairs.append((-iou, i, j))
        for _, i, j in sorted(pairs):
            if i in matched_prev or j in matched_new:
                continue
            matched_prev.add(i)
            matched_new.add(j)
            new[j].id = prev[i].id
            if new[j].color != prev[i].color:
                recolored.append((prev[i].id, prev[i].color, new[j].color))
            else:
                reshaped.append((prev[i].id, prev[i].size, new[j].size))
        appeared = []
        for j, e in enumerate(new):
            if j not in matched_new:
                e.id = self._new_id()
                appeared.append(e.id)
        disappeared = [prev[i].id for i in range(len(prev)) if i not in matched_prev]
        # Occlusion: a change whose cells all lie under a mover (before or after) is terrain being covered or
        # uncovered, not an event. Such entities keep their id and are reported as unchanged.
        shape = tuple(int(v) for v in np.asarray(grid).shape)
        moved_ids = {eid for eid, _, _ in moved}
        cover = self._mover_cells(prev, shape, moved_ids) | self._mover_cells(new, shape, moved_ids)
        if cover.any():
            keep_reshaped = []
            for eid, s0, s1 in reshaped:
                a = next(p for p in prev if p.id == eid)
                b = next(n for n in new if n.id == eid)
                d = self._cells(a, shape) ^ self._cells(b, shape)
                if (d & ~cover).any():
                    keep_reshaped.append((eid, s0, s1))
                else:
                    self.occluded_events += 1
            reshaped = keep_reshaped
            keep_moved = []
            for eid, dx, dy in moved:
                a = next(p for p in prev if p.id == eid)
                b = next(n for n in new if n.id == eid)
                d = self._cells(a, shape) ^ self._cells(b, shape)
                if self.moves.get(eid) or (d & ~cover).any() or not d.any():
                    keep_moved.append((eid, dx, dy))
                else:  # a terrain piece whose bbox shifted because a mover uncovered or covered its edge
                    self.occluded_events += 1
                    self.moves[eid].pop()
                    if not self.moves[eid]:
                        del self.moves[eid]
            moved = keep_moved
            keep_appeared = []
            for eid in appeared:
                b = next(n for n in new if n.id == eid)
                if (self._cells(b, shape) & ~cover).any() and not (self._cells(b, shape) & self._mover_cells(prev, shape, moved_ids)).all():
                    keep_appeared.append(eid)
                else:
                    self.occluded_events += 1  # uncovered terrain
            appeared = keep_appeared
            keep_gone = []
            for eid in disappeared:
                a = next(p for p in prev if p.id == eid)
                if (self._cells(a, shape) & ~cover).any():
                    keep_gone.append(eid)
                else:
                    self.occluded_events += 1  # covered terrain
            disappeared = keep_gone
        for eid, *_ in moved + recolored + reshaped:
            self.changed_ids[eid] += 1
        for eid in appeared + disappeared:
            self.changed_ids[eid] += 1
        # static layer: every cell not under a mover shows its terrain colour
        if self.under is not None:
            g = np.asarray(grid)
            if self.under.shape == g.shape:
                mv_now = self._mover_cells(new, shape, moved_ids)
                self.under[~mv_now] = g[~mv_now]
            else:
                self.under = g.copy()
        if action in ("UP", "DOWN", "LEFT", "RIGHT", 1, 2, 3, 4) and not moved:
            for p in prev:
                if self.moves.get(p.id):  # a key press that moved nothing: evidence of blocking for known movers
                    self.blocked.append((p.id, action, p.x0, p.y0))
        self.current = new
        for e in new:
            self.pos_hist[e.id].append((e.x0, e.y0))
        self.n_transitions += 1
        self.actions.append(action)
        self._snapshot()
        rec = {"action": action, "moved": moved, "appeared": appeared, "disappeared": disappeared,
               "recolored": recolored, "reshaped": reshaped, "same": len(new) - len(moved) - len(appeared) - len(recolored) - len(reshaped)}
        self.log.append(rec)
        del self.log[: -self.max_log]
        return rec

    # ---------------------------------------------------------------- derived
    def get(self, eid: int) -> Optional[Entity]:
        for e in self.current:
            if e.id == eid:
                return e
        return None

    def static_ids(self) -> list[int]:
        if self.n_transitions == 0:
            return []
        return [e.id for e in self.current if self.changed_ids[e.id] == 0]

    def hud_ids(self) -> list[int]:
        """Edge-hugging strips that were reshaped or recoloured in at least two transitions."""
        out = []
        for e in self.current:
            if not (e.touches_edge() and (e.w >= 20 or e.h >= 20) and self.changed_ids[e.id] >= 2):
                continue
            hist = self.pos_hist.get(e.id, [])
            # a bar may shift along its own axis as it shrinks from one end; it never moves across it
            along_axis = all(y == hist[0][1] for _, y in hist) if e.w >= e.h else all(x == hist[0][0] for x, _ in hist)
            if along_axis:
                out.append(e.id)
        return out

    def avatar(self) -> Optional[dict[str, Any]]:
        """The entity that moves most often on key actions, with its observed key map."""
        best: Optional[tuple[int, int]] = None
        for eid, mv in self.moves.items():
            n_keys = sum(1 for a, _, _ in mv if a in ("UP", "DOWN", "LEFT", "RIGHT") or a in (1, 2, 3, 4))
            if n_keys >= 2 and (best is None or n_keys > best[1]):
                best = (eid, n_keys)
        if best is None:
            return None
        eid = best[0]
        keymap: dict[str, tuple[int, int]] = {}
        for a in ("UP", "DOWN", "LEFT", "RIGHT"):
            ds = [(dx, dy) for act, dx, dy in self.moves[eid] if act == a or KEY_ACTIONS.get(act) == a]
            if ds:
                keymap[a] = Counter(ds).most_common(1)[0][0]
        e = self.get(eid)
        return {"id": eid, "key_moves": best[1], "keymap": keymap, "alive": e is not None,
                "entity": e.summary(self.tile) if e else None}

    def groups(self) -> list[list[int]]:
        """Entities that moved identically in the same transitions at least twice (multi-part sprites)."""
        sig: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for t, rec in enumerate(self.log):
            for eid, dx, dy in rec["moved"]:
                sig[eid].append((t, dx, dy))
        by_sig: dict[tuple, list[int]] = defaultdict(list)
        for eid, s in sig.items():
            if len(s) >= 2:
                by_sig[tuple(s)].append(eid)
        return [ids for ids in by_sig.values() if len(ids) > 1]

    def roles(self) -> dict[int, str]:
        r: dict[int, str] = {}
        for eid in self.static_ids():
            r[eid] = "static"
        for eid in self.hud_ids():
            r[eid] = "hud"
        av = self.avatar()
        if av:
            r[av["id"]] = "avatar"
        for e in self.current:
            r.setdefault(e.id, "dynamic" if self.changed_ids[e.id] else "unknown")
        return r

    # ---------------------------------------------------------------- compound sprites
    def compound_frames(self) -> list[tuple[Ent, ...]]:
        """The symbolic frames with multi-part sprites merged into one entity each: parts strictly inside a
        mover's bounding box (eyes, markings) and groups that always move together (stacked colour bands).
        Ids of parts are unstable, so merging is done on geometry and on the co-movement groups; the compound
        keeps the container's (or the smallest member's) id, the colour of its largest part, the union box and a
        shape hash of the multi-colour patch (registered so mask-based blocking keeps working)."""
        movers = {eid for eid, mv in self.moves.items() if mv}
        groups = [set(g) for g in self.groups()]
        out: list[tuple[Ent, ...]] = []
        for frame in self.frames:
            by_id = {e.id: e for e in frame}
            parts: dict[int, list[Ent]] = defaultdict(list)  # container id -> parts
            taken: set[int] = set()
            for e in frame:
                if e.id in movers and e.id not in taken:
                    for p in frame:
                        if p.id == e.id or p.id in taken or p.id in movers and p.size >= e.size:
                            continue
                        if e.x0 < p.x0 and p.x1 < e.x1 and e.y0 < p.y0 and p.y1 < e.y1:
                            parts[e.id].append(p)
                            taken.add(p.id)
            for g in groups:
                members = [by_id[i] for i in g if i in by_id and i not in taken]
                if len(members) > 1:
                    head = min(members, key=lambda m: m.id)
                    for m in members:
                        if m.id != head.id:
                            parts[head.id].append(m)
                            taken.add(m.id)
            if not parts:
                out.append(frame)
                continue
            merged = []
            for e in frame:
                if e.id in taken:
                    continue
                if e.id in parts:
                    merged.append(self._merge(e, parts[e.id]))
                else:
                    merged.append(e)
            out.append(tuple(merged))
        return out

    def _merge(self, head: Ent, parts: list[Ent]) -> Ent:
        members = [head] + parts
        x0, y0 = min(m.x0 for m in members), min(m.y0 for m in members)
        x1, y1 = max(m.x0 + m.w - 1 for m in members), max(m.y0 + m.h - 1 for m in members)
        patch = np.full((y1 - y0 + 1, x1 - x0 + 1), -1, dtype=np.int16)
        for m in sorted(members, key=lambda m: m.size):  # small parts drawn last so they show through
            mask = self.shapes.get(m.shape)
            if mask is None or mask.shape != (m.h, m.w):
                mask = np.ones((m.h, m.w), dtype=bool)
            region = patch[m.y0 - y0:m.y0 - y0 + m.h, m.x0 - x0:m.x0 - x0 + m.w]
            region[mask] = m.color
        shape = "c" + hashlib.blake2b(patch.tobytes(), digest_size=6).hexdigest()
        union = patch >= 0
        if shape not in self.shapes:
            self.shapes[shape] = union
            register_shapes({shape: union})
        big = max(members, key=lambda m: m.size)
        return Ent(head.id, big.color, x0, y0, x1 - x0 + 1, y1 - y0 + 1, int(union.sum()), shape)

    def entities_summary(self, limit: int = 40) -> list[dict[str, Any]]:
        roles = self.roles()
        ents = sorted(self.current, key=lambda e: (roles.get(e.id) == "static", -e.size))
        return [e.summary(self.tile, roles.get(e.id)) for e in ents[:limit]]

    @staticmethod
    def describe(rec: dict[str, Any], tracker: Optional["Tracker"] = None, max_items: int = 6) -> str:
        """One-line human/LLM-readable description of an event record."""
        a = rec.get("action")
        name = KEY_ACTIONS.get(a, a) if isinstance(a, int) else a
        parts = []
        for eid, dx, dy in rec["moved"][:max_items]:
            e = tracker.get(eid) if tracker else None
            tag = f"#{eid}" + (f" (c{e.color} {e.w}x{e.h})" if e else "")
            parts.append(f"{tag} moved ({dx:+d},{dy:+d})")
        if len(rec["moved"]) > max_items:
            parts.append(f"+{len(rec['moved']) - max_items} more moved")
        if rec["appeared"]:
            parts.append("appeared " + ", ".join(f"#{i}" for i in rec["appeared"][:max_items]))
        if rec["disappeared"]:
            parts.append("disappeared " + ", ".join(f"#{i}" for i in rec["disappeared"][:max_items]))
        for eid, c0, c1 in rec["recolored"][:max_items]:
            parts.append(f"#{eid} colour {c0}->{c1}")
        for eid, s0, s1 in rec["reshaped"][:max_items]:
            parts.append(f"#{eid} size {s0}->{s1}")
        body = "; ".join(parts) if parts else "no entity changed"
        return f"{name}: {body}"
