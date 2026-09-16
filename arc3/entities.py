"""Entity tracking: persistent object ids across frames and symbolic transitions (plan-100 Part II, component A).

Everything here is exact code over the grid. The model receives sentences like
"UP: entity #3 (colour 9, 4x4) moved (0,-4); #7 disappeared" instead of "52 cells changed",
plus roles: static (never changes), HUD (edge strips that shrink/grow), avatar (moves with keys) and its key map.
"""
from __future__ import annotations

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

    # ---------------------------------------------------------------- core
    def reset(self, grid: np.ndarray) -> list[Entity]:
        self.__init__(self.max_log)
        self.bg = background_color(grid)
        self.tile = tile_size(grid)
        self.current = [_obj_to_entity(o, self._new_id()) for o in components(grid, ignore=(self.bg,))]
        for e in self.current:
            self.pos_hist[e.id].append((e.x0, e.y0))
        self._snapshot()
        return self.current

    def _snapshot(self) -> None:
        self.frames.append(tuple(ent_from_tracker(e) for e in self.current))
        for e in self.current:
            if e.shape_hash not in self.shapes:
                self.shapes[e.shape_hash] = e.mask
                register_shapes({e.shape_hash: e.mask})
        if len(self.frames) > self.max_log:
            del self.frames[0]
            del self.actions[0]

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
        for eid, *_ in moved + recolored + reshaped:
            self.changed_ids[eid] += 1
        for eid in appeared + disappeared:
            self.changed_ids[eid] += 1
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
            if e.touches_edge() and (e.w >= 20 or e.h >= 20) and self.changed_ids[e.id] >= 2 and not self.moves.get(e.id):
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
