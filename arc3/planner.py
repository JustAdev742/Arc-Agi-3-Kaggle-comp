"""Navigation planner over tracked entities (plan-100 Part II, component C, first slice).

From the tracker's evidence it fits the avatar's move rule (key -> displacement, from the key map), the obstacle
map (non-background cells the avatar has never occupied, plus cells it bumped into), and gives:
  * ``MoveModel.step`` / ``bfs``: shortest key sequence to a goal under the fitted rule;
  * ``MoveModel.predict``: a grid predictor usable with ``set_model`` so the verifier checks it.
The model calls ``plan_to_entity(id)`` or ``plan_to(x, y)`` from the REPL and executes the result with ``act``.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any, Optional

import numpy as np

from .entities import KEY_ACTIONS, Entity, Tracker
from .perception import components

KEYS = ("UP", "DOWN", "LEFT", "RIGHT")


def _norm_key(a: Any) -> Optional[str]:
    if isinstance(a, int):
        a = KEY_ACTIONS.get(a)
    return a if a in KEYS else None


class MoveModel:
    def __init__(self, tracker: Tracker):
        av = tracker.avatar()
        if av is None or not av["entity"]:
            raise ValueError("no avatar yet: press each arrow key at least once so the key map can be learned")
        self.tracker = tracker
        self.avatar_id = int(av["id"])
        self.keymap: dict[str, tuple[int, int]] = {k: tuple(v) for k, v in av["keymap"].items()}
        e = tracker.get(self.avatar_id)
        assert e is not None
        self.mask = e.mask.copy()
        self.color = e.color
        self.h, self.w = self.mask.shape
        self.pos = (e.x0, e.y0)
        self.bg = tracker.bg
        self.optimistic = False
        self.obstacles = self._fit_obstacles()
        self.companions = self._fit_companions()

    def _fit_companions(self) -> list[tuple[int, int, int, np.ndarray]]:
        """Parts that always move with the avatar (ka59's eye, ar25's pupils): (dx, dy, colour, mask) relative to
        the avatar's top-left, from the tracker's identical-move groups. They move with the sprite in predictions."""
        out = []
        for grp in self.tracker.groups():
            if self.avatar_id not in grp:
                continue
            for eid in grp:
                if eid == self.avatar_id:
                    continue
                e = self.tracker.get(eid)
                if e is not None:
                    out.append((e.x0 - self.pos[0], e.y0 - self.pos[1], e.color, e.mask.copy()))
        return out

    # ------------------------------------------------------------ fitting
    def _fit_obstacles(self) -> np.ndarray:
        n = 64
        occ = np.zeros((n, n), dtype=bool)  # cells the avatar has occupied: passable by evidence
        for (x0, y0) in self.tracker.pos_hist.get(self.avatar_id, []):
            occ[y0:y0 + self.h, x0:x0 + self.w] |= self.mask
        obs = np.zeros((n, n), dtype=bool)
        roles = self.tracker.roles()
        for e in self.tracker.current:
            if e.id == self.avatar_id or roles.get(e.id) == "hud":
                continue
            obs[e.y0:e.y1 + 1, e.x0:e.x1 + 1] |= e.mask
        obs &= ~occ
        # Terrain the avatar has stood on is walkable wherever it appears: a floor drawn as one big entity (ka59's
        # white rooms, exp-009) is not a wall. The static layer holds each cell's colour when no mover covers it.
        under = self.tracker.under
        if under is not None and under.shape == (n, n):
            walkable = {int(under[y, x]) for (x0, y0) in self.tracker.pos_hist.get(self.avatar_id, [])
                        for y in range(y0, min(n, y0 + self.h)) for x in range(x0, min(n, x0 + self.w)) if self.mask[y - y0, x - x0]}
            walkable.discard(self.color)
            if self.bg is not None:
                walkable.discard(int(self.bg))
            self.walkable = walkable
            if walkable:
                obs &= ~np.isin(under, sorted(walkable))
        else:
            self.walkable = set()
        # Bumps: a key press that moved nothing marks the cells the avatar would have entered.
        for eid, action, x0, y0 in self.tracker.blocked:
            k = _norm_key(action)
            if eid != self.avatar_id or k is None or k not in self.keymap:
                continue
            dx, dy = self.keymap[k]
            nx, ny = x0 + dx, y0 + dy
            if nx >= 0 and ny >= 0 and nx + self.w <= n and ny + self.h <= n:
                obs[ny:ny + self.h, nx:nx + self.w] |= self.mask
            # bumping the frame edge is handled by bounds in step()
        return obs

    def relax(self) -> "MoveModel":
        """Optimistic copy: only cells the avatar actually bumped into count as obstacles (unknown terrain is
        assumed passable). Plans under it are experiments that the verifier checks step by step."""
        import copy

        m = copy.copy(self)
        n = 64
        obs = np.zeros((n, n), dtype=bool)
        for eid, action, x0, y0 in self.tracker.blocked:
            k = _norm_key(action)
            if eid != self.avatar_id or k is None or k not in self.keymap:
                continue
            dx, dy = self.keymap[k]
            nx, ny = x0 + dx, y0 + dy
            if nx >= 0 and ny >= 0 and nx + self.w <= n and ny + self.h <= n:
                obs[ny:ny + self.h, nx:nx + self.w] |= self.mask
        m.obstacles = obs
        m.optimistic = True
        return m

    # ------------------------------------------------------------ dynamics
    def step(self, pos: tuple[int, int], key: str) -> tuple[int, int]:
        """Position after ``key`` from ``pos`` (unchanged if blocked or out of frame)."""
        if key not in self.keymap:
            return pos
        dx, dy = self.keymap[key]
        nx, ny = pos[0] + dx, pos[1] + dy
        if nx < 0 or ny < 0 or nx + self.w > 64 or ny + self.h > 64:
            return pos
        if (self.obstacles[ny:ny + self.h, nx:nx + self.w] & self.mask).any():
            return pos
        return (nx, ny)

    def locate(self, grid: np.ndarray) -> Optional[tuple[int, int]]:
        """Find the avatar sprite in ``grid`` (same colour and shape), nearest to its last known position.
        The predictor must read the position from the grid it is given, not from the tracker, because the
        tracker may already hold the next frame when a prediction is checked."""
        g = np.asarray(grid)
        best = None
        for o in components(g, ignore=(self.bg,) if self.bg is not None else ()):
            if o.color != self.color or o.mask.shape != self.mask.shape or not np.array_equal(o.mask, self.mask):
                continue
            d = abs(o.x0 - self.pos[0]) + abs(o.y0 - self.pos[1])
            if best is None or d < best[0]:
                best = (d, (o.x0, o.y0))
        return best[1] if best else None

    def predict(self, grid: Any, action: Any) -> np.ndarray:
        """Grid predictor for set_model: move the avatar sprite, everything else unchanged."""
        g = np.asarray(grid).copy()
        k = _norm_key(action if not isinstance(action, tuple) else action[0])
        if k is None:
            return g
        pos = self.locate(g) or self.pos
        new = self.step(pos, k)
        if new == pos:
            return g
        under = self.tracker.under
        fill = self.bg if self.bg is not None else 0

        def erase(x0: int, y0: int, mask: np.ndarray) -> None:
            h, w = mask.shape
            if x0 < 0 or y0 < 0 or x0 + w > 64 or y0 + h > 64:
                return
            region = g[y0:y0 + h, x0:x0 + w]
            if under is not None and under.shape == g.shape:
                # the terrain the sprite stood on, not the background colour (ka59's white floor, exp-009)
                src = under[y0:y0 + h, x0:x0 + w]
                keep = mask & (src != self.color)
                region[keep] = src[keep]
                region[mask & ~keep] = fill
            else:
                region[mask] = fill

        def draw(x0: int, y0: int, mask: np.ndarray, colour: int) -> None:
            h, w = mask.shape
            if x0 < 0 or y0 < 0 or x0 + w > 64 or y0 + h > 64:
                return
            g[y0:y0 + h, x0:x0 + w][mask] = colour

        parts = [(0, 0, self.color, self.mask), *getattr(self, "companions", [])]
        for dx, dy, _, mask in parts:
            erase(pos[0] + dx, pos[1] + dy, mask)
        for dx, dy, colour, mask in parts:
            draw(new[0] + dx, new[1] + dy, mask, colour)
        return g

    # ------------------------------------------------------------ search
    def bfs(self, goal: Callable[[tuple[int, int]], bool], *, start: Optional[tuple[int, int]] = None,
            max_nodes: int = 50000) -> Optional[list[str]]:
        start = start or self.pos
        if goal(start):
            return []
        prev: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
        q = deque([start])
        seen = {start}
        n = 0
        while q and n < max_nodes:
            p = q.popleft()
            n += 1
            for k in KEYS:
                if k not in self.keymap:
                    continue
                nxt = self.step(p, k)
                if nxt == p or nxt in seen:
                    continue
                seen.add(nxt)
                prev[nxt] = (p, k)
                if goal(nxt):
                    path = []
                    cur = nxt
                    while cur != start:
                        cur, key = prev[cur]
                        path.append(key)
                    return path[::-1]
                q.append(nxt)
        return None

    def plan_to_point(self, x: int, y: int) -> Optional[list[str]]:
        """Keys until the avatar's bounding box covers (x, y)."""
        return self.bfs(lambda p: p[0] <= x < p[0] + self.w and p[1] <= y < p[1] + self.h)

    def plan_to_entity(self, target: Entity, *, touch: bool = True) -> Optional[list[str]]:
        """Keys until the avatar overlaps (or, with touch, is adjacent to) the target's bounding box."""
        pad = 1 if touch else 0

        def goal(p: tuple[int, int]) -> bool:
            ax0, ay0, ax1, ay1 = p[0], p[1], p[0] + self.w - 1, p[1] + self.h - 1
            return not (ax1 + pad < target.x0 or ax0 - pad > target.x1 or ay1 + pad < target.y0 or ay0 - pad > target.y1)

        return self.bfs(goal)

    def summary(self) -> dict[str, Any]:
        return {"avatar": self.avatar_id, "keymap": self.keymap, "pos": self.pos, "sprite": (self.w, self.h), "walkable_colours": sorted(getattr(self, "walkable", ())),
                "obstacle_cells": int(self.obstacles.sum())}
