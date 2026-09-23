"""No-LLM novelty explorer: a breadth-first search over observed frame states.

Purpose: the control arm. It tells us which levels blind state-graph search can clear
and what that costs in actions, which calibrates the exploration budget that any
model-driven harness has to beat. It is also the crash-safe fallback on Kaggle.

Per level it builds a graph whose nodes are frame hashes and whose edges are the
actions tried from each node. Policy each step:
  1. GAME_OVER -> RESET (restarts the level; costs one action).
  2. Current node has untried actions -> try the most promising one.
  3. Otherwise walk (via known edges) to the nearest node with untried actions,
     using RESET as a one-action edge back to the level start when that is shorter.
  4. Nothing left to try -> random fallback (coarse click grid + simple actions).
Level completion clears the graph but keeps per-action "changed something" stats,
which carry over as a cheap prior for the next level.

State keys (``mask_volatile``, default on since 2026-09-23): a frame hash counts every cell, so a step-counter bar
or a blinking indicator makes every step a new state and the graph never closes (ls20's bar shrinks by one cell per
action). Cells that change in at least ``volatile_cell`` (0.5) of the observed transitions, and edge-band rows/columns
that change in at least ``volatile_line`` (0.4) of them, are masked out of the key once ``volatile_min`` transitions
are seen; the graph is re-keyed when the mask grows (nodes that differ only in masked cells merge). Statistics carry
over between levels by default (the HUD stays put; ``volatile_per_level`` restarts them). A 0.2 cell threshold masked
the avatar's own path on ls20 (253-328 cells, 23 states, graph exhausted); 0.5 does not.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from arcengine import GameAction, GameState

from ..env import Action, Frame
from ..perception import background_color, components, grid_hash
from . import register
from .base import Agent, AgentContext, stable_seed

ActionKey = tuple  # (action_id,) or (6, x, y)


@dataclass
class Node:
    h: str
    grid: np.ndarray
    untried: list[ActionKey]
    edges: dict[ActionKey, str] = field(default_factory=dict)
    noop: set = field(default_factory=set)

    @property
    def expanded(self) -> bool:
        return not self.untried


def _key(a: Action) -> ActionKey:
    return a.key()


def _from_key(k: ActionKey) -> Action:
    if k[0] == GameAction.ACTION6.value:
        return Action.click(k[1], k[2])
    return Action(GameAction.from_id(k[0]))


@register("explorer")
class ExplorerAgent(Agent):
    """Breadth-first novelty search over frame hashes. Config keys (ctx.config):
    max_click_targets (48), include_undo (False), click_grid (8), min_obj_size (2)."""

    def __init__(self, ctx: AgentContext):
        super().__init__(ctx)
        cfg = ctx.config
        self.rng = random.Random(ctx.seed * 7919 + stable_seed(ctx.game_id))
        self.max_click_targets = int(cfg.get("max_click_targets", 16))
        self.clicks_always = bool(cfg.get("clicks_always", False))
        self.shape_changed: dict[str, tuple[int, int]] = {}
        self.level_key_probe_done = False
        self.click_shape: dict[tuple[int, int], str] = {}
        self.include_undo = bool(cfg.get("include_undo", False))
        self.click_grid = int(cfg.get("click_grid", 8))
        self.min_obj_size = int(cfg.get("min_obj_size", 2))
        self.nodes: dict[str, Node] = {}
        self.cur: Optional[str] = None
        self.root: Optional[str] = None  # last observed post-RESET / level-start state
        self.level = -1
        self.plan: deque[ActionKey] = deque()
        self.pending: Optional[ActionKey] = None
        # Cross-level priors: how often did each simple action / any click change the frame?
        self.changed: dict[int, list[int]] = {i: [0, 0] for i in range(8)}
        self.n_fallback = 0
        self.n_replans = 0
        self.n_actions = 0
        self.keys_probed: set[int] = set()
        # volatility-masked state keys
        self.mask_volatile = bool(cfg.get("mask_volatile", True))
        self.volatile_cell = float(cfg.get("volatile_cell", 0.5))
        self.volatile_line = float(cfg.get("volatile_line", 0.4))
        self.volatile_min = int(cfg.get("volatile_min", 12))
        self.edge_band = int(cfg.get("edge_band", 6))
        self.cell_changes = np.zeros((64, 64), dtype=np.int32)
        self.row_changes = np.zeros(64, dtype=np.int32)
        self.col_changes = np.zeros(64, dtype=np.int32)
        self.n_trans = 0
        self.mask = np.zeros((64, 64), dtype=bool)
        self.n_rekeys = 0
        self.volatile_per_level = bool(cfg.get("volatile_per_level", False))  # carry-over measured better on ls20 (2 levels vs 1)

    # ---------- candidate actions ----------
    def _candidates(self, frame: Frame) -> list[ActionKey]:
        avail = frame.available_actions or [1, 2, 3, 4, 5, 6, 7]
        keys: list[ActionKey] = []
        simple = [a for a in avail if a in (1, 2, 3, 4, 5, 7) and (a != 7 or self.include_undo)]
        simple.sort(key=lambda a: -self._prior(a))
        keys += [(a,) for a in simple]
        # Clicks only if the game offers them AND keys have shown no effect so far in this level
        # (or the game is click-only). Branching is the enemy of a state-graph search.
        keys_work = any(self.changed[a][0] > 0 for a in (1, 2, 3, 4, 5)) and self.level_key_probe_done
        if 6 in avail and (not simple or not keys_work or self.clicks_always):
            g = frame.grid
            bg = background_color(g)
            objs = [o for o in components(g, ignore=(bg,)) if o.size >= self.min_obj_size]
            # Prefer object shapes that responded to clicks before, then small objects (buttons, pieces).
            objs.sort(key=lambda o: (-self._shape_prior(o.shape_hash), o.size, o.y0, o.x0))
            seen: set[tuple[int, int]] = set()
            clicks: list[ActionKey] = []
            for o in objs:
                x, y = o.center
                if (x, y) in seen:
                    continue
                seen.add((x, y))
                clicks.append((6, x, y))
                if len(clicks) >= self.max_click_targets:
                    break
            if len(clicks) < 4:
                step = 64 // self.click_grid
                for yy in range(step // 2, 64, step):
                    for xx in range(step // 2, 64, step):
                        if (xx, yy) not in seen:
                            clicks.append((6, xx, yy))
            keys += clicks
        return keys

    def _shape_prior(self, sh: str) -> float:
        c, n = self.shape_changed.get(sh, (0, 0))
        return 0.5 if n == 0 else (c + 0.5) / (n + 1.0)

    def _prior(self, action_id: int) -> float:
        n, c = self.changed[action_id][1], self.changed[action_id][0]
        return 1.0 if n == 0 else (c + 0.5) / (n + 1.0)

    # ---------- state keys ----------
    def _key_of(self, grid: np.ndarray) -> str:
        g = np.asarray(grid)
        if not self.mask_volatile or not self.mask.any() or g.shape != self.mask.shape:
            return grid_hash(g)
        return grid_hash(np.where(self.mask, -1, g).astype(np.int16))

    def _update_volatility(self, before: np.ndarray, after: np.ndarray) -> None:
        b, a = np.asarray(before), np.asarray(after)
        if b.shape != (64, 64) or a.shape != (64, 64):
            return
        d = b != a
        self.n_trans += 1
        self.cell_changes += d
        self.row_changes += d.any(axis=1)
        self.col_changes += d.any(axis=0)
        if self.n_trans < self.volatile_min or self.n_trans % 4:
            return
        n = float(self.n_trans)
        m = self.cell_changes / n >= self.volatile_cell
        band = np.zeros(64, dtype=bool)
        band[: self.edge_band] = True
        band[-self.edge_band:] = True
        rows = band & (self.row_changes / n >= self.volatile_line)
        cols = band & (self.col_changes / n >= self.volatile_line)
        m[rows, :] = True
        m[:, cols] = True
        if (m & ~self.mask).any():
            self.mask = self.mask | m
            self._rekey()

    def _rekey(self) -> None:
        """Merge nodes whose frames differ only in (newly) masked cells and re-point every edge."""
        self.n_rekeys += 1
        remap = {h: self._key_of(n.grid) for h, n in self.nodes.items()}
        merged: dict[str, Node] = {}
        for h, n in self.nodes.items():
            nk = remap[h]
            m = merged.get(nk)
            if m is None:
                m = Node(nk, n.grid, list(n.untried), {}, set(n.noop))
                merged[nk] = m
            else:
                m.untried = [k for k in m.untried if k in n.untried]
                m.noop |= n.noop
            for k, tgt in n.edges.items():
                m.edges[k] = remap.get(tgt, tgt)
        for m in merged.values():
            m.untried = [k for k in m.untried if k not in m.edges]
            m.noop = {k for k in m.noop if m.edges.get(k) == m.h}
        self.nodes = merged
        self.cur = remap.get(self.cur, self.cur) if self.cur else self.cur
        self.root = remap.get(self.root, self.root) if self.root else self.root
        self.plan.clear()  # paths were computed on the old keys

    # ---------- graph ----------
    def _node(self, frame: Frame) -> Node:
        h = self._key_of(frame.grid)
        n = self.nodes.get(h)
        if n is None:
            n = Node(h, frame.grid.copy(), self._candidates(frame))
            self.nodes[h] = n
        return n

    def _shape_at(self, frame: Frame, x: int, y: int) -> Optional[str]:
        g = frame.grid
        bg = background_color(g)
        for o in components(g, ignore=(bg,)):
            if o.y0 <= y <= o.y1 and o.x0 <= x <= o.x1 and o.mask[y - o.y0, x - o.x0]:
                return o.shape_hash
        return None

    def _new_level(self, frame: Frame) -> None:
        self.level = frame.levels_completed
        if self.volatile_per_level:
            self.cell_changes[:] = 0
            self.row_changes[:] = 0
            self.col_changes[:] = 0
            self.n_trans = 0
            self.mask[:] = False
        self.keys_probed = set()
        self.level_key_probe_done = False
        self.nodes.clear()
        self.plan.clear()
        self.pending = None
        n = self._node(frame)
        self.cur = n.h
        self.root = n.h

    def _path_to_frontier(self) -> Optional[list[ActionKey]]:
        """BFS over known edges from the current node (and from root via RESET) to the
        nearest node with untried actions. Returns the action sequence, or None."""
        if self.cur is None:
            return None
        starts = [(self.cur, [])]
        if self.root is not None and self.root != self.cur and self.root in self.nodes:
            starts.append((self.root, [(0,)]))
        best: Optional[list[ActionKey]] = None
        for start, prefix in starts:
            q = deque([(start, prefix)])
            seen = {start}
            while q:
                h, path = q.popleft()
                if best is not None and len(path) >= len(best):
                    break
                n = self.nodes.get(h)
                if n is None:
                    continue
                if not n.expanded and path:
                    best = path
                    break
                for k, nh in n.edges.items():
                    if nh not in seen and nh in self.nodes:
                        seen.add(nh)
                        q.append((nh, [*path, k]))
        return best

    # ---------- policy ----------
    def act(self, frame: Frame) -> Action:
        if frame.state in (GameState.NOT_PLAYED,):
            self.pending = (0,)
            return Action.reset()
        if frame.levels_completed != self.level:
            self._new_level(frame)
        if frame.state is GameState.GAME_OVER:
            self.plan.clear()
            self.pending = (0,)
            return Action.reset()
        node = self.nodes.get(self.cur or "") or self._node(frame)
        if self.plan:
            k = self.plan.popleft()
            self.pending = k
            return _from_key(k)
        if not node.expanded:
            k = node.untried.pop(0)
            self.pending = k
            return _from_key(k)
        path = self._path_to_frontier()
        if path:
            self.n_replans += 1
            self.plan.extend(path[1:])
            self.pending = path[0]
            return _from_key(path[0])
        # Exhausted: random fallback keeps the game moving (and may reveal new states).
        self.n_fallback += 1
        avail = [a for a in (frame.available_actions or [1, 2, 3, 4, 5, 6]) if a != 0 and (a != 7 or self.include_undo)]
        a = self.rng.choice(avail or [1])
        k = (6, self.rng.randint(0, 63), self.rng.randint(0, 63)) if a == 6 else (a,)
        self.pending = k
        return _from_key(k)

    def observe(self, action: Action, before: Frame, after: Frame) -> None:
        self.n_actions += 1
        k = self.pending or _key(action)
        self.pending = None
        changed = not np.array_equal(before.grid, after.grid)
        if self.mask_volatile and action.action is not GameAction.RESET and after.levels_completed == before.levels_completed \
                and not after.game_over:
            self._update_volatility(before.grid, after.grid)
        self.changed[action.action.value][1] += 1
        self.changed[action.action.value][0] += int(changed)
        if action.action is GameAction.ACTION6 and action.x is not None:
            sh = self._shape_at(before, action.x, action.y)
            if sh is not None:
                c, n = self.shape_changed.get(sh, (0, 0))
                self.shape_changed[sh] = (c + int(changed), n + 1)
        if action.action.value in (1, 2, 3, 4, 5):
            self.keys_probed.add(action.action.value)
            avail = {a for a in (before.available_actions or [1, 2, 3, 4, 5]) if a in (1, 2, 3, 4, 5)}
            if avail <= self.keys_probed:
                self.level_key_probe_done = True
        if after.levels_completed != self.level:
            return  # act() will rebuild the graph for the new level
        prev = self.nodes.get(self.cur or "")
        node = self._node(after)
        if action.action is GameAction.RESET:
            self.root = node.h
        elif prev is not None:
            prev.edges[k] = node.h
            if node.h == prev.h:
                prev.noop.add(k)
        self.cur = node.h
        # If a planned walk diverged from the known graph, drop the plan and re-plan.
        if self.plan and prev is not None and prev.edges.get(k) not in (None, node.h):
            self.plan.clear()

    def stats(self) -> dict:
        return {"nodes_last_level": len(self.nodes), "replans": self.n_replans, "fallbacks": self.n_fallback,
                "masked_cells": int(self.mask.sum()), "rekeys": self.n_rekeys,
                "change_rate": {a: round(c / n, 3) for a, (c, n) in self.changed.items() if n}}
