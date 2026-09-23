#!/usr/bin/env python
"""Optimal-play oracle on the public games: breadth-first search inside the real engine.

A research instrument, never used at play time (on the hidden games the agent has no engine access). It answers
"how short can each level be?" and therefore "how much exploration can an agent afford and still score 100%?":
every completed level is scored min((baseline / actions)^2, 1.15), levels weigh their index, so optimal play on the
heavy later levels can bank the 1.15 cap and pay for an expensive level 1 (docs/research/road-to-100.md, part II).

Search: nodes are deep copies of the engine's game object (about 12 ms each). A node's key is the board rendered
without UI overlays, the game's own `_get_hidden_state()` (what ARC Prize's graph builder keys on), the level index,
and the full render (UI included) with the step-counter lines masked: edge rows and columns (6 px band) that change
on at least half of the state-changing actions of a short random calibration rollout. The full render matters: a
selection shown only in the HUD (ar25) is otherwise merged away and the search reports "exhausted" wrongly. The
action alphabet is `_get_valid_actions()` (legal keys and the sprites tagged sys_click); when ACTION6 is legal but
the engine lists no clickable sprite (bp35, cd82, ft09, s5i5, su15 and others tag none), one click per visible
4-connected component of the full render is added (its pixel nearest the box centre, at most 80), so click levels
are solved "under object clicks": an upper bound on the true optimum when a level needs clicks between objects
(su15 pulls toward the click point). GAME_OVER children are pruned. The first child whose level counter increases
ends the search: its path length is the level's optimal action count (exact within the action alphabet when the
search completes within budget; the key ignores the step counter, which cannot shorten a path). When the budget
runs out first, ``depth_lb`` is a lower bound on the optimum (every shorter path was checked). The winning child is
at the next level's start, so levels are solved in sequence.

Usage: .venv/bin/python scripts/oracle_bfs.py [--games ls20,vc33] [--split all] [--max-nodes 40000] [--max-s 240]
       [--workers 4] [--out runs/oracle-bfs/summary.json]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arc3.scoring import LEVEL_CAP, level_score  # noqa: E402

EDGE = 6
MAX_OBJECT_CLICKS = 80


def full_render(g: Any) -> np.ndarray:
    return np.asarray(g.camera.render(g.current_level.get_sprites()), dtype=np.int16)


def board_key(g: Any, mask: Any = None) -> bytes:
    raw = np.asarray(g.camera._raw_render(g.current_level.get_sprites()))
    hidden = np.asarray(g._get_hidden_state())
    h = hashlib.blake2b(digest_size=12)
    h.update(raw.tobytes())
    h.update(hidden.tobytes())
    h.update(int(g._score).to_bytes(2, "little"))
    if mask is not None:
        h.update(np.where(mask, -1, full_render(g)).astype(np.int16).tobytes())
    return h.digest()


def candidate_actions(g: Any) -> list[Any]:
    """The engine's valid actions, plus one click per visible object when ACTION6 is legal and no sprite is tagged."""
    from arcengine import GameAction
    from arcengine.enums import ActionInput

    from arc3.perception import background_color, components

    acts = list(g._get_valid_actions())
    if 6 in g._available_actions and not any(int(getattr(a.id, "value", a.id)) == 6 for a in acts):
        grid = full_render(g)
        bg = background_color(grid)
        seen: set[tuple[int, int]] = set()
        for o in sorted(components(grid, ignore=(bg,)), key=lambda o: o.size)[:MAX_OBJECT_CLICKS]:
            x, y = o.center
            if (x, y) not in seen:
                seen.add((x, y))
                acts.append(ActionInput(id=GameAction.ACTION6.value, data={"x": int(x), "y": int(y)}))
    return acts


def calibrate_mask(game: Any, n: int = 60, seed: int = 0) -> np.ndarray:
    """Edge rows/columns of the full render that change on at least half of the state-changing random actions."""
    from arcengine import GameState

    rng = random.Random(seed)
    root = copy.deepcopy(game)
    g = copy.deepcopy(root)
    start = int(g._score)
    rows = np.zeros(64)
    cols = np.zeros(64)
    changed = 0
    prev = full_render(g)
    for _ in range(n):
        acts = candidate_actions(g)
        if not acts:
            break
        g.perform_action(rng.choice(acts), raw=True)
        if g._state in (GameState.GAME_OVER, GameState.WIN) or int(g._score) != start:
            g = copy.deepcopy(root)
            prev = full_render(g)
            continue
        cur = full_render(g)
        d = cur != prev
        if d.any():
            changed += 1
            rows += d.any(axis=1)
            cols += d.any(axis=0)
        prev = cur
    mask = np.zeros((64, 64), dtype=bool)
    if changed:
        band = np.zeros(64, dtype=bool)
        band[:EDGE] = True
        band[-EDGE:] = True
        mask[band & (rows / changed >= 0.5), :] = True
        mask[:, band & (cols / changed >= 0.5)] = True
    return mask


def bfs_level(game: Any, *, max_nodes: int, max_s: float) -> dict[str, Any]:
    from arcengine import GameState

    start = int(game._score)
    t0 = time.time()
    root = copy.deepcopy(game)
    mask = calibrate_mask(root)
    seen = {board_key(root, mask)}
    q: deque = deque([(root, 0)])
    expanded = 0
    branching: list[int] = []
    info = {"masked_lines": int(mask.all(axis=1).sum() + mask.all(axis=0).sum())}
    while q:
        if expanded >= max_nodes or time.time() - t0 > max_s:
            return {"status": "budget", "depth_lb": q[0][1] + 1, "expanded": expanded, "seen": len(seen),
                    "secs": round(time.time() - t0, 1), **info}
        g, depth = q.popleft()
        expanded += 1
        actions = candidate_actions(g)
        branching.append(len(actions))
        for a in actions:
            c = copy.deepcopy(g)
            c.perform_action(a, raw=True)
            if c._state == GameState.GAME_OVER:
                continue
            if int(c._score) > start or c._state == GameState.WIN:
                return {"status": "found", "optimal": depth + 1, "expanded": expanded, "seen": len(seen),
                        "secs": round(time.time() - t0, 1), "branching": round(float(np.mean(branching)), 1), **info, "next": c}
            k = board_key(c, mask)
            if k in seen:
                continue
            seen.add(k)
            q.append((c, depth + 1))
    return {"status": "exhausted", "expanded": expanded, "seen": len(seen), "secs": round(time.time() - t0, 1), **info}


def solve_game(game_id: str, max_nodes: int, max_s: float) -> dict[str, Any]:
    import logging
    logging.disable(logging.INFO)
    from arc3.env import LocalEnv, make_arcade
    arc = make_arcade(str(ROOT / "environment_files"))
    env = LocalEnv(arc, game_id)
    try:
        g = env.env._game
        baselines = list(env.baselines)
        levels: list[dict[str, Any]] = []
        for i, b in enumerate(baselines):
            r = bfs_level(g, max_nodes=max_nodes, max_s=max_s)
            nxt = r.pop("next", None)
            levels.append({"level": i + 1, "baseline": b, **r})
            if nxt is None:
                break  # later levels are unreachable for the oracle
            g = nxt
        return {"game": game_id, "levels": levels, "win_levels": len(baselines)}
    finally:
        env.close()


def level1_budget(levels: list[dict[str, Any]], n_levels: int) -> dict[str, Any]:
    """With every level from 2 on played at its oracle-optimal count, the most actions level 1 may take while the game
    still scores 100 percent (None when a later level has no oracle solution)."""
    opt = {lv["level"]: lv.get("optimal") for lv in levels}
    base = {lv["level"]: lv["baseline"] for lv in levels}
    if any(opt.get(level) is None for level in range(2, n_levels + 1)) or 1 not in base:
        return {"level1_budget": None}
    total_w = n_levels * (n_levels + 1) / 2
    later = sum(level * level_score(base[level], opt[level]) for level in range(2, n_levels + 1))
    need_s1 = 100.0 * total_w - later  # percent points level 1 must contribute
    if need_s1 <= 0:
        return {"level1_budget": "unbounded", "later_weighted_pct": round(later / (total_w - 1), 1)}
    if need_s1 > LEVEL_CAP:
        return {"level1_budget": 0, "later_weighted_pct": round(later / (total_w - 1), 1)}
    budget = int(base[1] / (need_s1 / 100.0) ** 0.5)
    return {"level1_budget": budget, "later_weighted_pct": round(later / (total_w - 1), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="")
    ap.add_argument("--split", default="all")
    ap.add_argument("--max-nodes", type=int, default=40000)
    ap.add_argument("--max-s", type=float, default=240.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "runs" / "oracle-bfs" / "summary.json"))
    a = ap.parse_args()
    from arc3.splits import resolve
    games = [g for g in a.games.split(",") if g] or resolve(a.split)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(solve_game, games, [a.max_nodes] * len(games), [a.max_s] * len(games)))
    for r in results:
        r.update(level1_budget(r["levels"], r["win_levels"]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"max_nodes": a.max_nodes, "max_s": a.max_s, "wall_s": round(time.time() - t0, 1), "games": results}, indent=1))
    for r in results:
        cells = []
        for lv in r["levels"]:
            o = lv.get("optimal")
            cells.append(f"L{lv['level']} {o if o is not None else lv['status']}/{lv['baseline']}")
        print(f"{r['game']}: {'  '.join(cells)}  | level-1 budget for 100%: {r.get('level1_budget')}")
    print(f"-> {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
