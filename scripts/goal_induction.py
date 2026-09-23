#!/usr/bin/env python
"""exp-028: goal induction by contrast, from free exploration (research instrument).

Question: once a level is solved by brute exploration (level 1 is nearly free under RHAE when the later levels beat
the human count; docs/research/road-to-100-v2.md), can the win condition be read off the data, and does it hold on
the next levels? The census (docs/research/win-conditions-dev.md) found that the kind of win condition never changes
within a game, so a goal induced on level 1 is worth every later level.

collect   Plays each game with a data source and keeps, per completed level: a sample of distinct states visited
          before the win (negatives: the goal did not hold there, or the level would have ended), the states on the
          winning path, the exact terminal frame (the engine's ``next_level()`` is wrapped to render the board at the
          moment the win is detected: instrumentation for measurement, never available to the agent) and the layers
          of the winning step (what the agent sees). Sources: ``explorer`` (the novelty explorer with
          volatility-masked keys: what an agent can do without a model) and ``oracle`` (breadth-first search in the
          engine: every state closer than the optimum, and more solved levels for the transfer test).
analyze   Goal candidates of ``arc3.dsl.goal_predicates`` (the existing, existential grammar) plus a for-all relational
          extension, scored by contrast: true on the terminal frame and false on every negative. Per game: survivors
          on level 1, which of them still hold on each later solved level (transfer), survivors of all levels
          jointly, and which winning-step layer equals the exact terminal frame (the REPL harness archives layers[0]).

Usage:
  .venv/bin/python scripts/goal_induction.py collect --source explorer --split dev --workers 3
  .venv/bin/python scripts/goal_induction.py collect --source oracle --games dc22,ls20 --workers 3
  .venv/bin/python scripts/goal_induction.py analyze [--run runs/exp028-goal-induction]
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
import random
import sys
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

RUN = ROOT / "runs" / "exp028-goal-induction"


def _full_render(game: Any) -> np.ndarray:
    return np.asarray(game.camera.render(game.current_level.get_sprites()), dtype=np.int8).copy()


def _wrap_next_level(game: Any, sink: dict[str, Any]) -> None:
    """Instrumentation: render the board at the moment the game declares the level won."""
    orig = game.next_level

    def wrapped() -> None:
        sink["terminal"] = _full_render(game)
        return orig()

    game.next_level = wrapped


class _Sampler:
    """Reservoir sample of distinct states (by key) plus the most recent distinct ones (near misses)."""

    def __init__(self, cap: int, recent: int, seed: int):
        self.cap = cap
        self.rng = random.Random(seed)
        self.keys: set[Any] = set()
        self.sample: list[np.ndarray] = []
        self.recent: deque[np.ndarray] = deque(maxlen=recent)
        self.n = 0

    def add(self, key: Any, grid: np.ndarray) -> None:
        if key in self.keys:
            return
        self.keys.add(key)
        g = np.asarray(grid, dtype=np.int8).copy()
        self.recent.append(g)
        self.n += 1
        if len(self.sample) < self.cap:
            self.sample.append(g)
        else:
            j = self.rng.randrange(self.n)
            if j < self.cap:
                self.sample[j] = g


# ---------------------------------------------------------------------------------------------------- collect
def collect_explorer(game_id: str, out: str, max_actions: int, max_s: float, max_levels: int, max_neg: int) -> dict[str, Any]:
    import logging

    logging.disable(logging.INFO)
    from arcengine import GameState

    from arc3.agents import get as get_agent
    from arc3.agents.base import AgentContext
    from arc3.env import LocalEnv, make_arcade

    arc = make_arcade(str(ROOT / "environment_files"))
    env = LocalEnv(arc, game_id)
    game = env.env._game
    exact: dict[str, Any] = {}
    _wrap_next_level(game, exact)
    ctx = AgentContext(game_id=game_id, seed=0, deadline=time.time() + max_s, max_actions=max_actions,
                       baselines=list(env.baselines), config={"mask_volatile": True},
                       log=logging.getLogger(f"exp028.{game_id}"))
    agent = get_agent("explorer")(ctx)
    levels: list[dict[str, Any]] = []
    frame = env.frame
    sampler = _Sampler(max_neg, 40, len(levels))
    start = np.asarray(frame.grid, dtype=np.int8).copy()
    level_actions = 0
    skip_next = False
    t0 = time.time()
    try:
        while env.step_count < max_actions and time.time() - t0 < max_s and not frame.done:
            action = agent.act(frame)
            before = frame
            frame = env.step(action)
            agent.observe(action, before, frame)
            level_actions += 1
            if frame.levels_completed > before.levels_completed:
                levels.append({
                    "level": before.levels_completed + 1, "source": "explorer", "actions": level_actions,
                    "baseline": env.baselines[before.levels_completed] if before.levels_completed < len(env.baselines) else None,
                    "start": start, "negatives": list(sampler.sample), "recent": list(sampler.recent), "n_distinct": sampler.n,
                    "pre": np.asarray(before.grid, dtype=np.int8).copy(), "win_action": str(action),
                    "terminal_exact": exact.pop("terminal", None),
                    "layers": [np.asarray(x, dtype=np.int8).copy() for x in frame.layers],
                    "switch_pending": bool(getattr(game, "_next_level", False)),
                })
                if len(levels) >= max_levels:
                    break
                sampler = _Sampler(max_neg, 40, len(levels))
                start = np.asarray(frame.grid, dtype=np.int8).copy()
                level_actions = 0
                skip_next = bool(getattr(game, "_next_level", False))  # the grid still shows the old level
                continue
            if frame.state is GameState.GAME_OVER or action.action.value == 0:
                continue
            if skip_next:
                skip_next = False
                start = np.asarray(frame.grid, dtype=np.int8).copy()
                continue
            sampler.add(agent._key_of(frame.grid), frame.grid)
    finally:
        env.close()
    rec = {"game": game_id, "source": "explorer", "actions": env.step_count, "secs": round(time.time() - t0, 1),
           "win_levels": env.frame.win_levels, "levels": levels}
    _save(out, game_id, rec)
    return {"game": game_id, "levels": len(levels), "actions": env.step_count, "secs": rec["secs"]}


def _bfs_record(game: Any, *, max_nodes: int, max_s: float, max_neg: int, seed: int) -> dict[str, Any]:
    """oracle_bfs.bfs_level with recording: a reservoir of visited states, the winning path, the exact terminal."""
    from arcengine import GameState
    from oracle_bfs import board_key, calibrate_mask, candidate_actions

    start_score = int(game._score)
    t0 = time.time()
    root = copy.deepcopy(game)
    mask = calibrate_mask(root)
    seen = {board_key(root, mask)}
    nodes: list[tuple[int, Any]] = [(-1, None)]  # (parent index, action) per node
    q: deque = deque([(root, 0, 0)])
    sampler = _Sampler(max_neg, 0, seed)
    expanded = 0
    while q:
        if expanded >= max_nodes or time.time() - t0 > max_s:
            return {"status": "budget", "depth_lb": q[0][1] + 1, "expanded": expanded, "secs": round(time.time() - t0, 1)}
        g, depth, idx = q.popleft()
        expanded += 1
        for a in candidate_actions(g):
            c = copy.deepcopy(g)
            c.perform_action(a, raw=True)
            if c._state == GameState.GAME_OVER:
                continue
            if int(c._score) > start_score or c._state == GameState.WIN:
                # replay the winning action on a fresh copy with the terminal instrumentation
                probe = copy.deepcopy(g)
                exact: dict[str, Any] = {}
                _wrap_next_level(probe, exact)
                raw = probe.perform_action(a, raw=True)
                path: list[Any] = [a]
                j = idx
                while nodes[j][0] >= 0:
                    path.append(nodes[j][1])
                    j = nodes[j][0]
                path.reverse()
                walk = copy.deepcopy(root)
                on_path = [_full_render(walk)]
                for pa in path[:-1]:
                    walk.perform_action(pa, raw=True)
                    on_path.append(_full_render(walk))
                return {"status": "found", "optimal": depth + 1, "expanded": expanded, "secs": round(time.time() - t0, 1),
                        "negatives": list(sampler.sample), "recent": on_path, "n_distinct": len(seen),
                        "start": on_path[0], "pre": _full_render(g), "win_action": _action_str(a),
                        "terminal_exact": exact.get("terminal"),
                        "layers": [np.asarray(x, dtype=np.int8).copy() for x in (raw.frame or [])],
                        "switch_pending": bool(getattr(probe, "_next_level", False)), "next": c}
            k = board_key(c, mask)
            if k in seen:
                continue
            seen.add(k)
            nodes.append((idx, a))
            sampler.add(k, _full_render(c))
            q.append((c, depth + 1, len(nodes) - 1))
    return {"status": "exhausted", "expanded": expanded, "secs": round(time.time() - t0, 1)}


def _action_str(a: Any) -> str:
    aid = int(getattr(a.id, "value", a.id))
    d = getattr(a, "data", None) or {}
    return f"ACTION{aid}" + (f"({d.get('x')},{d.get('y')})" if aid == 6 else "")


def collect_oracle(game_id: str, out: str, max_nodes: int, max_s: float, max_levels: int, max_neg: int) -> dict[str, Any]:
    import logging

    logging.disable(logging.INFO)
    from arc3.env import LocalEnv, make_arcade

    arc = make_arcade(str(ROOT / "environment_files"))
    env = LocalEnv(arc, game_id)
    t0 = time.time()
    levels: list[dict[str, Any]] = []
    stop = ""
    try:
        g = env.env._game
        for i, b in enumerate(env.baselines[:max_levels]):
            r = _bfs_record(g, max_nodes=max_nodes, max_s=max_s, max_neg=max_neg, seed=i)
            if r["status"] != "found":
                stop = f"L{i + 1} {r['status']}"
                break
            g = r.pop("next")
            levels.append({"level": i + 1, "source": "oracle", "actions": r["optimal"], "baseline": b, **r})
    finally:
        env.close()
    rec = {"game": game_id, "source": "oracle", "secs": round(time.time() - t0, 1), "win_levels": len(env.baselines),
           "levels": levels, "stop": stop}
    _save(out, game_id, rec)
    return {"game": game_id, "levels": len(levels), "secs": rec["secs"], "stop": stop}


def _save(out: str, game_id: str, rec: dict[str, Any]) -> None:
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / f"{game_id}.pkl.gz", "wb") as f:
        pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------------------------------- analyze
def _frames(grids: list[np.ndarray], bg: int) -> list[Any]:
    from arc3 import dsl

    return [dsl.frame_from_grid(np.asarray(g, dtype=np.int16), bg) for g in grids]


def _holds(p: Callable, f: Any) -> bool:
    try:
        return bool(p(f))
    except Exception:
        return False


def _score_level(preds: list[tuple[str, str, tuple, Callable]], terminal: Any, negatives: list[Any], limit: int) -> dict[str, int]:
    """For each predicate true on the terminal frame: on how many negatives it also holds (0 = survivor), counted up
    to ``limit + 1`` (beyond that it is neither a survivor nor a near miss)."""
    out: dict[str, int] = {}
    for name, _kind, _args, p in preds:
        if not _holds(p, terminal):
            continue
        n = 0
        for f in negatives:
            if _holds(p, f):
                n += 1
                if n > limit:
                    break
        out[name] = n
    return out


def _all_predicates(colors: list[int], start: Any) -> list[tuple[str, str, tuple, Callable]]:
    """dsl.goal_predicates' kinds, enumerated as plain predicates so every candidate can be scored on every level
    (goal_predicates itself returns only strict survivors), plus the universal kinds (dsl.FORALL_RELS)."""
    from arc3 import dsl

    preds: list[tuple[str, str, tuple, Callable]] = []
    shapes = {(e.color, e.shape) for e in start if e.size <= 400 and e.shape}
    for c, sh in sorted(x for x in shapes if sum(1 for e in start if (e.color, e.shape) == x) <= 4):
        preds.append((f"vanish(colour {c}, shape {sh[:8]})", "vanish_shape", (c, sh),
                      lambda f, c=c, sh=sh: not any(e.color == c and e.shape == sh for e in f)))
    for c in colors:
        preds.append((f"none_left(colour {c})", "none_left", (c,), lambda f, c=c: not any(e.color == c for e in f)))
        for n in range(1, 5):
            preds.append((f"count(colour {c}) == {n}", "count", (c, n), lambda f, c=c, n=n: sum(1 for e in f if e.color == c) == n))
        preds.append((f"aligned(colour {c})", "aligned", (c,), lambda f, c=c: dsl._aligned([e for e in f if e.color == c])))
    for a in colors:
        for b in colors:
            for kind in ("overlap", "touch", "same_box", "same_columns", "same_rows", "inside", "shape_matches"):
                if kind in ("overlap", "touch", "inside") and a == b:
                    continue
                p = dsl.goal_predicate(kind, (a, b))
                if p is not None:
                    preds.append((f"{kind}(colour {a}, colour {b})", kind, (a, b), p))
            for rel in dsl.FORALL_RELS:
                p = dsl.goal_predicate(f"every_{rel}", (a, b))
                if p is not None:
                    preds.append((f"every_{rel}(colour {a}, colour {b})", f"every_{rel}", (a, b), p))
    return preds


def _terminal_layer(lv: dict[str, Any]) -> str:
    t = lv.get("terminal_exact")
    if t is None:
        return "no exact terminal"
    for i, layer in enumerate(lv.get("layers") or []):
        if layer.shape == t.shape and np.array_equal(layer, t):
            n = len(lv["layers"])
            return f"layers[{i}] of {n}" + (" (= last)" if i == n - 1 else "")
    return f"none of {len(lv.get('layers') or [])} layers"


def analyze_game(path: str) -> dict[str, Any]:
    from arc3.perception import background_color

    with gzip.open(path, "rb") as f:
        rec = pickle.load(f)
    levels = []
    for lv in rec["levels"]:  # the solved prefix with an exact terminal frame
        if lv.get("terminal_exact") is None:
            break
        levels.append(lv)
    out: dict[str, Any] = {"game": rec["game"], "source": rec["source"], "levels_solved": len(rec["levels"]),
                           "terminal_layer": [_terminal_layer(lv) for lv in rec["levels"]]}
    if not levels:
        return out
    bg = background_color(levels[0]["start"])
    per_level = []
    colors: set[int] = set()
    for lv in levels:
        negs = _frames(list(lv["negatives"]) + list(lv["recent"]) + [lv["start"], lv["pre"]], bg)
        term = _frames([lv["terminal_exact"]], bg)[0]
        per_level.append((lv, term, negs))
        colors |= {e.color for e in term} | {e.color for e in negs[0]}
    preds = _all_predicates(sorted(colors), per_level[0][2][-2])  # the level-1 start frame
    scored = []
    for lv, term, negs in per_level:
        limit = max(1, len(negs) // 100)
        s = _score_level(preds, term, negs, limit)
        scored.append({"level": lv["level"], "n_neg": len(negs), "actions": lv["actions"], "baseline": lv["baseline"],
                       "survivors": sorted(k for k, v in s.items() if v == 0),
                       "near": sorted(k for k, v in s.items() if 0 < v <= limit)})
    l1 = scored[0]
    transfer = {}
    for name in l1["survivors"]:
        transfer[name] = [name in lv["survivors"] for lv in scored[1:]]
    joint = set(l1["survivors"])
    for lv in scored[1:]:
        joint &= set(lv["survivors"])
    # Lifted transfer: the same kind with the same colour-equality pattern (vc33's goal is "aligned" on every level,
    # but on colour 11 at level 1 and colour 14 at level 2, so exact predicates never transfer there).
    names = {n: (k, a) for n, k, a, _p in preds}

    def lifted(name: str) -> tuple:
        k, a = names.get(name, (name, ()))
        cols = [x for x in a if isinstance(x, int)]
        return (k, len(cols) == 2 and cols[0] == cols[1])

    l1_lifted = {lifted(n) for n in l1["survivors"]}
    lifted_transfer = [sorted({str(lifted(n)) for n in lv["survivors"]} & {str(x) for x in l1_lifted}) for lv in scored[1:]]
    # The realistic test (dsl.instantiate_goals): after level 1 only, instantiate its survivors' colour-free signatures
    # at each later level's start, falsify them with that level's non-winning states, and check that the level's
    # winning frame satisfies a survivor (recall) and how many survive (ambiguity left for experiments or a model).
    from arc3 import dsl

    sigs = {sg for n in l1["survivors"] for sg in [dsl.goal_signature(*names.get(n, (n, ())))] if sg is not None}
    instantiated = []
    for lv, term, negs in per_level[1:]:
        cands = dsl.instantiate_goals(sigs, negs[-2], max_goals=1000) if sigs else []  # negs[-2]: the level start
        alive = [g for g in cands if not any(_holds(g["predicate"], f) for f in negs)]
        instantiated.append({"level": lv["level"], "instantiated": len(cands), "alive_after_falsification": len(alive),
                             "winning_frame_satisfies": sorted(g["goal"] for g in alive if _holds(g["predicate"], term))})
    out.update({
        "levels": scored,
        "l1_survivors": len(l1["survivors"]),
        "l1_survivors_existing_grammar": sum(1 for n in l1["survivors"] if not n.startswith("every_")),
        "l1_survivors_forall": sum(1 for n in l1["survivors"] if n.startswith("every_")),
        "transfer": transfer,
        "l1_survivors_holding_on_all_later": sorted(n for n, t in transfer.items() if t and all(t)),
        "joint_survivors": sorted(joint),
        "l1_kinds": sorted(str(x) for x in l1_lifted),
        "lifted_transfer": lifted_transfer,
        "lifted_instantiation": instantiated,
    })
    return out


# ---------------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--source", choices=["explorer", "oracle"], default="explorer")
    c.add_argument("--games", default="")
    c.add_argument("--split", default="dev")
    c.add_argument("--workers", type=int, default=3)
    c.add_argument("--max-actions", type=int, default=20000)
    c.add_argument("--max-s", type=float, default=90.0, help="explorer: seconds per game; oracle: seconds per level")
    c.add_argument("--max-nodes", type=int, default=60000)
    c.add_argument("--max-levels", type=int, default=4)
    c.add_argument("--max-neg", type=int, default=600)
    c.add_argument("--run", default=str(RUN))
    a_ = sub.add_parser("analyze")
    a_.add_argument("--run", default=str(RUN))
    a_.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "collect":
        from arc3.splits import resolve

        games = [g for g in a.games.split(",") if g] or resolve(a.split)
        out = str(Path(a.run) / a.source)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            if a.source == "explorer":
                futs = [ex.submit(collect_explorer, g, out, a.max_actions, a.max_s, a.max_levels, a.max_neg) for g in games]
            else:
                futs = [ex.submit(collect_oracle, g, out, a.max_nodes, a.max_s, a.max_levels, a.max_neg) for g in games]
            for f in futs:
                try:
                    print(json.dumps(f.result()), flush=True)
                except Exception as e:
                    print(json.dumps({"error": repr(e)}), flush=True)
        return
    run = Path(a.run)
    paths = sorted(str(p) for p in run.glob("*/*.pkl.gz"))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(analyze_game, paths))
    summary = {"run": str(run), "games": results}
    (run / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    for r in results:
        lv = r.get("levels") or []
        print(f"{r['source']:8s} {r['game']}: solved {r['levels_solved']}, terminal {r['terminal_layer'][:2]}; "
              f"L1 survivors {r.get('l1_survivors', '-')} (existing {r.get('l1_survivors_existing_grammar', '-')}, "
              f"for-all {r.get('l1_survivors_forall', '-')}); hold on all later {len(r.get('l1_survivors_holding_on_all_later', []))}"
              f"/{r.get('l1_survivors', 0)} over {max(0, len(lv) - 1)} later levels; joint {len(r.get('joint_survivors', []))}")
    print(f"-> {run / 'summary.json'}")


if __name__ == "__main__":
    main()
