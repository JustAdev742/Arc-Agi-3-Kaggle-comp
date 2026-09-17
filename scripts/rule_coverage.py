"""Rule-library coverage on the public games without a model (plan-100 gate exp-006, code-only lower bound).

For each game, play level 1 with a fixed probe policy (each key several times, ACT, clicks on entity centres),
track entities, fit rules with arc3.dsl.auto_rules and report the fraction of observed entity events the rule
set explains, the rules found, and the unexplained event kinds. This says which mechanics the DSL can express
today and where to extend it; it is not a score.

    .venv/bin/python scripts/rule_coverage.py --games all --actions 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arc3 import dsl  # noqa: E402
from arc3.entities import Tracker  # noqa: E402
from arc3.env import Action, LocalEnv, make_arcade  # noqa: E402
from arc3.prompts import ACTION_NAMES  # noqa: E402
from arc3.splits import ALL_GAMES, DEV_GAMES, VAL_GAMES  # noqa: E402

KEY_IDS = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4}


def probe_policy(frame, tracker: Tracker, i: int, clicked: set[tuple[int, int]]):
    """A fixed exploration order: keys in pairs, ACT, then clicks on the largest non-static entities."""
    avail = [ACTION_NAMES[a] for a in (frame.available_actions or []) if a in ACTION_NAMES]
    keys = [k for k in ("UP", "DOWN", "LEFT", "RIGHT") if k in avail]
    order: list = []
    for _ in range(3):
        for k in keys:
            order.append(k)
            order.append(k)
        if "ACT" in avail:
            order.append("ACT")
    if i < len(order):
        a = order[i]
        return (Action.simple(5) if a == "ACT" else Action.simple(KEY_IDS[a])), a
    if "CLICK" in avail:
        roles = tracker.roles()
        cands = sorted((e for e in tracker.current if roles.get(e.id) not in ("static", "hud")), key=lambda e: -e.size)
        cands += sorted((e for e in tracker.current if roles.get(e.id) == "static"), key=lambda e: -e.size)
        for e in cands:
            cx, cy = e.center
            if (cx, cy) not in clicked:
                clicked.add((cx, cy))
                return Action.click(cx, cy), ("CLICK", cx, cy)
    if keys:
        k = keys[i % len(keys)]
        return Action.simple(KEY_IDS[k]), k
    if "ACT" in avail:
        return Action.simple(5), "ACT"
    return Action.click(32, 32), ("CLICK", 32, 32)


def run_game(arc, game: str, n_actions: int) -> dict:
    env = LocalEnv(arc, game)
    tr = Tracker()
    tr.reset(env.frame.grid)
    clicked: set[tuple[int, int]] = set()
    levels_done = 0
    logs: list[tuple[list[dsl.Transition], list[int]]] = []  # (log, hud ids to ignore)
    t0 = time.time()
    i = 0
    while i < n_actions:
        act, label = probe_policy(env.frame, tr, i, clicked)
        f = env.step(act)
        i += 1
        if f.levels_completed > levels_done or f.state.name in ("WIN", "GAME_OVER"):
            levels_done = f.levels_completed
            logs.append((dsl.make_log(tr.compound_frames(), tr.actions, tr.unders, tr.bg), tr.hud_ids()))
            tr.reset(f.grid)
            clicked.clear()
            if f.state.name in ("WIN", "GAME_OVER"):
                break
            continue
        tr.update(f.grid, label)
    logs.append((dsl.make_log(tr.compound_frames(), tr.actions, tr.unders, tr.bg), tr.hud_ids()))
    env.close()
    out = {"game": game, "actions": i, "levels_completed": levels_done, "play_s": round(time.time() - t0, 1)}
    per_level = []
    kinds: Counter = Counter()
    t1 = time.time()
    for lg, hud in logs:
        if not lg:
            continue
        rules, rep = dsl.auto_rules(lg, ignore_ids=hud)
        for u in rep["unexplained"]:
            kinds[u["event"]] += 1
        per_level.append({"transitions": rep["transitions"], "events": rep["events"], "explained": rep["explained"],
                          "coverage": round(rep["coverage"], 3), "fully_explained": rep["fully_explained"],
                          "rules": [r.describe() for r in rules],
                          "unexplained_sample": [f"#{u['id']} c{u['color']} {u['event']} on {u['action']}" for u in rep["unexplained"][:4]]})
    out["fit_s"] = round(time.time() - t1, 1)
    out["levels"] = per_level
    ev = sum(p["events"] for p in per_level)
    ex = sum(p["explained"] for p in per_level)
    out["events"] = ev
    out["coverage"] = round(ex / ev, 3) if ev else None
    out["unexplained_kinds"] = dict(kinds)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", default="all", help="all | dev | val | comma-separated ids")
    p.add_argument("--actions", type=int, default=60)
    p.add_argument("--out", default="runs/rule-coverage/summary.json")
    a = p.parse_args()
    games = {"all": ALL_GAMES, "dev": DEV_GAMES, "val": VAL_GAMES}.get(a.games) or a.games.split(",")
    arc = make_arcade(ROOT / "environment_files")
    results = []
    for g in games:
        try:
            r = run_game(arc, g, a.actions)
        except Exception as e:
            r = {"game": g, "error": f"{type(e).__name__}: {e}"[:200]}
        results.append(r)
        cov = r.get("coverage")
        rules = r["levels"][0]["rules"] if r.get("levels") else []
        print(f"{g}: actions {r.get('actions')} events {r.get('events')} coverage {cov} rules {len(rules)} "
              f"unexplained {r.get('unexplained_kinds')} fit {r.get('fit_s')}s" + (f" ERROR {r['error']}" if "error" in r else ""), flush=True)
        for t in rules[:4]:
            print(f"    {t}")
    covs = [r["coverage"] for r in results if r.get("coverage") is not None]
    summary = {"games": results, "mean_coverage": round(sum(covs) / len(covs), 3) if covs else None,
               "n_games": len(results), "actions_per_game": a.actions}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1, default=str))
    print(f"mean coverage {summary['mean_coverage']} over {len(covs)} games -> {a.out}")


if __name__ == "__main__":
    main()
