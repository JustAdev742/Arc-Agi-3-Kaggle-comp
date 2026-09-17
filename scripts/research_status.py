#!/usr/bin/env python
"""Answers from the machine-readable run records (runs/*/summary.json): what is best, what changed since it, which
failure category dominates, seconds per action, which games eat the time, per-game outcomes across runs.

Usage: .venv/bin/python scripts/research_status.py [--split dev] [--agent repl] [--runs runs] [--json out.json]
       .venv/bin/python scripts/research_status.py --matrix          # per game x run failure matrix for the split
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_runs(runs_dir: Path, *, split: str | None, agent: str | None) -> list[dict[str, Any]]:
    out = []
    for p in sorted(runs_dir.glob("*/summary.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if not d.get("results") or d.get("score") is None:
            continue
        if split and d.get("split") != split:
            continue
        if agent and d.get("agent") != agent:
            continue
        d["_name"] = p.parent.name
        out.append(d)
    return out


def failure_of(r: dict[str, Any]) -> str:
    """The failure category of one game result: a solved game, a run-time failure tag, or a stuck level."""
    if r.get("levels_completed", 0) >= r.get("win_levels", 1):
        return "solved"
    if r.get("error"):
        return "crash"
    f = r.get("failure") or "unknown"
    st = r.get("agent_stats") or {}
    if st.get("actions_fallback", 0) and not st.get("actions_model", 0):
        return "no_model_actions"
    return f


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(runs, key=lambda d: d["score"])
    by_name = {d["_name"]: d for d in runs}
    per_game: dict[str, dict[str, Any]] = defaultdict(dict)
    fail = Counter()
    for d in runs:
        for r in d["results"]:
            per_game[r["game_id"]][d["_name"]] = {"levels": r["levels_completed"], "score": round(r["score"], 2), "actions": r["actions"],
                                                   "wall": round(r["wall_s"]), "failure": failure_of(r)}
    for r in best["results"]:
        fail[failure_of(r)] += 1
    spa = [r["seconds_per_action"] for r in best["results"] if r.get("seconds_per_action")]
    time_hogs = sorted(best["results"], key=lambda r: -r["wall_s"])[:5]
    stats = [r.get("agent_stats") or {} for r in best["results"]]
    lat = [s.get("model_latency_p50_s") for s in stats if s.get("model_latency_p50_s")]
    tokens = sum((s.get("prompt_tokens") or 0) + (s.get("completion_tokens") or 0) for s in stats)
    stable = [g for g, rs in per_game.items() if len(rs) >= 3 and all(v["levels"] > 0 for v in rs.values())]
    never = [g for g, rs in per_game.items() if len(rs) >= 3 and all(v["levels"] == 0 for v in rs.values())]
    scores = sorted(d["score"] for d in runs)
    return {
        "runs": len(runs),
        "best": {"run": best["_name"], "score": round(best["score"], 3), "levels": best["levels_completed"], "actions": best["actions"],
                 "wall_s": best["wall_s"], "commit": best.get("harness_commit"), "config": best.get("config"), "note": best.get("note"),
                 "started": best.get("started")},
        "score_spread": {"min": round(scores[0], 3), "median": round(statistics.median(scores), 3), "max": round(scores[-1], 3)},
        "failure_categories_best": dict(fail.most_common()),
        "seconds_per_action_best": {"median": round(statistics.median(spa), 1) if spa else None, "max": round(max(spa), 1) if spa else None},
        "model_latency_p50_best": round(statistics.median(lat), 1) if lat else None,
        "tokens_best": tokens,
        "time_hogs_best": [{"game": r["game_id"], "wall_s": round(r["wall_s"]), "actions": r["actions"], "levels": r["levels_completed"]} for r in time_hogs],
        "games_solved_in_every_run": sorted(stable),
        "games_never_solved": sorted(never),
        "runs_by_score": [{"run": d["_name"], "score": round(d["score"], 3), "levels": d["levels_completed"], "actions": d["actions"],
                           "commit": d.get("harness_commit"), "note": (d.get("note") or "")[:80]} for d in sorted(runs, key=lambda d: -d["score"])],
        "_per_game": per_game, "_by_name": list(by_name),
    }


def changed_since(commit: str | None) -> str:
    if not commit:
        return "(no commit recorded)"
    try:
        out = subprocess.run(["git", "diff", "--stat", f"{commit}..HEAD", "--", "arc3", "agent"], cwd=ROOT, capture_output=True, text=True, timeout=30)
        return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "no changes to arc3/ or agent/"
    except (OSError, subprocess.SubprocessError) as e:
        return f"(git unavailable: {e})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--agent", default="repl")
    ap.add_argument("--matrix", action="store_true", help="print the game x run matrix (levels/failure)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    runs = load_runs(Path(a.runs), split=a.split or None, agent=a.agent or None)
    if not runs:
        print("no runs", file=sys.stderr)
        raise SystemExit(1)
    s = summarise(runs)
    b = s["best"]
    print(f"best {a.agent} run on {a.split}: {b['run']} score {b['score']} levels {b['levels']} actions {b['actions']} wall {b['wall_s']} s "
          f"commit {b['commit']} ({b['note']})")
    print(f"  config: {json.dumps(b['config'])}")
    print(f"  changed since its commit: {changed_since(b['commit'])}")
    print(f"  score spread over {s['runs']} runs: {s['score_spread']}")
    print(f"  failure categories (best run): {s['failure_categories_best']}")
    print(f"  seconds per action: {s['seconds_per_action_best']}; model p50 latency {s['model_latency_p50_best']} s; tokens {s['tokens_best']}")
    print(f"  time hogs: {s['time_hogs_best']}")
    print(f"  solved in every run: {s['games_solved_in_every_run']}; never solved: {s['games_never_solved']}")
    print("  runs by score:")
    for r in s["runs_by_score"]:
        print(f"    {r['run']:26s} {r['score']:6.3f} L{r['levels']:<3d} acts {r['actions']:5d} {str(r['commit'])[:8]} {r['note']}")
    if a.matrix:
        names = s["_by_name"]
        print("\ngame x run (levels completed; '.' = 0):")
        print("  game  " + " ".join(n.replace("kaggle-repl-dev-", "")[:5].rjust(5) for n in names))
        for g, rs in sorted(s["_per_game"].items()):
            print(f"  {g:5s} " + " ".join((str(rs[n]["levels"]) if rs.get(n, {}).get("levels") else ".").rjust(5) for n in names))
    if a.json:
        Path(a.json).write_text(json.dumps({k: v for k, v in s.items() if not k.startswith("_")}, indent=1, default=str))
        print(f"-> {a.json}")


if __name__ == "__main__":
    main()
