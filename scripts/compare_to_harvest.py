#!/usr/bin/env python
"""Place one of our public-25 runs within the harvested distribution of other teams' runs of a reference configuration.

    .venv/bin/python scripts/compare_to_harvest.py runs/exp035-ours-b/summary.json [--models flash-next] [--anim any]

Reference runs come from runs/public-harvest (scripts/harvest_public_runs.py), filtered by model substring and the
animation-aware flag. Reports: our mean against the reference means (rank, percentile), levels, and a per-game table of
our score against the reference per-game mean. Because each reference run is itself one noisy draw, the "rank among
reference runs" is the honest reading of a single run (lesson 0018).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_reference(models: str, anim: str) -> list[dict]:
    refs = []
    for f in sorted((ROOT / "runs" / "public-harvest").glob("*__*.json")):
        d = json.loads(f.read_text())
        if d.get("games") != 25 or models not in ",".join(d.get("models", [])):
            continue
        if anim != "any" and bool(d.get("anim")) != (anim == "yes"):
            continue
        # Stock per-game cap only (one run gave every game 27,000 s), and not a run that broke (8 levels in 25 games).
        caps = [float(x) for x in (d.get("knobs") or {}).get("max_runtime_s_per_game", [])
                if x.replace(".", "", 1).isdigit()]
        if any(c > 8000 for c in caps) or d.get("levels", 0) < 15:
            continue
        refs.append(d)
    return refs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--models", default="flash-next")
    ap.add_argument("--anim", choices=["any", "yes", "no"], default="any")
    args = ap.parse_args()
    ours = json.loads(Path(args.summary).read_text())
    our_games = {g["game_id"]: g for g in ours["results"]}
    refs = load_reference(args.models, args.anim)
    if not refs:
        raise SystemExit("no reference runs")
    means = sorted(r["score_all"] for r in refs)
    our_mean = ours["score_all"]
    rank = sum(1 for m in means if m < our_mean)
    print(f"{ours['run_name']}: mean {our_mean:.2f}, levels {ours['levels_completed']}; reference runs n={len(refs)} "
          f"mean {statistics.mean(means):.2f} sd {statistics.pstdev(means):.2f} min {means[0]:.2f} max {means[-1]:.2f}; "
          f"ours beats {rank} of {len(refs)} ({100 * rank / len(refs):.0f}th percentile)")
    print(f"  reference levels: mean {statistics.mean(r['levels'] for r in refs):.1f} "
          f"(min {min(r['levels'] for r in refs)}, max {max(r['levels'] for r in refs)})")
    if ours.get("server"):
        print("  our server:", ours["server"])
    # Minutes from the previous level's completion (or the start) to each completed level; the stock-cap harvest
    # (19 runs, research log 2026-09-23) has medians 24.2 / 25.6 / 30.8 / 24.3 for levels 1-4.
    per_level: dict[int, list[float]] = {}
    for g in ours["results"]:
        prev = 0.0
        for k, t in enumerate(g.get("level_done_s") or [], start=1):
            if t is None:
                break
            per_level.setdefault(k, []).append((t - prev) / 60.0)
            prev = t
    if per_level:
        print("  our minutes per solved level (median, count):",
              ", ".join(f"L{k} {statistics.median(v):.1f} ({len(v)})" for k, v in sorted(per_level.items())),
              "| stock-cap harvest: L1 24.2, L2 25.6, L3 30.8, L4 24.3")
    per_game: dict[str, list[float]] = {}
    for r in refs:
        for g in r["results"]:
            per_game.setdefault(g["game_id"], []).append(g["score"])
    rows = []
    for gid, xs in per_game.items():
        mine = our_games.get(gid, {}).get("score", 0.0)
        solved = sum(1 for x in xs if x > 0) / len(xs)
        rows.append((mine - statistics.mean(xs), gid, mine, statistics.mean(xs), statistics.pstdev(xs), solved,
                     our_games.get(gid, {}).get("levels_completed", 0)))
    print(f"  {'game':6s} {'ours':>6s} {'ref':>6s} {'sd':>6s} {'ref>0':>6s} {'lv':>3s}  delta")
    for delta, gid, mine, mu, sd, solved, lv in sorted(rows, reverse=True):
        print(f"  {gid:6s} {mine:6.2f} {mu:6.2f} {sd:6.2f} {solved:6.0%} {lv:3d}  {delta:+.2f}")


if __name__ == "__main__":
    main()
