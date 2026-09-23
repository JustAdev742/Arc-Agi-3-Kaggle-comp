#!/usr/bin/env python
"""Per-game standardized comparison of our runs with the harvested base runs (a lower-noise yardstick than the mean).

    .venv/bin/python scripts/zsum.py [runs/exp042-ours-g ...]      # default: every Duck run of ours with a summary

The public-25 mean is dominated by a few games whose score swings by 10-25 points between identical runs (ft09, lp85,
ar25), so one lucky game moves the mean by a point. Here each game's score is standardized against that game's
distribution over the reference runs (mean and sd per game, sd floored at 2 points so near-constant games cannot
dominate), and the 25 z-scores are summed. The null distribution of that sum comes from the reference runs themselves,
each scored leave-one-out against the others. Reference: scripts/compare_to_harvest.load_reference (stock-cap
Flash-Next Duck runs). Prints each run's z-sum, its percentile among the reference runs, and the mean-based rank for
comparison.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_to_harvest import load_reference  # noqa: E402

SD_FLOOR = 2.0


def per_game(run: dict) -> dict[str, float]:
    return {r["game_id"][:4]: float(r.get("score", 0.0)) for r in run["results"]}


def zsum(scores: dict[str, float], refs: list[dict[str, float]]) -> float:
    total = 0.0
    for game, value in scores.items():
        vals = [r[game] for r in refs if game in r]
        if len(vals) < 3:
            continue
        mean = statistics.fmean(vals)
        sd = max(SD_FLOOR, statistics.pstdev(vals))
        total += (value - mean) / sd
    return total


def ours(path: Path) -> dict[str, float] | None:
    s = json.loads((path / "summary.json").read_text())
    if "results" not in s:
        return None
    return {r["game_id"][:4]: float(r.get("score", r.get("taaf_score", 0.0)) or 0.0) for r in s["results"]}


def main() -> None:
    refs_raw = load_reference("flash-next", "any")
    refs = [per_game(r) for r in refs_raw if r.get("results")]
    null = [zsum(r, refs[:i] + refs[i + 1:]) for i, r in enumerate(refs)]
    mu, sd = statistics.fmean(null), statistics.pstdev(null)
    means = sorted(statistics.fmean(r.values()) for r in refs)
    print(f"reference: {len(refs)} runs; z-sum null mean {mu:+.1f}, sd {sd:.1f}")
    paths = [Path(a) for a in sys.argv[1:]] or sorted(
        p.parent for p in (ROOT / "runs").glob("exp0[3-9]*/summary.json"))
    for path in paths:
        scores = ours(path)
        if not scores:
            continue
        z = zsum(scores, refs)
        mean = statistics.fmean(scores.values())
        pz = sum(n < z for n in null) / len(null)
        pm = sum(m < mean for m in means) / len(means)
        print(f"{path.name:24s} mean {mean:6.2f} (beats {pm:4.0%} of refs)   z-sum {z:+6.1f} "
              f"(beats {pz:4.0%}; {(z - mu) / sd:+.1f} sd)")


if __name__ == "__main__":
    main()
