#!/usr/bin/env python
"""Score of a TAAF run (benchmark.json) over a common wall clock: what the run would have scored had it stopped at
a fraction of its length, and the local elasticity d ln(score) / d ln(time).

    .venv/bin/python scripts/score_over_time.py RUN_DIR/benchmark.json

All games start together in Franzen's notebook, so a shorter run is (approximately) the same run cut at an earlier
time, and extra decode throughput is (approximately) extra time. The left derivative at the end of the run is an
estimate of how score responds to more tokens per game; it is shaped by the scheduler's tail fade near the end.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arc3.scoring import game_score  # noqa: E402

FRACTIONS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0)


def score_at(run: dict, t: float, t0: datetime) -> float:
    """Our score for one game run counting only actions taken by `t` seconds after the benchmark start."""
    offset = (datetime.fromisoformat(run["started_at"]) - t0).total_seconds()
    cut = sum(1 for x in run["history"] if offset + x["wallclock_seconds"] <= t)
    completed, used = [], 0
    for acts in run["actions_per_level"][: run["levels_completed"]]:
        used += acts
        if used > cut:
            break
        completed.append(acts)
    return game_score(completed, run["base_actions_per_level"])


def curve(bench: dict) -> tuple[float, dict[float, float]]:
    t0 = datetime.fromisoformat(bench["start_time"])
    runs = bench["game_runs"]
    end = max((datetime.fromisoformat(r["started_at"]) - t0).total_seconds()
              + (r["history"][-1]["wallclock_seconds"] if r["history"] else 0.0) for r in runs)
    return end, {f: statistics.mean(score_at(r, f * end, t0) for r in runs) for f in FRACTIONS}


def main() -> None:
    bench = json.loads(Path(sys.argv[1]).read_text())
    end, s = curve(bench)
    print(f"{len(bench['game_runs'])} game runs; last action {end / 60:.1f} min after the start")
    for f, v in s.items():
        print(f"  {f:4.2f} of the run ({f * end / 60:5.0f} min): mean score {v:6.2f}")
    for a in (0.5, 0.7, 0.8, 0.9):
        if s[a] > 0:
            print(f"  elasticity over [{a}, 1.0]: {math.log(s[1.0] / s[a]) / math.log(1 / a):.3f}")


if __name__ == "__main__":
    main()
