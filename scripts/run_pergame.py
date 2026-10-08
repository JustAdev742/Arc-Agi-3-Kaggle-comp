"""Per-game scores and levels across full-length runs (runs/<run>/report.json from scripts/franzen_report.py).

    .venv/bin/python -I scripts/run_pergame.py runs/exp073-... runs/exp073b-... [...]

Prints one row per game (score/levels per run), the run means, and for the last run the mean difference against the
mean of the others with a per-game paired SD, so a single run's gap can be read against game-level noise.
"""
import json
import statistics
import sys
from pathlib import Path

runs = [Path(p) for p in sys.argv[1:]]
data = []
for r in runs:
    rep = json.loads((r / "report.json").read_text())
    data.append({g[:4]: (v["score_ours"], v["levels"], v["total_levels"]) for g, v in rep["games"].items()})
games = sorted(set().union(*data))
names = [r.name.split("-")[0] for r in runs]
print("game  " + "".join(f"{n:>14s}" for n in names))
for g in games:
    cells = []
    for d in data:
        s, lv, tot = d.get(g, (float("nan"), 0, 0))
        cells.append(f"{s:7.1f} {lv:2d}/{tot:<2d} ")
    print(f"{g}  " + "".join(f"{c:>14s}" for c in cells))
means = [statistics.mean(d[g][0] for g in games if g in d) for d in data]
levels = [sum(d[g][1] for g in games if g in d) for d in data]
print("mean  " + "".join(f"{m:7.2f} {lv:3d}lv   " for m, lv in zip(means, levels)))
if len(data) >= 2:
    last, others = data[-1], data[:-1]
    diffs = [last[g][0] - statistics.mean(o[g][0] for o in others) for g in games if all(g in d for d in data)]
    sd = statistics.stdev(diffs)
    print(f"{names[-1]} minus mean of the others: {statistics.mean(diffs):+.2f} (per-game SD {sd:.1f}; "
          f"SE of the mean {sd / len(diffs) ** 0.5:.2f}, n={len(diffs)})")
