"""Were the levels ours-07 (fresh start) touched solved later? Against how often other runs solved the same levels.

    .venv/bin/python -I scripts/fresh_start_outcomes.py --base runs/<run> [--base ...] --arm SA_DIR:runs/<run> [...]

SA_DIR holds the run's solver_analysis/*.html (Franzen's per-game transcripts; pull them with `kaggle kernels output
OWNER/KERNEL -p SA_DIR --file-pattern 'solver_analysis/.*\\.html$'`); runs/<run>/report.json is scripts/franzen_report.py's.
The other runs' rate overstates the counterfactual (fresh starts fire on levels this run found hard); compare with the
hazard of levels unsolved at 20 minutes too (docs/research/beat-tufa/time-allocation.md, lesson 0037).
"""
import argparse
import glob
import html
import json
import re

FRESH = re.compile(r"ours_fresh_start: conversation history cleared at step (\d+) \(fresh start (\d+) of at most (\d+) "
                   r"on level (\d+)\)")


def levels(run):
    return {g[:4]: v["levels"] for g, v in json.load(open(f"{run}/report.json"))["games"].items()}


ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--base", action="append", required=True, help="runs/<run> without the patch")
ap.add_argument("--arm", action="append", required=True, help="SA_DIR:runs/<run> with the patch")
args = ap.parse_args()
base = {}
for r in args.base:
    for g, lv in levels(r).items():
        base.setdefault(g, []).append(lv)
for arm in args.arm:
    sa, run = arm.split(":", 1)
    rep = levels(run)
    touched = {}
    for f in sorted(glob.glob(sa + "/solver_analysis/*.html")):
        g = f.split("/")[-1][:4]
        t = html.unescape(re.sub(r"<[^>]+>", "\n", open(f, errors="replace").read()))
        for m in FRESH.finditer(t):
            touched[(g, int(m.group(4)))] = touched.get((g, int(m.group(4))), 0) + 1
    if not touched:
        print(f"{run}: no fresh starts found in {sa}")
        continue
    solved = sum(rep[g] >= lv for g, lv in touched)
    rates = [sum(b >= lv for b in base[g]) / len(base[g]) for g, lv in touched]
    print(f"{run}: {len(touched)} touched levels, solved later {solved}; the same levels in the {len(args.base)} base runs: "
          f"{sum(rates) / len(rates):.2f}")
    for (g, lv), n in sorted(touched.items()):
        print(f"   {g} L{lv} x{n}: {'SOLVED' if rep[g] >= lv else 'unsolved'}; base runs "
              f"{sum(b >= lv for b in base[g])}/{len(base[g])}")
