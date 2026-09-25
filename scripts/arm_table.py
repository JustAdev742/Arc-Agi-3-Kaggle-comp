#!/usr/bin/env python
"""One row per Duck run: score, levels and the serving and behaviour numbers the arms are judged on.

    .venv/bin/python scripts/arm_table.py [runs/exp032-anim-flashnext runs/exp045-gate ...]   # default: every Duck run

Reads runs/<run>/summary.json as written by scripts/pull_taaf_run.py (``server``, ``behaviour``) and, when the kernel
log is there, the last line the P21 gate printed. Harvest reference for the base configuration: n=37, mean 7.15, sd 1.65.
``val`` is the held-out validation split's score; ``hardL1`` counts the 8 hard public games (docs/research/
hard-games-level1.md) whose level 1 was solved, with the median minutes to that solve: most hidden games are expected
to stay on level 1, so this column tracks what the leaderboard rewards better than the public-25 mean does. Reference:
the 39 harvested base runs solve 4.5 of the 8 on average (sd 1.4, range 2-7).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARD = ("sk48", "bp35", "g50t", "dc22", "tn36", "ls20", "tr87", "m0r0")


def hard_l1(results: list[dict]) -> str:
    solved = [r for r in results if r["game_id"][:4] in HARD and int(r.get("levels_completed") or 0) >= 1]
    minutes = sorted(r["level_done_s"][0] / 60 for r in solved if r.get("level_done_s"))
    median = f"{minutes[len(minutes) // 2]:.0f}m" if minutes else "-"
    return f"{len(solved)}/8 {median:>4}"


def gate_line(run: Path) -> str:
    logs = sorted((run / "kernel-output").glob("*.log"))
    last = ""
    for log in logs:
        for line in log.read_text(errors="replace").splitlines():
            if "ours-gate:" in line:
                last = line[line.index("ours-gate:"):]
    m = re.search(r"admitted (\d+), mean wait ([\d.]+)s, timeouts (\d+)", last)
    return f"gate {m.group(1)} adm, wait {m.group(2)}s, {m.group(3)} t/o" if m else ""


def row(run: Path) -> str | None:
    path = run / "summary.json"
    if not path.exists():
        return None
    s = json.loads(path.read_text())
    if "score_all" not in s:
        return None
    sv, b = s.get("server") or {}, s.get("behaviour") or {}
    rates = b.get("calls_per_min_by_level")
    if rates is None and (run / "kernel-output" / "transcripts").is_dir():  # older summaries: recompute from transcripts
        sys.path.insert(0, str(ROOT / "scripts"))
        import taaf_mechanisms

        rates = taaf_mechanisms.run_stats(run).get("calls_per_min_by_level")
    rates = rates or {}
    rate_txt = " ".join(f"{k} {v:.2f}" for k, v in rates.items())
    val = f"{s['score_val']:5.2f}" if isinstance(s.get("score_val"), (int, float)) else "    ?"
    return (f"{run.name:24s} {s['score_all']:6.2f} {val} {hard_l1(s.get('results') or []):>8} "
            f"{s.get('levels_completed', '?'):>3} "
            f"{sv.get('requests', '?'):>5} {sv.get('prompt_tokens_per_request', '?'):>6} "
            f"{sv.get('generated_tokens_per_request', '?'):>5} {sv.get('preemptions', '?'):>5} "
            f"{sv.get('running_mean', '?'):>5} {b.get('acting_share', '?'):>6}  {rate_txt:28s} {gate_line(run)}")


def main() -> None:
    runs = [Path(a) for a in sys.argv[1:]] or sorted(p.parent for p in (ROOT / "runs").glob("exp0[3-9]*/summary.json"))
    print(f"{'run':24s} {'score':>6} {'val':>5} {'hardL1':>8} {'lv':>3} {'reqs':>5} {'prompt':>6} {'gen':>5} {'preem':>5} {'run':>5} "
          f"{'acting':>6}  calls/min by level")
    for run in runs:
        line = row(run)
        if line:
            print(line)


if __name__ == "__main__":
    main()
