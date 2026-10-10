#!/usr/bin/env python
"""Full-length runs side by side: score, levels on the hard 15 and the easy 10, serving and token use.

    .venv/bin/python -I scripts/run_compare.py runs/exp073-... runs/exp073b-... [...] runs/<arm>

Reads runs/<run>/report.json (scripts/franzen_report.py). "Hard 15" are the 15 games Franzen's demo leaves out (his
`demo_excluded_games`), "easy 10" the 10 it plays. Per run: mean score, levels in all / hard / easy, actions, output
tokens per action and per request (serve.log), MTP accept length and output tok/s over the run. The last run is the
arm: each of its columns is placed against the range and mean of the other runs.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HARD = {"bp35", "cd82", "cn04", "dc22", "g50t", "ka59", "lf52", "ls20", "m0r0", "s5i5", "sk48", "sp80", "su15",
        "tn36", "wa30"}


def row(run: Path) -> dict:
    rep = json.loads((run / "report.json").read_text())
    games, serve = rep["games"], rep.get("serve") or {}
    hard = [v for g, v in games.items() if g[:4] in HARD]
    easy = [v for g, v in games.items() if g[:4] not in HARD]
    actions = sum(v["actions"] for v in games.values())
    out_tokens = serve.get("output_tokens") or 0
    finished = serve.get("requests_finished") or 0
    return {"run": run.name.split("-")[0], "games": len(games),
            "score": statistics.fmean(v["score_ours"] for v in games.values()),
            "hard_score": statistics.fmean(v["score_ours"] for v in hard) if hard else float("nan"),
            "easy_score": statistics.fmean(v["score_ours"] for v in easy) if easy else float("nan"),
            "levels": sum(v["levels"] for v in games.values()), "hard_levels": sum(v["levels"] for v in hard),
            "easy_levels": sum(v["levels"] for v in easy), "actions": actions,
            "tok_per_action": out_tokens / actions if actions else float("nan"),
            "tok_per_request": out_tokens / finished if finished else float("nan"),
            "accept": serve.get("accept_len_mean") or float("nan"),
            "tok_s": serve.get("output_tok_s_over_span") or float("nan")}


COLUMNS = [("score", "{:7.2f}"), ("hard_score", "{:7.2f}"), ("easy_score", "{:7.2f}"), ("levels", "{:5d}"),
           ("hard_levels", "{:5d}"), ("easy_levels", "{:5d}"), ("actions", "{:6d}"), ("tok_per_action", "{:7.0f}"),
           ("tok_per_request", "{:7.0f}"), ("accept", "{:6.2f}"), ("tok_s", "{:6.0f}")]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    rows = [row(Path(p)) for p in argv]
    print("run      " + "".join(f"{name:>16s}" for name, _ in COLUMNS))
    for r in rows:
        print(f"{r['run']:8s} " + "".join(f"{fmt.format(r[name]):>16s}" for name, fmt in COLUMNS))
    arm, others = rows[-1], rows[:-1]
    print(f"\n{arm['run']} against the other {len(others)} runs (min / mean / max):")
    for name, fmt in COLUMNS:
        values = [o[name] for o in others if o[name] == o[name]]
        if not values or arm[name] != arm[name]:
            continue
        above = sum(v < arm[name] for v in values)
        print(f"  {name:16s} {fmt.format(arm[name]).strip():>8s}   vs {fmt.format(min(values)).strip()} / "
              f"{statistics.fmean(values):.2f} / {fmt.format(max(values)).strip()}   (above {above} of {len(values)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
