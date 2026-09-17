#!/usr/bin/env python
"""Run an agent on a split with the local harness. See arc3/eval.py.

Examples:
    .venv/bin/python scripts/eval.py --agent random --split smoke --max-actions 100
    .venv/bin/python scripts/eval.py --agent explorer --split dev --time-per-game 300 --workers 4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arc3.eval import run_eval


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True)
    p.add_argument("--split", default="dev", help="dev | val | all | smoke | comma-separated game ids")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-per-game", type=float, default=600.0, help="seconds")
    p.add_argument("--max-actions", type=int, default=5000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--run-name", default=None)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--config", default="{}", help="JSON dict passed to the agent")
    p.add_argument("--record-frames", action="store_true")
    p.add_argument("--note", default="")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("arc_agi").setLevel(logging.WARNING)
    run_eval(a.agent, a.split, seed=a.seed, time_budget_s=a.time_per_game, max_actions=a.max_actions,
             workers=a.workers, run_name=a.run_name, runs_dir=a.runs_dir, config=json.loads(a.config),
             record_frames=a.record_frames, note=a.note)


if __name__ == "__main__":
    main()
