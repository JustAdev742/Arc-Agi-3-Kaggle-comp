#!/usr/bin/env python
"""Download (or refresh) the public game sources into environment_files/ (anonymous key ok)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import arc_agi
from arc_agi import OperationMode


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    envs = arc.get_environments()
    card = arc.open_scorecard(tags=["download"])
    ok = 0
    for e in envs:
        gid = e.game_id.split("-")[0]
        env = arc.make(gid, scorecard_id=card)
        print(f"{e.game_id:16} {'ok' if env else 'FAILED'} baseline={e.baseline_actions} tags={e.tags}")
        ok += env is not None
    arc.close_scorecard(card)
    print(f"{ok}/{len(envs)} games available under environment_files/")


if __name__ == "__main__":
    main()
