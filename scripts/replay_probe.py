"""Replay a short action sequence of one public game through the tracker and move model, without a model.

The cheapest test of a planner or tracker change: it reproduces what the REPL helpers would have seen in a real run.
Written for the exp-009 ka59 post-mortem (docs/postmortems/exp009-ka59-floor-entity-2026-09-16.md).

    .venv/bin/python scripts/replay_probe.py ka59 UP DOWN LEFT RIGHT CLICK:36,30 RIGHT RIGHT
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.INFO)
from arc3.entities import Tracker  # noqa: E402
from arc3.env import Action, LocalEnv, make_arcade  # noqa: E402
from arc3.planner import MoveModel  # noqa: E402

KEYS = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "ACT": 5, "UNDO": 7}


def parse(tok: str):
    if tok.upper().startswith("CLICK:"):
        x, y = tok[6:].split(",")
        return ("CLICK", int(x), int(y))
    return tok.upper()


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    game, actions = sys.argv[1], [parse(t) for t in sys.argv[2:]]
    env = LocalEnv(make_arcade("environment_files"), game, seed=0)
    g = np.asarray(env.reset().grid, dtype=np.int16)
    t = Tracker()
    t.reset(g)
    for a in actions:
        act = Action.click(a[1], a[2]) if isinstance(a, tuple) else Action.simple(KEYS[a])
        fr = env.step(act)
        g = np.asarray(fr.grid, dtype=np.int16)
        rec = t.update(g, a)
        print(f"{a!s:18s} {Tracker.describe(rec, t)[:150]}")
        av = t.avatar()
        if av and av["alive"]:
            try:
                m = MoveModel(t)
                m.predict(g.copy(), "RIGHT")  # exercises the predictor on the current frame
                print(f"{'':18s} avatar #{av['id']} at {m.pos} keymap {av['keymap']} walkable {sorted(m.walkable)} "
                      f"companions {len(m.companions)} obstacles {int(m.obstacles.sum())} hud {t.hud_ids()}")
            except ValueError as e:
                print(f"{'':18s} move model: {e}")
        elif av:
            print(f"{'':18s} avatar #{av['id']} dead; no hand-over yet")
    print("levels", fr.levels_completed, "state", fr.state)


if __name__ == "__main__":
    main()
