import sys
import time
from pathlib import Path

import numpy as np

from arc3.sandbox import PersistentSandbox

ROOT = str(Path(__file__).resolve().parents[1])


def state(grid=None, frames=None):
    g = np.zeros((64, 64), dtype=int) if grid is None else grid
    return {"grid": g.tolist(), "frames": [f.tolist() for f in (frames or [g])], "level": 1, "levels_completed": 0,
            "win_levels": 3, "step": 0, "level_step": 0, "state": "NOT_FINISHED", "available": ["UP", "CLICK"],
            "history": [], "last": None}


def test_persistence_helpers_and_timeout():
    sb = PersistentSandbox(sys_path=[ROOT] + sys.path)
    g = np.zeros((64, 64), dtype=int)
    g[10:14, 10:14] = 8
    r = sb.run("x = 41\nobjs = objects()\nprint(len(objs), objs[0]['color'], scale)\nnote('seen red block')", state(g), timeout_s=20)
    assert r["error"] == "", r
    assert r["stdout"].strip() == "1 8 2"
    assert r["notes"] == ["seen red block"]
    r = sb.run("x += 1\nprint(x)\nresult = {'x': x}", state(g), timeout_s=20)
    assert r["stdout"].strip() == "42" and r["result"] == {"x": 42}
    r = sb.run("print(diff()['changed'])", state(g, [np.zeros((64, 64), dtype=int), g]), timeout_s=20)
    assert r["stdout"].strip() == "16"
    t0 = time.time()
    r = sb.run("while True: pass", state(g), timeout_s=2)
    assert r["timed_out"] and r["restarted"] and time.time() - t0 < 10
    r = sb.run("print(x)", state(g), timeout_s=20)
    assert "NameError" in r["error"]  # variables were lost on restart
    sb.stop()


def test_actions_round_trip():
    sb = PersistentSandbox(sys_path=[ROOT] + sys.path)
    seen = []

    def handler(actions):
        seen.extend(actions)
        g = np.full((64, 64), len(seen), dtype=int)
        results = [{"changed": 1, "level_completed": False, "state": "NOT_FINISHED"} for _ in actions]
        return results, state(g)

    code = "r = act('UP')\nprint(r['changed'], int(grid[0,0]))\nr2 = act(('CLICK', 3, 4), 'LEFT')\nprint(len(r2), int(grid[0,0]))\nclick(1,2)\nprint(int(grid[0,0]))"
    r = sb.run(code, state(), timeout_s=20, action_handler=handler)
    assert r["error"] == "", r
    assert r["stdout"].split("\n")[:3] == ["1 1", "2 3", "4"]
    assert seen[1] == {"action": "CLICK", "x": 3, "y": 4}
    assert r["actions"] == 4
    sb.stop()
