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


def test_world_model_predictions_are_checked_and_batches_stop_on_mismatch():
    sb = PersistentSandbox(sys_path=[ROOT] + sys.path)
    grids = []

    def handler(actions):
        # Environment: UP paints cell (0,0) with 5; anything else does nothing.
        g = np.array(grids[-1]) if grids else np.zeros((64, 64), dtype=int)
        g = g.copy()
        if actions[0]["action"] == "UP":
            g[0, 0] = 5
        grids.append(g)
        return [{"changed": 1, "level_completed": False, "state": "NOT_FINISHED"}], state(g, [g])

    code = """
def predict(grid, action):
    g = grid.copy()
    if action == 'UP':
        g[0, 0] = 5   # correct
    elif action == 'DOWN':
        g[1, 1] = 7   # wrong: DOWN does nothing in this game
    return g
print(set_model(predict))
r = act('UP')
print('up', r['pred_ok'], r['pred_wrong_cells'])
rs = act('DOWN', 'UP', 'UP')
print('batch', len(rs), rs[0]['pred_ok'], rs[0]['pred_wrong_cells'], 'mismatch' in rs[0].get('batch_stopped', ''))
print(world_model_stats()['checked'], world_model_stats()['matched'])
"""
    r = sb.run(code, state(), timeout_s=20, action_handler=handler)
    assert r["error"] == "", r
    lines = r["stdout"].strip().splitlines()
    assert lines[0] == "world model registered"
    assert lines[1] == "up True 0"
    assert lines[2] == "batch 1 False 1 True"
    assert lines[3] == "2 1"
    assert r["world_model"]["checked"] == 2 and r["world_model"]["matched"] == 1
    assert r["actions"] == 2  # the batch stopped after the first mismatching action
    sb.stop()


def test_transition_log_and_verify_model_counter_examples():
    sb = PersistentSandbox(sys_path=[ROOT] + sys.path)
    grids = [np.zeros((64, 64), dtype=int)]

    def handler(actions):
        g = grids[-1].copy()
        a = actions[0]
        if a["action"] == "UP":
            g[0, 0] += 1
        elif a["action"] == "CLICK":
            g[a["y"], a["x"]] = 9
        grids.append(g)
        return [{"changed": 1, "level_completed": False, "state": "NOT_FINISHED"}], state(g, [g])

    code = """
act('UP', 'UP', ('CLICK', 3, 4))
print(len(transitions()), transitions()[2][1])
def good(grid, action):
    g = grid.copy()
    if action == 'UP': g[0, 0] += 1
    elif isinstance(action, tuple) and action[0] == 'CLICK': g[action[2], action[1]] = 9
    return g
def bad(grid, action):
    return grid
v = verify_model(good); print('good', v['checked'], v['correct'], len(v['counter_examples']))
v = verify_model(bad); print('bad', v['checked'], v['correct'], v['counter_examples'][0]['index'], v['counter_examples'][0]['wrong_cells'])
print(set_model(good)); r = act('UP'); print(r['pred_ok'], len(transitions()))
"""
    r = sb.run(code, state(grids[0], [grids[0]]), timeout_s=20, action_handler=handler)
    assert r["error"] == "", r
    lines = r["stdout"].strip().splitlines()
    assert lines[0] == "3 ('CLICK', 3, 4)"
    assert lines[1] == "good 3 3 0"
    assert lines[2] == "bad 3 0 0 1"
    assert lines[3] == "world model registered" and lines[4] == "True 4"
    # a level change clears the log
    r = sb.run("print(len(transitions()))", {**state(grids[-1], [grids[-1]]), "level": 2}, timeout_s=20, action_handler=handler)
    assert r["stdout"].strip() == "0"
    sb.stop()
