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


def test_competing_hypotheses_die_on_first_wrong_prediction():
    sb = PersistentSandbox(sys_path=[ROOT] + sys.path)
    grids = [np.zeros((64, 64), dtype=int)]

    def handler(actions):
        g = grids[-1].copy()
        if actions[0]["action"] == "UP":
            g[0, 0] += 1  # the truth: UP increments (0,0)
        grids.append(g)
        return [{"changed": 1, "level_completed": False, "state": "NOT_FINISHED"}], state(g, [g])

    code = """
def inc(grid, action):
    g = grid.copy()
    if action == 'UP': g[0, 0] += 1
    return g
def dec(grid, action):
    g = grid.copy()
    if action == 'UP': g[0, 0] -= 1
    return g
def noop(grid, action):
    return grid
print(set_models({'inc': inc, 'dec': dec, 'noop': noop}))
r = act('DOWN'); print(r['alive_models'])    # nobody predicts a change: all survive
r = act('UP'); print(r['alive_models'])      # only inc survives
s = world_model_stats()['hypotheses']; print(s['dec']['alive'], s['inc']['correct'], 'UP' in s['noop']['killed_by'])
"""
    r = sb.run(code, state(grids[0], [grids[0]]), timeout_s=20, action_handler=handler)
    assert r["error"] == "", r
    lines = r["stdout"].strip().splitlines()
    assert lines[0] == "3 hypotheses registered"
    assert lines[1] == "['dec', 'inc', 'noop']"
    assert lines[2] == "['inc']"
    assert lines[3] == "False 2 True"
    assert r["world_model"]["hypotheses"]["inc"]["alive"] is True
    sb.stop()


def test_entities_exposed_in_sandbox():
    sb = PersistentSandbox(sys_path=[ROOT] + sys.path)
    grids = []

    def scene(x):
        g = np.zeros((64, 64), dtype=int)
        g[60:64, :] = 3
        g[20:24, x:x + 4] = 9
        return g

    pos = {"x": 4}

    def handler(actions):
        if actions[0]["action"] == "RIGHT":
            pos["x"] += 4
        g = scene(pos["x"])
        grids.append(g)
        return [{"changed": 32, "level_completed": False, "state": "NOT_FINISHED"}], state(g, [g])

    code = """
e0 = ents(); print(len(e0), sorted(x['color'] for x in e0))
act('RIGHT'); act('RIGHT'); act('UP')
ev = events(3); print([len(r['moved']) for r in ev])
print(describe_events(1)[0])
a = avatar(); print(a['id'], a['keymap'].get('RIGHT'))
print(roles()[a['id']], tile)
"""
    r = sb.run(code, state(scene(4), [scene(4)]), timeout_s=20, action_handler=handler)
    assert r["error"] == "", r
    lines = r["stdout"].strip().splitlines()
    assert lines[0] == "2 [3, 9]"
    assert lines[1] == "[1, 1, 0]"
    assert lines[2].startswith("UP: no entity changed")
    assert lines[3].endswith("(4, 0)")
    assert lines[4].startswith("avatar")
    sb.stop()


def test_plan_to_entity_in_sandbox():
    from tests.test_planner import GridWorld

    sb = PersistentSandbox(sys_path=[ROOT] + sys.path)
    w = GridWorld()

    def handler(actions):
        g = w.act(actions[0]["action"])
        return [{"changed": 32, "level_completed": False, "state": "NOT_FINISHED"}], state(g, [g])

    code = """
act('RIGHT', 'UP', 'DOWN', 'LEFT')
tid = [e['id'] for e in ents() if e['color'] == 12][0]
plan = plan_to_entity(tid); print(len(plan) >= 10, plan[0])
print(set_model(move_model().predict))
rs = act(plan); print(all(r['pred_ok'] for r in rs), len(rs) == len(plan))
"""
    r = sb.run(code, state(w.grid(), [w.grid()]), timeout_s=30, action_handler=handler)
    assert r["error"] == "", r
    lines = r["stdout"].strip().splitlines()
    assert lines[0].startswith("True")
    assert lines[1] == "world model registered"
    assert lines[2] == "True True"
    assert abs(w.pos[0] - w.target[0]) <= 4 and abs(w.pos[1] - w.target[1]) <= 4
    sb.stop()


def test_rules_fit_plan_and_goal_candidates_in_sandbox():
    from tests.test_planner import GridWorld

    sb = PersistentSandbox(sys_path=[ROOT] + sys.path)
    w = GridWorld()
    lvl = {"n": 1}

    def handler(actions):
        g = w.act(actions[0]["action"])
        x, y = w.pos
        tx, ty = w.target
        done = abs(x - tx) <= 4 and abs(y - ty) <= 4  # touching the target completes the level
        if done:
            lvl["n"] += 1
            w.pos = (8, 32)  # next level: same layout, avatar back at the start
            g = w.grid()
        st = state(g, [g])
        st["level"] = lvl["n"]
        return [{"changed": 32, "level_completed": done, "state": "NOT_FINISHED"}], st

    st0 = state(w.grid(), [w.grid()])
    st0["level"] = 1
    code = """
act('RIGHT', 'UP', 'DOWN', 'LEFT')
rep = auto_rules(); print(rep['coverage'], rep['contradictions'], len(rep['rules']))
print(rep['rules'][0][:12])
tid = [e['id'] for e in ents() if e['color'] == 12][0]
hint_ok = goal_hints()[0]['goal'] == {'reach_entity': tid}
ps = probe_suggestions(); ps_ok = all(p['action'] not in ('UP', 'DOWN', 'LEFT', 'RIGHT') for p in ps)  # all keys were pressed
plan = plan_rules({'reach_entity': tid}); print(plan is not None and len(plan) >= 10)
print(set_model(rules_predictor()))
rs = act(plan); print(all(r.get('pred_ok', True) for r in rs), rs[-1]['level_completed'])
print(goal_candidates())
print(level, len(symlog()))
print(hint_ok, ps_ok)
"""
    r = sb.run(code, st0, timeout_s=60, action_handler=handler)
    assert r["error"] == "", r
    lines = r["stdout"].strip().splitlines()
    assert lines[0] == "1.0 0 1", lines
    assert lines[1].startswith("move[colour"), lines
    assert lines[2] == "True", lines
    assert lines[3] == "world model registered"
    assert lines[4] == "True True", lines
    assert "touch(colour 9, colour 12)" in lines[5], lines
    assert lines[6] == "2 0", lines  # new level: the symbolic log restarted
    assert lines[7] == "True True", lines  # the unique-colour target is the first goal hint; no key left to probe
    sb.stop()


def test_optimistic_plan_when_strict_planner_has_no_path():
    """A target inside terrain the avatar has never walked on: the strict planners return None, the optimistic
    fallback returns a path and PLAN['optimistic'] says so."""
    from tests.test_planner import GridWorld

    class TwoFloors(GridWorld):
        def grid(self):
            g = super().grid()
            g[(g == 0) & (np.arange(64)[None, :] >= 36)] = 7  # the right half has a floor colour never walked on
            return g

    sb = PersistentSandbox(sys_path=[ROOT] + sys.path)
    w = TwoFloors()

    def handler(actions):
        g = w.act(actions[0]["action"])
        return [{"changed": 32, "level_completed": False, "state": "NOT_FINISHED"}], state(g, [g])

    code = """
act('UP', 'DOWN', 'LEFT', 'RIGHT')  # the avatar only ever stands on colour-0 floor
tid = [e['id'] for e in ents() if e['color'] == 12][0]
p_strict = plan_to_entity(tid, optimistic=False)
p_opt = plan_to_entity(tid)
print(p_strict is None, p_opt is not None and len(p_opt) >= 8, PLAN['optimistic'])
rep = auto_rules()
r_strict = plan_rules({'reach_entity': tid}, optimistic=False)
r_opt = plan_rules({'reach_entity': tid})
print(r_strict is None, r_opt is not None and len(r_opt) >= 8, PLAN['optimistic'])
"""
    r = sb.run(code, state(w.grid(), [w.grid()]), timeout_s=60, action_handler=handler)
    assert r["error"] == "", r
    lines = r["stdout"].strip().splitlines()
    assert lines[0] == "True True True", lines
    assert lines[1] == "True True True", lines
    sb.stop()
