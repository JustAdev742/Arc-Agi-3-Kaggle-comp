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
    assert lines[1].startswith("world model registered")
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
    assert lines[3].startswith("world model registered")
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


def test_act_results_carry_events_live_move_model_and_retirement():
    """Every act() result carries the entity events of its transition; set_model(move_model().predict) re-fits before
    each prediction; a predictor that is wrong three times running is retired (result gains 'pred_retired')."""
    from tests.test_planner import GridWorld

    sb = PersistentSandbox(sys_path=[ROOT] + sys.path)
    w = GridWorld()

    def handler(actions):
        g = w.act(actions[0]["action"])
        return [{"changed": 32, "level_completed": False, "state": "NOT_FINISHED"}], state(g, [g])

    code = """
rs = act('RIGHT', 'UP', 'DOWN', 'LEFT')
print(all('moved' in r['events'] for r in rs), rs[0]['events'][:40])
print(set_model(move_model().predict))
r = act('RIGHT'); print(r['pred_ok'], 'moved (+4,+0)' in r['events'])
set_model(lambda grid, action: grid)  # always predicts no change: wrong on every move
rs = act('LEFT', 'LEFT', 'LEFT', 'LEFT')
print([r.get('pred_ok') for r in rs], 'pred_retired' in rs[-1] or any('pred_retired' in r for r in rs))
print(world_model_stats()['checked'])
r = act('RIGHT'); print('pred_ok' in r)
"""
    r = sb.run(code, state(w.grid(), [w.grid()]), timeout_s=30, action_handler=handler)
    assert r["error"] == "", r
    lines = r["stdout"].strip().splitlines()
    assert lines[0].startswith("True #"), lines[0]
    assert lines[1].startswith("world model registered (live move model")
    assert lines[2] == "True True"
    # the always-wrong model stops the batch on its first miss; the batch is re-issued by the model in real play,
    # here three single mismatches retire it
    assert lines[3].startswith("[False]")
    sb.stop()


def test_world_model_retires_after_three_consecutive_mismatches():
    from tests.test_planner import GridWorld

    sb = PersistentSandbox(sys_path=[ROOT] + sys.path)
    w = GridWorld()

    def handler(actions):
        g = w.act(actions[0]["action"])
        return [{"changed": 32, "level_completed": False, "state": "NOT_FINISHED"}], state(g, [g])

    code = """
set_model(lambda grid, action: grid)
out = [act('RIGHT') for _ in range(3)]
print([r['pred_ok'] for r in out], 'pred_retired' in out[-1], 'pred_retired' in out[0])
r = act('RIGHT'); print('pred_ok' in r, world_model_stats()['checked'])
"""
    r = sb.run(code, state(w.grid(), [w.grid()]), timeout_s=30, action_handler=handler)
    assert r["error"] == "", r
    lines = r["stdout"].strip().splitlines()
    assert lines[0] == "[False, False, False] True False"
    assert lines[1] == "False 3"  # retired: the fourth action is not checked
    sb.stop()


def test_rebound_helpers_are_restored_and_rules_fit_lazily():
    """exp-009: `for act in plan` broke every later act(); plan_rules() before auto_rules() raised. Both are harness-side."""
    from tests.test_planner import GridWorld

    sb = PersistentSandbox(sys_path=[ROOT] + sys.path)
    w = GridWorld()

    def handler(actions):
        g = w.act(actions[0]["action"])
        return [{"changed": 32, "level_completed": False, "state": "NOT_FINISHED"}], state(g, [g])

    r = sb.run("for act in ['UP', 'DOWN']:\n    pass\nrules = 'shadow'\nprint(type(act).__name__)",
               state(w.grid(), [w.grid()]), timeout_s=30, action_handler=handler)
    assert r["error"] == "" and "str" in r["stdout"] and "act, rules" in r["stdout"] and "restored" in r["stdout"]
    r = sb.run("rs = act('UP', 'DOWN', 'LEFT', 'RIGHT'); print(len(rs), callable(rules))", state(w.grid(), [w.grid()]),
               timeout_s=30, action_handler=handler)
    assert r["error"] == "" and r["stdout"].strip() == "4 True", r
    r = sb.run("p = rules_predictor(); print(callable(p), len(rules()) >= 1)", state(w.grid(), [w.grid()]), timeout_s=30, action_handler=handler)
    assert r["error"] == "" and r["stdout"].strip() == "True True", r
    r = sb.run("def state():\n    return 'mine'\nact('UP'); print(state(), STATE['state'])", state(w.grid(), [w.grid()]), timeout_s=30, action_handler=handler)
    assert r["error"] == "" and r["stdout"].strip().startswith("mine NOT_FINISHED"), r
    sb.stop()


def test_idle_batch_stops_and_optimistic_rules_plan_is_verified_optimistically():
    """dc22 (exp-009): an optimistic plan_rules() plan ran under the strict rules predictor, so 48 blocked moves were
    'predicted correctly' as standing still. Now (1) a batch stops after three no-change actions and (2) an optimistic
    plan is checked against the optimistic rules, so the first blocked step is a mismatch."""
    from tests.test_planner import GridWorld

    sb = PersistentSandbox(sys_path=[ROOT] + sys.path)
    w = GridWorld()

    def handler(actions):
        before = w.pos
        g = w.act(actions[0]["action"])
        return [{"changed": 0 if w.pos == before else 32, "level_completed": False, "state": "NOT_FINISHED"}], state(g, [g])

    code = """
rs = act('LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT')  # one step to the left wall, then nothing changes
print(len(rs), rs[-1].get('batch_stopped', '')[:40])
act('RIGHT', 'UP', 'DOWN')
r = auto_rules(); print(r['coverage'] > 0)
print(set_model(rules_predictor())[:60])
PLAN['optimistic'] = True
rs = act(['RIGHT'] * 6)  # the middle wall (colour 3, never touched) stands at x=28: strict says blocked, optimistic says move
print([x.get('pred_ok') for x in rs], 'batch_stopped' in rs[-1])
"""
    r = sb.run(code, state(w.grid(), [w.grid()]), timeout_s=60, action_handler=handler)
    assert r["error"] == "", r
    lines = r["stdout"].strip().splitlines()
    assert lines[0] == "4 3 actions in a row changed nothing; stop", lines[0]
    assert lines[1] == "True"
    assert lines[2].startswith("world model registered (rules predictor")
    assert lines[3] == "[True, True, True, True, False] True", lines[3]
    sb.stop()


def test_single_result_iterates_region_degrades_and_cell_events_are_reported():
    """exp-011 tool errors: `for r in act('UP')` iterated dict keys; ascii(region=big) refused nine times; ents() entries
    lacked 'role'; and 77 inspection-only calls re-read events that the harness can attach to the cell's output."""
    from tests.test_planner import GridWorld

    sb = PersistentSandbox(sys_path=[ROOT] + sys.path)
    w = GridWorld()

    def handler(actions):
        g = w.act(actions[0]["action"])
        return [{"changed": 32, "level_completed": False, "state": "NOT_FINISHED"}], state(g, [g])

    code = """
r = act('UP')
print([x.get('changed') for x in r], isinstance(r, dict), r['changed'])
big = ascii(region=(0, 0, 63, 63)); print(big.splitlines()[0][:40], len(big.splitlines()) > 5)
print(all('role' in e for e in ents()))
"""
    r = sb.run(code, state(w.grid(), [w.grid()]), timeout_s=30, action_handler=handler)
    assert r["error"] == "", r
    lines = r["stdout"].strip().splitlines()
    assert lines[0] == "[32] True 32", lines[0]
    assert lines[1].startswith("(region 0,0-63,63 is 64x64 pixels") and lines[1].endswith("True")
    assert lines[2] == "True"
    assert r["events"] and "moved" in r["events"][0], r["events"]
    r2 = sb.run("x = 1", state(w.grid(), [w.grid()]), timeout_s=30, action_handler=handler)
    assert r2["events"] == []
    sb.stop()


def test_observed_terminal_frame_feeds_goal_candidates_and_dict_goals():
    """The harness attaches the completed level's observed winning frame ('terminal') to the level-completing action
    result; the sandbox archives it (not a simulation), goal_candidates() sees the sprite parked exactly on its slot
    (same_box) and plan_rules accepts the new relation goals as dicts."""
    from tests.test_planner import GridWorld

    sb = PersistentSandbox(sys_path=[ROOT] + sys.path)
    w = GridWorld()
    lvl = {"n": 1}

    def handler(actions):
        g = w.act(actions[0]["action"])
        x, y = w.pos
        tx, ty = w.target
        done = abs(x - tx) <= 4 and abs(y - ty) <= 4
        res = {"changed": 32, "level_completed": done, "state": "NOT_FINISHED"}
        if done:
            # observed terminal: the avatar drawn exactly over the target box (same colour 9 sprite, same 4x4 box)
            term = g.copy()
            term[y:y + 4, x:x + 4] = 0
            term[ty:ty + 4, tx:tx + 4] = 9
            res["terminal"] = term.tolist()
            lvl["n"] += 1
            w.pos = (8, 32)
            g = w.grid()
        st = state(g, [g])
        st["level"] = lvl["n"]
        return [res], st

    st0 = state(w.grid(), [w.grid()])
    st0["level"] = 1
    code = """
act('RIGHT', 'UP', 'DOWN', 'LEFT')
auto_rules()
tid = [e['id'] for e in ents() if e['color'] == 12][0]
rs = act(plan_rules({'reach_entity': tid}))
print(rs[-1]['level_completed'], 'terminal' in rs[-1])
g = goal_candidates(); print(g)
print(plan_rules({'same_box': (9, 12)}) is not None, plan_rules({'inside': (9, 12)}) is None)  # equal sizes: inside is impossible
try:
    plan_rules({'bogus': (1, 2)})
except ValueError as e:
    print('same_box' in str(e))
"""
    r = sb.run(code, st0, timeout_s=60, action_handler=handler)
    assert r["error"] == "", r
    lines = r["stdout"].strip().splitlines()
    assert lines[0] == "True False", lines  # the terminal grid never reaches the model's result dict
    assert "same_box(colour 9, colour 12)" in lines[1], lines
    assert lines[2] == "True True", lines
    assert lines[3] == "True", lines
    sb.stop()
