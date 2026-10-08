"""kaggle/franzen/patches/ours-04-search-helper.patch (method M3, docs/research/beat-tufa/patch-search-helper.md).

Behind OURS_SEARCH_HELPER=1 (read at call time, off by default) the python sandbox provides ``search`` and
``run_plan``, six lines of the system prompt describe them, and a call that calls ``search(`` may run its largest
time_limit + 15 s. With the flag off the harness must behave byte for byte as before: same bootstrap, system prompt,
tool time limits and tool results (checked against the tree with only ours-sandbox-timeout-keeps-work.patch).

The search module is stdlib-only, so its tests run on the copy inside the patch file itself. The rest runs the
patched tree in the bed venv (scripts/franzen_bed.py creates it) and needs his repo (scripts/franzen_tree.py).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import franzen_bed  # noqa: E402
import franzen_tree  # noqa: E402

PATCHES = ROOT / "kaggle" / "franzen" / "patches"
SANDBOX_FIX = PATCHES / "ours-sandbox-timeout-keeps-work.patch"
PATCH = PATCHES / "ours-04-search-helper.patch"
MODULE = "ARC3-Inference/inference/utils/search_helper.py"
BED_PY = Path(franzen_bed.DEFAULT_VENV) / "bin" / "python"

needs_tree = pytest.mark.skipif(not (franzen_tree.his_repo_path() / "ARC3-Inference").is_dir(),
                                reason="needs Franzen's repo (scripts/franzen_tree.py)")
needs_bed = pytest.mark.skipif(not BED_PY.exists(), reason="needs the franzen_bed venv (scripts/franzen_bed.py)")


def _new_file(patch: Path, rel: str) -> str:
    """The text of a file the patch adds, from its own hunk."""
    lines = patch.read_text(encoding="utf-8").splitlines()
    start = lines.index(f"+++ b/{rel}")
    out = []
    for line in lines[start + 1:]:
        if line.startswith("diff --git "):
            break
        if line.startswith("+"):
            out.append(line[1:])
    return "\n".join(out) + "\n"


@pytest.fixture(scope="module")
def sh(tmp_path_factory):
    path = tmp_path_factory.mktemp("sh") / "search_helper.py"
    path.write_text(_new_file(PATCH, MODULE))
    spec = importlib.util.spec_from_file_location("ours_search_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tools(sh, timeout=10**6, started=None, **runtime):
    runtime_globals = dict(runtime)
    return runtime_globals, sh.search_helper_globals(runtime_globals, timeout, started)


MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
MAZE = ["S...#.....",
        ".##.#.###.",
        ".#..#...#.",
        ".#.####.#.",
        ".#......#.",
        ".######.#G",
        "........##"]


def _grid_step(rows, walls="#"):
    def step(pos, action):
        r, c = pos[0] + MOVES[action][0], pos[1] + MOVES[action][1]
        if not (0 <= r < len(rows) and 0 <= c < len(rows[0])) or rows[r][c] in walls:
            return None
        return "dead" if rows[r][c] == "X" else (r, c)
    return step


def _find(rows, ch):
    return next((r, row.index(ch)) for r, row in enumerate(rows) if ch in row)


def _reference_distance(rows, start, goal, walls="#X"):
    dist, queue = {start: 0}, deque([start])
    while queue:
        pos = queue.popleft()
        for dr, dc in MOVES.values():
            r, c = pos[0] + dr, pos[1] + dc
            if 0 <= r < len(rows) and 0 <= c < len(rows[0]) and rows[r][c] not in walls and (r, c) not in dist:
                dist[(r, c)] = dist[pos] + 1
                queue.append((r, c))
    return dist.get(goal)


def _replay(step, start, plan):
    state = start
    for action in plan:
        state = step(state, action)
        assert state is not None and state != "dead", (action, plan)
    return state


# --- search -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [
    MAZE,
    ["S..X..G", ".#.#.#.", "......."],
    ["S#G", ".#.", "..."],
], ids=["maze", "hazard", "u-turn"])
def test_bfs_and_astar_find_shortest_paths_on_toy_grids(sh, rows):
    _, tools = _tools(sh)
    start, goal = _find(rows, "S"), _find(rows, "G")
    step = _grid_step(rows)
    shortest = _reference_distance(rows, start, goal)
    assert shortest
    bfs = tools["search"](start, step, lambda p: p == goal, list(MOVES))
    assert bfs["status"] == "found" and len(bfs["plan"]) == shortest and bfs["best_partial"] is None
    assert _replay(step, start, bfs["plan"]) == goal
    manhattan = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])  # noqa: E731
    astar = tools["search"](start, step, lambda p: p == goal, list(MOVES), heuristic=manhattan)
    assert astar["status"] == "found" and len(astar["plan"]) == shortest
    assert _replay(step, start, astar["plan"]) == goal
    assert astar["expanded"] <= bfs["expanded"]
    beam = tools["search"](start, step, lambda p: p == goal, list(MOVES), heuristic=manhattan, beam=3)
    assert beam["status"] == "found" and _replay(step, start, beam["plan"]) == goal


def test_the_start_can_be_the_goal_and_actions_can_depend_on_the_state(sh):
    _, tools = _tools(sh)
    assert tools["search"]((0, 0), _grid_step(MAZE), lambda p: True, list(MOVES))["plan"] == []
    calls = []

    def actions(pos):  # only RIGHT and DOWN are offered
        calls.append(pos)
        return ["RIGHT", "DOWN"]

    result = tools["search"]((0, 0), _grid_step(["....", "....", "...."]), lambda p: p == (2, 3), actions)
    assert result["status"] == "found" and sorted(result["plan"]) == ["DOWN"] * 2 + ["RIGHT"] * 3 and calls


def test_states_are_deduplicated_and_parents_stay_correct(sh):
    _, tools = _tools(sh)
    rows = ["......"] * 6
    calls = []
    step = _grid_step(rows)

    def counted(pos, action):
        calls.append((pos, action))
        return step(pos, action)

    result = tools["search"]((0, 0), counted, lambda p: False, list(MOVES))
    assert result["status"] == "exhausted" and result["seen"] == 36 and result["expanded"] == 36
    assert len(calls) == 36 * 4  # every state expanded once, however many paths reach it
    deepest = result["best_partial"]
    assert deepest["depth"] == 10 and _replay(step, (0, 0), deepest["plan"]) == deepest["state"] == (5, 5)
    # unhashable states are made hashable by default; a custom key merges states that differ only in a counter
    as_lists = lambda pos, action: None if step(tuple(pos), action) is None else list(step(tuple(pos), action))  # noqa: E731
    listed = tools["search"]([0, 0], as_lists, lambda p: p == [5, 5], list(MOVES))
    assert listed["status"] == "found" and len(listed["plan"]) == 10
    with_steps = lambda s, a: None if step(s[0], a) is None else (step(s[0], a), s[1] + 1)  # noqa: E731
    keyed = tools["search"](((0, 0), 0), with_steps, lambda s: False, list(MOVES), key=lambda s: s[0])
    assert keyed["status"] == "exhausted" and keyed["seen"] == 36


def test_astar_reopens_a_state_and_keeps_its_better_parent(sh):
    # inconsistent but admissible heuristic: C is first reached through B, D (3 moves), later through A (2 moves)
    graph = {"S": ["A", "B"], "A": ["C"], "B": ["D"], "D": ["C"], "C": ["E"], "E": ["F"], "F": ["G"], "G": []}
    h = {"S": 0, "A": 4, "B": 0, "D": 0, "C": 0, "E": 0, "F": 0, "G": 0}
    _, tools = _tools(sh)
    result = tools["search"]("S", lambda s, a: a if a in graph[s] else None, lambda s: s == "G",
                             lambda s: graph[s], heuristic=h.get)
    assert result["status"] == "found" and result["plan"] == ["A", "C", "E", "F", "G"]


def test_illegal_and_fatal_moves_are_skipped(sh):
    rows = ["S.X.G", ".....", "....."]
    _, tools = _tools(sh)
    result = tools["search"]((0, 0), _grid_step(rows), lambda p: p == (0, 4), list(MOVES))
    assert result["status"] == "found" and len(result["plan"]) == 6 and (0, 2) not in _path((0, 0), result["plan"])


def _path(start, plan):
    cells, pos = [start], start
    for action in plan:
        pos = (pos[0] + MOVES[action][0], pos[1] + MOVES[action][1])
        cells.append(pos)
    return cells


def test_an_unreachable_goal_is_exhausted_with_the_closest_partial(sh):
    rows = ["S..#G", "...#.", "...#."]
    _, tools = _tools(sh)
    goal = (0, 4)
    result = tools["search"]((0, 0), _grid_step(rows), lambda p: p == goal, list(MOVES),
                             heuristic=lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1]))
    assert result["status"] == "exhausted" and result["plan"] is None and result["seen"] == 9
    partial = result["best_partial"]
    assert partial["state"] == (0, 2) and partial["h"] == 2 and partial["plan"] == ["RIGHT", "RIGHT"]


def test_a_timeout_returns_the_best_partial(sh):
    _, tools = _tools(sh)

    def slowish(s, a):  # an endless state space, ~10 us a step: the time limit ends it, not the node cap
        for _ in range(200):
            pass
        return 2 * s + a

    t = time.monotonic()
    result = tools["search"](0, slowish, lambda s: False, [1, 2], heuristic=lambda s: -s, time_limit=0.5)
    assert result["status"] == "timeout" and result["plan"] is None and time.monotonic() - t < 1.5
    partial = result["best_partial"]
    assert partial["plan"] and _replay(slowish, 0, partial["plan"]) == partial["state"] == -partial["h"]
    # a step slower than the clock check interval: the alarm ends the search about 1 s after its limit
    t = time.monotonic()
    result = tools["search"](0, lambda s, a: time.sleep(0.4) or s + a, lambda s: False, [1, 2], time_limit=0.5)
    assert result["status"] == "timeout" and result["best_partial"] is not None and time.monotonic() - t < 2.5


def test_time_nodes_and_the_call_budget_are_capped(sh):
    _, tools = _tools(sh)
    result = tools["search"](0, lambda s, a: s + a, lambda s: s == 3, [1], time_limit=1000)
    assert result["status"] == "found" and result["time_limit"] == 60.0
    result = tools["search"](0, lambda s, a: 2 * s + a, lambda s: False, [1, 2], max_nodes=10**9)
    assert result["status"] == "max_nodes" and result["seen"] == 300_000
    _, short = _tools(sh, timeout=8, started=time.monotonic())  # a call that may run 8 s keeps 5 s in reserve
    result = short["search"](0, lambda s, a: s + a, lambda s: s == 3, [1])
    assert result["status"] == "found" and result["time_limit"] <= 3.0
    _, spent = _tools(sh, timeout=4, started=time.monotonic())
    assert spent["search"](0, lambda s, a: s + a, lambda s: s == 3, [1])["status"] == "timeout"


def test_action_cannot_run_inside_search_and_is_restored(sh):
    def real_action(actions):
        raise AssertionError("a real action ran inside search")

    runtime, tools = _tools(sh, action=real_action)
    with pytest.raises(RuntimeError, match="cannot run inside search"):
        tools["search"](0, lambda s, a: runtime["action"]([a]), lambda s: False, ["UP"])
    assert runtime["action"] is real_action


# --- run_plan -----------------------------------------------------------------------------------------------------


class _Frame:
    def __init__(self, pos, step):
        self.pos, self.step = pos, step


class _World:
    """The real game for run_plan: a grid the model's step function may not know fully."""

    def __init__(self, rows, runtime):
        self.rows, self.runtime = rows, runtime
        self.pos, self.steps, self.refuse = _find(rows, "S"), 0, set()
        runtime["current_frame"] = _Frame(self.pos, 0)

    def action(self, actions):
        (action,) = actions
        if action in self.refuse:
            return {"executed": False, "stop_reason": "known_noop"}
        nxt = _grid_step(self.rows)(self.pos, action)
        self.steps += 1
        if nxt == "dead":
            self.pos = _find(self.rows, "S")
            result = {"executed": True, "game_over": True}
        else:
            self.pos = nxt or self.pos
            result = {"executed": True, "level_completed": self.rows[self.pos[0]][self.pos[1]] == "G"}
        self.runtime["current_frame"] = _Frame(self.pos, self.steps)
        return result


def _world(sh, rows, timeout=10**6, started=None):
    runtime = {}
    world = _World(rows, runtime)
    runtime["action"] = world.action
    return world, sh.search_helper_globals(runtime, timeout, started)


def test_run_plan_executes_a_correct_plan_to_the_level_end(sh):
    rows = ["S...", "....", "...G"]
    world, tools = _world(sh, rows)
    model = _grid_step(rows)
    plan = tools["search"](world.pos, model, lambda p: p == (2, 3), list(MOVES))["plan"]
    out = tools["run_plan"](plan, lambda f: f.pos, model)
    assert out == {"executed": 5, "mismatch_at": None, "predicted": (2, 3), "observed": None,
                   "stop": "level_completed"}


@pytest.mark.parametrize("predicted", ["step", "list"])
def test_run_plan_stops_at_a_planted_mismatch(sh, predicted):
    believed = ["S....", ".....", "....G"]  # the model's map
    real = ["S.#..", ".....", "....G"]  # a wall it does not know about
    world, tools = _world(sh, real)
    model = _grid_step(believed)
    plan = ["RIGHT", "RIGHT", "RIGHT", "DOWN"]
    states, pos = [], world.pos
    for action in plan:
        pos = model(pos, action)
        states.append(pos)
    out = tools["run_plan"](plan, lambda f: f.pos, model if predicted == "step" else states)
    assert out == {"executed": 2, "mismatch_at": 1, "predicted": (0, 2), "observed": (0, 1), "stop": "mismatch"}
    assert world.steps == 2  # nothing ran after the surprise


def test_run_plan_stops_before_an_action_its_model_calls_illegal_or_fatal(sh):
    world, tools = _world(sh, ["SX..", "....", "...G"])
    out = tools["run_plan"](["DOWN", "UP", "UP"], lambda f: f.pos, _grid_step(["SX..", "....", "...G"]))
    assert out == {"executed": 2, "mismatch_at": 2, "predicted": None, "observed": (0, 0), "stop": "illegal"}
    out = tools["run_plan"](["RIGHT"], lambda f: f.pos, _grid_step(["SX..", "....", "...G"]))
    assert out["stop"] == "dead" and out["executed"] == 0 and world.steps == 2


def test_run_plan_reports_refusals_game_over_keys_and_the_call_clock(sh):
    world, tools = _world(sh, ["SX..", "....", "...G"])
    world.refuse.add("DOWN")
    assert tools["run_plan"](["DOWN"], lambda f: f.pos, [None])["stop"] == "known_noop" and world.steps == 0
    out = tools["run_plan"](["RIGHT", "DOWN"], lambda f: f.pos, [(0, 1), (1, 1)])  # the model missed the hazard
    assert out["stop"] == "game_over" and out["executed"] == 1 and world.steps == 1
    world, tools = _world(sh, ["S...", "...G"])
    out = tools["run_plan"](["RIGHT"], lambda f: (f.pos, f.step), [((0, 1), 99)], key=lambda s: s[0])
    assert out["stop"] == "done" and out["mismatch_at"] is None  # compared through key: the counter is ignored


class StaleStateActionError(Exception):
    pass


class KnownNoOpActionError(BaseException):
    pass


def test_run_plan_reports_the_stale_state_refusal_and_lets_the_others_end_the_snippet(sh):
    runtime = {"StaleStateActionError": StaleStateActionError}
    world = _World(["S...", "...G"], runtime)
    tools = sh.search_helper_globals(runtime, 10**6)
    calls = []

    def guarded(actions):  # like the harness: the action after one that changed nothing is refused
        calls.append(actions)
        if len(calls) == 2:
            raise StaleStateActionError("the previous action changed nothing")
        return world.action(actions)

    runtime["action"] = guarded
    out = tools["run_plan"](["UP", "RIGHT"], lambda f: f.pos, [(0, 0), (0, 1)])
    assert out == {"executed": 1, "mismatch_at": None, "predicted": (0, 0), "observed": (0, 0), "stop": "stale_state"}

    def refused(actions):
        raise KnownNoOpActionError("known no-op")

    runtime["action"] = refused
    with pytest.raises(KnownNoOpActionError):
        tools["run_plan"](["UP"], lambda f: f.pos, [(0, 0)])
    world, late = _world(sh, ["S..G"], timeout=4, started=time.monotonic() - 2)
    out = late["run_plan"](["RIGHT", "RIGHT"], lambda f: f.pos, [(0, 1), (0, 2)])
    assert out["stop"] == "time" and out["executed"] == 0 and world.steps == 0


# --- the patch on the notebook's tree, in the bed venv --------------------------------------------------------------


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    base, patched = tmp_path_factory.mktemp("base"), tmp_path_factory.mktemp("patched")
    franzen_tree.notebook_bundle(base, [SANDBOX_FIX])
    logs = franzen_tree.notebook_bundle(patched, [SANDBOX_FIX, PATCH])
    assert PATCH.name in logs and logs[PATCH.name].count("Applied patch") == 3
    return base / "src" / "ARC3-Inference", patched / "src" / "ARC3-Inference"


@needs_tree
def test_the_patch_applies_after_his_and_the_sandbox_fix(trees):
    _, patched = trees
    assert (patched / "inference/utils/search_helper.py").read_text() == _new_file(PATCH, MODULE)
    agent = (patched / "inference/agent/tool_agent.py").read_text()
    assert 'startswith("Tool timed out")' in agent and "_search_call_timeout(code, self._kept_functions" in agent
    assert agent.count('_get_env_bool("OURS_SEARCH_HELPER", False)') == 1


# One process per tree and flag: a set of snippets through ToolAgent._run_python_tool (fake game, real sandbox), the
# tool time limit each call got, the rendered tool results, the system prompt and the bootstrap's hash.
PROBE = r'''
import hashlib, json, sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from inference.agent import python_tool_sandbox as sb
from inference.agent import tool_agent as ta
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state

limits, real = [], ta.run_sandboxed_python
def spy(**kwargs):
    limits.append(kwargs["timeout_seconds"])
    return real(**kwargs)
ta.run_sandboxed_python = spy
agent = ta.ToolAgent(model="mock")
state_path = Path(tempfile.mkdtemp()) / "game" / "tool_runtime_state.json"
grid = tuple(tuple((r * 7 + c) % 3 for c in range(8)) for r in range(8))
write_runtime_state(state_path, current_frame=Frame(grid=grid, step=0, level=1), history=[])
agent._current_valid_actions = ["UP", "DOWN"]
count = [0]
def env(arguments):
    acts = arguments.get("actions") or []
    count[0] += len(acts)
    frame = Frame(grid=grid, step=count[0], level=1)
    write_runtime_state(state_path, current_frame=frame,
                        history=[HistoryEntry(action=str(a.get("action")), frame=frame) for a in acts])
    return {"executed": True, "action_num": count[0], "level": 1, "state": "NOT_FINISHED", "valid_actions": ["UP", "DOWN"],
            "board_changed": True, "gameplay_changed": True, "executed_actions": [a["action"] for a in acts],
            "requested_count": len(acts), "executed_count": len(acts)}
agent._step_env_callback = env
results = [agent._run_python_tool(state_path, {"code": code}).content for code in json.loads(sys.argv[2])]
print(json.dumps({"bootstrap": hashlib.sha256(sb._SANDBOX_BOOTSTRAP.encode()).hexdigest(), "limits": limits,
                  "prompt": agent._system_prompt, "results": results}))
'''

SNIPPETS = [
    "print(sorted(valid_actions), current_frame.step, current_frame.shape)",
    "print(search)",
    "def search(n):\n    return n + 1\nprint(search(1))",
    "r = action(['UP'])\nprint(r.get('executed'), current_frame.step)",
    "def helper(x):\n    return x * 2\nprint(helper(3))",
    "print(helper(4))",
]

# Flag on: search + run_plan against the fake game, a retained function that calls search (and its later call's
# limit), a search ending on its own time limit and one cut by the call's budget (both keep stdout), and a call the
# host kills, which keeps the retained functions (ours-sandbox-timeout-keeps-work.patch).
ON_SNIPPETS = [
    "s0 = current_frame.step\nr = search(s0, lambda s, a: s + 1, lambda s: s == s0 + 2, ['UP', 'DOWN'])\n"
    "print('plan', r['status'], r['plan'], r['time_limit'])\n"
    "print('run', run_plan(r['plan'], lambda f: f.step, lambda s, a: s + 1))\n"
    "print('planted', run_plan(['UP'], lambda f: f.step, [s0 + 99]))",
    "def plan_to(n):\n    return search(0, lambda s, a: s + 1, lambda s: s == n, ['UP'], time_limit=20)['plan']\n",
    "print(plan_to(3))",
    "def slow(s, a):\n    for _ in range(2000):\n        pass\n    return 2 * s + a\n"
    "r = search(0, slow, lambda s: False, [1, 2], time_limit=1)\nprint('own', r['status'], r['best_partial'] is not None)",
    "f = search\nr = f(0, slow, lambda s: False, [1, 2])\nprint('budget', r['status'], r['time_limit'], r['best_partial'] is not None)",
    "r = search(0, lambda s, a: s + 1, lambda s: s == 2, ['UP'], time_limit=0)\nimport time\ntime.sleep(60)",
    "print(plan_to(2))",
]


def _probe(tree: Path, snippets: list[str], **env) -> dict:
    nb_env, _ = franzen_bed.notebook_env(franzen_bed._cell(franzen_tree.notebook_cells(), "setup_env = {"))
    child = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/tmp"), **nb_env, **env}
    r = subprocess.run([str(BED_PY), "-c", PROBE, str(tree), json.dumps(snippets)], capture_output=True, text=True,
                       timeout=300, env=child)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@needs_tree
@needs_bed
def test_flag_off_is_byte_identical_to_the_sandbox_fix_alone(trees):
    base, patched = trees
    before = _probe(base, SNIPPETS)
    for flag in ({}, {"OURS_SEARCH_HELPER": "0"}):
        after = _probe(patched, SNIPPETS, **flag)
        assert after == before
    assert before["limits"] == [30] * len(SNIPPETS)
    assert "NameError: name 'search' is not defined" in before["results"][1]
    assert "Retained your function search(n)" in before["results"][2]


@needs_tree
@needs_bed
def test_flag_on_adds_the_helpers_their_lines_and_the_longer_limit(trees):
    base, patched = trees
    off = _probe(base, SNIPPETS[:1])
    on = _probe(patched, ON_SNIPPETS, OURS_SEARCH_HELPER="1", LOCAL_ANALYZER_TOOL_TIMEOUT="10")
    head, rules, tail = off["prompt"].partition("\n\nTool session rules:")
    assert rules and on["prompt"].startswith(head) and on["prompt"].endswith(rules + tail)
    added = on["prompt"][len(head):len(on["prompt"]) - len(rules + tail)]  # the lines sit before the session rules
    lines = added.split("\n")
    assert added.endswith("\n") and len(lines) - 1 <= 10 and all(line.startswith("- ") for line in lines[:-1])
    assert "search(start, step, is_goal, actions" in added and "run_plan(plan, observe, predicted" in added
    assert "solver" not in added.lower()
    # tool limits: 10 s base (this probe's env); a call with search( in its code, or calling a retained function that
    # calls it, gets its largest time_limit + 15 s (default 60); the call that reaches search only through `f` does not
    assert on["limits"] == [75, 35, 35, 16, 10, 15, 35], on["limits"]
    run, define, through, own, budget, killed, after = (json.loads(r) for r in on["results"])
    assert "plan found ['UP', 'UP'] 60.0" in run["stdout"]
    assert "'executed': 2, 'mismatch_at': None" in run["stdout"] and "'stop': 'done'" in run["stdout"]
    assert "'executed': 1, 'mismatch_at': 0" in run["stdout"] and "'stop': 'mismatch'" in run["stdout"]
    assert "Retained your function plan_to(n)" in define["function_retention"]
    assert through["stdout"].strip() == "['UP', 'UP', 'UP']"
    assert own["stdout"].strip() == "own timeout True" and "error" not in own
    word, status, limit, partial = budget["stdout"].split()  # cut to the call's 10 s minus its 5 s reserve
    assert (word, status, partial) == ("budget", "timeout", "True") and 4.0 <= float(limit) <= 5.0
    assert killed["error"] == "Tool timed out after 15s" and "still available" in killed["function_retention"]
    assert after["stdout"].strip() == "['UP', 'UP']"


TIMEOUTS = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from inference.agent import tool_agent as ta
cases = json.loads(sys.argv[2])
print(json.dumps([ta._search_call_timeout(code, retained, 30) for code, retained in cases]))
'''


@needs_tree
@needs_bed
def test_only_calls_that_call_search_get_the_longer_limit(trees):
    _, patched = trees
    plan = {"plan": "def plan():\n    return search(1, 2, 3, 4, time_limit=20)\n"}
    chain = {"outer": "def outer():\n    return inner()\n", "inner": "def inner():\n    return search(0, f, g, [1])\n"}
    cases = [
        ("print(1)", {}, 30),
        ("search(s, st, g, a)", {}, 75),
        ("search(s, st, g, a, time_limit=20)", {}, 35),
        ("search(s, st, g, a, time_limit=5)", {}, 30),
        ("search(s, st, g, a, None, None, None, 1000, 40)", {}, 55),
        ("search(s, st, g, a, time_limit=t)", {}, 75),
        ("search(*args)", {}, 75),
        ("search(s, st, g, a, **kw)", {}, 75),
        ("search(s, st, g, a, time_limit=1e9)", {}, 75),
        ("search(s, st, g, a, time_limit=2.5)", {}, 30),
        ("import re\nre.search('a', 'b')", {}, 30),
        ("bfs_search(1)", {}, 30),
        ("x = 'search(1)'", {}, 30),
        ("plan()", plan, 35),
        ("plan", plan, 30),
        ("outer()", chain, 75),
        ("def f():\n    return search(0, a, b, c, time_limit=33)\n", {}, 48),
    ]
    r = subprocess.run([str(BED_PY), "-c", TIMEOUTS, str(patched), json.dumps([[c, k] for c, k, _ in cases])],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-3000:]
    assert json.loads(r.stdout) == [want for *_, want in cases]
