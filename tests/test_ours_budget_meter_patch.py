"""kaggle/franzen/patches/ours-02-budget-meter.patch: the budget meter (method M2), behind OURS_BUDGET_METER.

The detector (inference/utils/budget_bar.py, a new file in the patch) is loaded straight from the patch text, so its
unit tests need neither Franzen's repo nor the game files: synthetic boards, and the edge region of every frame of
his 10-game demo run (tests/fixtures/franzen/budget-bar/demo-edges.json). The harness check runs in the bed venv on
the tree his notebook builds (his patch, then ours-sandbox-timeout-keeps-work.patch, then this one): with the flag
off the system prompt, the user prompts, the sandbox's state and the tool result are byte-identical to the tree
without this patch; with it on, the line appears after "Current state: ...", a budget death is named, and `budget`
reaches the sandbox.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import franzen_tree  # noqa: E402

PATCHES = ROOT / "kaggle" / "franzen" / "patches"
SANDBOX = PATCHES / "ours-sandbox-timeout-keeps-work.patch"
PATCH = PATCHES / "ours-02-budget-meter.patch"
FIXTURE = ROOT / "tests" / "fixtures" / "franzen" / "budget-bar" / "demo-edges.json"
BED_PY = Path.home() / ".cache" / "arc3-franzen-bed" / "venv" / "bin" / "python"
NEW_FILE = "ARC3-Inference/inference/utils/budget_bar.py"


def _load_meter() -> types.ModuleType:
    """The new module exactly as the patch adds it."""
    text = PATCH.read_text()
    section = text.split(f"+++ b/{NEW_FILE}\n", 1)[1].split("\ndiff --git ", 1)[0]
    body = section.split("\n", 1)[1]  # after the @@ line
    source = "\n".join(line[1:] for line in body.splitlines() if line.startswith("+")) + "\n"
    module = types.ModuleType("budget_bar")
    exec(compile(source, NEW_FILE, "exec"), module.__dict__)
    return module


bb = _load_meter()


# --- synthetic boards ---------------------------------------------------------------------------------------------


def board(bar=None, *, obj=0, row=63, colour=6, empty=0, extra=None):
    """64x64 board of colour 5: a bar of `bar` cells on `row` (colour, then `empty`), an object at (30, 10+obj)."""
    g = [[5] * 64 for _ in range(64)]
    if bar is not None:
        g[row] = [colour] * bar + [empty] * (64 - bar)
    g[30][10 + obj % 40] = 9
    for r, cells in (extra or {}).items():
        g[r] = list(cells)
    return tuple(tuple(r) for r in g)


def drain(cells_per_step, steps, *, start=64, action="RIGHT", level=1, first_step=1, **kw):
    """Entries for `steps` actions that each move the object and leave the bar at cells_per_step(i) cells."""
    return [(action, board(max(0, cells_per_step(i)), obj=first_step + i, **kw), level, first_step + i)
            for i in range(steps)]


def history(*parts, start=64, **kw):
    out = [("", board(start, **kw), 1, 0)]
    for p in parts:
        out.extend(p)
    return out


def reading(entries):
    return bb.public(bb.BudgetMeter().read(entries))


def test_silent_until_three_drops_then_exact():
    h = history(drain(lambda i: 62 - 2 * i, 12))
    assert reading(h[:3]) is None  # two drops
    r = reading(h[:4])
    assert r["line"] == "row 63" and r["colour"] == "M" and r["cells_left"] == 58
    assert r["cells_per_action"] == 2.0 and r["actions_left"] == 29
    r = reading(h)
    assert r["cells_left"] == 40 and r["actions_left"] == 20 and r["drops_seen"] == 12
    assert bb.prompt_line(bb.BudgetMeter().read(h)) == (
        "Budget bar (row 63, colour 'M'): about 20 more actions before it is empty (40 cells left, about 2.0 per "
        "action). `budget` in python has these numbers.")


def test_slow_bar_counts_the_actions_since_its_last_tick():
    # one cell every 3 actions: 64 -> 63 at action 3, 62 at 6, 61 at 9
    h = history(drain(lambda i: 64 - (i + 1) // 3, 11))
    r = reading(h[:10])
    assert r["cells_per_action"] == 0.33 and r["actions_left"] == 183
    assert reading(h[:11])["actions_left"] == 182 and reading(h[:12])["actions_left"] == 181


def test_fast_rounded_bar_reads_like_tu93_near_the_end():
    # 20 steps on 64 cells, drawn as round(64 * c / 20): 3.2 cells per action
    h = history(drain(lambda i: round(64 * (19 - i) / 20), 20))
    lefts = [reading(h[:t + 1])["actions_left"] for t in range(15, 21)]
    assert lefts == [4, 4, 3, 1, 0, 0]  # cells 16, 13, 10, 6, 3, 0; the game's counter says 5, 4, 3, 2, 1, 0


def test_reset_and_level_change_start_a_new_attempt():
    h = history(drain(lambda i: 62 - 2 * i, 6))
    assert reading(h) is not None
    after_reset = h + [("RESET", board(64, obj=50), 1, 7)] + drain(lambda i: 62 - 2 * i, 2, first_step=8)
    assert reading(after_reset) is None
    assert reading(after_reset + drain(lambda i: 58, 1, first_step=10))["actions_left"] == 29
    level_two = h + drain(lambda i: 64 - i, 2, level=2, first_step=7)
    assert reading(level_two) is None


def test_silent_on_lines_that_move_or_are_thick():
    # an object on row 1: its colour falls three times, then rises by one cell
    moving = [("", board(None, extra={1: [3] * 20 + [5] * 44}), 1, 0)]
    for i, n in enumerate([18, 16, 14, 15]):
        moving.append(("UP", board(None, obj=i + 1, extra={1: [3] * n + [5] * (64 - n)}), 1, i + 1))
    assert reading(moving[:4]) is not None  # three steady drops look like a bar ...
    assert reading(moving) is None  # ... until it rises a little
    # three identical rows draining together are an object in the border region, not a meter
    thick = [("", board(None, extra={r: [3] * 40 + [5] * 24 for r in (0, 1, 2)}), 1, 0)]
    for i in range(1, 7):
        n = 40 - 4 * i
        thick.append(("UP", board(None, obj=i, extra={r: [3] * n + [5] * (64 - n) for r in (0, 1, 2)}), 1, i))
    assert reading(thick) is None


def test_two_bars_that_disagree_are_silent():
    def two(n_top, n_bottom, i):
        return board(n_bottom, obj=i, extra={0: [7] * n_top + [4] * (64 - n_top)})
    h = [("", two(64, 64, 0), 1, 0)] + [("RIGHT", two(64 - 4 * i, 64 - i, i), 1, i) for i in range(1, 6)]
    assert reading(h) is None  # 11 actions left on row 0, 59 on row 63


def test_a_known_bar_outranks_a_new_line_in_later_attempts():
    # attempt 1 establishes row 63; in attempt 2 row 1 drains fast while row 63 has not yet dropped 3 times
    h = history(drain(lambda i: 62 - 2 * i, 6), extra={1: [3] * 48 + [5] * 16})
    h += [("RESET", board(64, extra={1: [3] * 48 + [5] * 16}), 1, 7)]
    slow = [62, 62, 62, 60, 60, 60]
    fast = [44, 40, 36, 32, 28, 24]
    for i in range(6):
        h.append(("RIGHT", board(slow[i], obj=8 + i, extra={1: [3] * fast[i] + [5] * (64 - fast[i])}), 1, 8 + i))
        if i < 5:
            assert reading(h) is None
    # a fresh game with the same frames has no known bar: there row 1 drains alone and would be read
    assert reading([("", h[7][1], 1, 0)] + [(a, g, lv, s - 7) for a, g, lv, s in h[8:12]])["line"] == "row 1"


def test_a_bar_that_drains_into_a_new_colour_is_silent():
    # level 1: the bar drains into colour 0; level 2: the same bar drains into colour 12 (a second layer under it?)
    h = history(drain(lambda i: 62 - 2 * i, 5)) + drain(lambda i: 64 - 2 * i, 6, level=2, first_step=6, empty=12)
    assert reading(h) is None
    same = history(drain(lambda i: 62 - 2 * i, 5)) + drain(lambda i: 64 - 2 * i, 6, level=2, first_step=6)
    assert reading(same)["cells_left"] == 54


def test_free_kinds_and_blocked_moves_are_not_charged():
    h = [("", board(64), 1, 0)]
    bar, step = 64, 0
    click = "MOUSE(row=30, col=12)"
    for kind in ["RIGHT", "RIGHT", click, "RIGHT", click, "RIGHT", click, "RIGHT"]:
        step += 1
        bar -= 2 if kind == "RIGHT" else 0
        h.append((kind, board(bar, obj=step), 1, step))  # every action moves the object
    r = reading(h)
    assert r["free_actions"] == ["MOUSE"] and r["cells_per_action"] == 2.0 and r["actions_left"] == 27
    assert "; MOUSE did not lower it." in bb.prompt_line(bb.BudgetMeter().read(h))
    # a move that changed nothing on the board and did not lower the bar was not charged either
    blocked = h + [("LEFT", board(bar, obj=step), 1, step + 1)]
    assert reading(blocked)["actions_left"] == 27


def test_budget_death_is_named_only_when_the_bar_ran_out():
    run_out = history(drain(lambda i: 62 - 2 * i, 31)) + [("RIGHT", board(0, obj=40), 1, 32),
                                                         ("RESET", board(64), 1, 33)]
    verdict = bb.BudgetMeter().death(run_out)
    assert verdict is not None and verdict["cells_left_at_death"] == 0
    assert bb.death_line(verdict) == ("Budget meter: the bar (row 63, colour 'M') ran out at the fatal action, so "
                                      "this was a budget death.")
    hazard = history(drain(lambda i: 62 - 2 * i, 10)) + [("RIGHT", board(40, obj=40), 1, 11),
                                                        ("RESET", board(64), 1, 12)]
    assert bb.BudgetMeter().death(hazard) is None
    assert bb.BudgetMeter().death(run_out[:-1]) is None  # not a history that ends in the automatic RESET


def test_the_meter_cache_follows_a_growing_history_and_starts_over_for_a_new_one():
    h = history(drain(lambda i: 62 - 2 * i, 12))
    meter = bb.BudgetMeter()
    assert [bb.public(meter.read(h[:t])) for t in range(2, 14)] == [reading(h[:t]) for t in range(2, 14)]
    other = history(drain(lambda i: 63 - i, 8))
    assert bb.public(meter.read(other)) == reading(other)


# --- the recorded demo boards (Franzen's run of 2026-09-30) ---------------------------------------------------------

HEX = "0123456789abcdef"


def _demo_games() -> dict:
    data = json.loads(FIXTURE.read_text())
    games = {}
    for game in data["games"]:
        cur, frames = {}, []
        for f in game["frames"]:
            cur.update(f["e"])
            g = [[0] * 64 for _ in range(64)]
            for key, rle in cur.items():
                cells = []
                for run in rle.split("."):
                    cells += [HEX.index(run[0])] * (int(run[1:]) if len(run) > 1 else 1)
                if key[0] == "r":
                    g[int(key[1:])] = cells
                else:
                    for r, c in enumerate(cells, start=4):
                        g[r][int(key[1:])] = c
            g[4][4], g[4][5], g[4][6] = f["i"] % 16, f["i"] // 16 % 16, f["i"] // 256 % 16
            frames.append((f["a"], tuple(tuple(r) for r in g), f["l"], f["s"], f["x"]))
        games[game["game"].split("-")[0]] = frames
    return games


@pytest.fixture(scope="module")
def demo():
    games = _demo_games()
    out = {}
    for name, frames in games.items():
        entries = [f[:4] for f in frames]
        meter = bb.BudgetMeter()
        out[name] = {"entries": entries, "over": [f[4] for f in frames],
                     "readings": [None] + [bb.public(meter.read(entries[:t + 1])) for t in range(1, len(entries))]}
    return out


def test_demo_bar_found_in_nine_of_ten_games_and_silent_on_sb26(demo):
    found = {g: sorted({(r["line"], r["colour"]) for r in d["readings"] if r}) for g, d in demo.items()}
    assert sorted(g for g, lines in found.items() if lines) == [
        "ar25", "ft09", "lp85", "r11l", "re86", "sc25", "tr87", "tu93", "vc33"]
    assert found["sb26"] == []
    assert found["sc25"] == [("columns 62-63", "N")] and found["tu93"] == [("row 63", "M")]
    assert all(len(lines) == 1 for lines in found.values() if lines)  # one bar per game, never a second line


def test_demo_reads_zero_before_each_budget_death_and_names_them(demo):
    sc25, tu93 = demo["sc25"]["readings"], demo["tu93"]["readings"]
    assert [sc25[t]["actions_left"] for t in range(89, 94)] == [4, 3, 2, 1, 0]  # death at action 94
    assert [sc25[t]["actions_left"] for t in range(124, 129)] == [4, 3, 2, 1, 0]  # death at action 129
    assert [tu93[t]["actions_left"] for t in range(102, 106)] == [4, 3, 1, 0]  # death at action 106
    named = {}
    for g in ("sc25", "tu93"):
        d = demo[g]
        named[g] = [t for t, x in enumerate(d["over"]) if x and bb.BudgetMeter().death(d["entries"][:t + 2])]
        deaths = [t for t, x in enumerate(d["over"]) if x]
        assert set(named[g]) <= set(deaths)
    assert named == {"sc25": [94, 129], "tu93": [106]}  # tu93's deaths at 52 and 85 were collisions: not named


# --- the harness, in the bed venv, on the tree the notebook builds --------------------------------------------------


def _has_tree_source() -> bool:
    return (franzen_tree.his_repo_path() / "ARC3-Inference").is_dir()


needs_tree = pytest.mark.skipif(not _has_tree_source(), reason="needs Franzen's repo (scripts/franzen_tree.py)")
needs_bed = pytest.mark.skipif(not BED_PY.exists(), reason="needs the franzen_bed venv (scripts/franzen_bed.py)")


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    base = tmp_path_factory.mktemp("base")
    franzen_tree.notebook_bundle(base, [SANDBOX])
    patched = tmp_path_factory.mktemp("patched")
    logs = franzen_tree.notebook_bundle(patched, [SANDBOX, PATCH])
    assert PATCH.name in logs
    return base / "src" / "ARC3-Inference", patched / "src" / "ARC3-Inference"


@needs_tree
def test_patch_applies_after_his_patch_and_the_sandbox_patch(trees):
    _, patched = trees
    agent = (patched / "inference/agent/tool_agent.py").read_text()
    assert agent.count('_get_env_bool("OURS_BUDGET_METER", False)') == 2
    assert 'startswith("Tool timed out")' in agent  # the earlier patch is still there
    assert (patched / "inference/utils/budget_bar.py").read_text().startswith('"""Budget meter:')
    assert 'runtime_globals["budget"] = state_payload.get("budget")' in (
        patched / "inference/agent/python_tool_sandbox.py").read_text()


CHECK = textwrap.dedent('''
    import json, os, sys, tempfile
    from pathlib import Path
    sys.path.insert(0, sys.argv[1])
    os.environ.update({"ARC3_ACTION_INFO": "1", "EXPOSE_UNDO": "on", "EXPOSE_RESET": "on",
                       "ARC3_PERSISTENT_FUNCTIONS": "1", "ARC3_PERSISTENT_FUNCTIONS_SCOPE": "game",
                       "ARC3_EXPLAIN_GAMEPLAY_CHANGED": "1", "ARC3_NEW_CHANGED_PROMPTS": "1",
                       "LOCAL_ANALYZER_TOOL_TIMEOUT": "20"})
    if sys.argv[2] == "on":
        os.environ["OURS_BUDGET_METER"] = "1"
    from inference.agent import tool_agent as ta
    from inference.agent.runtime_state import Frame, HistoryEntry, RUNTIME_STATE_FILENAME, write_runtime_state

    def grid(bar, obj):
        g = [[5] * 64 for _ in range(64)]
        g[63] = [6] * bar + [0] * (64 - bar)
        g[30][10 + obj % 40] = 9
        return tuple(tuple(r) for r in g)

    def entry(action, bar, step, **result):
        res = {"executed": True, "action_display": action, "gameplay_changed": True, **result} if action else {}
        return HistoryEntry(action=action, frame=Frame(grid=grid(bar, step), step=step, level=1), result=res)

    def summary(n, actions=("RIGHT",), **kw):
        return {"executed_count": len(actions), "executed_actions": list(actions), "gameplay_changed": True,
                "board_changed": True, "level": 1, "action_num": n, **kw}

    def agent():
        return ta.ToolAgent(model="flashnext", base_url="http://127.0.0.1:9/v1", provider="vllm")

    valid = ["UP", "DOWN", "LEFT", "RIGHT", "RESET"]
    h1 = [entry("", 64, 0)] + [entry("RIGHT", 64 - 2 * s, s) for s in range(1, 13)]
    h2 = ([entry("", 64, 0)] + [entry("RIGHT", 64 - 2 * s, s) for s in range(1, 32)]
          + [entry("RIGHT", 0, 32, game_over=True), entry("RESET", 64, 33, automatic=True)])
    h3 = h1 + [entry("RESET", 64, 13, automatic=False), entry("RIGHT", 62, 14), entry("RIGHT", 60, 15)]
    out = {"system": agent()._system_prompt}
    out["p1"] = agent()._build_user_prompt(12, valid_actions=valid, current_frame=h1[-1].frame, history_entries=h1,
                                           previous_step_summary=summary(12))
    out["p2"] = agent()._build_user_prompt(33, valid_actions=valid, current_frame=h2[-1].frame, history_entries=h2,
                                           previous_step_summary=summary(32, game_over=True, fatal_action="RIGHT"))
    out["p3"] = agent()._build_user_prompt(15, valid_actions=valid, current_frame=h3[-1].frame, history_entries=h3,
                                           previous_step_summary=summary(15, ("RESET", "RIGHT", "RIGHT")))
    captured = {}
    real = ta.run_sandboxed_python

    def spy(**kwargs):
        captured["state"] = kwargs["initial_state"]
        return real(**kwargs)

    ta.run_sandboxed_python = spy
    with tempfile.TemporaryDirectory() as d:
        state_path = Path(d) / RUNTIME_STATE_FILENAME
        write_runtime_state(state_path, current_frame=h1[-1].frame, history=h1)
        code = "import json\\ntry:\\n    b = budget\\nexcept NameError:\\n    b = 'unset'\\nprint(json.dumps(b, sort_keys=True))\\n"
        out["tool"] = agent()._run_python_tool(state_path, {"code": code}).content
    out["state"] = json.dumps(captured["state"], sort_keys=True)
    print(json.dumps(out))
''')


def _check(tree: Path, mode: str, tmp_path: Path) -> dict:
    script = tmp_path / f"check_{mode}.py"
    script.write_text(CHECK)
    r = subprocess.run([str(BED_PY), "-I", str(script), str(tree), mode], capture_output=True, text=True,
                       timeout=180, cwd=tmp_path)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@needs_tree
@needs_bed
def test_flag_off_is_byte_identical_and_flag_on_adds_only_the_meter(trees, tmp_path):
    base, patched = trees
    before = _check(base, "off", tmp_path)
    off = _check(patched, "off", tmp_path)
    on = _check(patched, "on", tmp_path)
    assert off == before  # system prompt, three user prompts, the sandbox state and the tool result
    assert _check(base, "on", tmp_path) == before  # the flag alone does nothing without the patch
    assert on["system"] == before["system"]
    line = ("Budget bar (row 63, colour 'M'): about 20 more actions before it is empty (40 cells left, about 2.0 per "
            "action). `budget` in python has these numbers.")
    assert f"Current state: step 13, level 1.\n{line}\nValid actions right now:" in on["p1"]
    assert on["p1"].replace(line + "\n", "", 1) == before["p1"]
    death = ("Budget meter: the bar (row 63, colour 'M') ran out at the fatal action, so this was a budget death.")
    assert death in on["p2"] and on["p2"].replace(death + "\n", "", 1) == before["p2"]
    assert "Budget bar" not in on["p2"]  # the new attempt after the automatic RESET has no drops yet
    # EXPOSE_RESET: a deliberate RESET starts a new attempt; two drops since then are not enough to read
    assert on["p3"] == before["p3"] and "You deliberately reset the current level" in on["p3"]
    state = json.loads(on["state"])
    assert state["budget"]["actions_left"] == 20 and state["budget"]["line"] == "row 63"
    assert {k: v for k, v in state.items() if k != "budget"} == json.loads(before["state"])
    assert "budget" not in json.loads(before["state"])
    printed = json.loads(json.loads(on["tool"])["stdout"])
    assert printed == state["budget"] and json.loads(json.loads(before["tool"])["stdout"]) == "unset"
