"""kaggle/franzen/patches/ours-10-untried.patch (OURS_UNTRIED): the "not yet tried on this level" block (I1) and the
"new colour" line (I2) in the next user prompt; see docs/research/beat-tufa/patch-untried.md.

The module (inference/utils/ours_untried.py, a new file in the patch) is loaded straight from the patch text, so its
unit tests need neither Franzen's repo nor the bed: synthetic boards, explicit clock values. The patch applies alone,
on exp-083's stack (the sandbox fix) and on exp-084's (sandbox, 02, 04, 03b, 05, 08b), and adds lines only. In the bed
venv the real ToolAgent, driven through _build_user_prompt and _run_python_tool against a small fake game with a fake
clock, builds byte-identical prompts and tool results with the flag off, and with it on puts the lines just before
the tool rules, timed from the action() call that reached the level.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import franzen_bed  # noqa: E402
import franzen_tree  # noqa: E402

PATCHES = ROOT / "kaggle" / "franzen" / "patches"
PATCH = PATCHES / "ours-10-untried.patch"
SANDBOX = PATCHES / "ours-sandbox-timeout-keeps-work.patch"
EXP084 = [PATCHES / name for name in (
    "ours-sandbox-timeout-keeps-work.patch", "ours-02-budget-meter.patch", "ours-04-search-helper.patch",
    "ours-03b-win-ledger-on-02-04.patch", "ours-05-level-mem.patch", "ours-08b-perception-on-01-02-04-03b-05.patch")]
STACKS = {"083": [SANDBOX], "084": EXP084}
BED_PY = Path.home() / ".cache" / "arc3-franzen-bed" / "venv" / "bin" / "python"
NEW_FILE = "ARC3-Inference/inference/utils/ours_untried.py"
SENTENCE = "Each is a one-action test of whether it is interactive."
BOTH = frozenset({"i1", "i2"})

needs_tree = pytest.mark.skipif(not (franzen_tree.his_repo_path() / "ARC3-Inference").is_dir(),
                                reason="needs Franzen's repo (scripts/franzen_tree.py)")
needs_bed = pytest.mark.skipif(not BED_PY.exists(), reason="needs the franzen_bed venv (scripts/franzen_bed.py)")


def _module_source() -> str:
    """The new module exactly as the patch adds it."""
    section = PATCH.read_text().split(f"+++ b/{NEW_FILE}\n", 1)[1].split("\ndiff --git ", 1)[0]
    body = section.split("\n", 1)[1]  # after the @@ line
    return "\n".join(line[1:] for line in body.splitlines() if line.startswith("+")) + "\n"


def _load() -> types.ModuleType:
    module = types.ModuleType("ours_untried")
    exec(compile(_module_source(), NEW_FILE, "exec"), module.__dict__)
    return module


u = _load()


# --- synthetic boards -------------------------------------------------------------------------------------------------


def rect(r0, c0, r1, c1):
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def board(*paint, size=64, bg=0):
    """A size x size board of colour bg with (colour, cells) painted in order."""
    g = [[bg] * size for _ in range(size)]
    for colour, cells in paint:
        for r, c in cells:
            g[r][c] = colour
    return tuple(tuple(row) for row in g)


def entry(action, grid, step, level=1, **result):
    """A harness HistoryEntry as the tracker reads it."""
    return types.SimpleNamespace(action=action, frame=types.SimpleNamespace(grid=grid, step=step, level=level),
                                 result=result)


L_SHAPE = [(2, 2), (3, 2), (4, 2), (4, 3)]
L_TURNED = [(10, 10), (10, 11), (10, 12), (11, 10)]  # the same L, turned 90 degrees
J_SHAPE = [(20, 3), (21, 3), (22, 3), (22, 2)]  # its mirror image: not a rotation


# --- the flag ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("raw, want", [(None, set()), ("", set()), ("0", set()), ("off", set()), ("i3", set()),
                                       ("1", {"i1", "i2"}), ("on", {"i1", "i2"}), ("TRUE", {"i1", "i2"}),
                                       ("yes", {"i1", "i2"}), ("i1", {"i1"}), (" I2 ", {"i2"})])
def test_the_flag_is_read_at_call_time(monkeypatch, raw, want):
    if raw is None:
        monkeypatch.delenv("OURS_UNTRIED", raising=False)
    else:
        monkeypatch.setenv("OURS_UNTRIED", raw)
    assert u.mode() == frozenset(want)


# --- objects and kinds ------------------------------------------------------------------------------------------------


def test_a_kind_is_a_colour_and_a_shape_up_to_rotation_and_copies_collapse():
    g = board((8, L_SHAPE), (8, L_TURNED), (8, J_SHAPE), (9, [(r, c + 18) for r, c in L_SHAPE]),
              (4, rect(26, 5, 26, 8)), (4, rect(14, 20, 17, 20)), size=32)
    kinds = u.board_kinds(g)
    by_colour = sorted((k[0], len(objs)) for k, objs in kinds.items())
    assert by_colour == [(4, 2), (8, 1), (8, 2), (9, 1)]  # the background (all 32 columns) is not a kind
    l_kind = u.kind_key(next(o for o in u.components(g) if o.box == (2, 2, 4, 3)))
    assert [o.box for o in kinds[l_kind]] == [(2, 2, 4, 3), (10, 10, 11, 12)]  # reading order of the boxes
    assert l_kind[1] != u.kind_key(next(o for o in u.components(g) if o.box == (20, 2, 22, 3)))[1]  # J is not L
    assert kinds[(9, l_kind[1])][0].text() == "b r2-4 c20-21"
    assert [o.text() for o in kinds[next(k for k in kinds if k[0] == 4)]] == ["c r14-17 c20", "c r26 c5-8"]


def test_components_cope_with_ragged_and_empty_grids():
    assert u.components(()) == [] and u.components(None) == []
    [one] = u.components(((1, 1), (1,)))
    assert one.colour == 1 and one.cells == {(0, 0), (0, 1), (1, 0)} and one.box == (0, 0, 1, 1)


def test_areas_panels_and_edge_hud_bars_are_not_objects():
    g = board((3, rect(0, 50, 49, 55)),  # a panel: 50 of 64 rows
              (2, rect(30, 2, 47, 19)),  # 324 cells
              (6, rect(0, 30, 0, 45)), (6, rect(62, 10, 63, 40)),  # HUD bars on the top and bottom edges
              (7, rect(5, 1, 20, 1)), (7, rect(5, 63, 20, 63)),  # and on the left and right
              (8, rect(25, 25, 25, 40)),  # a thin bar mid-board: kept
              (9, rect(0, 20, 3, 23)),  # four rows thick, touching the top edge: kept
              (10, rect(40, 0, 41, 2)))  # three columns wide at the left edge: kept
    kept = u.salient(u.components(g), u._shape(g))
    assert sorted((o.colour, o.box) for o in kept) == [(8, (25, 25, 25, 40)), (9, (0, 20, 3, 23)),
                                                       (10, (40, 0, 41, 2))]


def test_a_click_hits_the_object_under_it_else_the_nearest_small_box():
    objs = u.salient(u.components(board((9, rect(9, 9, 20, 20)), (8, rect(10, 10, 11, 11)))), (64, 64))
    small = next(o for o in objs if o.colour == 8)
    assert u.click_target(objs, 10, 10) is small
    assert u.click_target(objs, 12, 12).colour == 9  # on the big square's own cell
    objs = [small]
    assert u.click_target(objs, 12, 12) is small  # one cell off its box
    assert u.click_target(objs, 13, 13) is None
    assert u.mouse_cell("MOUSE(row=3, col=12)") == (3, 12) and u.mouse_cell("UP") is None
    assert [u.action_type(a) for a in ("", "RESET", "up", "ACTION6", "MOUSE(row=1, col=2)", "UNDO")] == [
        "", "", "UP", "MOUSE", "MOUSE", "UNDO"]


# --- the I1 block -----------------------------------------------------------------------------------------------------


EIGHT_KINDS = board(  # all at least 3 cells from the edges, where thin things count as HUD
    *[(8, rect(2, c, 3, c + 1)) for c in (6, 10, 14)],  # R 2x2, three copies (on the previous level)
    (9, [(6, 6)]),  # b 1x1 (on the previous level)
    (10, rect(10, 10, 12, 12)),  # S 3x3
    *[(11, rect(16, c, 17, c + 1)) for c in (6, 10)],  # Y 2x2, two copies
    (12, rect(20, 6, 20, 7)),  # O 1x2
    (13, [(24, 6)]),  # r 1x1
    (14, rect(28, 6, 28, 8)),  # N 1x3 (on the previous level, turned)
    *[(15, rect(r, c, r + 1, c + 2)) for r in (32, 36) for c in (6, 12)])  # p 2x3, four copies
PREVIOUS = board((8, rect(40, 40, 41, 41)), (9, [(50, 50)]), (14, rect(44, 30, 46, 30)))


def test_i1_puts_kinds_new_on_this_level_first_then_fewest_copies_then_larger_and_names_six():
    kinds = u.board_kinds(EIGHT_KINDS)
    assert len(kinds) == 8
    text = u.i1_block(unused_actions=["UP", "SPACE"], kinds=kinds, prev_kinds=set(u.board_kinds(PREVIOUS)))
    assert text == ("Not yet tried on this level: UP, SPACE never used; no click has hit S r10-12 c10-12, O r20 c6-7, "
                    "r r24 c6, Y r16-17 c6-7 x2, p r32-33 c6-8 x4, N r28 c6-8; 2 more kinds. " + SENTENCE)
    first_level = u.i1_block(unused_actions=[], kinds=kinds, prev_kinds=None)
    assert first_level == ("Not yet tried on this level: no click has hit S r10-12 c10-12, N r28 c6-8, O r20 c6-7, "
                           "b r6 c6, r r24 c6, Y r16-17 c6-7 x2; 2 more kinds. " + SENTENCE)


def test_i1_is_empty_with_nothing_to_name_and_lists_actions_alone_without_kinds():
    assert u.i1_block(unused_actions=[], kinds={}) == ""
    assert u.i1_block(unused_actions=["MOUSE"], kinds={}) == "Not yet tried on this level: MOUSE never used. " + SENTENCE


# --- the I2 line ------------------------------------------------------------------------------------------------------


BUTTON, GATE, OTHER = rect(10, 10, 11, 11), rect(30, 20, 30, 23), rect(40, 40, 42, 42)
START = board((8, BUTTON), (12, GATE), (9, OTHER))


def i2_lines(*entries, modes=BOTH):
    t = u.Tracker()
    return [line for line in t.lines([entry("", START, 0), *entries], ["UP", "MOUSE"], 0.0, modes)
            if not line.startswith("Not yet tried")]


def test_i2_names_an_object_other_than_the_click_target_that_turns_a_colour_new_to_the_game():
    after = board((8, BUTTON), (14, GATE), (9, OTHER))
    assert i2_lines(entry("MOUSE(row=10, col=10)", after, 1)) == [
        "After MOUSE(row=10, col=10), O r30 c20-23 turned N, a colour new to this game; no click has hit it since."]
    assert i2_lines(entry("MOUSE(row=10, col=10)", after, 1), modes=frozenset({"i1"})) == []
    appeared = board((8, BUTTON), (12, GATE), (9, OTHER), (13, rect(50, 50, 51, 51)))
    assert i2_lines(entry("SPACE", appeared, 1)) == [
        "After SPACE, r r50-51 c50-51 appeared, a colour new to this game; no click has hit it since."]
    four = board(*[(12, [(r, c)]) for r in (5, 20) for c in (5, 20)])
    t = u.Tracker()
    t.lines([entry("", four, 0)], ["UP"], 0.0, BOTH)
    [line] = t.lines([entry("", four, 0), entry("UP", board(*[(14, [(r, c)]) for r in (5, 20) for c in (5, 20)]), 1)],
                     ["UP"], 1.0, BOTH)
    assert line == ("After UP, O r5 c5 turned N; O r5 c20 turned N; O r20 c5 turned N (and 1 more), a colour new to "
                    "this game; no click has hit them since.")


def test_i2_is_silent_for_the_click_target_level_switches_game_overs_and_colours_seen_before():
    gate_new = board((8, BUTTON), (14, GATE), (9, OTHER))
    t = u.Tracker()
    assert t.lines([entry("", START, 0), entry("MOUSE(row=30, col=21)", gate_new, 1)], ["UP"], 0.0, BOTH) == []
    assert u.kind_key(u.Obj(14, frozenset(GATE), (30, 20, 30, 23))) in t.levels[1].clicked  # its new look too
    level_two = board((8, BUTTON), (13, GATE))
    assert i2_lines(entry("UP", level_two, 1, level=2, level_completed=True),
                    entry("UP", board((13, BUTTON), (13, GATE)), 2, level=2)) == []  # seen at the switch
    assert i2_lines(entry("UP", gate_new, 1, game_over=True)) == []
    assert i2_lines(entry("UP", board((8, BUTTON), (9, GATE), (9, OTHER)), 1)) == []  # b is on the board already
    # an object clicked after its event and before the prompt is dropped
    assert i2_lines(entry("SPACE", gate_new, 1), entry("MOUSE(row=30, col=21)", gate_new, 2)) == []


# --- timing -----------------------------------------------------------------------------------------------------------


def test_i1_fires_at_ten_minutes_then_every_ten_and_after_a_game_over_past_the_mark():
    t, valid, h = u.Tracker(), ["UP", "MOUSE"], [entry("", START, 0)]
    assert t.lines(h, valid, 100.0, BOTH) == []  # the first level starts when the harness first sees it
    assert t.lines(h, valid, 699.0, BOTH) == []
    [line] = t.lines(h, valid, 700.0, BOTH)
    assert line == ("Not yet tried on this level: UP, MOUSE never used; no click has hit b r40-42 c40-42, R r10-11 "
                    "c10-11, O r30 c20-23. " + SENTENCE)  # one copy each: the larger first
    assert t.lines(h, valid, 1299.0, BOTH) == []
    assert len(t.lines(h, valid, 1300.0, BOTH)) == 1
    h += [entry("UP", START, 1, game_over=True), entry("RESET", START, 2, automatic=True)]
    [line] = t.lines(h, valid, 1400.0, BOTH)  # 100 s after the last one: a GAME OVER past the mark fires at once
    assert line.startswith("Not yet tried on this level: MOUSE never used; no click")  # the fatal UP counts as used
    assert t.lines(h, valid, 1500.0, BOTH) == []
    assert [f["reason"] for f in t.fired] == ["stall", "repeat", "game over"]
    assert t.lines(h, valid, 2000.0, frozenset({"i2"})) == []


def test_a_game_over_before_ten_minutes_does_not_fire_and_is_not_kept():
    t, h = u.Tracker(), [entry("", START, 0)]
    t.lines(h, ["UP"], 0.0, BOTH)
    h += [entry("UP", START, 1, game_over=True), entry("RESET", START, 2, automatic=True)]
    assert t.lines(h, ["UP"], 300.0, BOTH) == []
    assert t.lines(h, ["UP", "LEFT"], 600.0, BOTH) == ["Not yet tried on this level: LEFT never used. " + SENTENCE]
    assert t.fired[-1]["reason"] == "stall"


def test_a_level_starts_when_the_action_call_that_reached_it_returned():
    t, valid, h = u.Tracker(), ["UP"], [entry("", START, 0)]
    t.lines(h, valid, 0.0, BOTH)
    t.stamp(3, 50.0)  # the action() call that ran steps 1-3 returned at t=50
    t.stamp(3, 60.0)
    t.stamp(2, 70.0)  # not a later call: ignored
    assert t.stamps == [(3, 50.0)]
    h += [entry("LEFT", START, 1), entry("LEFT", START, 2), entry("LEFT", board((9, OTHER)), 3, level=2)]
    assert t.lines(h, valid, 649.0, BOTH) == []  # seen first at 649, but the level began at 50
    assert t.levels[2].start == 50.0 and len(t.lines(h, valid, 650.0, BOTH)) == 1
    t2, h2 = u.Tracker(), [entry("", START, 0)]
    t2.lines(h2, valid, 0.0, BOTH)
    t2.lines([*h2, entry("UP", board((9, OTHER)), 1, level=2)], valid, 400.0, BOTH)  # not stamped: the prompt's time
    assert t2.levels[2].start == 400.0


def test_unused_actions_never_list_reset_or_undo_and_kinds_need_mouse():
    t, h = u.Tracker(), [entry("", START, 0), entry("UP", START, 1), entry("UNDO", START, 2)]
    t.lines(h, ["ACTION1", "ACTION2", "ACTION5", "ACTION7", "RESET"], 0.0, BOTH)
    assert t.lines(h, ["ACTION1", "ACTION2", "ACTION5", "ACTION7", "RESET"], 600.0, BOTH) == [
        "Not yet tried on this level: DOWN, SPACE never used. " + SENTENCE]
    clicked = u.Tracker()
    hc = [entry("", START, 0), entry("MOUSE(row=10, col=10)", START, 1), entry("MOUSE(row=30, col=22)", START, 2)]
    clicked.lines(hc, ["MOUSE"], 0.0, BOTH)
    assert clicked.lines(hc, ["MOUSE"], 600.0, BOTH) == [
        "Not yet tried on this level: no click has hit b r40-42 c40-42. " + SENTENCE]


# --- harness glue -----------------------------------------------------------------------------------------------------


def test_the_glue_does_nothing_with_the_flag_off(monkeypatch):
    monkeypatch.delenv("OURS_UNTRIED", raising=False)
    agent = types.SimpleNamespace(_session_runtime_dir="/runs/g1")
    u.stamp(agent, {"executed": True, "action_num": 3})
    assert u.prompt_lines(agent, [entry("", START, 0)], ["UP"]) == []
    assert not hasattr(agent, "_ours_untried")


def test_the_glue_keeps_one_tracker_per_session_and_turns_itself_off_on_an_error(monkeypatch, caplog):
    monkeypatch.setenv("OURS_UNTRIED", "1")
    now = [0.0]
    monkeypatch.setattr(u, "_clock", lambda: now[0])
    agent, h = types.SimpleNamespace(_session_runtime_dir="/runs/g1"), [entry("", START, 0)]
    assert u.prompt_lines(agent, h, ["UP"]) == []
    now[0] = 30.0
    u.stamp(agent, {"executed": True, "action_num": 1})
    u.stamp(agent, {"executed": False, "action_num": 2})  # a refused call ran nothing
    u.stamp(agent, None)
    assert agent._ours_untried[1].stamps == [(1, 30.0)]
    now[0] = 600.0
    assert u.prompt_lines(agent, h, ["UP"]) == ["Not yet tried on this level: UP never used. " + SENTENCE]
    agent._session_runtime_dir = "/runs/g2"  # the next game
    assert u.prompt_lines(agent, h, ["UP"]) == [] and agent._ours_untried[0] == "/runs/g2"
    with caplog.at_level(logging.WARNING):
        u.stamp(agent, {"executed": True, "action_num": "x"})  # logged, never raised
        real = u.Tracker.lines
        monkeypatch.setattr(u.Tracker, "lines", lambda *a, **k: 1 / 0)
        assert u.prompt_lines(agent, h, ["UP"]) == []
        monkeypatch.setattr(u.Tracker, "lines", real)
    assert agent._ours_untried[1].error == "internal error (ZeroDivisionError)"
    now[0] = 5000.0
    assert u.prompt_lines(agent, h, ["UP"]) == []  # off for the rest of this game
    assert "stamp failed" in caplog.text and "lines off for this game" in caplog.text


# --- the patch on the notebook's trees --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    out = {}
    for name, patches in (("base", []), ("ours", [PATCH]), *[(f"base{s}", p) for s, p in STACKS.items()],
                          *[(s, [*p, PATCH]) for s, p in STACKS.items()]):
        dest = tmp_path_factory.mktemp(name)
        logs = franzen_tree.notebook_bundle(dest, patches)
        assert [p.name for p in patches] == [k for k in logs if k != "his"]
        out[name] = dest / "src" / "ARC3-Inference"
    return out


@needs_tree
@pytest.mark.parametrize("stack", ["", "083", "084"])
def test_the_patch_applies_alone_and_on_both_stacks_and_only_adds_lines(trees, stack):
    base, ours = trees[f"base{stack}"], trees[stack or "ours"]
    before = (base / "inference/agent/tool_agent.py").read_text().splitlines()
    after = (ours / "inference/agent/tool_agent.py").read_text().splitlines()
    assert [line for line in before if line not in set(after)] == []  # nothing removed or changed
    added = [line for line in after if line not in set(before)]
    plus = {line[1:] for line in PATCH.read_text().splitlines() if line.startswith("+") and not line.startswith("+++")}
    assert len(added) == 6 and set(added) <= plus
    assert (ours / "inference/utils/ours_untried.py").read_text() == _module_source()
    text = "\n".join(after)
    hook = text.index("_ours_untried_lines(self, history_entries")
    assert text.index("self._death_ledger_lines(", hook - 2000) < hook < text.index("Only tool: `python`", hook)
    stamp = text.index("_ours_untried_stamp(self, raw_payload)")
    assert 0 < stamp - text.index("raw_payload = self._step_env_callback(step_arguments)") < 600


# --- the real ToolAgent in the bed venv -------------------------------------------------------------------------------

DRIVER = textwrap.dedent('''
    import json, sys
    from pathlib import Path
    import tempfile
    sys.path.insert(0, sys.argv[1])
    from inference.agent import tool_agent as ta
    from inference.agent.runtime_state import (RUNTIME_STATE_FILENAME, Frame, HistoryEntry, load_runtime_state,
                                               write_runtime_state)
    try:
        from inference.utils import ours_untried
    except ImportError:  # the tree without the patch
        ours_untried = None
    NOW = [1000.0]
    if ours_untried is not None:
        ours_untried._clock = lambda: NOW[0]
    VALID = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6", "ACTION7"]

    def board(level, l_colour, hud, dead):
        g = [[5] * 24 for _ in range(24)]
        for c in range(hud):
            g[23][c] = 6  # a HUD bar on the bottom edge
        if level == 1:
            for c0 in (2, 8):
                for r, c in ((2, c0), (2, c0 + 1), (3, c0), (3, c0 + 1)):
                    g[r][c] = 9
            for c in range(4, 8):
                g[10][c] = 8
            for r, c in ((15, 15), (16, 15), (17, 15), (17, 16)):
                g[r][c] = l_colour
        else:
            for r, c in ((2, 2), (2, 3), (3, 2), (3, 3)):
                g[r][c] = 9
            for r in range(5, 9):
                g[r][20] = 8
            for r in range(12, 15):
                for c in range(12, 15):
                    g[r][c] = 11
            if dead:
                g[13][13] = 13  # a colour new to the game, on the game-over frame only
        return tuple(tuple(row) for row in g)

    class Game:
        """MOUSE on the red bar turns the orange L a colour new to the game; RIGHT shortens the HUD bar; DOWN wins
        level 1; LEFT on level 2 is a GAME OVER, after which the solver adds the automatic RESET between turns."""
        def __init__(self, path):
            self.path, self.step, self.level, self.l_colour, self.hud = path, 0, 1, 12, 12
            self.history = [HistoryEntry(action="", frame=self.frame())]
            self.write()
        def frame(self, dead=False):
            return Frame(grid=board(self.level, self.l_colour, self.hud, dead), step=self.step, level=self.level)
        def write(self):
            write_runtime_state(self.path, current_frame=self.history[-1].frame, history=self.history)
        def payload(self, name, display, data, changed, won=False, over=False, automatic=False, index=1, size=1):
            return {"executed": True, "action_num": self.step, "level": self.level, "score": self.level - 1,
                    "state": "GAME_OVER" if over else "NOT_FINISHED",
                    "valid_actions": ["UP", "DOWN", "LEFT", "RIGHT", "MOUSE", "UNDO"], "board_changed": changed,
                    "gameplay_changed": changed, "no_op": not changed, "done": False, "level_completed": won,
                    "game_over": over, "run_complete": False, "action_name": name, "action_data": data,
                    "action_display": display, "automatic": automatic, "batch_index": index, "batch_size": size}
        def __call__(self, arguments):
            if "actions" not in arguments:
                return {}
            last, actions = None, arguments["actions"]
            for index, item in enumerate(actions, start=1):
                name = item["action"].upper()
                self.step += 1
                data, display = {}, name
                if name == "MOUSE":
                    data = {"row": item["row"], "col": item["col"]}
                    display = f"MOUSE(row={item['row']}, col={item['col']})"
                    if item["row"] == 10 and 4 <= item["col"] <= 7:
                        self.l_colour = 14
                if name == "RIGHT":
                    self.hud -= 1
                won, over = name == "DOWN" and self.level == 1, name == "LEFT" and self.level == 2
                if won:
                    self.level = 2
                last = self.payload(name, display, data, True, won, over, index=index, size=len(actions))
                self.history.append(HistoryEntry(action=display, frame=self.frame(dead=over),
                                                 result={k: v for k, v in last.items() if k not in ("batch_index",
                                                                                                   "batch_size")}))
                self.write()
                if won or over:
                    break
            return {**last, "requested_count": len(actions), "executed_count": index}
        def auto_reset(self):
            self.step += 1
            result = self.payload("RESET", "RESET", {}, True, automatic=True)
            self.history.append(HistoryEntry(action="RESET", frame=self.frame(), result=result))
            self.write()

    game = Game(Path(tempfile.mkdtemp()) / f"g_{RUNTIME_STATE_FILENAME}")
    agent = ta.ToolAgent(model="mock", base_url="http://127.0.0.1:9/v1")
    out = {"system": agent._system_prompt, "prompts": [], "tools": []}

    def prompt(at):
        NOW[0] = at
        agent._ensure_session(game.path)  # as analyze() does
        agent._step_env_callback = game
        agent._current_valid_actions = ta._normalize_valid_actions(VALID)
        frame, history = load_runtime_state(game.path)
        out["prompts"].append(agent._build_user_prompt(game.step, valid_actions=VALID, current_frame=frame,
                                                       history_entries=history,
                                                       previous_step_summary=agent._last_step_summary))

    def tool(at, code):
        NOW[0] = at
        out["tools"].append(agent._run_python_tool(game.path, {"code": code}).content)

    prompt(1000.0)
    tool(1030.0, 'action({"action": "MOUSE", "row": 10, "col": 5})')
    prompt(1040.0)
    tool(1100.0, 'action("RIGHT")')
    prompt(1600.0)
    tool(1610.0, 'action("DOWN")')
    prompt(2205.0)
    prompt(2210.0)
    tool(2300.0, 'action("LEFT")')
    game.auto_reset()
    prompt(2310.0)
    state = getattr(agent, "_ours_untried", None)
    out["fired"] = None if state is None else [[f["line"], f.get("reason")] for f in state[1].fired]
    out["stamps"] = None if state is None else state[1].stamps
    print(json.dumps(out))
''')

I2_P2 = "After MOUSE(row=10, col=5), O r15-17 c15-16 turned N, a colour new to this game; no click has hit it since."
I1_P3 = ("Not yet tried on this level: UP, DOWN, LEFT never used; no click has hit N r15-17 c15-16, b r2-3 c2-3 x2. "
         + SENTENCE)
I1_P5 = ("Not yet tried on this level: UP, DOWN, LEFT, RIGHT, MOUSE never used; no click has hit Y r12-14 c12-14, "
         "b r2-3 c2-3, R r5-8 c20. " + SENTENCE)
I1_P6 = ("Not yet tried on this level: UP, DOWN, RIGHT, MOUSE never used; no click has hit Y r12-14 c12-14, b r2-3 c2-3, "
         "R r5-8 c20. " + SENTENCE)
ANCHOR = "Only tool: `python`."


def _clean_env() -> dict[str, str]:
    """This process's environment without anything that steers the harness."""
    prefixes = ("ARC3_", "LOCAL_ANALYZER_", "OURS_", "OPENAI_", "INFERENCE_", "MULTIMODAL_", "EXPOSE_")
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith(prefixes)}


@pytest.fixture(scope="module")
def bed(trees, tmp_path_factory):
    """The scenario under the notebook's harness environment, per (stack, tree, OURS_UNTRIED)."""
    work = tmp_path_factory.mktemp("driver")
    script = work / "drive.py"
    script.write_text(DRIVER)
    notebook, _ = franzen_bed.notebook_env(franzen_bed._cell(franzen_tree.notebook_cells(), "setup_env = {"))
    out = {}
    for stack, tree, flag in (("083", "base", None), ("083", "base", "1"), ("083", "patched", None),
                              ("083", "patched", "0"), ("083", "patched", "1"), ("083", "patched", "i1"),
                              ("083", "patched", "i2"), ("084", "base", None), ("084", "patched", None),
                              ("084", "patched", "1")):
        src = trees[f"base{stack}" if tree == "base" else stack]
        env = {**_clean_env(), **notebook, **({} if flag is None else {"OURS_UNTRIED": flag})}
        r = subprocess.run([str(BED_PY), "-I", str(script), str(src)], capture_output=True, text=True, env=env,
                           cwd=work, timeout=300, check=False)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
        out[stack, tree, flag] = json.loads(r.stdout.strip().splitlines()[-1])
    return out


def _with(prompt: str, *lines: str) -> str:
    assert prompt.count(ANCHOR) == 1
    return prompt.replace(ANCHOR, "\n".join(lines) + "\n" + ANCHOR, 1) if lines else prompt


@needs_tree
@needs_bed
@pytest.mark.parametrize("stack", ["083", "084"])
def test_flag_off_the_real_agent_is_byte_identical_to_the_tree_without_the_patch(bed, stack):
    base = bed[stack, "base", None]
    assert {k: v for k, v in bed[stack, "patched", None].items() if k not in ("fired", "stamps")} == {
        k: v for k, v in base.items() if k not in ("fired", "stamps")}
    assert bed[stack, "patched", None]["fired"] is None  # nothing kept on the agent
    if stack == "083":
        assert bed[stack, "patched", "0"] == bed[stack, "patched", None]
        assert bed[stack, "base", "1"] == base  # the flag alone does nothing without the patch
    assert len(base["prompts"]) == 6 and all("Not yet tried" not in p and "a colour new to this game" not in p
                                             for p in base["prompts"])


@needs_tree
@needs_bed
@pytest.mark.parametrize("stack", ["083", "084"])
def test_flag_on_the_lines_reach_the_next_prompt_just_before_the_tool_rules(bed, stack):
    base, on = bed[stack, "base", None], bed[stack, "patched", "1"]
    assert on["system"] == base["system"] and on["tools"] == base["tools"]  # actions run exactly as before
    want = [[], [I2_P2], [I1_P3], [], [I1_P5], [I1_P6]]
    for got, before, lines in zip(on["prompts"], base["prompts"], want, strict=True):
        assert got == _with(before, *lines)
    # level 2 began when the DOWN call returned (t=1610), not at the prompt that first saw it (t=2205)
    assert on["stamps"] == [[1, 1030.0], [2, 1100.0], [3, 1610.0], [4, 2300.0]]
    assert on["fired"] == [["i2", None], ["i1", "stall"], ["i1", "stall"], ["i1", "game over"]]
    assert "turned" not in on["prompts"][5]  # the game-over frame's new colour is not an I2


@needs_tree
@needs_bed
def test_i1_and_i2_each_alone(bed):
    base = bed["083", "base", None]["prompts"]
    i1, i2 = bed["083", "patched", "i1"]["prompts"], bed["083", "patched", "i2"]["prompts"]
    assert i1 == [base[0], base[1], _with(base[2], I1_P3), base[3], _with(base[4], I1_P5), _with(base[5], I1_P6)]
    assert i2 == [base[0], _with(base[1], I2_P2), *base[2:]]
