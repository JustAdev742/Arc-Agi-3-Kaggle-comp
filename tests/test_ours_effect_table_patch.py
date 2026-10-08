"""kaggle/franzen/patches/ours-06-effect-table.patch (OURS_EFFECT_TABLE): an exact per-level action-effect table in
Franzen's harness, exposed in the sandbox as `effects()`.

Unit tests of inference/utils/ours_effect_table.py (loaded from the patched tree) on synthetic frames and on dc22's
recorded actions replayed through the local engine; then, in the bed venv, the real ToolAgent and sandbox: with the
flag off the system prompt and every tool result are byte-identical to the tree without the patch, and with it on
`effects()` works inside snippets and in retained functions, and the note appears only with OURS_EFFECT_TABLE=note.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import franzen_bed  # noqa: E402
import franzen_tree  # noqa: E402

PATCHES = ROOT / "kaggle" / "franzen" / "patches"
OURS01 = PATCHES / "ours-sandbox-timeout-keeps-work.patch"
PATCH = PATCHES / "ours-06-effect-table.patch"
BED_PY = Path.home() / ".cache" / "arc3-franzen-bed" / "venv" / "bin" / "python"
ENV_DIR = ROOT / "environment_files"
DC22_FIRST = [("ACTION3", {}), ("ACTION3", {}), ("ACTION1", {}), ("ACTION6", {"x": 24, "y": 21}), ("ACTION1", {}),
              ("ACTION4", {}), ("ACTION4", {}), ("ACTION6", {"x": 12, "y": 38}), ("ACTION6", {"x": 12, "y": 42}),
              ("ACTION6", {"x": 48, "y": 19})]  # exp-073b's first 10 dc22 actions; the 10th turns the red bar

needs_tree = pytest.mark.skipif(not (franzen_tree.his_repo_path() / "ARC3-Inference").is_dir(),
                                reason="needs Franzen's repo (scripts/franzen_tree.py)")
needs_bed = pytest.mark.skipif(not BED_PY.exists(), reason="needs the franzen_bed venv (scripts/franzen_bed.py)")
needs_games = pytest.mark.skipif(not (ENV_DIR / "dc22").exists(), reason="needs environment_files/")


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    """(his patch + ours-01, his patch + ours-01 + ours-06), as the notebook's cell 4 builds them."""
    out = []
    for name, patches in (("base", [OURS01]), ("ours06", [OURS01, PATCH])):
        dest = tmp_path_factory.mktemp(name)
        logs = franzen_tree.notebook_bundle(dest, patches)
        assert all(p.name in logs for p in patches)
        out.append(dest / "src" / "ARC3-Inference")
    return out


@pytest.fixture(scope="module")
def et(trees):
    path = trees[1] / "inference" / "utils" / "ours_effect_table.py"
    spec = importlib.util.spec_from_file_location("ours_effect_table_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- synthetic frames ----------------------------------------------------------------------------------------------


def blank(color: int = 2) -> list:
    g = [[color] * 64 for _ in range(64)]
    for c in range(64):
        g[63][c] = 0  # a bar in the bottom edge band
    return g


def paint(g: list, cells, color: int) -> list:
    for r, c in cells:
        g[r][c] = color
    return g


def block(r: int, c: int, h: int = 2, w: int = 2) -> list:
    return [(r + i, c + j) for i in range(h) for j in range(w)]


def history(frames: list, actions: list, results: list | None = None, levels: list | None = None) -> list:
    hist = [{"action": "", "frame": {"grid": frames[0], "step": 0, "level": (levels or [1])[0]}, "result": {}}]
    for i, action in enumerate(actions, 1):
        hist.append({"action": action, "frame": {"grid": frames[i], "step": i, "level": (levels or [1] * len(frames))[i]},
                     "result": (results or [{}] * len(actions))[i - 1]})
    return hist


def table(et, hist: list, level: int = 1) -> str:
    ledger = et.EffectLedger(border=4)
    for k in range(1, len(hist) + 1):  # one update per action, as the harness serializes state
        ledger.update(hist[:k])
    return ledger.payload()["tables"][str(level)]


def line(text: str, start: str) -> str:
    hits = [x for x in text.splitlines() if x.startswith(start)]
    assert len(hits) == 1, text
    return hits[0]


@needs_tree
def test_patch_applies_after_ours01_and_hooks_are_flag_gated(trees):
    agent = (trees[1] / "inference/agent/tool_agent.py").read_text()
    sandbox = (trees[1] / "inference/agent/python_tool_sandbox.py").read_text()
    assert agent.count("ours_effect_table import") == 3 and 'startswith("Tool timed out")' in agent
    assert 'runtime_globals["effects"] = _ours_effects_helper' in sandbox and '        "time",\n' in sandbox


@needs_tree
def test_moves_no_change_and_edge_band(et):
    f0 = paint(blank(), block(20, 20), 9)
    f1 = paint(blank(), block(20, 22), 9)
    f1[63][0] = 2  # the bar ticks
    f2 = [row[:] for row in f1]
    f2[63][1] = 2  # UP: only the bar changes
    f3 = paint(blank(), block(20, 24), 9)
    f3[63][0] = f3[63][1] = 2
    t = table(et, history([f0, f1, f2, f3], ["RIGHT", "UP", "RIGHT"]))
    assert t.startswith("Level 1: 3 actions. Board = frame without the 4-cell edge band")
    assert line(t, "UP ") == "UP x1: no change x1"
    assert line(t, "RIGHT ") == "RIGHT x2: moved b 4px (+0,+2) x2 [last step 3: (20,22)->(20,24)]"
    assert t.endswith("Edge band changed on 2 of 3 actions.")


@needs_tree
def test_recolor_rotate_appear_disappear(et):
    ell_a = [(10, 10), (11, 10), (11, 11)]
    ell_b = [(10, 10), (10, 11), (11, 10)]  # the same L turned 90 degrees clockwise
    f0 = paint(paint(paint(blank(), [(30, 30), (30, 31), (30, 32)], 9), ell_a, 8), [(45, 10), (45, 11)], 14)
    f1 = paint(paint(blank(), [(30, 30), (30, 31), (30, 32)], 12), ell_b, 8)
    f1[50][50] = 11
    t = line(table(et, history([f0, f1], ["SPACE"])), "SPACE ")
    for part in ("recolored b->O 3px x1 [step 1: at (30,30)]", "rotated R 3px 90 (+0,+0) x1 [step 1: (10,10)->(10,10)]",
                 "disappeared N 2px x1 [step 1: at (45,10)]", "appeared Y 1px x1 [step 1: at (50,50)]"):
        assert part in t, t


@needs_tree
def test_group_moves_and_ambiguous_pairs(et):
    f0 = paint(paint(blank(), [(20, 20), (20, 30)], 9), [(40, 20), (40, 30)], 8)
    f1 = paint(paint(blank(), [(21, 20), (21, 30)], 9), [(40, 21), (39, 30)], 8)
    t = line(table(et, history([f0, f1], ["DOWN"])), "DOWN ")
    assert "moved 2x b 1px (+1,+0) x1 [step 1]" in t
    assert "ambiguous R 1px objects 2->2 x1 [step 1: (40,20) (40,30) -> (39,30) (40,21)]" in t


@needs_tree
def test_reshape_listed_only_when_no_other_change_explains_it(et):
    region = [(r, c) for r in range(10, 20) for c in range(10, 20)]
    f0 = paint(paint(blank(), region, 3), block(12, 12), 9)
    f1 = paint(paint(blank(), region, 3), block(12, 14), 9)  # the hole in the G region moves with the block
    f2 = paint([row[:] for row in f1], [(r, c) for r in range(30, 33) for c in range(30, 34)], 3)
    f2 = paint(f2, [(30, 33), (31, 33), (32, 33)], 2)  # a separate 3x3 G block ...
    f3 = paint([row[:] for row in f2], [(30, 33), (31, 33), (32, 33)], 3)  # ... grows to 3x4
    t = table(et, history([f0, f1, f2, f3], ["RIGHT", "SPACE", "SPACE"]))
    assert line(t, "RIGHT ") == "RIGHT x1: moved b 4px (+0,+2) x1 [step 1: (12,12)->(12,14)]"
    assert "reshaped G 9px->12px x1 [step 3: at (30,30)]" in line(t, "SPACE ")


@needs_tree
def test_game_over_overlap_level_completion_and_automatic_reset(et):
    same = history([paint(paint(blank(), block(20, 20), 9), block(20, 22), 8), paint(blank(), block(20, 22), 9)],
                   ["RIGHT"], [{"game_over": True}])
    assert line(table(et, same), "Game overs: ") == (  # same cells in a new colour is "recolored", before any move
        "Game overs: after RIGHT at step 1: recolored R->b 4px; disappeared b 4px at (20,20), its cells now g:4, "
        "edge band unchanged")
    f0 = paint(paint(blank(), block(20, 20), 9), block(20, 22, 2, 1), 8)
    f1 = paint(blank(), block(20, 22), 9)  # the block moved onto the red bar
    f2 = [row[:] for row in f0]  # automatic RESET
    f3 = paint(paint(blank(), block(22, 20), 9), block(20, 22, 2, 1), 8)
    f4 = paint(blank(5), block(30, 30), 9)  # level 2 starts
    hist = history([f0, f1, f2, f3, f4], ["RIGHT", "RESET", "DOWN", "DOWN"],
                   [{"game_over": True}, {"automatic": True}, {}, {"level_completed": True}], [1, 1, 1, 1, 2])
    ledger = et.EffectLedger(border=4)
    ledger.update(hist)
    p = ledger.payload()
    t = p["tables"]["1"]
    assert p["level"] == 2 and p["tables"]["2"] == "Level 2: no actions yet."
    assert t.startswith("Level 1: 3 actions (1 game over, 1 automatic RESET not counted).")
    assert line(t, "DOWN ") == "DOWN x2: level completed x1; moved b 4px (+2,+0) x1 [step 3: (20,20)->(22,20)]"
    assert line(t, "Game overs: ") == (
        "Game overs: after RIGHT at step 1: moved b 4px (+0,+2) (20,20)->(20,22) onto g:2 R:2; disappeared R 2px at "
        "(20,22), its cells now b:2, edge band unchanged")
    assert t.endswith("Edge band changed on 1 of 3 actions.")  # the level change redraws the band


@needs_tree
def test_periodic_mover_is_reported_and_an_avatar_walked_back_and_forth_is_not(et):
    cols = [30, 32, 34, 32] * 4
    frames = [paint(paint(blank(), block(20, 20), 9), block(40, c), 13) for c in cols[:13]]
    t = table(et, history(frames, ["UP"] * 12))
    assert line(t, "Objects whose ") == ("Objects whose places and turns repeated, period in actions: r 4px every 4 "
                                         "(steps 0-12), last 4 corners: (40,32) (40,34) (40,32) (40,30)")
    frames = [paint(blank(), block(20, 20 + 2 * (i % 2)), 9) for i in range(13)]
    t = table(et, history(frames, ["RIGHT", "LEFT"] * 6))
    assert "Objects whose" not in t and line(t, "LEFT ").startswith("LEFT x6: moved b 4px (+0,-2) x6")


@needs_tree
def test_click_targets_and_the_no_change_line(et):
    f0 = paint(blank(), block(20, 20), 9)
    f0[40][40] = 12
    lit = [row[:] for row in f0]
    lit[40][40] = 14
    clicks = ["MOUSE(row=20, col=21)", "MOUSE(row=5, col=5)", "MOUSE(row=20, col=21)", "MOUSE(row=1, col=1)",
              "MOUSE(row=40, col=40)"]
    t = table(et, history([f0, f0, f0, f0, f0, lit], clicks))
    assert line(t, "MOUSE(row=40") == ("MOUSE(row=40, col=40) x1 (on O 1px at (40,40)): recolored O->N 1px x1 "
                                       "[step 5: at (40,40)]")
    assert line(t, "MOUSE with") == "MOUSE with no change: 4 uses at 3 cells (row,col), latest first: (1,1) (20,21)x2 (5,5)"
    assert et.EffectLedger._target(et._Frame(tuple(map(tuple, f0)), 4), "MOUSE(row=5, col=5)") == "g area 3131px"


@needs_tree
def test_incremental_equals_batch_and_a_rewritten_history_restarts(et):
    frames = [paint(blank(), block(20, 20 + 2 * i), 9) for i in range(6)]
    hist = history(frames, ["RIGHT"] * 5)
    batch = et.EffectLedger(border=4)
    batch.update(hist)
    assert table(et, hist) == batch.payload()["tables"]["1"]
    other = history([paint(blank(), block(30, 30 + i), 9) for i in range(3)], ["RIGHT"] * 2)
    assert batch.update(other) == 3  # not a continuation: everything reprocessed
    fresh = et.EffectLedger(border=4)
    fresh.update(other)
    assert batch.payload() == fresh.payload()


@needs_tree
def test_attach_and_note_follow_the_flag(et, monkeypatch):
    f = [paint(blank(), block(20, 20 + 2 * i), 9) for i in range(4)]
    f[3][40][40] = 12  # the third RIGHT also lights a cell: a kind of change not seen before on the level
    hist = history(f, ["RIGHT"] * 3)

    class Agent:
        pass

    def serialize(**_):
        return {"history": hist[:n]}

    monkeypatch.delenv("OURS_EFFECT_TABLE", raising=False)
    assert et.attach(Agent(), serialize) is serialize and et.system_prompt_lines() == ""
    for flag, want_note in (("1", False), ("note", True)):
        monkeypatch.setenv("OURS_EFFECT_TABLE", flag)
        agent = Agent()
        n = 3
        wrapped = et.attach(agent, serialize)
        wrapped()  # the snippet starts after two actions ...
        n = 4
        state = wrapped()  # ... and its action() returns after the third
        assert "RIGHT x3: moved b 4px (+0,+2) x3" in state["ours_effects"]["tables"]["1"]
        payload = {}
        et.note(agent, payload)
        assert ("effects_note" in payload) == want_note
        if want_note:
            assert payload["effects_note"] == ("RIGHT at step 3 (used 2x before on this level): appeared O 1px at "
                                               "(40,40), a kind of change not seen before on level 1. effects() has "
                                               "the table.")
        assert "`effects(level=None)`" in et.system_prompt_lines()


@needs_tree
def test_an_internal_error_disables_the_table_without_raising(et, monkeypatch):
    monkeypatch.setenv("OURS_EFFECT_TABLE", "1")
    frames = [paint(blank(), block(20, 20 + 2 * i), 9) for i in range(3)]
    hist = history(frames, ["RIGHT"] * 2)

    def boom(*_):
        raise RuntimeError("planted")

    monkeypatch.setattr(et, "diff_frames", boom)
    wrapped = et.attach(type("Agent", (), {})(), lambda **_: {"history": hist})
    state = wrapped()
    assert state["history"] == hist and "internal error (RuntimeError)" in state["ours_effects"]["error"]


@needs_tree
def test_runtime_per_call_stays_small_on_a_long_64x64_game(et):
    frames, actions = [], []
    for i in range(401):
        g = paint(blank(), block(20 + (i // 50) % 10, 10 + (i % 40)), 9)
        g = paint(g, block(40, 10 + 2 * (i % 8)), 13)  # a patroller
        for k in range(30):  # static clutter: 30 small objects
            g[8 + 2 * (k // 6)][45 + 3 * (k % 6)] = 3 + k % 5
        g[63][:] = [0] * max(0, 64 - i % 64) + [2] * min(64, i % 64)
        frames.append(g)
        actions.append(["RIGHT", "DOWN", "MOUSE(row=30, col=30)"][i % 3])
    hist = history(frames, actions[:400])
    ledger = et.EffectLedger(border=4)
    times = []
    for k in range(1, len(hist) + 1):
        t0 = time.perf_counter()
        ledger.update(hist[:k])
        ledger.payload()
        times.append(time.perf_counter() - t0)
    assert max(times) < 0.5 and sum(times) / len(times) < 0.05, (max(times), sum(times) / len(times))


# --- recorded frames -----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dc22(trees):
    import effect_table_replay as replay_check

    game = {"game_id": "dc22", "history": [{"action": {"id": i, "data": d}} for i, d in DC22_FIRST]}
    return replay_check, replay_check.replay(game, ENV_DIR)


@needs_tree
@needs_games
def test_dc22_red_bar_click_is_in_the_table_after_its_first_use(et, dc22):
    replay_check, hist = dc22
    t = table(et, hist)
    assert line(t, "MOUSE(row=19, col=48)") == (
        "MOUSE(row=19, col=48) x1 (on R 47px at (17,42)): rotated R 24px 90 (-6,+6) x1 [step 10: (30,12)->(24,18)]; "
        "disappeared 2x W 40px x1 [step 10]")
    assert line(t, "LEFT ") == "LEFT x2: no change x1; moved N 4px (+0,-2) x1 [step 1: (40,10)->(40,8)]"
    errors: list = []
    assert replay_check.check_table(replay_check.Facts(hist, 4), 10, 1, t, errors, "dc22") == len(t.splitlines())
    assert errors == []


@needs_tree
@needs_games
def test_the_replay_check_flags_a_wrong_line(et, dc22):
    replay_check, hist = dc22
    t = table(et, hist)
    facts = replay_check.Facts(hist, 4)
    for wrong in (t.replace("(-6,+6)", "(-6,+7)"), t.replace("RIGHT x2", "RIGHT x3"),
                  t.replace("(on R 47px at (17,42))", "(on R 47px at (17,41))"), re.sub(r"on \d+ of", "on 1 of", t)):
        errors: list = []
        replay_check.check_table(facts, 10, 1, wrong, errors, "dc22")
        assert errors, wrong


# --- the real harness, in the bed venv -----------------------------------------------------------------------------

CHILD = r'''
import json, os, sys, tempfile
from pathlib import Path

tree, mode = sys.argv[1], sys.argv[2]
os.environ.update(json.loads(sys.argv[3]))
os.environ.pop("OURS_EFFECT_TABLE", None)
if mode != "off":
    os.environ["OURS_EFFECT_TABLE"] = mode
sys.path.insert(0, tree)
from inference.agent import tool_agent as ta
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state


class Game:
    """A blue 2x2 block moved by the arrows (a wall above), an orange cell that turns green once the block reaches
    column 26, and a bar in the bottom edge band that loses a cell per action."""

    def __init__(self, path):
        self.path, self.step, self.pos, self.lit = path, 0, (20, 20), False
        self.hist = [HistoryEntry(action="", frame=self.frame())]
        write_runtime_state(self.path, current_frame=self.frame(), history=self.hist)

    def grid(self):
        g = [[2] * 64 for _ in range(64)]
        for c in range(max(0, 64 - self.step)):
            g[63][c] = 0
        for dr in (0, 1):
            for dc in (0, 1):
                g[self.pos[0] + dr][self.pos[1] + dc] = 9
        g[40][40] = 14 if self.lit else 12
        return tuple(tuple(row) for row in g)

    def frame(self):
        return Frame(grid=self.grid(), step=self.step, level=1)

    def __call__(self, args):
        if "query" in args:
            return {"record": None}
        done = []
        for a in args["actions"]:
            name = a["action"].upper()
            before = self.grid()
            dr, dc = {"DOWN": (2, 0), "LEFT": (0, -2), "RIGHT": (0, 2)}.get(name, (0, 0))
            self.pos = (self.pos[0] + dr, self.pos[1] + dc)
            self.lit = self.lit or self.pos[1] >= 26
            self.step += 1
            after = self.grid()
            changed = any(before[r][4:60] != after[r][4:60] for r in range(4, 60))
            display = f"MOUSE(row={a['row']}, col={a['col']})" if name == "MOUSE" else name
            result = {"executed": True, "action_num": self.step, "level": 1, "score": 0, "state": "NOT_FINISHED",
                      "board_changed": before != after, "gameplay_changed": changed, "no_op": not changed,
                      "done": False, "level_completed": False, "game_over": False, "run_complete": False,
                      "action_name": name, "action_display": display, "automatic": False}
            self.hist.append(HistoryEntry(action=display, frame=self.frame(), result=dict(result)))
            write_runtime_state(self.path, current_frame=self.frame(), history=self.hist)
            done.append(display)
        result.update(executed_actions=done, executed_count=len(done), requested_count=len(done),
                      valid_actions=["UP", "DOWN", "LEFT", "RIGHT", "MOUSE"])
        return result


SNIPPETS = [
    "print(sorted(k for k in dir() if not k.startswith('_')))",
    "r = action(['RIGHT', 'RIGHT'])\nprint(r.get('executed_count'), r.get('gameplay_changed'))",
    "action([{'action': 'MOUSE', 'row': 5, 'col': 5}])\naction(['UP'])\nprint(current_frame.step)",
    "r = action(['RIGHT'])\nprint(r.get('executed_count'))",
    "print(len(history), last_action)",
]
EFFECTS = [
    "print(effects())",
    "effects()",
    "print(effects(1).splitlines()[0][:9]); print(effects(7))",
    "def first_line():\n    return effects().splitlines()[0][:9]\nprint(first_line())",
    "print(first_line())",
]
with tempfile.TemporaryDirectory() as tmp:
    state = Path(tmp) / "run" / "tool_runtime_state.json"
    game = Game(state)
    agent = ta.ToolAgent(model="local")
    agent._step_env_callback = game
    agent._current_valid_actions = ["UP", "DOWN", "LEFT", "RIGHT", "MOUSE"]
    out = {"prompt": agent._system_prompt, "results": []}
    for code in SNIPPETS + (EFFECTS if mode != "off" else []):
        out["results"].append(agent._run_python_tool(state, {"code": code}).content)
print(json.dumps(out))
'''


def run_child(tree: Path, mode: str, tmp_path: Path) -> dict:
    cells = franzen_tree.notebook_cells(franzen_tree.NOTEBOOK)
    env, _ = franzen_bed.notebook_env(franzen_bed._cell(cells, "setup_env = {"))
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    r = subprocess.run([str(BED_PY), str(script), str(tree), mode, json.dumps(env)], capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@needs_tree
@needs_bed
def test_flag_off_prompt_and_tool_results_are_byte_identical(trees, tmp_path):
    base = run_child(trees[0], "off", tmp_path)
    ours = run_child(trees[1], "off", tmp_path)
    assert ours["prompt"] == base["prompt"]
    assert ours["results"] == base["results"]
    assert "effects" not in ours["results"][0] and "effects_note" not in json.dumps(ours)


@needs_tree
@needs_bed
def test_flag_on_effects_in_the_real_sandbox_and_note_only_in_note_mode(et, trees, tmp_path):
    for mode in ("1", "note"):
        out = run_child(trees[1], mode, tmp_path)
        assert et.PROMPT_LINES.format(border=4) in out["prompt"]
        results = [json.loads(r) for r in out["results"]]
        assert "'effects'" in results[0]["stdout"]
        printed = results[5]["stdout"]
        assert "RIGHT x3: moved b 4px (+0,+2) x3 [last step 5: (20,24)->(20,26)]; recolored O->N 1px x1" in printed
        assert "UP x1: no change x1" in printed and "Edge band changed on 5 of 5 actions." in printed
        assert results[6]["stdout"].startswith("Level 1: 5 actions.")  # a bare effects() prints as plain text
        assert results[7]["stdout"] == "Level 1: \nLevel 7: no actions yet.\n"
        assert results[8]["stdout"] == "Level 1: \n" and results[9]["stdout"] == "Level 1: \n"  # retained
        notes = [r.get("effects_note") for r in results if r.get("effects_note")]
        if mode == "note":
            assert notes == ["RIGHT at step 5 (used 2x before on this level): recolored O->N 1px at (40,40), a kind of "
                             "change not seen before on level 1. effects() has the table."]
        else:
            assert notes == []
