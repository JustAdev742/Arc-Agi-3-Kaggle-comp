"""kaggle/franzen/patches/ours-08-perception.patch (OURS_PERCEPTION): exact perception helpers in Franzen's harness,
computed from the frames the harness already holds - a whole-view shift estimate with a running world offset and a
`[view]` line in the action echo, `left_view` for objects whose colour left the view at its edge, `logical_grid()` on
the frame's cell lattice, and an 8-connected option for his segmentation (`frame.segmentation8`).

Unit tests of inference/utils/ours_perception.py and ours_perception_sandbox.py (loaded from the patched tree) on
synthetic frames; the shift estimate against the engine's own scroll state on exp-073b's recorded bp35 and lf52
actions (slow: on every recorded run); the lattice on real level starts. In the bed venv, the real ToolAgent and
sandbox: with the flag off the system prompt and every tool result are byte-identical to the tree without the patch
(also over a whole analyze() drive), and with it on the helpers work on real games in the sandbox.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import franzen_bed  # noqa: E402
import franzen_tree  # noqa: E402

PATCHES = ROOT / "kaggle" / "franzen" / "patches"
SEVEN = [PATCHES / name for name in (
    "ours-sandbox-timeout-keeps-work.patch", "ours-02-budget-meter.patch", "ours-04-search-helper.patch",
    "ours-03b-win-ledger-on-02-04.patch", "ours-05-level-mem.patch", "ours-06b-effect-table-on-02-04-03b-05.patch",
    "ours-07-fresh-start.patch")]
PATCH = PATCHES / "ours-08-perception.patch"
BED_PY = Path.home() / ".cache" / "arc3-franzen-bed" / "venv" / "bin" / "python"
GAMES = ROOT / "environment_files"
LEDGER_CHECKS = ROOT / "tests" / "franzen_ledger_checks.py"
EXP073B = ROOT / "runs" / "exp073b-dprime-reap448-r14-accept05-full" / "kernel-output" / "benchmark.json"
OTHERS = {"OURS_BUDGET_METER": "1", "OURS_SEARCH_HELPER": "1", "OURS_WIN_LEDGER": "1", "OURS_LEVEL_MEM": "1",
          "OURS_EFFECT_TABLE": "1", "OURS_FRESH_START": "1", "EXPOSE_RESET": "on"}
NAMES = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT", "ACTION5": "SPACE",
         "ACTION7": "UNDO", "RESET": "RESET"}

needs_tree = pytest.mark.skipif(not (franzen_tree.his_repo_path() / "ARC3-Inference").is_dir(),
                                reason="needs Franzen's repo (scripts/franzen_tree.py)")
needs_bed = pytest.mark.skipif(not BED_PY.exists(), reason="needs the franzen_bed venv (scripts/franzen_bed.py)")
needs_games = pytest.mark.skipif(not (GAMES / "bp35").is_dir(), reason="needs environment_files/")
needs_run = pytest.mark.skipif(not EXP073B.exists(), reason="needs exp-073b's benchmark.json")


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    """The notebook's src/ after cell 4: the seven patches of the bundle arm, without this patch and with it."""
    out = {}
    for name, patches in (("base", SEVEN), ("ours", [*SEVEN, PATCH])):
        dest = tmp_path_factory.mktemp(name)
        logs = franzen_tree.notebook_bundle(dest, patches)
        assert [p.name for p in patches] == [k for k in logs if k != "his"]
        out[name] = dest / "src" / "ARC3-Inference"
    return out


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def op(trees):
    return _load(trees["ours"] / "inference/utils/ours_perception.py", "ours_perception_under_test")


@pytest.fixture(scope="module")
def sb(trees):
    return _load(trees["ours"] / "inference/utils/ours_perception_sandbox.py", "ours_perception_sandbox_under_test")


@pytest.fixture(scope="module")
def segs(trees):
    return (_load(trees["base"] / "inference/utils/segmentation.py", "segmentation_base"),
            _load(trees["ours"] / "inference/utils/segmentation.py", "segmentation_ours"))


# --- synthetic frames ------------------------------------------------------------------------------------------------


def world(h: int = 200, w: int = 200, seed: int = 1, background: int = 5, pieces: int = 140) -> list:
    """A large scene of random coloured rectangles, to look at through a 64x64 camera."""
    rng = random.Random(seed)
    g = [[background] * w for _ in range(h)]
    for _ in range(pieces):
        r, c = rng.randrange(h - 8), rng.randrange(w - 8)
        colour = rng.choice([2, 3, 4, 8, 9, 10, 11, 12, 14])
        for i in range(rng.randint(2, 7)):
            for j in range(rng.randint(2, 7)):
                g[r + i][c + j] = colour
    return g


def view(g: list, top: int, left: int, size: int = 64) -> list:
    return [row[left:left + size] for row in g[top:top + size]]


def paint(frame: list, cells, colour: int) -> list:
    frame = [row[:] for row in frame]
    for r, c in cells:
        frame[r][c] = colour
    return frame


def box(r: int, c: int, h: int, w: int) -> list:
    return [(r + i, c + j) for i in range(h) for j in range(w)]


def with_hud_and_player(frame: list) -> list:
    """A status bar in the bottom row and a player the camera follows: both stay put while the view moves."""
    return paint(paint(frame, [(63, c) for c in range(40)], 13), box(30, 30, 5, 5), 7)


def entry(action: str, frame: list, step: int, level: int = 1) -> dict:
    return {"action": action, "frame": {"grid": frame, "step": step, "level": level}, "result": {}}


def fed(op, history: list):
    """A tracker fed one entry at a time, as the harness serializes state after every action."""
    tracker = op.ViewTracker()
    for k in range(1, len(history) + 1):
        tracker.update(history[:k])
    return tracker


# --- the patch -------------------------------------------------------------------------------------------------------


def test_patch_is_a_notebook_patch_confined_to_the_harness():
    text = PATCH.read_text()
    names = [line.split()[2][2:] for line in text.splitlines() if line.startswith("diff --git ")]
    assert sorted(names) == ["ARC3-Inference/inference/agent/python_tool_sandbox.py",
                             "ARC3-Inference/inference/agent/tool_agent.py",
                             "ARC3-Inference/inference/utils/ours_perception.py",
                             "ARC3-Inference/inference/utils/ours_perception_sandbox.py",
                             "ARC3-Inference/inference/utils/segmentation.py"]
    assert "GIT binary patch" not in text and '"OURS_PERCEPTION"' in text


@needs_tree
def test_patch_applies_after_the_seven_and_its_hooks_are_flag_gated(trees):
    agent = (trees["ours"] / "inference/agent/tool_agent.py").read_text()
    sandbox = (trees["ours"] / "inference/agent/python_tool_sandbox.py").read_text()
    assert "def _ours_fresh_start(" in agent and "ours_effect_table import attach" in agent  # the seven are under it
    assert agent.count("_ours_perception.") == 3 and "perception=perception" in agent
    assert sandbox.count("__OURS_PERCEPTION_") == 6  # three placeholders, each written once and replaced once
    assert "ours_perception" not in (trees["base"] / "inference/agent/tool_agent.py").read_text()


@needs_tree
def test_flag_off_bootstraps_differ_only_by_the_segmentation_option(trees):
    """With the flag off a snippet runs the bootstrap from before the patch, apart from his segmentation source,
    which gained the connectivity option (the default path is unchanged; the next test runs it on real frames)."""
    def bootstraps(tree: Path) -> tuple:
        code = ("import json, sys; sys.path.insert(0, sys.argv[1]); from inference.agent import python_tool_sandbox "
                "as s; print(json.dumps([s._SANDBOX_BOOTSTRAP, s._SANDBOX_BOOTSTRAP_SEARCH, "
                "getattr(s, '_OURS_PERCEPTION_BOOTSTRAPS', {})]))")
        r = subprocess.run([sys.executable, "-I", "-c", code, str(tree)], capture_output=True, text=True, check=True)
        return tuple(json.loads(r.stdout))

    base, ours = bootstraps(trees["base"]), bootstraps(trees["ours"])
    seg_base = (trees["base"] / "inference/utils/segmentation.py").read_text()
    seg_ours = (trees["ours"] / "inference/utils/segmentation.py").read_text()
    for before, after in zip(base[:2], ours[:2]):
        assert seg_base in before and seg_ours in after
        assert after.replace(seg_ours, seg_base) == before
        assert "__OURS_PERCEPTION" not in after and "ours_perception_install" not in after
    on_plain, on_search = ours[2]["false"], ours[2]["true"]
    for text in (on_plain, on_search):
        assert "ours_perception_install(runtime_globals, FrameView, segment_layer, COLOR_CHARS)" in text
        assert '_ours_view_bind(runtime_globals, state_payload["ours_view"])' in text
        assert "__OURS_PERCEPTION" not in text
        compile(text, "<bootstrap>", "exec")
    assert "search_helper_globals" in on_search and "search_helper_globals" not in on_plain


@needs_tree
@needs_games
def test_segment_layer_default_unchanged_and_8_connected(segs, monkeypatch):
    base, ours = segs
    chars = "WwgGcBMPRbSYOrNp"
    for frame in level_starts(monkeypatch, [g.name for g in sorted(GAMES.iterdir())], first_only=True).values():
        assert ours.segment_layer(frame, chars) == base.segment_layer(frame, chars)
        assert ours.segment_layer(frame, chars, connectivity=4) == base.segment_layer(frame, chars)
    # a diagonal sprite (su15's invaders) is one object 8-connected and pixels 4-connected; a diamond ring of
    # diagonal steps encloses its centre both ways (the complement is still flooded 4-connected)
    grid = [[0] * 12 for _ in range(12)]
    for i in range(5):
        grid[1 + i][1 + i] = 9
    for r, c in ((5, 8), (6, 7), (6, 9), (7, 6), (7, 10), (8, 7), (8, 9), (9, 8)):
        grid[r][c] = 3
    grid[7][8] = 4
    four, eight = ours.segment_layer(grid, chars), ours.segment_layer(grid, chars, connectivity=8)
    assert sum(n["color"] == "b" for n in four["nodes"]) == 5 and sum(n["color"] == "b" for n in eight["nodes"]) == 1
    ring = [n for n in eight["nodes"] if n["color"] == "G"]
    centre = [n for n in eight["nodes"] if n["color"] == "c"]
    assert len(ring) == 1 and len(centre) == 1 and ring[0]["children"] == [centre[0]["id"]]
    assert ring[0]["pixels"] == 8 and len(ring[0]["boundary"]) >= 4
    with pytest.raises(ValueError):
        ours.segment_layer(grid, chars, connectivity=6)


# --- the view shift --------------------------------------------------------------------------------------------------

SCROLLS = [(0, 6), (0, -8), (18, 0), (-24, 0), (6, 6), (-42, 0), (0, 40), (3, -5), (48, 0)]


@needs_tree
@pytest.mark.parametrize("camera_move", SCROLLS)
def test_a_scrolled_view_is_measured_exactly(op, camera_move):
    """The camera moves by (dy, dx) over a textured scene, a status bar and a followed player stay put on screen:
    the content moved by (-dy, -dx)."""
    g = world()
    dy, dx = camera_move
    a = with_hud_and_player(view(g, 60, 60))
    b = with_hud_and_player(view(g, 60 + dy, 60 + dx))
    assert op.view_shift(a, b) == (-dy, -dx)
    assert op.view_shift(view(g, 60, 60), view(g, 60 + dy, 60 + dx)) == (-dy, -dx)  # and without them


@needs_tree
def test_moving_objects_on_a_still_view_are_not_a_scroll(op):
    g = world()
    still = view(g, 60, 60)
    # one sprite steps right over a textured, static scene
    a = paint(still, box(30, 30, 5, 5), 7)
    b = paint(paint(still, box(30, 30, 5, 5), 5), box(30, 36, 5, 5), 7)
    assert op.measure(a, b)["verdict"] == "none"
    # a big sprite moves over a uniform floor: pixel for pixel that is also the view moving the other way, but the
    # evidence spans little of the frame
    floor = [[5] * 64 for _ in range(64)]
    a, b = paint(floor, box(20, 10, 20, 20), 9), paint(floor, box(20, 16, 20, 20), 9)
    assert op.measure(a, b)["verdict"] == "none"
    # five sprites move together by the same step, the scene stays
    sprites = [(10, 10), (10, 40), (40, 12), (45, 45), (25, 28)]
    a = still
    for r, c in sprites:
        a = paint(a, box(r, c, 4, 4), 7)
    b = still
    for r, c in sprites:
        b = paint(b, box(r + 3, c, 4, 4), 7)
    assert op.measure(a, b)["verdict"] == "none"
    # a periodic texture (bp35's dotted floor) with a sprite moving by the texture's period
    dots = [[5 if (r % 6, c % 6) not in ((1, 2), (4, 1), (5, 4)) else 3 for c in range(64)] for r in range(64)]
    a, b = paint(dots, box(20, 20, 6, 6), 9), paint(dots, box(20, 26, 6, 6), 9)
    assert op.measure(a, b)["verdict"] == "none"


@needs_tree
def test_wraparound_edges_and_odd_frames(op):
    g = world(seed=3)
    a = view(g, 70, 70)
    rolled = [row[-8:] + row[:-8] for row in a]  # a toroidal view: content leaving on the right comes back left
    assert op.view_shift(a, rolled) == (0, 8)
    rolled_up = a[5:] + a[:5]
    assert op.view_shift(a, rolled_up) == (-5, 0)
    assert op.view_shift(a, view(g, 70 + 60, 70)) is None  # 4 rows of overlap: too little to tell, never a guess
    assert op.view_shift(a, a) is None
    assert op.view_shift(a, [row[:32] for row in a[:32]]) is None  # different shapes
    assert op.view_shift(a, [[14] * 64 for _ in range(64)]) is None  # a flash
    assert op.view_shift(a, None) is None and op.as_rows([[1, 2], [3]]) is None and op.as_rows([[99]]) is None


@needs_tree
def test_unsure_makes_the_offset_unknown_until_a_reset(op):
    """A scroll under a static panel over half the board: the panel contradicts the shift, so the harness says it
    cannot tell (offset None) rather than guess, until the RESET restores the level start."""
    g = world(seed=5)
    panel = [(r, c) for r in range(8, 56) for c in range(8, 30) if (r // 3 + c // 3) % 2]
    a = paint(view(g, 60, 60), panel, 13)
    b = paint(view(g, 66, 60), panel, 13)
    m = op.measure(a, b)
    assert m["verdict"] == "unsure", m
    history = [entry("", a, 0), entry("RIGHT", b, 1), entry("RIGHT", view(g, 66, 54), 2), entry("RESET", a, 3)]
    tracker = fed(op, history)
    assert tracker.offsets == [(0, 0), None, None, (0, 0)]
    assert tracker.unsure == [False, True, False, False]
    assert tracker.line(1) == ("[view] the frame changed at step 1 in a way that may be a scroll; view_offset is "
                               "unknown until the next RESET or level")


@needs_tree
def test_tracker_offsets_resets_levels_and_lines(op):
    g = world(seed=7)
    cams = [(60, 60), (66, 60), (66, 52), (66, 52), (60, 60), (90, 90), (90, 96)]
    frames = [with_hud_and_player(view(g, y, x)) for y, x in cams]
    history = [entry("", frames[0], 0), entry("DOWN", frames[1], 1), entry("LEFT", frames[2], 2),
               entry("SPACE", frames[3], 3), entry("RESET", frames[4], 4), entry("RIGHT", frames[5], 5, level=2),
               entry("RIGHT", frames[6], 6, level=2)]
    tracker = fed(op, history)
    assert tracker.offsets == [(0, 0), (6, 0), (6, -8), (6, -8), (0, 0), (0, 0), (0, 6)]
    assert [m and m[1] for m in tracker.shifts] == [None, (-6, 0), (0, 8), None, None, None, (0, -6)]
    assert tracker.line(1) == ("[view] scrolled 2 times (steps 1, 2): content moved (-6,+0), (+0,+8); view_offset "
                               "now (+0,-6)")  # the call ran to the end of the history: the last offset
    two = fed(op, history[:3])
    assert two.line(2) == "[view] scrolled at step 2: content moved (+0,+8); view_offset now (+6,-8)"
    assert two.line(1).startswith("[view] scrolled 2 times (steps 1, 2)") and two.line(3) == "" and two.line(-1) == ""
    payload = tracker.payload(history[-1]["frame"]["grid"])
    assert payload["offsets"][2] == [6, -8] and payload["current"] == [0, 6] and payload["left"] == []
    assert tracker.payload(frames[0])["current"] is None  # a current frame that is not the last entry's
    # a rewritten history is processed again from the start
    assert tracker.update(history[:2]) == 2 and tracker.offsets == [(0, 0), (6, 0)]


@needs_tree
def test_left_view(op):
    floor = [[5] * 64 for _ in range(64)]
    for c in range(64):
        floor[63][c] = 13  # a bar in the edge band
    cart = box(20, 58, 6, 6)  # touches the right edge
    gem = box(40, 30, 3, 3)  # in the middle
    a = paint(paint(paint(floor, cart, 11), gem, 14), box(2, 10, 2, 8), 8)  # and a red HUD piece in the band
    b = floor  # the cart, the gem and the HUD piece are all gone
    history = [entry("", a, 0), entry("RIGHT", b, 1), entry("LEFT", paint(floor, box(20, 50, 6, 6), 11), 2)]
    tracker = fed(op, history[:2])
    record = {"colour": "Y", "pixels": 36, "step": 0, "box": [20, 58, 25, 63], "world_box": [20, 58, 25, 63],
              "edge": "right"}
    assert tracker.exits[1] == [record] and tracker.left == [record]  # not the gem (mid-view), not the HUD piece
    assert tracker.line(1) == ("[view] left the view: Y 36px at the right edge (last seen step 0, rows 20-25, cols "
                               "58-63); see left_view")
    assert fed(op, history).left == []  # the colour is back in view
    # pushed out by a scroll: the record carries the world box of the frame it was last seen in
    g = world(seed=9)
    a = paint(view(g, 60, 60), box(2, 20, 4, 4), 1)  # a w object near the top, unique colour
    b = view(g, 66, 60)  # the camera moved down 6: the content moved up 6, the object is gone
    assert 1 not in {v for row in b for v in row} and op.view_shift(a, b) == (-6, 0)
    t2 = fed(op, [entry("", view(g, 54, 60), 0), entry("DOWN", a, 1), entry("DOWN", b, 2)])
    assert t2.offsets[1] == (0, 0) or t2.offsets[1] is not None
    (gone,) = t2.exits[2]
    assert gone["colour"] == "w" and gone["edge"] == "top" and gone["step"] == 1 and gone["box"] == [2, 20, 5, 23]
    off = t2.offsets[1]
    assert gone["world_box"] == [2 + off[0], 20 + off[1], 5 + off[0], 23 + off[1]]
