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
    """The camera moves by (dy, dx) over a textured scene: the content moved by (-dy, -dx). With a status bar and a
    player the camera follows (both stay put on screen), a game's first scroll may come out unsure (the offset
    becomes unknown), never wrong; once the game has scrolled, it is exact."""
    dy, dx = camera_move
    for seed in (1, 2, 3):
        g = world(seed=seed)
        a, b = view(g, 60, 60), view(g, 60 + dy, 60 + dx)
        assert op.view_shift(a, b) == (-dy, -dx)
        a, b = with_hud_and_player(a), with_hud_and_player(b)
        assert op.view_shift(a, b) in ((-dy, -dx), "unsure")
        assert op.view_shift(a, b, scrolled=True) == (-dy, -dx)


@needs_tree
def test_moving_objects_on_a_still_view_are_not_a_scroll(op):
    g = world()
    still = view(g, 60, 60)
    # one sprite steps right over a textured, static scene
    a = paint(still, box(30, 30, 5, 5), 7)
    b = paint(paint(still, box(30, 30, 5, 5), 5), box(30, 36, 5, 5), 7)
    assert op.measure(a, b)["verdict"] == "none"
    # a big sprite moves over an empty uniform floor: pixel for pixel that is also the view moving the other way over
    # a lone object, so it is never claimed; with nothing in view that stays put, the offset may become unknown
    floor = [[5] * 64 for _ in range(64)]
    a, b = paint(floor, box(20, 10, 20, 20), 9), paint(floor, box(20, 16, 20, 20), 9)
    assert op.view_shift(a, b) in (None, "unsure") and op.view_shift(a, b, scrolled=True) in (None, "unsure")
    # the same sprite in a room whose walls stay put
    room = paint(floor, [(r, c) for r in range(64) for c in range(64) if r in (8, 55) or c in (6, 57)], 3)
    a, b = paint(room, box(20, 10, 20, 20), 9), paint(room, box(20, 16, 20, 20), 9)
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
    """Evidence for a scroll too weak to claim and too strong to ignore makes the offset unknown (None) rather than
    possibly wrong, until a RESET restores the level start: silently in a game whose view has not moved yet, with a
    [view] line once it has."""
    # a 24-row camera move with a status bar and a followed player, in a game that has not scrolled yet
    g = world(seed=1)
    a, b = with_hud_and_player(view(g, 60, 60)), with_hud_and_player(view(g, 36, 60))
    assert op.measure(a, b)["verdict"] == "unsure"
    history = [entry("", a, 0), entry("UP", b, 1), entry("UP", with_hud_and_player(view(g, 30, 60)), 2),
               entry("RESET", a, 3)]
    tracker = fed(op, history)
    assert tracker.offsets == [(0, 0), None, None, (0, 0)] and tracker.unsure == [False, True, False, False]
    assert fed(op, history[:2]).line(1) == ""
    assert fed(op, history[:3]).line(2) == "[view] scrolled at step 2: content moved (+6,+0); view_offset now unknown"
    # after a scroll, a camera move under a static panel over half the board: the panel contradicts the shift
    g = world(seed=5)
    panel = [(r, c) for r in range(8, 56) for c in range(8, 38) if (r // 3 + c // 3) % 2 == 0]
    frames = [view(g, 54, 60), view(g, 60, 60), paint(view(g, 60, 60), panel, 13), paint(view(g, 66, 60), panel, 13)]
    history = [entry("", frames[0], 0), entry("DOWN", frames[1], 1), entry("SPACE", frames[2], 2),
               entry("DOWN", frames[3], 3), entry("RESET", frames[0], 4)]
    tracker = fed(op, history)
    assert tracker.offsets == [(0, 0), (6, 0), (6, 0), None, (0, 0)]
    assert tracker.unsure == [False, False, False, True, False] and tracker.scrolled
    three = fed(op, history[:4])
    assert three.line(3) == ("[view] the frame changed at step 3 in a way that may be a scroll; view_offset is "
                             "unknown until the next RESET or level")
    assert three.payload(frames[3])["current"] is None


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
    # one call that ran to the end of the history: every scroll in it, the last offset; and the two orange objects
    # that the step-2 scroll pushed off the right edge
    assert tracker.line(1).splitlines() == [
        "[view] scrolled 3 times (steps 1, 2, 6): content moved (-6,+0), (+0,+8), (+0,-6); view_offset now (+0,+6)",
        "[view] left the view: O 8px at the right edge (last seen step 1, rows 52-53, cols 59-63); O 16px at the right "
        "edge (last seen step 1, rows 58-62, cols 59-63); see left_view"]
    assert tracker.exits[2][1] == {"colour": "O", "pixels": 16, "step": 1, "box": [58, 59, 62, 63],
                                   "world_box": [64, 59, 68, 63], "edge": "right"}
    two = fed(op, history[:3])
    assert two.line(2).splitlines()[0] == "[view] scrolled at step 2: content moved (+0,+8); view_offset now (+6,-8)"
    assert two.line(1).startswith("[view] scrolled 2 times (steps 1, 2)") and two.line(3) == "" and two.line(-1) == ""
    assert len(two.left) == 2 and tracker.left == []  # the RESET and the new level drop the records
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
    assert t2.offsets == [(0, 0), (6, 0), (12, 0)]
    (gone,) = t2.exits[2]
    assert gone == {"colour": "w", "pixels": 16, "step": 1, "box": [2, 20, 5, 23], "world_box": [8, 20, 11, 23],
                    "edge": "top"}


# --- the lattice and the logical grid --------------------------------------------------------------------------------

CHARS = "WwgGcBMPRbSYOrNp"


def board(cell: int, origin: tuple, pattern: list, size: int = 64, background: int = 5) -> list:
    """A frame drawn on a lattice: cell (i, j) of `pattern` covers rows origin[0] + cell * i.., cols origin[1] + .."""
    g = [[background] * size for _ in range(size)]
    for i, row in enumerate(pattern):
        for j, colour in enumerate(row):
            for r in range(origin[0] + cell * i, min(size, origin[0] + cell * (i + 1))):
                for c in range(origin[1] + cell * j, min(size, origin[1] + cell * (j + 1))):
                    g[r][c] = colour
    return g


def maze(n: int, seed: int) -> list:
    rng = random.Random(seed)
    return [[rng.choice((5, 5, 5, 3, 3, 4)) for _ in range(n)] for _ in range(n)]


@needs_tree
def test_lattice_on_synthetic_boards(sb):
    five = maze(12, 1)
    g = paint(board(5, (0, 4), five), [(31, 21), (32, 20), (32, 21), (32, 22), (33, 21)], 0)  # a cross in a cell
    assert sb.ours_find_lattice(g) == (5, (0, 4))
    view_ = sb.ours_logical_grid(g, None, None, CHARS)
    assert view_.cell == (5, 5) and view_.origin == (0, 4) and view_.shape == (12, 12)
    expected = ["".join(CHARS[v] for v in row) for row in five]
    expected[6] = expected[6][:3] + "*" + expected[6][4:]  # cell (6, 3) holds the cross
    assert view_.rows == expected and list(view_.mixed) == [(6, 3)]
    assert view_.mixed[(6, 3)].endswith("W5") and view_.box(6, 3) == (30, 19, 34, 23) and view_.cell_of(32, 21) == (6, 3)
    assert view_.cell_of(0, 2) is None and view_.cell_of(63, 63) is None  # left of the origin, below the last cell
    assert str(view_).splitlines()[0] == "cells 5x5 px from (0,4): 12 rows x 12 cols; '*' = mixed (1, see .mixed)"
    assert sb.ours_find_lattice(board(3, (2, 2), maze(20, 2))) == (3, (2, 2))
    assert sb.ours_find_lattice(view(world(seed=4), 50, 50)) is None  # rectangles at random places
    assert sb.ours_find_lattice([[5] * 64 for _ in range(64)]) is None  # nothing to read
    small = [[random.Random(r * 64 + c).choice((2, 5, 9)) for c in range(32)] for r in range(32)]
    assert sb.ours_find_lattice([[small[r // 2][c // 2] for c in range(64)] for r in range(64)]) == (2, (0, 0))
    # a lattice set by hand, and one that does not fit
    assert sb.ours_logical_grid(g, (5, 5), (0, 4), CHARS).rows == view_.rows
    assert sb.ours_logical_grid(g, 4, None, CHARS).cell == (4, 4)
    with pytest.raises(ValueError):
        sb.ours_logical_grid(g, (0, 5), None, CHARS)


def level_starts(monkeypatch, games: list, first_only: bool = False) -> dict:
    """{(game, level): the level's first frame} from the engine: set_level, then a level RESET."""
    monkeypatch.setenv("ONLY_RESET_LEVELS", "true")
    monkeypatch.syspath_prepend(str(ROOT))
    from arcengine import ActionInput, GameAction

    from arc3.env import LocalEnv, make_arcade

    arc = make_arcade(GAMES)
    out = {}
    for name in games:
        g = LocalEnv(arc, name, seed=0).env._game
        for level in range(1 if first_only else len(g._levels)):
            g.set_level(level)
            last = g.perform_action(ActionInput(id=GameAction.RESET), raw=True).frame[-1]
            out[(name, level + 1)] = [[int(v) for v in row] for row in (last.tolist() if hasattr(last, "tolist")
                                                                         else last)]
    return out


@needs_tree
@needs_games
def test_lattice_on_real_level_starts(sb, monkeypatch):
    starts = level_starts(monkeypatch, ["ls20", "cn04", "tu93", "bp35", "lf52", "ft09"])
    found = {key: sb.ours_find_lattice(frame) for key, frame in starts.items()}
    assert all(found[("ls20", level)] == (5, (0, 4)) for level in range(1, 8))  # the maze's 5-pixel cells
    assert all(found[("cn04", level)] == (3, (2, 2)) for level in range(1, 7))  # the pieces' 3-pixel cells
    assert all(found[("tu93", level)][0] == 3 for level in range(1, 10))
    assert all(found[("ft09", level)] == (2, (0, 0)) for level in range(1, 7))
    assert not any(found[("bp35", level)] for level in range(1, 10))  # dotted texture: no lattice claimed
    assert not any(found[("lf52", level)] for level in range(1, 11))
    grid = sb.ours_logical_grid(starts[("ls20", 1)], None, None, CHARS)
    assert grid.rows[5] == "ccGGGGGGGGcc" and grid.rows[0] == "cccccccccccc" and len(grid.mixed) == 15
    pieces = sb.ours_logical_grid(starts[("cn04", 1)], None, None, CHARS)
    assert pieces.rows[3] == "SSSWWWWWSSSSSSSSSSSS" and pieces.rows[8] == "SSSSRSRSSSSSSSSSSSSS" and not pieces.mixed


# --- recorded runs against the engine ---------------------------------------------------------------------------------


def engine_origin(name: str, game) -> tuple:
    """Where the engine draws the world from, in frame pixels: bp35's scene camera (frame = world - camera), lf52's
    world node (frame = world + node), else the engine camera (which no other public game moves)."""
    if name == "bp35":
        return game.oztjzzyqoek.camera.y, game.oztjzzyqoek.camera.x
    if name == "lf52":
        return -game.ikhhdzfmarl.hncnfaqaddg.y, -game.ikhhdzfmarl.hncnfaqaddg.x
    return game._camera.y, game._camera.x


def replay_with_engine(game: dict, monkeypatch) -> tuple:
    """The harness's history for one recorded game (start frame, then one entry per action) and the engine's
    origin after each entry."""
    monkeypatch.syspath_prepend(str(ROOT))
    from arcengine import GameAction

    from arc3.env import Action, LocalEnv, make_arcade

    name = game["game_id"].split("-")[0]
    env = LocalEnv(make_arcade(GAMES), game["game_id"], seed=0)
    frame = env.frame
    history = [entry("", frame.grid.tolist(), 0, frame.levels_completed + 1)]
    origins = [engine_origin(name, env.env._game)]
    for i, rec in enumerate(game["history"], 1):
        a = rec["action"]
        if a["id"] == "ACTION6":
            action, display = Action.click(a["data"]["x"], a["data"]["y"]), \
                f"MOUSE(row={int(a['data']['y'])}, col={int(a['data']['x'])})"
        else:
            action, display = Action(GameAction[a["id"]]), NAMES[a["id"]]
        before = frame
        frame = env.step(action)
        won = frame.state.name == "WIN"
        history.append(entry(display, frame.grid.tolist(), i,
                             before.levels_completed + 1 if won else frame.levels_completed + 1))
        origins.append(engine_origin(name, env.env._game))
    return history, origins


def compare(op, history: list, origins: list) -> dict:
    """Per frame, the tracker's offset against the engine's (relative to the level start or the last RESET); per
    action, the claimed shift against the engine's."""
    tracker = fed(op, history)
    out = {"frames": len(history), "agree": 0, "disagree": 0, "unknown": 0, "scrolls": 0, "claimed_exactly": 0,
           "claimed_wrong": 0, "unsure_on_scroll": 0, "missed": 0, "false_claims": 0, "unsure_on_still": 0}
    base = None
    for i, origin in enumerate(origins):
        if tracker.resets[i]:
            base = origin
        else:
            moved = origin != origins[i - 1]
            truth = (origins[i - 1][0] - origin[0], origins[i - 1][1] - origin[1])
            claimed = tracker.shifts[i] and tracker.shifts[i][1]
            if moved:
                out["scrolls"] += 1
                key = ("claimed_exactly" if claimed == truth else "claimed_wrong" if claimed else
                       "unsure_on_scroll" if tracker.offsets[i] is None else "missed")
                out[key] += 1
            else:
                out["false_claims"] += bool(claimed)
                out["unsure_on_still"] += bool(tracker.unsure[i])
        offset = tracker.offsets[i]
        truth_offset = (origin[0] - base[0], origin[1] - base[1])
        out["unknown" if offset is None else "agree" if tuple(offset) == truth_offset else "disagree"] += 1
    return out


@needs_tree
@needs_games
@needs_run
@pytest.mark.parametrize("name,scrolls", [("bp35", 27), ("lf52", 12)])
def test_recorded_scrolls_agree_with_the_engine_at_every_step(op, monkeypatch, name, scrolls):
    data = json.loads(EXP073B.read_text())
    game = next(g for g in data["game_runs"] if g["game_id"].startswith(name))
    result = compare(op, *replay_with_engine(game, monkeypatch))
    assert result == {"frames": len(game["history"]) + 1, "agree": len(game["history"]) + 1, "disagree": 0,
                      "unknown": 0, "scrolls": scrolls, "claimed_exactly": scrolls, "claimed_wrong": 0,
                      "unsure_on_scroll": 0, "missed": 0, "false_claims": 0, "unsure_on_still": 0}


@pytest.mark.slow
@needs_tree
@needs_games
def test_every_recorded_run(op, monkeypatch):
    """Every game of every run under runs/ (about 36 runs x 25 games): no shift is claimed where the engine did not
    move the view, no claimed shift is wrong, no scroll goes unnoticed (at worst the offset becomes unknown), and no
    offset disagrees with the engine's."""
    totals: dict = {}
    for path in sorted(glob.glob(str(ROOT / "runs" / "*" / "kernel-output" / "benchmark.json"))):
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            continue
        for game in data.get("game_runs", []):
            if game.get("history"):
                for key, value in compare(op, *replay_with_engine(game, monkeypatch)).items():
                    totals[key] = totals.get(key, 0) + value
    assert totals["scrolls"] > 200 and totals["frames"] > 50_000, totals
    assert totals["claimed_wrong"] == totals["missed"] == totals["false_claims"] == totals["disagree"] == 0, totals


# --- the real harness, in the bed venv -----------------------------------------------------------------------------

CHILD = r'''
import json, os, sys, tempfile
from pathlib import Path

tree, mode, game_name, env_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
os.environ.update(json.loads(sys.argv[5]))
snippets = json.loads(sys.argv[6])
os.environ.pop("OURS_PERCEPTION", None)
if mode != "off":
    os.environ["OURS_PERCEPTION"] = mode
sys.path.insert(0, tree)
from inference.agent import tool_agent as ta
from inference.agent.action_names import to_engine_action
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state


class Scripted:
    """A blue 2x2 block moved by the arrows, an orange cell that turns green once the block reaches column 26, and a
    bar in the bottom edge band that loses a cell per action."""

    def __init__(self, path):
        self.path, self.step, self.pos, self.lit = path, 0, (20, 20), False
        self.hist = [HistoryEntry(action="", frame=self.frame())]
        write_runtime_state(self.path, current_frame=self.frame(), history=self.hist)
        self.valid = ["UP", "DOWN", "LEFT", "RIGHT", "MOUSE"]

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
                      valid_actions=list(self.valid))
        return result


class Engine:
    """A real public game in the offline engine, reduced to what the agent reads (no guards); `origins` holds the
    engine's own scroll state after every entry (bp35's scene camera, else the engine camera)."""

    def __init__(self, path, prefix):
        import arc_agi
        from arc_agi import OperationMode

        arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=env_dir)
        game_id = next(e.game_id for e in arc.available_environments if e.game_id.startswith(prefix))
        self.env = arc.make(game_id, seed=0)
        self.raw = self.env.observation_space
        self.path, self.prefix = path, prefix
        self.hist = [HistoryEntry(action="", frame=Frame(grid=self.grid(), step=0, level=self.level()))]
        self.origins = [self.origin()]
        write_runtime_state(self.path, current_frame=self.hist[-1].frame, history=self.hist)

    def origin(self):
        game = self.env._game
        if self.prefix == "bp35":
            return [game.oztjzzyqoek.camera.y, game.oztjzzyqoek.camera.x]
        return [game._camera.y, game._camera.x]

    def grid(self):
        layer = self.raw.frame[-1]
        rows = layer.tolist() if hasattr(layer, "tolist") else layer
        return tuple(tuple(int(v) for v in row) for row in rows)

    def level(self):
        return max(1, min(int(self.raw.win_levels), int(self.raw.levels_completed) + 1))

    @property
    def valid(self):
        from arcengine import GameAction
        from inference.agent.action_names import to_model_actions

        names = [GameAction.from_id(int(a)).name for a in self.raw.available_actions or []]
        return to_model_actions([n for n in names if n != "RESET"])

    def __call__(self, args):
        from arcengine import GameAction, GameState

        if "query" in args:
            return {"record": None}
        done, stop = [], None
        for a in args["actions"]:
            label = str(a["action"]).upper()
            if label == "MOUSE":
                name, data, display = "ACTION6", {"x": int(a["col"]), "y": int(a["row"])}, \
                    f"MOUSE(row={int(a['row'])}, col={int(a['col'])})"
            else:
                name, data, display = to_engine_action(label), {}, label
            before, completed = self.grid(), int(self.raw.levels_completed)
            self.raw = self.env.step(GameAction[name], data=data)
            after = self.grid()
            won = self.raw.state == GameState.WIN
            result = {"executed": True, "action_num": len(self.hist), "level": self.level(),
                      "score": int(self.raw.levels_completed), "state": self.raw.state.name,
                      "board_changed": after != before,
                      "gameplay_changed": any(x[4:-4] != y[4:-4] for x, y in zip(before[4:-4], after[4:-4])),
                      "done": won, "level_completed": int(self.raw.levels_completed) > completed and not won,
                      "game_over": self.raw.state == GameState.GAME_OVER, "run_complete": won,
                      "action_name": name, "action_display": display, "automatic": False}
            self.hist.append(HistoryEntry(action=display, frame=Frame(grid=after, step=len(self.hist),
                                                                       level=self.level()), result=dict(result)))
            self.origins.append(self.origin())
            write_runtime_state(self.path, current_frame=self.hist[-1].frame, history=self.hist)
            done.append(display)
            stop = next((k for k in ("run_complete", "game_over", "level_completed") if result[k]), None)
            if stop:
                break
        result.update(executed_actions=done, executed_count=len(done), requested_count=len(args["actions"]),
                      stopped_early=len(done) < len(args["actions"]), valid_actions=list(self.valid))
        if stop:
            result["stop_reason"] = stop
        return result


with tempfile.TemporaryDirectory() as tmp:
    state = Path(tmp) / "run" / "tool_runtime_state.json"
    game = Scripted(state) if game_name == "scripted" else Engine(state, game_name)
    agent = ta.ToolAgent(model="local")
    agent._step_env_callback = game
    agent._current_valid_actions = list(game.valid)
    out = {"prompt": agent._system_prompt, "results": []}
    for code in snippets:
        out["results"].append(agent._run_python_tool(state, {"code": code}).content)
    out["origins"] = getattr(game, "origins", None)
print(json.dumps(out))
'''

FLAG_OFF_SNIPPETS = [
    "print(sorted(k for k in dir() if not k.startswith('_')))",
    "print(sorted(k for k in dir(current_frame) if not k.startswith('_')))",
    "r = action(['RIGHT', 'RIGHT'])\nprint(r.get('executed_count'), r.get('gameplay_changed'))",
    "nodes = current_frame.segmentation['nodes']\nprint(len(nodes), nodes[0]['boundary'][:3], nodes[-1]['color'])",
    "action([{'action': 'MOUSE', 'row': 5, 'col': 5}])\naction(['UP'])\nprint(current_frame.step)",
    "r = action(['RIGHT'])\nprint(r)",
    "print(len(history), last_action, previous_frame.step)",
]


def run_child(tree: Path, mode: str, game: str, snippets: list, tmp_path: Path, extra: dict | None = None) -> dict:
    env, _ = franzen_bed.notebook_env(franzen_bed._cell(franzen_tree.notebook_cells(), "setup_env = {"))
    env.update(extra or {})
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    r = subprocess.run([str(BED_PY), str(script), str(tree), mode, game, str(GAMES), json.dumps(env),
                        json.dumps(snippets)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@needs_tree
@needs_bed
@pytest.mark.parametrize("others", ["off", "on"])
def test_flag_off_prompt_and_tool_results_are_byte_identical(trees, tmp_path, others):
    extra = dict(OTHERS) if others == "on" else {}
    base = run_child(trees["base"], "off", "scripted", FLAG_OFF_SNIPPETS, tmp_path, extra)
    for mode in ("off", "0"):
        ours = run_child(trees["ours"], mode, "scripted", FLAG_OFF_SNIPPETS, tmp_path, extra)
        assert ours["prompt"] == base["prompt"], mode
        assert ours["results"] == base["results"], mode
    stdout = json.loads(base["results"][0])["stdout"] + json.loads(base["results"][1])["stdout"]
    for name in ("view_offset", "left_view", "logical_grid", "segmentation8"):
        assert name not in stdout and name not in base["prompt"]


@needs_tree
@needs_bed
@needs_games
def test_flag_on_bp35_scroll_line_and_offsets_match_the_engine(op, trees, tmp_path):
    out = run_child(trees["ours"], "1", "bp35", [
        "r = action(['RIGHT'] * 4)\nprint(r.get('executed_count'))",
        "print(view_offset, current_frame.view_offset, history[3].frame.view_offset, history[4].frame.view_offset)\n"
        "print(left_view, sorted(k for k in dir() if k in ('view_offset', 'left_view', 'logical_grid')))",
    ], tmp_path, OTHERS)
    assert op.PROMPT_LINES in out["prompt"]
    first, second = (json.loads(r)["stdout"] for r in out["results"])
    assert first.splitlines() == ["[view] scrolled at step 4: content moved (+18,+0); view_offset now (-18,+0)", "4"]
    camera = out["origins"]
    engine = (camera[4][0] - camera[0][0], camera[4][1] - camera[0][1])  # frame = world - camera
    assert engine == (-18, 0)
    assert second.splitlines() == [f"{engine} {engine} (0, 0) {engine}",
                                   "[] ['left_view', 'logical_grid', 'view_offset']"]


@needs_tree
@needs_bed
@needs_games
def test_flag_on_ls20_logical_grid_and_segmentation8_in_the_sandbox(trees, tmp_path):
    out = run_child(trees["ours"], "1", "ls20", [
        "print(logical_grid())",
        "g = logical_grid()\nprint(g.cell, g.origin, g.shape, len(g.mixed), g.box(5, 2), g.cell_of(27, 16))",
        "a, b = current_frame.segmentation, current_frame.segmentation8\n"
        "print(len(b['nodes']) <= len(a['nodes']), sorted(a) == sorted(b), sorted(b['nodes'][0]) == sorted(a['nodes'][0]))",
        "def row5():\n    return logical_grid(cell=5, origin=(0, 4)).rows[5]\nprint(row5())",
        "print(row5(), logical_grid(history[0].frame).rows == logical_grid().rows)",
    ], tmp_path, OTHERS)
    printed = [json.loads(r)["stdout"].splitlines() for r in out["results"]]
    assert printed[0][0] == "cells 5x5 px from (0,4): 12 rows x 12 cols; '*' = mixed (15, see .mixed)"
    assert printed[0][6] == "  5 ccGGGGGGGGcc" and len(printed[0]) == 13
    assert printed[1] == ["(5, 5) (0, 4) (12, 12) 15 (25, 14, 29, 18) (5, 2)"]
    assert printed[2] == ["True True True"]
    assert printed[3] == ["ccGGGGGGGGcc"] and printed[4] == ["ccGGGGGGGGcc True"]  # a retained function uses it


@needs_tree
@needs_bed
@needs_games
@needs_run
def test_flag_on_lf52_scroll_and_the_cart_that_left_the_view(trees, tmp_path):
    """exp-073b's lf52 actions, sent in snippets: the RIGHT at step 155 scrolls the world by 8 columns; on level 4
    the RIGHT at step 204 takes the cart off the view's right edge."""
    data = json.loads(EXP073B.read_text())
    game = next(g for g in data["game_runs"] if g["game_id"].startswith("lf52"))

    def acts(first: int, last: int) -> str:
        out = []
        for rec in game["history"][first - 1:last]:
            a = rec["action"]
            out.append({"action": "MOUSE", "row": int(a["data"]["y"]), "col": int(a["data"]["x"])}
                       if a["id"] == "ACTION6" else NAMES[a["id"]])
        # a one-action reply has no executed_count; both kinds list executed_actions
        return f"r = action({json.dumps(out)})\nprint(len(r.get('executed_actions') or ()), r.get('level'))"

    out = run_child(trees["ours"], "1", "lf52", [acts(1, 19), acts(20, 118), acts(119, 154), acts(155, 155),
                                                 acts(156, 198), acts(199, 203), acts(204, 204),
                                                 "print(view_offset, left_view)"], tmp_path, OTHERS)
    printed = [json.loads(r)["stdout"].splitlines() for r in out["results"]]
    assert [p[-1] for p in printed[:7]] == ["19 2", "99 3", "36 3", "1 3", "43 4", "5 4", "1 4"]
    assert not any(line.startswith("[view]") for p in printed[:3] for line in p)
    assert printed[3][0] == "[view] scrolled at step 155: content moved (+0,-8); view_offset now (+0,+8)"
    assert printed[6][0] == ("[view] left the view: Y 14px at the right edge (last seen step 203, rows 23-28, cols "
                             "59-63); see left_view")
    assert printed[7] == ["(0, 0) [{'colour': 'Y', 'pixels': 14, 'step': 203, 'box': [23, 59, 28, 63], "
                          "'world_box': [23, 59, 28, 63], 'edge': 'right'}]"]


def _harness_env(**extra: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith("OURS_")
           and k != "EXPOSE_RESET"}
    nb_env, _ = franzen_bed.notebook_env(franzen_bed._cell(franzen_tree.notebook_cells(), "setup_env = {"))
    env.update(nb_env)
    env.update(franzen_bed.BED_CONTEXT)
    env.update({"ARC3_MAX_ACTIVE_STREAMS": "0", "PYTHONHASHSEED": "0", **extra})
    return env


@needs_tree
@needs_bed
@needs_games
@pytest.mark.parametrize("game,others", [("ls20", "on"), ("ls20", "off"), ("vc33", "on")])
def test_flag_off_drive_is_byte_identical(trees, tmp_path, game, others):
    """His real ToolAgent.analyze() over 22 turns of a real game (tests/franzen_ledger_checks.py drive): with the
    flag unset or "0" the requests on the wire, the stored history and the transcript equal the tree without it."""
    flags = dict(OTHERS) if others == "on" else {}
    runs = {"base": ("base", flags), "off": ("ours", flags), "zero": ("ours", {**flags, "OURS_PERCEPTION": "0"})}
    procs = {}
    for name, (tree, extra) in runs.items():
        out = tmp_path / f"{game}-{name}.json"
        procs[name] = (out, subprocess.Popen(
            [str(BED_PY), str(LEDGER_CHECKS), "drive", str(trees[tree]), str(GAMES), str(out), game, "22"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_harness_env(**extra)))
    results = {}
    for name, (out, proc) in procs.items():
        stdout, stderr = proc.communicate(timeout=600)
        assert proc.returncode == 0, stdout[-2000:] + stderr[-4000:]
        results[name] = json.loads(out.read_text())
    base = results["base"]
    assert base["level_ups"] >= 1 and len(base["requests"]) >= 40, "the drive must reach a level-up and trims"
    for name in ("off", "zero"):
        assert [q["sha"] for q in results[name]["requests"]] == [q["sha"] for q in base["requests"]], name
        assert (results[name]["history_sha"], results[name]["transcript_sha"]) == (
            base["history_sha"], base["transcript_sha"]), name


@pytest.mark.slow
@needs_tree
@needs_bed
@needs_games
def test_bed_with_all_eight_patches_and_flags(tmp_path):
    result = franzen_bed.run_bed(tmp_path / "bed", games=["ls20", "vc33", "sb26", "bp35"], seconds=120,
                                 patches=[*SEVEN, PATCH], env_add={**OTHERS, "OURS_PERCEPTION": "1"},
                                 expect="[view] scrolled at step", expect_in="any",
                                 programs=["search", "mem", "effects", "perception"])
    failed = [name for name, ok in result["checks"].items() if not ok]
    assert not failed, (failed, result["facts"])


def test_flag_name_does_not_collide_with_cell_4():
    """--env-add refuses a key cell 4 already sets; an arm adds OURS_PERCEPTION that way."""
    assert "OURS_PERCEPTION" not in franzen_bed._cell(franzen_tree.notebook_cells(), "setup_env = {")
