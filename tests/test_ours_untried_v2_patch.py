"""kaggle/franzen/patches/ours-10v2-untried.patch ("notice v2", OURS_UNTRIED=v2): ours-10's module and call sites, with
the new flag value v2 selecting I1 alone under three changes from docs/research/beat-tufa/notice-response-2026-10-10.md:
at most 3 kinds named, a level's first line at 5 minutes when MOUSE is valid and unused on it, and the first level's
clock started by the game's first action() return instead of its first prompt; no I2. Every other flag value behaves
exactly as ours-10 (docs/research/beat-tufa/patch-untried.md, section "v2").

The module is loaded from the patch text (ours-10's too, for the comparison), so the unit tests need neither Franzen's
repo nor the bed: synthetic boards, explicit clock values, and random games driven through both modules under the old
flag values. The patch applies alone, on exp-083's stack (the sandbox fix) and on exp-084's (sandbox, 02, 04, 03b, 05,
08b), adds lines only, and leaves tool_agent.py exactly as ours-10 does. In the bed venv the real ToolAgent, driven as
in tests/test_ours_untried_patch.py plus a slow-boot schedule, is byte-identical to the tree without the patch with
the flag off, to ours-10's tree under 1/i1/i2, and shows the v2 lines where the rules put them.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import random
import re
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
PATCH = PATCHES / "ours-10v2-untried.patch"
OURS10 = PATCHES / "ours-10-untried.patch"
SANDBOX = PATCHES / "ours-sandbox-timeout-keeps-work.patch"
EXP084 = [PATCHES / name for name in (
    "ours-sandbox-timeout-keeps-work.patch", "ours-02-budget-meter.patch", "ours-04-search-helper.patch",
    "ours-03b-win-ledger-on-02-04.patch", "ours-05-level-mem.patch", "ours-08b-perception-on-01-02-04-03b-05.patch")]
STACKS = {"083": [SANDBOX], "084": EXP084}
BED_PY = Path.home() / ".cache" / "arc3-franzen-bed" / "venv" / "bin" / "python"
NEW_FILE = "ARC3-Inference/inference/utils/ours_untried.py"
SENTENCE = "Each is a one-action test of whether it is interactive."
BOTH, I1, I2 = frozenset({"i1", "i2"}), frozenset({"i1"}), frozenset({"i2"})
V2 = frozenset({"i1", "v2"})
OLD_MODES = (BOTH, I1, I2)

needs_tree = pytest.mark.skipif(not (franzen_tree.his_repo_path() / "ARC3-Inference").is_dir(),
                                reason="needs Franzen's repo (scripts/franzen_tree.py)")
needs_bed = pytest.mark.skipif(not BED_PY.exists(), reason="needs the franzen_bed venv (scripts/franzen_bed.py)")


def _module_source(patch: Path = PATCH) -> str:
    """The new module exactly as the patch adds it."""
    section = patch.read_text().split(f"+++ b/{NEW_FILE}\n", 1)[1].split("\ndiff --git ", 1)[0]
    body = section.split("\n", 1)[1]  # after the @@ line
    return "\n".join(line[1:] for line in body.splitlines() if line.startswith("+")) + "\n"


def _load(patch: Path = PATCH) -> types.ModuleType:
    module = types.ModuleType("ours_untried")
    exec(compile(_module_source(patch), NEW_FILE, "exec"), module.__dict__)
    return module


u = _load()
u10 = _load(OURS10)


# --- synthetic boards (as in tests/test_ours_untried_patch.py) --------------------------------------------------------


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
BUTTON, GATE, OTHER = rect(10, 10, 11, 11), rect(30, 20, 30, 23), rect(40, 40, 42, 42)
START = board((8, BUTTON), (12, GATE), (9, OTHER))
START_KINDS = "no click has hit b r40-42 c40-42, R r10-11 c10-11, O r30 c20-23"  # one copy each: the larger first


def _named(line: str) -> int:
    """How many kinds an I1 line names."""
    part = line.split("no click has hit ", 1)[1].split(". " + SENTENCE)[0].split("; ")[0] if "no click" in line else ""
    return len(part.split(", ")) if part else 0


# --- the flag ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("raw, want", [(None, set()), ("", set()), ("0", set()), ("off", set()), ("i3", set()),
                                       ("v3", set()), ("2", set()), ("v2x", set()),
                                       ("1", {"i1", "i2"}), ("on", {"i1", "i2"}), ("TRUE", {"i1", "i2"}),
                                       ("yes", {"i1", "i2"}), ("i1", {"i1"}), (" I2 ", {"i2"}),
                                       ("v2", {"i1", "v2"}), (" V2 ", {"i1", "v2"})])
def test_v2_is_a_new_flag_value_and_the_old_values_keep_their_meaning(monkeypatch, raw, want):
    if raw is None:
        monkeypatch.delenv("OURS_UNTRIED", raising=False)
    else:
        monkeypatch.setenv("OURS_UNTRIED", raw)
    assert u.mode() == frozenset(want)
    assert u10.mode() == (frozenset() if "v2" in want else u.mode())  # ours-10 reads v2 as off


# --- I1 under v2 ------------------------------------------------------------------------------------------------------


def test_v2_names_at_most_three_kinds_and_counts_the_rest():
    kinds = u.board_kinds(EIGHT_KINDS)
    assert u.i1_block(unused_actions=["UP", "SPACE"], kinds=kinds, prev_kinds=set(u.board_kinds(PREVIOUS)),
                      max_kinds=u.MAX_KINDS_V2) == (
        "Not yet tried on this level: UP, SPACE never used; no click has hit S r10-12 c10-12, O r20 c6-7, r r24 c6; "
        "5 more kinds. " + SENTENCE)
    assert u.MAX_KINDS_V2 == 3 and u.MAX_KINDS == 6
    # through the tracker: the same first-level board names 6 kinds under ours-10's values and 3 under v2
    h = [entry("", EIGHT_KINDS, 0), entry("UP", EIGHT_KINDS, 1)]
    lines = {}
    for name, modes in (("old", BOTH), ("v2", V2)):
        t = u.Tracker()
        t.lines(h[:1], ["UP", "MOUSE"], 0.0, modes)
        t.stamp(1, 0.0)
        lines[name] = t.lines(h, ["UP", "MOUSE"], 600.0, modes)
    assert lines["old"] == ["Not yet tried on this level: MOUSE never used; no click has hit S r10-12 c10-12, N r28 "
                            "c6-8, O r20 c6-7, b r6 c6, r r24 c6, Y r16-17 c6-7 x2; 2 more kinds. " + SENTENCE]
    assert lines["v2"] == ["Not yet tried on this level: MOUSE never used; no click has hit S r10-12 c10-12, N r28 "
                           "c6-8, O r20 c6-7; 5 more kinds. " + SENTENCE]


def _v2_tracker(stamp_at: float = 10.0, valid=("UP", "MOUSE")) -> tuple:
    """A v2 tracker past its first answered call (one UP returning at stamp_at): (tracker, history, valid)."""
    t, valid, h = u.Tracker(), list(valid), [entry("", START, 0)]
    assert t.lines(h, valid, 0.0, V2) == []
    h.append(entry("UP", START, 1))
    t.stamp(1, stamp_at)
    return t, h, valid


def test_v2_first_line_comes_at_five_minutes_when_mouse_is_valid_and_unused_then_every_ten():
    t, h, valid = _v2_tracker()
    assert t.lines(h, valid, 309.0, V2) == []
    [line] = t.lines(h, valid, 310.0, V2)  # 300 s after the level's start (the first stamp, t=10)
    assert line == f"Not yet tried on this level: MOUSE never used; {START_KINDS}. " + SENTENCE
    assert {k: t.fired[-1][k] for k in ("reason", "start", "mark", "max_kinds")} == {
        "reason": "stall", "start": 10.0, "mark": 300.0, "max_kinds": 3}
    assert t.lines(h, valid, 909.0, V2) == []  # repeats as before: 10 minutes after the last line
    assert len(t.lines(h, valid, 910.0, V2)) == 1 and t.fired[-1]["reason"] == "repeat"
    assert t.lines(h, valid, 1509.0, V2) == [] and len(t.lines(h, valid, 1510.0, V2)) == 1


def test_v2_first_line_stays_at_ten_minutes_when_mouse_is_not_valid_or_was_used_on_the_level():
    t, h, valid = _v2_tracker(valid=("UP", "DOWN"))  # no MOUSE in this game
    assert t.lines(h, valid, 609.0, V2) == []
    assert t.lines(h, valid, 610.0, V2) == ["Not yet tried on this level: DOWN never used. " + SENTENCE]
    assert t.fired[-1]["mark"] == 600.0
    t, h, valid = _v2_tracker()
    h.append(entry("MOUSE(row=10, col=10)", START, 2))  # MOUSE used on this level
    t.stamp(2, 20.0)
    assert t.lines(h, valid, 609.0, V2) == []
    assert t.lines(h, valid, 610.0, V2) == [
        "Not yet tried on this level: no click has hit b r40-42 c40-42, O r30 c20-23. " + SENTENCE]
    # MOUSE used on level 1 only: level 2, reached by the call that returned at t=650, gets its line at 5 minutes
    h.append(entry("DOWN", board((9, OTHER), (13, GATE)), 3, level=2, level_completed=True))
    t.stamp(3, 650.0)
    assert t.lines(h, valid, 949.0, V2) == []
    assert t.lines(h, valid, 950.0, V2) == [  # r is new on this level, so it comes first
        "Not yet tried on this level: UP, MOUSE never used; no click has hit r r30 c20-23, b r40-42 c40-42. "
        + SENTENCE]
    assert t.levels[2].start == 650.0 and not t.levels[2].before_reply  # later levels: the action that reached them


def test_v2_keeps_the_game_over_rule_at_the_ten_minute_mark():
    t, h, valid = _v2_tracker()
    assert len(t.lines(h, valid, 310.0, V2)) == 1  # the 5-minute line
    h += [entry("UP", START, 2, game_over=True), entry("RESET", START, 3, automatic=True)]
    t.stamp(2, 420.0)
    assert t.lines(h, valid, 430.0, V2) == []  # a GAME OVER before the 10-minute mark fires nothing ...
    assert t.lines(h, valid, 909.0, V2) == []  # ... and is not kept
    assert len(t.lines(h, valid, 910.0, V2)) == 1  # the repeat, 10 minutes after the last line
    h += [entry("UP", START, 4, game_over=True), entry("RESET", START, 5, automatic=True)]
    t.stamp(4, 950.0)
    [line] = t.lines(h, valid, 960.0, V2)  # past the mark: at once, 50 s after the last line
    assert line == f"Not yet tried on this level: MOUSE never used; {START_KINDS}. " + SENTENCE
    assert [f["reason"] for f in t.fired] == ["stall", "repeat", "game over"]


def test_v2_level_one_clock_starts_at_the_first_answered_call_not_at_the_first_prompt():
    """exp-085r: a slow server boot counted toward level 1, and 13 lines fired at step 1-2 of level 1."""
    valid = ["UP", "DOWN", "MOUSE"]
    h0 = [entry("", START, 0)]
    h1 = [*h0, entry("UP", START, 1)]
    h2 = [*h1, entry("DOWN", START, 2)]
    shown = {}
    for name, modes in (("old", BOTH), ("v2", V2)):
        t = u.Tracker()
        got = [t.lines(h0, valid, 0.0, modes)]  # the first prompt; its request waits 13 minutes for the server
        t.stamp(1, 795.0)  # the first reply's action() call returns
        got.append(t.lines(h1, valid, 800.0, modes))  # step 1
        t.stamp(2, 830.0)
        got.append(t.lines(h2, valid, 840.0, modes))  # step 2
        got.append(t.lines(h2, valid, 1094.0, modes))
        got.append(t.lines(h2, valid, 1095.0, modes))  # 300 s after the first answered call
        shown[name] = got
        assert t.levels[1].start == 0.0 and t.levels[1].before_reply  # ours-10's start is kept for its values
        assert t.level_start(t.levels[1], v2=True) == 795.0 and t.level_start(t.levels[1]) == 0.0
    assert shown["old"] == [[], [f"Not yet tried on this level: DOWN, MOUSE never used; {START_KINDS}. " + SENTENCE],
                            [], [], []]  # ours-10: at step 1, "800 s into the level"
    assert shown["v2"] == [[], [], [], [], [f"Not yet tried on this level: MOUSE never used; {START_KINDS}. " + SENTENCE]]


def test_v2_has_no_level_clock_before_any_action_call_returned():
    t, h = u.Tracker(), [entry("", START, 0)]
    assert t.lines(h, ["UP", "MOUSE"], 0.0, V2) == []
    assert t.level_start(t.levels[1], v2=True) is None
    assert t.lines(h, ["UP", "MOUSE"], 5000.0, V2) == []  # replies that never acted: nothing to say yet
    t.stamp(2, 5000.0)
    t.stamp(1, 4000.0)  # not a later call: ignored, as before
    assert t.level_start(t.levels[1], v2=True) == 5000.0
    assert t.lines(h, ["UP", "MOUSE"], 5299.0, V2) == [] and len(t.lines(h, ["UP", "MOUSE"], 5300.0, V2)) == 1
    old = u.Tracker()
    old.lines(h, ["UP", "MOUSE"], 0.0, BOTH)
    assert len(old.lines(h, ["UP", "MOUSE"], 5000.0, BOTH)) == 1  # ours-10's values: from the first prompt


def test_v2_shows_no_i2():
    after = board((8, BUTTON), (14, GATE), (9, OTHER))  # the click on the button turns the gate N, new to the game
    h = [entry("", START, 0), entry("MOUSE(row=10, col=10)", after, 1)]
    shown, fired = {}, {}
    for name, modes in (("old", BOTH), ("v2", V2)):
        t = u.Tracker()
        t.lines(h[:1], ["MOUSE"], 0.0, modes)
        t.stamp(1, 5.0)
        shown[name] = t.lines(h, ["MOUSE"], 10.0, modes)
        fired[name] = [f["line"] for f in t.fired]
    assert shown["old"] == ["After MOUSE(row=10, col=10), O r30 c20-23 turned N, a colour new to this game; no click "
                            "has hit it since."]
    assert shown["v2"] == [] and fired == {"old": ["i2"], "v2": []}


# --- the old flag values are ours-10's: random games through both modules --------------------------------------------

KEYS = ("UP", "DOWN", "LEFT", "RIGHT", "SPACE", "UNDO", "ACTION1", "ACTION5", "RESET")
VALID_SETS = (["ACTION1", "ACTION2", "ACTION6"], ["UP", "DOWN", "LEFT", "RIGHT"], ["MOUSE", "UNDO", "RESET"],
              ["ACTION1", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"], ["SPACE", "MOUSE"], [])


def random_game(seed: int, grid: bool = False) -> tuple:
    """(entries, ops): a game's history and the harness's calls on it, ("stamp", action_num, t) after an action()
    call and ("prompt", entries seen, valid actions, t) for a user prompt. Clicks mostly land on objects; boards
    gain colours, levels change, GAME OVERs come with the solver's automatic RESET; some stamps are missing. grid:
    every time a multiple of 100 s, so the 300/600 s marks are hit exactly."""
    rng = random.Random(seed)
    call_gaps = (100.0, 200.0, 300.0, 600.0) if grid else (2.0, 30.0, 120.0, 290.0, 310.0, 590.0, 610.0, 900.0)
    prompt_gaps = (0.0, 100.0, 300.0) if grid else (0.5, 5.0, 100.0, 400.0)
    size = rng.choice((12, 16, 24))

    def paint(grid, palette, n):
        g = [list(row) for row in grid]
        for _ in range(n):
            colour, r0, c0 = rng.choice(palette), rng.randrange(size), rng.randrange(size)
            for r in range(r0, min(size, r0 + rng.randint(1, 4))):
                for c in range(c0, min(size, c0 + rng.randint(1, 4))):
                    g[r][c] = colour
        return tuple(tuple(row) for row in g)

    blank = tuple((0,) * size for _ in range(size))
    grid, level, step, t = paint(blank, range(1, 7), rng.randint(3, 9)), 1, 0, 0.0
    entries = [entry("", grid, 0, level)]
    ops = [("prompt", 1, rng.choice(VALID_SETS), t)]
    for _call in range(rng.randint(6, 18)):
        t += rng.choice(call_gaps)
        reset = False
        for _ in range(rng.randint(1, 3)):
            step += 1
            cells = [(r, c) for r in range(size) for c in range(size) if grid[r][c]]
            if rng.random() < 0.5:
                r, c = rng.choice(cells) if cells and rng.random() < 0.8 else divmod(rng.randrange(size * size), size)
                action = f"MOUSE(row={r}, col={c})"
            else:
                action = rng.choice(KEYS)
            roll, result = rng.random(), {}
            if roll < 0.1:
                level += 1
                grid, result = paint(blank, range(1, 16), rng.randint(3, 9)), {"level_completed": True}
            elif roll < 0.18:
                result, reset = {"game_over": True}, True
            elif roll < 0.6:
                grid = paint(grid, range(1, 16) if rng.random() < 0.3 else range(1, 7), rng.randint(1, 2))
            entries.append(entry(action, grid, step, level, **result))
            if result:
                break
        if rng.random() < 0.9:
            ops.append(("stamp", step, t))
        if reset:  # the solver's RESET between turns: not part of the call
            step += 1
            entries.append(entry("RESET", grid, step, level, automatic=True))
        for _ in range(rng.choice((0, 1, 1, 1, 2))):
            t += rng.choice(prompt_gaps)
            ops.append(("prompt", len(entries), rng.choice(VALID_SETS), t))
    return entries, ops


GAMES = [(seed, False) for seed in range(60)] + [(seed, True) for seed in range(40)]


def _plain(x):
    """Records and state in a form two modules' objects compare in."""
    if hasattr(x, "cells") and hasattr(x, "box"):
        return ("obj", x.colour, x.box, x.px, tuple(sorted(x.cells)))
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    return x


def _drive(module, entries, ops, modes) -> tuple:
    t = module.Tracker()
    shown = []
    for op in ops:
        if op[0] == "stamp":
            t.stamp(op[1], op[2])
        else:
            shown.append(t.lines(entries[:op[1]], op[2], op[3], modes))
    levels = {n: (lv.number, lv.start, lv.last_i1, lv.game_over, sorted(lv.used), sorted(lv.clicked))
              for n, lv in t.levels.items()}
    state = (t.n, t.stamps, sorted(t.seen), t.level, levels, _plain(t.pending), t.error)
    return shown, _plain(t.fired), state


def test_old_flag_values_behave_exactly_as_ours10_on_random_games():
    reasons, i2 = {}, 0
    for seed, grid in GAMES:
        entries, ops = random_game(seed, grid)
        for modes in OLD_MODES:
            ours10 = _drive(u10, entries, ops, modes)
            assert _drive(u, entries, ops, modes) == ours10, (seed, grid, sorted(modes))
            if modes == BOTH:
                for rec in ours10[1]:
                    reasons[rec.get("reason", "i2")] = reasons.get(rec.get("reason", "i2"), 0) + 1
                    i2 += rec["line"] == "i2"
    # the games reach every trigger (100+ times each), so the comparison is not vacuous; on the 100 s grid they hit
    # the marks exactly, which is what catches a >= turned into >
    assert min(reasons.get(k, 0) for k in ("stall", "repeat", "game over")) >= 50 and i2 >= 50, reasons


def test_v2_on_random_games_keeps_its_rules():
    lines = 0
    for seed, grid in GAMES:
        entries, ops = random_game(seed, grid)
        t, first_stamp = u.Tracker(), None
        for op in ops:
            if op[0] == "stamp":
                t.stamp(op[1], op[2])
                first_stamp = first_stamp if first_stamp is not None else op[2]
                continue
            got = t.lines(entries[:op[1]], op[2], op[3], V2)
            lines += len(got)
            assert all(line.startswith("Not yet tried on this level: ") for line in got)  # no I2
            assert all(_named(line) <= 3 for line in got)
            assert not got or first_stamp is not None  # no line before an action() call returned
        for rec in t.fired:
            assert rec["time"] - rec["start"] >= (rec["mark"] if rec["reason"] == "stall" else u.STALL_SECONDS)
            if rec["mark"] == u.MOUSE_STALL_SECONDS:  # the 5-minute mark only with MOUSE valid and unused
                assert "MOUSE" in rec["unused"]
    assert lines >= 50


# --- harness glue -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("flag", [None, "", "0", "v3", "1", "on", "i1", "i2"])
def test_the_glue_is_ours10s_for_every_other_flag_value(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv("OURS_UNTRIED", raising=False)
    else:
        monkeypatch.setenv("OURS_UNTRIED", flag)
    entries, ops = random_game(7)
    out = []
    for module in (u10, u):
        now = [0.0]
        monkeypatch.setattr(module, "_clock", lambda now=now: now[0])
        agent, shown = types.SimpleNamespace(_session_runtime_dir="/runs/g1"), []
        for op in ops:
            now[0] = op[-1]
            if op[0] == "stamp":
                module.stamp(agent, {"executed": True, "action_num": op[1]})
            else:
                shown.append(module.prompt_lines(agent, entries[:op[1]], op[2]))
        state = getattr(agent, "_ours_untried", None)
        out.append((shown, None if state is None else _plain(state[1].fired)))
    assert out[0] == out[1]
    assert (out[0][1] is None) == (flag in (None, "", "0", "v3"))


def test_the_glue_does_nothing_with_the_flag_off_and_ours10_ignores_v2(monkeypatch):
    monkeypatch.delenv("OURS_UNTRIED", raising=False)
    agent = types.SimpleNamespace(_session_runtime_dir="/runs/g1")
    u.stamp(agent, {"executed": True, "action_num": 3})
    assert u.prompt_lines(agent, [entry("", START, 0)], ["UP"]) == []
    assert not hasattr(agent, "_ours_untried")
    monkeypatch.setenv("OURS_UNTRIED", "v2")
    u10.stamp(agent, {"executed": True, "action_num": 3})
    assert u10.prompt_lines(agent, [entry("", START, 0)], ["UP"]) == []
    assert not hasattr(agent, "_ours_untried")


def test_v2_glue_starts_the_clock_at_the_first_stamp_and_turns_itself_off_on_an_error(monkeypatch, caplog):
    monkeypatch.setenv("OURS_UNTRIED", "v2")
    now = [0.0]
    monkeypatch.setattr(u, "_clock", lambda: now[0])
    agent, h = types.SimpleNamespace(_session_runtime_dir="/runs/g1"), [entry("", START, 0)]
    assert u.prompt_lines(agent, h, ["UP", "MOUSE"]) == []
    now[0] = 780.0
    u.stamp(agent, {"executed": False, "action_num": 1})  # a refused call ran nothing and starts nothing
    assert u.prompt_lines(agent, h, ["UP", "MOUSE"]) == []
    u.stamp(agent, {"executed": True, "action_num": 1})
    h = [*h, entry("UP", START, 1)]
    now[0] = 1079.0
    assert u.prompt_lines(agent, h, ["UP", "MOUSE"]) == []
    now[0] = 1080.0
    assert u.prompt_lines(agent, h, ["UP", "MOUSE"]) == [
        f"Not yet tried on this level: MOUSE never used; {START_KINDS}. " + SENTENCE]
    agent._session_runtime_dir = "/runs/g2"  # the next game: a new tracker, no clock until its first call returns
    assert u.prompt_lines(agent, h[:1], ["UP", "MOUSE"]) == [] and agent._ours_untried[1].stamps == []
    with caplog.at_level(logging.WARNING):
        u.stamp(agent, {"executed": True, "action_num": "x"})  # logged, never raised
        real = u.Tracker.lines
        monkeypatch.setattr(u.Tracker, "lines", lambda *a, **k: 1 / 0)
        assert u.prompt_lines(agent, h, ["UP", "MOUSE"]) == []
        monkeypatch.setattr(u.Tracker, "lines", real)
    assert agent._ours_untried[1].error == "internal error (ZeroDivisionError)"
    now[0] = 9000.0
    assert u.prompt_lines(agent, h, ["UP", "MOUSE"]) == []  # off for the rest of this game
    assert "stamp failed" in caplog.text and "lines off for this game" in caplog.text


# --- the patch --------------------------------------------------------------------------------------------------------


def test_the_patch_is_ours10s_hooks_plus_the_module_and_only_adds_lines():
    old, new = OURS10.read_text(), PATCH.read_text()
    cut = f"diff --git a/{NEW_FILE} b/{NEW_FILE}\n"
    assert new.split(cut)[0] == old.split(cut)[0]  # tool_agent.py: the same two hooks, byte for byte
    assert [line for line in new.splitlines() if line.startswith("-") and not line.startswith("---")] == []
    assert re.findall(r"^\+\+\+ b/(\S+)", new, re.M) == re.findall(r"^\+\+\+ b/(\S+)", old, re.M)
    tree = ast.parse(_module_source())
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imported == {"__future__", "bisect", "logging", "os", "re", "time", "collections"}  # stdlib only


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    out = {}
    for name, patches in (("base", []), ("v2", [PATCH]), *[(f"base{s}", p) for s, p in STACKS.items()],
                          *[(f"{s}v2", [*p, PATCH]) for s, p in STACKS.items()],
                          *[(f"{s}o10", [*p, OURS10]) for s, p in STACKS.items()]):
        dest = tmp_path_factory.mktemp(name)
        logs = franzen_tree.notebook_bundle(dest, patches)
        assert [p.name for p in patches] == [k for k in logs if k != "his"]
        out[name] = dest / "src" / "ARC3-Inference"
    return out


@needs_tree
@pytest.mark.parametrize("stack", ["", "083", "084"])
def test_the_patch_applies_alone_and_on_both_stacks_and_only_adds_lines(trees, stack):
    base, ours = trees[f"base{stack}"], trees[f"{stack}v2"]
    before = (base / "inference/agent/tool_agent.py").read_text().splitlines()
    after = (ours / "inference/agent/tool_agent.py").read_text().splitlines()
    before_set, after_set = set(before), set(after)
    assert [line for line in before if line not in after_set] == []  # nothing removed or changed
    added = [line for line in after if line not in before_set]
    plus = {line[1:] for line in PATCH.read_text().splitlines() if line.startswith("+") and not line.startswith("+++")}
    assert len(added) == 6 and set(added) <= plus
    assert (ours / "inference/utils/ours_untried.py").read_text() == _module_source()
    text = "\n".join(after)
    hook = text.index("_ours_untried_lines(self, history_entries")
    assert text.index("self._death_ledger_lines(", hook - 2000) < hook < text.index("Only tool: `python`", hook)
    stamp = text.index("_ours_untried_stamp(self, raw_payload)")
    assert 0 < stamp - text.index("raw_payload = self._step_env_callback(step_arguments)") < 600
    if stack:  # on the candidate stacks, tool_agent.py is exactly ours-10's
        assert (ours / "inference/agent/tool_agent.py").read_bytes() == (
            trees[f"{stack}o10"] / "inference/agent/tool_agent.py").read_bytes()


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

    def board(level, l_colour, hud, dead, extra):
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
            if extra:  # two more kinds (5 in all), for the v2 cap: a yellow dot and a pink pair
                g[6][18] = 11
                g[19][4] = g[19][5] = 15
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
        def __init__(self, path, extra):
            self.path, self.step, self.level, self.l_colour, self.hud, self.extra = path, 0, 1, 12, 12, extra
            self.history = [HistoryEntry(action="", frame=self.frame())]
            self.write()
        def frame(self, dead=False):
            return Frame(grid=board(self.level, self.l_colour, self.hud, dead, self.extra), step=self.step,
                         level=self.level)
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

    def scenario(extra, schedule):
        game = Game(Path(tempfile.mkdtemp()) / f"g_{RUNTIME_STATE_FILENAME}", extra)
        agent = ta.ToolAgent(model="mock", base_url="http://127.0.0.1:9/v1")
        out = {"system": agent._system_prompt, "prompts": [], "tools": []}
        for what, at, *code in schedule:
            NOW[0] = at
            if what == "prompt":
                agent._ensure_session(game.path)  # as analyze() does
                agent._step_env_callback = game
                agent._current_valid_actions = ta._normalize_valid_actions(VALID)
                frame, history = load_runtime_state(game.path)
                out["prompts"].append(agent._build_user_prompt(game.step, valid_actions=VALID, current_frame=frame,
                                                               history_entries=history,
                                                               previous_step_summary=agent._last_step_summary))
            elif what == "tool":
                out["tools"].append(agent._run_python_tool(game.path, {"code": code[0]}).content)
            else:
                game.auto_reset()
        state = getattr(agent, "_ours_untried", None)
        out["fired"] = None if state is None else [[f["line"], f.get("reason"), f.get("start"), f.get("mark")]
                                                   for f in state[1].fired]
        out["stamps"] = None if state is None else state[1].stamps
        return out

    STANDARD = [("prompt", 1000.0), ("tool", 1030.0, 'action({"action": "MOUSE", "row": 10, "col": 5})'),
                ("prompt", 1040.0), ("tool", 1100.0, 'action("RIGHT")'), ("prompt", 1600.0),
                ("tool", 1610.0, 'action("DOWN")'), ("prompt", 2205.0), ("prompt", 2210.0),
                ("tool", 2300.0, 'action("LEFT")'), ("reset", 2300.0), ("prompt", 2310.0)]
    # a 13-minute server boot: the first prompt's request is answered, and its action() returns, at t=1790
    BOOT = [("prompt", 1000.0), ("tool", 1790.0, 'action("RIGHT")'), ("prompt", 1800.0),
            ("tool", 1810.0, 'action("UP")'), ("prompt", 1820.0), ("prompt", 2089.0), ("prompt", 2090.0)]
    print(json.dumps({"standard": scenario(False, STANDARD), "boot": scenario(True, BOOT)}))
''')

# ours-10's lines in the standard schedule (tests/test_ours_untried_patch.py)
I2_P2 = "After MOUSE(row=10, col=5), O r15-17 c15-16 turned N, a colour new to this game; no click has hit it since."
I1_P3 = ("Not yet tried on this level: UP, DOWN, LEFT never used; no click has hit N r15-17 c15-16, b r2-3 c2-3 x2. "
         + SENTENCE)
I1_P5 = ("Not yet tried on this level: UP, DOWN, LEFT, RIGHT, MOUSE never used; no click has hit Y r12-14 c12-14, "
         "b r2-3 c2-3, R r5-8 c20. " + SENTENCE)
I1_P6 = ("Not yet tried on this level: UP, DOWN, RIGHT, MOUSE never used; no click has hit Y r12-14 c12-14, b r2-3 c2-3, "
         "R r5-8 c20. " + SENTENCE)
# the slow-boot schedule: ours-10 fires at step 1 (800 s after the first prompt), v2 300 s after the first answer
OLD_BOOT = ("Not yet tried on this level: UP, DOWN, LEFT, MOUSE never used; no click has hit R r10 c4-7, O r15-17 c15-16, "
            "p r19 c4-5, Y r6 c18, b r2-3 c2-3 x2. " + SENTENCE)
V2_BOOT = ("Not yet tried on this level: DOWN, LEFT, MOUSE never used; no click has hit R r10 c4-7, O r15-17 c15-16, "
           "p r19 c4-5; 2 more kinds. " + SENTENCE)
ANCHOR = "Only tool: `python`."
RUNS = (("083", "base", None), ("083", "base", "v2"), ("083", "o10", "1"), ("083", "o10", "i1"), ("083", "o10", "i2"),
        ("083", "o10", "v2"), ("083", "v2", None), ("083", "v2", "0"), ("083", "v2", "1"), ("083", "v2", "i1"),
        ("083", "v2", "i2"), ("083", "v2", "v2"), ("084", "base", None), ("084", "o10", "1"), ("084", "v2", None),
        ("084", "v2", "1"), ("084", "v2", "v2"))


def _clean_env() -> dict[str, str]:
    """This process's environment without anything that steers the harness."""
    prefixes = ("ARC3_", "LOCAL_ANALYZER_", "OURS_", "OPENAI_", "INFERENCE_", "MULTIMODAL_", "EXPOSE_")
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith(prefixes)}


@pytest.fixture(scope="module")
def bed(trees, tmp_path_factory):
    """Both schedules under the notebook's harness environment, per (stack, tree, OURS_UNTRIED)."""
    work = tmp_path_factory.mktemp("driver")
    script = work / "drive.py"
    script.write_text(DRIVER)
    notebook, _ = franzen_bed.notebook_env(franzen_bed._cell(franzen_tree.notebook_cells(), "setup_env = {"))
    out = {}
    for stack, tree, flag in RUNS:
        src = trees[f"base{stack}" if tree == "base" else f"{stack}{tree}"]
        env = {**_clean_env(), **notebook, **({} if flag is None else {"OURS_UNTRIED": flag})}
        r = subprocess.run([str(BED_PY), "-I", str(script), str(src)], capture_output=True, text=True, env=env,
                           cwd=work, timeout=300, check=False)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
        out[stack, tree, flag] = json.loads(r.stdout.strip().splitlines()[-1])
    return out


def _with(prompt: str, *lines: str) -> str:
    assert prompt.count(ANCHOR) == 1
    return prompt.replace(ANCHOR, "\n".join(lines) + "\n" + ANCHOR, 1) if lines else prompt


def _strip(run: dict) -> dict:
    return {s: {k: v for k, v in run[s].items() if k not in ("fired", "stamps")} for s in run}


@needs_tree
@needs_bed
@pytest.mark.parametrize("stack", ["083", "084"])
def test_flag_off_the_real_agent_is_byte_identical_to_the_tree_without_the_patch(bed, stack):
    base = bed[stack, "base", None]
    assert _strip(bed[stack, "v2", None]) == _strip(base)
    assert all(bed[stack, "v2", None][s]["fired"] is None for s in ("standard", "boot"))  # nothing kept on the agent
    if stack == "083":
        assert bed[stack, "v2", "0"] == bed[stack, "v2", None]
        assert bed[stack, "base", "v2"] == base  # the flag alone does nothing without the patch
        assert bed[stack, "o10", "v2"] == bed[stack, "v2", None]  # and ours-10 reads v2 as off
    assert [len(base[s]["prompts"]) for s in ("standard", "boot")] == [6, 5]
    assert all("Not yet tried" not in p for s in ("standard", "boot") for p in base[s]["prompts"])


@needs_tree
@needs_bed
@pytest.mark.parametrize("stack, flag", [("083", "1"), ("083", "i1"), ("083", "i2"), ("084", "1")])
def test_the_old_flag_values_give_ours10s_prompts_on_the_real_agent(bed, stack, flag):
    assert bed[stack, "v2", flag] == bed[stack, "o10", flag]
    if flag == "1":
        base, on = bed[stack, "base", None], bed[stack, "v2", flag]
        for s, want in (("standard", [[], [I2_P2], [I1_P3], [], [I1_P5], [I1_P6]]),
                        ("boot", [[], [OLD_BOOT], [], [], []])):
            assert on[s]["prompts"] == [_with(p, *lines) for p, lines in zip(base[s]["prompts"], want, strict=True)]


@needs_tree
@needs_bed
@pytest.mark.parametrize("stack", ["083", "084"])
def test_v2_on_the_real_agent(bed, stack):
    base, on = bed[stack, "base", None], bed[stack, "v2", "v2"]
    for s in ("standard", "boot"):
        assert on[s]["system"] == base[s]["system"] and on[s]["tools"] == base[s]["tools"]  # actions as before
        assert not any("a colour new to this game" in p for p in on[s]["prompts"])  # no I2
    # standard: nothing on level 1 at t=1600 (its clock began at the first call's return, t=1030, and MOUSE was used:
    # the 10-minute mark), level 2's line at 595 s (MOUSE unused there), no repeat at 600 s, the GAME OVER past the mark
    want = [[], [], [], [I1_P5], [], [I1_P6]]
    assert on["standard"]["prompts"] == [_with(p, *lines) for p, lines in
                                         zip(base["standard"]["prompts"], want, strict=True)]
    assert on["standard"]["stamps"] == [[1, 1030.0], [2, 1100.0], [3, 1610.0], [4, 2300.0]]
    assert on["standard"]["fired"] == [["i1", "stall", 1610.0, 300.0], ["i1", "game over", 1610.0, 600.0]]
    # slow boot: no line at step 1-2; the first comes 300 s after the first answered call, naming 3 of 5 kinds
    want = [[], [], [], [], [V2_BOOT]]
    assert on["boot"]["prompts"] == [_with(p, *lines) for p, lines in zip(base["boot"]["prompts"], want, strict=True)]
    assert on["boot"]["stamps"] == [[1, 1790.0], [2, 1810.0]]
    assert on["boot"]["fired"] == [["i1", "stall", 1790.0, 300.0]]
