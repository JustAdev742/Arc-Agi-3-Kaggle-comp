"""Regression guard for the rule fitter on real public games (blind 60-action probe, level 1).

These numbers come from runs/rule-coverage; a change to the tracker or the DSL that lowers them is a regression
(the probe is deterministic: fixed policy, fixed game seeds).
"""
from pathlib import Path

import pytest

from arc3.env import make_arcade
from scripts.rule_coverage import run_game

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "environment_files" / "ar25").exists(), reason="game files not downloaded")


@pytest.fixture(scope="module")
def arc():
    return make_arcade(ROOT / "environment_files")


@pytest.mark.parametrize("game,floor", [("ar25", 1.0), ("m0r0", 1.0), ("ft09", 1.0), ("re86", 0.9), ("ls20", 0.7), ("dc22", 0.7)])
def test_probe_coverage_floor(arc, game, floor):
    r = run_game(arc, game, 60)
    assert r["coverage"] is not None and r["coverage"] >= floor, (game, r["coverage"], r["levels"][0]["unexplained_sample"])
    lv = r["levels"][0]
    assert lv["rules"], (game, "no rules fitted")
    assert all("contradict" not in x for x in lv["rules"])


def test_ar25_mirror_twin_rules(arc):
    r = run_game(arc, "ar25", 40)
    rules = r["levels"][0]["rules"]
    moves = [x for x in rules if x.startswith("move[")]
    assert len(moves) == 2, rules  # the avatar and its horizontally mirrored twin
    assert any("LEFT:(+3,+0)" in x for x in moves) and any("LEFT:(-3,+0)" in x for x in moves)
