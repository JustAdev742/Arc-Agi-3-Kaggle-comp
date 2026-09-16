"""Parity of arc3.scoring with the toolkit's EnvironmentScoreCalculator (ground truth)."""
import random

import pytest
from arc_agi.scorecard import EnvironmentScoreCalculator

from arc3 import scoring


def toolkit_game_score(level_actions, baselines):
    calc = EnvironmentScoreCalculator()
    for i, b in enumerate(baselines):
        acts = level_actions[i] if i < len(level_actions) else None
        calc.add_level(level_index=i + 1, completed=acts is not None, actions_taken=acts or 0, baseline_actions=b)
    return calc.to_score().score


@pytest.mark.parametrize("baseline,actions,expected", [
    (20, 20, 100.0), (20, 40, 25.0), (10, 100, 1.0), (20, 2, 115.0), (20, 18, 115.0), (20, 19, 110.80332409972299),
])
def test_level_score(baseline, actions, expected):
    assert scoring.level_score(baseline, actions) == pytest.approx(expected)


def test_level_not_completed_is_zero():
    assert scoring.level_score(20, 5, completed=False) == 0.0
    assert scoring.level_score(20, 0) == 0.0


def test_game_score_matches_toolkit_random():
    rng = random.Random(7)
    for _ in range(500):
        n = rng.randint(1, 10)
        baselines = [rng.randint(5, 300) for _ in range(n)]
        k = rng.randint(0, n)
        level_actions = [rng.randint(1, 600) for _ in range(k)]
        ours = scoring.game_score(level_actions, baselines)
        theirs = toolkit_game_score(level_actions, baselines)
        assert ours == pytest.approx(theirs), (level_actions, baselines)


def test_environment_cap_limits_overachievers():
    # Level 1 done at 115% but level 2 not done: cap = 1/3 * 100
    assert scoring.game_score([1], [20, 20]) == pytest.approx(100.0 / 3)


def test_actions_for_score():
    assert scoring.actions_for_score(20, 100.0) == 20
    assert scoring.actions_for_score(20, 25.0) == 40
    assert scoring.level_score(20, scoring.actions_for_score(20, 50.0)) >= 50.0
