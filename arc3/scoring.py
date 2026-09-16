"""Pure-python re-implementation of the toolkit's RHAE scorer.

Ground truth is ``arc_agi.scorecard.EnvironmentScoreCalculator`` (arc-agi 0.9.9);
``tests/test_scoring.py`` checks parity against it. This module exists so agents
and the governor can ask "what is this level worth?" without touching the toolkit.

Per level:   S = min((baseline / actions) ** 2 * 100, 115)     (0 if not completed)
Per game:    E = min(sum(w_l * S_l) / sum(w_l),  sum(w_l for completed) / sum(w_l) * 100)
             with w_l = l (1-indexed level number)
Total:       mean of E over games
RESET counts as one action (toolkit ``Card.inc_reset_count``); the per-level action
count is the number of actions between the previous level completion and this one.
"""
from __future__ import annotations

from typing import Iterable, Sequence

LEVEL_CAP = 115.0


def level_score(baseline: int, actions: int, completed: bool = True) -> float:
    """Score for one level in percent (0..115)."""
    if not completed or actions <= 0:
        return 0.0
    return min((baseline / actions) ** 2 * 100.0, LEVEL_CAP)


def game_score(level_actions: Sequence[int | None], baselines: Sequence[int]) -> float:
    """Score for one game in percent.

    ``level_actions[i]`` is the number of actions spent on level ``i`` (0-indexed) if it
    was completed, else ``None`` (uncompleted levels score 0 and still carry weight).
    Levels beyond ``len(level_actions)`` are treated as not completed.
    """
    n = len(baselines)
    if n == 0:
        return 0.0
    total = 0.0
    total_w = 0
    completed_w = 0
    for i in range(n):
        w = i + 1
        acts = level_actions[i] if i < len(level_actions) else None
        s = level_score(baselines[i], acts if acts is not None else 0, completed=acts is not None)
        total += w * s
        total_w += w
        if s > 0:
            completed_w += w
    score = total / total_w
    return min(score, completed_w / total_w * 100.0)


def total_score(game_scores: Iterable[float]) -> float:
    scores = list(game_scores)
    return sum(scores) / len(scores) if scores else 0.0


def actions_for_score(baseline: int, target_pct: float) -> int:
    """Largest action count that still scores at least ``target_pct`` on a level."""
    if target_pct <= 0:
        return 10**9
    import math

    return int(math.floor(baseline / math.sqrt(target_pct / 100.0)))
