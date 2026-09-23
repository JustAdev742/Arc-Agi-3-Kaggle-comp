"""Named agent configurations. ``CHAMPION`` is the measured best configuration and the one the submission notebook
runs unless a challenger wins on the fixed evaluation; every knob added since is off in it, so a control run at any
later commit reproduces the champion's behaviour (docs/champion.md). ``BUNDLE`` is the current experimental arm
(exp-019/020/022/028: memory, level boundary, action-budget notice, probe sweep, goal hypotheses with the
universal and colour-free goal kinds).

Precedence when building a notebook or an eval: preset first, then ``--config`` on top.
"""
from __future__ import annotations

from typing import Any

CHAMPION: dict[str, Any] = {
    "context_tokens": 32768,
    "reasoning_effort": "low",        # exp-004 (thinking off) and exp-013/014 (medium) lost or tied; low is the base
    "max_output_tokens": 3072,
    "effort_policy": "adaptive",      # raise to medium for one turn when stagnant (exp-011 harness)
    "tool_events_line": False,        # exp-012/012b lost
    "memory": False,                  # exp-019 arm, unmeasured
    "level_consolidation": False,     # exp-020 arm, unmeasured
    "level_action_notice": 0,         # exp-020 arm, unmeasured
    "explore_first": 0,               # exp-022 arm, unmeasured
    "goal_progress_in_prompt": False,  # exp-022 arm, unmeasured
    "bg_holes": False,                # exp-021b perception knob, unmeasured (enclosed background-coloured islands are entities)
    "noop_memory": False,             # exp-023 arm, unmeasured (known no-ops shown and flagged)
    "noop_skip": False,               # exp-023b arm, unmeasured (known no-ops not sent)
    "postmortem": False,              # research data only (one call at the end of an unsolved game)
    "goal_forall": False,             # exp-028 goal kinds, unmeasured with a model ("every target" relations)
    "goal_lifted": False,             # exp-028 goal kinds, unmeasured with a model (colour-free re-instantiation per level)
}

BUNDLE: dict[str, Any] = {
    **CHAMPION,
    "memory": True,
    "level_consolidation": True,
    "level_action_notice": 120,
    "explore_first": 8,
    "explore_first_clicks": 4,
    "goal_progress_in_prompt": True,
    "bg_holes": True,
    "noop_memory": True,
    "postmortem": True,
    "goal_forall": True,
    "goal_lifted": True,
}

PRESETS: dict[str, dict[str, Any]] = {"champion": CHAMPION, "bundle": BUNDLE}


def resolve(preset: str | None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """The agent config for a preset name ('' or None = no preset) with ``overrides`` applied on top."""
    base = dict(PRESETS[preset]) if preset else {}
    base.update(overrides or {})
    return base
