"""Thin local environment wrapper over the ``arc-agi`` toolkit.

Mirrors the Kaggle gateway's competition-mode semantics (arc_agi/api.py):
  * one scorecard per (game, play) so the toolkit computes RHAE for us;
  * the first RESET (inside ``make``) registers the single play;
  * a RESET sent when the level has just started (engine ``_action_count == 0``) is
    NOT stepped (it would be a full game reset locally); like the gateway we only
    update the scorecard, so it still costs one action;
  * any other RESET is a level reset; every action, RESET included, costs one.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import FrameDataRaw, GameAction, GameState  # noqa: E402


log = logging.getLogger(__name__)

SIMPLE_ACTIONS = (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                  GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION7)


@dataclass
class Frame:
    """One observation. ``grid`` is the last rendered layer (64x64 int16)."""

    grid: np.ndarray
    layers: list[np.ndarray]
    state: GameState
    levels_completed: int
    win_levels: int
    available_actions: list[int]
    full_reset: bool = False
    game_id: str = ""
    step: int = 0  # actions taken so far in this play (RESET included)
    level_step: int = 0  # actions since the current level started
    t: float = field(default_factory=time.time)

    @property
    def level(self) -> int:
        return self.levels_completed + 1

    @property
    def done(self) -> bool:
        return self.state is GameState.WIN

    @property
    def game_over(self) -> bool:
        return self.state is GameState.GAME_OVER

    def actions(self) -> list[GameAction]:
        return [GameAction.from_id(a) for a in self.available_actions]

    @classmethod
    def from_raw(cls, raw: FrameDataRaw, *, step: int = 0, level_step: int = 0) -> "Frame":
        layers = [np.asarray(l, dtype=np.int16) for l in raw.frame] if raw.frame else [np.zeros((64, 64), np.int16)]
        return cls(
            grid=layers[-1], layers=layers, state=raw.state,
            levels_completed=int(raw.levels_completed), win_levels=int(raw.win_levels),
            available_actions=list(raw.available_actions or []), full_reset=bool(raw.full_reset),
            game_id=raw.game_id or "", step=step, level_step=level_step,
        )


@dataclass
class Action:
    action: GameAction
    x: Optional[int] = None
    y: Optional[int] = None
    reasoning: Any = None

    @property
    def data(self) -> dict[str, int]:
        if self.action.is_complex():
            return {"x": int(self.x or 0), "y": int(self.y or 0)}
        return {}

    def key(self) -> tuple:
        return (self.action.value, self.x, self.y) if self.action.is_complex() else (self.action.value,)

    def __str__(self) -> str:
        return f"{self.action.name}({self.x},{self.y})" if self.action.is_complex() else self.action.name

    @classmethod
    def reset(cls) -> "Action":
        return cls(GameAction.RESET)

    @classmethod
    def click(cls, x: int, y: int, reasoning: Any = None) -> "Action":
        return cls(GameAction.ACTION6, int(x), int(y), reasoning)

    @classmethod
    def simple(cls, n: int, reasoning: Any = None) -> "Action":
        return cls(GameAction.from_id(n), reasoning=reasoning)


def make_arcade(environments_dir: str | Path = "environment_files", *, download: bool = False) -> arc_agi.Arcade:
    """Offline arcade over the cached game sources (``make games`` downloads them)."""
    mode = OperationMode.NORMAL if download else OperationMode.OFFLINE
    return arc_agi.Arcade(operation_mode=mode, environments_dir=str(environments_dir), logger=log)


def baselines_for(arc: arc_agi.Arcade) -> dict[str, list[int]]:
    return {e.game_id.split("-")[0]: list(e.baseline_actions or []) for e in arc.get_environments()}


class LocalEnv:
    """One play of one game with its own scorecard."""

    def __init__(self, arc: arc_agi.Arcade, game_id: str, *, seed: int = 0, tags: Optional[list[str]] = None):
        self.arc = arc
        self.game_id = game_id.split("-")[0]
        self.card_id = arc.open_scorecard(tags=tags or ["arc3-eval"])
        env = arc.make(self.game_id, seed=seed, scorecard_id=self.card_id)
        if env is None:
            raise RuntimeError(f"could not create environment for {game_id!r}")
        self.env = env
        self.step_count = 0
        self.level_step = 0
        self._levels = 0
        self.history: list[tuple[Action, Frame]] = []
        self.frame = self._wrap(env.observation_space)
        self.info = env.info
        self.baselines = list(env.info.baseline_actions or [])

    def _wrap(self, raw: FrameDataRaw) -> Frame:
        f = Frame.from_raw(raw, step=self.step_count, level_step=self.level_step)
        if f.levels_completed != self._levels:
            self._levels = f.levels_completed
            self.level_step = 0
            f.level_step = 0
        return f

    def _level_just_started(self) -> bool:
        g = getattr(self.env, "_game", None)
        return g is not None and getattr(g, "_action_count", 1) == 0 and self.frame.state is not GameState.WIN

    def step(self, action: Action) -> Frame:
        reasoning = action.reasoning
        if reasoning is not None and not isinstance(reasoning, dict):
            reasoning = {"text": str(reasoning)[:2000]}
        if action.action is GameAction.RESET and self._level_just_started():
            # Competition-mode gateway behaviour: no game reset, but the action is billed.
            raw = self.env.observation_space
            if raw is not None and raw.guid:
                self.arc.scorecard_manager.update_scorecard(raw.guid, raw, False)
        else:
            raw = self.env.step(action.action, data=action.data, reasoning=reasoning)
        if raw is None:
            raise RuntimeError(f"environment returned no frame for {action}")
        self.step_count += 1
        self.level_step += 1
        self.frame = self._wrap(raw)
        self.history.append((action, self.frame))
        return self.frame

    def reset(self) -> Frame:
        return self.step(Action.reset())

    def close(self):
        """Close the scorecard and return the toolkit's EnvironmentScorecard."""
        return self.arc.close_scorecard(self.card_id)
