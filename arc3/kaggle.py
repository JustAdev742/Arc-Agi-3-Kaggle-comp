"""Glue between the ARC-AGI-3-Agents framework (used by the Kaggle gateway) and arc3.

The framework instantiates one ``MyAgent`` per game and runs all of them in parallel
threads (``Swarm.main``). Each ``choose_action`` call must return one ``GameAction``.
This module provides:
  * a process-wide deadline shared by all games (``global_deadline``),
  * ``FrameData`` -> ``arc3.env.Frame`` conversion,
  * ``Driver``: runs an arc3 agent behind the single-action interface with crash
    recovery (any exception switches that game to the explorer).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

from .agents import get as get_agent
from .agents.base import Agent, AgentContext
from .env import Action, Frame

log = logging.getLogger("arc3.kaggle")

DEFAULT_BUDGET_S = float(os.environ.get("ARC3_TIME_BUDGET_S", str(9 * 3600)))  # conservative: see docs/status.md
DEFAULT_RESERVE_S = float(os.environ.get("ARC3_RESERVE_S", str(15 * 60)))

_lock = threading.Lock()
_deadline: Optional[float] = None


def global_deadline() -> float:
    """Absolute wall-clock deadline shared by every game in this process."""
    global _deadline
    with _lock:
        if _deadline is None:
            start = float(os.environ.get("ARC3_START_EPOCH", time.time()))
            _deadline = start + DEFAULT_BUDGET_S - DEFAULT_RESERVE_S
            log.info("global deadline in %.0f s", _deadline - time.time())
        return _deadline


def agent_config() -> dict[str, Any]:
    raw = os.environ.get("ARC3_AGENT_CONFIG", "")
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        log.warning("bad ARC3_AGENT_CONFIG, ignoring")
        return {}


def to_frame(fd: FrameData, *, step: int, level_step: int) -> Frame:
    layers = [np.asarray(l, dtype=np.int16) for l in (fd.frame or [])] or [np.zeros((64, 64), np.int16)]
    return Frame(grid=layers[-1], layers=layers, state=fd.state, levels_completed=int(fd.levels_completed),
                 win_levels=int(fd.win_levels), available_actions=list(fd.available_actions or []),
                 full_reset=bool(fd.full_reset), game_id=fd.game_id or "", step=step, level_step=level_step)


class Driver:
    """Owns one arc3 agent for one game and exposes the framework's contract."""

    def __init__(self, game_id: str, agent_name: str, *, config: Optional[dict[str, Any]] = None,
                 deadline: Optional[float] = None, fallback: str = "explorer"):
        self.game_id = game_id
        self.agent_name = agent_name
        self.fallback_name = fallback
        self.deadline = global_deadline() if deadline is None else deadline
        self.ctx = AgentContext(game_id=game_id, deadline=self.deadline, config=dict(config or {}),
                                log=logging.getLogger(f"arc3.{game_id}"))
        self.agent: Agent = self._make(agent_name)
        self.step = 0
        self.level_step = 0
        self.levels = 0
        self.prev: Optional[Frame] = None
        self.last_action: Optional[Action] = None
        self.crashes = 0

    def _make(self, name: str) -> Agent:
        return get_agent(name)(self.ctx)

    def out_of_time(self) -> bool:
        return time.time() >= self.deadline

    def _wrap(self, fd: FrameData) -> Frame:
        if int(fd.levels_completed) != self.levels:
            self.levels = int(fd.levels_completed)
            self.level_step = 0
        return to_frame(fd, step=self.step, level_step=self.level_step)

    def choose(self, latest: FrameData) -> GameAction:
        frame = self._wrap(latest)
        # Deliver the previous step's outcome first (framework gives us only the latest frame).
        if self.prev is not None and self.last_action is not None:
            try:
                self.agent.observe(self.last_action, self.prev, frame)
            except Exception as e:  # noqa: BLE001
                self._crash("observe", e)
        try:
            action = self.agent.act(frame)
        except Exception as e:  # noqa: BLE001
            self._crash("act", e)
            action = self.agent.act(frame)
        if not isinstance(action, Action):
            action = Action.reset()
        self.prev = frame
        self.last_action = action
        self.step += 1
        self.level_step += 1
        ga = action.action
        if ga.is_complex():
            ga.set_data({"x": int(action.x or 0), "y": int(action.y or 0)})
        ga.reasoning = action.reasoning if isinstance(action.reasoning, (dict, str)) else None
        return ga

    def _crash(self, where: str, e: Exception) -> None:
        self.crashes += 1
        log.exception("%s: agent %s crashed in %s (%d); switching to %s", self.game_id, self.agent_name, where,
                      self.crashes, self.fallback_name)
        try:
            self.agent.close()
        except Exception:  # noqa: BLE001
            pass
        self.agent = self._make(self.fallback_name)
        self.agent_name = self.fallback_name

    def done(self, latest: FrameData) -> bool:
        if latest.state is GameState.WIN:
            return True
        return self.out_of_time()

    def close(self) -> None:
        try:
            self.agent.close()
        except Exception:  # noqa: BLE001
            pass
