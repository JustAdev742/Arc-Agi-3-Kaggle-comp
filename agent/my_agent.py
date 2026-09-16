"""Kaggle entry point: the framework-facing ``MyAgent``.

The notebook copies this file into ``ARC-AGI-3-Agents/agents/templates/my_agent.py`` and
puts ``/kaggle/working`` (which holds the ``arc3`` package) on PYTHONPATH. Everything
interesting lives in ``arc3``; this file only adapts the framework's contract.

Environment knobs (set by the notebook):
  ARC3_AGENT         repl | explorer | random      (default explorer when no model server)
  ARC3_AGENT_CONFIG  JSON dict passed to the agent (base_url, model, ...)
  ARC3_TIME_BUDGET_S total wall-clock budget for the whole run (default 9h)
  ARC3_START_EPOCH   notebook start time (so setup time is charged to the budget)
"""
from __future__ import annotations

import logging
import os
from typing import Any

from arcengine import FrameData, GameAction

from agents.agent import Agent

from arc3.kaggle import Driver, agent_config

log = logging.getLogger("arc3.my_agent")


class MyAgent(Agent):
    MAX_ACTIONS = 10**9  # the governor, not the framework, decides when to stop

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        name = os.environ.get("ARC3_AGENT", "explorer")
        self.driver = Driver(self.game_id, name, config=agent_config())
        log.info("%s: agent=%s deadline in %.0fs", self.game_id, name, self.driver.deadline - __import__("time").time())

    @property
    def name(self) -> str:
        return f"{super().name}.arc3"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        d = self.driver.done(latest_frame)
        if d:
            self.driver.close()
        return d

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        try:
            return self.driver.choose(latest_frame)
        except Exception as e:  # last line of defence: never kill the game thread
            log.exception("%s: driver failed (%s); sending RESET", self.game_id, e)
            self.driver.last_data, self.driver.last_reasoning = {}, None
            return GameAction.RESET

    def do_action_request(self, action: GameAction) -> FrameData:
        """Send the action with *this game's* data. The framework's default reads the (x, y) off the
        shared GameAction enum member, which every game thread mutates: a race under Swarm."""
        data = dict(self.driver.last_data) if action.is_complex() else {}
        reasoning = self.driver.last_reasoning
        if reasoning is not None and not isinstance(reasoning, dict):
            reasoning = {"text": str(reasoning)}
        raw = self.arc_env.step(action, data=data, reasoning=reasoning)
        return self._convert_raw_frame_data(raw)
