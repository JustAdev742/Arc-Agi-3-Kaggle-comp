"""Agent interface used by the local harness and by the Kaggle adapter."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from ..env import Action, Frame


@dataclass
class AgentContext:
    """Everything an agent may know about its situation, provided by the harness."""

    game_id: str
    seed: int = 0
    deadline: Optional[float] = None  # absolute time.time() after which the harness stops
    max_actions: Optional[int] = None
    baselines: list[int] = field(default_factory=list)  # human baseline actions per level (may be empty on Kaggle)
    config: dict[str, Any] = field(default_factory=dict)
    out_dir: Optional[str] = None  # where the agent may write transcripts/artifacts for this game
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("arc3.agent"))

    def time_left(self) -> float:
        return float("inf") if self.deadline is None else max(0.0, self.deadline - time.time())


class Agent(ABC):
    name: str = "base"

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx
        self.log = ctx.log

    @abstractmethod
    def act(self, frame: Frame) -> Action:
        """Choose the next action given the latest frame."""

    def observe(self, action: Action, before: Frame, after: Frame) -> None:  # noqa: B027  (optional hook)
        """Called after every environment step (optional hook)."""

    def is_done(self, frame: Frame) -> bool:
        return frame.done

    def stats(self) -> dict[str, Any]:
        """Extra per-game numbers for the run record (tokens, tool calls, ...)."""
        return {}

    def close(self) -> None:  # noqa: B027  (optional hook)
        """Release resources (sandbox processes, sessions). Called once by the harness."""
