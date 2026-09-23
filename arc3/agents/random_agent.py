"""Uniform-random baseline (same policy as the Kaggle starter's Stochastic Goose port)."""
from __future__ import annotations

import random

from arcengine import GameAction, GameState

from ..env import Action, Frame
from . import register
from .base import Agent, AgentContext, stable_seed


@register("random")
class RandomAgent(Agent):
    def __init__(self, ctx: AgentContext):
        super().__init__(ctx)
        self.rng = random.Random(ctx.seed * 1_000_003 + stable_seed(ctx.game_id))

    def act(self, frame: Frame) -> Action:
        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return Action.reset()
        avail = [a for a in frame.actions() if a is not GameAction.RESET] or [GameAction.ACTION1]
        a = self.rng.choice(avail)
        if a.is_complex():
            return Action.click(self.rng.randint(0, 63), self.rng.randint(0, 63))
        return Action(a)
