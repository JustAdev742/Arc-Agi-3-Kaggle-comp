"""The code-only rules agent must probe, fit, plan and complete a level in the synthetic grid world."""
import time

import numpy as np

from arc3.agents import get
from arc3.agents.base import AgentContext
from arc3.env import Action, Frame, GameState
from tests.test_planner import GridWorld


class FakeEnv:
    """Minimal env: the level completes when the avatar touches the colour-12 target; then the layout repeats."""

    def __init__(self):
        self.w = GridWorld()
        self.levels = 0
        self.frame = self._frame(GameState.NOT_PLAYED)

    def _frame(self, state=GameState.NOT_FINISHED):
        g = self.w.grid()
        return Frame(grid=g, layers=[g], state=state, levels_completed=self.levels, win_levels=3, available_actions=[1, 2, 3, 4], game_id="fake")

    def step(self, a: Action) -> Frame:
        if a.action.value == 0:
            self.frame = self._frame()
            return self.frame
        key = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}.get(a.action.value)
        if key:
            self.w.act(key)
        x, y = self.w.pos
        tx, ty = self.w.target
        if abs(x - tx) <= 4 and abs(y - ty) <= 4:
            self.levels += 1
            self.w.pos = (8, 32)
            self.frame = self._frame(GameState.WIN if self.levels >= 3 else GameState.NOT_FINISHED)
        else:
            self.frame = self._frame()
        return self.frame


def test_rules_agent_completes_gridworld_levels():
    env = FakeEnv()
    ctx = AgentContext(game_id="fake", deadline=time.time() + 60, max_actions=400, config={})
    agent = get("rules")(ctx)
    f = env.frame
    n = 0
    while n < 300 and not agent.is_done(f):
        a = agent.act(f)
        before = f
        f = env.step(a)
        agent.observe(a, before, f)
        n += 1
    st = agent.stats()
    assert env.levels >= 2, (env.levels, n, st)
    assert st["plans"] >= 1 and st["plan_actions"] >= 10, st
    # the second level should be solved from the goal learned on the first (fewer probes than a blind level)
    assert n < 200, (n, st)
