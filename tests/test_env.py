"""Competition-mode reset semantics and scorecard parity of the local env wrapper."""
import pytest

from arc3.env import Action, LocalEnv, make_arcade
from arc3.scoring import game_score


@pytest.fixture(scope="module")
def arc():
    return make_arcade("environment_files")


def test_actions_are_billed_and_reset_at_level_start_is_noop(arc):
    env = LocalEnv(arc, "ls20")
    f0 = env.frame
    assert f0.levels_completed == 0 and not f0.done
    # RESET right at the start: must not start a new play, must cost one action.
    f1 = env.step(Action.reset())
    assert f1.levels_completed == 0 and (f1.grid == f0.grid).all()
    env.step(Action.simple(1))
    f3 = env.step(Action.reset())  # level reset after a real action
    assert f3.levels_completed == 0
    card = env.close()
    run = card.find_environment("ls20").runs
    assert len(run) == 1, "competition mode: exactly one play"
    assert run[0].actions == 3
    assert run[0].resets == 2


def test_game_score_parity_with_toolkit_on_partial_play(arc):
    env = LocalEnv(arc, "vc33")
    for _ in range(5):
        env.step(Action.click(10, 10))
    card = env.close()
    run = card.find_environment("vc33").runs[0]
    assert run.levels_completed == 0
    assert run.actions == 5
    assert run.score == pytest.approx(game_score([], env.baselines))
