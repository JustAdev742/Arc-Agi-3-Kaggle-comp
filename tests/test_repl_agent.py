"""End-to-end plumbing test of the REPL agent with a scripted mock model on a real game."""
import time

from arc3.agents import get
from arc3.agents.base import AgentContext
from arc3.env import LocalEnv, make_arcade
from arc3.llm import MockClient


def run(agent, env, n):
    frame = env.frame
    for _ in range(n):
        if agent.is_done(frame):
            break
        a = agent.act(frame)
        before = frame
        frame = env.step(a)
        agent.observe(a, before, frame)
    return frame


def test_repl_agent_executes_model_actions_and_falls_back():
    script = [
        MockClient.tool("print(len(objects()), level, available)\nr = act('UP')\nprint('changed', r['changed'])\nnote('UP moves something' if r['changed'] else 'UP did nothing')"),
        MockClient.tool("rs = act('LEFT', 'DOWN', ('CLICK', 5, 5))\nprint([x['changed'] for x in rs])\nresult = {'n': len(rs)}"),
        MockClient.tool("while True: pass"),  # sandbox timeout -> restart
        MockClient.say("Let me think."),  # idle turns follow -> fallback explorer takes over
    ]
    mock = MockClient(script)
    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    ctx = AgentContext(game_id="ls20", deadline=time.time() + 120,
                       config={"client": mock, "tool_timeout_s": 2, "idle_turns_before_fallback": 2, "image": True, "image_scale": 2})
    agent = get("repl")(ctx)
    try:
        run(agent, env, 12)
        st = agent.stats()
        assert env.step_count == 12
        assert st["actions_model"] == 4, st
        assert st["actions_fallback"] == 8, st
        assert st["sandbox_timeouts"] == 1 and st["sandbox_restarts"] >= 1
        assert st["idle_turns"] >= 2
        assert agent.notes and agent.notes[0].startswith("UP")
        # The tool result of the first call reached the model as a tool message with the printed output.
        tool_msgs = [m for m in mock.calls[1] if m.get("role") == "tool"]
        assert tool_msgs and "changed" in tool_msgs[0]["content"]
        # Images are attached to user turns.
        assert any(isinstance(m.get("content"), list) for m in mock.calls[0])
    finally:
        agent.close()
        env.close()


def test_eviction_keeps_context_bounded():
    big = "x" * 4000
    mock = MockClient(lambda msgs: MockClient.tool(f"print('{big}'[:10])\nact('UP')"))
    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    ctx = AgentContext(game_id="ls20", deadline=time.time() + 120,
                       config={"client": mock, "context_tokens": 6000, "max_output_tokens": 500, "image": False})
    agent = get("repl")(ctx)
    try:
        run(agent, env, 15)
        assert agent.stats()["evictions"] > 0
        assert agent._estimate_tokens(agent.messages) < 6000
        assert agent.messages[0]["role"] == "system"
    finally:
        agent.close()
        env.close()
