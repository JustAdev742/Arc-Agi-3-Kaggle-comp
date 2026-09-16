import time

from arc3.agents import get
from arc3.agents.base import AgentContext
from arc3.env import LocalEnv, make_arcade
from arc3.llm import ChatResponse, MockClient


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


def test_council_injects_specialist_reports_into_coordinator_turn():
    seen_roles = []

    def specialist(messages):
        role = messages[0]["content"].split("Role: ")[1].split(".")[0]
        seen_roles.append(role)
        assert isinstance(messages[1]["content"], list)  # image attached for the VL specialist
        return ChatResponse(content=f"{role} report: object #0 is the avatar", prompt_tokens=50, completion_tokens=10)

    # One action per coordinator turn: a tool call, then a plain reply that ends the turn.
    coord = MockClient([MockClient.tool("act('UP')"), MockClient.say("ok"), MockClient.tool("act('DOWN')"), MockClient.say("ok"),
                        MockClient.tool("act('LEFT')"), MockClient.say("ok")])
    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    ctx = AgentContext(game_id="ls20", deadline=time.time() + 300,
                       config={"client": coord, "specialist_client": MockClient(specialist), "image": False,
                               "specialist_every": 2, "idle_turns_before_fallback": 5})
    agent = get("council")(ctx)
    try:
        run(agent, env, 3)
        st = agent.stats()
        assert st["council"]["rounds"] == 2 and st["council"]["calls"] == 12  # turns 1 and 3 (every 2nd)
        assert sorted(set(seen_roles)) == ["EXPLORER", "FALSIFIER", "GOAL ANALYST", "MECHANICS", "PERCEPTION", "PLANNER"]
        first_user = [m for m in coord.calls[0] if m["role"] == "user"][0]["content"]
        assert "Specialist reports" in first_user and "PERCEPTION report" in first_user
        assert st["actions_model"] == 3
        assert st["council"]["shared_model"] is False
    finally:
        agent.close()
        env.close()


def test_council_shared_model_and_timeout_tolerance():
    def slow_or_tool(messages):
        if "Role: " in messages[0]["content"]:
            time.sleep(3)  # specialists too slow: the round times out, coordinator still acts
            return ChatResponse(content="late", prompt_tokens=1, completion_tokens=1)
        return MockClient.tool("act('UP')")

    client = MockClient(slow_or_tool)
    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    ctx = AgentContext(game_id="ls20", deadline=time.time() + 300,
                       config={"client": client, "image": False, "roles": ["perception", "planner"], "specialist_timeout_s": 1})
    agent = get("council")(ctx)
    try:
        t0 = time.time()
        run(agent, env, 1)
        assert time.time() - t0 < 30
        st = agent.stats()
        assert st["council"]["shared_model"] is True and st["council"]["timeouts"] == 1
        assert st["actions_model"] == 1
    finally:
        agent.close()
        env.close()
