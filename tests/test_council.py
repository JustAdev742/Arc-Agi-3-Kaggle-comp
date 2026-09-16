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
                               "specialist_schedule": "every", "specialist_every": 2, "idle_turns_before_fallback": 5,
                               "specialist_sync": True})
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


def test_council_events_schedule_and_rich_state():
    """Events schedule: a round on the first turn (level start) and after a turn without an action; the specialists
    read the coordinator's observation (entities, avatar, rules line) and their reports are dated."""
    seen_states = []

    def specialist(messages):
        role = messages[0]["content"].split("Role: ")[1].split(".")[0]
        text = messages[1]["content"][0]["text"] if isinstance(messages[1]["content"], list) else messages[1]["content"]
        seen_states.append(text)
        return ChatResponse(content=f"{role}: use plan_to_entity(3)", prompt_tokens=50, completion_tokens=10)

    # turn 1: four key presses (acts); turn 2: talk only, also after the no-code nudge (idle) -> turn 3 gets an 'idle_turn' round
    coord = MockClient([MockClient.tool("act('UP','DOWN','LEFT','RIGHT')"), MockClient.say("ok"),
                        MockClient.say("thinking only"), MockClient.say("still thinking"),
                        MockClient.tool("act('UP')"), MockClient.say("ok")])
    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    ctx = AgentContext(game_id="ls20", deadline=time.time() + 300,
                       config={"client": coord, "specialist_client": MockClient(specialist), "image": False,
                               "roles": ["mechanics", "planner"], "specialist_every": 10, "idle_turns_before_fallback": 5,
                               "specialist_sync": True})
    agent = get("council")(ctx)
    try:
        run(agent, env, 5)
        st = agent.stats()["council"]
        assert st["reasons"].get("level_start") == 1 and st["reasons"].get("idle_turn") == 1, st
        assert st["rounds"] == 2 and "periodic" not in st["reasons"]
        later = [t for t in seen_states if "Rules (auto-fitted" in t]
        assert later and "Entities (persistent ids" in later[0] and "Avatar: #" in later[0], later[0][:500] if later else seen_states[-1][:500]
        assert "STAGNATION" not in later[0] and "took NO action" not in later[0]  # nudges are for the coordinator only
        third_user = [m for m in coord.calls[4] if m["role"] == "user"][0]["content"]
        assert "[mechanics] MECHANICS: use plan_to_entity(3)" in third_user, third_user[:400]
    finally:
        agent.close()
        env.close()


def test_council_async_round_is_injected_within_the_turn():
    """Default mode: the round runs concurrently with the coordinator's first call and its reports are handed over
    as a user message before the coordinator's next call of the same turn."""
    def specialist(messages):
        time.sleep(0.3)
        role = messages[0]["content"].split("Role: ")[1].split(".")[0]
        return ChatResponse(content=f"{role}: target is #6", prompt_tokens=10, completion_tokens=5)

    def coordinator(messages):
        time.sleep(0.6)  # slower than the round: the reports are ready before the second call
        n = sum(1 for m in messages if m["role"] == "assistant")
        return MockClient.tool("print(1)") if n == 0 else (MockClient.tool("act('UP')") if n == 1 else MockClient.say("ok"))

    coord = MockClient(coordinator)
    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    ctx = AgentContext(game_id="ls20", deadline=time.time() + 300,
                       config={"client": coord, "specialist_client": MockClient(specialist), "image": False,
                               "roles": ["goal"], "inspect_steps_before_nudge": 5})
    agent = get("council")(ctx)
    try:
        run(agent, env, 1)
        first_user = [m for m in coord.calls[0] if m["role"] == "user"][0]["content"]
        assert "Specialist reports" not in first_user  # not waited for
        second_call_users = [m["content"] for m in coord.calls[1] if m["role"] == "user"]
        assert any(str(u).startswith("New Specialist reports") and "GOAL ANALYST: target is #6" in str(u) for u in second_call_users), second_call_users
        assert agent.stats()["council"]["late_injections"] == 1
    finally:
        agent.close()
        env.close()


def test_council_disables_itself_after_two_empty_rounds():
    def failing_specialist(messages):
        raise RuntimeError("specialist server down")

    coord = MockClient([MockClient.tool("act('UP')"), MockClient.say("ok")] * 4)
    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    ctx = AgentContext(game_id="ls20", deadline=time.time() + 300,
                       config={"client": coord, "specialist_client": MockClient(failing_specialist), "image": False,
                               "roles": ["planner"], "specialist_schedule": "every", "specialist_every": 1, "specialist_sync": True})
    agent = get("council")(ctx)
    try:
        run(agent, env, 4)
        st = agent.stats()["council"]
        assert st["rounds"] == 2 and st["errors"] == 2 and st["disabled_at_turn"] == 2 and st["roles"] == [], st
        assert agent.stats()["actions_model"] == 4
    finally:
        agent.close()
        env.close()
