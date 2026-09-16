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
                       config={"client": mock, "tool_timeout_s": 2, "idle_turns_before_fallback": 2, "fallback_burst": 2,
                               "image": True, "image_scale": 2})
    agent = get("repl")(ctx)
    try:
        run(agent, env, 12)
        st = agent.stats()
        assert env.step_count == 12
        assert st["actions_model"] == 4, st
        assert st["actions_fallback"] == 8, st  # bursts of 2 after every 2 idle turns
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
        agent._evict()  # the newest pair may have been appended after the last in-turn eviction
        budget = agent.context_tokens - agent.max_output_tokens - 1024
        # Eviction keeps the system prompt and the current turn's user message and drops everything else it can:
        # either we are under budget, or only one assistant/tool pair is left (a single turn can exceed a tiny budget).
        roles = [m["role"] for m in agent.messages]
        assert roles[0] == "system" and roles[1] == "user", roles
        n_pairs = roles.count("assistant")
        assert agent._estimate_tokens(agent.messages) <= budget or n_pairs <= 1, (roles, agent._estimate_tokens(agent.messages))
    finally:
        agent.close()
        env.close()


def test_inspection_nudge_and_no_action_notice():
    calls = []

    def script(messages):
        calls.append(messages)
        n = len(calls)
        if n <= 3:
            return MockClient.tool("print(level)")  # inspection only
        if n == 4:
            return MockClient.say("still thinking")  # ends turn 1 without acting
        return MockClient.tool("act('UP')")

    mock = MockClient(script)
    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    ctx = AgentContext(game_id="ls20", deadline=time.time() + 300, config={"client": mock, "image": False})
    agent = get("repl")(ctx)
    try:
        run(agent, env, 1)
        nudges = [m for m in calls[3] if m["role"] == "user" and "inspection steps used" in str(m["content"])]
        assert nudges, "a nudge must follow three inspection-only steps"
        second_turn_user = [m for m in calls[4] if m["role"] == "user"][-1]["content"]
        assert "took NO action" in second_turn_user
        assert agent.stats()["actions_model"] == 1
    finally:
        agent.close()
        env.close()


def test_timeouts_do_not_trigger_fallback_only_and_widen_the_timeout():
    import requests

    calls = {"n": 0}

    def flaky(messages):
        calls["n"] += 1
        if calls["n"] <= 6:
            raise requests.exceptions.ReadTimeout("HTTPConnectionPool: Read timed out.")
        return MockClient.tool("act('UP')")

    mock = MockClient(flaky)
    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    ctx = AgentContext(game_id="ls20", deadline=time.time() + 600,
                       config={"client": mock, "image": False, "model_timeout_s": 100, "max_model_errors": 3})
    agent = get("repl")(ctx)
    try:
        run(agent, env, 1)
        st = agent.stats()
        assert not agent.use_fallback_only
        assert st["model_errors"] == 6 and st["actions_model"] == 1 and st["actions_fallback"] == 0
        assert agent.model_timeout_s > 100
    finally:
        agent.close()
        env.close()


def test_dead_server_switches_to_capped_fallback():
    class DeadClient:
        model = "dead"

        def chat(self, *a, **k):
            raise RuntimeError("Connection refused")

        def models(self):
            raise RuntimeError("Connection refused")

    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    ctx = AgentContext(game_id="ls20", deadline=time.time() + 600,
                       config={"client": DeadClient(), "image": False, "max_model_errors": 2, "fallback_cap_dead": 5})
    agent = get("repl")(ctx)
    try:
        frame = env.frame
        n = 0
        while not agent.is_done(frame) and n < 50:
            a = agent.act(frame)
            before = frame
            frame = env.step(a)
            agent.observe(a, before, frame)
            n += 1
        assert agent.use_fallback_only
        assert agent.stats()["actions_fallback"] <= 6  # cap + at most one stop action
        assert agent.stop_requested
    finally:
        agent.close()
        env.close()


def test_context_overflow_evicts_and_retries_without_counting_an_error():
    calls = {"n": 0}

    def script(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("chat completion failed 400: This model's maximum context length is 32768 tokens.")
        r = MockClient.tool("act('UP')")
        r.prompt_tokens = 0  # no server count: skip calibration so the overflow bump is observable
        return r

    mock = MockClient(script)
    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    ctx = AgentContext(game_id="ls20", deadline=time.time() + 300, config={"client": mock, "image": False})
    agent = get("repl")(ctx)
    try:
        # Seed some history so there is something to evict.
        agent.messages += [{"role": "user", "content": "old turn"}, {"role": "assistant", "content": "ok"},
                           {"role": "user", "content": "older turn 2"}, {"role": "assistant", "content": "ok"}]
        ratio0 = agent.token_ratio
        run(agent, env, 1)
        st = agent.stats()
        assert st["context_overflows"] == 1 and st["model_errors"] == 0 and st["actions_model"] == 1
        assert agent.token_ratio > ratio0
        assert not any(m.get("content") == "old turn" for m in agent.messages)
    finally:
        agent.close()
        env.close()


def test_turn_log_uses_entity_events():
    mock = MockClient([MockClient.tool("act('RIGHT')"), MockClient.say("ok"), MockClient.tool("act('LEFT')"), MockClient.say("ok")])
    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    ctx = AgentContext(game_id="ls20", deadline=time.time() + 300, config={"client": mock, "image": False})
    agent = get("repl")(ctx)
    try:
        run(agent, env, 2)
        second_turn_user = [m for m in mock.calls[2] if m["role"] == "user"][-1]["content"]
        assert "Since your last turn:" in second_turn_user and "moved (" in second_turn_user, second_turn_user[:400]
        assert "Entities (persistent ids" in second_turn_user
    finally:
        agent.close()
        env.close()


def test_transcript_is_written_on_close(tmp_path):
    mock = MockClient([MockClient.tool("print(1)\nact('UP')"), MockClient.say("done")])
    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    ctx = AgentContext(game_id="ls20", deadline=time.time() + 300, config={"client": mock, "image": False}, out_dir=str(tmp_path))
    agent = get("repl")(ctx)
    run(agent, env, 1)
    agent.close()
    env.close()
    import json as _json
    with open(tmp_path / "ls20.transcript.jsonl") as f:
        lines = [_json.loads(line) for line in f]
    assert lines[0]["kind"] == "meta" and lines[0]["stats"]["actions_model"] == 1
    kinds = [l["kind"] for l in lines[1:]]
    assert "assistant" in kinds and "tool" in kinds
    assert any("act('UP')" in c for l in lines if l["kind"] == "assistant" for c in l["code"])


def test_stop_near_deadline_spends_no_action():
    mock = MockClient([MockClient.tool("act('UP')")])
    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    ctx = AgentContext(game_id="ls20", deadline=time.time() + 5, config={"client": mock, "image": False, "min_time_for_turn_s": 45})
    agent = get("repl")(ctx)
    try:
        assert agent.is_done(env.frame)  # stop requested before any action is asked for
        assert env.step_count == 0 and mock.calls == []
    finally:
        agent.close()
        env.close()
