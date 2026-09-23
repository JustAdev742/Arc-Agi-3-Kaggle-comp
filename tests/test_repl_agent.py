"""End-to-end plumbing test of the REPL agent with a scripted mock model on a real game."""
import json
import time

from arc3.agents import get
from arc3.agents.base import AgentContext
from arc3.agents.repl_agent import TURN_DONE
from arc3.env import Action, Frame, GameState, LocalEnv, make_arcade
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
        nudges = [m for m in calls[1] if m["role"] == "user" and "inspection step(s) used" in str(m["content"])]
        assert nudges, "a nudge must follow the first inspection-only step"
        assert "calls remain" in nudges[0]["content"]
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


def test_turn_log_uses_entity_events_and_rules_summary():
    mock = MockClient([MockClient.tool("act('RIGHT')"), MockClient.say("ok"), MockClient.tool("act('LEFT', 'UP', 'DOWN')"), MockClient.say("ok"),
                       MockClient.tool("act('RIGHT')"), MockClient.say("ok")])
    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    ctx = AgentContext(game_id="ls20", deadline=time.time() + 300, config={"client": mock, "image": False})
    agent = get("repl")(ctx)
    try:
        run(agent, env, 5)
        second_turn_user = [m for m in mock.calls[2] if m["role"] == "user"][-1]["content"]
        assert "Since your last turn:" in second_turn_user and "moved (" in second_turn_user, second_turn_user[:400]
        assert "Entities (persistent ids" in second_turn_user
        third_turn_user = [m for m in mock.calls[4] if m["role"] == "user"][-1]["content"]
        assert "Avatar: #" in third_turn_user, third_turn_user[:600]  # key labels reach the parent tracker
        assert "Rules (auto-fitted from 4 transitions" in third_turn_user and "move[colour" in third_turn_user, third_turn_user[:800]
        assert agent.stats()["rules_fits"] >= 1
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
    kinds = [rec["kind"] for rec in lines[1:]]
    assert "assistant" in kinds and "tool" in kinds
    assert any("act('UP')" in c for rec in lines if rec["kind"] == "assistant" for c in rec["code"])


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


def _frame(grid, levels=0, level_step=0, state=GameState.NOT_FINISHED):
    return Frame(grid=grid, layers=[grid], state=state, levels_completed=levels, win_levels=3, available_actions=[1, 2, 3, 4],
                 game_id="fake", step=level_step, level_step=level_step)


def test_level_and_stagnation_notices_and_adaptive_effort():
    import numpy as np

    mock = MockClient([MockClient.tool("act('UP')"), MockClient.say("ok")] * 6)
    ctx = AgentContext(game_id="fake", deadline=time.time() + 300, config={"client": mock, "image": False, "stagnation_actions": 3})
    agent = get("repl")(ctx)
    try:
        g0 = np.zeros((64, 64), dtype=np.int16)
        g0[10:14, 10:14] = 9
        g1 = g0.copy()
        g1[10:14, 10:14] = 0
        g1[6:10, 10:14] = 9
        f0, f1 = _frame(g0), _frame(g1, level_step=1)
        agent.act(f0)  # starts turn 1; the mock's act('UP') arrives as a request
        agent.observe(Action.simple(1), f0, f1)
        # level completes on the next action: the new level shows a different board
        g2 = np.zeros((64, 64), dtype=np.int16)
        g2[30:34, 30:34] = 9
        f2 = _frame(g2, levels=1, level_step=0)
        agent.observe(Action.simple(1), f1, f2)
        agent.frame = f2
        text = agent._observation_text()
        assert "LEVEL 1 COMPLETED after 2 actions" in text and "win condition" in text.lower(), text[:400]
        assert len(agent.level_archive) == 1
        assert agent.stats()["levels_completed"] == 1
        # three actions that change nothing -> stagnation notice with untested actions, and a raised effort for the turn
        for i in range(3):
            agent.observe(Action.simple(2), f2, _frame(g2, levels=1, level_step=i + 1))
        agent.frame = f2
        text = agent._observation_text()
        assert "STAGNATION: the last 3 actions changed nothing" in text and "Untested here" in text, text[:600]
        assert agent._effort_for_turn() == "medium" and agent.stats()["effort_raises"] == 1
        agent.recent_changes.append(40)  # something changed again: back to the configured effort
        assert agent._effort_for_turn() == agent.reasoning_effort  # back to the configured effort (None here)
    finally:
        agent.close()


def test_tool_text_does_not_echo_a_printed_act_result():
    from arc3.agents.repl_agent import ReplAgent

    res = {"action": "ACTION6(48,15)", "changed": 2, "events": "#27 size 60->58"}
    shown = ReplAgent._tool_text({"stdout": "{'action': 'ACTION6(48,15)', 'changed': 2}\n", "result": res, "actions": 1})
    assert "result:" not in shown and "1 action(s) executed" in shown
    hidden = ReplAgent._tool_text({"stdout": "probing\n", "result": [res], "actions": 1})
    assert "result: [{\"action\": \"ACTION6(48,15)\"" in hidden
    assert "result:" in ReplAgent._tool_text({"stdout": "", "result": 42})


def test_tool_text_appends_cell_events_unless_printed():
    from arc3.agents.repl_agent import ReplAgent

    ev = ["UP: #3 (c9 4x4) moved (+0,-4)", "RIGHT: no entity changed"]
    out = ReplAgent._tool_text({"stdout": "ok\n", "events": ev, "actions": 2}, events_line=True)
    assert "events: UP: #3 (c9 4x4) moved (+0,-4) | RIGHT: no entity changed" in out
    printed = ReplAgent._tool_text({"stdout": "['UP: #3 (c9 4x4) moved (+0,-4)', 'RIGHT: no entity changed']\n", "events": ev, "actions": 2}, events_line=True)
    assert "events:" not in printed
    assert "events:" not in ReplAgent._tool_text({"stdout": "ok\n", "events": ev, "actions": 2})  # off by default (exp-012/012b)


def test_image_cap_keeps_prompt_under_the_server_limit():
    # exp-018: terse turns let 17+ images accumulate under the token budget; vLLM rejects the prompt (limit 16).
    calls = {"n": 0}

    def script(messages):  # one action per turn: act, then end the turn with a plain answer
        calls["n"] += 1
        return MockClient.tool("act('UP')") if calls["n"] % 2 else MockClient.say("moved")

    mock = MockClient(script)
    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    ctx = AgentContext(game_id="ls20", deadline=time.time() + 300,
                       config={"client": mock, "image": True, "max_images": 3, "context_tokens": 200000})
    agent = get("repl")(ctx)
    try:
        run(agent, env, 8)
        agent._evict()
        assert agent._image_count() <= 3
        assert agent.stats()["evictions"] > 0
        users = [m for m in agent.messages if m["role"] == "user" and isinstance(m.get("content"), list)]
        assert len(users) >= 4, [m["role"] for m in agent.messages]
        # Old observations keep their text; only the image part is replaced by a note.
        stripped = [m for m in users if any("image dropped" in str(p.get("text", "")) for p in m["content"])]
        assert stripped, [[p.get("type") for p in m["content"]] for m in users]
        assert all(any((p.get("type") == "text" and "act(" in str(p.get("text", ""))) or "board" in str(p.get("text", "")).lower()
                       for p in m["content"]) for m in stripped)
        # The newest observation still carries its image.
        assert any(p.get("type") == "image_url" for p in users[-1]["content"])
    finally:
        agent.close()
        env.close()


def test_image_limit_error_strips_images_and_retries_without_counting_an_error():
    calls = {"n": 0}

    def script(messages):
        calls["n"] += 1
        n_img = sum(1 for m in messages if isinstance(m.get("content"), list)
                    for p in m["content"] if p.get("type") == "image_url")
        if n_img > 2:
            raise RuntimeError('chat completion failed 400: {"error":{"message":"At most 2 image(s) may be provided '
                               'in one prompt. (parameter=image)","type":"BadRequestError","param":"image","code":400}}')
        return MockClient.tool("act('UP')") if calls["n"] % 2 else MockClient.say("moved")

    mock = MockClient(script)
    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    ctx = AgentContext(game_id="ls20", deadline=time.time() + 300,
                       config={"client": mock, "image": True, "max_images": 12, "context_tokens": 200000})
    agent = get("repl")(ctx)
    try:
        run(agent, env, 6)
        st = agent.stats()
        assert st["actions_model"] == 6 and st["turns"] >= 6, st
        assert st["model_errors"] == 0 and st["context_overflows"] >= 1, st
        assert agent.max_images <= 2 and agent._image_count() <= 2
    finally:
        agent.close()
        env.close()


def test_lessons_from_model_and_flags_reach_the_observation(tmp_path):
    calls = {"n": 0}

    def script(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return MockClient.tool("learn('walls of colour 5 block UP', kind='mechanic')\nact('UP')")
        return MockClient.say("ok")

    mock = MockClient(script)
    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    ctx = AgentContext(game_id="ls20", deadline=time.time() + 300, out_dir=str(tmp_path),
                       config={"client": mock, "image": False})
    agent = get("repl")(ctx)
    try:
        run(agent, env, 1)
        for _ in range(100):  # the cell's final message (with the lesson) lands after the action was observed
            if agent.memory.model_lessons:
                break
            time.sleep(0.1)
        assert agent.memory.model_lessons == 1
        # Harness-written lessons from the sandbox flags.
        agent._learn_from_result({"flags": {"pred_retired": {"action": "UP", "wrong_cells": 4, "recent": ["UP", "UP", "LEFT"]},
                                            "batch_stopped": {"done": 3, "planned": 9, "idle": ["RIGHT", "RIGHT", "RIGHT"]}}})
        obs = agent._observation_text()
        assert "Lessons (this game" in obs
        assert "[mechanic L1] walls of colour 5 block UP" in obs
        assert "world model was retired" in obs and "batch of 9 actions stopped after 3" in obs
        assert "What did we learn?" in obs  # asked once after the retired model...
        assert "What did we learn?" not in agent._observation_text()  # ...and not again
        st = agent.stats()
        assert st["lessons_model"] == 1 and st["lessons_auto"] == 2
        # Another game in the same run sees the shared mechanic, not the private mistakes.
        other = get("repl")(AgentContext(game_id="ft09", deadline=time.time() + 300, out_dir=str(tmp_path),
                                         config={"client": mock, "image": False}))
        try:
            txt = other.memory.render_others()
            assert "[ls20 mechanic] walls of colour 5 block UP" in txt and "retired" not in txt
        finally:
            other.close()
    finally:
        agent.close()
        env.close()
    saved = json.loads((tmp_path / "ls20.lessons.json").read_text())
    assert saved["model_lessons"] == 1 and len(saved["lessons"]) == 3
    meta = json.loads(open(tmp_path / "ls20.transcript.jsonl").readline())
    assert len(meta["lessons"]) == 3


def test_level_consolidation_records_lessons_refuses_actions_and_clears_history(tmp_path):
    import numpy as np

    calls = {"n": 0}

    def script(messages):
        calls["n"] += 1
        last = messages[-1]
        text = last.get("content") if isinstance(last.get("content"), str) else ""
        if "Consolidation step" in text:
            return MockClient.tool("learn('walk the avatar onto the odd-coloured tile', kind='goal')\n"
                                   "learn('UP moves the avatar 4 px', kind='mechanic')\nnote('level 1 target was #3')\n"
                                   "r = act('UP')")
        if last.get("role") == "tool":
            return MockClient.say("done\nFRICTION: the tile map is hard to read")
        return MockClient.tool("act('UP')") if calls["n"] % 2 else MockClient.say("moved")

    mock = MockClient(script)
    ctx = AgentContext(game_id="fake", deadline=time.time() + 300, out_dir=str(tmp_path),
                       config={"client": mock, "image": False, "consolidation_calls": 2})
    agent = get("repl")(ctx)
    try:
        g0 = np.zeros((64, 64), dtype=np.int16)
        g0[10:14, 10:14] = 9
        g1 = g0.copy()
        g1[10:14, 10:14] = 0
        g1[6:10, 10:14] = 9
        f0, f1 = _frame(g0), _frame(g1, level_step=1)
        agent.act(f0)
        agent.observe(Action.simple(1), f0, f1)
        # The level completes: the engine returns the terminal frame of level 1 first and the new level's start last.
        g_term = g1.copy()
        g_term[6:10, 10:14] = 0
        g_term[2:6, 10:14] = 9
        g2 = np.zeros((64, 64), dtype=np.int16)
        g2[30:34, 30:34] = 9
        f2 = Frame(grid=g2, layers=[g_term, g2], state=GameState.NOT_FINISHED, levels_completed=1, win_levels=3,
                   available_actions=[1, 2, 3, 4], game_id="fake", step=2, level_step=0)
        agent.observe(Action.simple(1), f1, f2)
        assert agent.consolidate_pending and agent.consolidate_pending["level"] == 1
        assert "moved" in agent.consolidate_pending["winning_move"]  # the winning move was tracked on the terminal frame
        assert "terminal" not in agent.last_result and "terminal" not in json.dumps(agent._state_payload()["last"])
        assert len(agent.level_archive) == 1 and len(agent.level_archive[0][0]) == 3  # start, after UP, terminal (observed)
        n_msgs = len(agent.messages)
        agent.frame = f2
        agent._consolidate_level()
        assert agent.consolidate_pending is None and agent.stats()["consolidations"] == 1
        assert [m["role"] for m in agent.messages] == ["system"]  # conversation cleared at the level boundary
        assert n_msgs > 1
        lessons = {(it["kind"], it["text"]) for it in agent.memory.to_list()}
        assert ("goal", "walk the avatar onto the odd-coloured tile") in lessons
        assert ("mechanic", "UP moves the avatar 4 px") in lessons
        assert any(it["kind"] == "recipe" for it in agent.memory.to_list())
        assert agent.notes == ["level 1 target was #3"]
        assert agent.friction == ["the tile map is hard to read"]
        assert not agent.no_actions
        rec = [r for r in agent.transcript if r["kind"] == "consolidation"]
        assert len(rec) == 2
        # act() inside the consolidation cell was refused: only the first turn's UP was ever served.
        assert agent.stats()["actions_model"] == 1
        while not agent.action_q.empty():
            assert agent.action_q.get_nowait() is TURN_DONE
        text = agent._observation_text()
        assert "LEVEL 1 COMPLETED after 2 actions" in text and "Lessons (this game" in text
    finally:
        agent.close()
    meta = json.loads(open(tmp_path / "fake.transcript.jsonl").readline())
    assert meta["friction"] == ["the tile map is hard to read"]


def test_consolidation_is_skipped_when_disabled_or_out_of_time():
    import numpy as np

    mock = MockClient([MockClient.tool("act('UP')"), MockClient.say("ok")] * 4)
    ctx = AgentContext(game_id="fake", deadline=time.time() + 300, config={"client": mock, "image": False, "level_consolidation": False})
    agent = get("repl")(ctx)
    try:
        g0 = np.zeros((64, 64), dtype=np.int16)
        g0[10:14, 10:14] = 9
        g1 = np.zeros((64, 64), dtype=np.int16)
        g1[30:34, 30:34] = 9
        f0, f1 = _frame(g0), _frame(g1, levels=1)
        agent.act(f0)
        agent.observe(Action.simple(1), f0, f1)
        assert agent.consolidate_pending is None
    finally:
        agent.close()
    mock = MockClient([MockClient.tool("act('UP')"), MockClient.say("ok")] * 4)
    ctx = AgentContext(game_id="fake", deadline=time.time() + 60, config={"client": mock, "image": False})
    agent = get("repl")(ctx)
    try:
        g0 = np.zeros((64, 64), dtype=np.int16)
        g0[10:14, 10:14] = 9
        g1 = np.zeros((64, 64), dtype=np.int16)
        g1[30:34, 30:34] = 9
        f0, f1 = _frame(g0), _frame(g1, levels=1)
        agent.act(f0)
        agent.observe(Action.simple(1), f0, f1)
        assert agent.consolidate_pending is not None
        agent.messages.append({"role": "user", "content": "old"})
        calls_before = agent.stats()["model_calls"]
        agent._consolidate_level()  # under 3 turns of time left: no model call, history cleared anyway
        assert agent.stats()["model_calls"] == calls_before and [m["role"] for m in agent.messages] == ["system"]
    finally:
        agent.close()


def test_action_budget_notice_fires_at_the_threshold_and_doubles():
    import numpy as np

    mock = MockClient([MockClient.tool("act('UP')"), MockClient.say("ok")] * 6)
    ctx = AgentContext(game_id="fake", deadline=time.time() + 300, config={"client": mock, "image": False, "level_action_notice": 4})
    agent = get("repl")(ctx)
    try:
        g = np.zeros((64, 64), dtype=np.int16)
        g[10:14, 10:14] = 9
        agent.frame = _frame(g, level_step=3)
        agent.tracker.reset(g)
        assert "ACTION BUDGET" not in agent._observation_text()
        agent.frame = _frame(g, level_step=4)
        text = agent._observation_text()
        assert "ACTION BUDGET: 4 actions spent on level 1" in text and agent.stats()["action_budget_notices"] == 1
        agent.frame = _frame(g, level_step=6)
        assert "ACTION BUDGET" not in agent._observation_text()  # next notice at the doubling (8)
        agent.frame = _frame(g, level_step=9)
        assert "ACTION BUDGET: 9 actions" in agent._observation_text()
        # a completed level resets the threshold for the next level
        g2 = np.zeros((64, 64), dtype=np.int16)
        g2[30:34, 30:34] = 9
        agent.observe(Action.simple(1), _frame(g, level_step=9), _frame(g2, levels=1, level_step=0))
        assert agent._action_notice_next == 4
        # off by default
        agent2 = get("repl")(AgentContext(game_id="fake", deadline=time.time() + 300, config={"client": mock, "image": False}))
        try:
            agent2.frame = _frame(g, level_step=500)
            agent2.tracker.reset(g)
            assert "ACTION BUDGET" not in agent2._observation_text()
        finally:
            agent2.close()
    finally:
        agent.close()


def test_explore_first_sweep_is_built_once_per_level_and_reported_once():
    import numpy as np

    mock = MockClient([MockClient.tool("act('UP')"), MockClient.say("ok")] * 4)
    ctx = AgentContext(game_id="fake", deadline=time.time() + 300,
                       config={"client": mock, "image": False, "explore_first": 8, "explore_first_clicks": 2})
    agent = get("repl")(ctx)
    try:
        g = np.zeros((64, 64), dtype=np.int16)
        g[10:14, 10:14] = 9   # two entity classes: a 4x4 colour-9 square and a 2x2 colour-3 square
        g[30:32, 40:42] = 3
        g[0:64, 0:64][g == 0] = 0
        f0 = Frame(grid=g, layers=[g], state=GameState.NOT_FINISHED, levels_completed=0, win_levels=3,
                   available_actions=[1, 2, 3, 4, 5, 6], game_id="fake", step=1, level_step=1)
        agent.frame = f0
        agent.tracker.reset(g)
        assert agent._sweep_due(f0)
        sweep = agent._build_sweep(f0)
        names = [a.action.name for a in sweep]
        # four keys, ACT, then one click per entity class (2), capped at explore_first (8) -> 7 actions
        assert names[:5] == ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"] and len(sweep) == 7, names
        clicks = [(a.x, a.y) for a in sweep[5:]]
        assert (11, 11) in clicks and (40, 30) in clicks, clicks  # centres of the two squares, largest first
        # a lower cap truncates; a level already played by the model gets no sweep; a game over gets none
        agent.explore_first = 3
        assert len(agent._build_sweep(f0)) == 3
        assert not agent._sweep_due(Frame(grid=g, layers=[g], state=GameState.NOT_FINISHED, levels_completed=0, win_levels=3,
                                          available_actions=[1], game_id="fake", step=5, level_step=5))
        assert not agent._sweep_due(Frame(grid=g, layers=[g], state=GameState.GAME_OVER, levels_completed=0, win_levels=3,
                                          available_actions=[1], game_id="fake", step=1, level_step=1))
        # act() serves the sweep before any model turn; observe() files the effects in the sweep log, not the turn log
        agent.explore_first = 8
        a1 = agent.act(f0)
        assert agent.pending is not None and agent.pending.sweep and a1.action.name == "ACTION1"
        g1 = g.copy()
        g1[10:14, 10:14] = 0
        g1[9:13, 10:14] = 9  # the square moved up
        f1 = Frame(grid=g1, layers=[g1], state=GameState.NOT_FINISHED, levels_completed=0, win_levels=3,
                   available_actions=[1, 2, 3, 4, 5, 6], game_id="fake", step=2, level_step=2)
        agent.observe(a1, f0, f1)
        assert agent.sweep_log and "moved" in agent.sweep_log[0] and not agent.turn_log
        assert agent.stats()["sweep_actions"] == 1 and agent.stats()["actions_model"] == 0
        # the remaining sweep actions are served without a model turn, once each
        served = 1
        while agent._sweep:
            a = agent.act(f1)
            assert agent.pending.sweep
            agent.observe(a, f1, f1)
            served += 1
        assert served == 7 and agent.stats()["sweep_actions"] == 7 and mock.calls == []
        text = agent._observation_text()
        assert "PROBE SWEEP: the harness spent 7 actions" in text and "moved" in text
        assert not agent.sweep_log and "PROBE SWEEP" not in agent._observation_text()  # shown once
        # a level completed by a sweep action drops the rest of the sweep; the next level gets its own sweep
        agent._sweep = deque_of(agent, [Action.simple(1), Action.simple(2)])
        agent.pending = agent_req(agent, Action.simple(1))
        g2 = np.zeros((64, 64), dtype=np.int16)
        g2[20:24, 20:24] = 9
        f2 = Frame(grid=g2, layers=[g, g2], state=GameState.NOT_FINISHED, levels_completed=1, win_levels=3,
                   available_actions=[1, 2, 3, 4], game_id="fake", step=9, level_step=0)
        agent.observe(Action.simple(1), f1, f2)
        assert not agent._sweep and "LEVEL COMPLETED" in agent.sweep_log[-1]
        assert agent._sweep_due(f2) and len(agent._build_sweep(f2)) == 4
    finally:
        agent.close()


def deque_of(agent, actions):
    from collections import deque
    return deque(actions)


def agent_req(agent, action):
    from arc3.agents.repl_agent import _Req
    return _Req(action, auto=True, sweep=True)


def test_explore_first_sweep_precedes_the_first_model_turn_on_a_real_game():
    mock = MockClient([MockClient.tool("act('UP')"), MockClient.say("ok")] * 6)
    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    ctx = AgentContext(game_id="ls20", deadline=time.time() + 120,
                       config={"client": mock, "image": False, "explore_first": 6, "explore_first_clicks": 1})
    agent = get("repl")(ctx)
    try:
        run(agent, env, 10)
        st = agent.stats()
        assert 1 <= st["sweep_actions"] <= 6, st
        assert env.step_count == 10 and st["actions_model"] + st["sweep_actions"] + st["actions_fallback"] == 10, st
        first_user = next(m for m in mock.calls[0] if m.get("role") == "user")
        text = first_user["content"] if isinstance(first_user["content"], str) else first_user["content"][0]["text"]
        assert "PROBE SWEEP" in text, text[:300]
        # shown once: the last call's history holds the first observation and nothing later repeats the line
        users = [m["content"] if isinstance(m["content"], str) else m["content"][0]["text"] for m in mock.calls[-1] if m.get("role") == "user"]
        assert len(users) >= 2 and sum("PROBE SWEEP" in u for u in users) == 1, users
        # off by default
        agent2 = get("repl")(AgentContext(game_id="ls20", deadline=time.time() + 120, config={"client": MockClient([MockClient.say("ok")]), "image": False}))
        try:
            assert agent2.explore_first == 0 and not agent2._sweep_due(env.frame)
        finally:
            agent2.close()
    finally:
        agent.close()
        env.close()


def test_goal_hypotheses_line_ranks_and_falsifies_from_level_two():
    import numpy as np

    mock = MockClient([MockClient.say("ok")])
    agent = get("repl")(AgentContext(game_id="fake", deadline=time.time() + 300, config={"client": mock, "image": False}))
    try:
        g = np.zeros((64, 64), dtype=np.int16)
        g[10:14, 10:14] = 9
        g[30:32, 40:42] = 3
        f0 = _frame(g, levels=1, level_step=0)
        agent.frame = f0
        agent.tracker.reset(g)
        agent.tracker_level = 1  # what act() records; observe() then updates the tracker instead of resetting it
        assert "Goal hypotheses" not in agent._observation_text()  # nothing known before a level is completed
        from arc3 import dsl
        agent.goal_info = [{"goal": "none_left(colour 9)", "kind": "none_left", "args": (9,), "predicate": lambda f: not any(e.color == 9 for e in f)},
                           {"goal": "none_left(colour 3)", "kind": "none_left", "args": (3,), "predicate": lambda f: not any(e.color == 3 for e in f)}]
        assert dsl.goal_kind(agent.goal_info[0]) == ("none_left", (9,))
        text = agent._observation_text()
        assert "Goal hypotheses (code-computed; 2 live, 0 falsified): none_left(colour 9) dist 1" in text, text
        # the colour-3 entity vanishes without the level completing: that hypothesis is falsified, the other stays live
        g1 = g.copy()
        g1[30:32, 40:42] = 0
        f1 = _frame(g1, levels=1, level_step=1)
        agent.observe(Action.simple(1), f0, f1)
        text = agent._observation_text()
        assert "1 live, 1 falsified" in text and "Falsified this level (came true without completing it): none_left(colour 3)" in text, text
        assert agent._goal_falsified == {"none_left(colour 3)": 1} and agent._goal_checked == 2
        # config knob off: no line
        agent.goal_progress_in_prompt = False
        assert "Goal hypotheses" not in agent._observation_text()
    finally:
        agent.close()


def test_noop_memory_records_flags_and_optionally_skips():
    """exp-023: an action that changed nothing from an exact frame is remembered per frame hash; re-sending it is
    flagged in the result (and counted), the observation lists the frame's known no-ops, and with noop_skip the
    harness answers without spending the action unless the model forces it. RESET is never a no-op."""
    import numpy as np

    from arc3.agents.repl_agent import _Req
    mock = MockClient([MockClient.say("ok")])
    agent = get("repl")(AgentContext(game_id="fake", deadline=time.time() + 300, config={"client": mock, "image": False}))
    try:
        g = np.zeros((64, 64), dtype=np.int16)
        g[10:14, 10:14] = 9
        f0 = _frame(g, level_step=1)
        f1 = _frame(g, level_step=2)  # same grid: UP changed nothing
        agent.frame = f0
        agent.tracker.reset(g)
        agent.tracker_level = 0
        agent.observe(Action.simple(1), f0, f1)
        assert agent._known_noop(Action.simple(1)) and not agent._known_noop(Action.simple(2)) and not agent._known_noop(Action.reset())
        assert "Known no-ops from this exact frame" in agent._observation_text() and "ACTION1" in agent._observation_text()
        # a re-send is executed but flagged and counted (soft mode)
        agent.pending = _Req(Action.simple(1), known_noop=True)
        agent.observe(Action.simple(1), f1, _frame(g, level_step=3))
        assert agent.pending is None and agent.st.noop_repeats == 1
        assert not agent._should_skip(Action.simple(1), True)  # noop_skip off
        # hard mode: not sent, answered from memory; force and RESET always go through
        agent.noop_skip = True
        assert agent._should_skip(Action.simple(1), True) and not agent._should_skip(Action.simple(1), True, force=True)
        assert not agent._should_skip(Action.reset(), True)
        results, _state = agent._handle_actions([{"action": "UP"}])
        assert results[0]["skipped_known_noop"] and results[0]["changed"] == 0 and agent.st.noop_skipped == 1 and agent.action_q.empty()
        assert "noop_skip" in agent._observation_text()
        # a changed frame is not a no-op; the memory is per exact frame
        g2 = g.copy()
        g2[20:24, 20:24] = 3
        agent.observe(Action.simple(2), f1, _frame(g2, level_step=4))
        assert not agent._known_noop(Action.simple(2))
        # the champion preset keeps the memory off entirely
        from arc3.presets import CHAMPION
        agent2 = get("repl")(AgentContext(game_id="fake", deadline=time.time() + 300, config={"client": mock, "image": False, **CHAMPION}))
        try:
            agent2.frame = f0
            agent2.tracker.reset(g)
            agent2.tracker_level = 0
            agent2.observe(Action.simple(1), f0, f1)
            assert not agent2.noops and "Known no-ops" not in agent2._observation_text()
        finally:
            agent2.close()
    finally:
        agent.close()


def test_postmortem_call_at_the_end_of_an_unsolved_game(tmp_path):
    """Brief item 15: one structured post-mortem call under fixed headings when an unsolved game ends with time left;
    saved in the transcript meta and as <game>.postmortem.md; never for a won game or when the knob is off."""
    calls = {"n": 0}

    def script(messages):
        calls["n"] += 1
        text = messages[-1].get("content") if isinstance(messages[-1].get("content"), str) else ""
        if "WHAT DID WE BELIEVE?" in text:
            return MockClient.say("WHAT DID WE BELIEVE?\nThat UP moves the avatar.\nWHAT ACTUALLY HAPPENED?\nNothing moved.\nCONFIDENCE: low")
        return MockClient.tool("act('UP')") if calls["n"] % 2 else MockClient.say("ok")

    mock = MockClient(script)
    arc = make_arcade("environment_files")
    env = LocalEnv(arc, "ls20")
    ctx = AgentContext(game_id="ls20", deadline=time.time() + 300,
                       config={"client": mock, "image": False, "postmortem": True, "postmortem_min_s": 0, "transcript_dir": str(tmp_path)})
    agent = get("repl")(ctx)
    try:
        run(agent, env, 3)
        agent.close()
        assert agent.stats()["postmortems"] == 1 and agent.postmortem_text.startswith("WHAT DID WE BELIEVE?")
        meta = json.loads((tmp_path / "ls20.transcript.jsonl").read_text().splitlines()[0])
        assert meta["postmortem"].startswith("WHAT DID WE BELIEVE?")
        assert (tmp_path / "ls20.postmortem.md").read_text().startswith("# ls20: post-mortem (level 1/7")
        pm_calls = [c for c in mock.calls if any("WHAT DID WE BELIEVE?" in (m.get("content") if isinstance(m.get("content"), str) else "") for m in c if m.get("role") == "user")]
        assert len(pm_calls) == 1 and not pm_calls[0][-1].get("tool_calls")
        agent.close()  # idempotent
        assert agent.stats()["postmortems"] == 1
    finally:
        agent.close()
        env.close()
    # off by default and in the champion preset
    from arc3.presets import CHAMPION
    assert CHAMPION["postmortem"] is False
    env2 = LocalEnv(arc, "ls20")
    agent2 = get("repl")(AgentContext(game_id="ls20", deadline=time.time() + 300, config={"client": MockClient(script), "image": False, "transcript_dir": str(tmp_path / "b")}))
    try:
        run(agent2, env2, 2)
        agent2.close()
        assert agent2.stats()["postmortems"] == 0 and not (tmp_path / "b" / "ls20.postmortem.md").exists()
    finally:
        agent2.close()
        env2.close()


def test_animated_win_archives_the_final_board_not_the_first_layer():
    import numpy as np

    mock = MockClient(lambda messages: MockClient.tool("act('UP')"))
    ctx = AgentContext(game_id="fake", deadline=time.time() + 300, config={"client": mock, "image": False})
    agent = get("repl")(ctx)
    try:
        g0 = np.zeros((64, 64), dtype=np.int16)
        g0[10:14, 10:14] = 9
        g1 = g0.copy()
        g1[10:14, 10:14] = 0
        g1[6:10, 10:14] = 9
        f0, f1 = _frame(g0), _frame(g1, level_step=1)
        agent.act(f0)
        agent.observe(Action.simple(1), f0, f1)
        # An animated winning move (cd82's pour, 2026-09-23): the first layer is still the board before the move, the
        # finished board is second to last, the next level's start is last.
        g_mid = g1.copy()
        g_mid[6:10, 10:14] = 0
        g_mid[4:8, 10:14] = 9
        g_term = g1.copy()
        g_term[6:10, 10:14] = 0
        g_term[2:6, 10:14] = 9
        g2 = np.zeros((64, 64), dtype=np.int16)
        g2[30:50, 30:50] = 9
        f2 = Frame(grid=g2, layers=[g1.copy(), g_mid, g_term, g2], state=GameState.NOT_FINISHED, levels_completed=1,
                   win_levels=3, available_actions=[1, 2, 3, 4], game_id="fake", step=2, level_step=0)
        agent.observe(Action.simple(1), f1, f2)
        final = agent.level_archive[0][0][-1]
        assert any(e.color == 9 and e.y0 == 2 for e in final)  # the archived winning frame is the finished board
    finally:
        agent.close()
