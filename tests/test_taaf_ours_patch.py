"""Tests for scripts/taaf_ours_patch.py, the source patches every Duck-fork notebook applies at run time.

A patch that stops matching the upstream text fails the notebook at its first cell, and a patch that breaks the
harness can hang or crash games in a competition rerun. These run against a verbatim copy of the upstream modules
(tests/fixtures/taaf_anim, see its NOTICE.md), so they need neither the Kaggle dataset nor a model server.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "taaf_anim"
sys.path.insert(0, str(ROOT / "scripts"))
import taaf_ours_patch as tp  # noqa: E402

ALL = list(tp.PATCHES)


def _copy(dst: Path) -> Path:
    shutil.copytree(FIXTURE, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


# --- patch application ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_each_patch_applies_alone(tmp_path, name):
    bundle = _copy(tmp_path / "b")
    assert len(tp.apply(bundle, [name])) == len(tp.PATCHES[name])


@pytest.mark.parametrize("order", [ALL, list(reversed(ALL))], ids=["forward", "reversed"])
def test_all_patches_apply_together_in_any_order_and_compile(tmp_path, order):
    bundle = _copy(tmp_path / "b")
    tp.apply(bundle, order)
    for rel in {rel for pairs in tp.PATCHES.values() for rel, _, _ in pairs}:
        compile((bundle / rel).read_text(encoding="utf-8"), rel, "exec")


def test_a_patch_applied_twice_fails_loudly(tmp_path):
    bundle = _copy(tmp_path / "b")
    tp.apply(bundle, ["P2"])
    with pytest.raises(RuntimeError, match="expected exactly one match"):
        tp.apply(bundle, ["P2"])


# --- the patched harness ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def h(tmp_path_factory):
    """The fully patched harness modules, imported from a temporary copy."""
    bundle = _copy(tmp_path_factory.mktemp("patched") / "b")
    tp.apply(bundle)
    src = str(bundle / "src" / "ARC3-Inference")
    saved_env = {k: os.environ.get(k) for k in ("LOCAL_ANALYZER_MODEL_ID", "LOCAL_ANALYZER_BASE_URL")}
    os.environ.update(LOCAL_ANALYZER_MODEL_ID="mock", LOCAL_ANALYZER_BASE_URL="http://127.0.0.1:9/v1")
    sys.path.insert(0, src)
    try:
        from inference.agent import action_names, prompts, python_tool_sandbox, tool_agent
        from inference.agent.runtime_state import Frame, HistoryEntry
        from inference.utils import openai_compat

        yield types.SimpleNamespace(ta=tool_agent, sandbox=python_tool_sandbox, prompts=prompts, names=action_names,
                                    compat=openai_compat, Frame=Frame, HistoryEntry=HistoryEntry)
    finally:
        sys.path.remove(src)
        for mod in [m for m in sys.modules if m == "inference" or m.startswith("inference.")]:
            del sys.modules[mod]
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _grid(cells: dict[tuple[int, int], int], size: int = 8) -> tuple[tuple[int, ...], ...]:
    g = [[0] * size for _ in range(size)]
    for (r, c), v in cells.items():
        g[r][c] = v
    return tuple(tuple(row) for row in g)


def test_p1_note_read_from_reasoning(h):
    note = h.ta._extract_scientist_note_from_reasoning(
        "thinking about it\n**World model:** the red block is the player\nPlan: move left twice\n\nmore thoughts")
    assert note["world_model"] == "the red block is the player"
    assert note["current_plan"] == "move left twice"


def test_p2_undo_maps_to_action7(h):
    assert h.names.to_engine_action("UNDO") == "ACTION7"
    assert h.names.to_model_action("ACTION7") == "UNDO"


def test_p3_goal_and_action_models_survive_a_level_change_but_not_a_game_over_reset(h):
    agent = h.ta.ToolAgent(model="mock")
    base = {"world_model": "W", "goal_model": "G", "action_model": "A", "recent_findings": "R",
            "open_questions": "O", "current_plan": "P", "cross_level_notes": "C"}
    agent._summarized_knowledge = dict(base)
    agent._last_step_summary = {"game_over": True}
    agent._update_summarized_knowledge_from_step_summary()
    assert agent._summarized_knowledge == base
    agent._last_step_summary = {"level_transition": True}
    agent._update_summarized_knowledge_from_step_summary()
    k = agent._summarized_knowledge
    assert k["goal_model"].startswith("[from an earlier level") and k["goal_model"].endswith(" G")
    assert k["action_model"].endswith(" A") and k["cross_level_notes"] == "C"
    assert k["world_model"] == k["current_plan"] == k["recent_findings"] == ""


def test_p4_older_user_turns_are_compressed_and_past_reasoning_dropped(h):
    user = {"role": "user", "content": [
        {"type": "text", "text": "Executed actions: LEFT.\nCurrent state: step 3, level 1.\nOnly tool: `python`. It receives..."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}
    out = h.ta._compress_history_message(user)
    text = out["content"] if isinstance(out["content"], str) else json.dumps(out["content"])
    assert "Current state: step 3" in text and "Only tool" not in text and "image_url" not in text
    reply = h.ta._compress_history_message({"role": "assistant", "content": "", "reasoning": "long thoughts"})
    assert not reply.get("reasoning")


def test_p6_persisted_definitions_keep_defs_imports_and_constants_only(h):
    store: dict[str, str] = {}
    h.ta._persist_definitions(store, "import math\nfrom itertools import *\nfrom math import *\n"
                                     "def f(x):\n    return x\nclass C:\n    pass\nX = 3\ny = 4\n")
    assert {"f", "C", "const:X"} <= set(store)
    assert len([k for k in store if k.startswith("import:")]) == 3
    assert not any("y = 4" in v for v in store.values())
    h.ta._persist_definitions(store, "def broken(:\n")
    assert "broken" not in store


def _run(h, code, prelude=(), handler=None):
    grid = [[0] * 8 for _ in range(8)]
    state = {"current_frame": {"ascii": "", "step": 0, "level": 1, "shape": [8, 8], "grid": grid},
             "history": [], "valid_actions": ["UP", "DOWN"], "last_action_result": {}}
    handler = handler or (lambda actions: {"action_result": {"executed": True}, "state": state})
    return h.sandbox.run_sandboxed_python(code=code, timeout_seconds=10, initial_state=state,
                                          action_handler=handler, prelude=list(prelude))


def test_p6_p7_replayed_code_that_prints_or_acts_cannot_hang_or_repeat_actions(h):
    sent = []

    def handler(actions):
        sent.append(actions)
        return {"action_result": {"executed": True}, "state": {}}

    prelude = ["class Probe:\n    print('building')\n    r = action(['UP'])\n", "def g():\n    return 7\n"]
    out = _run(h, "print(g())\nr = action(['DOWN'])\n", prelude=prelude, handler=handler)
    assert out["error"] == "" and out["stdout"].strip() == "7"
    assert sent == [[{"action": "DOWN"}]]  # the kept class body's action(["UP"]) is not replayed


def test_p7_class_idioms_and_exception_names_work(h):
    code = ("class A:\n    def v(self):\n        return 1\nclass B(A, object):\n    @property\n    def w(self):\n"
            "        return super().v() + 1\ntry:\n    {}['k']\nexcept KeyError:\n    print('caught', B().w, __name__)\n"
            "class R:\n    def __str__(self):\n        print('side effect')\n        return 'R'\nresult = R()\n")
    out = _run(h, code)
    assert out["error"] == ""
    assert "caught 2 __main__" in out["stdout"]


def test_p9_board_diff_line(h):
    before = _grid({(1, 1): 9})
    after = _grid({(1, 2): 9})
    line = h.ta._board_diff_line(before, after)
    assert line.startswith("Board diff over that sequence: 2 cells changed in 1 region(s)")
    assert "rows 1-1, cols 1-2" in line
    assert h.ta._board_diff_line(before, before).endswith("no cell changed on the final board.")


def test_p11_reasoning_effort_and_preserve_thinking_knobs(h, monkeypatch):
    kw = {"provider": "vllm", "model": "m", "messages": [], "max_tokens": None, "temperature": 0.6, "top_p": 0.95,
          "top_k": 20}
    monkeypatch.setenv("OURS_REASONING_EFFORT", "medium")
    monkeypatch.setenv("OURS_PRESERVE_THINKING", "0")
    on = h.compat.build_chat_payload(thinking=True, **kw)["chat_template_kwargs"]
    off = h.compat.build_chat_payload(thinking=False, **kw)["chat_template_kwargs"]
    assert on == {"enable_thinking": True, "reasoning_effort": "medium", "preserve_thinking": False}
    assert "reasoning_effort" not in off
    monkeypatch.setenv("OURS_REASONING_EFFORT", "bogus")
    monkeypatch.delenv("OURS_PRESERVE_THINKING")
    assert h.compat.build_chat_payload(thinking=True, **kw)["chat_template_kwargs"] == {"enable_thinking": True}


def test_p12_code_that_cannot_compile_is_a_tool_error_not_a_crash(h):
    stub = types.SimpleNamespace(_ensure_session=lambda path: None)
    for code in ('print("\ud800")', "x = " + "1+" * 200000 + "1", "def f(:\n  pass"):
        result = h.ta.ToolAgent._run_python_tool(stub, None, {"code": code})
        assert '"error"' in str(getattr(result, "content", result))


def test_p13_score_semantics_in_the_system_prompt(h):
    assert "counts the levels completed so far" in h.prompts.STRUCTURED_RUNTIME_STATE_ADDENDUM


def _history(h, levels_and_actions):
    """[(level, action), ...] -> HistoryEntry list whose frames carry those levels (first entry: the start board)."""
    return [h.HistoryEntry(action, h.Frame(_grid({(0, i % 8): 1 + level}), i, level))
            for i, (level, action) in enumerate(levels_and_actions)]


def test_p14_p16_level_start_comparison_and_record(h):
    agent = h.ta.ToolAgent(model="mock")
    hist = _history(h, [(1, ""), (1, "LEFT"), (1, "LEFT"), (1, "UP"), (2, "UP")])
    current = h.Frame(_grid({(3, 3): 12, (5, 5): 12}), 4, 2)
    prompt = agent._build_user_prompt(
        4, valid_actions=["ACTION1", "ACTION2", "ACTION3"], current_frame=current, history_entries=hist,
        previous_step_summary={"executed_count": 1, "executed_actions": ["UP"], "level": 2, "level_transition": True})
    assert "Harness comparison with the first board of level 1 (exact): colours new on this level:" in prompt
    assert "Actions you have never tried in this game: DOWN." in prompt
    assert "level 1 took 4 actions; the last 4: LEFT x2, UP x2" in prompt
    again = agent._build_user_prompt(
        5, valid_actions=["ACTION1", "ACTION2", "ACTION3"], current_frame=current, history_entries=hist,
        previous_step_summary={"executed_count": 1, "executed_actions": ["UP"], "level": 2, "level_transition": True})
    assert "Harness comparison" not in again  # once per level
    assert "level 1 took 4 actions" in again


def test_p15_p18_p19_idle_probe_expectation_and_supervisor(h, monkeypatch):
    monkeypatch.setenv("OURS_GOVERNOR_IDLE_MIN", "0")
    monkeypatch.setenv("OURS_GOVERNOR_LEVEL_MIN", "0")
    monkeypatch.setenv("OURS_SUPERVISOR_MIN", "1")
    agent = h.ta.ToolAgent(model="mock")
    calls = []

    def fake_chat(messages, tools=None, request_timeout_seconds=None):
        calls.append(messages)
        return types.SimpleNamespace(message={"content": "The goal is untested.\nProbe: press UP twice."})

    agent._chat_completion = fake_chat
    hist = _history(h, [(1, ""), (1, "LEFT")])
    current = h.Frame(_grid({(1, 1): 9}), 2, 1)
    kw = {"valid_actions": ["ACTION1", "ACTION2"], "current_frame": current, "history_entries": hist,
          "previous_step_summary": {"executed_count": 0, "executed_actions": [], "level": 1}}
    first = agent._build_user_prompt(2, **kw)
    assert "No action for 0 minutes on this level" in first
    assert "print one line starting with `expect:`" in first
    agent._ours_sup_t -= 120.0  # two minutes on this level since the last review (deterministic, no sleep)
    second = agent._build_user_prompt(2, **kw)
    assert len(calls) == 1 and "You are reviewing an agent" in calls[0][0]["content"]
    assert "Supervisor review of this stalled level" in second
    assert "The goal is untested. | Probe: press UP twice." in second


def test_p17_animation_hint_fires_again_only_after_a_new_animation(h):
    agent = types.SimpleNamespace(
        _animation_awareness_enabled=True, _animation_hint_level=1, _animation_turns_without_progress=0,
        _animation_transient_animations=0, _animation_hint_follow_window=0, _animation_turns_since_hint=99,
        _animation_counted_action=None, _bump_animation_counter=lambda name: None)
    fired, marker = [], None
    for turn in range(1, 40):
        if turn in (1, 2, 3, 20):
            marker = turn
        summary = {"animation": {"transient_pixels": 999}, "end_action_num": marker}
        if h.ta.ToolAgent._animation_hint_line(agent, summary, 1):
            fired.append(turn)
    assert fired == [6, 20]


# --- the notebook builder --------------------------------------------------------------------------------------


def test_builder_inlines_the_patch_source_and_every_requested_patch_applies(tmp_path):
    names = ["P1", "P1B", "P2", "P3", "P7", "P4", "P8", "P9", "P10", "P6", "P12", "P13", "P14", "P15", "P16", "P17",
             "P18", "P19"]
    out = tmp_path / "nb"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_taaf_nb.py"), "--out", str(out), "--slug", "t-arm",
                    "--patches", *names, "--wavefit", "--knob", "OURS_REASONING_EFFORT=medium"],
                   check=True, capture_output=True, text=True)
    nb = json.loads((out / "t-arm.ipynb").read_text())
    cells = ["".join(c["source"]) for c in nb["cells"]]
    ns: dict = {}
    exec(next(c for c in cells if c.startswith("# ours: source of scripts/taaf_ours_patch.py")), ns)
    assert ns["_OURS_PATCH_SOURCE"] == (ROOT / "scripts" / "taaf_ours_patch.py").read_text()
    apply_cell = next(c for c in cells if '_ours_ns["apply"](_OURS_BUNDLE' in c)
    assert repr(names) in apply_cell
    assert any("'OURS_REASONING_EFFORT': 'medium'" in c and "assert os.environ[_k] == _v" in c for c in cells)
    patch_ns = {"__name__": "taaf_ours_patch"}
    exec(compile(ns["_OURS_PATCH_SOURCE"], "taaf_ours_patch.py", "exec"), patch_ns)
    applied = patch_ns["apply"](_copy(tmp_path / "b"), names)
    assert len(applied) == sum(len(tp.PATCHES[n]) for n in names)
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["is_private"] is True and meta["enable_internet"] is False


def test_arm_registry_names_only_known_patches_and_builds(tmp_path):
    registry = json.loads((ROOT / "kaggle" / "taaf" / "arms.json").read_text())
    for arm in registry["arms"]:
        assert set(arm["patches"]) <= set(tp.PATCHES), arm["exp"]
    out = tmp_path / "arms"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_arms.py"), "--out", str(out),
                    "--arms", "exp050", "exp040", "exp043"], check=True, capture_output=True, text=True)
    assert not (out / "exp040").exists()  # dropped arms are not built without --force
    assert not (out / "exp043").exists()  # nor arms already pushed
    nb50 = json.loads((out / "exp050" / "arc3-taaf-fix-kv65-obj.ipynb").read_text())
    text50 = "\n".join("".join(c["source"]) for c in nb50["cells"])
    applied50 = ast.literal_eval(re.search(r'_ours_ns\["apply"\]\(_OURS_BUNDLE, (\[[^\]]*\])\)', text50).group(1))
    assert "P23" in applied50 and "P24" in applied50 and "P21" not in applied50
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_arms.py"), "--out", str(out), "--force",
                    "--arms", "exp043"], check=True, capture_output=True, text=True)
    nb = json.loads((out / "exp043" / "arc3-taaf-ours-h.ipynb").read_text())
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    applied = ast.literal_eval(re.search(r'_ours_ns\["apply"\]\(_OURS_BUNDLE, (\[[^\]]*\])\)', text).group(1))
    knobs = ast.literal_eval(re.search(r"^_KNOBS = (\{.*\})$", text, re.MULTILINE).group(1))
    assert "P19" in applied and knobs["LOCAL_ANALYZER_YIELD_SECONDS"] == "180"
    # exp-035: P4 with a smaller history budget regressed, so the new arms keep the base's budget
    assert "P4" not in applied
    assert "LOCAL_ANALYZER_CONTEXT_WINDOW" not in knobs and "LOCAL_ANALYZER_MAX_OUTPUT" not in knobs


def test_p20_no_impact_learner_and_wrappers():
    """Lever L1 (ported): rows that change on >= 90% of actions become the counter band after 20 actions; an action that
    changes only band rows is flagged no_impact with board_changed False, and the flag reaches the step summary."""

    class Session:
        def __init__(self):
            self.game = types.SimpleNamespace(current_state=None)

        def _execute_action(self, action, **kw):
            self.game.current_state = kw.pop("next_grid")
            return {"executed": True, "board_changed": True, "level_completed": False}

    class Agent:
        def _compact_action_result(self, payload):
            return {"board_changed": payload.get("board_changed")}

        def _summarize_step_sequence(self, action_results):
            return {"executed_count": len(action_results)}

        def _describe_last_outcome(self, summary):
            return "Last executed sequence."

    ns = {"_HarnessGameSession": Session, "ToolAgent": Agent, "_grid_from_state": lambda state: state or ()}
    exec(tp.P20_NEW, ns)
    learner = ns["_ours_hud_update"]
    st = ns["_ours_hud_new"]()
    for i in range(24):
        learner(st, {0, 10 + (i % 7)})
    assert st["band"] == {0}
    assert learner(st, {0}) is True and learner(st, {0, 5}) is False
    st = ns["_ours_hud_new"]()
    assert not any(learner(st, set(range(6))) for _ in range(30)) and st["band"] is None

    session, move = Session(), types.SimpleNamespace(id=types.SimpleNamespace(name="ACTION1"))
    board = [[0] * 4 for _ in range(4)]
    session.game.current_state = tuple(tuple(r) for r in board)
    payload = None
    for step in range(1, 26):  # row 3 (a counter) ticks every action; row step % 3 moves as well until the last one
        board[3][0] = step
        if step < 25:
            board[step % 3][1] = step
        grid = tuple(tuple(r) for r in board)
        payload = Session._execute_action(session, move, next_grid=grid)
    assert payload["no_impact"] is True and payload["board_changed"] is False and payload["hud_rows"] == [3]
    agent = Agent()
    assert agent._compact_action_result(payload)["no_impact"] is True
    summary = agent._summarize_step_sequence([payload])
    assert summary["no_impact_count"] == 1 and "NO impact" in agent._describe_last_outcome(summary)


def test_p20_prompt_line(h):
    agent = h.ta.ToolAgent(model="mock")
    hist = _history(h, [(1, ""), (1, "LEFT")])
    prompt = agent._build_user_prompt(
        2, valid_actions=["ACTION1"], current_frame=h.Frame(_grid({}), 2, 1), history_entries=hist,
        previous_step_summary={"executed_count": 1, "executed_actions": ["LEFT"], "level": 1, "no_impact_count": 1,
                               "hud_rows": [7]})
    assert "1 of these actions changed only the game's counter strip (rows [7]) and had NO impact" in prompt


def test_p21_gate_serves_the_heaviest_waiter_first_and_never_blocks_forever(h):
    gate = h.ta._OursCallGate(1)
    assert gate.acquire(1.0, None) < 0.5  # a free slot is taken at once
    order, threads = [], []

    def wait(name, weight):
        gate.acquire(weight, 30.0)
        order.append(name)
        gate.release()

    for name, weight, pause in (("a", 1.0, 0.0), ("b", 1.0, 0.05), ("heavy", 4.0, 0.1)):
        time.sleep(pause)
        threads.append(threading.Thread(target=wait, args=(name, weight)))
        threads[-1].start()
    time.sleep(0.3)
    gate.release()
    for t in threads:
        t.join(10)
    assert order == ["heavy", "a", "b"]  # weight x wait: 4 x 0.2 s beats 1 x 0.35 s; equal weights go oldest first
    gate.acquire(1.0, None)
    with pytest.raises(h.ta.requests.Timeout):
        gate.acquire(1.0, 0.2)
    with pytest.raises(h.ta.requests.RequestException, match="stopped"):
        gate.acquire(1.0, 30.0, should_stop=lambda: True)
    gate.release()
    assert gate.timeouts == 1 and not gate._waiting


def test_p21_weight_grows_with_level_and_fades_on_a_stalled_level(h, monkeypatch):
    monkeypatch.setenv("OURS_GATE_LEVEL_WEIGHT", "1")
    monkeypatch.setenv("OURS_GATE_STALL_MIN", "45")
    now = time.monotonic()
    agent = types.SimpleNamespace(_ours_gate_level=4, _ours_gate_level_t0=now)
    assert h.ta._ours_gate_weight(agent) == pytest.approx(4.0)
    agent._ours_gate_level_t0 = now - 90 * 60
    assert h.ta._ours_gate_weight(agent) == pytest.approx(2.5, abs=0.01)
    assert h.ta._ours_gate_weight(types.SimpleNamespace()) == 1.0


def test_p21_chat_completion_goes_through_the_gate_and_keeps_the_time_budget(h, monkeypatch):
    monkeypatch.setenv("OURS_GATE_SLOTS", "2")
    h.ta._OURS_GATE_STATE.clear()
    seen = []

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.append(timeout)
        return types.SimpleNamespace(status_code=200, text="", raise_for_status=lambda: None,
                                     json=lambda: {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    monkeypatch.setattr(h.ta.requests, "post", fake_post)
    agent = h.ta.ToolAgent(model="mock")
    try:
        result = agent._chat_completion([{"role": "user", "content": "hi"}], tools=None, request_timeout_seconds=300)
        gate = h.ta._ours_call_gate()
        assert result.message["content"] == "ok" and gate.admitted == 1 and gate._busy == 0
        assert 299.0 <= seen[0] <= 300.0
        monkeypatch.setenv("OURS_GATE_SLOTS", "0")
        h.ta._OURS_GATE_STATE.clear()
        agent._chat_completion([{"role": "user", "content": "hi"}], tools=None, request_timeout_seconds=300)
        assert h.ta._ours_call_gate() is None and seen[1] == 300
    finally:
        h.ta._OURS_GATE_STATE.clear()


def test_p21_review_fixes_print_failure_keeps_the_slot_and_retries_keep_their_place(h, monkeypatch):
    monkeypatch.setenv("OURS_GATE_PRINT_S", "0")

    def broken_print(*args, **kwargs):
        raise OSError("stdout closed")

    monkeypatch.setattr(h.ta, "print", broken_print, raising=False)
    gate = h.ta._OursCallGate(1)
    gate.acquire(1.0, None)  # the print fails, yet the call gets its slot and the slot is counted once
    assert gate._busy == 1
    gate.release()
    assert gate._busy == 0
    monkeypatch.delattr(h.ta, "print")
    # a retry after a timeout keeps its accumulated wait: weight 1 waiting 100 s beats weight 9 waiting 0.3 s
    gate.acquire(1.0, None)
    order = []

    def wait(name, weight, since):
        gate.acquire(weight, 30.0, since=since)
        order.append(name)
        gate.release()

    heavy = threading.Thread(target=wait, args=("heavy", 9.0, None))
    heavy.start()
    time.sleep(0.1)
    old = threading.Thread(target=wait, args=("retry", 1.0, time.monotonic() - 100.0))
    old.start()
    time.sleep(0.2)
    gate.release()
    heavy.join(10)
    old.join(10)
    assert order == ["retry", "heavy"]


def test_p21_no_slot_with_too_little_budget_left(h, monkeypatch):
    monkeypatch.setenv("OURS_GATE_MIN_LEFT_S", "0.5")
    gate = h.ta._OursCallGate(1)
    gate.acquire(1.0, None)
    t0 = time.monotonic()
    with pytest.raises(h.ta.requests.Timeout):
        gate.acquire(1.0, 1.5)  # a 1.5 s budget with 0.5 s kept back gives up after about 1 s, not 1.5 s
    assert time.monotonic() - t0 < 1.3
    gate.release()
    assert gate.acquire(1.0, 0.8) < 0.1  # a budget under twice the reserve takes a free slot at once


@pytest.mark.parametrize("turn_calls,expect", [("2", 2), ("0", 3)])
def test_p21_turn_rule_two_calls_and_zero_means_off(h, monkeypatch, tmp_path, turn_calls, expect):
    import importlib

    rs = importlib.import_module("inference.agent.runtime_state")
    monkeypatch.setenv("OURS_GATE_SLOTS", "2")
    monkeypatch.setenv("OURS_GATE_TURN_CALLS", turn_calls)
    h.ta._OURS_GATE_STATE.clear()
    seen = []

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.append(timeout)
        time.sleep(0.1)
        return types.SimpleNamespace(status_code=200, text="", raise_for_status=lambda: None,
                                     json=lambda: {"choices": [{"message": {"content": "thinking"},
                                                                "finish_reason": "stop"}]})

    monkeypatch.setattr(h.ta.requests, "post", fake_post)
    state = tmp_path / "g_runtime_state.json"
    frame = h.Frame(_grid({}), 0, 1)
    rs.write_runtime_state(state, current_frame=frame, history=[h.HistoryEntry(action="", frame=frame)])
    agent = h.ta.ToolAgent(model="mock")
    agent._yield_seconds = 0.35  # the time rule alone would allow about three 0.1 s calls
    try:
        result = agent.analyze(state, 0, valid_actions=["ACTION1"], transcript_path=tmp_path / "t.txt",
                               request_timeout_seconds=300, should_stop=lambda: False)
    finally:
        h.ta._OURS_GATE_STATE.clear()
    assert getattr(result, "yielded_control", False)
    assert len(seen) == expect if turn_calls == "2" else len(seen) >= expect


def test_p22_a_level_up_is_not_offered_as_the_latest_change(h):
    level1 = [[0] * 8 for _ in range(8)]
    level2 = [[3] * 8 for _ in range(8)]
    history = [{"action": "", "frame": {"ascii": "", "step": 0, "level": 1, "shape": [8, 8], "grid": level1}},
               {"action": "UP", "frame": {"ascii": "", "step": 1, "level": 2, "shape": [8, 8], "grid": level2}}]
    state = {"current_frame": {"ascii": "", "step": 1, "level": 2, "shape": [8, 8], "grid": level2},
             "history": history, "valid_actions": ["UP"], "last_action_result": {}}
    code = "print(previous_frame is None, last_transition is None, len(transitions))\n"
    out = h.sandbox.run_sandboxed_python(code=code, timeout_seconds=10, initial_state=state,
                                         action_handler=lambda actions: {"action_result": {}, "state": state})
    assert out["error"] == "" and out["stdout"].split() == ["True", "True", "1"]
    history[1]["frame"]["level"] = 1  # the same step within a level is still the latest change
    state["current_frame"]["level"] = 1
    out = h.sandbox.run_sandboxed_python(code=code, timeout_seconds=10, initial_state=state,
                                         action_handler=lambda actions: {"action_result": {}, "state": state})
    assert out["stdout"].split() == ["False", "False", "1"]


def _board(objects: dict[tuple[int, int], int], size: int = 16, bg: int = 5) -> tuple[tuple[int, ...], ...]:
    g = [[bg] * size for _ in range(size)]
    for (r, c), v in objects.items():
        g[r][c] = v
    return tuple(tuple(row) for row in g)


L_SHAPE = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)]  # 5 cells in a 4x2 box; its mirror image is not a rotation of it


def _place(shape, r, c, colour):
    return {(r + dr, c + dc): colour for dr, dc in shape}


def _phrases(h, before, after):
    return h.ta._ours_diff_phrases(h.ta._ours_object_diff(before, after))


def test_p23_object_diff_names_moves_turns_appearances_and_bar_changes(h):
    before = _board({**_place(L_SHAPE, 2, 2, 9), (8, 8): 11, **{(15, c): 12 for c in range(10)}})
    turned = [(c, 3 - r) for r, c in L_SHAPE]  # the L rotated 90 degrees clockwise
    after = _board({**_place(turned, 2, 6, 9), (9, 12): 3, **{(15, c): 12 for c in range(9)}})
    text = "; ".join(_phrases(h, before, after))
    assert "b 4x2 (5 px) moved right 4 and rotated 90 clockwise (2,2)->(2,6)" in text
    assert "Y 1x1 vanished from (8,8)" in text and "G 1x1 appeared at (9,12)" in text
    assert "O 1x10 at (15,0) became 1x9" in text  # a shrinking bar is one change, not a vanish and an appear
    assert _phrases(h, _board(_place(L_SHAPE, 2, 2, 9)), _board(_place(L_SHAPE, 5, 2, 9))) == [
        "b 4x2 (5 px) moved down 3 (2,2)->(5,2)"]
    assert _phrases(h, _board(_place(L_SHAPE, 2, 2, 9)), _board(_place(L_SHAPE, 2, 2, 14))) == [
        "b 4x2 (5 px) at (2,2) changed colour to N"]
    many = {(r, c): 9 for r in range(0, 14, 2) for c in range(0, 6, 2)}
    assert _phrases(h, _board(many), _board({(r, c + 1): 9 for r, c in many}))[0].startswith(
        "21 objects moved right 1 together")
    assert _phrases(h, before, before) == []


def test_p23_review_cases_pairing_by_shared_offset_and_moves_before_colour_changes(h):
    blocks = {(5 + i, c + j): 3 for c in (0, 7) for i in range(2) for j in range(2)}
    moved = {(r, c + 5): v for (r, c), v in blocks.items()}
    assert _phrases(h, _board(blocks), _board(moved)) == ["G 2x2 at (5,0), G 2x2 at (5,7) all moved right 5"]
    dots = {(10, c): 3 for c in range(0, 21, 3)}
    assert _phrases(h, _board(dots, size=32), _board({(10, c + 2): 3 for c in range(0, 21, 3)}, size=32)) == [
        "7 objects moved right 2 together (now rows 10-10, cols 2-20)"]
    # g50t: the player moves on and a gray copy appears where it stood: a move and an appearance, not a colour change
    player = _place(L_SHAPE, 2, 2, 9)
    after = _board({**_place(L_SHAPE, 2, 8, 9), **_place(L_SHAPE, 2, 2, 2)})
    assert _phrases(h, _board(player), after) == ["b 4x2 (5 px) moved right 6 (2,2)->(2,8)", "g 4x2 (5 px) appeared at (2,2)"]
    # a selection that moves from one tile to another stays two colour changes
    tiles = {**_place(L_SHAPE, 2, 2, 9), **_place(L_SHAPE, 2, 8, 2)}
    swapped = {**_place(L_SHAPE, 2, 2, 2), **_place(L_SHAPE, 2, 8, 9)}
    assert sorted(_phrases(h, _board(tiles), _board(swapped))) == [
        "b 4x2 (5 px) at (2,2) changed colour to g", "g 4x2 (5 px) at (2,8) changed colour to b"]


def test_p23_review_cases_a_panel_around_a_changed_glyph_is_not_news(h):
    panel = {(r, c): 12 for r in range(2, 9) for c in range(2, 14)}
    ring = {(r, c): 9 for r in range(4, 7) for c in range(5, 8)}
    ring[(5, 6)] = 12
    plus = {(4, 6): 9, (5, 5): 9, (5, 6): 9, (5, 7): 9, (6, 6): 9}
    phrases = _phrases(h, _board({**panel, **ring}, bg=0), _board({**panel, **plus}, bg=0))
    assert "b 3x3 (8 px) at (4,5) became 3x3 (5 px)" in phrases
    assert not any(p.startswith("O 7x12") for p in phrases)
    # a room the sprite moves inside keeps its box and size: not reported
    room = {(r, c): 12 for r in range(2, 8) for c in range(2, 8)}
    inside = _phrases(h, _board({**room, (4, 4): 9}, bg=0), _board({**room, (4, 5): 9}, bg=0))
    assert inside == ["b 1x1 moved right 1 (4,4)->(4,5)"]


def test_p23_crowded_boards_are_summarised_and_fast(h):
    checker = tuple(tuple((r + c) % 2 for c in range(64)) for r in range(64))
    flipped = tuple(tuple(1 - v for v in row) for row in checker)
    t0 = time.time()
    diff = h.ta._ours_object_diff(checker, flipped)
    assert time.time() - t0 < 1.0
    assert diff.get("crowded") and h.ta._ours_diff_phrases(diff) == ["4096 cells changed, too many objects to list"]
    # three actions with 42-frame animations of about 110 flashing pixels over a board of objects
    base = [[0] * 64 for _ in range(64)]
    for r in range(4, 60, 6):
        for c in range(4, 60, 6):
            base[r][c] = base[r][c + 1] = 1 + (r + c) % 7
    base = tuple(tuple(row) for row in base)
    frames = []
    for k in range(42):
        g = [list(row) for row in base]
        for i in range(110):
            g[(i * 7 + k) % 64][(i * 13 + 3 * k) % 64] = 9
        frames.append(tuple(tuple(row) for row in g))
    frames.append(base)
    t0 = time.time()
    lines = h.ta._ours_effect_lines([base, base, base, base], ["A", "B", "C"], {0: frames, 1: frames, 2: frames})
    assert time.time() - t0 < 3.0 and len(lines) == 3


def test_p23_animation_that_moves_a_piece_and_brings_it_back(h):
    start = _board({**_place(L_SHAPE, 2, 2, 9)})
    down2 = _board({**_place(L_SHAPE, 4, 2, 9)})
    down4 = _board({**_place(L_SHAPE, 6, 2, 9)})
    flash = _board({**_place(L_SHAPE, 2, 2, 9), (12, 12): 8})
    lines = h.ta._ours_effect_lines([start, start], ["ACTION6 (5,5)"], {0: [start, down2, down4, flash, start]})
    assert lines == ["- ACTION6 (5,5): no change. During its animation (5 frames): b 4x2 (5 px) at (2,2) moved as far "
                     "as down 4 (frame 2) and was back in place at the end; R 1x1 showed at (12,12) from frame 3, "
                     "gone at the end."]  # frame numbers are those of animation(frame=k)
    # a piece that glides to where it stays is an ordinary move, not something that came back
    assert h.ta._ours_effect_lines([start, down4], ["DOWN"], {0: [start, down2, down4]}) == [
        "- DOWN: b 4x2 (5 px) moved down 4 (2,2)->(6,2)."]
    # objects that grow or fall into place during the animation are not "gone at the end"
    beam = [_board({(5, c): 2 for c in range(n)}) for n in (0, 2, 5, 8)]
    assert h.ta._ours_effect_lines([beam[0], beam[-1]], ["SPACE"], {0: beam}) == ["- SPACE: g 1x8 appeared at (5,0)."]
    fall = [_board({(r + i, 7 + j): 3 for i in range(2) for j in range(2)}) for r in (0, 3, 6, 9, 12)]
    assert h.ta._ours_effect_lines([_board({}), fall[-1]], ["DROP"], {0: [_board({})] + fall}) == [
        "- DROP: G 2x2 appeared at (12,7)."]


def test_p23_prompt_reports_each_sequence_once_unless_the_turn_was_dropped(h):
    agent = h.ta.ToolAgent(model="mock")
    b0 = _board(_place(L_SHAPE, 2, 2, 9))
    b1 = _board(_place(L_SHAPE, 2, 4, 9))
    b2 = _board({**_place(L_SHAPE, 2, 4, 9), (10, 10): 3})
    hist = [h.HistoryEntry("", h.Frame(b0, 0, 1)), h.HistoryEntry("RIGHT", h.Frame(b1, 1, 1)),
            h.HistoryEntry("SPACE", h.Frame(b2, 2, 1))]
    queries = []

    def step_env(args):
        queries.append(args)
        if args.get("action_num") == 1:
            raise RuntimeError("no record")  # a failing lookup costs the animation part only
        flash = _board({**_place(L_SHAPE, 2, 4, 9), (12, 1): 8})  # shaped like the solver's animation record
        return {"executed": False, "query": "animation",
                "record": {"action_num": 2, "before": b1, "frames": [b1, flash, b2], "summary": {}}}

    agent._step_env_callback = step_env
    summary = {"executed_count": 2, "executed_actions": ["RIGHT", "SPACE"], "level": 1, "start_action_num": 1,
               "end_action_num": 2}

    def build(n, entries, s):
        return agent._build_user_prompt(n, valid_actions=["ACTION3", "ACTION5"], current_frame=entries[-1].frame,
                                        history_entries=entries, previous_step_summary=s)

    prompt = build(2, hist, summary)
    assert "Object changes from your last actions 1-2 (computed by the harness" in prompt
    assert ("- RIGHT: b 4x2 (5 px) moved right 2 (2,2)->(2,4).\n- SPACE: G 1x1 appeared at (10,10). During its "
            "animation (3 frames): R 1x1 showed at (12,1) from frame 1, gone at the end.") in prompt
    assert [q["action_num"] for q in queries] == [1, 2] and all(q["query"] == "animation" for q in queries)
    assert "Object changes" in build(2, hist, summary)  # the first turn was dropped (a retry): the report comes again
    agent._history_messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]},
                               {"role": "assistant", "content": "ok"}]
    assert "Object changes" not in build(2, hist, summary)  # kept in history: not repeated
    up = {"executed_count": 1, "executed_actions": ["UP"], "level": 2, "start_action_num": 3, "end_action_num": 3,
          "level_transition": True}
    hist2 = hist + [h.HistoryEntry("UP", h.Frame(b0, 3, 2))]
    assert "Object changes" not in build(3, hist2, up)


def test_p24_shape_match_line_pairs_rotated_and_recoloured_copies_once_per_level(h):
    turned = [(c, 3 - r) for r, c in L_SHAPE]
    mirrored = [(r, 1 - c) for r, c in L_SHAPE]
    board = _board({**_place(L_SHAPE, 1, 1, 9), **_place(turned, 1, 8, 9), **_place(mirrored, 8, 1, 12),
                    **{(8 + r, 8 + c): 11 for r in range(2) for c in range(3)},   # a solid 2x3 rectangle: left out
                    **{(12 + r, 12 + c): 11 for r in range(3) for c in range(2)}})
    line = h.ta._ours_shape_match_line(board, 1)
    assert line.startswith("Harness shape matches, level 1: objects with the same shape up to rotation")
    assert "[1] 5 px: b (1,1), b (1,8) rotated 90 clockwise, O (8,1) mirrored left-right." in line
    assert "Y" not in line.split("):", 1)[1]
    assert h.ta._ours_shape_match_line(_board({})) == ""
    agent = h.ta.ToolAgent(model="mock")
    frame = h.Frame(board, 0, 1)

    def build(n, f, s):
        return agent._build_user_prompt(n, valid_actions=["ACTION1"], current_frame=f, history_entries=[],
                                        previous_step_summary=s)

    first = build(0, frame, None)
    assert "Harness shape matches, level 1" in first
    agent._history_messages = [{"role": "user", "content": first}]
    assert "Harness shape matches" not in build(0, frame, None)  # once per level while that turn is in history
    level2 = build(5, h.Frame(board, 5, 2), {"executed_count": 1, "executed_actions": ["UP"], "level": 2,
                                              "level_transition": True})
    assert "Harness shape matches, level 2" in level2
    many = {}
    for k in range(40):  # 40 pentomino pairs in two orientations: the line stays within its length cap
        r, c = 1 + (k // 8) * 12, 1 + (k % 8) * 8
        shape = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2 + k % 2)]
        many.update(_place(shape, r, c, k % 16))
        many.update(_place([(b, a) for a, b in shape], r + 5, c, k % 16))
    assert len(h.ta._ours_shape_match_line(_board(many, size=64, bg=0))) < 1700


def test_p23_identical_objects_that_vanish_together_are_one_phrase(h):
    dots = {(12, c): 3 for c in (1, 4, 7, 10, 13)}
    assert _phrases(h, _board(dots), _board({})) == ["5 x G 1x1 vanished from (12,1), (12,4), (12,7), (12,10), ..."]


def test_p23_far_apart_dots_are_not_one_moving_object(h):
    assert _phrases(h, _board({(1, 1): 0}), _board({(14, 14): 0})) == ["W 1x1 appeared at (14,14)",
                                                                       "W 1x1 vanished from (1,1)"]
    assert _phrases(h, _board({(1, 1): 0}), _board({(1, 4): 0})) == ["W 1x1 moved right 3 (1,1)->(1,4)"]


def test_p25_action_results_carry_the_object_report(h, tmp_path):
    import importlib

    rs = importlib.import_module("inference.agent.runtime_state")
    state = tmp_path / "g_runtime_state.json"
    b0 = _board(_place(L_SHAPE, 2, 2, 9))
    b1 = _board(_place(L_SHAPE, 2, 5, 9))
    start = h.Frame(b0, 0, 1)
    rs.write_runtime_state(state, current_frame=start, history=[h.HistoryEntry(action="", frame=start)])

    def step_env(args):  # a one-action environment that moves the L right by 3, as the solver would record it
        if args.get("query") == "animation":
            return {"executed": False, "query": "animation", "record": None}
        after = h.Frame(b1, 1, 1)
        rs.write_runtime_state(state, current_frame=after, history=[h.HistoryEntry(action="", frame=start),
                                                                    h.HistoryEntry(action="RIGHT", frame=after)])
        return {"executed": True, "action_num": 1, "level": 1, "score": 0, "state": "NOT_FINISHED",
                "valid_actions": ["ACTION4"], "board_changed": True, "action_display": "RIGHT"}

    agent = h.ta.ToolAgent(model="mock")
    agent._step_env_callback = step_env
    agent._current_valid_actions = ["RIGHT"]
    out = agent._run_python_tool(state, {"code": "r = action(['RIGHT'])\nprint(r['object_changes'])\n"
                                                 "print(last_action_result.get('object_changes'))"})
    text = str(getattr(out, "content", out))
    assert text.count("RIGHT: b 4x2 (5 px) moved right 3 (2,2)->(2,5).") == 2, text
    # a failing report never breaks the action: the result comes back without `object_changes`
    assert h.ta._ours_action_changes(agent, tmp_path / "missing.json", {"action_result": {"executed": True}}) == {
        "action_result": {"executed": True}}
    prompt = agent._build_user_prompt(1, valid_actions=["ACTION4"], current_frame=h.Frame(b1, 1, 1),
                                      history_entries=[], previous_step_summary=None)
    assert "Each `action(...)` result has `object_changes`" in prompt
