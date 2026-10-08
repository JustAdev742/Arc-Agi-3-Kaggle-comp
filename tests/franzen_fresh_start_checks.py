"""Checks of kaggle/franzen/patches/ours-07-fresh-start.patch that need the notebook tree and the franzen_bed venv.

Run by tests/test_ours_fresh_start_patch.py in that venv (arc-agi, arcengine), with the harness environment set:

    python tests/franzen_fresh_start_checks.py units TREE_SRC
    python tests/franzen_fresh_start_checks.py drive TREE_SRC ENV_DIR OUT.json GAME_PREFIX TURNS

TREE_SRC is a notebook tree's ARC3-Inference directory with his patch and ours-01, 02, 04, 03b, 05, 06b and 07
applied.

``units``: the stagnation rule on synthetic histories of generated tokens and clock time (both thresholds, growth, at
most N per level, a level-up starts over, settings read at call time) and the reset: the history is cleared, the trim
path then sends the system prompt, the rebuilt ledger and the opener with its line; the eviction is noted, so the gate
hands the slot over once at the next request and the counts it prices with are untouched; a turn rolled back after
the reset repeats the line.

``drive``: Franzen's real ToolAgent.analyze() over a real game, with the scripted model and the engine-backed step_env
of tests/franzen_ledger_checks.py (replies chosen by request count, so a run with fresh starts plays the same game as
one without). OUT gets, per request, the hashes franzen_ledger_checks records plus where the fresh-start line and the
ledger sit, the roles, whether it was the first request of an analyze() call and whether a tool call lacks its result;
the priority gate's handovers (the request index each came before); the transcript's fresh-start statuses; and the
counts the gate prices a game with.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import franzen_ledger_checks as flc  # reads the same argv (COMMAND TREE ...) and puts TREE on sys.path

ta = flc.ta
LEDGER_HEAD = flc.LEDGER_HEAD
FRESH = "Conversation history was cleared at step"
STATUS = re.compile(r"^ours_fresh_start: conversation history cleared at step (\d+) \(fresh start (\d+) of at most "
                    r"(\d+) on level (\d+)\) after (\d+) generated tokens and ([\d.]+) min on the level without a "
                    r"level-up; (\d+) history messages dropped; ledger (kept|off)\.$", re.M)


# --- units -----------------------------------------------------------------------------------------------------------

def _agent():
    agent = ta.ToolAgent(model="scripted", base_url="http://127.0.0.1:9/v1", provider="vllm")
    agent._session_runtime_dir = Path("/nonexistent/artifacts")
    return agent


def _fires(agent, steps: list[tuple[int, int, float]], clock: list[float]) -> list[tuple[int, int, int]]:
    """Feed (level, tokens generated, seconds passed) per turn start; (turn index, level, count) of each fresh start."""
    fired = []
    for index, (level, tokens, seconds) in enumerate(steps):
        agent._session_generated_tokens += tokens
        clock[0] += seconds
        state = agent._ours_fresh_due(level, clock[0])
        if state is not None:
            fired.append((index, level, state["count"]))
    return fired


def units_rule() -> None:
    env = {"OURS_FRESH_START_TOKENS": "1000", "OURS_FRESH_START_MINUTES": "10", "OURS_FRESH_START_GROWTH": "1.5",
           "OURS_FRESH_START_MAX": "2"}
    os.environ.update(env)
    # defaults, read at call time
    for key in env:
        os.environ.pop(key)
    assert ta.ToolAgent._ours_fresh_settings() == (80000, 1200.0, 1.5, 2), ta.ToolAgent._ours_fresh_settings()
    os.environ.update(env)
    assert ta.ToolAgent._ours_fresh_settings() == (1000, 600.0, 1.5, 2)
    # Each history starts with the first sight of its level (turn 0), from which tokens and time are counted.
    # 1. both thresholds must be reached: tokens come fast and the clock decides (100 tokens and 10 s a turn: 1000
    # tokens at turn 10, 600 s at turn 60) ...
    agent, clock = _agent(), [0.0]
    assert _fires(agent, [(2, 0, 0.0)] + [(2, 100, 10.0)] * 70, clock) == [(60, 2, 1)]
    # ... and the other way round (10 tokens and 60 s a turn: 600 s at turn 10, 1000 tokens at turn 100)
    agent, clock = _agent(), [0.0]
    assert _fires(agent, [(2, 0, 0.0)] + [(2, 10, 60.0)] * 110, clock) == [(100, 2, 1)]
    # 2. growth: the second needs 1.5x the first gap in both (1500 tokens and 900 s after the first), then never again
    agent, clock = _agent(), [0.0]
    fired = _fires(agent, [(3, 0, 0.0)] + [(3, 100, 60.0)] * 80, clock)
    assert fired == [(10, 3, 1), (25, 3, 2)], fired
    # 3. a level-up starts over: the count restarts, the first threshold applies again, and nothing fires on the turn
    # of the level-up however long the old level ran
    agent, clock = _agent(), [0.0]
    steps = [(3, 0, 0.0)] + [(3, 100, 60.0)] * 40 + [(4, 0, 0.0)] + [(4, 100, 60.0)] * 12
    fired = _fires(agent, steps, clock)
    assert fired == [(10, 3, 1), (25, 3, 2), (51, 4, 1)], fired
    # ... and what was generated before the new level was first seen (the turn that won the old one) is not its cost
    agent, clock = _agent(), [0.0]
    fired = _fires(agent, [(1, 999, 599.0), (2, 5000, 0.0), (2, 0, 3000.0), (2, 1000, 0.0)], clock)
    assert fired == [(3, 2, 1)], fired
    # 4. a new game session starts over too
    agent, clock = _agent(), [0.0]
    assert _fires(agent, [(2, 2000, 700.0)], clock) == []  # first sight of the level: nothing yet
    agent._session_runtime_dir = Path("/nonexistent/other")
    assert _fires(agent, [(2, 2000, 700.0)], clock) == []
    assert _fires(agent, [(2, 1000, 600.0)], clock) == [(0, 2, 1)]
    # 5. read at call time: raising the threshold mid-level holds the next fresh start back; at most 0 disables
    agent, clock = _agent(), [0.0]
    _fires(agent, [(2, 0, 0.0)], clock)
    os.environ["OURS_FRESH_START_TOKENS"] = "5000"
    assert _fires(agent, [(2, 1000, 600.0)], clock) == []
    os.environ["OURS_FRESH_START_TOKENS"] = "1000"
    os.environ["OURS_FRESH_START_MAX"] = "0"
    assert _fires(agent, [(2, 1000, 600.0)] * 3, clock) == []
    os.environ["OURS_FRESH_START_MAX"] = "2"
    assert _fires(agent, [(2, 0, 0.0)], clock) == [(0, 2, 1)]
    # 6. the defaults: 80k tokens and 20 min, then 120k and 30 min more
    for key in env:
        os.environ.pop(key)
    agent, clock = _agent(), [0.0]
    fired = _fires(agent, [(5, 0, 0.0)] + [(5, 4000, 60.0)] * 70, clock)  # 4k tokens a minute
    assert fired == [(20, 5, 1), (50, 5, 2)], fired
    os.environ.update(env)


def _turn(i: int) -> list[dict]:
    filler = "x" * 9000
    return [{"role": "user", "content": f"opener {i} " + filler},
            {"role": "assistant", "content": None, "reasoning_content": f"thinking {i} " + filler,
             "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "python", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": f"result {i}"}]


def units_reset() -> None:
    """The reset with the ledger on: everything after the system prompt goes, the ledger comes back (rebuilt, exact),
    the opener carries the line, the retained functions and the state; the gate is asked once; a rollback repeats."""
    os.environ.update(OURS_WIN_LEDGER="1", OURS_FRESH_START_TOKENS="1000", OURS_FRESH_START_MINUTES="0",
                      ARC3_MAX_ACTIVE_STREAMS="1")
    g0, g1 = flc.board(flc.rect(30, 30, 2, 2, 9)), flc.board(flc.rect(26, 30, 2, 2, 9))
    nxt = flc.board(flc.rect(0, 0, 20, 20, 11))
    entries = [flc.entry("", g0, 0, 1), flc.entry("UP", g1, 1, 1), flc.entry("UP", nxt, 2, 2, level_completed=True),
               flc.entry("LEFT", nxt, 3, 2)]
    agent = _agent()
    agent._ours_ledger_observe(entries)
    agent._kept_functions = {"probe": "def probe(frame):\n    return frame\n"}
    agent._tokens_at_level_start, agent._actions_at_level_start = 7, 3
    system = {"role": "system", "content": agent._system_prompt}
    history: list = []
    for i in range(8):
        history += _turn(i)
        history = agent._trim_messages_for_context([system, *history])[1:]
    assert ta._ours_is_ledger(history[0]), "the setup trimmed, so a ledger is pinned"
    agent._history_messages = list(history)
    agent._resume_after_yield = True  # a resumption: the fresh start must open like a new turn
    agent._context_was_trimmed = False
    frame = ta.Frame(grid=nxt, step=3, level=2)
    state_path = Path("/nonexistent/artifacts/ls20-0000_p0_" + ta.RUNTIME_STATE_FILENAME)
    assert agent._ours_fresh_start(state_path, frame, 3) == ""  # first sight of level 2
    assert agent._history_messages == history and not agent._context_was_trimmed
    agent._session_generated_tokens += 1000
    line = agent._ours_fresh_start(state_path, frame, 3)
    assert line == ("Conversation history was cleared at step 4 after 1,000 generated tokens on this level without a "
                    "level-up; the facts stated by the harness are exact."), line
    assert agent._history_messages == [] and agent._context_was_trimmed and agent._has_evicted
    assert agent._resume_after_yield is False
    status = agent._ours_fresh["status"]
    dropped = sum(1 for m in history if not ta._ours_is_ledger(m))
    assert STATUS.match(status) and "(fresh start 1 of at most 2 on level 2)" in status, status
    assert f"; {dropped} history messages dropped; ledger kept." in status and dropped >= 6, status
    # the opener, built as analyze() builds it: the line first, then his text; the retained functions follow
    opener = agent._build_user_message(line + "\nCurrent state: step 4, level 2.", None)
    assert opener["content"].startswith(line) and "probe(frame)" in opener["content"], opener["content"]
    messages = agent._trim_messages_for_context([system, opener])
    assert len(messages) == 3 and messages[0] is system and messages[2] is opener, len(messages)
    ledger = messages[1]
    assert ta._ours_is_ledger(ledger) and ledger is not history[0], "a ledger rebuilt at the reset"
    text = ledger["content"]
    assert text.startswith(LEDGER_HEAD) and "- Level 1: won in 2 actions (steps 1-2), no game over." in text, text
    assert "- Level 2 (current, since step 2): 1 action so far, no game over." in text, text
    assert "- Retained functions: probe." in text, text
    # the gate: one handover at the next request, at the counts a trim would give it; then none
    calls = []
    gate = ta._priority_gate()
    gate.configure_clock(time.monotonic(), time.monotonic() + 3600.0)  # the solver does this; the tail fade needs it
    original = type(gate).handover
    type(gate).handover = lambda self, priority, snapshot=None: calls.append(priority)
    try:
        agent._maybe_handover()
        agent._maybe_handover()
    finally:
        type(gate).handover = original
    assert len(calls) == 1, calls
    assert (agent._tokens_at_level_start, agent._actions_at_level_start) == (7, 3), "the gate's counts are untouched"
    # the turn's later requests and the commit keep the same ledger object right after the system prompt
    turn = [{"role": "assistant", "content": None, "reasoning_content": "r",
             "tool_calls": [{"id": "z", "type": "function", "function": {"name": "python", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "z", "content": "ok"}]
    again = agent._trim_messages_for_context([*messages, *turn])
    assert again[1] is ledger and again[2] is opener and len(again) == 5
    agent._history_messages = agent._persistent_history_messages(again)
    assert agent._history_messages[0] is ledger and agent._history_messages[1] is opener
    assert agent._ours_fresh_start(state_path, frame, 3) == ""  # committed: the line is not repeated
    # a second one needs 1.5x the gap (1500 tokens); a turn rolled back after it repeats the line, once committed not
    agent._session_generated_tokens += 1499
    assert agent._ours_fresh_start(state_path, frame, 5) == ""
    agent._session_generated_tokens += 1
    second = agent._ours_fresh_start(state_path, frame, 5)
    assert second.startswith("Conversation history was cleared at step 6 after 2,500 generated tokens"), second
    assert agent._ours_fresh_start(state_path, frame, 5) == second  # history still empty: the failed turn's retry
    assert agent._ours_fresh["status"].startswith("ours_fresh_start_repeated:")
    agent._history_messages = [opener, *turn]
    assert agent._ours_fresh_start(state_path, frame, 6) == ""
    # at most two on this level, however much more it costs
    agent._session_generated_tokens += 10 ** 6
    assert agent._ours_fresh_start(state_path, frame, 7) == "" and agent._ours_fresh["count"] == 2
    # a level-up: a new count
    agent._session_generated_tokens += 1000
    assert agent._ours_fresh_start(state_path, ta.Frame(grid=g0, step=9, level=3), 9) == ""
    agent._session_generated_tokens += 1000
    assert agent._ours_fresh_start(state_path, ta.Frame(grid=g0, step=12, level=3), 12).startswith(
        "Conversation history was cleared at step 13 after 1,000 generated tokens")
    # without the ledger: the same reset, no ledger message
    os.environ["OURS_WIN_LEDGER"] = "0"
    plain = _agent()
    plain._history_messages = _turn(0) + _turn(1)
    plain._ours_fresh_start(state_path, frame, 3)
    plain._session_generated_tokens += 1000
    line = plain._ours_fresh_start(state_path, frame, 3)
    assert line and plain._history_messages == [] and plain._ours_fresh["status"].endswith("ledger off.")
    opener = plain._build_user_message(line, None)
    assert plain._trim_messages_for_context([system, opener]) == [system, opener]
    os.environ.update(OURS_WIN_LEDGER="1", ARC3_MAX_ACTIVE_STREAMS="0")


def units() -> None:
    units_rule()
    units_reset()
    print("ok")


# --- drive -----------------------------------------------------------------------------------------------------------

def _dangling(wire: list[dict]) -> bool:
    """A tool call without its result, or a result without its call, anywhere in a request."""
    calls: dict[str, int] = {}
    for index, message in enumerate(wire):
        for call in message.get("tool_calls") or []:
            calls[str(call.get("id"))] = index
        if message.get("role") == "tool":
            if str(message.get("tool_call_id")) not in calls:
                return True
            calls.pop(str(message.get("tool_call_id")))
    return bool(calls)


class FreshModel(flc.ScriptedModel):
    """franzen_ledger_checks' scripted model, recording a few more facts per request."""

    def __init__(self, agent, prefix: str):
        super().__init__(agent, prefix)
        self.turn_start = False

    def __call__(self, messages, **kwargs):
        wire = ta._strip_control_keys(ta._apply_summary_visibility(
            list(messages), evicted=getattr(self.agent, "_has_evicted", False)))
        first, self.turn_start = self.turn_start, False
        result = super().__call__(messages, **kwargs)
        self.requests[-1].update(
            n=len(wire), roles=[m.get("role") for m in wire], first_of_turn=first, dangling=_dangling(wire),
            fresh_at=[i for i, m in enumerate(wire) if FRESH in json.dumps(m)])
        return result


def drive(env_dir: str, out: Path, game: str, turns: int) -> None:
    from arcengine import GameState

    work = out.parent / f"{out.stem}-work"
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / f"{game}_p0_{ta.RUNTIME_STATE_FILENAME}"
    transcript = work / "transcript.txt"
    env = flc.EngineEnv(game, env_dir, state_path)
    agent = ta.ToolAgent(model="scripted", base_url="http://127.0.0.1:9/v1", provider="vllm")
    model = FreshModel(agent, game)
    agent._chat_completion = model
    handovers: list[int] = []
    original = ta._PriorityGate.handover

    def handover(self, priority, snapshot=None):
        handovers.append(len(model.requests))  # the index of the request it came before
        return original(self, priority, snapshot)

    ta._PriorityGate.handover = handover
    gate = ta._priority_gate()
    if gate is not None:  # the solver configures the gate's clock before admitting a game; the tail fade needs it
        gate.configure_clock(time.monotonic(), time.monotonic() + 86400.0)
    warnings: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: warnings.append(record.getMessage())
    ta.log.addHandler(handler)
    for turn in range(1, turns + 1):
        if env.raw.state == GameState.GAME_OVER:  # the solver's auto-RESET
            env.execute("RESET", {}, "RESET", automatic=True)
        if env.raw.state == GameState.WIN:
            break
        model.turn_start = True
        agent.analyze(state_path, len(env.entries) - 1, valid_actions=env.valid_actions(), step_env=env.step_env,
                      transcript_path=transcript, analysis_step=turn)
    history = ta._strip_control_keys(list(agent._history_messages))
    text = transcript.read_text()
    out.write_text(json.dumps({
        "requests": model.requests,
        "handovers": handovers,
        "actions": len(env.entries) - 1,
        "game_overs": sum(bool(e.result.get("game_over")) for e in env.entries),
        "level_ups": sum(bool(e.result.get("level_completed")) for e in env.entries),
        "history_sha": hashlib.sha256(json.dumps(history, sort_keys=True).encode()).hexdigest(),
        "statuses": [m.group(0) for m in STATUS.finditer(text)],
        "status_fields": [list(m.groups()) for m in STATUS.finditer(text)],
        "repeated": text.count("ours_fresh_start_repeated:"),
        "warnings": [w for w in warnings if w.startswith("ours fresh start")],
        "gate_inputs": [agent._session_generated_tokens, agent._tokens_at_level_start, agent._actions_at_level_start,
                        agent._pace_last_completed_level],
    }))


if __name__ == "__main__":
    if flc.COMMAND == "units":
        units()
    elif flc.COMMAND == "drive":
        drive(sys.argv[3], Path(sys.argv[4]), sys.argv[5], int(sys.argv[6]))
    else:
        raise SystemExit(f"unknown command {flc.COMMAND!r}")
