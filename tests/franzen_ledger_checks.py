"""Checks of kaggle/franzen/patches/ours-03-win-ledger.patch that need the notebook tree and the franzen_bed venv.

Run by tests/test_ours_win_ledger_patch.py in that venv (arc-agi, arcengine), with the harness environment set:

    python tests/franzen_ledger_checks.py units TREE_SRC
    python tests/franzen_ledger_checks.py drive TREE_SRC ENV_DIR OUT.json GAME_PREFIX TURNS

TREE_SRC is a notebook tree's ARC3-Inference directory.

``units``: the record builder on synthetic boards, the trim re-insertion of the ledger, the sandbox's `level_wins`.

``drive``: Franzen's real ToolAgent.analyze() turn after turn - openers, the python sandbox, retained functions,
history trimming - against a real game in the offline engine, with two stand-ins: the model (a scripted reply chosen
by request count, so it does not depend on what the prompts say) and the solver's step_env (the same payload fields,
no guards, the auto-RESET after a game over, no clock fields). Two trees therefore see the same game, replies and tool
calls, and their requests can be compared byte for byte. OUT gets, per request, a hash of the exact wire messages,
the hash of each message, and where the ledger and the level records sit.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

COMMAND, TREE = sys.argv[1], sys.argv[2]
sys.path.insert(0, TREE)

from inference.agent import tool_agent as ta  # noqa: E402
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state  # noqa: E402

LEDGER_HEAD = "Ledger of exact facts from the recorded boards"
RECORD_MARK = " record (exact facts from the recorded boards)"


# --- units -----------------------------------------------------------------------------------------------------------

def board(cells: dict, background: int = 5) -> tuple:
    rows = [[background] * 64 for _ in range(64)]
    for (r, c), value in cells.items():
        rows[r][c] = value
    return tuple(tuple(row) for row in rows)


def rect(r: int, c: int, h: int, w: int, value: int) -> dict:
    return {(r + i, c + j): value for i in range(h) for j in range(w)}


def entry(action: str, grid: tuple, step: int, level: int, result: dict | None = None, **flags) -> HistoryEntry:
    return HistoryEntry(action=action, frame=Frame(grid=grid, step=step, level=level), result={**(result or {}), **flags})


def units() -> None:
    # 1. exact object changes: a unique object moved and turned, alike objects listed by place (never paired), a
    # recolour; a HUD bar in the edge band and the background reshaped around the changes are not compared.
    hud_a, hud_b = rect(63, 0, 1, 40, 14), rect(63, 0, 1, 30, 14)
    common = {**rect(30, 10, 2, 2, 9), **rect(30, 20, 2, 2, 9), **rect(5, 40, 2, 3, 6)}
    a = board({**common, (10, 10): 8, (11, 10): 8, (11, 11): 8, **rect(30, 30, 2, 2, 9), **rect(50, 10, 1, 3, 11),
               **hud_a})
    b = board({**common, (20, 30): 8, (20, 31): 8, (21, 30): 8, **rect(40, 40, 2, 2, 9), **rect(50, 10, 1, 3, 12),
               **hud_b})
    assert ta._ours_board_changes(a, b, 4) == [
        "moved: color R 3px [10, 10]->[20, 30] rotated_by 90",
        "recolored Y->O (3px) at [50, 10]",
        "disappeared: color b 4px at [30, 30] (3 alike before, 3 after)",
        "appeared: color b 4px at [40, 40] (3 alike before, 3 after)",
    ], ta._ours_board_changes(a, b, 4)
    assert ta._ours_board_changes(a, a, 4) == []
    assert ta._ours_board_changes(a, b[:10], 4) is None
    # 2. what only follows from a move is left out: the eye carried inside a moved ring (one of three alike dots), a
    # marker the ring now covers
    ring_a = {**rect(10, 10, 3, 3, 9), (11, 11): 0}
    ring_b = {**rect(10, 30, 3, 3, 9), (11, 31): 0}
    dots = {(40, 40): 0, (40, 50): 0}
    a = board({**ring_a, **dots, (12, 32): 14})
    b = board({**ring_b, **dots})
    assert ta._ours_board_changes(a, b, 4) == ["moved: color b 8px [10, 10]->[10, 30]"], ta._ours_board_changes(a, b, 4)
    # 3. more than five changes: five are shown and the rest is announced, not counted
    a = board({(10 + 4 * i, 10): 6 + i for i in range(7)})
    b = board({(10 + 4 * i, 20): 6 + i for i in range(7)})
    text = ta._ours_shown_changes(ta._ours_board_changes(a, b, 4), 5)
    assert text.count("moved: ") == 5 and text.endswith("; more not listed"), text
    # 4. run-length encoding; a long one keeps its head and last run and says how many actions it left out
    assert ta._ours_rle(["UP", "UP", "UP", "LEFT"]) == "UP x3, LEFT"
    long = [f"MOUSE(row={i}, col={i})" for i in range(30)] + ["UP"] * 4
    text = ta._ours_rle(long, 160)
    head, skipped, tail = re.match(r"^(.*), \.\.\. (\d+) more actions \.\.\., (.*)$", text).groups()
    assert len(text) <= 190 and tail == "UP x4" and head.count("MOUSE") + int(skipped) == 30, text
    # 5. the finished board of a level-completing action: the layer before the jump to the next level, else unknown
    pre = board(rect(20, 20, 2, 2, 9))
    done = board(rect(20, 24, 2, 2, 9))
    nxt = board(rect(0, 0, 20, 20, 11))
    assert ta._ours_terminal_grid([pre, done, nxt], pre, nxt) == done
    assert ta._ours_terminal_grid([pre, nxt], pre, nxt) is None
    assert ta._ours_terminal_grid([pre, done, board(rect(20, 28, 2, 2, 9))], pre, board(rect(20, 28, 2, 2, 9))) is None
    assert ta._ours_terminal_grid([pre, done, nxt], done, nxt) is None
    # 6. a level with a game over: the header, the attempt after the RESET, its actions and changes
    g0, g1, g2 = board(rect(30, 30, 2, 2, 9)), board(rect(26, 30, 2, 2, 9)), board(rect(22, 30, 2, 2, 9))
    g4 = board(rect(30, 26, 2, 2, 9))
    entries = [entry("", g0, 0, 1), entry("UP", g1, 1, 1), entry("UP", g2, 2, 1, game_over=True),
               entry("RESET", g0, 3, 1, automatic=True), entry("LEFT", g4, 4, 1),
               entry("RIGHT", nxt, 5, 2, level_completed=True)]
    record = ta._ours_win_record(entries, ta._ours_level_spans(entries)[0], [g4, g0, nxt], 4)
    assert record["opener_lines"][:3] == [
        "Level 1 record (exact facts from the recorded boards): won in 5 actions (steps 1-5), 1 game over: step 2 "
        "(after UP); the winning attempt began after the RESET at step 3.",
        "Actions after that RESET: LEFT, RIGHT.",
        "Before the winning action, changes since that RESET include: moved: color b 4px [30, 30]->[30, 26].",
    ], record["opener_lines"]
    assert record["opener_lines"][3] == ("The winning action itself, before the level ended, made changes including: "
                                         "moved: color b 4px [30, 26]->[30, 30].")
    assert (record["attempt_index"], record["prewin_index"], record["terminal"]) == (3, 4, g0)
    spans = ta._ours_level_spans([*entries, entry("UP", nxt, 6, 2, run_complete=True)])
    assert [s["end"] for s in spans] == [5, 6], spans  # the last level won: no current level is reported
    units_trim(entries)
    units_sandbox(g0)
    print("ok")


def units_trim(entries: list) -> None:
    """The ledger goes in at index 1 once history is dropped from the front, then stays the same object until the
    next drop; it is never pruned or duplicated, and a forced reduction drops a real block instead of it."""
    agent = ta.ToolAgent(model="flashnext", base_url="http://127.0.0.1:9/v1", provider="vllm")
    agent._ours_ledger_observe(entries)
    system = {"role": "system", "content": "system prompt " * 50}
    filler = "x" * 9000
    history: list = []

    def turn(i: int) -> list:
        return [{"role": "user", "content": f"opener {i} " + filler},
                {"role": "assistant", "content": None, "reasoning_content": f"thinking {i} " + filler,
                 "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "python", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": f"c{i}", "content": f"result {i}"}]

    requests, ledgers = [], []
    for i in range(14):
        history += turn(i)
        messages = agent._trim_messages_for_context([system, *history])
        assert agent._estimate_request_input_tokens(messages) <= agent._context_budget_tokens
        found = [m for m in messages if ta._ours_is_ledger(m)]
        assert len(found) <= 1 and all(m is messages[1] for m in found), "one ledger, right after the system prompt"
        requests.append(messages)
        ledgers.append(found[0] if found else None)
        history = messages[1:]
    assert ledgers[0] is None and ledgers[-1] is not None, "no ledger before the first trim, one after it"
    breaks = 0
    for before, after in itertools.pairwise(requests):
        if after[:len(before)] == before:
            if len(before) > 1 and ta._ours_is_ledger(before[1]):
                assert after[1] is before[1], "between trims the ledger is the same message"
            continue
        breaks += 1
        assert after[0] is before[0] and after[1] is not before[1], "a trim diverges right after the system prompt"
    assert breaks >= 2, breaks
    wire = ta._strip_control_keys(requests[-1])[1]
    assert wire["role"] == "user" and wire["content"].startswith(LEDGER_HEAD) and ta._CONTROL_MESSAGE_KEY not in wire
    assert "- Level 1: won in 5 actions (steps 1-5), 1 game over: step 2 (after UP)" in wire["content"]
    assert "- Level 2 (current, since step 5): 0 actions so far, no game over." in wire["content"]
    # never pruned (it is replaced at trims only), so a commit with every pruning knob on keeps it
    assert not ta._prune_control_kind_enabled(ta._OURS_LEDGER_KIND)
    agent._context_was_trimmed = False  # consumed by the gate handover in a real turn
    kept = agent._persistent_history_messages(requests[-1])
    assert kept[0] is requests[-1][1] and not agent._context_was_trimmed, "a commit without a trim keeps it in place"
    # a forced reduction (the server rejected the request) drops a real block: the next trim then rebuilds the ledger
    reduced = agent._force_reduce_messages(requests[-1])
    assert ta._ours_count_history(reduced) < ta._ours_count_history(requests[-1])
    assert not any(ta._ours_is_ledger(m) for m in reduced)
    again = agent._trim_messages_for_context(reduced)
    assert ta._ours_is_ledger(again[1]) and again[1] is not requests[-1][1] and again[2:] == reduced[1:]


def units_sandbox(grid: tuple) -> None:
    from inference.agent.python_tool_sandbox import run_sandboxed_python

    frame = {"ascii": "", "step": 3, "level": 1, "shape": [64, 64], "grid": [list(row) for row in grid]}
    history = [{"action": "", "frame": {**frame, "step": 0}, "result": {}},
               {"action": "UP", "frame": {**frame, "step": 1}, "result": {}}]
    state = {"current_frame": frame, "history": history, "valid_actions": ["UP"], "last_action_call_result": {}}
    code = ("try:\n    w = level_wins\nexcept NameError:\n    print('absent')\nelse:\n"
            "    print(w[0]['level'], w[0]['start_frame'].step, w[0]['prewin_frame'].step, w[0]['win_frame'].level)\n")
    out = run_sandboxed_python(code=code, timeout_seconds=10, initial_state=state, action_handler=lambda *a, **k: {})
    assert (out.get("stdout") or "").strip() == "absent", out
    wins = [{"level": 1, "start_index": 0, "prewin_index": 1, "win_frame": {**frame, "level": 1}}]
    out = run_sandboxed_python(code=code, timeout_seconds=10, initial_state={**state, "level_wins": wins},
                               action_handler=lambda *a, **k: {})
    assert (out.get("stdout") or "").strip() == "1 0 1 1", out


# --- drive -----------------------------------------------------------------------------------------------------------

# Level 1 solutions found by breadth-first search in the engine (scripts/franzen_bed.py uses the same), then wandering
# that runs the budget bar down to a game over.
PLANS = {
    "ls20": (["LEFT"] * 3 + ["UP"] * 4 + ["RIGHT"] * 3 + ["UP"] * 3,
             [["UP", "UP", "LEFT", "DOWN", "RIGHT", "RIGHT"], ["LEFT", "LEFT", "DOWN", "RIGHT", "UP", "UP"]]),
    "vc33": ([{"action": "MOUSE", "row": 32, "col": 60}] * 3,
             [[{"action": "MOUSE", "row": r, "col": c} for r, c in ((17, 1), (25, 1), (45, 1))],
              [{"action": "MOUSE", "row": r, "col": c} for r, c in ((37, 1), (45, 1), (25, 1))]]),
}


def _grid(layer) -> tuple:
    rows = layer.tolist() if hasattr(layer, "tolist") else layer
    return tuple(tuple(int(v) for v in row) for row in rows)


class EngineEnv:
    """The solver's step_env, reduced to what the agent reads, over the offline engine."""

    def __init__(self, prefix: str, env_dir: str, state_path: Path):
        import arc_agi
        from arc_agi import OperationMode

        arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=env_dir)
        game_id = next(e.game_id for e in arc.available_environments if e.game_id.startswith(prefix))
        self.env = arc.make(game_id, seed=0)
        self.raw = self.env.observation_space
        self.state_path = state_path
        self.animation_record = None
        self.entries = [entry("", _grid(self.raw.frame[-1]), 0, 1)]
        self.write()

    def level(self) -> int:
        from arcengine import GameState

        if self.raw.state == GameState.WIN:
            return int(self.raw.win_levels)
        return max(1, min(int(self.raw.win_levels), int(self.raw.levels_completed) + 1))

    def valid_actions(self) -> list[str]:
        from arcengine import GameAction

        names = [GameAction.from_id(int(a)).name for a in self.raw.available_actions or []]
        return [n for n in names if n != "RESET"]

    def write(self) -> None:
        write_runtime_state(self.state_path, current_frame=self.entries[-1].frame, history=self.entries)

    def execute(self, name: str, data: dict, display: str, *, automatic: bool = False, index: int = 1,
                size: int = 1) -> dict:
        from arcengine import GameAction, GameState
        from inference.agent.action_names import to_model_actions
        from inference.utils.animation import collapse_frames, normalize_frames, summarize_animation

        before = self.entries[-1].frame.grid
        completed = int(self.raw.levels_completed)
        self.raw = self.env.step(GameAction[name], data=data)
        frames = normalize_frames(self.raw.frame)
        grid = frames[-1] if frames else before
        interior = any(a[4:-4] != b[4:-4] for a, b in zip(before[4:-4], grid[4:-4]))
        chain = collapse_frames(before, frames)
        summary = summarize_animation(chain) if name != "RESET" else None
        if name != "RESET":
            self.animation_record = {"action_display": display, "chain": chain, "summary": summary} if summary else None
        payload = {
            "executed": True, "action_num": len(self.entries), "level": self.level(),
            "score": int(self.raw.levels_completed), "state": self.raw.state.name,
            "valid_actions": to_model_actions(self.valid_actions()), "board_changed": grid != before,
            "gameplay_changed": interior, "no_op": not interior, "done": self.raw.state == GameState.WIN,
            "level_completed": int(self.raw.levels_completed) > completed and self.raw.state != GameState.WIN,
            "game_over": self.raw.state == GameState.GAME_OVER, "run_complete": self.raw.state == GameState.WIN,
            "action_name": name, "action_data": data, "action_display": display, "batch_index": index,
            "batch_size": size, "automatic": automatic,
        }
        if summary:
            payload["animation"] = summary
            payload["animation_chain"] = chain
        keys = ("executed", "action_num", "level", "score", "state", "valid_actions", "board_changed",
                "gameplay_changed", "no_op", "done", "level_completed", "game_over", "run_complete", "action_name",
                "action_data", "action_display", "animation", "automatic")
        result = json.loads(json.dumps({k: payload[k] for k in keys if k in payload}))
        self.entries.append(entry(display, grid, len(self.entries), self.level(), result))
        self.write()
        return payload

    def step_env(self, arguments: dict) -> dict:
        from inference.agent.action_names import to_engine_action

        if str(arguments.get("query") or "") == "animation":
            return {"executed": False, "query": "animation", "record": self.animation_record}
        requested = list(arguments.get("actions") or [])
        executed, stop = [], None
        for index, item in enumerate(requested, 1):
            label = str(item.get("action", "")).upper()
            if label == "MOUSE":
                name, data = "ACTION6", {"x": int(item["col"]), "y": int(item["row"])}
                display = f"MOUSE(row={int(item['row'])}, col={int(item['col'])})"
            else:
                name, data, display = to_engine_action(label), {}, label
            payload = self.execute(name, data, display, index=index, size=len(requested))
            executed.append(payload)
            stop = next((k for k in ("run_complete", "game_over", "level_completed") if payload[k]), None)
            if stop:
                break
        final = dict(executed[-1])
        final.update(requested_count=len(requested), executed_count=len(executed),
                     executed_actions=[p["action_display"] for p in executed],
                     board_changed=any(p["board_changed"] for p in executed),
                     gameplay_changed=any(p["gameplay_changed"] for p in executed),
                     stopped_early=len(executed) < len(requested))
        if stop:
            final["stop_reason"] = stop
        return final


class ScriptedModel:
    """Replies chosen by request count: inspect, act, a long print (fills the context), act, a reply without a tool
    call. Acting replies take the next batch of the game's plan; every tenth inspection defines a function."""

    CYCLE = ("inspect", "act", "print", "act", "chat")

    def __init__(self, agent, prefix: str):
        self.agent, self.n = agent, 0
        self.solution, self.wander = PLANS[prefix]
        self.solved_sent = False
        self.wander_index = 0
        self.requests: list[dict] = []

    def next_batch(self) -> list:
        if not self.solved_sent:
            self.solved_sent = True
            return list(self.solution)
        batch = self.wander[self.wander_index % len(self.wander)]
        self.wander_index += 1
        return list(batch)

    def __call__(self, messages, *, tools=None, request_timeout_seconds=None, max_tokens=None, enable_thinking=None):
        wire = ta._strip_control_keys(ta._apply_summary_visibility(
            list(messages), evicted=getattr(self.agent, "_has_evicted", False)))
        text = json.dumps(wire, sort_keys=True)
        tool_text = " ".join(str(m.get("content")) for m in wire if m.get("role") == "tool")
        self.requests.append({
            "sha": hashlib.sha256(text.encode()).hexdigest(),
            "keys": [hashlib.sha1(json.dumps(m, sort_keys=True).encode()).hexdigest()[:16] for m in wire],
            "ledger_at": [i for i, m in enumerate(wire) if str(m.get("content", "")).startswith(LEDGER_HEAD)],
            "ledger": next((m["content"] for m in wire if str(m.get("content", "")).startswith(LEDGER_HEAD)), None),
            "records": text.count(RECORD_MARK),
            "level_wins_seen": max([0, *map(int, re.findall(r"LEVEL_WINS (\d+)", tool_text))]),
        })
        self.n += 1
        program = self.CYCLE[(self.n - 1) % len(self.CYCLE)]
        reasoning = f"Scripted reasoning for request {self.n}: running the {program} step. " * 6
        usage = {"prompt_tokens": len(text) // 4, "completion_tokens": 120, "total_tokens": len(text) // 4 + 120}
        if program == "chat":
            message = {"role": "assistant", "content": f"Request {self.n}: I will look again.",
                       "reasoning_content": reasoning}
            return ta._ChatCompletionResult(message=message, finish_reason="stop", usage=usage)
        if program == "inspect":
            code = ("nodes = current_frame.segmentation['nodes']\nprint(len(nodes), current_frame.level, "
                    "current_frame.step)\ntry:\n    print('LEVEL_WINS', len(level_wins))\nexcept NameError:\n"
                    "    print('no level_wins')")
            if self.n % 10 == 1:
                code += "\n\ndef drive_probe(frame):\n    return len(frame.segmentation['nodes'])\n"
        elif program == "print":
            code = "for _ in range(6):\n    print(current_frame.ascii)"
        else:
            code = f"r = action({json.dumps(self.next_batch())})\nprint(r.get('executed_count'), r.get('level'))"
        call = {"id": f"call_{self.n}", "type": "function",
                "function": {"name": "python", "arguments": json.dumps({"code": code})}}
        message = {"role": "assistant", "content": None, "reasoning_content": reasoning, "tool_calls": [call]}
        return ta._ChatCompletionResult(message=message, finish_reason="tool_calls", usage=usage)


def drive(env_dir: str, out: Path, game: str, turns: int) -> None:
    from arcengine import GameState

    work = out.parent / f"{out.stem}-work"
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / "game_tool_runtime_state.json"
    transcript = work / "transcript.txt"
    env = EngineEnv(game, env_dir, state_path)
    agent = ta.ToolAgent(model="flashnext", base_url="http://127.0.0.1:9/v1", provider="vllm")
    model = ScriptedModel(agent, game)
    agent._chat_completion = model
    for turn in range(1, turns + 1):
        if env.raw.state == GameState.GAME_OVER:  # the solver's auto-RESET
            env.execute("RESET", {}, "RESET", automatic=True)
        if env.raw.state == GameState.WIN:
            break
        agent.analyze(state_path, len(env.entries) - 1, valid_actions=env.valid_actions(), step_env=env.step_env,
                      transcript_path=transcript, analysis_step=turn)
    history = ta._strip_control_keys(list(agent._history_messages))
    text = transcript.read_text()
    out.write_text(json.dumps({
        "requests": model.requests,
        "actions": len(env.entries) - 1,
        "game_overs": sum(bool(e.result.get("game_over")) for e in env.entries),
        "level_ups": sum(bool(e.result.get("level_completed")) for e in env.entries),
        "history_sha": hashlib.sha256(json.dumps(history, sort_keys=True).encode()).hexdigest(),
        # turn headers carry the wall-clock time (`| 16:40:18 | tool-agent ---`); nothing else in it is timed
        "transcript_sha": hashlib.sha256(re.sub(r"\| \d\d:\d\d:\d\d \|", "| T |", text).encode()).hexdigest(),
        "transcript_records": [line for line in text.splitlines() if RECORD_MARK in line],
    }))


if __name__ == "__main__":
    if COMMAND == "units":
        units()
    elif COMMAND == "drive":
        drive(sys.argv[3], Path(sys.argv[4]), sys.argv[5], int(sys.argv[6]))
    else:
        raise SystemExit(f"unknown command {COMMAND!r}")
