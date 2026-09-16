"""Duck-style REPL agent: one strong local model plays by writing and running Python.

Architecture (see CLAUDE.md "Architecture stance", item 1-3):
  * exact perception is handed to the model as variables/helpers (arc3.perception);
  * the model keeps its own world model in a persistent REPL (arc3.sandbox);
  * a time governor stops model calls when the game's deadline nears;
  * a no-LLM explorer is the fallback for idle turns and model/server failures.

Threading: the harness (or the Kaggle framework) calls ``act(frame)`` for one action at a
time. A model turn runs in a worker thread; when model code calls ``act(...)`` the
sandbox request is handed to the harness thread through a queue, and the environment
result is delivered back through ``observe``. This keeps the single-action interface,
so the same class works unchanged in the local harness and in the Kaggle framework.
"""
from __future__ import annotations

import base64
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from arcengine import GameAction, GameState

from ..env import Action, Frame
from ..llm import ChatClient, ChatResponse
from ..perception import ascii as grid_ascii
from ..perception import detect_scale, diff, render_png
from ..prompts import ACTION_NAMES, NAME_TO_ID, SYSTEM_PROMPT, TOOLS
from ..sandbox import PersistentSandbox
from . import register
from .base import Agent, AgentContext
from .explorer import ExplorerAgent

TURN_DONE = object()


@dataclass
class _Req:
    action: Action
    auto: bool = False  # substituted by the agent (e.g. RESET after game over)
    result: Optional[dict[str, Any]] = None


@dataclass
class Stats:
    model_calls: int = 0
    model_errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    sandbox_timeouts: int = 0
    turns: int = 0
    idle_turns: int = 0
    evictions: int = 0
    actions_model: int = 0
    actions_fallback: int = 0
    model_time_s: float = 0.0
    tool_time_s: float = 0.0
    latencies: list[float] = field(default_factory=list)


@register("repl")
class ReplAgent(Agent):
    """Config keys (ctx.config): base_url, model, api_key, context_tokens, max_output_tokens, temperature,
    top_p, thinking, model_timeout_s, tool_timeout_s, max_tool_steps, image, image_scale, image_tokens,
    ascii, history_frames, idle_turns_before_fallback, max_model_errors, client (object, tests only)."""

    def __init__(self, ctx: AgentContext):
        super().__init__(ctx)
        c = ctx.config
        self.client = c.get("client") or ChatClient(base_url=c.get("base_url", "http://127.0.0.1:8000/v1"),
                                                     model=c.get("model", ""), api_key=c.get("api_key", "EMPTY"),
                                                     timeout_s=float(c.get("model_timeout_s", 180)))
        self.context_tokens = int(c.get("context_tokens", 32768))
        self.max_output_tokens = int(c.get("max_output_tokens", 4096))
        self.temperature = float(c.get("temperature", 0.6))
        self.top_p = float(c.get("top_p", 0.95))
        self.thinking = c.get("thinking", None)
        self.model_timeout_s = float(c.get("model_timeout_s", 180))
        self.tool_timeout_s = float(c.get("tool_timeout_s", 30))
        self.max_tool_steps = int(c.get("max_tool_steps", 12))
        self.use_image = bool(c.get("image", True))
        self.image_scale = int(c.get("image_scale", 6))
        self.image_tokens = int(c.get("image_tokens", 400))
        self.use_ascii = bool(c.get("ascii", True))
        self.history_frames = int(c.get("history_frames", 12))
        self.idle_limit = int(c.get("idle_turns_before_fallback", 3))
        self.max_model_errors = int(c.get("max_model_errors", 5))
        self.min_time_for_turn_s = float(c.get("min_time_for_turn_s", 20))
        root = str(Path(__file__).resolve().parents[2])
        self.sandbox = PersistentSandbox(sys_path=[root] + [p for p in sys.path if p])
        self.fallback = ExplorerAgent(ctx)
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.notes: list[str] = []
        self.turn_log: list[str] = []  # compact action summaries for the next user message
        self.frame: Optional[Frame] = None
        self.frames: list[np.ndarray] = []
        self.history: list[dict[str, Any]] = []
        self.last_result: Optional[dict[str, Any]] = None
        self.action_q: "queue.Queue[Any]" = queue.Queue()
        self.result_q: "queue.Queue[Any]" = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.pending: Optional[_Req] = None
        self.closed = False
        self.consecutive_idle = 0
        self.consecutive_errors = 0
        self.use_fallback_only = False
        self.st = Stats()
        self.level_seen = -1

    # ------------------------------------------------------------------ harness interface
    def act(self, frame: Frame) -> Action:
        self.frame = frame
        if not self.frames or self.frames[-1] is not frame.grid:
            self.frames.append(frame.grid)
            del self.frames[: -self.history_frames]
        if frame.state is GameState.NOT_PLAYED:
            self.pending = _Req(Action.reset(), auto=True)
            return self.pending.action
        while True:
            if frame.game_over:
                # Only RESET is legal now; if the model asked for something else, substitute and tell it.
                req = self._take_request(block=False)
                self.pending = _Req(Action.reset(), auto=True) if req is None or req.action.action is not GameAction.RESET else req
                if req is not None and self.pending is not req:
                    req.result = {"auto_reset": True}
                    self._unblocked_request = req
                return self.pending.action
            req = self._take_request(block=self._worker_alive())
            if req is not None:
                self.pending = req
                self.st.actions_model += 1
                return req.action
            if self._worker_alive():
                continue
            # No turn in flight. Decide: model turn, or fallback.
            if self.use_fallback_only or self.consecutive_idle >= self.idle_limit or \
                    self.ctx.time_left() < self.min_time_for_turn_s:
                if self.consecutive_idle >= self.idle_limit:
                    self.consecutive_idle = 0  # give the model another chance after a burst of fallback actions
                    self._fallback_burst = 8
                if getattr(self, "_fallback_burst", 0) > 0 or self.use_fallback_only or self.ctx.time_left() < self.min_time_for_turn_s:
                    self._fallback_burst = max(0, getattr(self, "_fallback_burst", 0) - 1)
                    a = self.fallback.act(frame)
                    self.pending = _Req(a, auto=True)
                    self.st.actions_fallback += 1
                    return a
            self._start_turn()

    def observe(self, action: Action, before: Frame, after: Frame) -> None:
        self.frame = after
        self.frames.append(after.grid)
        del self.frames[: -self.history_frames]
        d = diff(before.grid, after.grid)
        res = {"action": str(action), "changed": int(d.changed), "levels_completed": int(after.levels_completed),
               "level_completed": bool(after.levels_completed > before.levels_completed),
               "game_over": bool(after.game_over), "won": bool(after.done), "state": after.state.name,
               "level_step": int(after.level_step)}
        if d.changed:
            res["changed_bbox"] = d.bbox
        self.history.append({"a": str(action), "changed": int(d.changed), "level": int(after.levels_completed) + 1})
        del self.history[:-40]
        self.last_result = res
        self.turn_log.append(f"{action} -> {'changed %d cells' % d.changed if d.changed else 'no change'}"
                             + (" LEVEL COMPLETED" if res["level_completed"] else "") + (" GAME OVER" if after.game_over else ""))
        self.fallback.observe(action, before, after)
        req = self.pending
        self.pending = None
        if req is not None and not req.auto:
            req.result = {**res, **(req.result or {})}
            self.result_q.put(req)
        elif req is not None and req.auto and getattr(self, "_unblocked_request", None) is not None:
            # The model asked for X during game over; we sent RESET instead. Unblock it with the RESET outcome.
            r = self._unblocked_request
            self._unblocked_request = None
            r.result = {**res, "auto_reset": True, "note": "game was over: RESET was sent instead of your action"}
            self.result_q.put(r)

    def stats(self) -> dict[str, Any]:
        s = self.st.__dict__.copy()
        lat = s.pop("latencies")
        s["model_latency_p50_s"] = round(float(np.median(lat)), 2) if lat else None
        s["sandbox_restarts"] = self.sandbox.restarts
        s["fallback"] = self.fallback.stats()
        return s

    def close(self) -> None:
        self.closed = True
        self.result_q.put(None)
        self.sandbox.stop()

    # ------------------------------------------------------------------ worker side
    def _worker_alive(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def _take_request(self, *, block: bool) -> Optional[_Req]:
        try:
            item = self.action_q.get(block=block, timeout=1.0 if block else None)
        except queue.Empty:
            return None
        if item is TURN_DONE:
            return None
        return item

    def _start_turn(self) -> None:
        self.st.turns += 1
        self.worker = threading.Thread(target=self._turn, name=f"repl-{self.ctx.game_id}", daemon=True)
        self.worker.start()

    def _state_payload(self) -> dict[str, Any]:
        f = self.frame
        assert f is not None
        return {"grid": f.grid.tolist(), "frames": [g.tolist() for g in self.frames[-self.history_frames:]],
                "level": f.level, "levels_completed": f.levels_completed, "win_levels": f.win_levels,
                "step": f.step, "level_step": f.level_step, "state": f.state.name,
                "available": [ACTION_NAMES[a] for a in (f.available_actions or [1, 2, 3, 4, 5, 6]) if a in ACTION_NAMES],
                "history": self.history[-30:], "last": self.last_result}

    def _handle_actions(self, actions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for a in actions:
            act = self._to_action(a)
            req = _Req(act)
            self.action_q.put(req)
            while True:
                if self.closed:
                    raise RuntimeError("agent closed")
                try:
                    got = self.result_q.get(timeout=5.0)
                except queue.Empty:
                    continue
                if got is None:
                    raise RuntimeError("agent closed")
                if got is req:
                    break
            results.append(req.result or {})
            if req.result and (req.result.get("won") or req.result.get("game_over") or req.result.get("level_completed")):
                break
        return results, self._state_payload()

    @staticmethod
    def _to_action(a: dict[str, Any]) -> Action:
        name = str(a.get("action", "")).upper().strip()
        if name not in NAME_TO_ID:
            raise ValueError(f"unknown action {name!r}; use UP/DOWN/LEFT/RIGHT/ACT/CLICK/UNDO/RESET")
        aid = NAME_TO_ID[name]
        if aid == 6:
            x, y = a.get("x"), a.get("y")
            if x is None or y is None:
                raise ValueError("CLICK needs x and y, e.g. act(('CLICK', 12, 30))")
            x, y = int(x), int(y)
            if not (0 <= x <= 63 and 0 <= y <= 63):
                raise ValueError(f"CLICK out of range: ({x},{y}); 0..63")
            return Action.click(x, y)
        return Action(GameAction.from_id(aid))

    def _user_message(self) -> dict[str, Any]:
        f = self.frame
        assert f is not None
        parts: list[str] = []
        left = self.ctx.time_left()
        left_s = "unlimited" if left == float("inf") else f"{int(left // 60)}m{int(left % 60):02d}s"
        avail = ", ".join(ACTION_NAMES[a] for a in (f.available_actions or [1, 2, 3, 4, 5, 6]) if a in ACTION_NAMES)
        parts.append(f"Level {f.level}/{f.win_levels} | step {f.step} | this level: {f.level_step} actions | "
                     f"time left {left_s} | legal: {avail} | state {f.state.name}")
        if self.turn_log:
            parts.append("Since your last turn: " + "; ".join(self.turn_log[-12:]))
            self.turn_log.clear()
        if self.notes:
            parts.append("Your notes:\n- " + "\n- ".join(self.notes[-20:]))
        if self.use_ascii:
            s = detect_scale(f.grid)
            small = f.grid[::s, ::s]
            parts.append(f"Board (ascii, {small.shape[0]}x{small.shape[1]}, scale {s}, hex colours):\n" + grid_ascii(f.grid))
        content: Any = "\n\n".join(parts)
        if self.use_image:
            png = render_png(f.grid, scale=self.image_scale)
            content = [{"type": "text", "text": content},
                       {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]
        return {"role": "user", "content": content}

    def _estimate_tokens(self, msgs: list[dict[str, Any]]) -> int:
        n = 0
        for m in msgs:
            c = m.get("content")
            if isinstance(c, list):
                for part in c:
                    n += self.image_tokens if part.get("type") == "image_url" else len(str(part.get("text", ""))) // 3
            else:
                n += len(str(c or "")) // 3
            for tc in m.get("tool_calls") or []:
                n += len(json.dumps(tc)) // 3
        return n

    def _evict(self) -> None:
        """Keep the estimated prompt under budget: drop whole old turns first, then the oldest
        assistant/tool pairs inside the current turn (always keeping its user message and the
        newest pair)."""
        budget = self.context_tokens - self.max_output_tokens - 1024
        while len(self.messages) > 2 and self._estimate_tokens(self.messages) > budget:
            users = [k for k, m in enumerate(self.messages) if m["role"] == "user"]
            if len(users) >= 2:
                del self.messages[users[0]:users[1]]
                self.st.evictions += 1
                continue
            # Only the current turn is left: trim inside it.
            start = users[0] + 1 if users else 1
            pairs = [k for k in range(start, len(self.messages)) if self.messages[k]["role"] == "assistant"]
            if len(pairs) < 2:
                break
            end = pairs[1]
            del self.messages[pairs[0]:end]
            self.st.evictions += 1

    def _turn(self) -> None:
        acted = False
        nudged = False
        try:
            self.messages.append(self._user_message())
            for _ in range(self.max_tool_steps):
                if self.closed or self.ctx.time_left() < self.min_time_for_turn_s / 2:
                    break
                self._evict()
                t0 = time.time()
                try:
                    resp: ChatResponse = self.client.chat(
                        self.messages, tools=TOOLS, max_tokens=self.max_output_tokens, temperature=self.temperature,
                        top_p=self.top_p, thinking=self.thinking,
                        timeout_s=max(10.0, min(self.model_timeout_s, self.ctx.time_left() - 5)))
                except Exception as e:  # noqa: BLE001
                    self.st.model_errors += 1
                    self.consecutive_errors += 1
                    self.log.warning("model call failed (%d in a row): %s", self.consecutive_errors, e)
                    if self.consecutive_errors >= self.max_model_errors:
                        self.log.error("switching to fallback explorer for the rest of %s", self.ctx.game_id)
                        self.use_fallback_only = True
                    time.sleep(min(5.0, 0.5 * self.consecutive_errors))
                    break
                self.consecutive_errors = 0
                dt = time.time() - t0
                self.st.model_calls += 1
                self.st.model_time_s += dt
                self.st.latencies.append(dt)
                self.st.prompt_tokens += resp.prompt_tokens
                self.st.completion_tokens += resp.completion_tokens
                self.messages.append(resp.assistant_message())
                if not resp.tool_calls:
                    if not acted and not nudged:
                        nudged = True
                        self.messages.append({"role": "user", "content": "No code was run. Use the python tool and end with act(...)."})
                        continue
                    break
                stop = False
                for tc in resp.tool_calls:
                    self.st.tool_calls += 1
                    if tc.name != "python":
                        self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": f"unknown tool {tc.name}; only `python` exists"})
                        continue
                    code = str(tc.arguments.get("code", "") or "")
                    t1 = time.time()
                    r = self.sandbox.run(code, self._state_payload(), timeout_s=self.tool_timeout_s,
                                         action_handler=self._handle_actions)
                    self.st.tool_time_s += time.time() - t1
                    if r.get("timed_out"):
                        self.st.sandbox_timeouts += 1
                    if r.get("error"):
                        self.st.tool_errors += 1
                    if r.get("notes"):
                        self.notes = list(r["notes"])[-40:]
                    if r.get("actions"):
                        acted = True
                    self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": self._tool_text(r)})
                    lr = self.last_result or {}
                    if r.get("actions") and (lr.get("level_completed") or lr.get("game_over") or lr.get("won")):
                        stop = True
                        break
                if stop:
                    break
            self.consecutive_idle = 0 if acted else self.consecutive_idle + 1
            if not acted:
                self.st.idle_turns += 1
        except Exception as e:  # noqa: BLE001
            self.log.exception("turn crashed: %s", e)
            self.st.model_errors += 1
        finally:
            self.action_q.put(TURN_DONE)

    @staticmethod
    def _tool_text(r: dict[str, Any]) -> str:
        parts = []
        if r.get("stdout"):
            parts.append(r["stdout"].rstrip())
        if r.get("result") is not None:
            parts.append("result: " + json.dumps(r["result"], ensure_ascii=False)[:1500])
        if r.get("error"):
            parts.append(r["error"])
        if r.get("actions"):
            parts.append(f"[{r['actions']} action(s) executed; variables refreshed]")
        return "\n".join(parts) or "(no output)"
