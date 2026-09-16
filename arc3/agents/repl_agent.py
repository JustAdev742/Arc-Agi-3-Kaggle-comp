"""Duck-style REPL agent: one strong local model plays by writing and running Python.

Architecture (see CLAUDE.md "Architecture stance", item 1-3):
  * exact perception is handed to the model as variables/helpers (arc3.perception);
  * the model keeps its own world model in a persistent REPL (arc3.sandbox);
  * a time governor stops model calls when the game's deadline nears;
  * a no-LLM rules agent (or the explorer, config fallback_agent) is the fallback for idle turns and model/server failures.

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
from ..entities import Tracker
from ..perception import ascii as grid_ascii
from ..perception import detect_scale, diff, render_png, tile_map
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
    context_overflows: int = 0
    actions_model: int = 0
    actions_fallback: int = 0
    model_time_s: float = 0.0
    tool_time_s: float = 0.0
    wm_checked: int = 0
    wm_matched: int = 0
    wm_errors: int = 0
    rules_fits: int = 0
    rules_time_s: float = 0.0
    rules_coverage: float = 0.0
    effort_raises: int = 0  # turns run at the raised reasoning effort (adaptive policy)
    stagnation_notices: int = 0
    levels_completed: int = 0
    latencies: list[float] = field(default_factory=list)


@register("repl")
class ReplAgent(Agent):
    """Config keys (ctx.config): base_url, model, api_key, context_tokens, max_output_tokens, temperature,
    top_p, thinking, reasoning_effort, model_timeout_s, tool_timeout_s, max_tool_steps, image, image_scale,
    image_tokens, ascii, history_frames, idle_turns_before_fallback, max_model_errors, client (tests only)."""

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
        self.reasoning_effort = c.get("reasoning_effort", None)  # Qwen3.8: low|medium|high|xhigh
        self.preserve_thinking = c.get("preserve_thinking", None)  # Qwen3.8: keep the turn's earlier reasoning in the prompt
        self.model_timeout_s = float(c.get("model_timeout_s", 480))
        self.tool_timeout_s = float(c.get("tool_timeout_s", 30))
        self.max_tool_steps = int(c.get("max_tool_steps", 8))
        self.inspect_steps_before_nudge = int(c.get("inspect_steps_before_nudge", 1))  # exp-007: 2.5 inspection calls per turn at ~30 s each
        self.use_image = bool(c.get("image", True))
        self.image_scale = int(c.get("image_scale", 4))
        self.image_tokens = int(c.get("image_tokens", 600))  # Qwen3.8 ViT: 576 tokens for a 384px image (diag v5)
        self.token_ratio = float(c.get("token_ratio", 1.3))  # server prompt_tokens / our estimate, calibrated per response
        self.context_margin = int(c.get("context_margin", 2048))
        ascii_cfg = c.get("ascii", "auto")  # auto: only when no image is attached
        self.use_ascii = (not self.use_image) if ascii_cfg == "auto" else bool(ascii_cfg)
        self.objects_in_prompt = int(c.get("objects_in_prompt", 16))
        self.auto_rules_in_prompt = bool(c.get("auto_rules_in_prompt", True))
        # Thinking policy: 'fixed' uses reasoning_effort every call; 'adaptive' (default) raises it to effort_raised for a
        # turn when the game is stagnant (the last stagnation_actions actions changed nothing, or two turns without an
        # action) and drops back afterwards, so extra thinking is bought only where it can change the next action.
        self.effort_policy = str(c.get("effort_policy", "adaptive"))
        self.effort_raised = str(c.get("effort_raised", "medium"))
        self.stagnation_actions = int(c.get("stagnation_actions", 6))
        self.recent_changes: list[int] = []  # cells changed by each of the last actions
        self.recent_actions: list[str] = []
        self.level_notice = ""  # shown once, at the first turn after a level is completed
        self.level_archive: list[tuple[list, bool]] = []  # completed levels' symbolic frames (+ simulated final frame)
        self.idle_turns_in_row = 0
        self.tile_map_max_cells = int(c.get("tile_map_max_cells", 1024))  # 32x32 at most in the observation
        self.history_frames = int(c.get("history_frames", 6))
        self.idle_limit = int(c.get("idle_turns_before_fallback", 3))
        self.fallback_burst = int(c.get("fallback_burst", 2))
        self.fallback_cap = int(c.get("fallback_cap", 30))  # per game, while the model is alive
        self.fallback_cap_dead = int(c.get("fallback_cap_dead", 40))  # per game, after the server died
        self.max_model_errors = int(c.get("max_model_errors", 5))
        self.min_time_for_turn_s = float(c.get("min_time_for_turn_s", 45))
        self.min_call_timeout_s = float(c.get("min_call_timeout_s", 120))
        root = str(Path(__file__).resolve().parents[2])
        self.sandbox = PersistentSandbox(sys_path=[root] + [p for p in sys.path if p], max_output_chars=int(c.get("tool_output_chars", 2500)))
        fb = str(c.get("fallback_agent", "rules"))
        if fb == "rules":
            from .rules_agent import RulesAgent
            self.fallback = RulesAgent(ctx)  # code-only probe -> fit -> plan agent (exp-006b 0.20 vs explorer 0.06)
        else:
            self.fallback = ExplorerAgent(ctx)
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.notes: list[str] = []
        self.tracker = Tracker()
        self.tracker_level = -1
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
        self.stop_requested = False
        self.last_turn_acted = True
        self.st = Stats()
        self.level_seen = -1
        self.transcript: list[dict[str, Any]] = []
        self.transcript_dir = c.get("transcript_dir") or getattr(ctx, "out_dir", None)

    # ------------------------------------------------------------------ harness interface
    def is_done(self, frame: Frame) -> bool:
        if frame.done:
            return True
        idle = not self._worker_alive() and self.action_q.empty()
        if idle and not self.use_fallback_only and self.ctx.time_left() < self.min_time_for_turn_s:
            self.stop_requested = True  # too little time for another model turn: stop without spending an action
        if self.stop_requested and idle:
            return True
        return False

    def act(self, frame: Frame) -> Action:
        self.frame = frame
        if self.tracker.bg is None or self.tracker_level != frame.levels_completed:
            self.tracker.reset(frame.grid)
            self.tracker_level = frame.levels_completed
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
            # No turn in flight. Decide: model turn, fallback, or stop.
            if self.use_fallback_only:
                if self.st.actions_fallback >= self.fallback_cap_dead:
                    self.stop_requested = True
                return self._fallback_action(frame)
            if self.ctx.time_left() < self.min_time_for_turn_s:
                # Not enough time for another model turn: stop cleanly rather than spend actions.
                self.stop_requested = True
                return self._fallback_action(frame) if self.st.actions_fallback < self.fallback_cap else Action.reset()
            if getattr(self, "_fallback_burst_left", 0) > 0 and self.st.actions_fallback < self.fallback_cap:
                self._fallback_burst_left -= 1
                return self._fallback_action(frame)
            if self.consecutive_idle >= self.idle_limit and self.st.actions_fallback < self.fallback_cap:
                # The model keeps ending turns without acting: a tiny probe burst to move the state, then back to it.
                self.consecutive_idle = 0
                self._fallback_burst_left = max(0, self.fallback_burst - 1)
                return self._fallback_action(frame)
            self._start_turn()

    def _fallback_action(self, frame: Frame) -> Action:
        a = self.fallback.act(frame)
        self.pending = _Req(a, auto=True)
        self.st.actions_fallback += 1
        return a

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
        self.recent_changes.append(int(d.changed))
        self.recent_actions.append(str(action))
        del self.recent_changes[:-24]
        del self.recent_actions[:-24]
        if res["level_completed"]:
            self.st.levels_completed += 1
            n_level = int(before.level_step) + 1
            last = ", ".join(self.recent_actions[-8:])
            goals = self._archive_level(action)
            goal_txt = (" Win conditions consistent with every completed level so far: " + "; ".join(goals[:5]) + "."
                        if goals else " No win condition is consistent with all completed levels yet.")
            self.level_notice = (f"LEVEL {before.level} COMPLETED after {n_level} actions on it (the last actions were: {last})."
                                 f"{goal_txt} You are now on level {after.level}: the layout changed, so re-read ents(); "
                                 "keep the key map and rules that worked.")
            self.recent_changes.clear()
        if res["level_completed"] or self.tracker_level != after.levels_completed:
            self.tracker.reset(after.grid)
            self.tracker_level = after.levels_completed
            summary = f"{action} -> changed {d.changed} cells"
        else:
            aid = int(action.action.value)
            label: Any = ("CLICK", int(action.x or 0), int(action.y or 0)) if aid == 6 else ACTION_NAMES.get(aid, str(action))
            rec = self.tracker.update(after.grid, label)
            summary = Tracker.describe({**rec, "action": str(action)}, self.tracker)
        self.turn_log.append(summary + (" LEVEL COMPLETED" if res["level_completed"] else "") + (" GAME OVER" if after.game_over else ""))
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

    def _record(self, kind: str, **fields: Any) -> None:
        self.transcript.append({"t": round(time.time(), 1), "kind": kind, **fields})

    def dump_transcript(self, path: Optional[str] = None) -> Optional[str]:
        """Write the compact transcript (model calls, code, tool outputs) as JSONL; returns the path."""
        d = path or self.transcript_dir
        if not d:
            return None
        p = Path(d) / f"{self.ctx.game_id}.transcript.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write(json.dumps({"kind": "meta", "game": self.ctx.game_id, "config": {k: v for k, v in self.ctx.config.items() if k not in ("client", "specialist_client")},
                                "notes": self.notes, "stats": self.stats()}, default=str) + "\n")
            f.writelines(json.dumps(rec, default=str) + "\n" for rec in self.transcript)
        return str(p)

    def close(self) -> None:
        self.closed = True
        self.result_q.put(None)
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=3.0)  # let a finishing turn record its tool output before the dump
        try:
            self.dump_transcript()
        except Exception:  # noqa: BLE001
            self.log.exception("transcript dump failed")
        self.sandbox.stop()

    # ------------------------------------------------------------------ worker side
    def _server_alive(self) -> bool:
        """Cheap liveness check before giving up on the model (mock clients count as alive)."""
        models = getattr(self.client, "models", None)
        if models is None:
            return True
        try:
            models()
            return True
        except Exception:  # noqa: BLE001
            return False

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

    def _archive_level(self, final_action: Action) -> list[str]:
        """Archive the completed level's symbolic frames with a simulated winning frame and return the goal
        predicates consistent with every completed level (the harness-side twin of the REPL's goal_candidates())."""
        try:
            from .. import dsl
            t = self.tracker
            frames = t.compound_frames()
            if not frames:
                return []
            aid = int(final_action.action.value)
            label: Any = ("CLICK", int(final_action.x or 0), int(final_action.y or 0)) if aid == 6 else ACTION_NAMES.get(aid, str(final_action))
            log = dsl.make_log(frames, t.actions, t.unders, t.bg)
            rules, _ = dsl.auto_rules(log, ignore_ids=t.hud_ids()) if len(log) >= 2 else ([], None)
            rules = rules or dsl.fallback_move_rules(t.avatar(), frames[-1])
            dsl.set_terrain(rules, t.under, t.bg)
            # with no rule to simulate the winning step, the last observed frame stands in for the final one
            final = dsl.simulate(frames[-1], label, rules) if rules else frames[-1]
            self.level_archive.append((list(frames) + [final], True))
            return [g["goal"] for g in dsl.goal_predicates(self.level_archive)]
        except Exception as e:  # noqa: BLE001
            self.log.warning("level archive failed: %s", e)
            return []

    def _stagnant(self) -> bool:
        k = self.stagnation_actions
        return k > 0 and len(self.recent_changes) >= k and not any(self.recent_changes[-k:])

    def _untested_actions(self, frame: Frame, limit: int = 5) -> list[str]:
        """Legal actions never tried on this level: keys, ACT, and one click per entity class never clicked."""
        avail = [ACTION_NAMES[a] for a in (frame.available_actions or []) if a in ACTION_NAMES]
        tried = set()
        clicked: set[tuple[int, int]] = set()
        for a in self.tracker.actions:
            if isinstance(a, tuple):
                tried.add("CLICK")
                clicked.add((int(a[1]), int(a[2])))
            else:
                tried.add(str(a))
        out = [k for k in ("UP", "DOWN", "LEFT", "RIGHT", "ACT") if k in avail and k not in tried]
        if "CLICK" in avail and self.tracker.frames:
            roles = self.tracker.roles()
            seen_cls: set[tuple] = set()
            for e in sorted(self.tracker.compound_frames()[-1], key=lambda e: -e.size):
                key = (e.color, e.shape)
                if key in seen_cls or roles.get(e.id) == "hud" or any(e.contains(x, y) for x, y in clicked):
                    continue
                seen_cls.add(key)
                out.append(f"click #{e.id} (colour {e.color}) at ({(e.x0 + e.x1) // 2},{(e.y0 + e.y1) // 2})")
        return out[:limit]

    def _observation_text(self, *, include_nudges: bool = True) -> str:
        """The text the coordinator sees each turn (also given to council specialists without the nudges)."""
        f = self.frame
        assert f is not None
        parts: list[str] = []
        left = self.ctx.time_left()
        left_s = "unlimited" if left == float("inf") else f"{int(left // 60)}m{int(left % 60):02d}s"
        avail = ", ".join(ACTION_NAMES[a] for a in (f.available_actions or [1, 2, 3, 4, 5, 6]) if a in ACTION_NAMES)
        parts.append(f"Level {f.level}/{f.win_levels} | step {f.step} | this level: {f.level_step} actions | "
                     f"time left {left_s} | legal: {avail} | state {f.state.name}")
        if self.level_notice:
            parts.append(self.level_notice)
            if include_nudges:
                self.level_notice = ""
        if self.turn_log:
            parts.append("Since your last turn: " + "; ".join(self.turn_log[-12:]))
            if include_nudges:
                self.turn_log.clear()
        elif include_nudges and self.st.turns > 1 and not self.last_turn_acted:
            parts.append("Your previous turn took NO action. Inspection alone makes no progress: act this turn.")
        if include_nudges and self._stagnant():
            self.st.stagnation_notices += 1
            untested = self._untested_actions(f)
            parts.append(f"STAGNATION: the last {self.stagnation_actions} actions changed nothing. Do not repeat them. "
                         + (f"Untested here: {', '.join(untested)}. " if untested else "")
                         + "Write down the hypotheses you have not tested, then act on the cheapest one.")
        if self.objects_in_prompt > 0:
            ents = self.tracker.entities_summary(self.objects_in_prompt)
            av = self.tracker.avatar()
            parts.append(f"Entities (persistent ids; tile {self.tracker.tile}; roles from evidence): " + "; ".join(
                f"#{e['id']} c{e['color']} @({e['x']},{e['y']}) {e['w']}x{e['h']}" + (f" [{e['role']}]" if e.get('role') and e['role'] != 'unknown' else "")
                for e in ents))
            if av:
                parts.append(f"Avatar: #{av['id']} moves with keys {av['keymap']}")
        if getattr(self, "wm_summary", ""):
            parts.append(self.wm_summary)
        rs = self._rules_summary()
        if rs:
            parts.append(rs)
        if self.notes:
            parts.append("Your notes:\n- " + "\n- ".join(self.notes[-20:]))
        if self.use_ascii:
            s = detect_scale(f.grid)
            small = f.grid[::s, ::s]
            parts.append(f"Board (ascii, {small.shape[0]}x{small.shape[1]}, scale {s}, hex colours):\n" + grid_ascii(f.grid))
        else:
            tile = max(1, int(self.tracker.tile or 1))
            n = -(-f.grid.shape[0] // tile)
            if tile > 1 and n * n <= self.tile_map_max_cells:
                parts.append(f"Board as a tile map ({n}x{n}, one hex colour per {tile}x{tile} tile; row = y // {tile}, column = x // {tile}; "
                             f"pixel (x, y) = (column*{tile}, row*{tile})):\n" + tile_map(f.grid, tile))
            parts.append(f"Board image attached (64x64, logical tile {tile}); grid/objects()/ascii()/tilemap() are in the REPL.")
        return "\n\n".join(parts)

    def _user_message(self) -> dict[str, Any]:
        f = self.frame
        assert f is not None
        content: Any = self._observation_text()
        if self.use_image:
            png = render_png(f.grid, scale=self.image_scale)
            content = [{"type": "text", "text": content},
                       {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]
        return {"role": "user", "content": content}

    def _estimate_tokens(self, msgs: list[dict[str, Any]]) -> int:
        return int(self._raw_tokens(msgs) * self.token_ratio)

    def _raw_tokens(self, msgs: list[dict[str, Any]]) -> int:
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
        budget = self.context_tokens - self.max_output_tokens - self.context_margin
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

    def _force_evict(self) -> bool:
        """Drop the oldest evictable block regardless of the estimate. False if nothing is left to drop."""
        users = [k for k, m in enumerate(self.messages) if m["role"] == "user"]
        if len(users) >= 2:
            del self.messages[users[0]:users[1]]
            self.st.evictions += 1
            return True
        start = users[0] + 1 if users else 1
        pairs = [k for k in range(start, len(self.messages)) if self.messages[k]["role"] == "assistant"]
        if len(pairs) >= 2:
            del self.messages[pairs[0]:pairs[1]]
            self.st.evictions += 1
            return True
        return False

    def _before_model_call(self) -> None:  # noqa: B027
        """Hook run before every model call of a turn (the council injects late specialist reports here)."""

    def _effort_for_turn(self) -> Optional[str]:
        """Reasoning effort for this turn: the configured one, raised when the game is stagnant (adaptive policy)."""
        if self.effort_policy != "adaptive":
            return self.reasoning_effort
        stuck = self._stagnant() or self.idle_turns_in_row >= 2
        if stuck and self.effort_raised and self.effort_raised != self.reasoning_effort:
            self.st.effort_raises += 1
            return self.effort_raised
        return self.reasoning_effort

    def _turn(self) -> None:
        acted = False
        nudged = False
        errored = False
        inspect_only = 0
        effort = self._effort_for_turn()
        try:
            um = self._user_message()
            self.messages.append(um)
            head = um["content"][0]["text"] if isinstance(um["content"], list) else str(um["content"])
            self._record("observation", turn=self.st.turns, effort=effort, text=head[:1500])
            for _ in range(self.max_tool_steps):
                if self.closed or self.ctx.time_left() < self.min_time_for_turn_s / 2:
                    break
                self._before_model_call()
                if not acted and inspect_only >= self.inspect_steps_before_nudge and not nudged:
                    nudged = True
                    lat = self.st.latencies[-8:]
                    per_call = (sum(lat) / len(lat)) if lat else 25.0
                    calls_left = int(self.ctx.time_left() / max(per_call, 5.0))
                    self.messages.append({"role": "user", "content": (
                        f"{inspect_only} inspection step(s) used. Each call costs about {per_call:.0f} s of model time and roughly "
                        f"{calls_left} calls remain for this game. The entity list, events and rules summary above already describe "
                        "the board: call act(...) in your next python call (a single probe is fine), then re-inspect.")})
                self._evict()
                t0 = time.time()
                raw_before = self._raw_tokens(self.messages)
                try:
                    resp: ChatResponse = self.client.chat(
                        self.messages, tools=TOOLS, max_tokens=self.max_output_tokens, temperature=self.temperature,
                        top_p=self.top_p, thinking=self.thinking, reasoning_effort=effort, preserve_thinking=self.preserve_thinking,
                        timeout_s=max(self.min_call_timeout_s, min(self.model_timeout_s, self.ctx.time_left() - 5)))
                except Exception as e:  # noqa: BLE001
                    msg = str(e).lower()
                    if "maximum context length" in msg or "context length" in msg or "too many tokens" in msg:
                        # Our estimate was low: assume the worst, drop history and retry at once (not an outage).
                        self.st.context_overflows += 1
                        self.token_ratio = min(self.token_ratio * 1.25, 3.0)
                        if not self._force_evict():
                            self.log.error("context overflow with nothing left to evict; ending turn")
                            errored = True
                            break
                        continue
                    self.st.model_errors += 1
                    self.consecutive_errors += 1
                    errored = True
                    timed_out = "timed out" in msg or "timeout" in type(e).__name__.lower()
                    self.log.warning("model call failed (%d in a row%s): %s", self.consecutive_errors,
                                     ", timeout" if timed_out else "", e)
                    if timed_out:
                        # Slow, not dead (exp-003: 8 concurrent games pushed calls past 180 s). Give the next call
                        # more room instead of counting toward the dead-server switch.
                        self.model_timeout_s = min(self.model_timeout_s * 1.5, 900.0)
                        self.consecutive_errors = 0
                    elif self.consecutive_errors >= self.max_model_errors and not self._server_alive():
                        self.log.error("model server unreachable: fallback explorer for the rest of %s", self.ctx.game_id)
                        self.use_fallback_only = True
                    time.sleep(min(5.0, 0.5 * self.consecutive_errors))
                    break
                self.consecutive_errors = 0
                if resp.prompt_tokens and raw_before:
                    # Exponential moving average of the server's own count over our estimate.
                    self.token_ratio = 0.7 * self.token_ratio + 0.3 * max(0.5, min(3.0, resp.prompt_tokens / raw_before))
                dt = time.time() - t0
                self.st.model_calls += 1
                self.st.model_time_s += dt
                self.st.latencies.append(dt)
                self.st.prompt_tokens += resp.prompt_tokens
                self.st.completion_tokens += resp.completion_tokens
                self.messages.append(resp.assistant_message(with_reasoning=bool(self.preserve_thinking)))
                self._record("assistant", turn=self.st.turns, reasoning=resp.reasoning[:1500], content=resp.content[:1500],
                             code=[tc.arguments.get("code", "")[:3000] for tc in resp.tool_calls], latency_s=round(dt, 1),
                             prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens)
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
                    wm = r.get("world_model")
                    if wm:
                        self.st.wm_checked, self.st.wm_matched, self.st.wm_errors = int(wm["checked"]), int(wm["matched"]), int(wm["errors"])
                        self.wm_summary = f"world model: {wm['matched']}/{wm['checked']} predictions correct" + (
                            f"; last mismatch {wm['recent_mismatches'][-1]}" if wm["recent_mismatches"] else "")
                        if wm.get("hypotheses"):
                            alive = [k for k, v in wm["hypotheses"].items() if v["alive"]]
                            self.wm_summary += f"; hypotheses alive: {alive} of {list(wm['hypotheses'])}"
                    if r.get("actions"):
                        acted = True
                    else:
                        inspect_only += 1
                    self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": self._tool_text(r)})
                    self._record("tool", turn=self.st.turns, output=self._tool_text(r)[:2000], actions=r.get("actions", 0),
                                 error=bool(r.get("error")), step=self.frame.step if self.frame else None,
                                 level=self.frame.level if self.frame else None)
                    lr = self.last_result or {}
                    if r.get("actions") and (lr.get("level_completed") or lr.get("game_over") or lr.get("won")):
                        stop = True
                        break
                if stop:
                    break
            if acted:
                self.consecutive_idle = 0
                self.idle_turns_in_row = 0
            elif not errored:  # the model answered and chose not to act; an error-ended turn is not idleness
                self.consecutive_idle += 1
                self.idle_turns_in_row += 1
                self.st.idle_turns += 1
            self.last_turn_acted = acted or errored
        except Exception as e:  # noqa: BLE001
            self.log.exception("turn crashed: %s", e)
            self.st.model_errors += 1
        finally:
            self.action_q.put(TURN_DONE)

    def _rules_summary(self) -> str:
        """Auto-fitted rules for the level, shown every turn (time-boxed): the model reads coverage and the
        unexplained items without spending a call, and knows when plan_rules() is worth calling."""
        if not self.auto_rules_in_prompt:
            return ""
        t = self.tracker
        n = len(t.actions)
        if n < 3:
            return ""
        cache = getattr(self, "_rules_cache", None)
        if cache and cache[0] == n:
            return cache[1]
        last_cost = getattr(self, "_rules_cost", 0.0)
        skip = getattr(self, "_rules_skip", 0)
        if last_cost > 2.5 and skip < 3:  # expensive fits: refresh every 4th turn only
            self._rules_skip = skip + 1
            return cache[1] if cache else ""
        self._rules_skip = 0
        t0 = time.time()
        try:
            from .. import dsl
            frames, actions, unders = t.compound_frames()[-121:], t.actions[-120:], t.unders[-121:]
            log = dsl.make_log(frames, actions, unders, t.bg)
            rules, rep = dsl.auto_rules(log, ignore_ids=t.hud_ids())
            self.st.rules_fits += 1
            self.st.rules_coverage = float(rep["coverage"])
            txt = "; ".join(r.describe() for r in rules[:6]) or "none"
            un = rep["unexplained"][:3]
            un_txt = "; ".join(f"#{u['id']} c{u['color']} {u['event']} on {u['action']}" for u in un)
            line = (f"Rules (auto-fitted from {rep['transitions']} transitions, coverage {rep['coverage']:.2f}"
                    f"{', contradictions ' + str(rep['contradictions']) if rep['contradictions'] else ''}): {txt}."
                    + (f" Unexplained: {un_txt}." if un_txt else " Every observed event is explained: set_model(rules_predictor()) and plan_rules(goal) apply.")
                    )
        except Exception as e:  # noqa: BLE001
            self.log.warning("rules summary failed: %s", e)
            line = ""
        self._rules_cost = time.time() - t0
        self.st.rules_time_s += self._rules_cost
        self._rules_cache = (n, line)
        return line

    @staticmethod
    def _result_shown(result: Any, stdout: str) -> bool:
        """True when the cell already printed its act() result (the action label appears in stdout), so echoing
        the same JSON again would only cost prompt tokens (exp-009 transcripts showed every result twice)."""
        first = result[0] if isinstance(result, list) and result else result
        if isinstance(first, dict) and first.get("action"):
            return str(first["action"]) in stdout
        return False

    @staticmethod
    def _tool_text(r: dict[str, Any]) -> str:
        parts = []
        if r.get("stdout"):
            parts.append(r["stdout"].rstrip())
        wm = r.get("world_model")
        if wm and wm.get("checked"):
            parts.append(f"[world model: {wm['matched']}/{wm['checked']} predictions correct" + (
                f"; mismatches: {wm['recent_mismatches'][-2:]}]" if wm["recent_mismatches"] else "]"))
        if r.get("result") is not None and not ReplAgent._result_shown(r["result"], r.get("stdout") or ""):
            parts.append("result: " + json.dumps(r["result"], ensure_ascii=False)[:1500])
        if r.get("error"):
            parts.append(r["error"])
        if r.get("actions"):
            parts.append(f"[{r['actions']} action(s) executed; variables refreshed]")
        return "\n".join(parts) or "(no output)"
