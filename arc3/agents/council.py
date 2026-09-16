"""Council agent: six specialist passes feeding one coordinator that plays through the REPL.

This is the user's "6 x Qwen3-VL-8B specialists + 1 x 27B coordinator" design as an ablation arm.
It subclasses the single-model REPL agent and only changes what the coordinator sees each turn:
a block of short reports from the specialist roles, produced concurrently before the turn.

Config (ctx.config), on top of ReplAgent's keys:
  roles                 list of role names (default: all six)
  specialist            {"base_url", "model", "api_key"} for the specialist server; if absent the
                        coordinator's own client answers the role prompts (shared-model arm)
  specialist_schedule   "events" (default): a round on the first turn of a level, after a world-model mismatch,
                        after a turn without an action, on stagnation, and at least every specialist_every turns;
                        "every": a round every specialist_every turns
  specialist_every      periodic floor / period in turns (default 3 for events, 1 for every)
  specialist_max_round_s when the average round takes longer than this, the periodic floor is dropped and only
                        events trigger a round (default 30; calls are the budget, lesson 0009)
  specialist_sync       True: run the round before the coordinator's call (adds its latency to the turn);
                        False (default): run it concurrently with the coordinator's first call and inject the
                        reports before the next call of the turn, or at the next turn if the turn already ended
  specialist_timeout_s  wall-clock cap for the whole specialist round (default 60)
  specialist_max_tokens (default 400), specialist_thinking (default False), specialist_image (default True)
  specialist_client     object, tests only

The specialists read the same observation as the coordinator (entities with ids and roles, the avatar's key
map, the auto-fitted rules line with coverage and unexplained items, the coordinator's notes, the tile map,
the recent transitions) plus the frame image, so their reports can cite ids and helper calls.
"""
from __future__ import annotations

import base64
import concurrent.futures as cf
import time
from typing import Any, Optional

from ..council_prompts import REPORT_HEADER, ROLES
from ..llm import ChatClient
from ..perception import render_png
from . import register
from .base import AgentContext
from .repl_agent import ReplAgent


@register("council")
class CouncilAgent(ReplAgent):
    def __init__(self, ctx: AgentContext):
        super().__init__(ctx)
        c = ctx.config
        self.roles: list[str] = [r for r in c.get("roles", list(ROLES)) if r in ROLES]
        sp = c.get("specialist") or {}
        self.specialist_client = c.get("specialist_client") or (
            ChatClient(base_url=sp["base_url"], model=sp.get("model", ""), api_key=sp.get("api_key", "EMPTY"),
                       timeout_s=float(c.get("specialist_timeout_s", 60))) if sp.get("base_url") else self.client)
        self.shared_model = self.specialist_client is self.client
        self.specialist_schedule = str(c.get("specialist_schedule", "events"))
        self.specialist_every = max(1, int(c.get("specialist_every", 3 if self.specialist_schedule == "events" else 1)))
        self.specialist_max_round_s = float(c.get("specialist_max_round_s", 30))
        self.specialist_timeout_s = float(c.get("specialist_timeout_s", 60))
        self.specialist_max_tokens = int(c.get("specialist_max_tokens", 400))
        self.specialist_thinking = c.get("specialist_thinking", False)
        self.specialist_image = bool(c.get("specialist_image", True))
        self.specialist_image_scale = int(c.get("specialist_image_scale", 4))
        self.reports: dict[str, tuple[int, str]] = {}  # role -> (turn, text)
        self.cst: dict[str, Any] = {"rounds": 0, "calls": 0, "errors": 0, "timeouts": 0, "prompt_tokens": 0,
                                    "completion_tokens": 0, "time_s": 0.0, "reasons": {}}
        self._last_round_turn = 0
        self._last_round_level = -1
        self._last_round_mismatches = 0
        self.specialist_sync = bool(c.get("specialist_sync", False))
        self._round_pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"council-{ctx.game_id}")
        self._round_future: Optional[cf.Future] = None
        self._reports_version = 0  # bumps when a round delivers reports
        self._injected_version = 0  # the version the coordinator has already seen

    # ------------------------------------------------------------------ specialist round
    def _state_text(self) -> str:
        """What every specialist reads: the coordinator's observation (without its nudges) plus the recent
        transitions and the previous reports."""
        lines = [self._observation_text(include_nudges=False)]
        if self.history:
            lines.append("RECENT TRANSITIONS: " + "; ".join(
                f"{i + 1}. {h['a']} -> {'changed ' + str(h['changed']) + ' cells' if h['changed'] else 'no change'} (L{h['level']})"
                for i, h in enumerate(self.history[-12:])))
        if self.reports:
            lines.append("PREVIOUS REPORTS: " + " || ".join(f"[{r} @turn {t}] {txt[:300]}" for r, (t, txt) in self.reports.items()))
        return "\n".join(lines)

    def _round_reason(self) -> Optional[str]:
        """Why a specialist round should run now, or None."""
        f = self.frame
        if f is None:
            return None
        if self.specialist_schedule != "events":
            return "periodic" if (self.st.turns - 1) % self.specialist_every == 0 else None
        if self.st.turns == 1 or f.levels_completed != self._last_round_level:
            return "level_start"
        wm_mis = int(self.st.wm_checked - self.st.wm_matched)
        if wm_mis > self._last_round_mismatches:
            return "mismatch"
        if not self.last_turn_acted:
            return "idle_turn"
        if self._stagnant():
            return "stagnation"
        avg_round = (self.cst["time_s"] / self.cst["rounds"]) if self.cst["rounds"] else 0.0
        if avg_round <= self.specialist_max_round_s and self.st.turns - self._last_round_turn >= self.specialist_every:
            return "periodic"
        return None

    def _ask(self, role: str, state_text: str, image_b64: Optional[str]) -> tuple[str, str]:
        content: Any = state_text + "\n\nWrite your report."
        if image_b64:
            content = [{"type": "text", "text": content},
                       {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}}]
        msgs = [{"role": "system", "content": ROLES[role]}, {"role": "user", "content": content}]
        r = self.specialist_client.chat(msgs, max_tokens=self.specialist_max_tokens, temperature=0.3,
                                        thinking=self.specialist_thinking, timeout_s=self.specialist_timeout_s)
        self.cst["prompt_tokens"] += r.prompt_tokens
        self.cst["completion_tokens"] += r.completion_tokens
        return role, (r.content or "").strip()[:1200]

    def _run_specialists(self, reason: str = "periodic") -> None:
        if self.frame is None or not self.roles:
            return
        if self.ctx.time_left() < self.specialist_timeout_s + self.min_time_for_turn_s:
            return
        t0 = time.time()
        self.cst["rounds"] += 1
        self.cst["reasons"][reason] = self.cst["reasons"].get(reason, 0) + 1
        self._last_round_turn = self.st.turns
        self._last_round_level = self.frame.levels_completed
        self._last_round_mismatches = int(self.st.wm_checked - self.st.wm_matched)
        state_text = self._state_text()
        img = base64.b64encode(render_png(self.frame.grid, scale=self.specialist_image_scale)).decode() if self.specialist_image else None
        new: dict[str, str] = {}
        with cf.ThreadPoolExecutor(max_workers=len(self.roles)) as ex:
            futs = {ex.submit(self._ask, role, state_text, img): role for role in self.roles}
            try:
                for fut in cf.as_completed(futs, timeout=self.specialist_timeout_s):
                    role = futs[fut]
                    self.cst["calls"] += 1
                    try:
                        _, text = fut.result()
                        if text:
                            new[role] = text
                    except Exception as e:  # noqa: BLE001
                        self.cst["errors"] += 1
                        self.log.warning("specialist %s failed: %s", role, e)
            except cf.TimeoutError:
                self.cst["timeouts"] += 1
                for fut in futs:
                    fut.cancel()
        for role, text in new.items():
            self.reports[role] = (self.st.turns, text)
        if new:
            self._reports_version += 1
            self._empty_rounds = 0
        else:
            # nothing came back (server down, every call failed or timed out): after two such rounds stop asking, so a
            # dead specialist server costs the coordinator nothing more than the single-model arm
            self._empty_rounds = getattr(self, "_empty_rounds", 0) + 1
            if self._empty_rounds >= 2 and self.roles:
                self.log.error("council disabled for %s: two specialist rounds returned nothing", self.ctx.game_id)
                self.cst["disabled_at_turn"] = self.st.turns
                self.roles = []
        dt = time.time() - t0
        self.cst["time_s"] += dt
        self._record("council", turn=self.st.turns, reason=reason, roles=sorted(new), round_s=round(dt, 1),
                     reports={r: t[:400] for r, t in new.items()})

    # ------------------------------------------------------------------ coordinator turn
    def _report_block(self) -> str:
        return REPORT_HEADER + "\n" + "\n".join(
            f"[{role}{'' if t == self.st.turns else f', from turn {t}'}] {text[:600]}" for role, (t, text) in self.reports.items())

    def _user_message(self) -> dict[str, Any]:
        reason = self._round_reason()
        if reason:
            if self.specialist_sync or self.shared_model:
                self._run_specialists(reason)  # a shared server gains nothing from overlap; keep it simple
            elif self._round_future is None or self._round_future.done():
                self._round_future = self._round_pool.submit(self._run_specialists, reason)
        msg = super()._user_message()
        if not self.reports:
            return msg
        self._injected_version = self._reports_version
        block = self._report_block()
        if isinstance(msg["content"], list):
            msg["content"][0]["text"] = block + "\n\n" + msg["content"][0]["text"]
        else:
            msg["content"] = block + "\n\n" + msg["content"]
        return msg

    def _before_model_call(self) -> None:
        """Reports that arrived while the coordinator was busy are handed over before its next call."""
        if self._reports_version > self._injected_version and self.reports:
            self._injected_version = self._reports_version
            self.cst["late_injections"] = self.cst.get("late_injections", 0) + 1
            self.messages.append({"role": "user", "content": "New " + self._report_block()})

    def close(self) -> None:
        try:
            self._round_pool.shutdown(wait=False, cancel_futures=True)
        finally:
            super().close()

    def stats(self) -> dict[str, Any]:
        s = super().stats()
        s["council"] = dict(self.cst, roles=self.roles, shared_model=self.shared_model)
        return s
