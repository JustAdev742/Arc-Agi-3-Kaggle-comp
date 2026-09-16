"""Council agent: six specialist passes feeding one coordinator that plays through the REPL.

This is the user's "6 x Qwen3-VL-8B specialists + 1 x 27B coordinator" design as an ablation arm.
It subclasses the single-model REPL agent and only changes what the coordinator sees each turn:
a block of short reports from the specialist roles, produced concurrently before the turn.

Config (ctx.config), on top of ReplAgent's keys:
  roles                 list of role names (default: all six)
  specialist            {"base_url", "model", "api_key"} for the specialist server; if absent the
                        coordinator's own client answers the role prompts (shared-model arm)
  specialist_every      run specialists every k coordinator turns (default 1)
  specialist_timeout_s  wall-clock cap for the whole specialist round (default 60)
  specialist_max_tokens (default 400), specialist_thinking (default False), specialist_image (default True)
  specialist_client     object, tests only
"""
from __future__ import annotations

import base64
import concurrent.futures as cf
import time
from typing import Any, Optional

from ..council_prompts import REPORT_HEADER, ROLES
from ..llm import ChatClient
from ..perception import objects_summary, render_png
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
        self.specialist_every = max(1, int(c.get("specialist_every", 1)))
        self.specialist_timeout_s = float(c.get("specialist_timeout_s", 60))
        self.specialist_max_tokens = int(c.get("specialist_max_tokens", 400))
        self.specialist_thinking = c.get("specialist_thinking", False)
        self.specialist_image = bool(c.get("specialist_image", True))
        self.specialist_image_scale = int(c.get("specialist_image_scale", 4))
        self.reports: dict[str, str] = {}
        self.cst = {"rounds": 0, "calls": 0, "errors": 0, "timeouts": 0, "prompt_tokens": 0,
                    "completion_tokens": 0, "time_s": 0.0}

    # ------------------------------------------------------------------ specialist round
    def _state_text(self) -> str:
        f = self.frame
        assert f is not None
        objs = objects_summary(f.grid, limit=30)
        lines = [f"Level {f.level}/{f.win_levels}, step {f.step}, actions this level {f.level_step}, state {f.state.name}."]
        lines.append("OBJECTS (colour, x, y, w, h, size, rect): " + "; ".join(
            f"#{o['id']} c{o['color']} @({o['x']},{o['y']}) {o['w']}x{o['h']} n={o['size']}{' rect' if o['rect'] else ''}" for o in objs))
        if self.history:
            lines.append("RECENT TRANSITIONS: " + "; ".join(
                f"{i + 1}. {h['a']} -> {'changed ' + str(h['changed']) + ' cells' if h['changed'] else 'no change'} (L{h['level']})"
                for i, h in enumerate(self.history[-12:])))
        if self.notes:
            lines.append("COORDINATOR NOTES: " + " | ".join(self.notes[-10:]))
        if self.reports:
            lines.append("PREVIOUS REPORTS: " + " || ".join(f"[{r}] {t[:300]}" for r, t in self.reports.items()))
        return "\n".join(lines)

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

    def _run_specialists(self) -> None:
        if self.frame is None or not self.roles:
            return
        if self.ctx.time_left() < self.specialist_timeout_s + self.min_time_for_turn_s:
            return
        t0 = time.time()
        self.cst["rounds"] += 1
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
        if new:
            self.reports = {r: new.get(r, self.reports.get(r, "")) for r in self.roles if new.get(r) or self.reports.get(r)}
        self.cst["time_s"] += time.time() - t0

    # ------------------------------------------------------------------ coordinator turn
    def _user_message(self) -> dict[str, Any]:
        if (self.st.turns - 1) % self.specialist_every == 0:
            self._run_specialists()
        msg = super()._user_message()
        if not self.reports:
            return msg
        block = REPORT_HEADER + "\n" + "\n".join(f"[{role}] {text}" for role, text in self.reports.items())
        if isinstance(msg["content"], list):
            msg["content"][0]["text"] = block + "\n\n" + msg["content"][0]["text"]
        else:
            msg["content"] = block + "\n\n" + msg["content"]
        return msg

    def stats(self) -> dict[str, Any]:
        s = super().stats()
        s["council"] = dict(self.cst, roles=self.roles, shared_model=self.shared_model)
        return s
