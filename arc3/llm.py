"""Minimal OpenAI-compatible chat client (vLLM / OpenRouter) plus a scriptable mock.

Only what the REPL agent needs: one call, optional tools, optional images, usage
accounting, and tolerant parsing of tool calls (structured ``tool_calls`` first, then
Qwen-style ``<tool_call>`` markup in the content as a fallback).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_FN_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*</tool_call>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    raw: Any = None
    finish_reason: str = ""

    def assistant_message(self, *, with_reasoning: bool = False) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if with_reasoning and self.reasoning:
            msg["reasoning_content"] = self.reasoning  # vLLM/Qwen3.8 read this back when preserve_thinking is on
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                for tc in self.tool_calls
            ]
        return msg


def parse_tool_calls_from_text(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for i, m in enumerate(_TOOL_CALL_RE.finditer(text or "")):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = str(obj.get("name", ""))
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"code": args}
        if name:
            calls.append(ToolCall(f"text-{i}", name, dict(args)))
    for i, m in enumerate(_FN_RE.finditer(text or "")):
        params = {k.strip(): v for k, v in _PARAM_RE.findall(m.group(2))}
        calls.append(ToolCall(f"fn-{i}", m.group(1).strip(), params))
    return calls


def strip_tool_call_markup(text: str) -> str:
    text = _TOOL_CALL_RE.sub("", text or "")
    text = _FN_RE.sub("", text)
    return text.strip()


class ChatClient:
    """POST /chat/completions against a vLLM-style server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1", model: str = "", api_key: str = "EMPTY",
                 timeout_s: float = 180.0, extra_body: Optional[dict[str, Any]] = None, session: Any = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.extra_body = dict(extra_body or {})
        self._session = session

    def _http(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def models(self) -> list[str]:
        r = self._http().get(f"{self.base_url}/models", headers=self._headers(), timeout=30)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def chat(self, messages: list[dict[str, Any]], *, tools: Optional[list[dict[str, Any]]] = None,
             max_tokens: int = 4096, temperature: float = 0.6, top_p: float = 0.95,
             thinking: Optional[bool] = None, reasoning_effort: Optional[str] = None,
             preserve_thinking: Optional[bool] = None,
             timeout_s: Optional[float] = None, extra: Optional[dict[str, Any]] = None) -> ChatResponse:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                                "temperature": temperature, "top_p": top_p}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        body.update(self.extra_body)
        if thinking is not None or reasoning_effort or preserve_thinking is not None:
            ctk = dict(body.get("chat_template_kwargs") or {})
            if thinking is not None:
                ctk["enable_thinking"] = bool(thinking)
            if reasoning_effort:  # Qwen3.8 template knob: low | medium | high | xhigh
                ctk["reasoning_effort"] = reasoning_effort
            if preserve_thinking is not None:  # Qwen3.8: keep earlier assistant reasoning in the prompt (continuity within a turn)
                ctk["preserve_thinking"] = bool(preserve_thinking)
            body["chat_template_kwargs"] = ctk
        if extra:
            body.update(extra)
        t0 = time.time()
        r = self._http().post(f"{self.base_url}/chat/completions", headers=self._headers(), json=body,
                              timeout=timeout_s or self.timeout_s)
        if r.status_code >= 400:
            raise RuntimeError(f"chat completion failed {r.status_code}: {r.text[:500]}")
        data = r.json()
        return self._parse(data, time.time() - t0)

    @staticmethod
    def _parse(data: dict[str, Any], latency: float) -> ChatResponse:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        m = _THINK_RE.search(content)
        if m and not reasoning:
            reasoning = m.group(0)
            content = _THINK_RE.sub("", content)
        calls: list[ToolCall] = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"code": args}
            calls.append(ToolCall(str(tc.get("id") or f"call-{i}"), str(fn.get("name") or ""), dict(args)))
        if not calls:
            calls = parse_tool_calls_from_text(content)
            if calls:
                content = strip_tool_call_markup(content)
        usage = data.get("usage") or {}
        return ChatResponse(content=content, reasoning=reasoning, tool_calls=calls,
                            prompt_tokens=int(usage.get("prompt_tokens") or 0),
                            completion_tokens=int(usage.get("completion_tokens") or 0),
                            latency_s=latency, raw=data, finish_reason=str(choice.get("finish_reason") or ""))


class MockClient:
    """Scripted stand-in: ``script`` is a list of ChatResponse (or dicts) or a callable(messages)->ChatResponse."""

    def __init__(self, script: list[Any] | Callable[[list[dict[str, Any]]], Any]):
        self.script = script
        self.calls: list[list[dict[str, Any]]] = []
        self.kwargs: list[dict[str, Any]] = []  # the keyword arguments of each call (reasoning_effort, max_tokens, ...)
        self.model = "mock"
        self.i = 0

    @staticmethod
    def tool(code: str, text: str = "") -> ChatResponse:
        return ChatResponse(content=text, tool_calls=[ToolCall(f"m{time.time_ns()}", "python", {"code": code})],
                            prompt_tokens=100, completion_tokens=50)

    @staticmethod
    def say(text: str) -> ChatResponse:
        return ChatResponse(content=text, prompt_tokens=100, completion_tokens=20)

    def chat(self, messages, **kw) -> ChatResponse:
        self.calls.append([dict(m) for m in messages])
        self.kwargs.append(dict(kw))
        if callable(self.script):
            out = self.script(messages)
        else:
            out = self.script[min(self.i, len(self.script) - 1)] if self.script else self.say("")
            self.i += 1
        if isinstance(out, dict):
            out = ChatClient._parse(out, 0.0)
        return out
