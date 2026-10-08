"""Test helpers for the fidelity probe: synthetic request logs in Franzen's format and a fake SGLang-like server."""
from __future__ import annotations

import hashlib
import http.server
import json
import threading
import time
from pathlib import Path

PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAwS2OUAAAAABJRU5ErkJggg=="
TOOL = {"type": "function", "function": {"name": "python", "description": "Run one snippet.",
                                         "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                                                        "required": ["code"]}}}


def _user(text: str, control: str | None = None) -> dict:
    message = {"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": PNG}}]}
    if control:
        message["_arc3_control"] = control
    return message


def write_fake_logs(folder: Path, games: tuple = ("aa01", "bb02"), n: int = 16, trim_at: int = 11,
                    unanswered: int = 5) -> list[Path]:
    """One <game>-<id>_p0_requests.jsonl per game: n requests whose messages extend the previous request's, a history
    trim at ``trim_at``, a resume message tagged with ``_arc3_control`` every 4th turn, and no response for request
    ``unanswered`` (a read timeout)."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for g, game in enumerate(games):
        system = {"role": "system", "content": f"You are a coding agent. Game {game}."}
        history = [system, _user("step 1")]
        window = 0
        lines = []
        for k in range(n):
            if k == trim_at:
                history, window = [system, _user(f"after trim {k}")], k
            request = {"messages": json.loads(json.dumps(history)), "tools": [TOOL],
                       "chat_template_kwargs": {"preserve_thinking": True}, "event": "request", "tool_choice": "auto",
                       "analysis_step": k + 1, "action": k + 1, "request_index_within_turn": 1}
            lines.append(request)
            if k != unanswered:
                usage = {"prompt_tokens": 5000 + 3000 * (k - window) + 100 * g, "completion_tokens": 300,
                         "total_tokens": 0, "prompt_tokens_details": {"cached_tokens": 4000, "image_tokens": 400}}
                lines.append({**request, "event": "response", "usage": usage, "finish_reason": "tool_calls"})
            history.append({"role": "assistant", "reasoning_content": f"thinking about {game} turn {k}",
                            "tool_calls": [{"id": f"call{k}", "type": "function",
                                            "function": {"name": "python", "arguments": json.dumps({"code": f"a({k})"})}}]})
            history.append({"role": "tool", "tool_call_id": f"call{k}", "content": f"result {k}"})
            history.append(_user(f"step {k + 2}", control="resume" if k % 4 == 3 else None))
        path = folder / f"{game}-{g:08x}_p0_requests.jsonl"
        path.write_text("".join(json.dumps(line) + "\n" for line in lines))
        paths.append(path)
    (folder / "requests.jsonl").write_text(json.dumps({"event": "response", "messages": []}) + "\n")  # unattributed
    return paths


class FakeServer:
    """/health, /server_info, /flush_cache and /v1/chat/completions with logprobs, like Pennyroyal SGLang.

    Greedy output is a deterministic function of the messages. ``shift`` adds to every logprob; ``diverge_at``
    makes the server choose its runner-up token from that position on (a different model); ``reject`` lists request
    fields answered with HTTP 400 (a server without an extension)."""

    def __init__(self, *, healthy_after: float = 0.0, reject: tuple = (), info: dict | None = None,
                 shift: float = 0.0, diverge_at: int | None = None, delay: float = 0.02, length: int = 24,
                 fail_first: int = 0):
        self.healthy_after, self.reject, self.shift, self.diverge_at = healthy_after, set(reject), shift, diverge_at
        self.delay, self.length, self.fail_first = delay, length, fail_first  # fail_first: 503s before answering
        self.info = {"json_model_override_args": "{}", "speculative_accept_threshold_single": 1.0,
                     "speculative_accept_threshold_acc": 1.0, "max_running_requests": 10, "version": "fake"}
        self.info.update(info or {})
        self.bodies: list[dict] = []
        self.flushes = 0
        self.lock = threading.Lock()
        self.in_flight = self.max_in_flight = 0

    def __enter__(self) -> FakeServer:
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code: int, obj) -> None:
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/health":
                    self._send(200 if time.time() - server.started >= server.healthy_after else 503, {})
                elif self.path == "/server_info":
                    self._send(200, server.info)
                elif self.path.startswith("/flush_cache"):
                    with server.lock:
                        server.flushes += 1
                    self._send(200, {})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                if self.path != "/v1/chat/completions":
                    self._send(404, {"error": "not found"})
                    return
                with server.lock:
                    transient = server.fail_first > 0
                    server.fail_first -= transient
                if transient:
                    self._send(503, {"object": "error", "message": "not ready"})
                    return
                bad = server.reject & set(body)
                if bad:
                    self._send(400, {"object": "error", "message": f"extra fields not permitted: {sorted(bad)}"})
                    return
                with server.lock:
                    server.bodies.append(body)
                    server.in_flight += 1
                    server.max_in_flight = max(server.max_in_flight, server.in_flight)
                time.sleep(server.delay)
                try:
                    self._send(200, server.completion(body))
                finally:
                    with server.lock:
                        server.in_flight -= 1

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.started = time.time()
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def completion(self, body: dict) -> dict:
        key = hashlib.sha256(json.dumps(body["messages"], sort_keys=True).encode()).digest()
        n = min(int(body.get("max_tokens") or 16), self.length)
        ids, lps, tops = [], [], []
        for j in range(n):
            first = 1000 + 10 * key[j % len(key)]
            cands = [first + c for c in range(5)]
            vals = [-0.05 * (1 + j % 3) + self.shift, -0.08 * (1 + j % 3) + self.shift, -2.0, -3.0, -4.0]
            if self.diverge_at is not None and j >= self.diverge_at:
                cands[0], cands[1] = cands[1], cands[0]
            ids.append(cands[0])
            lps.append(vals[0])
            tops.append(list(zip(vals, cands, strict=True)))
        content = [{"token": f"t{i}", "logprob": lp, "bytes": [],
                    "top_logprobs": [{"token": f"t{c}", "logprob": v, "bytes": []} for v, c in top]}
                   for i, lp, top in zip(ids, lps, tops, strict=True)]
        choice = {"index": 0, "message": {"role": "assistant", "content": "", "reasoning_content": "r" * n,
                                          "tool_calls": None},
                  "logprobs": {"content": content}, "finish_reason": "length", "matched_stop": None}
        if body.get("return_token_ids"):
            choice["response_token_ids"] = ids
        if body.get("return_meta_info"):
            choice["meta_info"] = {"output_token_logprobs": [[lp, i, f"t{i}"] for i, lp in zip(ids, lps, strict=True)],
                                   "output_top_logprobs": [[[v, c, f"t{c}"] for v, c in top] for top in tops],
                                   "spec_accept_length": 2.5, "spec_verify_ct": max(1, n // 2), "e2e_latency": 0.01}
        return {"id": "x", "object": "chat.completion", "model": body["model"], "choices": [choice],
                "usage": {"prompt_tokens": 100, "completion_tokens": n, "total_tokens": 100 + n,
                          "prompt_tokens_details": {"cached_tokens": 0}}}
