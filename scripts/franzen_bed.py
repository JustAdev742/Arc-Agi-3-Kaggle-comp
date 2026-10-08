#!/usr/bin/env python
"""CPU test bed for Daniel Franzen's harness: his real harness, real public games, a scripted mock model server.

    .venv/bin/python scripts/franzen_bed.py                                  # his notebook, ls20 + vc33 + sb26, 150 s
    .venv/bin/python scripts/franzen_bed.py --patch ours.patch --set OURS_X=1 --expect "some prompt text"
    .venv/bin/python scripts/franzen_bed.py --notebook built/arm.ipynb --games ls20,ft09 --seconds 300

What runs is what the notebook runs, minus the GPU: the arm notebook (his, or one built on the fly by
scripts/build_franzen_nb.py from --env/--env-add/--patch, or --notebook) gives his patch (cell 2), our patches
(their %%writefile cells), the harness environment (cell 4's ``setup_env``, evaluated from the cell's own code) and
the solver settings (cell 16, executed as in a Save & Run). scripts/franzen_tree.py builds the exact source bundle
and applies the patches the way cell 4 does. A child process with a Kaggle-like venv (arc-agi 0.9.9, arcengine
0.9.3, numpy, scipy, matplotlib, imageio(-ffmpeg), pillow, requests; created once with uv under
~/.cache/arc3-franzen-bed/venv, or --python) then does cells 10, 14, 16 and 20: source roots on sys.path, the pickled
TAAF benchmark and deployment target, offline arcade on environment_files/, ``await bm.run(...)``.

The model is a scripted OpenAI-compatible server in this process. Each reply carries one ``python`` tool call (or
deliberately none) from a fixed cycle of small programs, chosen statelessly from the request: a marker comment in
the last tool call it emitted says which snippet comes next. The programs exercise, in turn:

- ``solve``: on a game's first turn, a known level-1 solution (ls20: 13 moves; vc33: 3 clicks, found by BFS in the
  engine), so later turns run on level 2 where his per-action guards are armed (ARC3_GUARDS_FROM_LEVEL=2);
- ``define``: defines ``bed_pick`` (retained by ARC3_PERSISTENT_FUNCTIONS, scope game) and calls ``frame_diff()``;
  the next call uses ``bed_pick`` without defining it and acts;
- ``noop``: a batch with a likely no-op (walking into a wall, clicking a border cell): ARC3_BATCH_NOOP_BLOCK;
- ``print``: a long board dump (tool-output middle truncation; fills the context, so history is trimmed and the
  priority gate hands the slot over), then acts;
- ``stale``: repeats one action until it stops changing the board: ARC3_STALE_STATE_BLOCK refuses the next call;
- ``undo``: UNDO where the game offers it (sb26; EXPOSE_UNDO=on), else a plain action;
- ``chat``: a reply without a tool call (the "You have not acted yet" nudge);
- ``think``: ~3k tokens of reasoning without a tool call (LOCAL_ANALYZER_YIELD_TOKENS=2048: the turn yields);
- every ``--overflow-every`` requests (if the request is long enough) an SGLang context-length error, which the
  harness answers by force-draining history and retrying.

Usage it reports: prompt tokens estimated from the text and images, cached tokens as the longest message prefix
shared with a recent request (a radix cache in miniature), completion tokens from the reply. It also writes an
SGLang-style serve.log, so scripts/franzen_report.py reads the bed's output like a real run (the server numbers are
the mock's, not a GPU's).

Bed overrides on top of the notebook's environment, all printed: the model URLs; ARC3_MAX_ACTIVE_STREAMS = games - 1
(--slots), so games wait at the priority gate; ARC3_DIAG_CONCURRENCY=1 (gate counters in the log); and, unless
--keep-context, a 24k context window, 4k output and an 8k drain so history is trimmed within minutes.

The run directory (--out, default a new temp dir) is laid out like /kaggle/working: benchmark.json, summary.txt,
transcripts/, *_requests.jsonl, serve.log (mock), plus bed.log (the child's output), mock.jsonl and
bed_report.json. The script prints the coverage checks and the franzen_report table, and exits 1 if a required
check failed.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import itertools
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
from collections import Counter, deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV = Path(os.environ.get("FRANZEN_BED_VENV", Path.home() / ".cache" / "arc3-franzen-bed" / "venv"))
REQUIREMENTS = ["arc-agi==0.9.9", "arcengine==0.9.3", "numpy", "scipy", "matplotlib", "imageio", "imageio-ffmpeg",
                "pillow", "requests", "python-dotenv"]
DEFAULT_GAMES = ["ls20", "vc33", "sb26"]
BED_CONTEXT = {"LOCAL_ANALYZER_CONTEXT_WINDOW": "24576", "LOCAL_ANALYZER_MAX_OUTPUT": "4096",
               "ARC3_CONTEXT_DRAIN_TOKENS": "8192"}
NUDGE = "You have not acted yet."
MARKER = re.compile(r"bed:([a-z]+):(\d+)")
TAGS = (("retained", ("BED retained True",)), ("not_retained", ("BED retained False",)),
        ("frame_diff", ("BED frame_diff",)), ("batch_noop", ("no_op_action",)),
        ("stale_state", ("stale_state", "StaleStateActionError")), ("undo", ("BED undo executed",)),
        ("solve", ("BED solve",)), ("long_output", ("BED long output",)))

# --- the scripted model -----------------------------------------------------------------------------------------

_PICK = """def bed_pick(actions, step):
    keys = [a for a in actions if a not in ('RESET', 'UNDO', 'MOUSE')]
    if keys:
        return [keys[step % len(keys)]]
    return [{'action': 'MOUSE', 'row': 8 + (7 * step) % 48, 'col': 8 + (13 * step) % 48}]
"""

SNIPPETS = {
    "solve": ["""# bed:solve:0
acts = set(valid_actions)
moved = [t for t in transitions if 'RESET' not in str(t.action)]
plan = None
if current_frame.level == 1 and not moved:
    if acts == {'UP', 'DOWN', 'LEFT', 'RIGHT'}:
        plan = ['LEFT'] * 3 + ['UP'] * 4 + ['RIGHT'] * 3 + ['UP'] * 3   # ls20 level 1 (BFS in the engine)
    elif acts == {'MOUSE'}:
        plan = [{'action': 'MOUSE', 'row': 32, 'col': 60}] * 3           # vc33 level 1 (BFS in the engine)
if plan is None:
    keys = [a for a in valid_actions if a not in ('RESET', 'UNDO', 'MOUSE')]
    plan = [keys[0]] if keys else [{'action': 'MOUSE', 'row': 20, 'col': 20}]
r = action(plan)
print('BED solve', len(plan), r.get('executed_count'), r.get('stop_reason'), 'level', current_frame.level)
"""],
    "define": ["""# bed:define:0
""" + _PICK + """
def bed_summary(frame):
    return {'level': frame.level, 'step': frame.step, 'objects': len(frame.segmentation['nodes'])}

print('BED define', bed_summary(current_frame))
d = frame_diff()
print('BED frame_diff', sorted(d)[:6] if isinstance(d, dict) else type(d).__name__)
""", None],  # the second snippet is _use(...)
    "noop": ["""# bed:noop:0
keys = [a for a in valid_actions if a in ('UP', 'DOWN', 'LEFT', 'RIGHT')]
if keys:
    batch = [keys[current_frame.step % len(keys)]] * 8      # walks into a wall: the batch stops at the first no-op
else:
    batch = [{'action': 'MOUSE', 'row': 0, 'col': 0}] * 3  # a border click changes nothing inside the board
r = action(batch)
print('BED noop', len(batch), r.get('executed_count'), r.get('stop_reason'))
"""],
    "print": ["""# bed:print:0
rows = current_frame.ascii.splitlines()
print('BED long output', len(rows), 'rows')
for _ in range(12):
    print('\\n'.join(rows))
""", None],
    "stale": ["""# bed:stale:0
keys = [a for a in valid_actions if a in ('UP', 'DOWN', 'LEFT', 'RIGHT')]
move = [keys[0]] if keys else [{'action': 'MOUSE', 'row': 0, 'col': 0}]
for i in range(10):
    r = action(move)
    print('BED stale step', i, r.get('executed_count'), r.get('stop_reason'), r.get('gameplay_changed'))
"""],
    "undo": ["""# bed:undo:0
if 'UNDO' in valid_actions:
    r = action(['UNDO'])
    print('BED undo executed', r.get('executed_count'), r.get('stop_reason'))
else:
    keys = [a for a in valid_actions if a not in ('RESET', 'MOUSE')]
    r = action([keys[0]] if keys else [{'action': 'MOUSE', 'row': 30, 'col': 30}])
    print('BED undo unavailable', r.get('executed_count'))
"""],
    "fallback": ["""# bed:fallback:0
keys = [a for a in valid_actions if a not in ('RESET', 'UNDO', 'MOUSE')]
r = action([keys[0]] if keys else [{'action': 'MOUSE', 'row': 24, 'col': 24}])
print('BED fallback', r.get('executed_count'), r.get('stop_reason'))
"""],
}


def _use(marker: str) -> str:
    return f"""# {marker}
try:
    plan = bed_pick(valid_actions, current_frame.step)
    print('BED retained True')
except NameError:
    print('BED retained False')
    plan = [valid_actions[0]] if valid_actions[0] != 'MOUSE' else [{{'action': 'MOUSE', 'row': 12, 'col': 12}}]
r = action(plan)
d = frame_diff()
print('BED acted', r.get('executed_count'), r.get('stop_reason'), 'changed', d.get('changed_cell_count') if isinstance(d, dict) else d)
"""


SNIPPETS["define"][1] = _use("bed:define:1")
SNIPPETS["print"][1] = _use("bed:print:1")
CYCLE = ["define", "noop", "print", "stale", "undo", "chat", "think"]
NO_TOOL = {"chat", "think"}


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return ""


def _images(message: dict) -> int:
    content = message.get("content")
    return sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image_url") if isinstance(content, list) else 0


def _message_tokens(message: dict) -> int:
    chars = len(_text(message.get("content")))
    chars += sum(len(str(message.get(k) or "")) for k in ("reasoning_content", "reasoning"))
    chars += sum(len(json.dumps(c.get("function", {}))) for c in message.get("tool_calls") or [])
    return int(chars / 3.3) + 402 * _images(message) + 4


def _last_marker(messages: list[dict]) -> tuple[str, int] | None:
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        texts = [str(c.get("function", {}).get("arguments", "")) for c in m.get("tool_calls") or []]
        texts.append(_text(m.get("content")))
        for t in texts:
            hit = MARKER.search(t)
            if hit:
                return hit.group(1), int(hit.group(2))
    return None


def next_step(messages: list[dict]) -> tuple[str, int]:
    """(program, snippet index) for this request, from the conversation alone."""
    last = _last_marker(messages)
    role = messages[-1].get("role") if messages else "user"
    if role == "tool" and last is not None:
        program, index = last
        if program in SNIPPETS and index + 1 < len(SNIPPETS[program]):
            return program, index + 1
        return "fallback", 0  # the acting snippet was refused before anything executed; the turn is still open
    if last is None:
        return "solve", 0
    program = last[0]
    following = CYCLE[(CYCLE.index(program) + 1) % len(CYCLE)] if program in CYCLE else CYCLE[0]
    return following, 0


class MockModel:
    """Replies, usage, the request log (mock.jsonl) and an SGLang-style serve.log."""

    def __init__(self, out_dir: Path, *, latency: float = 0.15, decode_tok_s: float = 3000.0,
                 overflow_every: int = 41, expect: str = "", reasoning_chars: int = 700, think_chars: int = 9500):
        self.out_dir = Path(out_dir)
        self.latency, self.decode_tok_s, self.overflow_every = latency, decode_tok_s, overflow_every
        self.expect, self.reasoning_chars, self.think_chars = expect, reasoning_chars, think_chars
        self.lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.n = 0
        self.inflight = 0
        self.recent: deque = deque(maxlen=64)
        self.seen_tool_results: set[str] = set()
        self.records = (self.out_dir / "mock.jsonl").open("a", encoding="utf-8")
        self.serve_log = (self.out_dir / "serve.log").open("a", encoding="utf-8")

    def close(self) -> None:
        self.records.close()
        self.serve_log.close()

    def _log(self, line: str) -> None:
        with self.log_lock:
            self.serve_log.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {line}\n")
            self.serve_log.flush()

    def _cached(self, keys: list[str], cumulative: list[int]) -> int:
        best = 0
        for other in self.recent:
            k = 0
            while k < min(len(keys), len(other)) and keys[k] == other[k]:
                k += 1
            best = max(best, k)
        return cumulative[best - 1] if best else 0

    def handle(self, request: dict) -> tuple[int, dict]:
        messages = request.get("messages") or []
        tools_tokens = int(len(json.dumps(request.get("tools") or [])) / 3.3)
        tokens = [_message_tokens(m) for m in messages]
        if tokens:
            tokens[0] += tools_tokens
        cumulative = [sum(tokens[:i + 1]) for i in range(len(tokens))]
        prompt = cumulative[-1] if cumulative else 0
        keys = [hashlib.sha1(json.dumps(m, sort_keys=True).encode()).hexdigest() for m in messages]
        with self.lock:
            self.n += 1
            n = self.n
            cached = min(self._cached(keys, cumulative) // 64 * 64, max(0, prompt - 1))
            self.recent.append(keys)
            self.inflight += 1
            running = self.inflight
        program, index = next_step(messages)
        last = messages[-1] if messages else {}
        with self.lock:  # each tool result once, wherever it sits (an acting call's is followed by the opener)
            fresh = [m for m in messages if m.get("role") == "tool"
                     and str(m.get("tool_call_id")) not in self.seen_tool_results]
            self.seen_tool_results.update(str(m.get("tool_call_id")) for m in fresh)
        tool_text = "\n".join(_text(m.get("content")) for m in fresh)
        record = {"n": n, "t": round(time.time(), 3), "messages": len(messages), "prompt_tokens": prompt,
                  "cached_tokens": cached, "last_role": last.get("role"), "program": program, "snippet": index,
                  "nudge": last.get("role") == "user" and NUDGE in _text(last.get("content")),
                  "tags": [tag for tag, needles in TAGS if any(n in tool_text for n in needles)]}
        if self.expect:
            record["expect_in_system"] = bool(messages) and self.expect in _text(messages[0].get("content"))
            record["expect_at"] = next((i for i, m in enumerate(messages) if self.expect in _text(m.get("content"))),
                                       None)
        try:
            if self.overflow_every and n % self.overflow_every == 0 and len(messages) >= 12:
                record["status"] = 400
                limit = max(1, prompt - 1)
                self._log('INFO:     127.0.0.1:1 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request')
                return 400, {"object": "error", "type": "BadRequestError", "code": 400, "param": None,
                             "message": f"The input ({prompt} tokens) is longer than the model's context length "
                                        f"({limit} tokens)."}
            message, finish = self._reply(n, program, index)
            generated = int((len(str(message.get("content") or "")) + len(message.get("reasoning_content", ""))
                             + sum(len(c["function"]["arguments"]) for c in message.get("tool_calls", []))) / 3.3)
            seconds = self.latency + generated / self.decode_tok_s
            time.sleep(seconds)
            record.update(status=200, completion_tokens=generated)
            self._log(f"Prefill batch, #new-seq: 1, #new-token: {prompt - cached}, #cached-token: {cached}, "
                      f"full token usage: 0.00, #running-req: {running - 1}, #queue-req: 0")
            self._log(f"Decode batch, #running-req: {running}, #full token: {prompt}, accept len: 3.00, "
                      f"accept rate: 0.67, cuda graph: True, gen throughput (token/s): {generated / seconds:.2f}, "
                      f"#queue-req: 0")
            self._log(f"ReqTimeStats(rid=mock{n}, input_len={prompt}, cached_input_len={cached}, output_len={generated}, "
                      f"attempts=0, type=unified): queue_duration=0.10ms, initial_prefill_elapsed="
                      f"{1000 * self.latency:.2f}ms, post_prefill_elapsed={1000 * (seconds - self.latency):.2f}ms, "
                      f"forward_duration={1000 * seconds:.2f}ms, entry_time={time.time():.3f}")
            self._log('INFO:     127.0.0.1:1 - "POST /v1/chat/completions HTTP/1.1" 200 OK')
            usage = {"prompt_tokens": prompt, "completion_tokens": generated, "total_tokens": prompt + generated,
                     "prompt_tokens_details": {"cached_tokens": cached, "image_tokens": 402 * sum(map(_images, messages))},
                     "reasoning_tokens": int(len(message.get("reasoning_content", "")) / 3.3)}
            return 200, {"id": f"mock-{n}", "object": "chat.completion", "model": request.get("model", "mock"),
                         "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}
        finally:
            with self.lock:
                self.inflight -= 1
                self.records.write(json.dumps(record) + "\n")
                self.records.flush()

    def _reply(self, n: int, program: str, index: int) -> tuple[dict, str]:
        thinking = (f"Mock reasoning for request {n}: the board has objects; testing the {program} step next. ")
        if program == "think":
            reasoning = (thinking * (self.think_chars // len(thinking) + 1))[:self.think_chars]
            return {"role": "assistant", "content": "bed:think:0 long deliberation, no tool call",
                    "reasoning_content": reasoning}, "stop"
        if program == "chat":
            return {"role": "assistant", "content": "bed:chat:0 I will inspect the board on the next call.",
                    "reasoning_content": thinking}, "stop"
        reasoning = (thinking * (self.reasoning_chars // len(thinking) + 1))[:self.reasoning_chars]
        code = SNIPPETS[program][index]
        call = {"id": f"call_{n}", "type": "function",
                "function": {"name": "python", "arguments": json.dumps({"code": code})}}
        return {"role": "assistant", "content": None, "reasoning_content": reasoning, "tool_calls": [call]}, "tool_calls"


def serve(model: MockModel, port: int = 0) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):  # the harness gave up (read timeout)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"object": "list", "data": [{"id": "flashnext", "object": "model"}]})

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0)
            try:
                status, reply = model.handle(json.loads(body or b"{}"))
            except Exception as exc:  # a bug in the mock must not look like a harness failure
                status, reply = 500, {"object": "error", "message": f"mock error: {exc!r}"}
            self._send(status, reply)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# --- the notebook ------------------------------------------------------------------------------------------------


def _cell(cells: list[str], needle: str) -> str:
    hits = [c for c in cells if needle in c]
    if len(hits) != 1:
        raise SystemExit(f"notebook: {needle!r} found in {len(hits)} cells (expected one)")
    return hits[0]


def notebook_env(cell4: str) -> tuple[dict[str, str], dict]:
    """The environment cell 4 sets for the harness, from the cell's own statements (TRUE_SUBMISSION = False).

    Runs only its assignments of names and ``os.environ[...]``, the ``if USE_PRIORITY_SCHEDULING`` block and the
    ``.update(...)`` calls, against a dict standing in for os.environ; shell lines, the git apply, prints and loops
    are skipped.
    """
    source = "\n".join(line for line in cell4.splitlines() if not line.lstrip().startswith("!"))
    tree = ast.parse(source)
    skip = {"TRUE_SUBMISSION", "NOTEBOOK_START_TIME", "NOTEBOOK_START_EPOCH"}

    def wanted(node) -> bool:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            environ_item = (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute)
                            and target.value.attr == "environ")
            return environ_item or (isinstance(target, ast.Name) and target.id not in skip)
        if isinstance(node, ast.If):
            return isinstance(node.test, ast.Name) and node.test.id == "USE_PRIORITY_SCHEDULING"
        return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "update")

    keep = [node for node in tree.body if wanted(node)]
    environ: dict[str, str] = {}
    namespace = {"os": types.SimpleNamespace(environ=environ), "Path": Path, "TRUE_SUBMISSION": False, "time": time}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<cell 4>", "exec"), namespace)
    return environ, namespace


def arm_notebook(out: Path, notebook: Path | None, env: dict, env_add: dict, patches: list[Path]) -> Path:
    if notebook is not None:
        return Path(notebook)
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_franzen_nb
    build_franzen_nb.build(out, "bed-arm", env=env, env_add=env_add, patches=patches, apply_check=False)
    return out / "bed-arm.ipynb"


# --- the child: notebook cells 10, 14, 16, 20 in the bed venv ------------------------------------------------------


def inner(config_path: Path) -> None:
    import asyncio
    import pickle
    from datetime import timedelta

    cfg = json.loads(Path(config_path).read_text())
    bundle, job = Path(cfg["bundle"]), Path(cfg["job_dir"])
    entries = []
    for repo in sorted((bundle / "src").iterdir(), reverse=True):  # cell 10
        for candidate in (repo / "src", repo):
            if candidate.is_dir():
                entries.append(candidate)
    for entry in entries:
        sys.path.insert(0, str(entry))
    with open(bundle / "deploy_target.pkl", "rb") as f:  # cell 14
        target = pickle.load(f)
    target.actual_run_as_submission = False
    target.is_competition_rerun = False
    with open(bundle / "benchmark_initial.pkl", "rb") as f:
        bm = pickle.load(f)
    bm.job_dir = job
    namespace = {"bm": bm, "target": target, "TRUE_SUBMISSION": False,
                 "USE_PRIORITY_SCHEDULING": cfg["use_priority_scheduling"]}
    exec(cfg["cell16"], namespace)  # cell 16 as in a Save & Run; then the bed's game count and time
    bm.solver.max_runtime_s_per_game = float(cfg["seconds"])
    bm.solver.concurrency = len(cfg["games"]) * bm.n_passes
    import arc_agi  # cell 20, offline, the bed's games
    import taaf.game_api
    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=cfg["env_dir"])
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=cfg["env_dir"])
    available = [e.game_id for e in arcade.available_environments]
    missing = [prefix for prefix in cfg["games"] if not any(g.startswith(prefix) for g in available)]
    if missing:
        raise SystemExit(f"bed: no game files for {missing} under {cfg['env_dir']}")
    game_ids = [next(g for g in available if g.startswith(prefix)) for prefix in cfg["games"]]
    bm.games = [taaf.game_api.GameAPI(env_name=g, arcade_spec=spec) for g in game_ids]
    (job / "git_status.txt").write_text((bundle / "git_status.txt").read_text())
    os.environ.setdefault("RECORDINGS_DIR", str(job / "server_recording"))
    budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)
    soft_end = (datetime.fromtimestamp(cfg["start"]) + timedelta(seconds=budget - min(600.0, budget / 2))
                if budget > 0 else None)
    print(f"bed: games {game_ids}, {cfg['seconds']} s each, solver {bm.solver}", flush=True)
    asyncio.run(bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=False))
    from inference.agent import tool_agent
    report = {"gate_stats": dict(tool_agent._GATE_STATS), "gate_slots": tool_agent._max_active_streams(), "games": {}}
    for run in bm.game_runs:
        ids = Counter((getattr(h.action, "id", None).name if getattr(h, "action", None) is not None else "?")
                      for h in run.history)
        report["games"][run.game_id] = {"state": run.state, "levels": run.levels_completed,
                                        "actions": len(run.history), "actions_per_level": list(run.actions_per_level),
                                        "score": run.final_score, "note": run.solver_note, "action_ids": dict(ids)}
    (job / "bed_inner.json").write_text(json.dumps(report, indent=1, default=str))


# --- the outer process ---------------------------------------------------------------------------------------------


def ensure_venv(venv: Path) -> Path:
    """A venv with what the notebook's Kaggle image provides for the harness; created once, reused after."""
    python = venv / "bin" / "python"
    stamp = venv / "franzen-bed-requirements.txt"
    wanted = "\n".join(REQUIREMENTS) + "\n"
    if python.exists() and stamp.exists() and stamp.read_text() == wanted:
        return python
    venv.parent.mkdir(parents=True, exist_ok=True)
    uv = next((p for p in (os.environ.get("UV"), "uv", str(Path.home() / ".local/bin/uv")) if p and _which(p)), None)
    if uv:
        subprocess.run([uv, "venv", "-q", "--python", "3.12", str(venv)], check=True)
        subprocess.run([uv, "pip", "install", "-q", "--python", str(python), *REQUIREMENTS], check=True)
    else:
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "-q", *REQUIREMENTS], check=True)
    stamp.write_text(wanted)
    return python


def _which(cmd: str) -> bool:
    import shutil
    return shutil.which(cmd) is not None or Path(cmd).is_file()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def coverage(out: Path, inner_report: dict, records: list[dict], analysis: dict, log_text: str, expect: str,
             games: list[str], expect_in: str = "system") -> dict:
    tags = Counter(tag for r in records for tag in r.get("tags", []))
    gate = inner_report.get("gate_stats", {})
    game_runs = inner_report.get("games", {})
    statuses = Counter()
    for t in (analysis.get("transcripts") or {}).values():
        statuses.update(t["statuses"])
    following = {}
    for prev, cur in itertools.pairwise(records):
        following.setdefault(prev.get("program"), Counter())[(cur.get("last_role"), cur.get("nudge"))] += 1
    undo_games = [g for g in game_runs if g.startswith("sb26") or g.startswith("ar25")]
    facts = {
        "requests": len(records),
        "http_400_overflow": sum(r.get("status") == 400 for r in records),
        "overflow_recovered": statuses.get("context_overflow_recovered", 0),
        "games_crashed": [g for g, r in game_runs.items() if r["state"] == "crashed"],
        "levels_completed": {g: r["levels"] for g, r in game_runs.items()},
        "gate_slots": inner_report.get("gate_slots"),
        "gate_admissions": gate.get("enqueued", 0),
        "gate_blocked": gate.get("blocked", 0),
        "gate_blocked_ms": gate.get("blocked_ms", 0),
        "tail_fade_logged": "tail fade phase" in log_text,
        "prefix_breaks": analysis["totals"]["prefix_breaks"],
        "retained_function_used": tags["retained"], "retained_function_missing": tags["not_retained"],
        "frame_diff_calls": tags["frame_diff"], "batch_noop_stops": tags["batch_noop"],
        "stale_state_refusals": tags["stale_state"], "undo_executed": tags["undo"],
        "undo_actions": sum(r["action_ids"].get("ACTION7", 0) for r in game_runs.values()),
        "nudges": sum(bool(r.get("nudge")) for r in records),
        "yields_after_think": sum(n for (role, nudge), n in following.get("think", {}).items() if role == "user" and not nudge),
        "tracebacks_in_log": log_text.count("Traceback (most recent call last)"),
    }
    if expect:
        facts["expect_seen"] = sum(bool(r.get("expect_in_system")) for r in records)
        # requests by the index of the first message holding the text (0 is the system prompt)
        facts["expect_at"] = dict(sorted(Counter(str(r["expect_at"]) for r in records
                                                 if r.get("expect_at") is not None).items()))
    checks = {
        "harness made requests": facts["requests"] > 0,
        "no game crashed": not facts["games_crashed"],
        "a scripted level-1 win (ls20/vc33)": any(v >= 1 for v in facts["levels_completed"].values())
        or not any(g in ("ls20", "vc33") for g in games),
        "priority gate admitted more than once per game": facts["gate_admissions"] > len(game_runs),
        "games waited at the gate": facts["gate_blocked"] > 0 or (facts["gate_slots"] or 0) >= len(game_runs),
        "history trimmed (prefix breaks)": facts["prefix_breaks"] > 0,
        "retained function reused": facts["retained_function_used"] > 0,
        "frame_diff called": facts["frame_diff_calls"] > 0,
        "batch no-op guard stopped a batch": facts["batch_noop_stops"] > 0,
        "stale-state guard refused an action": facts["stale_state_refusals"] > 0,
        "context overflow recovered": facts["overflow_recovered"] > 0 or facts["http_400_overflow"] == 0,
        "no tracebacks": facts["tracebacks_in_log"] == 0,
    }
    if undo_games:
        checks["UNDO executed"] = facts["undo_actions"] > 0
    if expect and expect_in == "any":
        checks[f"--expect text in a request ({expect!r})"] = bool(facts["expect_at"])
    elif expect:
        checks[f"--expect text in the system prompt ({expect!r})"] = facts["expect_seen"] > 0
    return {"facts": facts, "checks": checks}


def run_bed(out: Path, *, games: list[str], seconds: float, slots: int | None = None, notebook: Path | None = None,
            env: dict | None = None, env_add: dict | None = None, patches: list[Path] | tuple = (),
            sets: dict | None = None, expect: str = "", expect_in: str = "system", keep_context: bool = False,
            python: Path | None = None,
            env_dir: Path | None = None, overflow_every: int = 41, latency: float = 0.15, verbose: bool = False,
            his_repo: Path | None = None, bundle: Path | None = None) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    import franzen_report
    import franzen_tree

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    nb = arm_notebook(out / "notebook", notebook, env or {}, env_add or {}, list(patches))
    cells = franzen_tree.notebook_cells(nb)
    franzen_tree.notebook_bundle(out / "bundle", franzen_tree.our_patch_cells(nb), his_repo=his_repo, bundle=bundle,
                                 his_patch=franzen_tree.his_patch_text(nb))
    nb_env, names = notebook_env(_cell(cells, "setup_env = {"))
    slots = slots if slots is not None else max(1, len(games) - 1)
    model = MockModel(out, latency=latency, overflow_every=overflow_every, expect=expect)
    server = serve(model, _free_port())
    url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    overrides = {"LOCAL_ANALYZER_BASE_URL": url, "OPENAI_BASE_URL": url, "ARC3_MAX_ACTIVE_STREAMS": str(slots),
                 "ARC3_DIAG_CONCURRENCY": "1", **({} if keep_context else BED_CONTEXT), **(sets or {})}
    changed = {k: (nb_env.get(k), v) for k, v in overrides.items() if nb_env.get(k) != v}
    child_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    child_env.update(nb_env)
    child_env.update(overrides)
    child_env["PYTHONUNBUFFERED"] = "1"
    config = {"bundle": str(out / "bundle"), "job_dir": str(out), "games": games, "seconds": seconds,
              "env_dir": str(env_dir or ROOT / "environment_files"), "start": time.time(),
              "use_priority_scheduling": bool(names.get("USE_PRIORITY_SCHEDULING", True)),
              "cell16": _cell(cells, "bm.solver.max_runtime_s_per_game")}
    (out / "bed_config.json").write_text(json.dumps({**config, "overrides": changed}, indent=1))
    py = python or ensure_venv(DEFAULT_VENV)
    print(f"bed: {out}\n  notebook {nb}\n  games {games}, {seconds:g} s each, gate slots {slots}\n  overrides of the "
          f"notebook environment: " + ", ".join(f"{k}: {a!r} -> {b!r}" for k, (a, b) in changed.items()), flush=True)
    t0 = time.time()
    with open(out / "bed.log", "w") as log:
        proc = subprocess.Popen([str(py), str(Path(__file__).resolve()), "--inner", str(out / "bed_config.json")],
                                env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        watchdog = threading.Timer(seconds + 900, proc.kill)  # a hung harness must not hang the bed
        watchdog.start()
        for line in proc.stdout:
            log.write(line)
            if verbose:
                print("  | " + line, end="")
        code = proc.wait()
        watchdog.cancel()
    server.shutdown()
    model.close()
    print(f"bed: child exited {code} after {time.time() - t0:.0f} s", flush=True)
    inner_path = out / "bed_inner.json"
    inner_report = json.loads(inner_path.read_text()) if inner_path.exists() else {}
    records = [json.loads(line) for line in (out / "mock.jsonl").read_text().splitlines() if line.strip()]
    analysis = franzen_report.analyze(out, out / "bed.log") if (out / "benchmark.json").exists() else {
        "totals": {"prefix_breaks": 0}, "transcripts": {}}
    result = coverage(out, inner_report, records, analysis, (out / "bed.log").read_text(errors="replace"), expect,
                      games, expect_in)
    result.update(exit_code=code, seconds=round(time.time() - t0, 1), overrides=changed, inner=inner_report)
    if code != 0:
        result["checks"]["child exited 0"] = False
    (out / "bed_report.json").write_text(json.dumps(result, indent=1, default=str))
    for name, ok in result["checks"].items():
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    print("  facts: " + json.dumps(result["facts"], default=str))
    if (out / "benchmark.json").exists():
        print(franzen_report.render(analysis))
    return result


def _pairs(values: list[str]) -> dict[str, str]:
    return dict(v.split("=", 1) for v in values)


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--inner":
        inner(Path(sys.argv[2]))
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=None, help="run directory (default: a new temp dir)")
    ap.add_argument("--games", default=",".join(DEFAULT_GAMES), help="comma-separated game id prefixes")
    ap.add_argument("--seconds", type=float, default=150.0, help="per-game time budget")
    ap.add_argument("--slots", type=int, default=None, help="ARC3_MAX_ACTIVE_STREAMS (default: games - 1)")
    ap.add_argument("--notebook", type=Path, default=None, help="a built arm (default: build one from the flags below)")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="as build_franzen_nb.py --env")
    ap.add_argument("--env-add", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--patch", action="append", default=[], type=Path, metavar="FILE")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="extra environment for the harness process only (not in the notebook)")
    ap.add_argument("--expect", default="", help="text that must appear in the system prompt the mock receives")
    ap.add_argument("--expect-in", choices=("system", "any"), default="system",
                    help="where --expect must appear: the system prompt (default) or any message of a request")
    ap.add_argument("--keep-context", action="store_true", help="keep the notebook's 128k context settings")
    ap.add_argument("--python", type=Path, default=None, help="interpreter with the harness dependencies")
    ap.add_argument("--env-dir", type=Path, default=None, help="game files (default: environment_files/)")
    ap.add_argument("--overflow-every", type=int, default=41, help="answer every Nth request with a context error (0: never)")
    ap.add_argument("--latency", type=float, default=0.15, help="mock seconds per reply before decode time")
    ap.add_argument("--his-repo", type=Path, default=None)
    ap.add_argument("--bundle", type=Path, default=None)
    ap.add_argument("-v", "--verbose", action="store_true", help="echo the harness output")
    args = ap.parse_args()
    out = args.out or Path(tempfile.mkdtemp(prefix="franzen-bed-"))
    result = run_bed(out, games=[g.strip() for g in args.games.split(",") if g.strip()], seconds=args.seconds,
                     slots=args.slots, notebook=args.notebook, env=_pairs(args.env), env_add=_pairs(args.env_add),
                     patches=args.patch, sets=_pairs(args.set), expect=args.expect, expect_in=args.expect_in,
                     keep_context=args.keep_context,
                     python=args.python, env_dir=args.env_dir, overflow_every=args.overflow_every,
                     latency=args.latency, verbose=args.verbose, his_repo=args.his_repo, bundle=args.bundle)
    sys.exit(0 if all(result["checks"].values()) else 1)


if __name__ == "__main__":
    main()
