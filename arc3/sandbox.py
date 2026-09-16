"""Persistent Python REPL in a child process for model-written code.

Why a child process: model code can loop forever; killing and restarting a child is
the only reliable timeout. Why persistent: the model builds a world model (functions,
dicts, search code) that should survive between tool calls. On timeout the child is
restarted and the model is told its variables were lost.

Protocol (JSON lines over stdin/stdout of the child):
  parent -> child: {"type":"run","code":..., "state":{...}, "timeout":s}
  child  -> parent: {"type":"action","actions":[...]}    (blocking request)
  parent -> child: {"type":"action_result","result":{...},"state":{...}}
  child  -> parent: {"type":"final","stdout":...,"error":...,"result":...}

Preloaded names in the child (all from ``arc3.perception``, exact numpy code):
  grid, frames, level, levels_completed, win_levels, step, level_step, state, available,
  scale, objects(), components(), diff(), ascii(), downscale(), act(), note(), notes,
  history, last, np
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Optional

CHILD_SOURCE = r'''
import json, sys, io, traceback, contextlib, os
HOST_OUT = sys.stdout
sys.stdout = io.StringIO()  # everything the model prints is captured per run
import numpy as np
from arc3 import perception as _P

def _send(obj):
    HOST_OUT.write(json.dumps(obj, ensure_ascii=False, default=_default) + "\n"); HOST_OUT.flush()

def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)

def _recv():
    line = sys.stdin.readline()
    if not line:
        raise SystemExit(0)
    return json.loads(line)

G = {"__name__": "__repl__", "np": np, "notes": []}

def _to_grid(x):
    return np.asarray(x, dtype=np.int16)

def _refresh(state):
    G["grid"] = _to_grid(state["grid"])
    G["frames"] = [_to_grid(f) for f in state.get("frames", [])]
    for k in ("level", "levels_completed", "win_levels", "step", "level_step", "state", "available", "history", "last"):
        G[k] = state.get(k)
    G["scale"] = _P.detect_scale(G["grid"])

def objects(g=None, limit=60):
    g = G["grid"] if g is None else _to_grid(g)
    return _P.objects_summary(g, limit=limit)

def components(g=None, ignore_background=True):
    g = G["grid"] if g is None else _to_grid(g)
    ign = (_P.background_color(g),) if ignore_background else ()
    return _P.components(g, ignore=ign)

def diff(a=None, b=None):
    if a is None and b is None:
        fr = G["frames"]
        if len(fr) < 2:
            return {"changed": 0, "note": "fewer than two frames"}
        a, b = fr[-2], fr[-1]
    elif b is None:
        b, a = G["grid"], a
    return _P.diff(_to_grid(a), _to_grid(b)).summary()

def ascii(g=None, scale=None):
    g = G["grid"] if g is None else _to_grid(g)
    return _P.ascii(g, scale=scale)

def downscale(g=None):
    g = G["grid"] if g is None else _to_grid(g)
    return _P.downscale(g)[0]

def background(g=None):
    g = G["grid"] if g is None else _to_grid(g)
    return _P.background_color(g)

def moved(a=None, b=None):
    fr = G["frames"]
    a = fr[-2] if a is None and len(fr) > 1 else (G["grid"] if a is None else _to_grid(a))
    b = G["grid"] if b is None else _to_grid(b)
    ca = components(a); cb = components(b)
    return [{"color": o.color, "from": (o.x0, o.y0), "to": (n.x0, n.y0), "dx": dx, "dy": dy, "size": o.size}
            for o, n, dx, dy in _P.moved_objects(ca, cb)]

def note(text):
    G["notes"].append(str(text))
    return len(G["notes"])

def _is_single(a):
    if isinstance(a, (str, dict)):
        return True
    if isinstance(a, (list, tuple)):
        if len(a) == 3 and isinstance(a[0], str):
            return True  # ('CLICK', x, y)
        if len(a) == 2 and all(isinstance(v, (int, np.integer)) for v in a):
            return True  # (x, y) shorthand for a click
    return False

def act(*actions):
    items = []
    for a in actions:
        if _is_single(a):
            items.append(a)
        elif isinstance(a, (list, tuple)):
            items.extend(a)  # a list of actions
        else:
            raise TypeError(f"bad action {a!r}")
    norm = []
    for a in items:
        if isinstance(a, str):
            norm.append({"action": a.strip().upper()})
        elif isinstance(a, dict):
            norm.append({"action": str(a.get("action", "")).upper(), "x": a.get("x"), "y": a.get("y")})
        elif isinstance(a, (list, tuple)) and len(a) in (2, 3):
            if len(a) == 3:
                norm.append({"action": str(a[0]).upper(), "x": int(a[1]), "y": int(a[2])})
            else:
                norm.append({"action": "CLICK", "x": int(a[0]), "y": int(a[1])})
        else:
            raise TypeError(f"bad action {a!r}: use 'UP'/'DOWN'/'LEFT'/'RIGHT'/'ACT'/'UNDO'/'RESET' or ('CLICK', x, y)")
    if not norm:
        raise ValueError("act() needs at least one action")
    _send({"type": "action", "actions": norm})
    reply = _recv()
    if reply.get("type") != "action_result":
        raise RuntimeError(reply.get("error", "action failed"))
    _refresh(reply["state"])
    res = reply["result"]
    return res if len(norm) > 1 else res[0]

def click(x, y):
    return act(("CLICK", int(x), int(y)))

for _n in ("objects", "components", "diff", "ascii", "downscale", "background", "moved", "note", "act", "click"):
    G[_n] = globals()[_n]

while True:
    msg = _recv()
    if msg.get("type") != "run":
        continue
    _refresh(msg["state"])
    sys.stdout = buf = io.StringIO()
    err = ""
    result = None
    try:
        code = compile(msg["code"], "<repl>", "exec")
        exec(code, G, G)
        result = G.get("result")
    except SystemExit:
        raise
    except BaseException as e:
        tb = traceback.extract_tb(e.__traceback__)
        user = [f for f in tb if f.filename == "<repl>"]
        lines = [f'  line {f.lineno}, in {f.name}' for f in (user or tb[-1:])]
        err = "Traceback:\n" + "\n".join(lines) + f"\n{type(e).__name__}: {e}"
    out = buf.getvalue()
    _send({"type": "final", "stdout": out, "error": err, "result": result, "notes": G.get("notes", [])})
'''


class SandboxTimeout(Exception):
    pass


class PersistentSandbox:
    def __init__(self, *, python: Optional[str] = None, sys_path: Optional[list[str]] = None,
                 max_output_chars: int = 6000, env: Optional[dict[str, str]] = None):
        self.python = python or sys.executable
        self.sys_path = list(sys_path or sys.path)
        self.max_output_chars = max_output_chars
        self.env = env
        self.proc: Optional[subprocess.Popen] = None
        self.restarts = 0
        self.runs = 0
        self.timeouts = 0
        self._lock = threading.Lock()

    # ---------- process management ----------
    def start(self) -> None:
        self.stop()
        env = dict(os.environ if self.env is None else self.env)
        env["PYTHONPATH"] = os.pathsep.join(p for p in self.sys_path if p)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env["MPLBACKEND"] = "agg"
        self.proc = subprocess.Popen([self.python, "-c", CHILD_SOURCE], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, encoding="utf-8", env=env,
                                     start_new_session=True)
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc: subprocess.Popen) -> None:
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                self._q.put(line)
        finally:
            self._q.put(None)

    def stop(self) -> None:
        p = self.proc
        self.proc = None
        if p is None:
            return
        try:
            os.killpg(p.pid, 9)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        try:
            p.wait(timeout=2)
        except Exception:
            pass

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def restart(self) -> None:
        self.restarts += 1
        self.start()

    def _send(self, obj: dict[str, Any]) -> None:
        p = self.proc
        if p is None or p.stdin is None:
            raise RuntimeError("sandbox is not running")
        try:
            p.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
            p.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(f"sandbox pipe closed: {e}") from e

    # ---------- execution ----------
    def run(self, code: str, state: dict[str, Any], *, timeout_s: float = 30.0,
            action_handler: Optional[Callable[[list[dict[str, Any]]], tuple[list[dict[str, Any]], dict[str, Any]]]] = None,
            ) -> dict[str, Any]:
        """Execute ``code``; returns {stdout, error, result, notes, actions, timed_out, restarted}."""
        with self._lock:
            try:
                return self._run_locked(code, state, timeout_s=timeout_s, action_handler=action_handler)
            except RuntimeError as e:
                return {"stdout": "", "error": f"sandbox unavailable: {e}", "result": None, "notes": [], "actions": 0,
                        "timed_out": False, "restarted": False}

    def _run_locked(self, code: str, state: dict[str, Any], *, timeout_s: float, action_handler) -> dict[str, Any]:
        if True:
            restarted = False
            if not self.alive():
                self.start()
                restarted = self.runs > 0
            self.runs += 1
            self._send({"type": "run", "code": code, "state": state})
            deadline = time.monotonic() + timeout_s
            n_actions = 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.timeouts += 1
                    self.restart()
                    return {"stdout": "", "error": f"Tool timed out after {timeout_s:.0f}s; the REPL was restarted and "
                                                   f"all persistent variables were lost. Re-define what you need.",
                            "result": None, "notes": [], "actions": n_actions, "timed_out": True, "restarted": True}
                try:
                    line = self._q.get(timeout=remaining)
                except queue.Empty:
                    continue
                if line is None:
                    err = ""
                    try:
                        err = (self.proc.stderr.read() if self.proc and self.proc.stderr else "")[-1500:]
                    except Exception:
                        pass
                    self.restart()
                    return {"stdout": "", "error": f"REPL process died and was restarted (variables lost).\n{err}",
                            "result": None, "notes": [], "actions": n_actions, "timed_out": False, "restarted": True}
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = msg.get("type")
                if t == "action":
                    if action_handler is None:
                        self._send({"type": "error", "error": "actions are not allowed here"})
                        continue
                    try:
                        results, new_state = action_handler(list(msg.get("actions") or []))
                        n_actions += len(results)
                        self._send({"type": "action_result", "result": results, "state": new_state})
                    except Exception as e:  # noqa: BLE001
                        self._send({"type": "error", "error": f"action failed: {e}"})
                    # Actions may legitimately take a while (model-free, but the env may animate).
                    deadline = max(deadline, time.monotonic() + min(timeout_s, 10.0))
                    continue
                if t == "final":
                    out = str(msg.get("stdout") or "")
                    if len(out) > self.max_output_chars:
                        out = out[: self.max_output_chars] + f"\n...[truncated {len(out) - self.max_output_chars} chars]"
                    return {"stdout": out, "error": str(msg.get("error") or ""), "result": msg.get("result"),
                            "notes": list(msg.get("notes") or []), "actions": n_actions, "timed_out": False,
                            "restarted": restarted}

    def __del__(self):  # pragma: no cover
        try:
            self.stop()
        except Exception:
            pass
