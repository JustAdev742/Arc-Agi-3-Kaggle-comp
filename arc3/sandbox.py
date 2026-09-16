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
  history, last, np, set_model(predict), world_model_stats(), verify_model(predict), transitions(),
  ents(), events(n), event_log(), describe_events(n), avatar(), roles(), entity(id), tile,
  move_model(), plan_to(x, y), plan_to_entity(id)
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
from arc3.entities import Tracker as _Tracker
from arc3.planner import MoveModel as _MoveModel

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
WM = {"predict": None, "checked": 0, "matched": 0, "mismatches": [], "errors": 0}
LOG = {"level": None, "transitions": []}  # (before_grid, action_label, after_grid) for the current level
TRK = {"t": _Tracker()}

def ents(limit=40):
    """Tracked entities of the current frame with persistent ids and roles (static, hud, avatar, dynamic)."""
    return TRK["t"].entities_summary(limit)

def events(n=1):
    """The last n action transitions as entity events: moved/appeared/disappeared/recolored/reshaped."""
    return TRK["t"].log[-n:]

def event_log():
    return list(TRK["t"].log)

def describe_events(n=3):
    t = TRK["t"]
    return [_Tracker.describe(r, t) for r in t.log[-n:]]

def avatar():
    """{'id', 'keymap': {'UP': (dx,dy), ...}, 'entity'} for the entity that moves with the keys, or None."""
    return TRK["t"].avatar()

def roles():
    return TRK["t"].roles()

def entity(eid):
    e = TRK["t"].get(eid)
    return None if e is None else {**e.summary(TRK["t"].tile), "mask": e.mask}

def move_model():
    """Fitted avatar movement model (key map + obstacles from evidence). Needs a few observed key moves.
    Use move_model().predict with set_model(...) to have every move verified."""
    return _MoveModel(TRK["t"])

def plan_to(x, y):
    """Shortest key sequence (list of 'UP'/'DOWN'/...) until the avatar covers cell (x, y), or None."""
    return move_model().plan_to_point(int(x), int(y))

def plan_to_entity(eid, touch=True):
    """Shortest key sequence until the avatar touches (touch=True) or overlaps entity `eid`, or None."""
    e = TRK["t"].get(int(eid))
    if e is None:
        raise ValueError(f"no entity #{eid} in the current frame; see ents()")
    return move_model().plan_to_entity(e, touch=touch)

def transitions(last_n=None):
    """The level's recorded (before, action, after) triples, oldest first (lost if the REPL restarts)."""
    t = LOG["transitions"]
    return t[-last_n:] if last_n else list(t)

def verify_model(predict=None, last_n=None):
    """Replay the level's transitions through predict(grid, action) and return
    {'checked', 'correct', 'counter_examples': [{'index','action','wrong_cells','bbox','sample'}, ...]}.
    Use it before trusting a model and after every revision; a model with counter-examples is wrong."""
    fn = predict if predict is not None else WM["predict"]
    if fn is None:
        raise ValueError("no predictor: pass predict or call set_model(predict) first")
    items = transitions(last_n)
    out = {"checked": 0, "correct": 0, "counter_examples": []}
    for i, (before, action, after) in enumerate(items):
        try:
            pred = _to_grid(fn(before.copy(), action))
            if pred.shape != after.shape:
                raise ValueError(f"predicted shape {pred.shape} != {after.shape}")
        except Exception as e:  # noqa: BLE001
            out["counter_examples"].append({"index": i, "action": action, "error": f"{type(e).__name__}: {e}"[:160]})
            continue
        out["checked"] += 1
        wrong = pred != after
        n = int(wrong.sum())
        if n == 0:
            out["correct"] += 1
        elif len(out["counter_examples"]) < 8:
            ys, xs = np.nonzero(wrong)
            out["counter_examples"].append({"index": i, "action": action, "wrong_cells": n,
                                            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                                            "sample": [[int(x), int(y), int(pred[y, x]), int(after[y, x])] for y, x in list(zip(ys, xs))[:4]]})
    return out

def set_model(predict):
    """Register predict(grid, action) -> predicted next grid (64x64 array or list). The harness checks every
    real action against it and reports mismatches; see world_model_stats()."""
    if predict is not None and not callable(predict):
        raise TypeError("set_model expects a callable predict(grid, action) or None")
    WM["predict"] = predict
    WM["checked"] = WM["matched"] = WM["errors"] = 0
    WM["mismatches"] = []
    return "world model registered" if predict else "world model cleared"

def world_model_stats():
    out = {"checked": WM["checked"], "matched": WM["matched"], "errors": WM["errors"],
           "recent_mismatches": WM["mismatches"][-5:]}
    if HYP["models"]:
        out["hypotheses"] = {k: {"alive": k in HYP["alive"], "checked": HYP["checked"][k], "correct": HYP["correct"][k],
                                 "killed_by": HYP["killed_by"].get(k)} for k in HYP["models"]}
    return out

HYP = {"models": {}, "alive": set(), "checked": {}, "correct": {}, "killed_by": {}}

def set_models(models):
    """Register competing hypotheses: {name: predict(grid, action)}. Each real action checks all of them;
    a hypothesis dies on its first wrong prediction (see world_model_stats()['hypotheses'] and alive_models()).
    The single set_model() predictor is unaffected."""
    if not isinstance(models, dict) or not all(callable(f) for f in models.values()):
        raise TypeError("set_models expects {name: callable}")
    HYP["models"] = dict(models)
    HYP["alive"] = set(models)
    HYP["checked"] = {k: 0 for k in models}
    HYP["correct"] = {k: 0 for k in models}
    HYP["killed_by"] = {}
    return f"{len(models)} hypotheses registered"

def alive_models():
    return sorted(HYP["alive"])

def _check_hypotheses(before, action, after):
    if not HYP["models"]:
        return None
    for name in list(HYP["alive"]):
        fn = HYP["models"][name]
        try:
            pred = _to_grid(fn(before.copy(), action))
            ok = pred.shape == after.shape and bool((pred == after).all())
        except Exception as e:  # noqa: BLE001
            ok = False
            HYP["killed_by"][name] = f"error: {type(e).__name__}: {e}"[:120]
        HYP["checked"][name] += 1
        if ok:
            HYP["correct"][name] += 1
        else:
            HYP["alive"].discard(name)
            HYP["killed_by"].setdefault(name, f"{action} at transition {len(LOG['transitions'])}")
    return {"alive_models": sorted(HYP["alive"])}

def _action_label(a):
    if a.get("action") == "CLICK":
        return ("CLICK", a.get("x"), a.get("y"))
    return a.get("action")

def _check_prediction(before, action, after):
    fn = WM["predict"]
    if fn is None:
        return None
    try:
        pred = fn(before.copy(), _action_label(action))
        pred = _to_grid(pred)
        if pred.shape != after.shape:
            raise ValueError(f"predicted shape {pred.shape} != {after.shape}")
        wrong = int((pred != after).sum())
    except Exception as e:  # noqa: BLE001
        WM["errors"] += 1
        return {"pred_ok": False, "pred_error": f"{type(e).__name__}: {e}"[:200]}
    WM["checked"] += 1
    if wrong == 0:
        WM["matched"] += 1
        return {"pred_ok": True, "pred_wrong_cells": 0}
    ys, xs = np.nonzero(pred != after)
    d = {"pred_ok": False, "pred_wrong_cells": wrong,
         "pred_bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
         "pred_sample": [[int(x), int(y), int(pred[y, x]), int(after[y, x])] for y, x in list(zip(ys, xs))[:5]]}
    WM["mismatches"].append({"action": _action_label(action), "wrong_cells": wrong, "bbox": d["pred_bbox"]})
    return d

def _to_grid(x):
    return np.asarray(x, dtype=np.int16)

def _refresh(state, action=None):
    lvl = state.get("level")
    new_level = LOG["level"] is not None and lvl != LOG["level"]
    if new_level:
        LOG["transitions"] = []  # new level: the old transitions no longer describe this layout
    LOG["level"] = lvl
    G["grid"] = _to_grid(state["grid"])
    if TRK["t"].bg is None or new_level:
        TRK["t"].reset(G["grid"])
    elif action is not None:
        TRK["t"].update(G["grid"], action)
    G["tile"] = TRK["t"].tile
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
    if WM["predict"] is None:
        results = []
        for a in norm:  # one at a time so every transition is logged with its exact before/after grids
            before = G["grid"].copy()
            _send({"type": "action", "actions": [a]})
            reply = _recv()
            if reply.get("type") != "action_result":
                raise RuntimeError(reply.get("error", "action failed"))
            _refresh(reply["state"], _action_label(a))
            r = reply["result"][0]
            if not r.get("level_completed"):
                LOG["transitions"].append((before, _action_label(a), G["grid"].copy()))
                hyp = _check_hypotheses(before, _action_label(a), G["grid"])
                if hyp:
                    r.update(hyp)
            results.append(r)
            if r.get("level_completed") or r.get("game_over") or r.get("won"):
                break
        return results if len(norm) > 1 else results[0]
    # With a registered world model, actions go one at a time so each prediction is checked and a
    # mismatch stops the batch (the model must revise before spending more actions).
    results = []
    for a in norm:
        before = G["grid"].copy()
        _send({"type": "action", "actions": [a]})
        reply = _recv()
        if reply.get("type") != "action_result":
            raise RuntimeError(reply.get("error", "action failed"))
        _refresh(reply["state"], _action_label(a))
        r = reply["result"][0]
        chk = _check_prediction(before, a, G["grid"])
        if chk:
            r.update(chk)
        if not r.get("level_completed"):
            LOG["transitions"].append((before, _action_label(a), G["grid"].copy()))
            hyp = _check_hypotheses(before, _action_label(a), G["grid"])
            if hyp:
                r.update(hyp)
        results.append(r)
        if chk and not chk.get("pred_ok") and len(norm) > 1:
            r["batch_stopped"] = f"prediction mismatch after {len(results)} of {len(norm)} actions; revise the model"
            break
        if r.get("level_completed") or r.get("game_over") or r.get("won"):
            break
    return results if len(norm) > 1 else results[0]

def click(x, y):
    return act(("CLICK", int(x), int(y)))

for _n in ("objects", "components", "diff", "ascii", "downscale", "background", "moved", "note", "act", "click",
           "set_model", "world_model_stats", "verify_model", "transitions", "set_models", "alive_models",
           "ents", "events", "event_log", "describe_events", "avatar", "roles", "entity",
           "move_model", "plan_to", "plan_to_entity"):
    G[_n] = globals()[_n]

while True:
    msg = _recv()
    if msg.get("type") != "run":
        continue
    _last = msg["state"].get("last") or {}
    _prev = G.get("grid")
    _incoming = _to_grid(msg["state"]["grid"])
    if _prev is not None and TRK["t"].bg is not None and not np.array_equal(_prev, _incoming) and msg["state"].get("level") == LOG["level"]:
        _refresh(msg["state"], str(_last.get("action", "external")))  # actions taken outside this REPL (e.g. fallback)
    else:
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
    _send({"type": "final", "stdout": out, "error": err, "result": result, "notes": G.get("notes", []),
           "world_model": world_model_stats() if (WM["predict"] is not None or HYP["models"]) else None})
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
                            "restarted": restarted, "world_model": msg.get("world_model")}

    def __del__(self):  # pragma: no cover
        try:
            self.stop()
        except Exception:
            pass
