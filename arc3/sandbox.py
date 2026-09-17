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
  scale, objects(), components(), diff(), ascii(), downscale(), act(), note(), notes, learn(),
  history, last, np, set_model(predict), world_model_stats(), verify_model(predict), transitions(),
  ents(), events(n), event_log(), describe_events(n), avatar(), roles(), entity(id), tile,
  move_model(), plan_to(x, y), plan_to_entity(id)
  symlog(), fit_rules(kind), auto_rules(), rules(), explain_rules(), rules_predictor(), plan_rules(goal), goal_candidates(), goal_hints(), probe_suggestions()
"""
from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, Optional

CHILD_SOURCE = r'''
import json, sys, io, traceback, contextlib, os
HOST_OUT = sys.stdout
sys.stdout = io.StringIO()  # everything the model prints is captured per run
import numpy as np
from arc3 import perception as _P
from arc3.entities import Tracker as _Tracker
from arc3.planner import MoveModel as _MoveModel
from arc3 import dsl as _dsl

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

G = {"__name__": "__repl__", "np": np, "notes": [], "lessons": []}
CELL = {"pred_retired": None, "batch_stopped": None}  # decisive events of the current cell, reported to the agent
WM = {"predict": None, "checked": 0, "matched": 0, "mismatches": [], "errors": 0, "streak": 0}
WM_RETIRE_AFTER = 3  # consecutive mismatches after which the registered world model is dropped
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

def _last_event_text(max_len=220):
    """Entity events of the last transition, as the short text every act() result carries."""
    t = TRK["t"]
    if not t.log:
        return ""
    txt = _Tracker.describe(t.log[-1], t)
    txt = txt.split(": ", 1)[1] if ": " in txt else txt
    return txt if len(txt) <= max_len else txt[:max_len - 3] + "..."

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

PLAN = {"optimistic": False}

def plan_to(x, y, optimistic=True):
    """Shortest key sequence (list of 'UP'/'DOWN'/...) until the avatar covers cell (x, y), or None. When no path
    exists through known-passable cells and optimistic=True, returns a path that assumes unknown terrain is
    passable (PLAN['optimistic'] becomes True): execute it with set_model(move_model().predict) so the first
    wrong step stops the batch and becomes evidence."""
    m = move_model()
    p = m.plan_to_point(int(x), int(y))
    PLAN["optimistic"] = False
    if p is None and optimistic:
        p = m.relax().plan_to_point(int(x), int(y))
        PLAN["optimistic"] = p is not None
    return p

def plan_to_entity(eid, touch=True, optimistic=True):
    """Shortest key sequence until the avatar touches (touch=True) or overlaps entity `eid`, or None; falls back
    to an optimistic path (unknown terrain passable, PLAN['optimistic'] True) like plan_to."""
    e = TRK["t"].get(int(eid))
    if e is None:
        raise ValueError(f"no entity #{eid} in the current frame; see ents()")
    m = move_model()
    p = m.plan_to_entity(e, touch=touch)
    PLAN["optimistic"] = False
    if p is None and optimistic:
        p = m.relax().plan_to_entity(e, touch=touch)
        PLAN["optimistic"] = p is not None
    return p

ARCH = {"levels": []}  # completed levels of this game: {"frames", "actions", "final_action"} (symbolic)
RULES = {"rules": [], "report": None, "level": None}
GOALS = {}

def symlog(all_levels=False):
    """This level's transitions as entity-level records (before entities, action, after entities)."""
    t = TRK["t"]
    log = _dsl.make_log(t.compound_frames(), t.actions, t.unders, t.bg)
    if all_levels:
        for lv in ARCH["levels"]:
            log = _dsl.make_log(lv["frames"], lv["actions"], lv.get("unders"), lv.get("bg")) + log
    return log

def _hud_ids():
    return TRK["t"].hud_ids()

def _rule_view(rule, score=None):
    d = rule.to_dict()
    if score is not None:
        d["support"] = score.support
        d["contradictions"] = score.contradictions
    d["rule"] = rule
    return d

def fit_rules(kind=None, all_levels=False):
    """Fit rule types to the transition log by enumeration: kind in move, push, drift, vanish, overlap, recolor,
    counter (None = all). Returns {kind: [{'text', 'support', 'contradictions', 'rule'}, ...]}; only rules with
    zero contradictions are returned."""
    log = symlog(all_levels)
    return {k: [_rule_view(r, sc) for r, sc in lst] for k, lst in _dsl.fit(kind, log).items()}

def auto_rules(all_levels=False):
    """Fit and select a consistent rule set for this level's log (movement with blocking, pushing, vanish on
    overlap/click/ACT, recolouring, counters). Returns {'rules': [text], 'coverage', 'fully_explained', 'transitions',
    'unexplained': [...], 'contradictions'}. coverage 1.0 = every observed event is explained: then
    set_model(rules_predictor()) and plan with plan_rules(). Otherwise the unexplained items say what to probe."""
    log = symlog(all_levels)
    rules, rep = _dsl.auto_rules(log, ignore_ids=_hud_ids())
    RULES["rules"] = rules
    RULES["report"] = rep
    RULES["level"] = LOG["level"]
    out = {"rules": [r.describe() for r in rules], "coverage": round(rep["coverage"], 3), "fully_explained": rep["fully_explained"],
           "transitions": rep["transitions"], "contradictions": rep["contradictions"],
           "unexplained": [{k: v for k, v in u.items() if k != "observed"} | {"observed": _obs_text(u.get("observed"))} for u in rep["unexplained"][:6]]}
    return out

def _obs_text(o):
    if o is None:
        return None
    if o.gone:
        return "disappeared"
    parts = []
    if o.moved and o.moved != (0, 0):
        parts.append(f"moved ({o.moved[0]:+d},{o.moved[1]:+d})")
    if o.recolor:
        parts.append(f"colour {o.recolor[0]}->{o.recolor[1]}")
    if o.resized:
        parts.append(f"size {o.resized[0]}->{o.resized[1]}")
    return "; ".join(parts) or "unchanged"

def rules(rules=None):
    """The current rule objects (from auto_rules or set_rules); pass a list to replace them."""
    if rules is not None:
        RULES["rules"] = list(rules)
    return list(RULES["rules"])

def explain_rules(rules=None, all_levels=False):
    """Coverage report of a rule set against the log (default: the current rules)."""
    rep = _dsl.explain(list(rules) if rules is not None else RULES["rules"], symlog(all_levels), ignore_ids=_hud_ids())
    rep["unexplained"] = [{k: v for k, v in u.items() if k != "observed"} | {"observed": _obs_text(u.get("observed"))} for u in rep["unexplained"]]
    return rep

def rules_predictor(rules=None):
    """predict(grid, action) built from the rule set, for set_model(...) so every real action verifies it."""
    rs = list(rules) if rules is not None else RULES["rules"]
    if not rs:
        auto_rules()  # fit lazily (exp-009: seven calls lost to "call auto_rules() first")
        rs = list(RULES["rules"])
        if not rs:
            raise ValueError("no rules could be fitted from this level's transitions yet: probe more, then retry")
    t = TRK["t"]
    fn = _dsl.predictor(rs, t.shapes, t.bg if t.bg is not None else 0, ref=lambda: t.compound_frames()[-1] if t.frames else None,
                        under=lambda: t.under)
    fn._arc3_rules = rs  # noqa: SLF001  (set_model turns it into a live predictor that follows PLAN['optimistic'])
    return fn

def _goal_fn(goal):
    if callable(goal):
        return goal
    if isinstance(goal, str):
        if goal in GOALS:
            return GOALS[goal]
        raise ValueError(f"unknown goal name {goal!r}; see goal_candidates() or pass a dict/callable")
    if isinstance(goal, dict) and len(goal) == 1:
        (k, v), = goal.items()
        if k == "none_left":
            return lambda f: not any(e.color == int(v) for e in f)
        if k == "count":
            c, n = v
            return lambda f: sum(1 for e in f if e.color == int(c)) == int(n)
        if k in ("overlap", "touch"):
            a, b = v
            pad = 1 if k == "touch" else 0
            return lambda f: any(x.overlaps(y, pad) for x in f if x.color == int(a) for y in f if y.color == int(b))
        if k in ("same_box", "same_columns", "same_rows", "inside"):
            a, b = int(v[0]), int(v[1])
            def _rel(x, y, k=k):
                if x is y:
                    return False
                if k == "same_box":
                    return (x.x0, x.y0, x.x1, x.y1) == (y.x0, y.y0, y.x1, y.y1)
                if k == "same_columns":
                    return (x.x0, x.x1) == (y.x0, y.x1) and not (x.y0 <= y.y1 and y.y0 <= x.y1)
                if k == "same_rows":
                    return (x.y0, x.y1) == (y.y0, y.y1) and not (x.x0 <= y.x1 and y.x0 <= x.x1)
                return (y.x0 <= x.x0 and x.x1 <= y.x1 and y.y0 <= x.y0 and x.y1 <= y.y1
                        and (x.x1 - x.x0 + 1) * (x.y1 - x.y0 + 1) < (y.x1 - y.x0 + 1) * (y.y1 - y.y0 + 1))
            return lambda f: any(_rel(x, y) for x in f if x.color == a for y in f if y.color == b)
        if k == "reach":
            av = TRK["t"].avatar()
            if not av:
                raise ValueError("reach needs a known avatar")
            aid, (x, y) = int(av["id"]), v
            return lambda f: any(e.id == aid and e.contains(int(x), int(y)) for e in f)
        if k == "reach_entity":
            av = TRK["t"].avatar()
            tgt = TRK["t"].get(int(v))
            if not av or tgt is None:
                raise ValueError("reach_entity needs a known avatar and an existing target id")
            aid = int(av["id"])
            box = _dsl.Ent(-1, tgt.color, tgt.x0, tgt.y0, tgt.w, tgt.h, tgt.size, tgt.shape_hash)
            return lambda f: any(e.id == aid and e.overlaps(box, 1) for e in f)
    raise ValueError("goal must be a callable(frame)->bool, a goal_candidates() name, or one of "
                     "{'none_left': colour}, {'count': (colour, n)}, {'overlap': (a, b)}, {'touch': (a, b)}, {'same_box': (a, b)}, "
                     "{'same_columns': (a, b)}, {'same_rows': (a, b)}, {'inside': (a, b)}, {'reach': (x, y)}, {'reach_entity': id}")

def plan_rules(goal, rules=None, max_depth=200, max_nodes=40000, optimistic=True):
    """Shortest action list reaching `goal` in the rule-set simulation from the current frame (BFS), or None.
    goal: {'none_left': colour} | {'count': (colour, n)} | {'overlap': (a, b)} | {'touch': (a, b)} | {'same_box': (a, b)} |
    {'same_columns': (a, b)} | {'same_rows': (a, b)} | {'inside': (a, b)} | {'reach': (x, y)} | {'reach_entity': id} |
    a goal_candidates() name | callable(frame)->bool. Execute with act(plan) after
    set_model(rules_predictor()). If no path exists under the fitted rules, an optimistic path is returned (unknown
    colours assumed passable; PLAN['optimistic'] is True): its first wrong step is caught and becomes evidence."""
    rs = list(rules) if rules is not None else RULES["rules"]
    if not rs:
        auto_rules()  # fit lazily (exp-009: seven calls lost to "call auto_rules() first")
        rs = list(RULES["rules"])
        if not rs:
            raise ValueError("no rules could be fitted from this level's transitions yet: probe more, then retry")
    t = TRK["t"]
    frame = t.compound_frames()[-1]
    _dsl.set_terrain(rs, t.under, t.bg)
    acts = _dsl.planning_actions(rs, frame)
    fn = _goal_fn(goal)
    p = _dsl.plan(rs, frame, fn, acts, max_nodes=max_nodes, max_depth=max_depth)
    PLAN["optimistic"] = False
    if p is None and optimistic:
        # no path through known-passable cells: assume unknown colours are passable (bumps still count) and let
        # the verified execution find out; PLAN['optimistic'] tells you the plan is an experiment
        relaxed = _dsl.optimistic(rs)
        _dsl.set_terrain(relaxed, t.under, t.bg)
        p = _dsl.plan(relaxed, frame, fn, _dsl.planning_actions(relaxed, frame), max_nodes=max_nodes, max_depth=max_depth)
        PLAN["optimistic"] = p is not None
    return p

def goal_candidates():
    """Win-condition candidates consistent with every completed level so far: true at the winning frame, false
    before. Names can be passed to plan_rules(). Empty until a level has been completed."""
    levels = [(list(lv["frames"]) + [lv["final_frame"]], True) for lv in ARCH["levels"] if lv.get("final_frame") is not None]
    if not levels:
        return ["(no level completed yet in this game: nothing to infer the win condition from; use goal_hints() and the Goal line of the observation)"]
    out = _dsl.goal_predicates(levels)
    GOALS.clear()
    for g in out:
        GOALS[g["goal"]] = g["predicate"]
    return [g["goal"] for g in out]

def goal_hints(limit=8):
    """Structural goal hypotheses for a level whose win condition is unknown: unique-colour entities, entities
    shaped like the avatar (slots), collectible sets (several small same-colour entities), and the largest
    non-static entities. Each hint has 'hint', 'ids' and a suggested plan_rules goal."""
    t = TRK["t"]
    frame = t.compound_frames()[-1] if t.frames else ()
    roles = t.roles()
    av = t.avatar()
    aid = int(av["id"]) if av else None
    ents_ = [e for e in frame if roles.get(e.id) != "hud" and e.id != aid]
    by_color = {}
    for e in ents_:
        by_color.setdefault(e.color, []).append(e)
    hints = []
    for c, lst in sorted(by_color.items(), key=lambda kv: len(kv[1])):
        if len(lst) == 1 and lst[0].size <= 400:
            e = lst[0]
            hints.append({"hint": f"unique colour {c} entity #{e.id} ({e.w}x{e.h}) may be the target or exit", "ids": [e.id],
                          "goal": {"reach_entity": e.id} if aid else {"touch": (c, c)}})
    if aid:
        a = t.get(aid)
        if a is not None:
            for e in ents_:
                if e.shape == a.shape_hash or (e.w == a.w and e.h == a.h and e.color != a.color):
                    hints.append({"hint": f"entity #{e.id} colour {e.color} has the avatar's size: a slot or a twin", "ids": [e.id],
                                  "goal": {"reach_entity": e.id}})
    for c, lst in by_color.items():
        if 2 <= len(lst) <= 12 and all(x.size <= 64 for x in lst):
            hints.append({"hint": f"{len(lst)} small colour-{c} entities: collect or visit them all", "ids": [x.id for x in lst][:12],
                          "goal": {"none_left": c}})
    dyn = [e for e in ents_ if roles.get(e.id) in ("dynamic", "unknown") and e.size <= 400]
    for e in sorted(dyn, key=lambda e: -e.size)[:2]:
        if not any(e.id in h["ids"] for h in hints):
            hints.append({"hint": f"non-static entity #{e.id} colour {e.color} ({e.w}x{e.h})", "ids": [e.id], "goal": {"reach_entity": e.id} if aid else None})
    return hints[:limit]

def probe_suggestions(limit=8):
    """Cheapest untested actions on this level: legal keys no fitted rule responds to, ACT if never pressed, and
    one click per entity class (colour, shape) never clicked. Each item is {'action', 'why'}; act(item['action'])."""
    t = TRK["t"]
    avail = list(G.get("available") or [])
    tried = set()
    clicked_classes = set()
    for a in t.actions:
        if isinstance(a, (tuple, list)) and a and str(a[0]).upper() == "CLICK":
            tried.add("CLICK")
            x, y = int(a[1]), int(a[2])
            for e in t.compound_frames()[0]:
                if e.contains(x, y):
                    clicked_classes.add((e.color, e.shape))
        else:
            tried.add(str(a).upper())
    keymaps = set()
    for r in RULES["rules"]:
        mv = r.move if r.kind == "push" else r
        if getattr(mv, "keymap", None):
            keymaps |= set(mv.keymap)
    out = []
    for k in ("UP", "DOWN", "LEFT", "RIGHT"):
        if k in avail and k not in tried:
            out.append({"action": k, "why": "never pressed"})
        elif k in avail and k not in keymaps and RULES["rules"]:
            out.append({"action": k, "why": "pressed but no rule responds to it; press it next to something"})
    if "ACT" in avail and "ACT" not in tried:
        out.append({"action": "ACT", "why": "never pressed"})
    if "CLICK" in avail and t.frames:
        roles = t.roles()
        seen_cls = set()
        for e in sorted(t.compound_frames()[-1], key=lambda e: -e.size):
            key = (e.color, e.shape)
            if key in clicked_classes or key in seen_cls or roles.get(e.id) == "hud":
                continue
            seen_cls.add(key)
            out.append({"action": ("CLICK", (e.x0 + e.x1) // 2, (e.y0 + e.y1) // 2), "why": f"entity #{e.id} colour {e.color} {e.w}x{e.h}: class never clicked"})
    return out[:limit]

def _archive_level(final_action, terminal=None):
    """Archive the completed level for goal_candidates(): the observed terminal frame when the harness supplied it
    (the engine returns it as the first layer of the level-completing step), else a simulated winning frame."""
    t = TRK["t"]
    if not t.frames:
        return
    final = None
    if terminal is not None:
        try:
            t.update(_to_grid(terminal), final_action)
            frames = t.compound_frames()
            ARCH["levels"].append({"level": LOG["level"], "frames": list(frames[:-1]), "actions": list(t.actions), "unders": list(t.unders),
                                   "bg": t.bg, "final_action": final_action, "final_frame": frames[-1], "observed": True})
            return
        except Exception:  # noqa: BLE001
            pass
    frames = t.compound_frames()
    rules_ = RULES["rules"] or _dsl.fallback_move_rules(t.avatar(), frames[-1])
    try:
        _dsl.set_terrain(rules_, t.under, t.bg)
        final = _dsl.simulate(frames[-1], final_action, rules_) if rules_ else None
    except Exception:  # noqa: BLE001
        final = None
    ARCH["levels"].append({"level": LOG["level"], "frames": list(frames), "actions": list(t.actions), "unders": list(t.unders),
                           "bg": t.bg, "final_action": final_action, "final_frame": final})

def tilemap(g=None, tile=None):
    """Text view with one hex colour per logical tile (default: the tracker's tile size); rows are y // tile."""
    t = int(tile) if tile else int(G.get("tile") or 1)
    return _P.tile_map(G["grid"] if g is None else _to_grid(g), t)

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
        wrong = (pred != after) & ~_hud_mask(after.shape)
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
    live = isinstance(getattr(predict, "__self__", None), _MoveModel)
    rules_live = getattr(predict, "_arc3_rules", None) is not None
    if rules_live:
        # rules_predictor(): verify an optimistic plan against the optimistic rules, so a step the strict rules call
        # "blocked" is a mismatch that stops the batch instead of a correct prediction of standing still (dc22,
        # exp-009: 48 actions spent in place with "49/49 predictions correct").
        strict_rules = list(predict._arc3_rules)  # noqa: SLF001
        strict_fn = predict
        t = TRK["t"]
        relaxed_rules = _dsl.optimistic(strict_rules)
        relaxed_fn = _dsl.predictor(relaxed_rules, t.shapes, t.bg if t.bg is not None else 0,
                                    ref=lambda: t.compound_frames()[-1] if t.frames else None, under=lambda: t.under)

        def predict(grid, action):  # noqa: F811
            if PLAN["optimistic"]:
                _dsl.set_terrain(relaxed_rules, t.under, t.bg)
                return relaxed_fn(grid, action)
            _dsl.set_terrain(strict_rules, t.under, t.bg)
            return strict_fn(grid, action)
    if live:
        # move_model().predict is a snapshot of the evidence at fit time; the harness registers a predictor that
        # re-fits from the tracker before every prediction (ka59, exp-009: a stale fit mismatched 14 times) and
        # follows PLAN['optimistic'] so an optimistic plan is checked against the optimistic model.
        def predict(grid, action):  # noqa: F811
            m = _MoveModel(TRK["t"])
            if PLAN["optimistic"]:
                m = m.relax()
            return m.predict(grid, action)
    WM["predict"] = predict
    WM["checked"] = WM["matched"] = WM["errors"] = WM["streak"] = 0
    WM["mismatches"] = []
    if predict is None:
        return "world model cleared"
    if live:
        return "world model registered (live move model: re-fitted from the evidence before every prediction)"
    if rules_live:
        return "world model registered (rules predictor; an optimistic plan is verified against the optimistic rules)"
    return "world model registered"

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

def _hud_mask(shape):
    """Cells of HUD strips (edge bars that count actions): left out of prediction checks."""
    m = np.zeros(shape, dtype=bool)
    t = TRK["t"]
    for eid in t.hud_ids():
        e = t.get(eid)
        if e is not None:
            m[e.y0:e.y1 + 1, e.x0:e.x1 + 1] = True
    return m

def _check_prediction(before, action, after):
    fn = WM["predict"]
    if fn is None:
        return None
    try:
        pred = fn(before.copy(), _action_label(action))
        pred = _to_grid(pred)
        if pred.shape != after.shape:
            raise ValueError(f"predicted shape {pred.shape} != {after.shape}")
        diff = (pred != after) & ~_hud_mask(after.shape)  # HUD bars are not part of the mechanics being modelled
        wrong = int(diff.sum())
    except Exception as e:  # noqa: BLE001
        WM["errors"] += 1
        return {"pred_ok": False, "pred_error": f"{type(e).__name__}: {e}"[:200]}
    WM["checked"] += 1
    if wrong == 0:
        WM["matched"] += 1
        WM["streak"] = 0
        return {"pred_ok": True, "pred_wrong_cells": 0}
    ys, xs = np.nonzero(diff)
    d = {"pred_ok": False, "pred_wrong_cells": wrong,
         "pred_bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
         "pred_sample": [[int(x), int(y), int(pred[y, x]), int(after[y, x])] for y, x in list(zip(ys, xs))[:5]]}
    WM["mismatches"].append({"action": _action_label(action), "wrong_cells": wrong, "bbox": d["pred_bbox"]})
    WM["streak"] = WM.get("streak", 0) + 1
    if WM["streak"] >= WM_RETIRE_AFTER:
        # A model wrong three times running is not being revised; keep checking it and every batch stops after
        # one action (ka59, exp-009: 14 mismatches, one action per call). Drop it and say so.
        WM["predict"] = None
        WM["streak"] = 0
        d["pred_retired"] = (f"world model retired after {WM_RETIRE_AFTER} consecutive mismatches; re-fit it "
                             "(move_model() / rules_predictor()) or rewrite predict, then set_model again")
        CELL["pred_retired"] = {"action": _action_label(action), "wrong_cells": wrong,
                                "recent": [m["action"] for m in WM["mismatches"][-WM_RETIRE_AFTER:]]}
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
        if callable(G.get(k)) and not isinstance(G.get(k), type(None)):
            # the model defined a function with a state variable's name (wa30 def state(), exp-009): keep its function
            # and expose the value under STATE[k] instead of silently replacing the function with a string
            G.setdefault("STATE", {})[k] = state.get(k)
            continue
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

def ascii(g=None, scale=None, region=None):
    """Text view of the board. Without `region` this is the tile map (one hex colour per logical tile, the same map as
    in the observation). region=(x0, y0, x1, y1) shows those pixels at full resolution (at most 32x32 cells)."""
    g = G["grid"] if g is None else _to_grid(g)
    if region is not None:
        x0, y0, x1, y1 = (int(v) for v in region)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(g.shape[1] - 1, max(x0, x1)), min(g.shape[0] - 1, max(y0, y1))
        sub = g[y0:y1 + 1, x0:x1 + 1]
        if sub.size > 1024:
            # too big for pixels: show the region at tile resolution instead of refusing (exp-011: nine refused calls)
            t = int(G.get("tile") or 1)
            if t <= 1:
                t = int(np.ceil(np.sqrt(sub.size / 1024)))
            return (f"(region {x0},{y0}-{x1},{y1} is {sub.shape[1]}x{sub.shape[0]} pixels, more than 32x32: shown at one char per "
                    f"{t}x{t} block; ask for a smaller region for pixels)\n" + _P.tile_map(sub, t))
        return _P.ascii(sub, scale=1)
    t = int(G.get("tile") or 1)
    if t <= 1:
        return _P.ascii(g, scale=scale)
    return f"(tile map, one char per {t}x{t} tile; ascii(region=(x0,y0,x1,y1)) shows pixels)\n" + _P.tile_map(g, t)

def downscale(g=None):
    """The board at one cell per logical tile (numpy array). Uses the engine's pixel upscale when there is one,
    else the tracker's tile size (ft09's 6-px tiles with 1-px gaps, exp-009: the raw 64x64 grid came back and the
    model rebuilt the tile map by hand over four calls). tilemap() is the same view as text."""
    g = G["grid"] if g is None else _to_grid(g)
    small, s = _P.downscale(g)
    tile = int(G.get("tile") or 1)
    if s <= 1 and tile > 1:
        h, w = g.shape
        ys = [min(h - 1, y * tile + tile // 2) for y in range((h + tile - 1) // tile)]
        xs = [min(w - 1, x * tile + tile // 2) for x in range((w + tile - 1) // tile)]
        return g[np.ix_(ys, xs)]
    return small

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

def learn(text, kind="mistake"):
    """Record a lesson ("what did we learn?"): kept for the whole game, shown every turn, carried to the next level,
    and shared with the other games of this run when it is a mechanic, hazard, recipe or strategy."""
    G["lessons"].append({"kind": str(kind), "text": str(text)})
    return len(G["lessons"])

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
            r = reply["result"][0]
            if r.get("level_completed"):
                _archive_level(_action_label(a), r.pop("terminal", None))
            _refresh(reply["state"], _action_label(a))
            r["events"] = _last_event_text()
            if not r.get("level_completed"):
                LOG["transitions"].append((before, _action_label(a), G["grid"].copy()))
                hyp = _check_hypotheses(before, _action_label(a), G["grid"])
                if hyp:
                    r.update(hyp)
            results.append(r)
            if r.get("level_completed") or r.get("game_over") or r.get("won"):
                break
            if _idle_streak(results) >= IDLE_STOP and len(norm) > len(results):
                r["batch_stopped"] = f"{IDLE_STOP} actions in a row changed nothing; stopped after {len(results)} of {len(norm)} (the rest would be wasted)"
                CELL["batch_stopped"] = {"done": len(results), "planned": len(norm), "idle": [_action_label(a) for a in norm[len(results) - IDLE_STOP:len(results)]]}
                break
        return results if len(norm) > 1 else _Result(results[0])
    # With a registered world model, actions go one at a time so each prediction is checked and a
    # mismatch stops the batch (the model must revise before spending more actions).
    results = []
    for a in norm:
        before = G["grid"].copy()
        _send({"type": "action", "actions": [a]})
        reply = _recv()
        if reply.get("type") != "action_result":
            raise RuntimeError(reply.get("error", "action failed"))
        r = reply["result"][0]
        if r.get("level_completed"):
            _archive_level(_action_label(a), r.pop("terminal", None))
        _refresh(reply["state"], _action_label(a))
        r["events"] = _last_event_text()
        chk = None if r.get("level_completed") else _check_prediction(before, a, G["grid"])  # a new level's first frame is not a prediction target
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
        if _idle_streak(results) >= IDLE_STOP and len(norm) > len(results):
            r["batch_stopped"] = f"{IDLE_STOP} actions in a row changed nothing; stopped after {len(results)} of {len(norm)} (the rest would be wasted)"
            CELL["batch_stopped"] = {"done": len(results), "planned": len(norm), "idle": [_action_label(a) for a in norm[len(results) - IDLE_STOP:len(results)]]}
            break
        if WM["predict"] is None and len(norm) > len(results):  # retired mid-batch: the rest runs unverified
            return results + act(*norm[len(results):]) if len(norm) - len(results) > 1 else results + [act(norm[len(results)])]
        if r.get("level_completed") or r.get("game_over") or r.get("won"):
            break
    return results if len(norm) > 1 else _Result(results[0])

class _Result(dict):
    """A single action's result: a dict that also iterates as a one-item list, because `for r in act('ACT')` is
    what models write (exp-011 wa30: iterating the keys gave "'str' object has no attribute 'get'")."""

    def __iter__(self):
        yield self


IDLE_STOP = 3  # consecutive no-change actions that end a batch (dc22, exp-009: 13- and 21-action batches spent standing still)

def _idle_streak(results):
    n = 0
    for r in reversed(results):
        if int(r.get("changed") or 0) == 0 and not r.get("level_completed"):
            n += 1
        else:
            break
    return n

def click(x, y):
    return act(("CLICK", int(x), int(y)))

HELPER_NAMES = ("objects", "components", "diff", "ascii", "tilemap", "downscale", "background", "moved", "note", "learn", "act", "click",
                "set_model", "world_model_stats", "verify_model", "transitions", "set_models", "alive_models",
                "ents", "events", "event_log", "describe_events", "avatar", "roles", "entity",
                "move_model", "plan_to", "plan_to_entity",
                "symlog", "fit_rules", "auto_rules", "rules", "explain_rules", "rules_predictor", "plan_rules", "goal_candidates", "goal_hints", "probe_suggestions", "PLAN")
HELPERS = {_n: globals()[_n] for _n in HELPER_NAMES}
G.update(HELPERS)

def _restore_helpers():
    """Helpers the cell rebound (``for act in plan:`` or ``rules = auto_rules()``) come back, with a note: exp-009 lost
    eight calls to "'str' object is not callable" after such rebinding."""
    lost = [n for n in HELPER_NAMES if G.get(n) is not HELPERS[n]]
    for n in lost:
        G[n] = HELPERS[n]
    return lost

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
    _n_log = len(TRK["t"].log)
    CELL["pred_retired"] = CELL["batch_stopped"] = None
    G["lessons"] = []
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
    lost = _restore_helpers()
    if lost:
        out += f"\n[helper(s) {', '.join(lost)} were rebound by your code and have been restored; use other variable names]"
    shadowed = sorted(k for k in (G.get("STATE") or {}) if callable(G.get(k)))
    if shadowed and not G.get("_state_note_done"):
        G["_state_note_done"] = True
        out += f"\n[your function(s) {', '.join(shadowed)} shadow harness variables; their current values are in STATE[...]]"
    cell_events = [_Tracker.describe(rec, TRK["t"])[:220] for rec in TRK["t"].log[_n_log:]][:10]
    _send({"type": "final", "stdout": out, "error": err, "result": result, "notes": G.get("notes", []), "events": cell_events,
           "lessons": list(G.get("lessons") or []), "flags": {k: v for k, v in CELL.items() if v},
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
        self.closed = False  # set by stop(): a cell still running in another thread must not respawn the child
        self._lock = threading.Lock()

    # ---------- process management ----------
    def start(self) -> None:
        self.stop()
        self.closed = False
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
        """Kill the child. The sandbox stays closed until start() is called again explicitly: run() then returns an
        error instead of spawning a new process (the REPL agent's close() used to leave one behind when a cell was
        still executing on the worker thread)."""
        self.closed = True
        p = self.proc
        self.proc = None
        if p is None:
            return
        # Teardown must never raise: the group may already be gone, or the child may have changed session.
        try:
            os.killpg(p.pid, 9)
        except OSError:
            with contextlib.suppress(OSError):
                p.kill()
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            p.wait(timeout=2)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def restart(self) -> None:
        if self.closed:
            return
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
                if self.closed and self.runs > 0:
                    raise RuntimeError("sandbox closed")
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
                    with contextlib.suppress(OSError, ValueError):  # the pipe may be closed already
                        err = (self.proc.stderr.read() if self.proc and self.proc.stderr else "")[-1500:]
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
                    return {"stdout": out, "error": str(msg.get("error") or ""), "result": msg.get("result"), "events": msg.get("events") or [],
                            "notes": list(msg.get("notes") or []), "actions": n_actions, "timed_out": False,
                            "restarted": restarted, "world_model": msg.get("world_model"),
                            "lessons": list(msg.get("lessons") or []), "flags": dict(msg.get("flags") or {})}

    def __del__(self):  # pragma: no cover
        with contextlib.suppress(Exception):  # interpreter shutdown: modules may be half torn down
            self.stop()
