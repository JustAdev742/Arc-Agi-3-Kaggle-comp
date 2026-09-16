"""Rule language, fitter and verifier over the symbolic transition log (plan-100 Part II, component B).

The transition log is entity-level: a frame is a tuple of light entities (id, colour, box, size, shape), an
action is a label ('UP', 'ACT', ('CLICK', x, y), ...), and a transition is (before, action, after).

A rule is a parameterised claim about what entities of one class do on a transition. Code enumerates the
parameter space (``fit``), keeps the parameterisations with zero contradictions against the log and the
largest support, and reports what a rule set leaves unexplained (``explain``). ``simulate`` applies a rule
set to a frame so a planner can search it, and ``predictor`` wraps that as a grid predictor for the sandbox's
``set_model`` so every real action keeps verifying it.

The model only chooses rule types and reads counter-examples; it never has to write the arithmetic.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Iterable, Optional

import numpy as np

KEYS = ("UP", "DOWN", "LEFT", "RIGHT")
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
N = 64
SHAPES: dict[str, np.ndarray] = {}  # shape hash -> boolean mask; lets hollow entities (frames, walls) block by cells, not boxes


def register_shapes(shapes: dict[str, np.ndarray]) -> None:
    SHAPES.update(shapes)


def _dilate(m: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return m
    h, w = m.shape
    out = np.zeros((h + 2 * pad, w + 2 * pad), dtype=bool)
    for dy in range(2 * pad + 1):
        for dx in range(2 * pad + 1):
            out[dy:dy + h, dx:dx + w] |= m
    return out


# ---------------------------------------------------------------------------------------------- data model
@dataclass(frozen=True)
class Ent:
    id: int
    color: int
    x0: int
    y0: int
    w: int
    h: int
    size: int
    shape: str

    @property
    def x1(self) -> int:
        return self.x0 + self.w - 1

    @property
    def y1(self) -> int:
        return self.y0 + self.h - 1

    def moved(self, dx: int, dy: int) -> "Ent":
        return Ent(self.id, self.color, self.x0 + dx, self.y0 + dy, self.w, self.h, self.size, self.shape)

    def recolored(self, c: int) -> "Ent":
        return Ent(self.id, c, self.x0, self.y0, self.w, self.h, self.size, self.shape)

    def overlaps(self, o: "Ent", pad: int = 0) -> bool:
        """Do the two entities share a cell (pad=1: or touch), by mask when the shapes are known, else by box."""
        if self.x1 + pad < o.x0 or self.x0 - pad > o.x1 or self.y1 + pad < o.y0 or self.y0 - pad > o.y1:
            return False
        ma, mb = self._mask(), o._mask()
        if ma is None and mb is None:
            return True
        a = _dilate(ma if ma is not None else np.ones((self.h, self.w), dtype=bool), pad)
        b = mb if mb is not None else np.ones((o.h, o.w), dtype=bool)
        return _masks_intersect(a, self.x0 - pad, self.y0 - pad, b, o.x0, o.y0)

    def box_overlaps(self, x0: int, y0: int, w: int, h: int) -> bool:
        """Does this entity occupy any cell of the box (by mask when known)?"""
        if x0 + w - 1 < self.x0 or x0 > self.x1 or y0 + h - 1 < self.y0 or y0 > self.y1:
            return False
        m = self._mask()
        if m is None:
            return True
        return _masks_intersect(m, self.x0, self.y0, np.ones((h, w), dtype=bool), x0, y0)

    def _mask(self) -> Optional[np.ndarray]:
        m = SHAPES.get(self.shape)
        if m is None or m.shape != (self.h, self.w) or bool(m.all()):
            return None  # unknown or solid: the box is exact
        return m

    def contains(self, x: int, y: int) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def key(self) -> tuple:
        return (self.id, self.color, self.x0, self.y0, self.w, self.h, self.size)


def _masks_intersect(a: np.ndarray, ax: int, ay: int, b: np.ndarray, bx: int, by: int) -> bool:
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + a.shape[1], bx + b.shape[1]), min(ay + a.shape[0], by + b.shape[0])
    if x1 <= x0 or y1 <= y0:
        return False
    return bool((a[y0 - ay:y1 - ay, x0 - ax:x1 - ax] & b[y0 - by:y1 - by, x0 - bx:x1 - bx]).any())


Frame = tuple[Ent, ...]


def ent_from_tracker(e: Any) -> Ent:
    return Ent(int(e.id), int(e.color), int(e.x0), int(e.y0), int(e.w), int(e.h), int(e.size), str(e.shape_hash))


def action_kind(action: Any) -> str:
    """'UP'/'DOWN'/'LEFT'/'RIGHT'/'ACT'/'CLICK'/'UNDO'/'RESET'/other."""
    if isinstance(action, (tuple, list)):
        return str(action[0]).upper()
    return str(action).upper()


def click_xy(action: Any) -> Optional[tuple[int, int]]:
    if isinstance(action, (tuple, list)) and len(action) == 3 and str(action[0]).upper() == "CLICK":
        return int(action[1]), int(action[2])
    return None


@dataclass(frozen=True)
class Obs:
    """What one entity of the before-frame did: displacement, disappearance, recolouring, resizing."""
    moved: Optional[tuple[int, int]]  # None when gone
    gone: bool
    recolor: Optional[tuple[int, int]]
    resized: Optional[tuple[int, int]]

    def trivial(self) -> bool:
        return not self.gone and self.moved == (0, 0) and self.recolor is None and self.resized is None


@dataclass
class Transition:
    before: Frame
    action: Any
    after: Frame
    _b: dict[int, Ent] = field(default_factory=dict, repr=False)
    _a: dict[int, Ent] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._b = {e.id: e for e in self.before}
        self._a = {e.id: e for e in self.after}

    def obs(self, eid: int) -> Optional[Obs]:
        b = self._b.get(eid)
        if b is None:
            return None
        a = self._a.get(eid)
        if a is None:
            return Obs(None, True, None, None)
        return Obs((a.x0 - b.x0, a.y0 - b.y0), False,
                   (b.color, a.color) if a.color != b.color else None,
                   (b.size, a.size) if a.size != b.size else None)

    def appeared(self) -> list[Ent]:
        return [e for e in self.after if e.id not in self._b]


def make_log(frames: list[Frame], actions: list[Any]) -> list[Transition]:
    """Transitions from aligned frames/actions (frames[i] --actions[i]--> frames[i+1])."""
    n = min(len(actions), len(frames) - 1)
    return [Transition(frames[i], actions[i], frames[i + 1]) for i in range(n)]


# ---------------------------------------------------------------------------------------------- classes
@dataclass(frozen=True)
class Cls:
    """An entity class: by colour, optionally by shape, or an explicit id set."""
    color: Optional[int] = None
    shape: Optional[str] = None
    ids: Optional[frozenset[int]] = None

    def matches(self, e: Ent) -> bool:
        if self.ids is not None and e.id not in self.ids:
            return False
        if self.color is not None and e.color != self.color:
            return False
        return not (self.shape is not None and e.shape != self.shape)

    def select(self, frame: Iterable[Ent]) -> list[Ent]:
        return [e for e in frame if self.matches(e)]

    def __str__(self) -> str:
        parts = []
        if self.color is not None:
            parts.append(f"colour {self.color}")
        if self.shape is not None:
            parts.append(f"shape {self.shape[:6]}")
        if self.ids is not None:
            parts.append("ids " + ",".join(f"#{i}" for i in sorted(self.ids)))
        return " ".join(parts) or "anything"


def _classes_of(ents: Iterable[Ent], *, with_shape: bool = True) -> list[Cls]:
    out: list[Cls] = []
    seen: set[tuple] = set()
    for e in ents:
        for c in ((Cls(color=e.color), Cls(color=e.color, shape=e.shape)) if with_shape else (Cls(color=e.color),)):
            k = (c.color, c.shape)
            if k not in seen:
                seen.add(k)
                out.append(c)
    return out


# ---------------------------------------------------------------------------------------------- claims
@dataclass(frozen=True)
class Claim:
    """A rule's statement about one entity; None fields are 'no opinion'."""
    moved: Optional[tuple[int, int]] = None
    gone: Optional[bool] = None
    recolor: Optional[tuple[int, int]] = None
    resized: Optional[tuple[int, int]] = None

    def contradicts(self, o: Obs) -> bool:
        if self.gone is not None and self.gone != o.gone:
            return True
        if o.gone:
            return False  # a vanished entity has no position or colour to contradict
        if self.moved is not None and self.moved != o.moved:
            return True
        if self.recolor is not None and self.recolor != o.recolor:
            return True
        return self.resized is not None and self.resized != o.resized

    def events(self) -> set[str]:
        """The non-trivial event components this claim asserts."""
        s = set()
        if self.moved is not None and self.moved != (0, 0):
            s.add("moved")
        if self.gone:
            s.add("gone")
        if self.recolor is not None:
            s.add("recolor")
        if self.resized is not None:
            s.add("resized")
        return s


def obs_events(o: Obs) -> set[str]:
    s = set()
    if o.gone:
        s.add("gone")
    else:
        if o.moved != (0, 0):
            s.add("moved")
        if o.recolor is not None:
            s.add("recolor")
        if o.resized is not None:
            s.add("resized")
    return s


# ---------------------------------------------------------------------------------------------- rules
class Rule:
    kind = "rule"
    cls: Cls

    def claims(self, tr: Transition) -> dict[int, Claim]:
        raise NotImplementedError

    def apply(self, before: Frame, action: Any, frame: Frame) -> Frame:
        """Entity-level step: ``before`` is the frame at the start of the transition, ``frame`` the frame after
        the rules applied so far. Returns the updated frame."""
        return frame

    def describe(self) -> str:
        return self.kind

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.describe()}

    def __repr__(self) -> str:
        return f"<{self.describe()}>"


def _blocked(target: Ent, others: Iterable[Ent], blocked_by: Any, dx: int, dy: int) -> bool:
    x0, y0 = target.x0 + dx, target.y0 + dy
    if x0 < 0 or y0 < 0 or x0 + target.w > N or y0 + target.h > N:
        return True
    if blocked_by is None:
        return False
    for o in others:
        if o.id == target.id:
            continue
        if (blocked_by == "any" or o.color in blocked_by) and o.box_overlaps(x0, y0, target.w, target.h):
            return True
    return False


class Move(Rule):
    """Entities of ``cls`` move by keymap[key] on that key unless the target box is off-frame or overlaps an
    entity whose colour is in ``blocked_by`` ('any' = every other entity, None = nothing blocks)."""
    kind = "move"

    def __init__(self, cls: Cls, keymap: dict[str, tuple[int, int]], blocked_by: Any = "any"):
        self.cls = cls
        self.keymap = {k: (int(v[0]), int(v[1])) for k, v in keymap.items()}
        self.blocked_by = blocked_by if blocked_by in (None, "any") else frozenset(int(c) for c in blocked_by)

    def delta(self, action: Any) -> Optional[tuple[int, int]]:
        k = action_kind(action)
        if k in self.keymap:
            return self.keymap[k]
        if k in KEYS:
            return None  # unknown key: no opinion
        return (0, 0)  # non-key actions do not move this class

    def claims(self, tr: Transition) -> dict[int, Claim]:
        d = self.delta(tr.action)
        if d is None:
            return {}
        out = {}
        for e in self.cls.select(tr.before):
            if d == (0, 0) or _blocked(e, tr.before, self.blocked_by, *d):
                out[e.id] = Claim(moved=(0, 0))
            else:
                out[e.id] = Claim(moved=d)
        return out

    def apply(self, before: Frame, action: Any, frame: Frame) -> Frame:
        d = self.delta(action)
        if not d:
            return frame
        out = []
        for e in frame:
            if self.cls.matches(e) and not _blocked(e, frame, self.blocked_by, *d):
                out.append(e.moved(*d))
            else:
                out.append(e)
        return tuple(out)

    def describe(self) -> str:
        km = ", ".join(f"{k}:({v[0]:+d},{v[1]:+d})" for k, v in self.keymap.items())
        b = "nothing" if self.blocked_by is None else ("any entity" if self.blocked_by == "any" else f"colours {sorted(self.blocked_by)}")
        return f"move[{self.cls}] keys {{{km}}} blocked by {b}"

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "cls": str(self.cls), "keymap": self.keymap,
                "blocked_by": None if self.blocked_by is None else ("any" if self.blocked_by == "any" else sorted(self.blocked_by))}


class Push(Rule):
    """A mover (with its Move rule) pushes entities of ``pushable`` one step along its direction when the
    pushed entity's own target is free; otherwise neither moves."""
    kind = "push"

    def __init__(self, move: Move, pushable: Cls, blocked_by: Any = "any"):
        self.move = move
        self.cls = move.cls
        self.pushable = pushable
        self.blocked_by = blocked_by if blocked_by in (None, "any") else frozenset(int(c) for c in blocked_by)

    def _resolve(self, frame: Frame, action: Any) -> dict[int, tuple[int, int]]:
        """id -> displacement for movers and pushed entities this action."""
        d = self.move.delta(action)
        moves: dict[int, tuple[int, int]] = {}
        if not d or d == (0, 0):
            return moves
        dx, dy = d
        by_id = {e.id: e for e in frame}
        for m in self.move.cls.select(frame):
            x0, y0 = m.x0 + dx, m.y0 + dy
            if x0 < 0 or y0 < 0 or x0 + m.w > N or y0 + m.h > N:
                moves[m.id] = (0, 0)
                continue
            hit = [e for e in frame if e.id != m.id and self.pushable.matches(e) and e.box_overlaps(x0, y0, m.w, m.h)]
            if not hit:
                if _blocked(m, frame, self.move.blocked_by, dx, dy):
                    moves[m.id] = (0, 0)
                else:
                    moves[m.id] = (dx, dy)
                continue
            # a chain of one: the pushed entity must have a free target
            ok = True
            for p in hit:
                others = [e for e in frame if e.id not in (p.id, m.id)]
                if _blocked(p, others, self.blocked_by, dx, dy):
                    ok = False
            if ok:
                moves[m.id] = (dx, dy)
                for p in hit:
                    moves[p.id] = (dx, dy)
            else:
                moves[m.id] = (0, 0)
                for p in hit:
                    moves[p.id] = (0, 0)
        for e in frame:
            if self.pushable.matches(e) and e.id not in moves and e.id in by_id:
                moves[e.id] = (0, 0)  # not pushed this action
        return moves

    def claims(self, tr: Transition) -> dict[int, Claim]:
        return {eid: Claim(moved=d) for eid, d in self._resolve(tr.before, tr.action).items()}

    def apply(self, before: Frame, action: Any, frame: Frame) -> Frame:
        moves = self._resolve(frame, action)
        return tuple(e.moved(*moves[e.id]) if e.id in moves else e for e in frame)

    def describe(self) -> str:
        return f"push[{self.cls} pushes {self.pushable}] ({self.move.describe()})"

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "mover": str(self.cls), "pushable": str(self.pushable), "move": self.move.to_dict()}


class Drift(Rule):
    """Entities of ``cls`` move by (dx, dy) on every action (or on every key action when keys_only)."""
    kind = "drift"

    def __init__(self, cls: Cls, dx: int, dy: int, keys_only: bool = False):
        self.cls, self.dx, self.dy, self.keys_only = cls, int(dx), int(dy), keys_only

    def _active(self, action: Any) -> bool:
        return not self.keys_only or action_kind(action) in KEYS

    def claims(self, tr: Transition) -> dict[int, Claim]:
        d = (self.dx, self.dy) if self._active(tr.action) else (0, 0)
        return {e.id: Claim(moved=d) for e in self.cls.select(tr.before)}

    def apply(self, before: Frame, action: Any, frame: Frame) -> Frame:
        if not self._active(action):
            return frame
        return tuple(e.moved(self.dx, self.dy) if self.cls.matches(e) else e for e in frame)

    def describe(self) -> str:
        return f"drift[{self.cls}] ({self.dx:+d},{self.dy:+d}) per {'key' if self.keys_only else 'action'}"


class Vanish(Rule):
    """Entities of ``cls`` disappear on a trigger: an action kind ('ACT', 'UP', ...), 'CLICK@self' (clicked
    inside the entity), or 'any' (every action)."""
    kind = "vanish"

    def __init__(self, cls: Cls, trigger: str):
        self.cls, self.trigger = cls, trigger

    def _fires(self, e: Ent, action: Any) -> bool:
        if self.trigger == "any":
            return True
        if self.trigger == "CLICK@self":
            xy = click_xy(action)
            return xy is not None and e.contains(*xy)
        return action_kind(action) == self.trigger

    def claims(self, tr: Transition) -> dict[int, Claim]:
        return {e.id: Claim(gone=self._fires(e, tr.action)) for e in self.cls.select(tr.before)}

    def apply(self, before: Frame, action: Any, frame: Frame) -> Frame:
        return tuple(e for e in frame if not (self.cls.matches(e) and self._fires(e, action)))

    def describe(self) -> str:
        return f"vanish[{self.cls}] on {self.trigger}"


class OnOverlap(Rule):
    """When an ``actor`` entity ends the transition overlapping (or touching, pad=1) a ``target`` entity of the
    before-frame, the effect happens: 'target_vanishes', 'actor_vanishes' or ('target_recolor', c)."""
    kind = "overlap"

    def __init__(self, actor: Cls, target: Cls, effect: Any, pad: int = 0):
        self.actor, self.target, self.effect, self.pad = actor, target, effect, int(pad)
        self.cls = target

    def _actor_positions(self, tr: Transition) -> list[Ent]:
        out = []
        for a in self.actor.select(tr.before):
            after = tr._a.get(a.id)
            out.append(after if after is not None else a)
        return out

    def _hits(self, actors: list[Ent], targets: list[Ent]) -> tuple[set[int], set[int]]:
        hit_t, hit_a = set(), set()
        for t in targets:
            for a in actors:
                if a.id != t.id and a.overlaps(t, self.pad):
                    hit_t.add(t.id)
                    hit_a.add(a.id)
        return hit_t, hit_a

    def claims(self, tr: Transition) -> dict[int, Claim]:
        actors = self._actor_positions(tr)
        targets = self.target.select(tr.before)
        hit_t, hit_a = self._hits(actors, targets)
        out: dict[int, Claim] = {}
        if self.effect == "target_vanishes":
            for t in targets:
                out[t.id] = Claim(gone=t.id in hit_t)
        elif self.effect == "actor_vanishes":
            for a in actors:
                out[a.id] = Claim(gone=a.id in hit_a)
        elif isinstance(self.effect, tuple) and self.effect[0] == "target_recolor":
            c = int(self.effect[1])
            for t in targets:
                out[t.id] = Claim(recolor=(t.color, c) if (t.id in hit_t and t.color != c) else None) if t.id in hit_t else Claim(recolor=None)
        return out

    def apply(self, before: Frame, action: Any, frame: Frame) -> Frame:
        actors = self.actor.select(frame)
        targets = self.target.select(before)
        hit_t, hit_a = self._hits(actors, targets)
        if self.effect == "target_vanishes":
            return tuple(e for e in frame if e.id not in hit_t)
        if self.effect == "actor_vanishes":
            return tuple(e for e in frame if e.id not in hit_a)
        if isinstance(self.effect, tuple) and self.effect[0] == "target_recolor":
            c = int(self.effect[1])
            return tuple(e.recolored(c) if e.id in hit_t else e for e in frame)
        return frame

    def describe(self) -> str:
        eff = self.effect if isinstance(self.effect, str) else f"target recolours to {self.effect[1]}"
        return f"overlap[{self.actor} {'touches' if self.pad else 'covers'} {self.target}] -> {eff}"


class Recolor(Rule):
    """Entities of ``cls`` with colour c0 become c1 on the trigger (toggle: and back)."""
    kind = "recolor"

    def __init__(self, cls: Cls, trigger: str, c0: int, c1: int, toggle: bool = False):
        self.cls, self.trigger, self.c0, self.c1, self.toggle = cls, trigger, int(c0), int(c1), toggle

    def _fires(self, e: Ent, action: Any) -> bool:
        if self.trigger == "any":
            return True
        if self.trigger == "CLICK@self":
            xy = click_xy(action)
            return xy is not None and e.contains(*xy)
        return action_kind(action) == self.trigger

    def _next(self, e: Ent, action: Any) -> Optional[int]:
        if not self._fires(e, action):
            return None
        if e.color == self.c0:
            return self.c1
        if self.toggle and e.color == self.c1:
            return self.c0
        return None

    def claims(self, tr: Transition) -> dict[int, Claim]:
        out = {}
        for e in self.cls.select(tr.before):
            nxt = self._next(e, tr.action)
            out[e.id] = Claim(recolor=(e.color, nxt) if nxt is not None else None) if nxt is not None else Claim(recolor=None)
        return out

    def apply(self, before: Frame, action: Any, frame: Frame) -> Frame:
        out = []
        for e in frame:
            nxt = self._next(e, action) if self.cls.matches(e) else None
            out.append(e.recolored(nxt) if nxt is not None else e)
        return tuple(out)

    def describe(self) -> str:
        arrow = "<->" if self.toggle else "->"
        return f"recolor[{self.cls}] {self.c0}{arrow}{self.c1} on {self.trigger}"


class CounterRule(Rule):
    """A HUD-like entity of ``cls`` shrinks by ``per`` cells on every action (or every key action)."""
    kind = "counter"

    def __init__(self, cls: Cls, per: int, keys_only: bool = False):
        self.cls, self.per, self.keys_only = cls, int(per), keys_only

    def _active(self, action: Any) -> bool:
        return not self.keys_only or action_kind(action) in KEYS

    def claims(self, tr: Transition) -> dict[int, Claim]:
        out = {}
        for e in self.cls.select(tr.before):
            if self._active(tr.action) and e.size > self.per:
                out[e.id] = Claim(resized=(e.size, e.size - self.per))
            else:
                out[e.id] = Claim(resized=None)
        return out

    def apply(self, before: Frame, action: Any, frame: Frame) -> Frame:
        if not self._active(action):
            return frame
        return tuple(Ent(e.id, e.color, e.x0, e.y0, e.w, e.h, e.size - self.per, e.shape) if self.cls.matches(e) and e.size > self.per else e
                     for e in frame)

    def describe(self) -> str:
        return f"counter[{self.cls}] -{self.per} cells per {'key' if self.keys_only else 'action'}"


# ---------------------------------------------------------------------------------------------- verification
@dataclass
class Score:
    contradictions: int = 0
    support: int = 0  # non-trivial claims that matched observations
    examples: list[dict[str, Any]] = field(default_factory=list)

    def ok(self) -> bool:
        return self.contradictions == 0


def score_rule(rule: Rule, log: list[Transition], max_examples: int = 5) -> Score:
    s = Score()
    for i, tr in enumerate(log):
        for eid, cl in rule.claims(tr).items():
            o = tr.obs(eid)
            if o is None:
                continue
            if cl.contradicts(o):
                s.contradictions += 1
                if len(s.examples) < max_examples:
                    s.examples.append({"index": i, "action": tr.action, "id": eid, "claim": cl, "observed": o})
            else:
                s.support += len(cl.events() & obs_events(o))
    return s


def explain(rules: list[Rule], log: list[Transition], max_items: int = 12) -> dict[str, Any]:
    """Coverage of a rule set: contradictions, events explained, transitions fully explained, and the first
    unexplained (transition, entity, event) items: the counter-examples the model should look at."""
    total_events = explained = contradictions = full = 0
    unexplained: list[dict[str, Any]] = []
    contra: list[dict[str, Any]] = []
    for i, tr in enumerate(log):
        claimed: dict[int, set[str]] = defaultdict(set)
        bad = False
        for r in rules:
            for eid, cl in r.claims(tr).items():
                o = tr.obs(eid)
                if o is None:
                    continue
                if cl.contradicts(o):
                    bad = True
                    contradictions += 1
                    if len(contra) < max_items:
                        contra.append({"index": i, "action": tr.action, "id": eid, "rule": r.describe(), "claim": cl, "observed": o})
                else:
                    claimed[eid] |= cl.events()
        missing = False
        for e in tr.before:
            o = tr.obs(e.id)
            ev = obs_events(o) if o else set()
            total_events += len(ev)
            got = ev & claimed[e.id]
            explained += len(got)
            for kind in sorted(ev - claimed[e.id]):
                missing = True
                if len(unexplained) < max_items:
                    unexplained.append({"index": i, "action": tr.action, "id": e.id, "color": e.color, "event": kind, "observed": o})
        for e in tr.appeared():
            total_events += 1
            missing = True
            if len(unexplained) < max_items:
                unexplained.append({"index": i, "action": tr.action, "id": e.id, "color": e.color, "event": "appeared", "observed": None})
        if not bad and not missing:
            full += 1
    return {"transitions": len(log), "fully_explained": full, "events": total_events, "explained": explained,
            "coverage": (explained / total_events) if total_events else 1.0, "contradictions": contradictions,
            "unexplained": unexplained, "contradicted": contra}


# ---------------------------------------------------------------------------------------------- fitting
def _key_deltas(log: list[Transition], cls: Cls) -> dict[str, Counter]:
    out: dict[str, Counter] = defaultdict(Counter)
    for tr in log:
        k = action_kind(tr.action)
        if k not in KEYS:
            continue
        for e in cls.select(tr.before):
            o = tr.obs(e.id)
            if o and not o.gone and o.moved != (0, 0):
                out[k][o.moved] += 1
    return out


def _bump_colors(log: list[Transition], cls: Cls, keymap: dict[str, tuple[int, int]]) -> Counter:
    """Colours found in the target box on key presses where the class stayed put."""
    c: Counter = Counter()
    for tr in log:
        k = action_kind(tr.action)
        if k not in keymap:
            continue
        dx, dy = keymap[k]
        for e in cls.select(tr.before):
            o = tr.obs(e.id)
            if o and not o.gone and o.moved == (0, 0):
                for other in tr.before:
                    if other.id != e.id and other.box_overlaps(e.x0 + dx, e.y0 + dy, e.w, e.h):
                        c[other.color] += 1
    return c


def fit_move(log: list[Transition], max_rules: int = 6, allow_contradictions: bool = False) -> list[tuple[Move, Score]]:
    """Best Move rule per moving class. With allow_contradictions, the least-contradicted rule is kept even when
    no parameterisation fits (a mover that also pushes needs a Push rule to become consistent)."""
    if not log:
        return []
    out: list[tuple[Move, Score]] = []
    moved_ents = [e for tr in log if action_kind(tr.action) in KEYS for e in tr.before
                  if (o := tr.obs(e.id)) and not o.gone and o.moved != (0, 0)]
    for cls in _classes_of(moved_ents):
        deltas = _key_deltas(log, cls)
        if not deltas:
            continue
        base: dict[str, tuple[int, int]] = {k: c.most_common(1)[0][0] for k, c in deltas.items()}
        # unobserved keys: try the mirror of the opposite key, accepted only if consistent
        variants: list[dict[str, tuple[int, int]]] = [dict(base)]
        guess = dict(base)
        for k in KEYS:
            if k not in guess and OPPOSITE[k] in base:
                ox, oy = base[OPPOSITE[k]]
                guess[k] = (-ox, -oy)
        if guess != base:
            variants.insert(0, guess)
        bumps = _bump_colors(log, cls, base)
        cols = [c for c, _ in bumps.most_common(6)]
        block_options: list[Any] = ["any", None]
        for r in (1, 2, 3):
            block_options += [frozenset(s) for s in combinations(cols, r)]
        best: Optional[tuple[Move, Score]] = None
        for km in variants:
            for b in block_options:
                rule = Move(cls, km, b)
                s = score_rule(rule, log)
                if not s.ok() and not allow_contradictions:
                    continue
                # Preference on ties: 'any' (cautious: plans go around what has not been walked through), then the
                # smallest specific colour set, and 'nothing blocks' only when evidence rules out everything else.
                rank = 0 if b == "any" else (100 if b is None else len(b))
                key = (-s.contradictions, s.support, -rank, len(km))
                if best is None or key > best[2]:
                    best = (rule, s, key)  # type: ignore[assignment]
        if best is not None:
            out.append((best[0], best[1]))
    out.sort(key=lambda rs: -rs[1].support)
    return out[:max_rules]


def fit_push(log: list[Transition], moves: list[Move], max_rules: int = 4) -> list[tuple[Push, Score]]:
    out: list[tuple[Push, Score]] = []
    for mv in moves:
        # candidate pushables: classes that moved by the mover's delta in the same transition while adjacent
        cands: Counter = Counter()
        for tr in log:
            d = mv.delta(tr.action)
            if not d or d == (0, 0):
                continue
            for m in mv.cls.select(tr.before):
                for e in tr.before:
                    if e.id == m.id or mv.cls.matches(e):
                        continue
                    o = tr.obs(e.id)
                    if o and not o.gone and o.moved == d and e.box_overlaps(m.x0 + d[0], m.y0 + d[1], m.w, m.h):
                        cands[(e.color, e.shape)] += 1
        for (color, shape), _ in cands.most_common(4):
            for cls in (Cls(color=color), Cls(color=color, shape=shape)):
                for b in ("any", None):
                    rule = Push(mv, cls, b)
                    s = score_rule(rule, log)
                    if s.ok() and s.support > 0:
                        out.append((rule, s))
                        break
                else:
                    continue
                break
    out.sort(key=lambda rs: -rs[1].support)
    return out[:max_rules]


def fit_drift(log: list[Transition], max_rules: int = 4) -> list[tuple[Drift, Score]]:
    out: list[tuple[Drift, Score]] = []
    per_cls: dict[tuple, Counter] = defaultdict(Counter)
    for tr in log:
        for e in tr.before:
            o = tr.obs(e.id)
            if o and not o.gone and o.moved != (0, 0):
                per_cls[(e.color, e.shape)][o.moved] += 1
    for (color, shape), c in per_cls.items():
        (dx, dy), n = c.most_common(1)[0]
        if n < 2:
            continue
        for cls in (Cls(color=color), Cls(color=color, shape=shape)):
            for keys_only in (False, True):
                rule = Drift(cls, dx, dy, keys_only)
                s = score_rule(rule, log)
                if s.ok() and s.support > 0:
                    out.append((rule, s))
                    break
            else:
                continue
            break
    out.sort(key=lambda rs: -rs[1].support)
    return out[:max_rules]


def _triggers(log: list[Transition]) -> list[str]:
    kinds = sorted({action_kind(tr.action) for tr in log})
    return kinds + ["CLICK@self", "any"]


def fit_vanish(log: list[Transition], max_rules: int = 6) -> list[tuple[Vanish, Score]]:
    out: list[tuple[Vanish, Score]] = []
    gone_ents = [e for tr in log for e in tr.before if (o := tr.obs(e.id)) and o.gone]
    triggers = _triggers(log)
    for cls in _classes_of(gone_ents):
        best: Optional[tuple[Vanish, Score]] = None
        for t in triggers:
            rule = Vanish(cls, t)
            s = score_rule(rule, log)
            if s.ok() and s.support > 0 and (best is None or s.support > best[1].support):
                best = (rule, s)
        if best:
            out.append(best)
    out.sort(key=lambda rs: -rs[1].support)
    return out[:max_rules]


def fit_overlap(log: list[Transition], max_rules: int = 6) -> list[tuple[OnOverlap, Score]]:
    out: list[tuple[OnOverlap, Score]] = []
    movers = [e for tr in log for e in tr.before if (o := tr.obs(e.id)) and not o.gone and o.moved != (0, 0)]
    gone = [e for tr in log for e in tr.before if (o := tr.obs(e.id)) and o.gone]
    recol = [(e, o.recolor[1]) for tr in log for e in tr.before if (o := tr.obs(e.id)) and o.recolor]
    actor_classes = _classes_of(movers, with_shape=False)
    effects: list[tuple[Cls, Any]] = []
    for t in _classes_of(gone):
        effects.append((t, "target_vanishes"))
    for t in _classes_of([e for e, _ in recol]):
        for c in sorted({c for _, c in recol}):
            effects.append((t, ("target_recolor", c)))
    for a in actor_classes:
        for t, eff in effects:
            if a.color == t.color and a.shape is None:
                continue
            for pad in (0, 1):
                rule = OnOverlap(a, t, eff, pad)
                s = score_rule(rule, log)
                if s.ok() and s.support > 0:
                    out.append((rule, s))
                    break
        # actor vanishes when it meets something
        for t in _classes_of([e for tr in log for e in tr.before], with_shape=False):
            if t.color == a.color:
                continue
            for pad in (0, 1):
                rule = OnOverlap(a, t, "actor_vanishes", pad)
                s = score_rule(rule, log)
                if s.ok() and s.support > 0:
                    out.append((rule, s))
                    break
    out.sort(key=lambda rs: -rs[1].support)
    return out[:max_rules]


def fit_recolor(log: list[Transition], max_rules: int = 6) -> list[tuple[Recolor, Score]]:
    out: list[tuple[Recolor, Score]] = []
    pairs: Counter = Counter()
    ents_by_pair: dict[tuple[int, int], list[Ent]] = defaultdict(list)
    for tr in log:
        for e in tr.before:
            o = tr.obs(e.id)
            if o and o.recolor:
                pairs[o.recolor] += 1
                ents_by_pair[o.recolor].append(e)
    triggers = _triggers(log)
    for (c0, c1), _ in pairs.most_common(6):
        for cls in _classes_of(ents_by_pair[(c0, c1)]):
            cls = Cls(color=None, shape=cls.shape)  # colour is the thing that changes; class by shape (or any)
            best: Optional[tuple[Recolor, Score]] = None
            for t in triggers:
                for toggle in (True, False):
                    rule = Recolor(cls, t, c0, c1, toggle)
                    s = score_rule(rule, log)
                    if s.ok() and s.support > 0 and (best is None or s.support > best[1].support):
                        best = (rule, s)
            if best:
                out.append(best)
    # de-duplicate by description
    seen: set[str] = set()
    uniq = []
    for r, s in sorted(out, key=lambda rs: -rs[1].support):
        if r.describe() not in seen:
            seen.add(r.describe())
            uniq.append((r, s))
    return uniq[:max_rules]


def fit_counter(log: list[Transition], max_rules: int = 3) -> list[tuple[CounterRule, Score]]:
    out: list[tuple[CounterRule, Score]] = []
    per_cls: dict[tuple, Counter] = defaultdict(Counter)
    ents: dict[tuple, list[Ent]] = defaultdict(list)
    for tr in log:
        for e in tr.before:
            o = tr.obs(e.id)
            if o and o.resized and o.resized[0] > o.resized[1]:
                per_cls[(e.color,)][o.resized[0] - o.resized[1]] += 1
                ents[(e.color,)].append(e)
    for key, c in per_cls.items():
        per, n = c.most_common(1)[0]
        if n < 2:
            continue
        for keys_only in (False, True):
            rule = CounterRule(Cls(color=key[0]), per, keys_only)
            s = score_rule(rule, log)
            if s.ok() and s.support > 0:
                out.append((rule, s))
                break
    return out[:max_rules]


FITTERS: dict[str, Callable[..., list[tuple[Rule, Score]]]] = {
    "move": fit_move, "drift": fit_drift, "vanish": fit_vanish, "overlap": fit_overlap,
    "recolor": fit_recolor, "counter": fit_counter,
}


def fit(kind: Optional[str], log: list[Transition]) -> dict[str, list[tuple[Rule, Score]]]:
    """Consistent parameterisations per rule type (push needs the fitted moves, so it is derived here)."""
    kinds = [kind] if kind else list(FITTERS) + ["push"]
    out: dict[str, list[tuple[Rule, Score]]] = {}
    for k in kinds:
        if k == "push":
            movers = [r for r, _ in fit_move(log, allow_contradictions=True)]
            out[k] = fit_push(log, movers)  # type: ignore[assignment]
        elif k in FITTERS:
            out[k] = FITTERS[k](log)
        else:
            raise ValueError(f"unknown rule kind {k!r}; choose from {list(FITTERS) + ['push']}")
    return out


def auto_rules(log: list[Transition]) -> tuple[list[Rule], dict[str, Any]]:
    """Greedy rule-set selection: add consistent rules by support while they explain new events and add no
    contradictions; a push rule replaces the move it extends. Returns (rules, explain(rules))."""
    fitted = fit(None, log)
    cands: list[tuple[Rule, Score]] = []
    for lst in fitted.values():
        cands.extend(lst)
    prio = {"push": 0, "move": 1, "overlap": 2, "vanish": 3, "recolor": 4, "counter": 5, "drift": 6}  # drift last: it explains key moves only by coincidence
    cands.sort(key=lambda rs: (-rs[1].support, prio.get(rs[0].kind, 9)))
    chosen: list[Rule] = []
    best = explain(chosen, log)
    for r, _ in cands:
        trial = list(chosen)
        if isinstance(r, Push):
            trial = [x for x in trial if x is not r.move]
        trial.append(r)
        rep = explain(trial, log)
        if rep["contradictions"] == 0 and (rep["explained"] > best["explained"] or
                                            (rep["explained"] == best["explained"] and isinstance(r, Push))):
            chosen, best = trial, rep
    return chosen, best


# ---------------------------------------------------------------------------------------------- simulation
ORDER = ("push", "move", "drift", "overlap", "vanish", "recolor", "counter")


def simulate(frame: Frame, action: Any, rules: list[Rule]) -> Frame:
    before = frame
    cur = frame
    movers_done: set[Cls] = set()
    for kind in ORDER:
        for r in rules:
            if r.kind != kind:
                continue
            if isinstance(r, Move) and r.cls in movers_done:
                continue  # its push rule already moved this class
            cur = r.apply(before, action, cur)
            if isinstance(r, Push):
                movers_done.add(r.move.cls)
    return cur


def frame_key(frame: Frame) -> tuple:
    return tuple(sorted(e.key() for e in frame))


def frame_from_grid(grid: np.ndarray, bg: int, ref: Optional[Frame] = None) -> Frame:
    """Segment a grid into entities; ids are taken from ``ref`` when an entity matches one exactly (same colour,
    shape and position) or nearest same-shape one, else fresh negative ids."""
    from .perception import components  # local import keeps dsl importable without perception's deps at load

    objs = components(np.asarray(grid), ignore=(bg,))
    for o in objs:
        SHAPES.setdefault(str(o.shape_hash), o.mask)
    ents = [Ent(-(i + 1), int(o.color), int(o.x0), int(o.y0), int(o.w), int(o.h), int(o.size), str(o.shape_hash)) for i, o in enumerate(objs)]
    if not ref:
        return tuple(ents)
    used: set[int] = set()
    out = []
    for e in ents:
        best = None
        for r in ref:
            if r.id in used or r.color != e.color or r.shape != e.shape:
                continue
            d = abs(r.x0 - e.x0) + abs(r.y0 - e.y0)
            if best is None or d < best[0]:
                best = (d, r.id)
        if best is not None:
            used.add(best[1])
            e = Ent(best[1], e.color, e.x0, e.y0, e.w, e.h, e.size, e.shape)
        out.append(e)
    return tuple(out)


def render(grid: np.ndarray, before: Frame, after: Frame, shapes: dict[str, np.ndarray], bg: int) -> np.ndarray:
    """Paint the entity-level change ``before -> after`` onto ``grid``: erase entities that changed, then draw
    their new state (mask by shape hash, or a filled box when the shape is unknown)."""
    g = np.asarray(grid).copy()
    a_by_id = {e.id: e for e in after}
    changed_before = [e for e in before if a_by_id.get(e.id) is None or a_by_id[e.id].key() != e.key()]
    b_ids = {e.id for e in before}
    changed_after = [e for e in after if e.id not in b_ids or any(e.id == c.id for c in changed_before)]

    def paint(e: Ent, color: int) -> None:
        m = shapes.get(e.shape)
        if m is not None and m.shape == (e.h, e.w):
            region = g[e.y0:e.y0 + e.h, e.x0:e.x0 + e.w]
            if region.shape == m.shape:
                region[m] = color
                return
        g[max(0, e.y0):e.y0 + e.h, max(0, e.x0):e.x0 + e.w] = color

    for e in changed_before:
        paint(e, bg)
    for e in changed_after:
        paint(e, e.color)
    return g


def predictor(rules: list[Rule], shapes: dict[str, np.ndarray], bg: int, ref: Optional[Callable[[], Frame]] = None) -> Callable[[Any, Any], np.ndarray]:
    """A grid predictor ``predict(grid, action)`` for the sandbox's set_model, built from an entity rule set."""

    def predict(grid: Any, action: Any) -> np.ndarray:
        g = np.asarray(grid)
        before = frame_from_grid(g, bg, ref() if ref else None)
        after = simulate(before, action, rules)
        return render(g, before, after, shapes, bg)

    return predict


# ---------------------------------------------------------------------------------------------- planning
def plan(rules: list[Rule], start: Frame, goal: Callable[[Frame], bool], actions: list[Any], *,
         max_nodes: int = 40000, max_depth: int = 200) -> Optional[list[Any]]:
    """Breadth-first search over simulated frames; returns the shortest action list reaching ``goal``."""
    if goal(start):
        return []
    seen = {frame_key(start)}
    q: deque[tuple[Frame, int]] = deque([(start, 0)])
    prev: dict[tuple, tuple[tuple, Any]] = {}
    n = 0
    while q and n < max_nodes:
        f, depth = q.popleft()
        n += 1
        if depth >= max_depth:
            continue
        fk = frame_key(f)
        for a in actions:
            nf = simulate(f, a, rules)
            k = frame_key(nf)
            if k in seen:
                continue
            seen.add(k)
            prev[k] = (fk, a)
            if goal(nf):
                path = [a]
                cur = fk
                while cur in prev:
                    cur, act = prev[cur]
                    path.append(act)
                return path[::-1]
            q.append((nf, depth + 1))
    return None


def planning_actions(rules: list[Rule], frame: Frame) -> list[Any]:
    """The action alphabet a rule set responds to: keys in key maps, triggers, and clicks on triggerable entities."""
    acts: list[Any] = []
    for r in rules:
        if isinstance(r, Move):
            acts += [k for k in r.keymap if k not in acts]
        if isinstance(r, Push):
            acts += [k for k in r.move.keymap if k not in acts]
        trig = getattr(r, "trigger", None)
        if trig == "CLICK@self":
            for e in r.cls.select(frame):
                acts.append(("CLICK", (e.x0 + e.x1) // 2, (e.y0 + e.y1) // 2))
        elif trig and trig != "any" and trig not in acts and trig != "CLICK":
            acts.append(trig)
    return acts


# ---------------------------------------------------------------------------------------------- goals
def goal_predicates(frames_by_level: list[tuple[list[Frame], bool]]) -> list[dict[str, Any]]:
    """Goal candidates: predicates true at the final frame of every completed level and false at every earlier
    frame of those levels. Levels are (frames, won); for a won level the last frame is the simulated winning frame."""
    won = [fr for fr, w in frames_by_level if w and len(fr) >= 2]
    if not won:
        return []
    colors = sorted({e.color for fr in won for e in fr[-1]} | {e.color for fr in won for e in fr[0]})
    preds: list[tuple[str, Callable[[Frame], bool]]] = []
    for c in colors:
        preds.append((f"none_left(colour {c})", lambda f, c=c: not any(e.color == c for e in f)))
        for n in range(1, 5):
            preds.append((f"count(colour {c}) == {n}", lambda f, c=c, n=n: sum(1 for e in f if e.color == c) == n))
        preds.append((f"aligned(colour {c})", lambda f, c=c: _aligned([e for e in f if e.color == c])))
        preds.append((f"all_same_colour_as({c})", lambda f, c=c: len({e.color for e in f}) == 1 and all(e.color == c for e in f)))
    for a in colors:
        for b in colors:
            if a != b:
                preds.append((f"overlap(colour {a}, colour {b})",
                              lambda f, a=a, b=b: any(x.overlaps(y) for x in f if x.color == a for y in f if y.color == b)))
                preds.append((f"touch(colour {a}, colour {b})",
                              lambda f, a=a, b=b: any(x.overlaps(y, 1) for x in f if x.color == a for y in f if y.color == b)))
    out = []
    for name, p in preds:
        ok = True
        for fr in won:
            if not p(fr[-1]) or any(p(f) for f in fr[:-1]):
                ok = False
                break
        if ok:
            out.append({"goal": name, "levels": len(won), "predicate": p})
    return out


def _aligned(ents: list[Ent]) -> bool:
    if len(ents) < 2:
        return False
    return len({e.x0 for e in ents}) == 1 or len({e.y0 for e in ents}) == 1
