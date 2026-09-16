"""Rule-driven agent without a language model (plan-100 components A-C end to end, code only).

Per level: a short fixed probe (each key twice, ACT once, one click per entity class), then fit rules with
``arc3.dsl.auto_rules``, pick a goal (win conditions consistent with completed levels first, then structural
hints: a unique-colour entity, an avatar-sized slot, collectible sets), plan with BFS in the rule simulation
and execute the plan one action at a time, verifying every step against the simulation; a mismatch refits and
replans. When no goal yields a plan it probes further (untested classes, then random legal actions).

It is a measured fallback for the model-driven agent and an end-to-end test of the rule library on real
games; it is not expected to solve much on its own (goal inference without a model is weak).
"""
from __future__ import annotations

import random
from collections import deque
from typing import Any, Callable, Optional

import numpy as np

from .. import dsl
from ..entities import Tracker
from ..env import Action, Frame, GameState
from . import register
from .base import Agent, AgentContext

KEY_IDS = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4}
ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "ACT", 6: "CLICK", 7: "UNDO", 0: "RESET"}


def _label(a: Action) -> Any:
    if a.action.value == 6:
        return ("CLICK", int(a.x or 0), int(a.y or 0))
    return ACTION_NAMES.get(a.action.value, a.action.name)


def _to_action(label: Any) -> Action:
    if isinstance(label, (tuple, list)):
        return Action.click(int(label[1]), int(label[2]))
    if label == "ACT":
        return Action.simple(5)
    if label == "UNDO":
        return Action.simple(7)
    return Action.simple(KEY_IDS[label])


@register("rules")
class RulesAgent(Agent):
    def __init__(self, ctx: AgentContext):
        super().__init__(ctx)
        cfg = ctx.config
        self.rng = random.Random(ctx.seed * 7919 + (hash(ctx.game_id) & 0xFFFF))
        self.max_probe_clicks = int(cfg.get("max_probe_clicks", 10))
        self.max_plan_len = int(cfg.get("max_plan_len", 80))
        self.max_level_actions = int(cfg.get("max_level_actions", 300))
        self.tracker = Tracker()
        self.level = -1
        self.level_actions = 0
        self.probes: deque[Any] = deque()
        self.plan: deque[Any] = deque()
        self.plan_goal: Optional[str] = None
        self.rules: list[dsl.Rule] = []
        self.plan_rules: list[dsl.Rule] = []  # the rules the current plan was made with (exact or optimistic)
        self.archive: list[tuple[list[dsl.Frame], bool]] = []  # (frames incl. simulated final, won)
        self.tried_goals: set[str] = set()
        self.failed_steps: dict[tuple, int] = {}
        self.clicked_classes: set[tuple] = set()
        self.pending: Optional[Any] = None
        self.stats_ = {"plans": 0, "plan_actions": 0, "mismatches": 0, "probes": 0, "random": 0, "levels_won": 0, "goals_tried": 0}

    # ------------------------------------------------------------------ level bookkeeping
    def _new_level(self, frame: Frame) -> None:
        self.level = frame.levels_completed
        self.level_actions = 0
        self.tracker = Tracker()
        self.tracker.reset(frame.grid)
        self.plan.clear()
        self.plan_goal = None
        self.tried_goals.clear()
        self.failed_steps.clear()
        self.clicked_classes.clear()
        self.probes = deque(self._initial_probes(frame))

    def _initial_probes(self, frame: Frame) -> list[Any]:
        avail = [ACTION_NAMES[a] for a in (frame.available_actions or []) if a in ACTION_NAMES]
        out: list[Any] = []
        for k in ("UP", "DOWN", "LEFT", "RIGHT"):
            if k in avail:
                out += [k, k]
        if "ACT" in avail:
            out.append("ACT")
        if "CLICK" in avail:
            out += self._click_probes(self.max_probe_clicks)
        return out

    def _click_probes(self, n: int) -> list[Any]:
        frame = self.tracker.compound_frames()[-1] if self.tracker.frames else ()
        roles = self.tracker.roles()
        out = []
        seen = set(self.clicked_classes)
        for e in sorted(frame, key=lambda e: -e.size):
            key = (e.color, e.shape)
            if key in seen or roles.get(e.id) == "hud":
                continue
            seen.add(key)
            out.append(("CLICK", (e.x0 + e.x1) // 2, (e.y0 + e.y1) // 2))
            if len(out) >= n:
                break
        return out

    # ------------------------------------------------------------------ goals
    def _avatar_compound(self, frame: dsl.Frame) -> Optional[dsl.Ent]:
        av = self.tracker.avatar()
        if not av or not av.get("entity"):
            return None
        aid = int(av["id"])
        for e in frame:
            if e.id == aid:
                return e  # the compound keeps the mover's id
        raw = self.tracker.get(aid)
        if raw is None:
            return None
        cx, cy = raw.center
        inside = [e for e in frame if e.contains(cx, cy)]
        return min(inside, key=lambda e: e.size) if inside else None

    def _goal_list(self, frame: dsl.Frame) -> list[tuple[str, Callable[[dsl.Frame], bool]]]:
        goals: list[tuple[str, Callable[[dsl.Frame], bool]]] = []
        for g in dsl.goal_predicates(self.archive):
            goals.append((g["goal"], g["predicate"]))
        av = self._avatar_compound(frame)
        roles = self.tracker.roles()
        hud = set(self.tracker.hud_ids())
        ents = [e for e in frame if e.id not in hud and (av is None or e.id != av.id)]
        by_color: dict[int, list[dsl.Ent]] = {}
        for e in ents:
            by_color.setdefault(e.color, []).append(e)
        if av is not None:
            aid = av.id
            for c, lst in sorted(by_color.items(), key=lambda kv: len(kv[1])):
                if len(lst) == 1 and lst[0].size <= 400:
                    t = lst[0]
                    goals.append((f"reach #{t.id} (unique colour {c})", lambda f, aid=aid, t=t: any(e.id == aid and e.overlaps(t, 1) for e in f)))
            for e in ents:
                if (e.w == av.w and e.h == av.h) and e.color != av.color:
                    goals.append((f"reach #{e.id} (avatar-sized)", lambda f, aid=aid, t=e: any(x.id == aid and x.overlaps(t, 1) for x in f)))
            for e in sorted((x for x in ents if roles.get(x.id) in ("dynamic", "unknown") and x.size <= 400), key=lambda x: -x.size)[:2]:
                goals.append((f"reach #{e.id} (dynamic)", lambda f, aid=aid, t=e: any(x.id == aid and x.overlaps(t, 1) for x in f)))
        for c, lst in by_color.items():
            if 2 <= len(lst) <= 12 and all(x.size <= 64 for x in lst):
                goals.append((f"none_left(colour {c})", lambda f, c=c: not any(e.color == c for e in f)))
        return goals

    # ------------------------------------------------------------------ planning
    def _fit_and_plan(self) -> bool:
        if not self.tracker.frames:
            return False
        log = dsl.make_log(self.tracker.compound_frames(), self.tracker.actions, self.tracker.unders, self.tracker.bg)
        try:
            self.rules, _rep = dsl.auto_rules(log, ignore_ids=self.tracker.hud_ids())
        except Exception as e:  # noqa: BLE001
            self.log.warning("auto_rules failed: %s", e)
            self.rules = []
        if not self.rules:
            return False
        frame = self.tracker.compound_frames()[-1]
        dsl.set_terrain(self.rules, self.tracker.under, self.tracker.bg)
        actions = dsl.planning_actions(self.rules, frame)
        if not actions:
            return False
        goals = [(n, p) for n, p in self._goal_list(frame) if n not in self.tried_goals]
        # exact rules first; then optimistic rules (unknown colours assumed passable): the plan is an experiment
        for rules, tag in ((self.rules, ""), (dsl.optimistic(self.rules), " [optimistic]")):
            for name, pred in goals:
                key = name + tag
                if key in self.tried_goals:
                    continue
                self.stats_["goals_tried"] += 1
                self.tried_goals.add(key)
                try:
                    path = dsl.plan(rules, frame, pred, actions, max_nodes=20000, max_depth=self.max_plan_len)
                except Exception as e:  # noqa: BLE001
                    self.log.warning("plan failed: %s", e)
                    path = None
                if path:
                    self.plan = deque(path[: self.max_plan_len])
                    self.plan_goal = key
                    self.plan_rules = rules
                    self.stats_["plans"] += 1
                    if tag:
                        self.stats_["optimistic_plans"] = self.stats_.get("optimistic_plans", 0) + 1
                    return True
        return False

    # ------------------------------------------------------------------ agent API
    def act(self, frame: Frame) -> Action:
        if frame.state is GameState.NOT_PLAYED:
            self.pending = "RESET"
            return Action.reset()
        if frame.levels_completed != self.level:
            self._new_level(frame)
        if frame.state is GameState.GAME_OVER:
            self.pending = "RESET"
            self.plan.clear()
            self.probes.clear()
            return Action.reset()
        self.level_actions += 1
        if self.plan:
            label = self.plan.popleft()
            self.stats_["plan_actions"] += 1
        elif self.probes:
            label = self.probes.popleft()
            self.stats_["probes"] += 1
        elif self._fit_and_plan() and self.plan:
            label = self.plan.popleft()
            self.stats_["plan_actions"] += 1
        else:
            more = self._click_probes(4) if 6 in (frame.available_actions or []) else []
            if more:
                self.probes.extend(more)
                label = self.probes.popleft()
                self.stats_["probes"] += 1
            else:
                avail = [a for a in (frame.available_actions or [1, 2, 3, 4]) if a in (1, 2, 3, 4, 5, 6)]
                a = self.rng.choice(avail or [1])
                label = ("CLICK", self.rng.randint(0, 63), self.rng.randint(0, 63)) if a == 6 else ACTION_NAMES[a]
                self.stats_["random"] += 1
                if self.level_actions % 8 == 0:
                    self.tried_goals.clear()  # new evidence: allow the goals again
        self.pending = label
        return _to_action(label)

    def observe(self, action: Action, before: Frame, after: Frame) -> None:
        label = self.pending if self.pending is not None else _label(action)
        self.pending = None
        if label == "RESET":
            if after.levels_completed == self.level:
                self.tracker.reset(after.grid)  # the level restarted: same rules, fresh log
                self.plan.clear()
                self.probes = deque(self._initial_probes(after))
            return
        if after.levels_completed > self.level or after.state is GameState.WIN:
            # level completed: archive with the simulated winning frame for goal inference
            frames = self.tracker.compound_frames()
            final = None
            try:
                dsl.set_terrain(self.rules, self.tracker.under, self.tracker.bg)
                final = dsl.simulate(frames[-1], label, self.rules) if self.rules else None
            except Exception:  # noqa: BLE001
                final = None
            if final is not None:
                self.archive.append((list(frames) + [final], True))
            self.stats_["levels_won"] += 1
            return
        if isinstance(label, tuple):
            for e in self.tracker.compound_frames()[-1]:
                if e.contains(label[1], label[2]):
                    self.clicked_classes.add((e.color, e.shape))
        before_frame = self.tracker.compound_frames()[-1] if self.tracker.frames else None
        self.tracker.update(after.grid, label)
        if self.plan and self.plan_rules and before_frame is not None:
            try:
                dsl.set_terrain(self.plan_rules, self.tracker.under, self.tracker.bg)
                pred = dsl.simulate(before_frame, label, self.plan_rules)
                got = self.tracker.compound_frames()[-1]
                tr = dsl.Transition(before_frame, label, got, self.tracker.under, self.tracker.bg)
                claimed: set[int] = set()
                for r in self.plan_rules:
                    claimed |= set(r.claims(tr))
                # verify only what the rules speak about (position, colour, presence); counters and unmodelled
                # entities changing is not a plan failure
                p_by = {e.id: (e.x0, e.y0, e.color) for e in pred if e.id in claimed}
                g_by = {e.id: (e.x0, e.y0, e.color) for e in got if e.id in claimed}
                if p_by != g_by:
                    self.stats_["mismatches"] += 1
                    self.plan.clear()  # refit and replan on the next act(): the mismatch is new evidence
                    sig = (self.plan_goal, label, dsl.frame_key(before_frame))
                    self.failed_steps[sig] = self.failed_steps.get(sig, 0) + 1
                    banned = {g for (g, _, _), n in self.failed_steps.items() if n >= 2 and g}
                    self.tried_goals = set(banned)  # a goal whose plan failed twice at the same step stays banned
            except Exception:  # noqa: BLE001
                self.plan.clear()

    def stats(self) -> dict[str, Any]:
        return dict(self.stats_)
