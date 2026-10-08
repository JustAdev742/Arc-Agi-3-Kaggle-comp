#!/usr/bin/env python
"""Replay check of ours-06 (OURS_EFFECT_TABLE): every line of every effect table, checked against the frames.

Replays the recorded actions of a run (benchmark.json: action ids and click coordinates) through the local engine,
builds the harness's history from the frames (initial frame, then one entry per action with its result flags), feeds
it to the patched harness's EffectLedger one action at a time, the way the harness serializes state after every
action, and checks the tables with a second implementation of the rules in the module's docstring, written from that
text: breadth-first objects on numpy arrays, no code shared with the module. Every claim of a line is recomputed:
counts, changes and their multiplicities, last steps and positions, click targets, game-over overlaps, periods, edge
counts. It also checks the "note" lines (OURS_EFFECT_TABLE=note) and times each update.

    .venv/bin/python -I scripts/effect_table_replay.py runs/RUN/kernel-output/benchmark.json [--games dc22,tu93]
        [--tree ARC3-Inference] [--every 25]

Without --tree it builds Franzen's tree with his patch, ours-01 and ours-06 (scripts/franzen_tree.py) in a temp dir.
Exit 1 if any line is wrong.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATCHES = [ROOT / "kaggle/franzen/patches/ours-sandbox-timeout-keeps-work.patch",
           ROOT / "kaggle/franzen/patches/ours-06-effect-table.patch"]
COLORS = "WwgGcBMPRbSYOrNp"
AREA, MAX_DISTINCT, SHOWN, MAX_QUIET = 512, 12, 4, 24
NAMES = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT", "ACTION5": "SPACE",
         "ACTION7": "UNDO", "RESET": "RESET"}


# --- the history, from the local engine --------------------------------------------------------------------------


def replay(game: dict, env_dir: Path) -> list[dict]:
    """The harness's history for one recorded game: entry 0 is the start frame, then one entry per action."""
    sys.path.insert(0, str(ROOT))
    logging.disable(logging.INFO)
    from arcengine import GameAction

    from arc3.env import Action, LocalEnv, make_arcade

    env = LocalEnv(make_arcade(env_dir), game["game_id"], seed=0)
    frame = env.frame
    hist = [{"action": "", "frame": {"grid": frame.grid.tolist(), "step": 0, "level": frame.levels_completed + 1},
             "result": {}}]
    for i, rec in enumerate(game["history"], 1):
        a = rec["action"]
        if a["id"] == "ACTION6":
            action, display = Action.click(a["data"]["x"], a["data"]["y"]), \
                f"MOUSE(row={int(a['data']['y'])}, col={int(a['data']['x'])})"
        else:
            action, display = Action(GameAction[a["id"]]), NAMES[a["id"]]
        before = frame
        frame = env.step(action)
        won = frame.state.name == "WIN"
        level = before.levels_completed + 1 if won else frame.levels_completed + 1
        hist.append({"action": display, "frame": {"grid": frame.grid.tolist(), "step": i, "level": level},
                     "result": {"game_over": frame.state.name == "GAME_OVER", "run_complete": won,
                                "level_completed": frame.levels_completed > before.levels_completed and not won,
                                "automatic": display == "RESET"}})
    return hist


# --- the second implementation -------------------------------------------------------------------------------------


class Obj:
    def __init__(self, color: int, cells: frozenset):
        self.color, self.cells, self.px = color, cells, len(cells)
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        self.tl = (min(rows), min(cols))
        self.dims = (max(rows) - min(rows) + 1, max(cols) - min(cols) + 1)
        self.shape = tuple(sorted((r - self.tl[0], c - self.tl[1]) for r, c in cells))


def xy(p) -> str:
    return f"({p[0]},{p[1]})"


def board_objects(grid: np.ndarray, b: int) -> tuple[list[Obj], dict]:
    """4-connected one-colour objects of the board (the grid without the b-cell edge band), and cell -> object."""
    h, w = grid.shape
    owner: dict = {}
    objs: list[Obj] = []
    for r0 in range(b, h - b):
        for c0 in range(b, w - b):
            if (r0, c0) in owner:
                continue
            color = int(grid[r0, c0])
            cells, todo = {(r0, c0)}, deque([(r0, c0)])
            while todo:
                r, c = todo.popleft()
                for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if b <= rr < h - b and b <= cc < w - b and (rr, cc) not in cells and grid[rr, cc] == color:
                        cells.add((rr, cc))
                        todo.append((rr, cc))
            o = Obj(color, frozenset(cells))
            objs.append(o)
            for cell in cells:
                owner[cell] = o
    return objs, owner


def turned(shape: tuple) -> tuple:
    h = max(r for r, _ in shape) + 1
    cells = [(c, h - 1 - r) for r, c in shape]
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def orbit(shape: tuple) -> list:
    out = [shape]
    for _ in range(3):
        out.append(turned(out[-1]))
    return out


def text(sig: tuple, mult: int = 1) -> str:
    times = f"{mult}x " if mult > 1 else ""
    k = sig[0]
    if k == "moved":
        return f"moved {times}{COLORS[sig[1]]} {sig[2]}px ({sig[3]:+d},{sig[4]:+d})"
    if k == "rotated":
        return f"rotated {times}{COLORS[sig[1]]} {sig[2]}px {sig[3]} ({sig[4]:+d},{sig[5]:+d})"
    if k == "recolored":
        return f"recolored {times}{COLORS[sig[1]]}->{COLORS[sig[2]]} {sig[3]}px"
    if k in ("appeared", "disappeared"):
        return f"{k} {times}{COLORS[sig[1]]} {sig[2]}px"
    if k == "reshaped":
        return f"reshaped {times}{COLORS[sig[1]]} {'+'.join(map(str, sig[2]))}px->{'+'.join(map(str, sig[3]))}px"
    if k == "ambiguous":
        return f"ambiguous {times}{COLORS[sig[1]]} {sig[2]}px objects {sig[3]}->{sig[4]}"
    if k == "cells":
        return f"cells {COLORS[sig[1]]}->{COLORS[sig[2]]}"
    return "many changes"


class Transition:
    """One action's facts by the second implementation."""

    def __init__(self, A: np.ndarray, B: np.ndarray, b: int, objs_a, objs_b):
        self.A, self.B, self.b = A, B, b
        h, w = A.shape
        diff = A != B
        inner = np.zeros_like(diff)
        inner[b:h - b, b:w - b] = True
        self.edge = bool((diff & ~inner).any())
        self.changed = {(int(r), int(c)) for r, c in zip(*np.nonzero(diff & inner))}
        self.items = []  # (sig, where, before objs, after objs)
        if self.changed:
            self._pair(objs_a, objs_b)

    def _pair(self, objs_a, objs_b) -> None:
        A, B = self.A, self.B
        in_b = {(o.color, o.cells) for o in objs_b}
        in_a = {(o.color, o.cells) for o in objs_a}
        only_a = [o for o in objs_a if (o.color, o.cells) not in in_b]
        only_b = [o for o in objs_b if (o.color, o.cells) not in in_a]
        by_cells = {o.cells: o for o in only_b}
        items, rest_a, used = [], [], set()
        for x in only_a:
            y = by_cells.get(x.cells)
            if y is not None:
                items.append((("recolored", x.color, y.color, x.px), f"at {xy(x.tl)}", [x], [y]))
                used.add(id(y))
            else:
                rest_a.append(x)
        rest_b = [y for y in only_b if id(y) not in used]
        groups: dict = {}
        for side, objs in ((0, rest_a), (1, rest_b)):
            for o in objs:
                if o.px < AREA:
                    groups.setdefault((o.color, min(orbit(o.shape))), ([], []))[side].append(o)
        paired = set()
        for (color, _), (xs, ys) in groups.items():
            if not xs or not ys:
                continue
            paired |= {id(o) for o in xs + ys}
            if len(xs) == len(ys) == 1:
                x, y = xs[0], ys[0]
                k = orbit(x.shape).index(y.shape)
                d = (y.tl[0] - x.tl[0], y.tl[1] - x.tl[1])
                sig = ("moved", color, x.px, *d) if k == 0 else ("rotated", color, x.px, 90 * k, *d)
                items.append((sig, f"{xy(x.tl)}->{xy(y.tl)}", [x], [y]))
                continue
            if len(xs) == len(ys) and len({o.shape for o in xs + ys}) == 1:
                pa, pb = sorted(o.tl for o in xs), sorted(o.tl for o in ys)
                d = (pb[0][0] - pa[0][0], pb[0][1] - pa[0][1])
                if sorted((r + d[0], c + d[1]) for r, c in pa) == pb:
                    for i, x in enumerate(xs):
                        items.append((("moved", color, x.px, *d), "", [x], ys if i == len(xs) - 1 else []))
                    continue
            where = ""
            if len(xs) + len(ys) <= 6:
                where = " ".join(xy(p) for p in sorted(o.tl for o in xs)) + " -> " + " ".join(
                    xy(p) for p in sorted(o.tl for o in ys))
            items.append((("ambiguous", color, xs[0].px, len(xs), len(ys)), where, xs, ys))
        rest_a = [o for o in rest_a if id(o) not in paired]
        rest_b = [o for o in rest_b if id(o) not in paired]
        # same-colour clusters with shared cells, built as connected components of the "shares a cell" graph
        nodes = rest_a + rest_b
        adj = {id(o): set() for o in nodes}
        for x in rest_a:
            for y in rest_b:
                if x.color == y.color and x.cells & y.cells:
                    adj[id(x)].add(id(y))
                    adj[id(y)].add(id(x))
        seen, clusters = set(), []
        for x in rest_a:
            if id(x) in seen or not adj[id(x)]:
                continue
            comp, todo = set(), [id(x)]
            while todo:
                k = todo.pop()
                if k not in comp:
                    comp.add(k)
                    todo.extend(adj[k])
            seen |= comp
            clusters.append([o for o in nodes if id(o) in comp])
        for x in rest_a:
            if id(x) not in seen and x.px < AREA and all(B[r, c] != x.color for r, c in x.cells):
                items.append((("disappeared", x.color, x.px), f"at {xy(x.tl)}", [x], []))
        for y in rest_b:
            if id(y) not in seen and y.px < AREA and all(A[r, c] != y.color for r, c in y.cells):
                items.append((("appeared", y.color, y.px), f"at {xy(y.tl)}", [], [y]))
        covered = set()
        for _, _, before, after in items:
            for o in before + after:
                covered |= o.cells
        extra = []
        for comp in clusters:
            if any(o.px >= AREA for o in comp):
                continue
            ca, cb = [o for o in comp if o in rest_a], [o for o in comp if o in rest_b]
            cells = set().union(*(o.cells for o in comp)) & self.changed
            if cells - covered:
                sig = ("reshaped", ca[0].color, tuple(sorted(o.px for o in ca)), tuple(sorted(o.px for o in cb)))
                extra.append((sig, f"at {xy(min(o.tl for o in ca))}", ca, cb, cells))
        for sig, where, ca, cb, cells in extra:
            items.append((sig, where, ca, cb))
            covered |= cells
        left = Counter((int(A[r, c]), int(B[r, c])) for r, c in self.changed - covered)
        for (va, vb), n in left.items():
            items.append((("cells", va, vb), f"{n} cells", [], []))
        if len({it[0] for it in items}) > MAX_DISTINCT:
            kinds = Counter(it[0][0] for it in items if it[0][0] != "cells")
            summary = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0])))
            items = [(("many",), f"{len(self.changed)} cells" + (f": {summary}" if summary else ""), [], [])]
        self.items = items

    def counted(self) -> dict:
        """sig -> (multiplicity, where); where only for a sig that occurs once."""
        mult = Counter(it[0] for it in self.items)
        return {sig: (n, next(it[1] for it in self.items if it[0] == sig) if n == 1 else "") for sig, n in mult.items()}

    def death_facts(self) -> Counter:
        out = Counter()
        mult = Counter(it[0] for it in self.items)
        for sig, where, before, after in self.items:
            if sig[0] in ("moved", "rotated", "appeared") and len(after) == 1 and len(before) <= 1:
                o = after[0]
                under = Counter(int(self.A[r, c]) for r, c in o.cells if self.A[r, c] != o.color)
                out[f"{text(sig)} {where} onto {colours(under) or 'its own colour'}"] += 1
            elif sig[0] == "disappeared":
                o = before[0]
                out[f"{text(sig)} {where}, its cells now {colours(Counter(int(self.B[r, c]) for r, c in o.cells))}"] += 1
            else:
                out[text(sig, mult[sig]) + (f" ({where})" if sig[0] in ("cells", "many") else "")] = 1
        return out


def colours(counter: Counter) -> str:
    return " ".join(f"{COLORS[v]}:{n}" for v, n in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


class Facts:
    """Per-history-index facts of a game, computed lazily and cached."""

    def __init__(self, hist: list, b: int):
        self.hist, self.b = hist, b
        self.grids = [np.array(e["frame"]["grid"], dtype=np.int16) for e in hist]
        self._objs: dict = {}
        self._tr: dict = {}

    def objs(self, i: int):
        if i not in self._objs:
            self._objs[i] = board_objects(self.grids[i], self.b)
        return self._objs[i]

    def items(self, i: int) -> list:
        """The changes of action i; none for an action that completed a level or won the game."""
        return self.transition(i).items if self.outcome(i) in ("", "game over") else []

    def counted(self, i: int) -> dict:
        return self.transition(i).counted() if self.outcome(i) in ("", "game over") else {}

    def transition(self, i: int) -> Transition:
        if i not in self._tr:
            self._tr[i] = Transition(self.grids[i - 1], self.grids[i], self.b, self.objs(i - 1)[0], self.objs(i)[0])
        return self._tr[i]

    def outcome(self, i: int) -> str:
        res, e0, e1 = self.hist[i]["result"], self.hist[i - 1]["frame"], self.hist[i]["frame"]
        if res.get("game_over"):
            return "game over"
        if res.get("run_complete"):
            return "game won"
        if res.get("level_completed") or e0["level"] != e1["level"]:
            return "level completed"
        return ""

    def target(self, i: int) -> str:
        m = re.match(r"MOUSE\(row=(-?\d+), col=(-?\d+)\)", self.hist[i]["action"])
        r, c = int(m.group(1)), int(m.group(2))
        g = self.grids[i - 1]
        if not (0 <= r < g.shape[0] and 0 <= c < g.shape[1]):
            return "a cell off the frame"
        o = self.objs(i - 1)[1].get((r, c))
        if o is None:
            return "the edge band"
        return f"{COLORS[o.color]} area {o.px}px" if o.px >= AREA else f"{COLORS[o.color]} {o.px}px at {xy(o.tl)}"


# --- checking a table ----------------------------------------------------------------------------------------------


def uses_of(facts: Facts, upto: int, level: int) -> list:
    """History indices of the actions (not RESET) taken on `level` among entries 1..upto."""
    hist = facts.hist
    return [i for i in range(1, upto + 1) if hist[i]["action"] and hist[i]["action"] != "RESET"
            and hist[i - 1]["frame"]["level"] == level]


def segments_of(facts: Facts, upto: int, level: int) -> list:
    """Attempts on `level` as lists of history indices of their frames (start, then each frame after an action
    without an outcome)."""
    segs, cur = [], None
    hist = facts.hist
    for i in range(0, upto + 1):
        e = hist[i]
        start = i == 0 or not e["action"] or e["action"] == "RESET" or facts.outcome(i) in ("level completed",
                                                                                            "game won")
        if start:
            cur = [i] if e["frame"]["level"] == level else None
            if cur is not None:
                segs.append(cur)
        elif facts.outcome(i) == "game over":
            cur = None
        elif cur is not None:
            cur.append(i)
    return segs


def check_table(facts: Facts, upto: int, level: int, table: str, errors: list, where: str) -> int:
    """Check every line of one level's table; returns the number of lines checked."""
    def bad(msg: str) -> None:
        errors.append(f"{where} L{level}: {msg}")

    lines = table.split("\n")
    idx = uses_of(facts, upto, level)
    if not idx:
        if lines != [f"Level {level}: no actions yet."]:
            bad(f"empty level rendered as {table[:80]!r}")
        return 1
    hist = facts.hist
    deaths = [i for i in idx if facts.outcome(i) == "game over"]
    resets = [i for i in range(1, upto + 1) if hist[i]["action"] == "RESET" and hist[i - 1]["frame"]["level"] == level]
    auto = sum(1 for i in resets if hist[i]["result"].get("automatic"))
    extra = ([f"{len(deaths)} game over" + ("s" if len(deaths) > 1 else "")] if deaths else []) + (
        [f"{len(resets) - auto} RESET"] if len(resets) > auto else []) + (
        [f"{auto} automatic RESET not counted"] if auto else [])
    head = f"Level {level}: {len(idx)} actions" + (f" ({', '.join(extra)})" if extra else "")
    if not lines[0].startswith(head + ". Board = frame without the"):
        bad(f"header {lines[0][:90]!r} expected {head!r}")
    by_key: dict = {}
    for i in idx:
        by_key.setdefault(hist[i]["action"], []).append(i)
    quiet_keys = {k for k, ii in by_key.items() if k.startswith("MOUSE")
                  and all(not facts.items(i) and not facts.outcome(i) for i in ii)}
    seen_keys = set()
    checked = 1
    hidden_lines = 0
    for line in lines[1:]:
        checked += 1
        if line.startswith("Edge band changed on "):
            n = sum(1 for i in idx if facts.transition(i).edge)
            if line != f"Edge band changed on {n} of {len(idx)} actions.":
                bad(f"{line!r}: expected {n}")
        elif line.startswith("MOUSE with no change: "):
            seen_keys |= check_quiet(facts, by_key, quiet_keys, line, bad)
        elif line.startswith("Game overs: "):
            check_deaths(facts, deaths, line, bad)
        elif line.startswith("Objects whose places and turns repeated, period in actions: "):
            check_movers(facts, segments_of(facts, upto, level), line, bad)
        elif re.match(r"\+\d+ more action lines \(older MOUSE cells\) not shown$", line):
            hidden_lines = int(line[1:].split()[0])
        else:
            key = check_key_line(facts, by_key, line, bad)
            if key:
                seen_keys.add(key)
    missing = set(by_key) - seen_keys
    if len(missing) != hidden_lines:
        bad(f"{len(missing)} actions without a line, {hidden_lines} said hidden: {sorted(missing)[:5]}")
    return checked


def check_key_line(facts: Facts, by_key: dict, line: str, bad) -> str:
    m = re.match(r"(\S+(?: col=\d+\))?) x(\d+)(?: \((on|last on) (.+?)\))?: (.*)$", line)
    if not m or m.group(1) not in by_key:
        bad(f"unparsed or unknown action line {line[:100]!r}")
        return ""
    key, n, which, target, rest = m.group(1), int(m.group(2)), m.group(3), m.group(4), m.group(5)
    ii = by_key[key]
    if n != len(ii):
        bad(f"{key}: x{n}, recorded {len(ii)}")
    if key.startswith("MOUSE"):
        targets = [facts.target(i) for i in ii]
        want = ("on" if len(set(targets)) == 1 else "last on", targets[-1])
        if (which, target) != want:
            bad(f"{key}: target {(which, target)} expected {want}")
    stats: dict = {}
    for i in ii:
        for sig, (mult, where) in facts.counted(i).items():
            st = stats.setdefault((sig, mult), [0, 0, ""])
            st[0] += 1
            st[1], st[2] = facts.hist[i]["frame"]["step"], where
    expect = {}
    quiet = sum(1 for i in ii if not facts.items(i) and not facts.outcome(i))
    if quiet:
        expect["no change"] = quiet
    for outcome in ("game over", "level completed", "game won"):
        k = sum(1 for i in ii if facts.outcome(i) == outcome)
        if k:
            expect[outcome] = k
    shown = 0
    others = None
    min_shown = None
    texts = {text(sig, mult): st for (sig, mult), st in stats.items()}
    for part in rest.split("; "):
        mm = re.match(r"(no change|game over|level completed|game won) x(\d+)$", part)
        if mm:
            if expect.pop(mm.group(1), None) != int(mm.group(2)):
                bad(f"{key}: {part!r} wrong")
            continue
        mm = re.match(r"\+(\d+) other changes$", part)
        if mm:
            others = int(mm.group(1))
            continue
        mm = re.match(r"(.+) x(\d+) \[(?:last )?step (\d+)(?:: (.*))?\]$", part)
        if not mm or mm.group(1) not in texts:
            bad(f"{key}: change {part!r} not found in the frames")
            continue
        count, last, where = texts.pop(mm.group(1))
        got = (int(mm.group(2)), int(mm.group(3)), mm.group(4) or "")
        if got != (count, last, where):
            bad(f"{key}: {part!r} expected x{count} step {last} {where!r}")
        if ("[last " in part) != (count > 1):
            bad(f"{key}: 'last' wrong in {part!r}")
        shown += 1
        min_shown = count if min_shown is None else min(min_shown, count)
    if expect:
        bad(f"{key}: missing {expect}")
    if (others or 0) != len(texts) or (others is not None and shown != SHOWN):
        bad(f"{key}: +{others} other changes, {len(texts)} not shown, {shown} shown")
    if texts and min_shown is not None and max(st[0] for st in texts.values()) > min_shown:
        bad(f"{key}: a hidden change has a higher count than a shown one")
    return key


def check_quiet(facts: Facts, by_key: dict, quiet_keys: set, line: str, bad) -> set:
    m = re.match(r"MOUSE with no change: (\d+) uses at (\d+) cells \(row,col\), latest first: (.*?)"
                 r"(?: \+(\d+) more cells)?$", line)
    if not m:
        bad(f"unparsed {line[:80]!r}")
        return set()
    order = sorted(quiet_keys, key=lambda k: by_key[k][-1], reverse=True)
    want_cells = []
    for k in order[:MAX_QUIET]:
        r, c = re.match(r"MOUSE\(row=(-?\d+), col=(-?\d+)\)", k).groups()
        want_cells.append(f"({r},{c})" + (f"x{len(by_key[k])}" if len(by_key[k]) > 1 else ""))
    uses = sum(len(by_key[k]) for k in quiet_keys)
    more = len(order) - MAX_QUIET if len(order) > MAX_QUIET else None
    got = (int(m.group(1)), int(m.group(2)), m.group(3).split(" "), int(m.group(4)) if m.group(4) else None)
    if got != (uses, len(order), want_cells, more):
        bad(f"quiet MOUSE line {line[:120]!r}")
    return quiet_keys


def check_deaths(facts: Facts, deaths: list, line: str, bad) -> None:
    body = line[len("Game overs: "):]
    m = re.search(r" \(\+(\d+) earlier\)$", body)
    earlier = int(m.group(1)) if m else 0
    if m:
        body = body[:m.start()]
    parts = body.split(" | ")
    if earlier != max(0, len(deaths) - 3) or len(parts) != min(3, len(deaths)):
        bad(f"game-over count: {len(parts)} shown +{earlier}, recorded {len(deaths)}")
        return
    for part, i in zip(parts, deaths[-len(parts):]):
        e = facts.hist[i]
        tr = facts.transition(i)
        prefix = f"after {e['action']} at step {e['frame']['step']}: "
        suffix = ", edge band changed" if tr.edge else ", edge band unchanged"
        if not part.startswith(prefix) or not part.endswith(suffix):
            bad(f"game over {part[:100]!r}")
            continue
        facts_text = part[len(prefix):-len(suffix)]
        want = tr.death_facts() if tr.items else Counter({"no change": 1})
        if Counter(facts_text.split("; ")) != want:
            bad(f"game over facts {facts_text[:160]!r} expected {sorted(want)[:4]}")


def check_movers(facts: Facts, segs: list, line: str, bad) -> None:
    body = line.split(": ", 1)[1]
    for part in body.split(" | "):
        m = re.match(r"(?:(\d+)x )?(\S) (\d+)px every (\d+) \(steps (\d+)-(\d+)\)(?:, last \d+ corners: (.*))?$", part)
        if not m:
            bad(f"unparsed mover {part!r}")
            continue
        members, color, px, p = int(m.group(1) or 1), COLORS.index(m.group(2)), int(m.group(3)), int(m.group(4))
        a, z, corners = int(m.group(5)), int(m.group(6)), m.group(7)
        seg = next((s for s in segs if any(facts.hist[i]["frame"]["step"] == a for i in s)
                    and any(facts.hist[i]["frame"]["step"] == z for i in s)), None)
        if seg is None:
            bad(f"mover {part!r}: no attempt holds steps {a}-{z}")
            continue
        window = [i for i in seg if a <= facts.hist[i]["frame"]["step"] <= z]
        ok = False
        dims = {o.dims for i in window for o in facts.objs(i)[0] if o.color == color and o.px == px}
        for d in dims:
            track = []
            for i in window:
                track.append(tuple(sorted((o.tl, o.shape) for o in facts.objs(i)[0]
                                          if o.color == color and o.px == px and sorted(o.dims) == sorted(d))))
            n = len(track)
            if any(len(t) != members for t in track) or len(set(track)) < 2:
                continue
            periods = [q for q in range(2, (n - 1) // 2 + 1) if all(track[t] == track[t + q] for t in range(n - q))]
            if not periods or periods[0] != p:
                continue
            if corners is not None and corners != " ".join(xy(t[0][0]) for t in track[-p:]):
                continue
            ok = True
        if not ok:
            bad(f"mover {part!r} does not hold in the frames")


def check_note(facts: Facts, note: str, level_uses: list, bad) -> None:
    m = re.match(r"(\S+(?: col=\d+\))?) at step (\d+) \(used (\d+)x before on this level\): (.*), a kind of change "
                 r"not seen before on level (\d+)\. effects\(\) has the table\.$", note)
    if not m:
        bad(f"unparsed note {note[:100]!r}")
        return
    key, step, tries, listed = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    i = next((j for j in level_uses if facts.hist[j]["frame"]["step"] == step), None)
    if i is None or facts.hist[i]["action"] != key:
        bad(f"note {note[:80]!r}: no such action")
        return
    earlier = [j for j in level_uses if j < i]
    if sum(1 for j in earlier if facts.hist[j]["action"] == key) != tries:
        bad(f"note {note[:80]!r}: earlier uses")

    def coarse(sig):
        return sig[:3] if sig[0] == "recolored" else sig[:2]

    seen = {coarse(sig) for j in earlier for sig in facts.counted(j)}
    new = [(sig, n, w) for sig, (n, w) in facts.counted(i).items()
           if coarse(sig) not in seen and sig[0] in ("moved", "rotated", "recolored", "appeared", "disappeared")]
    texts = {text(sig, n) + (f" {w}" if w else "") for sig, n, w in new}
    listed = re.sub(r" \(\+\d+ more\)$", "", listed)
    if not new or not all(part in texts for part in listed.split("; ")):
        bad(f"note {note[:120]!r}: new changes are {sorted(texts)[:3]}")


# --- driver --------------------------------------------------------------------------------------------------------


def run(bench: Path, games: list[str], tree: Path, every: int, env_dir: Path) -> dict:
    sys.path.insert(0, str(tree))
    from inference.utils import ours_effect_table as et

    data = json.loads(bench.read_text())
    totals = Counter()
    errors: list[str] = []
    times: list[float] = []
    report = {}
    for game in data["game_runs"]:
        name = game["game_id"].split("-")[0]
        if games and name not in games:
            continue
        hist = replay(game, env_dir)
        if game.get("actions_per_level"):  # the replay must reproduce the recorded run
            per_level = Counter(e["frame"]["level"] for e in hist[:-1])  # the frame each action was taken on
            done = [per_level.get(k + 1, 0) for k in range(len(game["actions_per_level"]))]
            if done != list(game["actions_per_level"]):
                errors.append(f"{name}: replay gives actions per level {done}, recorded {game['actions_per_level']}")
        facts = Facts(hist, 4)
        ledger = et.EffectLedger(border=4)
        notes = 0
        lines = 0
        dc22 = None
        for k in range(1, len(hist) + 1):
            ledger.mark()
            t0 = time.perf_counter()
            ledger.update(hist[:k])
            payload = ledger.payload()
            times.append(time.perf_counter() - t0)
            note = ledger.surprise()
            upto = k - 1
            if note:
                notes += 1
                level = int(re.search(r"level (\d+)\. effects", note).group(1))
                check_note(facts, note, uses_of(facts, upto, level), lambda msg, name=name: errors.append(f"{name} note: {msg}"))
            level_now = hist[upto]["frame"]["level"]
            acted_on = hist[upto - 1]["frame"]["level"] if upto else level_now
            last = k == len(hist)
            if upto and (upto % every == 0 or acted_on != level_now or last):
                for lv, table in payload["tables"].items():
                    if last or int(lv) in (acted_on, level_now):
                        lines += check_table(facts, upto, int(lv), table, errors, f"{name}@{upto}")
            if name == "dc22" and upto == 10:
                dc22 = payload["tables"]["1"]
                lines += check_table(facts, upto, 1, dc22, errors, f"{name}@{upto}")
        totals["games"] += 1
        totals["actions"] += len(hist) - 1
        totals["lines"] += lines
        totals["notes"] += notes
        report[name] = {"actions": len(hist) - 1, "lines_checked": lines, "notes": notes}
        if dc22 is not None:
            report[name]["table_after_action_10"] = dc22
        print(f"{name}: {len(hist) - 1} actions, {lines} table lines checked, {notes} notes, errors so far "
              f"{len(errors)}", flush=True)
    totals["errors"] = len(errors)
    return {"totals": dict(totals), "errors": errors, "games": report,
            "update_ms": {"mean": 1000 * sum(times) / max(1, len(times)), "max": 1000 * max(times or [0]),
                          "p99": 1000 * sorted(times)[int(0.99 * (len(times) - 1))] if times else 0}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("benchmark", type=Path)
    ap.add_argument("--games", default="", help="comma-separated game id prefixes (default: all)")
    ap.add_argument("--tree", type=Path, default=None, help="an ARC3-Inference dir with the patched harness")
    ap.add_argument("--every", type=int, default=25, help="also check the current level's table every N actions")
    ap.add_argument("--env-dir", type=Path, default=ROOT / "environment_files")
    ap.add_argument("--json", type=Path, default=None, help="write the result here")
    args = ap.parse_args()
    games = [g for g in args.games.split(",") if g]
    with tempfile.TemporaryDirectory(prefix="effect-replay-") as tmp:
        tree = args.tree
        if tree is None:
            sys.path.insert(0, str(ROOT / "scripts"))
            import franzen_tree
            franzen_tree.notebook_bundle(Path(tmp) / "share", PATCHES)
            tree = Path(tmp) / "share" / "src" / "ARC3-Inference"
        result = run(args.benchmark, games, tree, args.every, args.env_dir)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1))
    for e in result["errors"][:30]:
        print("ERROR", e)
    t = result["totals"]
    u = result["update_ms"]
    print(f"{t.get('games', 0)} games, {t.get('actions', 0)} actions, {t.get('lines', 0)} table lines and "
          f"{t.get('notes', 0)} notes checked, {t['errors']} errors; update+render per action: mean {u['mean']:.2f} ms, "
          f"p99 {u['p99']:.1f} ms, max {u['max']:.1f} ms")
    sys.exit(1 if result["errors"] else 0)


if __name__ == "__main__":
    main()
