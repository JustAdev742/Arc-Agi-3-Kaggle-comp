#!/usr/bin/env python
"""Replay check of ours-10 (OURS_UNTRIED): the I1 and I2 lines over recorded runs, with the module the patch ships.

Replays the recorded actions of a run (`kernel-output/benchmark.json`: action ids, click coordinates, wall clock and
generated tokens per action) through the local engine, builds the harness's history from the frames (as
scripts/effect_table_replay.py does), and drives the patch's Tracker the way tool_agent.py does: `stamp()` when an
action() call returns, `lines()` at every user prompt. The module is loaded from the patch text (or from --tree), so
the replay runs the very code that ships.

The recorded runs keep no prompt times, so they are rebuilt from the clock and the tokens:
- an action with generated tokens starts an action() call; the zero-token actions after it belong to the same call,
  except RESETs, which the solver issues between turns (warmup and after a game over);
- a call is stamped with the time of its last action;
- prompts: one right after each call returns (the next turn's opener), and while the model thinks before the next
  call, one per YIELD_TOKENS generated (Franzen yields a turn at 2,048 tokens), the thinking placed just before that
  call at TOK_PER_S, so time the game spent parked comes before it; the game's first prompt is when its first call's
  thinking began. A real run builds no fewer prompts than this, so a line can show up to one turn earlier.

For every line shown it also reads, from the recorded run (the baseline, without the line), what a GPU arm should
read: was a named object clicked (or a named action used) within RESPONSE_ACTIONS actions, did that click change the
board inside the 4-cell edge band, and did a level-up follow within LEVEL_UP_SECONDS.

    .venv/bin/python -I scripts/untried_replay.py runs/RUN [runs/RUN2 ...] [--games tn36,vc33] [--patch P]
        [--tree ARC3-Inference] [--tokenizer tokenizer.json] [--json OUT]

--tokenizer counts the lines' tokens with a byte-level BPE tokenizer.json (ASCII text only); without it, tokens are
estimated as characters / 3.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
import statistics
import sys
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "kaggle" / "franzen" / "patches" / "ours-10-untried.patch"
NEW_FILE = "ARC3-Inference/inference/utils/ours_untried.py"
YIELD_TOKENS = 2048
TOK_PER_S = 55.0
RESPONSE_ACTIONS = 3
LEVEL_UP_SECONDS = 900.0
BORDER = 4  # ARC3_NOOP_GUARD_BORDER: a click "changed state" if the board changed inside this band
MODEL_NAMES = {0: "RESET", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SPACE", 6: "MOUSE", 7: "UNDO"}
DISPLAY = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT", "ACTION5": "SPACE",
           "ACTION7": "UNDO", "RESET": "RESET"}
# The post-mortem's seven losing game-runs, and the object each never clicked (docs/postmortems/bimodal-games-...).
CRITICAL = {
    ("tn36", 2): ("the demo's two boxes", lambda o: o.box[0] >= 52 and o.box[3] <= 28),
    ("vc33", 4): ("the orange gate", lambda o: o.colour == 12),
    ("sp80", 2): ("the red bars", lambda o: o.colour == 8),
    ("cn04", 2): ("the N and b 'sockets'", lambda o: o.colour in (9, 14)),
}
CRITICAL_RUNS = {("exp073", "tn36"), ("exp073", "vc33"), ("exp083", "vc33"), ("exp073", "sp80"), ("exp075", "sp80"),
                 ("exp083", "sp80"), ("exp075", "cn04")}


def load_module(patch: Path | None = None, tree: Path | None = None) -> types.ModuleType:
    """ours_untried as the patch adds it (or from an ARC3-Inference tree)."""
    if tree is not None:
        source = (Path(tree) / "inference" / "utils" / "ours_untried.py").read_text(encoding="utf-8")
    else:
        text = Path(patch or PATCH).read_text(encoding="utf-8")
        section = text.split(f"+++ b/{NEW_FILE}\n", 1)[1].split("\ndiff --git ", 1)[0]
        body = section.split("\n", 1)[1]  # after the @@ line
        source = "\n".join(line[1:] for line in body.splitlines() if line.startswith("+")) + "\n"
    module = types.ModuleType("ours_untried")
    module.__dict__["__name__"] = "ours_untried"
    exec(compile(source, NEW_FILE, "exec"), module.__dict__)
    return module


# --- token counting (optional) ------------------------------------------------------------------------------------


class BPE:
    """A byte-level BPE tokenizer.json, enough to count tokens of ASCII text (Qwen's pre-tokenizer rule, ASCII)."""

    SPLIT = re.compile(r"'(?:[sStTmMdD]|re|RE|ve|VE|ll|LL)|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]| ?[^\sA-Za-z0-9]+[\r\n]*"
                       r"|\s*[\r\n]+|\s+(?!\S)|\s+")

    def __init__(self, path: Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = data["model"]["merges"]
        self.ranks = {tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m): i for i, m in enumerate(merges)}
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
        cs, n = bs[:], 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        self.byte_char = {b: chr(c) for b, c in zip(bs, cs)}
        self.cache: dict = {}

    def _word(self, word: str) -> int:
        if word in self.cache:
            return self.cache[word]
        parts = [self.byte_char[b] for b in word.encode("utf-8")]
        while len(parts) > 1:
            ranked = [(self.ranks.get((a, b), 1 << 30), i) for i, (a, b) in enumerate(itertools.pairwise(parts))]
            rank, i = min(ranked)
            if rank == 1 << 30:
                break
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        self.cache[word] = len(parts)
        return len(parts)

    def count(self, text: str) -> int:
        return sum(self._word(w) for w in self.SPLIT.findall(text))


# --- the replay ---------------------------------------------------------------------------------------------------


def replay(game: dict, env_dir: Path) -> tuple[list, list]:
    """(entries, meta): the harness's history for one recorded game (entry 0 is the start frame, then one entry per
    action: action display, frame grid/step/level, result flags) and, per entry, the recorded clock, tokens and the
    model-facing valid actions after it."""
    sys.path.insert(0, str(ROOT))
    logging.disable(logging.INFO)
    from arcengine import GameAction

    from arc3.env import Action, LocalEnv, make_arcade

    env = LocalEnv(make_arcade(env_dir), game["game_id"], seed=0)
    frame = env.frame

    def entry(display, fr, step, level, result):
        grid = tuple(tuple(int(v) for v in row) for row in fr.grid.tolist())
        return types.SimpleNamespace(action=display, frame=types.SimpleNamespace(grid=grid, step=step, level=level),
                                     result=result)

    def valid(fr) -> list:
        return [MODEL_NAMES.get(int(a), str(a)) for a in fr.available_actions]

    entries = [entry("", frame, 0, frame.levels_completed + 1, {})]
    meta = [{"t": 0.0, "tok": 0, "valid": valid(frame), "id": ""}]
    for i, rec in enumerate(game["history"], 1):
        a = rec["action"]
        if a["id"] == "ACTION6":
            act = Action.click(a["data"]["x"], a["data"]["y"])
            display = f"MOUSE(row={int(a['data']['y'])}, col={int(a['data']['x'])})"
        else:
            act, display = Action(GameAction[a["id"]]), DISPLAY[a["id"]]
        before = frame
        frame = env.step(act)
        won = frame.state.name == "WIN"
        level = before.levels_completed + 1 if won else frame.levels_completed + 1
        result = {"game_over": frame.state.name == "GAME_OVER", "run_complete": won,
                  "level_completed": frame.levels_completed > before.levels_completed and not won,
                  "automatic": a["id"] == "RESET"}
        entries.append(entry(display, frame, i, level, result))
        meta.append({"t": float(rec["wallclock_seconds"]), "tok": int(rec.get("generated_tokens") or 0),
                     "valid": valid(frame), "id": a["id"]})
    return entries, meta


def schedule(meta: list) -> tuple[list, list]:
    """(calls, prompts): calls are (first index, last index, start time, end time, tokens); prompts are (time, number
    of history entries the prompt sees), in time order."""
    calls = []
    for i in range(1, len(meta)):
        m = meta[i]
        if m["id"] == "RESET":
            continue
        if not calls or m["tok"] > 0 or meta[i - 1]["id"] == "RESET":
            calls.append([i, i, m["t"], m["t"], m["tok"]])
        else:
            calls[-1][1], calls[-1][3] = i, m["t"]
    prompts = []
    prev_end = None
    for first, _last, start, end, tok in calls:
        sees = first  # every entry before this call's first action, the solver's RESETs included
        if prev_end is None:
            prompts.append((max(0.0, start - tok / TOK_PER_S), sees))
        else:
            prompts.append((prev_end + 0.001, sees))
        for j in range(1, tok // YIELD_TOKENS + 1):
            t = start - (tok - j * YIELD_TOKENS) / TOK_PER_S
            if prev_end is None or t > prev_end:
                prompts.append((min(t, start), sees))
        prev_end = end
    prompts.append((prev_end + 0.001 if prev_end is not None else 0.0, len(meta)))
    prompts.sort(key=lambda p: (p[0], p[1]))
    return [tuple(c) for c in calls], prompts


def _interior_changed(a, b) -> bool:
    h, w = len(a), len(a[0]) if a else 0
    return any(a[r][c] != b[r][c] for r in range(BORDER, h - BORDER) for c in range(BORDER, w - BORDER))


def run_game(u, run: str, game: dict, entries: list, meta: list) -> dict:
    """Drive the Tracker over one game; return its firings with what the recorded run did next."""
    tracker = u.Tracker()
    calls, prompts = schedule(meta)
    stamps = sorted((last, end) for _, last, _, end, _ in calls)
    out, si, shown = [], 0, 0
    modes = frozenset(("i1", "i2"))
    for now, sees in prompts:
        while si < len(stamps) and stamps[si][1] <= now:
            tracker.stamp(entries[stamps[si][0]].frame.step, stamps[si][1])
            si += 1
        tracker.lines(entries[:sees], meta[sees - 1]["valid"], now, modes)
        for rec in tracker.fired[shown:]:
            out.append(describe_firing(u, run, game, entries, meta, rec, sees))
        shown = len(tracker.fired)
    return {"firings": out, "levels": level_outcomes(entries, meta)}


def level_outcomes(entries: list, meta: list) -> dict:
    """level -> {"solved", "start", "end"} from the replayed frames and the recorded clock."""
    out: dict = {}
    for i, e in enumerate(entries):
        lv = e.frame.level
        d = out.setdefault(lv, {"solved": False, "start": meta[i]["t"], "end": meta[i]["t"]})
        d["end"] = meta[i]["t"]
        if i and entries[i - 1].frame.level < lv:
            out[entries[i - 1].frame.level]["solved"] = True
        if e.result.get("run_complete"):
            d["solved"] = True
    return out


def describe_firing(u, run, game, entries, meta, rec, sees) -> dict:
    gid = game["game_id"][:4]
    level = rec["level"]
    names = []
    if rec["line"] == "i1":
        order = sorted(rec["kinds"].items(), key=lambda kv: (rec["prev_kinds"] is not None and kv[0] in rec[
            "prev_kinds"], len(kv[1]), -kv[1][0].px, kv[1][0].box))[:u.MAX_KINDS]
        names = [o for _, objs in order for o in objs]
        keys = {k for k, _ in order}
    else:
        names = list(rec["objects"])
        keys = set()
    hit, used, changed = None, None, None
    for j in range(sees, min(len(entries), sees + RESPONSE_ACTIONS)):
        action = entries[j].action
        if rec["line"] == "i1" and u.action_type(action) in set(rec.get("unused") or []) and used is None:
            used = j - sees + 1
        cell = u.mouse_cell(action)
        if cell is None or hit is not None:
            continue
        before = entries[j - 1].frame.grid
        target = u.click_target(u.salient(u.components(before), (len(before), len(before[0]))), *cell)
        if target is None:
            continue
        if (rec["line"] == "i1" and u.kind_key(target) in keys) or any(target.cells & o.cells for o in names):
            hit = j - sees + 1
            changed = _interior_changed(before, entries[j].frame.grid)
    t = rec["time"]
    later = [meta[i]["t"] for i in range(sees, len(entries)) if entries[i].frame.level > level
             or entries[i].result.get("run_complete")]
    level_up = bool(later) and later[0] - t <= LEVEL_UP_SECONDS
    crit = None
    if (gid, level) in CRITICAL:
        crit = any(CRITICAL[(gid, level)][1](o) for o in names)
    return {"run": run, "game": gid, "level": level, "line": rec["line"], "reason": rec.get("reason", "event"),
            "time": round(t, 1), "text": rec["text"], "shown": (min(len(rec["kinds"]), u.MAX_KINDS) + len(
                rec.get("unused") or [])) if rec["line"] == "i1" else len(names), "hit_within": hit,
            "hit_changed": changed, "used_within": used, "level_up_15min": level_up, "critical_named": crit}


def summarise(results: dict, bpe: BPE | None) -> dict:
    rows = {}
    for run, games in results.items():
        firings = [f for g in games.values() for f in g["firings"]]
        solved = {(g, lv): d["solved"] for g, data in games.items() for lv, d in data["levels"].items()}
        row: dict = {}
        for line in ("i1", "i2"):
            fs = [f for f in firings if f["line"] == line]
            on_solved = [f for f in fs if solved.get((f["game"], f["level"]))]
            levels = {(f["game"], f["level"]) for f in fs}
            toks = [bpe.count(f["text"]) if bpe else round(len(f["text"]) / 3) for f in fs]
            chars = [len(f["text"]) for f in fs]
            row[line] = {
                "firings": len(fs), "on_solved_levels": len(on_solved), "on_unsolved_levels": len(fs) - len(on_solved),
                "levels_with_a_firing": len(levels),
                "levels_solved_with_a_firing": sum(1 for k in levels if solved.get(k)),
                "by_reason": dict(Counter(f["reason"] for f in fs)),
                "entries_median": statistics.median([f["shown"] for f in fs]) if fs else 0,
                "tokens_median": statistics.median(toks) if toks else 0,
                "tokens_p90": sorted(toks)[int(0.9 * (len(toks) - 1))] if toks else 0,
                "chars_median": statistics.median(chars) if chars else 0,
                "baseline_clicked_named_within_3": sum(1 for f in fs if f["hit_within"]),
                "baseline_click_changed_state": sum(1 for f in fs if f["hit_changed"]),
                "baseline_used_named_action_within_3": sum(1 for f in fs if f["used_within"]),
                "baseline_level_up_within_15min": sum(1 for f in fs if f["level_up_15min"]),
            }
        row["levels"] = len(solved)
        row["levels_solved"] = sum(solved.values())
        rows[run] = row
    return rows


def critical_table(results: dict) -> list:
    out = []
    for run, gid in sorted(CRITICAL_RUNS):
        games = results.get(run, {})
        if gid not in games:
            continue
        level = next(lv for (g, lv) in CRITICAL if g == gid)
        data = games[gid]
        start = data["levels"].get(level, {}).get("start", 0.0)
        first = {}
        for f in data["firings"]:
            if f["level"] == level and f["critical_named"] and f["line"] not in first:
                first[f["line"]] = round((f["time"] - start) / 60, 1)
        any_i1 = [f for f in data["firings"] if f["level"] == level and f["line"] == "i1"]
        out.append({"run": run, "game": gid, "level": level, "object": CRITICAL[(gid, level)][0],
                    "i1_first_naming_min_in": first.get("i1"), "i2_first_naming_min_in": first.get("i2"),
                    "i1_firings_on_level": len(any_i1),
                    "i1_first_text": any_i1[0]["text"] if any_i1 else ""})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="run directories (with kernel-output/benchmark.json)")
    ap.add_argument("--games", default="", help="comma-separated game id prefixes (default: all)")
    ap.add_argument("--patch", type=Path, default=PATCH)
    ap.add_argument("--tree", type=Path, default=None, help="an ARC3-Inference dir with the patch applied instead")
    ap.add_argument("--tokenizer", type=Path, default=None, help="a byte-level BPE tokenizer.json for token counts")
    ap.add_argument("--env-dir", type=Path, default=ROOT / "environment_files")
    ap.add_argument("--json", type=Path, default=None, help="write every firing and the summary here")
    args = ap.parse_args()
    u = load_module(args.patch, args.tree)
    bpe = BPE(args.tokenizer) if args.tokenizer else None
    prefixes = [g for g in args.games.split(",") if g]
    results: dict = {}
    for run_dir in args.runs:
        bench = run_dir / "kernel-output" / "benchmark.json" if run_dir.is_dir() else run_dir
        name = (run_dir.name if run_dir.is_dir() else run_dir.parent.parent.name).split("-")[0]
        data = json.loads(bench.read_text())
        results[name] = {}
        for game in data["game_runs"]:
            if prefixes and not any(game["game_id"].startswith(p) for p in prefixes):
                continue
            entries, meta = replay(game, args.env_dir)
            results[name][game["game_id"][:4]] = run_game(u, name, game, entries, meta)
    summary = summarise(results, bpe)
    crit = critical_table(results)
    if args.json:
        firings = [f for games in results.values() for g in games.values() for f in g["firings"]]
        levels = {run: {g: {str(k): v for k, v in d["levels"].items()} for g, d in games.items()}
                  for run, games in results.items()}
        args.json.write_text(json.dumps({"summary": summary, "critical": crit, "firings": firings, "levels": levels},
                                        indent=1))
    print(json.dumps({"summary": summary, "critical": crit}, indent=1))


if __name__ == "__main__":
    main()
