#!/usr/bin/env python
"""Measure the budget meter (kaggle/franzen/patches/ours-02-budget-meter.patch) on recorded games, offline, on CPU.

    .venv/bin/python scripts/budget_meter_eval.py runs/exp054-fix-kv775-obj runs/exp073b-... [--json out.json]

Each RUN_DIR is a run's output. Its frames come from the per-action boards in `*_events.jsonl` (the viewer events of
Tufa's harness and of Franzen's) when it has them, else from replaying the action histories in its benchmark.json in
the offline engine (environment_files/, ONLY_RESET_LEVELS=true as in his notebook); a replay of recorded boards
matches them exactly. Every recording is also replayed to read the game's own budget counter where its camera
interface keeps one (`current_steps`, 16 of the 25 public games).

At every step the meter runs on the history up to that step, which is what the harness gives it, and the reading is
checked against:
  - the true bar of each game (TRUE below, read off the frames of all 25 games; sb26 and sk48 draw no edge bar):
    a reading of any other line is a wrong-line claim;
  - the realized future of the true bar in the same recording: from step t, the next event is the bar emptying
    (exact: actions until then), a budget death at D (exact: D - 1 - t safe actions), or a refill from a full bar,
    a win, a hazard death, a RESET or the end of the recording (a lower bound: the attempt survived that long).
    Actions of a kind the reading calls free, and for a bar that drops on every charged action the actions that did
    not lower it, are not counted. A reading is near the end when it or the truth is at most 10, wrong when it is
    off by more than 2 from an exact truth or more than 2 below a lower bound;
  - the game's counter, where there is one (ls20's counts cells, 1 or 2 per move, so it is left out).
Every GAME_OVER is also checked: the harness names a budget death only when `budget_death` says so; truth is the
true bar empty in the death frame and, where there is a counter, the counter at most 1.

Validation games (cn04, g50t, lf52, r11l, sc25, sp80) are reported apart from the 19 dev games.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import types
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "kaggle" / "franzen" / "patches" / "ours-02-budget-meter.patch"
NEW_FILE = "ARC3-Inference/inference/utils/budget_bar.py"
VALIDATION = {"cn04", "g50t", "lf52", "r11l", "sc25", "sp80"}
# (axis, index, colour) of each game's budget bar; m0r0 drains rows 0 and 63 together, sp80 uses row 0 or row 63
TRUE = {
    "ar25": {("c", 63, 11)}, "bp35": {("r", 63, 0)}, "cd82": {("r", 63, 4)}, "cn04": {("r", 0, 4)},
    "dc22": {("r", 63, 0)}, "ft09": {("r", 63, 12)}, "g50t": {("r", 63, 9)}, "ka59": {("r", 63, 4)},
    "lf52": {("r", 0, 0)}, "lp85": {("c", 0, 14)}, "ls20": {("r", 61, 11), ("r", 62, 11)},
    "m0r0": {("r", 0, 5), ("r", 63, 5)}, "r11l": {("c", 0, 0)}, "re86": {("r", 63, 15)}, "s5i5": {("r", 63, 3)},
    "sb26": set(), "sc25": {("c", 62, 14), ("c", 63, 14)}, "sk48": set(), "sp80": {("r", 0, 14), ("r", 63, 14)},
    "su15": {("r", 63, 0)}, "tn36": {("r", 1, 9)}, "tr87": {("r", 63, 1)}, "tu93": {("r", 63, 6)},
    "vc33": {("r", 0, 7)}, "wa30": {("r", 63, 7)},
}
DISPLAY = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT", "ACTION5": "SPACE",
           "ACTION7": "UNDO", "RESET": "RESET"}
MOUSE = re.compile(r"MOUSE\(row=(\d+), col=(\d+)\)")


def load_meter(patch: Path = PATCH) -> types.ModuleType:
    """inference/utils/budget_bar.py exactly as the patch adds it."""
    section = patch.read_text().split(f"+++ b/{NEW_FILE}\n", 1)[1].split("\ndiff --git ", 1)[0]
    source = "\n".join(line[1:] for line in section.split("\n", 1)[1].splitlines() if line.startswith("+")) + "\n"
    module = types.ModuleType("budget_bar")
    exec(compile(source, NEW_FILE, "exec"), module.__dict__)
    return module


# --- recordings ---------------------------------------------------------------------------------------------------


def from_events(path: Path) -> dict:
    actions, levels, over, boards = [], [], [], []
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            b = e.get("board")
            if e.get("type") not in ("action", "initial") or not b or len(b) != 64 or len(b[0]) != 64:
                continue
            actions.append("" if e["type"] == "initial" else str(e.get("action_display") or e.get("action_name") or ""))
            levels.append(int(e.get("level") or 1))
            over.append(bool(e.get("game_over")) or e.get("state") == "GAME_OVER")
            boards.append(np.asarray(b, dtype=np.uint8))
    return {"game": path.name.split("_p")[0], "source": str(path), "actions": actions, "levels": levels,
            "over": over, "boards": np.stack(boards) if boards else np.zeros((0, 64, 64), np.uint8)}


def replay(env_dir: str, game_id: str, steps: list) -> dict:
    """Play (engine action name, data) steps in the offline engine; boards, levels and the interface counters."""
    os.environ["ONLY_RESET_LEVELS"] = "true"
    import logging

    import arc_agi
    from arcengine import GameAction, GameState
    logging.disable(logging.WARNING)
    env = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir).make(
        game_id.split("-")[0])
    game = env._game
    out = {"game": game_id, "actions": [""], "levels": [], "over": [], "boards": [], "counters": []}

    def record(raw):
        out["boards"].append(np.asarray(raw.frame[-1], dtype=np.uint8))
        n = int(raw.win_levels or 0)
        out["levels"].append(max(1, min(n, int(raw.levels_completed) + 1)) if raw.state != GameState.WIN else max(1, n))
        out["over"].append(raw.state == GameState.GAME_OVER)
        out["counters"].append({f"{i}.{k}": int(v) for i, iface in enumerate(game.camera._interfaces)
                                for k, v in vars(iface).items() if isinstance(v, int) and not isinstance(v, bool)})

    record(env.observation_space)
    for name, data in steps:
        raw = env.step(GameAction[name], data=dict(data or {}))
        out["actions"].append(f"MOUSE(row={data['y']}, col={data['x']})" if name == "ACTION6" else DISPLAY[name])
        record(raw)
    out["boards"] = np.stack(out["boards"])
    return out


def recordings(run: Path, env_dir: str) -> list[dict]:
    events = sorted(p for p in run.rglob("*_events.jsonl") if p.stat().st_size > 1000)
    recs = []
    for p in events:
        rec = from_events(p)
        if len(rec["actions"]) < 2:
            continue
        steps = []
        for a in rec["actions"][1:]:
            m = MOUSE.fullmatch(a)
            steps.append(("ACTION6", {"x": int(m.group(2)), "y": int(m.group(1))}) if m else
                         ({v: k for k, v in DISPLAY.items()}[a], {}))
        rr = replay(env_dir, rec["game"], steps)
        rec["counters"] = rr["counters"]
        rec["replay_matches"] = bool((rr["boards"] == rec["boards"]).all())
        recs.append(rec)
    if not events:
        for bm in run.rglob("benchmark.json"):
            for g in json.loads(bm.read_text()).get("game_runs", []):
                if g.get("history"):
                    steps = [(h["action"]["id"], h["action"].get("data") or {}) for h in g["history"]]
                    rec = replay(env_dir, g["game_id"], steps)
                    rec.update(source=str(bm), replay_matches=True)
                    recs.append(rec)
    return recs


# --- truth and readings -------------------------------------------------------------------------------------------


def true_count(board, keys):
    """Cells of the true bar's colour on its line (the fuller line where a game has two)."""
    return max(int(((board[i, :] if axis == "r" else board[:, i]) == c).sum()) for axis, i, c in keys)


def future_truth(counts, over, kinds, t, end, unit, free, every):
    n, costlier = 0, False
    for e in range(t + 1, end + 1):
        n += kinds[e] not in free and (not every or counts[e] < counts[e - 1] or over[e])
        costlier = costlier or counts[e - 1] - counts[e] > 2 * unit + 1
        if counts[e] > counts[e - 1]:
            return ("exact", n, costlier) if counts[e - 1] <= 2 * unit else ("lower", n - 1, costlier)
        if over[e]:
            return ("exact" if counts[e] <= unit else "lower"), n - 1, costlier
        if counts[e] == 0:
            return "exact", n, costlier
    return "lower", n, costlier


def evaluate(args) -> tuple[str, Counter, Counter]:
    patch, rec = args
    bb = load_meter(Path(patch))
    g = rec["game"].split("-")[0]
    keys = TRUE.get(g, set())
    boards, actions, levels, over = rec["boards"], rec["actions"], rec["levels"], rec["over"]
    records = [bb.frame_record(actions[i], boards[i].tolist(), levels[i], i) for i in range(len(actions))]
    starts = bb.attempt_starts(records)
    counts = [true_count(b, keys) for b in boards] if keys else None
    kinds = [bb._kind(a) for a in actions]
    cs = rec.get("counters") or []
    ckey = next(iter(sorted(k for k in (cs[0] if cs else {}) if k.endswith(".current_steps"))), None)
    s, errs, known = Counter(), Counter(), {}
    for a, b in zip(starts, [*starts[1:], len(actions)], strict=True):
        end, true_drops, sizes = b - 1, 0, Counter()
        for t in range(a + 1, end + 1):
            if counts is not None and counts[t] < counts[t - 1]:
                true_drops += 1
                sizes[counts[t - 1] - counts[t]] += 1
            r = bb.read(records, known, n=t)
            s["steps"] += 1
            s["readings"] += r is not None
            eligible = true_drops >= 3
            s["eligible"] += eligible
            s["eligible_read"] += eligible and r is not None
            truth = None
            if counts is not None:
                free = set(r["free_actions"]) if r else set()
                truth = future_truth(counts, over, kinds, t, end, min(sizes) if sizes else 1, free,
                                     bool(r and r["drops_every_action"]))
                if eligible and truth[0] == "exact" and truth[1] <= 10:
                    s["end_eligible"] += 1
                    s["end_read"] += r is not None
            if r is None:
                continue
            if not {("r" if k[0] == "row" else "c", k[1], c) for k, c in r["_keys"]} & keys:
                s["wrong_line"] += 1
                continue
            kind, value, costlier = truth
            costlier = costlier or r["largest_drop"] > 2 * max(1, round(r["cells_per_action"])) + 1
            if kind == "exact" and min(value, r["actions_left"]) <= 10:
                err = r["actions_left"] - value
                errs[max(-5, min(5, err))] += 1
                s["near"] += 1
                s["near_off"] += abs(err) > 2
                s["near_off_over"] += err > 2
                s["near_off_costlier"] += abs(err) > 2 and costlier
            elif kind == "lower" and r["actions_left"] <= 10:
                s["near_lower"] += 1
                s["near_lower_off"] += r["actions_left"] < value - 2
            c = cs[t].get(ckey) if ckey and t < len(cs) else None
            if c is not None and min(c, r["actions_left"]) <= 10 and g != "ls20":
                s["counter"] += 1
                s["counter_off"] += abs(r["actions_left"] - c) > 2
                s["counter_over"] += r["actions_left"] - c > 2
        bb.learn(known, records, a, end)
    for d in range(1, len(actions) - 1):  # game overs, each followed by the automatic RESET
        if not over[d] or actions[d + 1] != "RESET":
            continue
        s["game_overs"] += 1
        budget = False
        if keys:
            before, at = counts[d - 1], counts[d]
            budget = at <= max(1, before - at if before > at else 1)
            if budget and ckey:
                budget = cs[d].get(ckey, 0) <= 1
        named = bb.budget_death(records[:d + 2]) is not None
        s["budget_deaths"] += budget
        s["named"] += named
        s["named_wrong"] += named and not budget
        s["missed"] += budget and not named
    s["recordings"] += 1
    s["recordings_with_readings"] += s["readings"] > 0
    return g, s, errs


def report(results) -> dict:
    per, errs = defaultdict(Counter), defaultdict(Counter)
    for g, s, e in results:
        per[g].update(s)
        errs[g].update(e)
    print(f"{'game':5s} {'split':5s} {'recs':>4s} {'steps':>6s} {'reads':>6s} {'cover':>6s} {'end':>5s} {'wrong':>5s} "
          f"{'near':>5s} {'off>2':>5s} {'cost':>4s} {'cntr':>5s} {'off':>4s} {'deaths':>6s} {'budg':>4s} "
          f"{'named':>5s} {'false':>5s}  errors near the end")
    split = defaultdict(Counter)
    for g in sorted(per):
        s = per[g]
        tag = "val" if g in VALIDATION else "dev"
        split[tag].update(s)
        split[tag]["games"] += 1
        split[tag]["games_with_readings"] += s["readings"] > 0
        print(f"{g:5s} {tag:5s} {s['recordings']:4d} {s['steps']:6d} {s['readings']:6d} "
              f"{s['eligible_read'] / max(1, s['eligible']):6.0%} {s['end_read'] / max(1, s['end_eligible']):5.0%} "
              f"{s['wrong_line']:5d} {s['near']:5d} {s['near_off']:5d} {s['near_off_costlier']:4d} {s['counter']:5d} "
              f"{s['counter_off']:4d} {s['game_overs']:6d} {s['budget_deaths']:4d} {s['named']:5d} "
              f"{s['named_wrong']:5d}  {dict(sorted(errs[g].items()))}")
    for tag, s in sorted(split.items()):
        print(f"{tag}: {s['games']} games ({s['games_with_readings']} with readings), {s['recordings']} recordings, "
              f"{s['steps']} steps, {s['readings']} readings; coverage {s['eligible_read'] / max(1, s['eligible']):.1%} "
              f"of steps after 3 true drops, {s['end_read'] / max(1, s['end_eligible']):.1%} of those within 10 of "
              f"empty; wrong line {s['wrong_line']}; near the end {s['near']} readings with a realized truth, "
              f"{s['near_off']} off by more than 2 ({s['near_off_over']} too high; {s['near_off_costlier']} with a "
              f"costlier-than-usual action), {s['near_lower_off']} of {s['near_lower']} below a survived bound; vs "
              f"the game's counter {s['counter']}, {s['counter_off']} off by more than 2 ({s['counter_over']} too "
              f"high); game overs {s['game_overs']}, budget deaths {s['budget_deaths']}, named {s['named']} "
              f"({s['named_wrong']} wrongly), missed {s['missed']}")
    return {g: dict(s) for g, s in per.items()} | {f"split:{t}": dict(s) for t, s in split.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="run output directories")
    ap.add_argument("--env-dir", default=str(ROOT / "environment_files"), help="the offline game files")
    ap.add_argument("--patch", type=Path, default=PATCH, help="the patch whose meter is measured")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--json", type=Path, default=None, help="write the per-game and per-split counts here")
    args = ap.parse_args()
    t0 = time.time()
    recs = [r for run in args.runs for r in recordings(run, args.env_dir)]
    mismatched = [r["source"] for r in recs if not r["replay_matches"]]
    print(f"{len(recs)} recordings, {sum(len(r['actions']) for r in recs)} frames from {len(args.runs)} runs; "
          f"replays differing from the recorded boards: {len(mismatched)} ({time.time() - t0:.0f} s)")
    with Pool(args.jobs) as pool:
        results = pool.map(evaluate, [(str(args.patch), r) for r in recs], chunksize=2)
    counts = report(results)
    print(f"done in {time.time() - t0:.0f} s")
    if args.json:
        args.json.write_text(json.dumps(counts, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
