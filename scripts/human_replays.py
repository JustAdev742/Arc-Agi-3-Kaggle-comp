#!/usr/bin/env python
"""Summarise ARC Prize human play recordings (road-to-100 item 7; plan-100 section 4.1).

Input: the recording JSONL files the toolkit and the ARC Prize site write (docs.arcprize.org/recordings): one line per
action, ``{"timestamp": ..., "data": {"game_id", "state", "levels_completed", "win_levels", "action_input": {"id",
"data"}, "guid", "full_reset", "available_actions", "frame"?}}``, one file per play under ``data/human/`` (any
layout; files are found by suffix). The public human dataset (342 replays over the 25 public games) is what the owner
has to download by browser (docs/status.md, follow-ups); until then this runs on the synthetic file of its test.

Output (``--out``, default arc3/data/human_priors.json): per game the number of replays, wins, the median action count
per level index over winning replays (the same statistic the RHAE baseline is built from), the first three actions
humans take, the click fraction; and cross-game aggregates that are legitimate priors for hidden games (per-game
figures are for analysis of the dev split only: using them in play on the dev games would be leakage).

Usage: .venv/bin/python scripts/human_replays.py [--root data/human] [--out arc3/data/human_priors.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KEYS = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT", "ACTION5": "ACT", "ACTION6": "CLICK",
        "ACTION7": "UNDO", "RESET": "RESET"}
IDS = {0: "RESET", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "ACT", 6: "CLICK", 7: "UNDO"}  # the published files use ids


def action_name(action_input: dict[str, Any]) -> str | None:
    """'UP' / 'CLICK(x,y)' / 'RESET' from an action_input; the published recordings carry integer ids (0 = RESET),
    the docs example the ACTIONn strings. None when there is no action (the trailing scorecard line)."""
    act = action_input.get("id")
    if act is None:
        return None
    name = IDS.get(act) if isinstance(act, int) else KEYS.get(str(act), str(act))
    if name == "CLICK":
        xy = action_input.get("data") or {}
        return f"CLICK({xy.get('x')},{xy.get('y')})"
    return name


def _short(game_id: str) -> str:
    return str(game_id).split("-")[0]


def parse_recording(path: Path) -> dict[str, Any] | None:
    """One replay: the action sequence per level index (1-based; RESETs after the opening one included, as the scorer
    counts them)."""
    per_level: dict[int, list[str]] = defaultdict(list)
    game = None
    last_levels = 0
    won = False
    win_levels = None
    n = 0
    card_levels: list[int] | None = None  # the scorecard's own per-level cumulative action counts, when present
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line).get("data") or {}
            except json.JSONDecodeError:
                continue
            if "cards" in d and "action_input" not in d:
                # trailing scorecard line: {'won', 'played', 'total_actions', 'cards': {game_id: {'actions_by_level': [[[lvl, cum], ...]]}}}
                for card in (d.get("cards") or {}).values():
                    abl = (card.get("actions_by_level") or [[]])[0]
                    if abl:
                        card_levels = [int(c) for _, c in abl]
                continue
            game = game or _short(d.get("game_id", path.name))
            name = action_name(d.get("action_input") or {})
            if name is None:
                continue
            if n == 0 and name == "RESET" and not per_level:
                # the game-opening RESET is not billed: the ls20 recording of 2026-09-17 has 547 action lines and its
                # scorecard says total_actions 546 with level 1 at 21 (later RESETs are billed as level restarts)
                continue
            # the level the action was taken on = the level index before the action's effect
            per_level[last_levels + 1].append(name)
            n += 1
            last_levels = int(d.get("levels_completed", last_levels) or 0)
            win_levels = d.get("win_levels", win_levels)
            won = won or str(d.get("state", "")).upper() == "WIN"
    if game is None or n == 0:
        return None
    out = {"game": game, "file": path.name, "actions": n, "levels_completed": last_levels, "win_levels": win_levels,
           "won": won, "per_level": {str(k): v for k, v in sorted(per_level.items())}}
    if card_levels:
        # per-level counts as the scorecard bills them (differences of the cumulative counts); they must agree with ours
        counts = [c - (card_levels[i - 1] if i else 0) for i, c in enumerate(card_levels)]
        out["card_actions_per_level"] = counts
        out["card_agrees"] = counts == [len(out["per_level"].get(str(i + 1), [])) for i in range(len(counts))]
    return out


def summarise(replays: list[dict[str, Any]]) -> dict[str, Any]:
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in replays:
        by_game[r["game"]].append(r)
    games: dict[str, Any] = {}
    all_first: Counter = Counter()
    level1_actions: list[int] = []
    click_fracs: list[float] = []
    for game, rs in sorted(by_game.items()):
        wins = [r for r in rs if r["won"]]
        per_level_counts: dict[str, list[int]] = defaultdict(list)
        for r in wins:
            for lvl, acts in r["per_level"].items():
                if int(lvl) <= int(r["levels_completed"]):  # completed levels only (the last, unfinished one is not a baseline)
                    per_level_counts[lvl].append(len(acts))
        firsts = Counter(" ".join(a.split("(")[0] for a in list(r["per_level"].get("1", []))[:3]) for r in rs)
        clicks = sum(a.startswith("CLICK") for r in rs for acts in r["per_level"].values() for a in acts)
        total = sum(len(acts) for r in rs for acts in r["per_level"].values())
        frac = clicks / total if total else 0.0
        games[game] = {"replays": len(rs), "wins": len(wins),
                       "median_actions_per_level": {lvl: statistics.median(v) for lvl, v in sorted(per_level_counts.items(), key=lambda kv: int(kv[0]))},
                       "first_three_actions": firsts.most_common(3), "click_fraction": round(frac, 3)}
        all_first.update(firsts)
        level1_actions += per_level_counts.get("1", [])
        click_fracs.append(frac)
    cross = {"games": len(games), "replays": len(replays), "wins": sum(1 for r in replays if r["won"]),
             "median_level1_actions": statistics.median(level1_actions) if level1_actions else None,
             "first_three_actions": all_first.most_common(5),
             "median_click_fraction": statistics.median(click_fracs) if click_fracs else None}
    return {"version": 1, "cross_game": cross, "games": games}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "human"))
    ap.add_argument("--out", default=str(ROOT / "arc3" / "data" / "human_priors.json"))
    a = ap.parse_args()
    root = Path(a.root)
    files = sorted(root.rglob("*.jsonl")) if root.exists() else []
    replays = [r for r in (parse_recording(p) for p in files) if r]
    if not replays:
        print(f"[human_replays] no recordings under {root} (see docs/status.md follow-ups for the download)", file=sys.stderr)
        raise SystemExit(1)
    summary = summarise(replays)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    c = summary["cross_game"]
    print(f"[human_replays] {c['replays']} replays, {c['games']} games, {c['wins']} wins; median level-1 actions "
          f"{c['median_level1_actions']}; first actions {c['first_three_actions'][:3]} -> {out}")


if __name__ == "__main__":
    main()
