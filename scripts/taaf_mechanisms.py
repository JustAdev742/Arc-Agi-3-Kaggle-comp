#!/usr/bin/env python
"""Behaviour metrics of Duck (TAAF) runs from their transcripts: what our patches are meant to change.

    .venv/bin/python scripts/taaf_mechanisms.py runs/exp032-anim-flashnext runs/exp039-ours-d [...]

Per run (the folder holding ``kernel-output/transcripts`` or the transcripts themselves), over all games:
- turns and acting turns: a turn is one ``--- analysis_step=N | action=M | HH:MM:SS`` block; it acted when the next
  turn's action number is higher (the last turn of a game is not counted);
- longest action-free stretch per level (minutes between the start of a level or of an acting turn and the next
  acting turn), median over (game, level) pairs;
- minutes from a level's first turn to its first acting turn, for levels 2 and later (a new level's first move);
- tool errors per model response, by exception type (from ``[TOOL RESULT: python]`` blocks);
- carried-note updates: turns whose "Working world model carried" block differs from the previous turn's.
The stall analysis of the public thui run (research log 2026-09-23) is the reference these are compared with.
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import Counter
from pathlib import Path

TURN_RE = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d):(\d\d):(\d\d) \| tool-agent ---$", re.M)
LEVEL_RE = re.compile(r"^Current state: step \d+, level (\d+)", re.M)
NOTE_RE = re.compile(r"Working world model carried from earlier turns:\n(.*?)end of world model\.", re.S)
ERROR_TYPE_RE = re.compile(r"\b([A-Z][A-Za-z]*(?:Error|Exception))\b")


def transcripts_dir(path: Path) -> Path:
    for cand in (path / "kernel-output" / "transcripts", path / "transcripts", path):
        if cand.is_dir() and any(cand.glob("*.txt")):
            return cand
    raise SystemExit(f"no transcripts under {path}")


def game_turns(text: str) -> list[dict]:
    heads = list(TURN_RE.finditer(text))
    turns, day = [], 0
    last_t = None
    for i, m in enumerate(heads):
        body = text[m.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        t = int(m.group(3)) * 3600 + int(m.group(4)) * 60 + int(m.group(5))
        if last_t is not None and t + day < last_t - 3600:
            day += 86400
        last_t = t + day
        level = LEVEL_RE.search(body)
        note = NOTE_RE.search(body)
        results = re.findall(r"^\[TOOL RESULT: python\]\n(.*?)(?=^\[|\Z)", body, re.M | re.S)
        errors = []
        for r in results:
            m_err = re.search(r"^error:\n(.*)", r, re.M | re.S)  # the tool result prints stdout, then `error:` and the text
            if not m_err or not m_err.group(1).strip():
                continue
            found = ERROR_TYPE_RE.findall(m_err.group(1))
            errors.append(found[-1] if found else "other")
        turns.append({"action": int(m.group(2)), "t": t + day, "level": int(level.group(1)) if level else None,
                      "note": note.group(1).strip() if note else "", "responses": body.count("[MODEL RESPONSE META]"),
                      "errors": errors})
    return turns


def run_stats(path: Path) -> dict:
    acting = total = responses = note_changes = note_turns = 0
    idle, first_move = [], []
    errors: Counter = Counter()
    for f in sorted(transcripts_dir(path).glob("*.txt")):
        turns = game_turns(f.read_text(errors="replace"))
        for i, turn in enumerate(turns):
            responses += turn["responses"]
            errors.update(turn["errors"])
            if i > 0:
                note_turns += 1
                note_changes += turn["note"] != turns[i - 1]["note"]
        # acting turns and per-level stretches (the last turn's outcome is unknown)
        level_start: dict[int, float] = {}
        last_act: dict[int, float] = {}
        longest: dict[int, float] = {}
        firsts: dict[int, float] = {}
        for i, turn in enumerate(turns[:-1]):
            lvl = turn["level"] or 0
            level_start.setdefault(lvl, turn["t"])
            total += 1
            if turns[i + 1]["action"] > turn["action"]:
                acting += 1
                since = last_act.get(lvl, level_start[lvl])
                longest[lvl] = max(longest.get(lvl, 0.0), (turn["t"] - since) / 60.0)
                last_act[lvl] = turn["t"]
                if lvl >= 2 and lvl not in firsts:
                    firsts[lvl] = (turn["t"] - level_start[lvl]) / 60.0
        if turns:
            end_t = turns[-1]["t"]
            for lvl, start in level_start.items():
                longest[lvl] = max(longest.get(lvl, 0.0), (end_t - last_act.get(lvl, start)) / 60.0)
        idle.extend(longest.values())
        first_move.extend(firsts.values())
    return {"run": str(path), "turns": total, "acting_share": round(acting / total, 3) if total else None,
            "responses": responses,
            "idle_min_median": round(statistics.median(idle), 1) if idle else None,
            "first_move_L2plus_min_median": round(statistics.median(first_move), 1) if first_move else None,
            "tool_errors_per_response": round(sum(errors.values()) / responses, 3) if responses else None,
            "top_errors": dict(errors.most_common(6)),
            "note_update_share": round(note_changes / note_turns, 3) if note_turns else None}


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    for arg in sys.argv[1:]:
        s = run_stats(Path(arg))
        print(f"{s['run']}: turns {s['turns']}, acting {s['acting_share']}, longest idle/level median "
              f"{s['idle_min_median']} min, first move on L2+ median {s['first_move_L2plus_min_median']} min, "
              f"tool errors/response {s['tool_errors_per_response']} {s['top_errors']}, note updated in "
              f"{s['note_update_share']} of turns")


if __name__ == "__main__":
    main()
