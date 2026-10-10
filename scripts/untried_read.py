#!/usr/bin/env python
"""Read a GPU run with OURS_UNTRIED on: every I1/I2 line the model actually saw, and what it did in the next actions.

    .venv/bin/python -I scripts/untried_read.py runs/RUN [--json OUT]

scripts/untried_replay.py reconstructs when the lines would fire from a recorded run's clock and tokens; this reads the
lines as they were shown, from the run's per-game transcripts (`kernel-output/solver_analysis/*.html`, the "[USER
PROMPT]" sections with "Current state: step N, level L."). One firing per distinct (game, step, line). For each:
- "step N" counts the history entries the prompt saw (the start frame and N-1 actions: the step-24 prompt follows
  replayed entries 20-23), so the board it saw is entry N-1 (scripts/untried_replay.replay, the local engine) and
  the model's next action is entry N;
- I1: each "colour box" entry of the line is matched to the salient object with that colour and box on that board,
  and its kind (ours_untried.kind_key) is a named kind; I2: the named objects are matched by their new colour ("X box
  turned Y" is a Y object on that board; "Y box appeared" too);
- over the next RESPONSE_ACTIONS actions: a click whose target (ours_untried.click_target) is of a named kind (I1) or
  overlaps a named object (I2), whether that click changed the board inside the edge band, and (I1) a named unused
  action type used;
- a level-up within LEVEL_UP_SECONDS of action N's recorded time.
These are the quantities scripts/untried_replay.py reads from runs without the lines (its baselines), so the two
compare directly.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import untried_replay as ur  # noqa: E402

SECTION = re.compile(r"^\[USER PROMPT\]\n(.*?)(?=^\[[A-Z][A-Z ]+[^\n]*\]\n|\Z)", re.S | re.M)
STATE = re.compile(r"Current state: step (\d+), level (\d+)")
I1 = re.compile(r"Not yet tried on this level: (.*?)\. Each is a one-action test of whether it is interactive\.", re.S)
I2 = re.compile(r"After [^\n]*?, a colour new to this game; no click has hit (?:it|them) since\.")
ENTRY = re.compile(r"(?<![A-Za-z0-9])([A-Za-z0-9]) r(\d+)(?:-(\d+))? c(\d+)(?:-(\d+))?")
UNUSED = re.compile(r"^([A-Z, ]+) never used")
I2_PART = re.compile(r"([A-Za-z0-9]) r(\d+)(?:-(\d+))? c(\d+)(?:-(\d+))?(?: turned ([A-Za-z0-9]))?")


def lines_shown(path: Path) -> list:
    """[(step, level, kind, text)] in transcript order, one per distinct (step, kind, text)."""
    text = html.unescape(re.sub(r"<[^>]+>", "\n", path.read_text(encoding="utf-8", errors="replace")))
    seen, out = set(), []
    for m in SECTION.finditer(text):
        block = m.group(1)
        state = STATE.search(block)
        if not state:
            continue
        step, level = int(state.group(1)), int(state.group(2))
        for kind, rx in (("i1", I1), ("i2", I2)):
            for hit in rx.finditer(block):
                line = " ".join(hit.group(0).split())
                if (step, kind, line) not in seen:
                    seen.add((step, kind, line))
                    out.append((step, level, kind, line))
    return out


def _box(m) -> tuple:
    r0, r1, c0, c1 = int(m.group(2)), m.group(3), int(m.group(4)), m.group(5)
    return (r0, c0, int(r1) if r1 else r0, int(c1) if c1 else c0)


def read_firing(u, entries, meta, step, level, kind, line) -> dict:
    seen = step - 1  # the last entry the prompt saw
    grid = entries[seen].frame.grid
    shape = (len(grid), len(grid[0]))
    objs = u.salient(u.components(grid), shape)
    if kind == "i1":
        named_part = line.split("no click has hit ", 1)[1] if "no click has hit " in line else ""
        wanted = [(m.group(1), _box(m)) for m in ENTRY.finditer(named_part)]
    else:
        body = line.split(", a colour new to this game", 1)[0].split(", ", 1)[-1]
        wanted = []
        for part in body.split("; "):
            m = I2_PART.search(part)
            if m:
                wanted.append((m.group(6) or m.group(1), _box(m)))
    named = [o for c, box in wanted for o in objs if u.colour_char(o.colour) == c and o.box == box]
    kinds = {u.kind_key(o) for o in named}
    unused = set()
    if kind == "i1":
        head = line.split(": ", 1)[1]
        mu = UNUSED.match(head)
        if mu:
            unused = {a.strip() for a in mu.group(1).split(",") if a.strip()}
    hit = used = changed = None
    for j in range(step, min(len(entries), step + ur.RESPONSE_ACTIONS)):
        action = entries[j].action
        if unused and used is None and u.action_type(action) in unused:
            used = j - seen
        cell = u.mouse_cell(action)
        if cell is None or hit is not None:
            continue
        before = entries[j - 1].frame.grid
        target = u.click_target(u.salient(u.components(before), shape), *cell)
        if target is None:
            continue
        if (kind == "i1" and u.kind_key(target) in kinds) or any(target.cells & o.cells for o in named):
            hit, changed = j - seen, ur._interior_changed(before, entries[j].frame.grid)
    t0 = meta[seen]["t"]
    later = [meta[i]["t"] for i in range(step, len(entries))
             if entries[i].frame.level > level or entries[i].result.get("run_complete")]
    return {"step": step, "level": level, "line": kind, "text": line, "named_entries": len(wanted),
            "named_matched": len(named), "level_matches_replay": entries[seen].frame.level == level,
            "hit_within": hit, "hit_changed": changed, "used_within": used,
            "level_up_15min": bool(later) and later[0] - t0 <= ur.LEVEL_UP_SECONDS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--env-dir", type=Path, default=ROOT / "environment_files")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    u = ur.load_module()
    out = args.run / "kernel-output"
    bench = json.loads((out / "benchmark.json").read_text())
    firings = []
    for game in bench["game_runs"]:
        gid = game["game_id"]
        pages = sorted((out / "solver_analysis").glob(f"{gid}*.html"))
        shown = [x for p in pages for x in lines_shown(p)]
        if not shown:
            continue
        entries, meta = ur.replay(game, args.env_dir)
        for step, level, kind, line in shown:
            if not 1 <= step <= len(entries):
                continue
            rec = read_firing(u, entries, meta, step, level, kind, line)
            rec["game"] = gid[:4]
            firings.append(rec)
    summary = {}
    for kind in ("i1", "i2"):
        fs = [f for f in firings if f["line"] == kind]
        n = len(fs) or 1
        summary[kind] = {
            "firings": len(fs),
            "clicked_named_within_3": sum(f["hit_within"] is not None for f in fs),
            "clicked_named_within_3_share": round(sum(f["hit_within"] is not None for f in fs) / n, 3),
            "click_changed_board": sum(bool(f["hit_changed"]) for f in fs),
            "used_named_action_within_3": sum(f["used_within"] is not None for f in fs),
            "level_up_within_15min": sum(f["level_up_15min"] for f in fs),
            "level_up_within_15min_share": round(sum(f["level_up_15min"] for f in fs) / n, 3),
            "entries_named": sum(f["named_entries"] for f in fs), "entries_matched": sum(f["named_matched"] for f in fs),
            "level_mismatches": sum(not f["level_matches_replay"] for f in fs),
        }
    print(json.dumps(summary, indent=1))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "firings": firings}, indent=1))


if __name__ == "__main__":
    main()
