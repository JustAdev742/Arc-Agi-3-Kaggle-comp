#!/usr/bin/env python
"""Mine solved levels from run transcripts into the offline skill library ``arc3/data/skills.json``.

For every level a REPL agent completed in ``runs/*/<game>.transcript.jsonl`` the miner extracts what code can
extract without a model call: the level's coarse signature (avatar or click-only, the observed key map, entity
count, tile size, HUD present), the model's own goal statement at the moment it won (the ``# goal: ...`` comment
line every cell starts with), what it said it had learned, the number of actions the level took and a compressed
action sequence from the per-game action log. Runs of the same (game, level) are merged: the card keeps the
fewest-actions solution and counts the wins.

At play time ``arc3.memory.match_skills`` retrieves the cards whose signature agrees with the current level (never
the same game: on the dev split that would be leakage, on the hidden set the game is unknown anyway) and the
agent shows them as hints. The cards are text; nothing in them is executed.

Each card also carries its evidence and a status (brief item 16): ``wins`` (transcripts that solved the level),
``failures`` (transcripts of the same game that reached the level and did not solve it), ``confidence`` =
wins / (wins + failures), ``last_validated`` (the latest winning run) and ``status``: ``validated`` (won at least
twice), ``candidate`` (won once, failed at most twice since), ``deprecated`` (won once, failed more than twice: a
one-off that later runs did not reproduce; never shown). A one-off observation is never promoted to a rule.

Usage: .venv/bin/python scripts/mine_skills.py [--runs runs] [--out arc3/data/skills.json] [--min-actions 1]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arc3.prompts import ACTION_NAMES  # noqa: E402

_GOAL = re.compile(r"#\s*goal:\s*(.*?)\s*(?:\||$)", re.I)
_LEARNED = re.compile(r"\|\s*learned:\s*(.*?)\s*(?:\|\s*now:|$)", re.I)
_LEVEL_DONE = re.compile(r"LEVEL (\d+) COMPLETED after (\d+) actions on it", re.I)
_LEGAL = re.compile(r"legal:\s*([A-Z, ]+?)\s*\|")
_TILE = re.compile(r"tile (\d+)")
_KEYS = re.compile(r"moves with keys (\{.*?\})")
_ENT = re.compile(r"#\d+ c\d+ @\(")


def _signature_from_observations(obs_texts: list[str]) -> dict[str, Any]:
    avatar = False
    keys: list[str] = []
    click_only = False
    n_ents = 0
    tile = 1
    hud = False
    for txt in obs_texts:
        m = _LEGAL.search(txt)
        if m:
            legal = [a.strip() for a in m.group(1).split(",") if a.strip()]
            click_only = set(legal) <= {"CLICK", "RESET", "UNDO"}
        if "Avatar: #" in txt:
            avatar = True
            km = _KEYS.search(txt)
            if km:
                keys = sorted(set(re.findall(r"'([A-Z]+)'", km.group(1))))
        n_ents = max(n_ents, len(_ENT.findall(txt)))
        mt = _TILE.search(txt)
        if mt:
            tile = int(mt.group(1))
        if "[hud]" in txt:
            hud = True
    return {"avatar": avatar, "click_only": click_only, "keys": keys, "many_entities": n_ents >= 12, "tile": tile, "hud": hud}


def _compress(actions: list[str]) -> str:
    """UP, UP, UP, CLICK(1,2), CLICK(3,4) -> 'UPx3, CLICKx2' (click targets are game-specific, so only the count)."""
    out: list[str] = []
    for a in actions:
        name = "CLICK" if a.startswith("CLICK") else a
        if out and out[-1][0] == name:
            out[-1][1] += 1
        else:
            out.append([name, 1])
    parts = [f"{n}x{c}" if c > 1 else n for n, c in out]
    if len(parts) > 12:
        parts = [*parts[:6], "...", *parts[-5:]]
    return ", ".join(parts)


def _action_name(rec: dict[str, Any]) -> str:
    a = str(rec.get("action", ""))
    m = re.match(r"ACTION(\d)", a)
    if m:
        aid = int(m.group(1))
        if aid == 6:
            return f"CLICK({rec.get('x')},{rec.get('y')})"
        return ACTION_NAMES.get(aid, a)
    return a


def _level_actions(game_log: Path) -> dict[int, list[str]]:
    """Actions per level index (1-based) from the per-game action log written by arc3.eval."""
    per: dict[int, list[str]] = defaultdict(list)
    if not game_log.exists():
        return per
    with open(game_log) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a truncated last line of a killed run
            lvl = int(rec.get("levels", 0)) + 1
            per[lvl].append(_action_name(rec))
    return per


def mine_transcript(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if not recs or recs[0].get("kind") != "meta":
        return []
    meta = recs[0]
    game = meta.get("game") or path.name.split(".")[0]
    if not int((meta.get("stats") or {}).get("levels_completed", 0)):
        return []
    actions_per_level = _level_actions(path.with_name(f"{game}.jsonl"))
    cards: list[dict[str, Any]] = []
    level_obs: dict[int, list[str]] = defaultdict(list)
    goal: dict[int, str] = {}
    learned: dict[int, str] = {}
    cur = 1
    for r in recs[1:]:
        k = r.get("kind")
        if k == "observation":
            txt = r.get("text") or ""
            m = re.match(r"Level (\d+)/(\d+)", txt)
            if m:
                cur = int(m.group(1))
            level_obs[cur].append(txt)
            for done in _LEVEL_DONE.finditer(txt):
                lvl, n = int(done.group(1)), int(done.group(2))
                acts = actions_per_level.get(lvl, [])
                cards.append({"game": game, "level": lvl, "actions": n, "goal": goal.get(lvl, ""), "learned": learned.get(lvl, ""),
                              "sequence": _compress(acts) if acts else "", "signature": _signature_from_observations(level_obs[lvl]),
                              "run": path.parent.name})
        elif k == "assistant":
            for code in r.get("code") or []:
                first = (code or "").strip().splitlines()[:1]
                if not first:
                    continue
                g = _GOAL.search(first[0])
                if g and g.group(1) and g.group(1).lower() not in ("unknown", "?", "tbd"):
                    goal[cur] = g.group(1)[:140]
                le = _LEARNED.search(first[0])
                if le and le.group(1):
                    learned[cur] = le.group(1)[:160]
        elif k == "tool":
            lvl = r.get("level")
            if isinstance(lvl, int) and lvl > cur:
                cur = lvl
    return cards


def transcript_outcomes(runs_dir: Path) -> dict[str, list[tuple[str, int]]]:
    """game -> [(run, levels_completed)] from the meta line of every transcript (the cheap scan that counts failures)."""
    out: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for p in sorted(runs_dir.glob("*/*.transcript.jsonl")):
        try:
            with open(p) as f:
                meta = json.loads(f.readline())
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if meta.get("kind") != "meta":
            continue
        game = meta.get("game") or p.name.split(".")[0]
        out[game].append((p.parent.name, int((meta.get("stats") or {}).get("levels_completed", 0))))
    return out


def status_of(wins: int, failures: int) -> str:
    if wins >= 2:
        return "validated"
    return "candidate" if failures <= 2 else "deprecated"


def build_library(runs_dir: Path, *, min_actions: int = 1) -> dict[str, Any]:
    raw: list[dict[str, Any]] = []
    for p in sorted(runs_dir.glob("*/*.transcript.jsonl")):
        try:
            raw.extend(mine_transcript(p))
        except Exception as e:
            print(f"[mine_skills] skip {p}: {e}", file=sys.stderr)
    outcomes = transcript_outcomes(runs_dir)
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for c in raw:
        if c["actions"] >= min_actions:
            by_key[(c["game"], c["level"])].append(c)
    skills: list[dict[str, Any]] = []
    for (game, level), cards in sorted(by_key.items()):
        best = min(cards, key=lambda c: c["actions"])
        with_goal = [c for c in cards if c["goal"]] or cards
        goal = min(with_goal, key=lambda c: c["actions"])["goal"]
        learned = next((c["learned"] for c in sorted(cards, key=lambda c: c["actions"]) if c["learned"]), "")
        sig = best["signature"]
        shape = ("click-only" if sig["click_only"] else ("avatar with keys " + "/".join(sig["keys"]) if sig["keys"] else "avatar, key map not fitted")
                 if sig["avatar"] else "keys and clicks, no avatar found")
        strategy = f"{shape}: goal '{goal or 'not stated'}'; solved in {best['actions']} actions"
        if best["sequence"]:
            strategy += f" ({best['sequence']})"
        if learned:
            strategy += f". Learned: {learned}"
        runs = sorted({c["run"] for c in cards})
        # failures: transcripts of the game that reached this level (completed level - 1) and did not complete it
        failures = sorted(r for r, done in outcomes.get(game, []) if done == level - 1 and r not in runs)
        wins = len(cards)
        skills.append({"game": game, "level": level, "wins": wins, "actions": best["actions"], "signature": sig,
                       "strategy": strategy, "runs": runs, "failures": len(failures), "failed_runs": failures,
                       "confidence": round(wins / (wins + len(failures)), 2), "last_validated": runs[-1],
                       "status": status_of(wins, len(failures))})
    return {"version": 1, "source": str(runs_dir), "n_transcripts_with_wins": len({(c['game'], c['run']) for c in raw}),
            "skills": skills}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=str(ROOT / "arc3" / "data" / "skills.json"))
    ap.add_argument("--min-actions", type=int, default=1)
    a = ap.parse_args()
    lib = build_library(Path(a.runs), min_actions=a.min_actions)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lib, indent=1))
    print(f"[mine_skills] {len(lib['skills'])} skill cards from {lib['n_transcripts_with_wins']} winning transcripts -> {out}")
    for s in lib["skills"]:
        print(f"  {s['game']} L{s['level']} {s['status']} wins={s['wins']} failures={s['failures']} conf={s['confidence']} "
              f"actions={s['actions']}: {s['strategy'][:120]}")


if __name__ == "__main__":
    main()
