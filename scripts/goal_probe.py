#!/usr/bin/env python
"""Goal-predicate recall on solved levels (code-only, no model).

For every (game, level) card in arc3/data/skills.json, replay the recorded winning run's actions through the local
engine with the real REPL agent (mock model that just replays the actions), so the tracker and the level archive see
exactly what they see in production (including the observed terminal frame), and report whether
dsl.goal_predicates() returns a win condition consistent with the completed level(s) and which one comes first.

Second gate (road-to-100 item 4, 2026-09-17): on levels 2+ the harness ranks the level-1 candidates by code-computed
distance and falsifies the ones that come true without a win (dsl.goal_progress). For every replayed level >= 2 the
report says how many candidates were live one frame before the win, how many level 1 candidates the level falsified,
and the rank of the eventual winner (a predicate still consistent after the level) among the live ones: rank 1 means
goal_probe() would have planned the right goal first.

Human recordings (``--recordings data/human``, the ARC Prize dataset files: one line per action with integer ids and the
frame): the same replay and gates, with every level the human completed as a case, plus a determinism check of the
local engine against the recorded frames (mismatching frames are counted, not fatal).

Usage: .venv/bin/python scripts/goal_probe.py [--games vc33,ls20] [--max-levels 2] [--recordings data/human]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from arc3.agents import get  # noqa: E402
from arc3.agents.base import AgentContext  # noqa: E402
from arc3.env import LocalEnv, make_arcade  # noqa: E402
from arc3.llm import MockClient  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from human_replays import action_name  # noqa: E402


def recording_actions(path: Path) -> tuple[str, list[str], list]:
    """(game, act() literals, recorded frames after each action) from an ARC Prize recording file."""
    game = None
    acts: list[str] = []
    frames: list = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line).get("data") or {}
            if "action_input" not in d:
                continue
            game = game or str(d.get("game_id", path.name)).split("-")[0]
            name = action_name(d["action_input"])
            if name is None or name == "RESET":
                continue  # the agent's harness sends the opening RESET itself
            acts.append(f"('CLICK', {name[6:-1]})" if name.startswith("CLICK(") else repr(name))
            frames.append(d.get("frame"))
    return game or path.name[:4], acts, frames


def replay(game: str, run: str, target_level: int, arc) -> list[dict]:
    with open(ROOT / "runs" / run / f"{game}.jsonl") as f:
        recs = [json.loads(line) for line in f]
    acts = [f"('CLICK', {r['x']}, {r['y']})" if r["action"] == "ACTION6" else repr(r["action"]) for r in recs]
    return replay_actions(game, acts, target_level, arc)


def replay_actions(game: str, acts: list[str], target_level: int, arc, recorded_frames: list | None = None) -> list[dict]:
    recs = list(acts)
    acts = list(acts)
    frame_i = 0
    mismatches = 0
    out: list[dict] = []

    def script(messages):
        last = messages[-1]
        text = last.get("content") if isinstance(last.get("content"), str) else ""
        if "Consolidation step" in text or last.get("role") == "tool":
            return MockClient.say("done")
        if acts:
            batch = [acts.pop(0) for _ in range(min(6, len(acts)))]
            return MockClient.tool("act(" + ", ".join(batch) + ")")
        return MockClient.say("nothing left")

    env = LocalEnv(arc, game)
    ctx = AgentContext(game_id=game, deadline=time.time() + 900,
                       config={"client": MockClient(script), "image": False, "level_consolidation": False})
    agent = get("repl")(ctx)
    from arc3 import dsl
    try:
        frame = env.frame
        seen = 0
        rows_before = None  # goal_progress rows at the frame before the level-completing action (levels >= 2)
        for _ in range(len(recs) + 5):
            if not acts and seen == frame.levels_completed:
                break
            if agent.is_done(frame) or frame.levels_completed >= target_level:
                break
            if agent.goal_info and agent.tracker.frames:
                av = agent.tracker.avatar()
                rows_before = dsl.goal_progress(agent.goal_info, agent.tracker.compound_frames(),
                                                avatar_id=int(av["id"]) if av else None)
            a = agent.act(frame)
            before = frame
            frame = env.step(a)
            agent.observe(a, before, frame)
            if recorded_frames is not None and a.action.name != "RESET":
                rec = recorded_frames[frame_i] if frame_i < len(recorded_frames) else None
                frame_i += 1
                if rec and not np.array_equal(np.asarray(rec[-1], dtype=np.int16), frame.grid):
                    mismatches += 1
            if frame.levels_completed > seen:
                seen = frame.levels_completed
                arch = agent.level_archive[-1][0] if agent.level_archive else []
                preds = dsl.goal_predicates(agent.level_archive)
                row = {"game": game, "level": seen, "frames": len(arch), "n_goals": len(preds),
                       "goals": [g["goal"] for g in preds[:4]], "actions": before.level_step + 1}
                if rows_before is not None:
                    live = [r["goal"] for r in rows_before if not r["falsified"]]
                    winners = {g["goal"] for g in preds}
                    rank = next((i + 1 for i, g in enumerate(live) if g in winners), None)
                    row.update({"candidates_before": len(rows_before), "live_before_win": len(live),
                                "falsified": sum(1 for r in rows_before if r["falsified"]), "winner_rank": rank})
                if recorded_frames is not None:
                    row["frame_mismatches"] = mismatches
                out.append(row)
                rows_before = None
    finally:
        agent.close()
        env.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="")
    ap.add_argument("--max-levels", type=int, default=3)
    ap.add_argument("--recordings", default="", help="folder of ARC Prize recording files (or one file) to replay instead of runs/")
    a = ap.parse_args()
    want = set(a.games.split(",")) if a.games else None
    arc = make_arcade("environment_files")
    rows = []
    if a.recordings:
        root = Path(a.recordings)
        files = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
        for p in files:
            game, acts, frames = recording_actions(p)
            if want and game not in want:
                continue
            try:
                rows.extend(replay_actions(game, acts, a.max_levels, arc, recorded_frames=frames))
            except Exception as e:
                print(f"{p.name}: replay failed: {e}")
    else:
        skills = json.loads((ROOT / "arc3" / "data" / "skills.json").read_text())["skills"]
        by_game: dict[str, tuple[int, str]] = {}
        for s in skills:
            if want and s["game"] not in want:
                continue
            lvl = min(int(s["level"]), a.max_levels)
            if s["game"] not in by_game or lvl > by_game[s["game"]][0]:
                by_game[s["game"]] = (lvl, s["runs"][-1])
        for game, (lvl, run) in sorted(by_game.items()):
            try:
                rows.extend(replay(game, run, lvl, arc))
            except Exception as e:
                print(f"{game}: replay failed: {e}")
    hit = sum(1 for r in rows if r["n_goals"])
    print(f"\nsolved levels replayed: {len(rows)}; with >=1 consistent goal predicate: {hit}")
    for r in rows:
        print(f"  {r['game']} L{r['level']} frames={r['frames']} actions={r['actions']} goals={r['n_goals']}: {'; '.join(r['goals'])}")
    later = [r for r in rows if "winner_rank" in r]
    if later:
        first = sum(1 for r in later if r["winner_rank"] == 1)
        print(f"\nlevels >= 2 with level-1 candidates: {len(later)}; winner was the cheapest live hypothesis before the win: {first}")
        for r in later:
            print(f"  {r['game']} L{r['level']}: candidates {r['candidates_before']}, live before win {r['live_before_win']}, "
                  f"falsified by this level {r['falsified']}, winner rank {r['winner_rank']}")
    if any("frame_mismatches" in r for r in rows):
        print(f"\nengine vs recorded frames: {sum(r.get('frame_mismatches', 0) for r in rows if 'frame_mismatches' in r)} mismatching frames "
              f"over {len(rows)} completed levels (cumulative per recording)")


if __name__ == "__main__":
    main()
