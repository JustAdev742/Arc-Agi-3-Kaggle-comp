#!/usr/bin/env python
"""Goal-predicate recall on solved levels (code-only, no model).

For every (game, level) card in arc3/data/skills.json, replay the recorded winning run's actions through the local
engine with the real REPL agent (mock model that just replays the actions), so the tracker and the level archive see
exactly what they see in production (including the observed terminal frame), and report whether
dsl.goal_predicates() returns a win condition consistent with the completed level(s) and which one comes first.

Usage: .venv/bin/python scripts/goal_probe.py [--games vc33,ls20] [--max-levels 2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arc3.agents import get  # noqa: E402
from arc3.agents.base import AgentContext  # noqa: E402
from arc3.env import LocalEnv, make_arcade  # noqa: E402
from arc3.llm import MockClient  # noqa: E402


def replay(game: str, run: str, target_level: int, arc) -> list[dict]:
    with open(ROOT / "runs" / run / f"{game}.jsonl") as f:
        recs = [json.loads(line) for line in f]
    acts = [f"('CLICK', {r['x']}, {r['y']})" if r["action"] == "ACTION6" else repr(r["action"]) for r in recs]
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
    try:
        frame = env.frame
        seen = 0
        for _ in range(len(recs) + 5):
            if not acts and seen == frame.levels_completed:
                break
            if agent.is_done(frame) or frame.levels_completed >= target_level:
                break
            a = agent.act(frame)
            before = frame
            frame = env.step(a)
            agent.observe(a, before, frame)
            if frame.levels_completed > seen:
                seen = frame.levels_completed
                arch = agent.level_archive[-1][0] if agent.level_archive else []
                from arc3 import dsl
                preds = dsl.goal_predicates(agent.level_archive)
                out.append({"game": game, "level": seen, "frames": len(arch), "n_goals": len(preds),
                            "goals": [g["goal"] for g in preds[:4]], "actions": before.level_step + 1})
    finally:
        agent.close()
        env.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="")
    ap.add_argument("--max-levels", type=int, default=3)
    a = ap.parse_args()
    skills = json.loads((ROOT / "arc3" / "data" / "skills.json").read_text())["skills"]
    want = set(a.games.split(",")) if a.games else None
    arc = make_arcade("environment_files")
    by_game: dict[str, tuple[int, str]] = {}
    for s in skills:
        if want and s["game"] not in want:
            continue
        lvl = min(int(s["level"]), a.max_levels)
        if s["game"] not in by_game or lvl > by_game[s["game"]][0]:
            by_game[s["game"]] = (lvl, s["runs"][-1])
    rows = []
    for game, (lvl, run) in sorted(by_game.items()):
        try:
            rows.extend(replay(game, run, lvl, arc))
        except Exception as e:
            print(f"{game}: replay failed: {e}")
    hit = sum(1 for r in rows if r["n_goals"])
    print(f"\nsolved levels replayed: {len(rows)}; with >=1 consistent goal predicate: {hit}")
    for r in rows:
        print(f"  {r['game']} L{r['level']} frames={r['frames']} actions={r['actions']} goals={r['n_goals']}: {'; '.join(r['goals'])}")


if __name__ == "__main__":
    main()
