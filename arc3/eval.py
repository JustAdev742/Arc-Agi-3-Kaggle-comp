"""Local evaluation harness.

Runs any registered agent on a fixed list of games with fixed seeds and a per-game
wall-clock budget, scores every game with the toolkit's own scorer, and writes a
self-describing run record to ``runs/<run_name>/``:

    summary.json          run metadata + per-game results + aggregate scores
    <game>.jsonl          one line per action (step, action, state, levels, diff size)

Every number in docs/research_log.md must point at one of these directories.
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


from . import __version__
from .env import Action, LocalEnv, make_arcade
from .perception import diff as grid_diff
from .perception import grid_hash
from .scoring import total_score
from .splits import DEV_GAMES, VAL_GAMES, resolve

log = logging.getLogger("arc3.eval")


@dataclass
class GameResult:
    game_id: str
    version: str
    seed: int
    score: float = 0.0
    levels_completed: int = 0
    win_levels: int = 0
    actions: int = 0
    resets: int = 0
    level_actions: list[int] = field(default_factory=list)
    level_scores: list[float] = field(default_factory=list)
    baselines: list[int] = field(default_factory=list)
    wall_s: float = 0.0
    time_budget_s: float = 0.0
    state: str = "NOT_PLAYED"
    failure: str = ""  # solved | timeout | action_cap | crash | gave_up | no_progress
    error: Optional[str] = None
    agent_stats: dict[str, Any] = field(default_factory=dict)
    seconds_per_action: float = 0.0
    tags: list[str] = field(default_factory=list)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _score_from_card(card, game_id: str) -> dict[str, Any]:
    env_list = card.find_environment(game_id) if card is not None else None
    if env_list is None or not env_list.runs:
        return {}
    best = max(env_list.runs, key=lambda r: (r.levels_completed, r.score))
    return {
        "score": float(best.score),
        "levels_completed": int(best.levels_completed),
        "actions": int(best.actions),
        "resets": int(best.resets or 0),
        "level_actions": list(best.level_actions or []),
        "level_scores": [float(s) for s in (best.level_scores or [])],
        "baselines": list(best.level_baseline_actions or []),
        "state": best.state.name if best.state else "NOT_PLAYED",
        "message": best.message,
    }


def play_game(
    agent_name: str,
    game_id: str,
    *,
    seed: int = 0,
    time_budget_s: float = 600.0,
    max_actions: int = 5000,
    out_dir: Optional[Path] = None,
    config: Optional[dict[str, Any]] = None,
    environments_dir: str = "environment_files",
    record_frames: bool = False,
) -> GameResult:
    """Play one game to completion / timeout / action cap and score it."""
    from .agents import get as get_agent
    from .agents.base import AgentContext

    arc = make_arcade(environments_dir)
    env = LocalEnv(arc, game_id, seed=seed, tags=["arc3-eval", agent_name])
    version = env.info.game_id
    res = GameResult(game_id=game_id, version=version, seed=seed, time_budget_s=time_budget_s,
                     win_levels=env.frame.win_levels, baselines=list(env.baselines), tags=list(env.info.tags or []))
    deadline = time.time() + time_budget_s
    ctx = AgentContext(game_id=game_id, seed=seed, deadline=deadline, max_actions=max_actions,
                       baselines=list(env.baselines), config=dict(config or {}),
                       log=logging.getLogger(f"arc3.agent.{game_id}"))
    jsonl = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl = open(out_dir / f"{game_id}.jsonl", "w")
    t0 = time.time()
    agent = None
    try:
        agent = get_agent(agent_name)(ctx)
        frame = env.frame
        last_progress_step = 0
        while True:
            if agent.is_done(frame):
                res.failure = "solved" if frame.done else "gave_up"
                break
            if env.step_count >= max_actions:
                res.failure = "action_cap"
                break
            if time.time() >= deadline:
                res.failure = "timeout"
                break
            action = agent.act(frame)
            if not isinstance(action, Action):
                raise TypeError(f"agent returned {type(action)!r}, expected Action")
            before = frame
            frame = env.step(action)
            agent.observe(action, before, frame)
            if frame.levels_completed > before.levels_completed:
                last_progress_step = env.step_count
                log.info("%s level %d done at action %d (%.0fs)", game_id, frame.levels_completed, env.step_count, time.time() - t0)
            if jsonl is not None:
                d = grid_diff(before.grid, frame.grid)
                rec = {"step": env.step_count, "t": round(time.time() - t0, 3), "action": action.action.name,
                       "x": action.x, "y": action.y, "state": frame.state.name, "levels": frame.levels_completed,
                       "changed": d.changed, "hash": grid_hash(frame.grid)}
                if record_frames:
                    rec["grid"] = frame.grid.tolist()
                if action.reasoning is not None:
                    rec["reasoning"] = action.reasoning if isinstance(action.reasoning, (str, dict)) else str(action.reasoning)
                jsonl.write(json.dumps(rec) + "\n")
        if res.failure == "solved" and not frame.done:
            res.failure = "gave_up"
        res.state = frame.state.name
    except Exception as e:  # crash isolation: the run record survives, the game scores what it scored
        res.failure = "crash"
        res.error = "".join(traceback.format_exception(e))[-4000:]
        log.exception("%s crashed", game_id)
    finally:
        if jsonl is not None:
            jsonl.close()
        res.wall_s = round(time.time() - t0, 3)
        try:
            card = env.close()
            res.__dict__.update({k: v for k, v in _score_from_card(card, game_id).items() if k != "message"})
        except Exception as e:  # pragma: no cover
            log.exception("scorecard failed for %s: %s", game_id, e)
        if agent is not None:
            try:
                res.agent_stats = agent.stats()
            except Exception:  # pragma: no cover
                pass
            try:
                agent.close()
            except Exception:  # pragma: no cover
                pass
        res.seconds_per_action = round(res.wall_s / res.actions, 4) if res.actions else 0.0
        if res.failure == "" and res.state != "WIN":
            res.failure = "no_progress"
    return res


def _worker(args: dict[str, Any]) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out_dir = Path(args["out_dir"]) if args.get("out_dir") else None
    r = play_game(args["agent"], args["game_id"], seed=args["seed"], time_budget_s=args["time_budget_s"],
                  max_actions=args["max_actions"], out_dir=out_dir, config=args.get("config"),
                  environments_dir=args.get("environments_dir", "environment_files"),
                  record_frames=args.get("record_frames", False))
    return asdict(r)


def run_eval(
    agent_name: str,
    split: str = "dev",
    *,
    seed: int = 0,
    time_budget_s: float = 600.0,
    max_actions: int = 5000,
    workers: int = 1,
    run_name: Optional[str] = None,
    runs_dir: Path | str = "runs",
    config: Optional[dict[str, Any]] = None,
    environments_dir: str = "environment_files",
    record_frames: bool = False,
    note: str = "",
) -> dict[str, Any]:
    games = resolve(split)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    run_name = run_name or f"{stamp}-{agent_name}-{split}-s{seed}"
    out_dir = Path(runs_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [dict(agent=agent_name, game_id=g, seed=seed, time_budget_s=time_budget_s, max_actions=max_actions,
                 out_dir=str(out_dir), config=config or {}, environments_dir=environments_dir,
                 record_frames=record_frames) for g in games]
    t0 = time.time()
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for j in jobs:
            results.append(_worker(j))
            _print_result(results[-1])
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_worker, j): j["game_id"] for j in jobs}
            for f in as_completed(futs):
                results.append(f.result())
                _print_result(results[-1])
    results.sort(key=lambda r: games.index(r["game_id"]))
    by_game = {r["game_id"]: r["score"] for r in results}
    dev = [by_game[g] for g in games if g in DEV_GAMES]
    val = [by_game[g] for g in games if g in VAL_GAMES]
    summary = {
        "run_name": run_name,
        "agent": agent_name,
        "split": split,
        "games": games,
        "seed": seed,
        "time_budget_s": time_budget_s,
        "max_actions": max_actions,
        "config": config or {},
        "note": note,
        "harness_commit": git_commit(),
        "arc3_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "started": stamp,
        "wall_s": round(time.time() - t0, 1),
        "score": total_score(by_game.values()),
        "score_dev": total_score(dev) if dev else None,
        "score_val": total_score(val) if val else None,
        "levels_completed": sum(r["levels_completed"] for r in results),
        "levels_total": sum(r["win_levels"] for r in results),
        "games_solved": sum(1 for r in results if r["state"] == "WIN"),
        "actions": sum(r["actions"] for r in results),
        "failures": _count([r["failure"] for r in results]),
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(format_summary(summary))
    return summary


def _count(xs: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


def _print_result(r: dict[str, Any]) -> None:
    print(f"  {r['game_id']:6} score={r['score']:6.2f} levels={r['levels_completed']}/{r['win_levels']} "
          f"actions={r['actions']:5d} wall={r['wall_s']:7.1f}s {r['failure']}", flush=True)


def format_summary(s: dict[str, Any]) -> str:
    lines = [f"== {s['run_name']}  agent={s['agent']} split={s['split']} seed={s['seed']} commit={s['harness_commit']}",
             f"   score={s['score']:.3f}  dev={s['score_dev']}  val={s['score_val']}  "
             f"levels={s['levels_completed']}/{s['levels_total']}  solved={s['games_solved']}/{len(s['games'])}  "
             f"actions={s['actions']}  wall={s['wall_s']}s  failures={s['failures']}"]
    for r in s["results"]:
        la = ",".join(str(a) for a in r["level_actions"][: r["levels_completed"]])
        lines.append(f"   {r['game_id']:6} {r['score']:6.2f}  L{r['levels_completed']}/{r['win_levels']}  "
                     f"acts={r['actions']:5d} [{la}]  base={r['baselines']}  {r['wall_s']:6.1f}s  {r['failure']}")
    return "\n".join(lines)
