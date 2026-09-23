#!/usr/bin/env python
"""Pull a Kaggle run of a Tufa-harness (TAAF / Duck) notebook and score it with our own scorer.

    .venv/bin/python scripts/pull_taaf_run.py scottmahony/arc3-taaf-q38-27b-anim taaf-q38-27b-anim-public25

Downloads the kernel's output into ``runs/<run-name>/kernel-output`` (git-ignored), reads TAAF's ``benchmark.json``
(per game: levels completed, actions per level, human baselines, TAAF's own final score), rescores every game with
``arc3.scoring.game_score`` (parity-tested against the toolkit), and writes ``runs/<run-name>/summary.json`` with the
public-25 mean and the dev / val split means from ``arc3.splits`` so it compares directly with our harness's runs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arc3.scoring import game_score  # noqa: E402
from arc3.splits import resolve  # noqa: E402


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    kernel, run_name = sys.argv[1], sys.argv[2]
    out = ROOT / "runs" / run_name
    dl = out / "kernel-output"
    dl.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    token = ROOT / ".kaggle" / "access_token"
    if token.exists() and not env.get("KAGGLE_API_TOKEN"):
        env["KAGGLE_API_TOKEN"] = token.read_text().strip()
    kaggle = str(ROOT / ".venv" / "bin" / "kaggle")
    r = subprocess.run([kaggle, "kernels", "output", kernel, "-p", str(dl), "-o"], env=env, capture_output=True, text=True, check=False)
    print((r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else "no output")
    bench = sorted(dl.rglob("benchmark.json"), key=lambda p: p.stat().st_mtime)
    if not bench:
        raise SystemExit(f"no benchmark.json under {dl}")
    data = json.loads(bench[-1].read_text())
    runs = data.get("game_runs") or data.get("runs") or []
    if not runs:
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "game_id" in v[0]:
                runs = v
                break
    dev, val = set(resolve("dev")), set(resolve("val"))
    games = []
    for g in runs:
        gid = str(g["game_id"]).split("-")[0]
        n = int(g.get("levels_completed", 0))
        apl = list(g.get("actions_per_level") or [])
        base = list(g.get("base_actions_per_level") or [])
        ours = game_score([apl[i] if i < n else None for i in range(len(base))], base) if base else 0.0
        games.append({"game_id": gid, "state": g.get("state"), "levels_completed": n, "levels_total": len(base),
                      "level_actions": apl, "baselines": base, "score": round(ours, 4),
                      "taaf_final_score": g.get("final_score"), "actions": len(g.get("history") or []),
                      "wall_s": g.get("final_wallclock_seconds"), "generated_tokens": g.get("final_generated_tokens")})

    def mean(ids: set[str]) -> float | None:
        xs = [g["score"] for g in games if g["game_id"] in ids]
        return round(sum(xs) / len(xs), 4) if xs else None

    summary = {"run_name": run_name, "kernel": kernel, "source": str(bench[-1].relative_to(dl)), "games": len(games),
               "score_all": mean({g["game_id"] for g in games}), "score_dev": mean(dev), "score_val": mean(val),
               "levels_completed": sum(g["levels_completed"] for g in games),
               "levels_total": sum(g["levels_total"] for g in games), "results": games}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{run_name}: all {summary['score_all']}  dev {summary['score_dev']}  val {summary['score_val']}  "
          f"levels {summary['levels_completed']}/{summary['levels_total']}")
    for g in sorted(games, key=lambda g: -g["score"]):
        if g["levels_completed"]:
            print(f"  {g['game_id']}: {g['levels_completed']}/{g['levels_total']} score {g['score']:.2f} "
                  f"(taaf {g['taaf_final_score']}) actions {g['level_actions'][:g['levels_completed']]} base {g['baselines'][:g['levels_completed']]}")


if __name__ == "__main__":
    main()
