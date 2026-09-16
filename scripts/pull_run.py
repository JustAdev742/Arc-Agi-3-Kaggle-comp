"""Download a finished Kaggle evaluation kernel's output and file it under runs/<run_name>/.

    .venv/bin/python scripts/pull_run.py scottmahony/arc3-eval-dev-b kaggle-repl-dev-007

Copies summary.json, eval_result.json, per-game action logs and transcripts, prints the per-game table and
the tail of the kernel log. Needs the Kaggle token in .kaggle/access_token or KAGGLE_API_TOKEN.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    kernel, run_name = sys.argv[1], sys.argv[2]
    token_file = ROOT / ".kaggle" / "access_token"
    env = dict(os.environ)
    if token_file.exists() and not env.get("KAGGLE_API_TOKEN"):
        env["KAGGLE_API_TOKEN"] = token_file.read_text().strip()
    out = ROOT / "runs" / "_kaggle_output" / kernel.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    kaggle = ROOT / ".venv" / "bin" / "kaggle"
    r = subprocess.run([str(kaggle), "kernels", "output", kernel, "-p", str(out)], env=env, capture_output=True, text=True, check=False)
    print(r.stdout[-2000:], r.stderr[-1000:])
    dest = ROOT / "runs" / run_name
    dest.mkdir(parents=True, exist_ok=True)
    src_runs = out / "runs" / run_name
    copied = []
    for f in [*src_runs.glob("*")] if src_runs.exists() else []:
        if f.suffix in (".json", ".jsonl") and f.stat().st_size < 20_000_000:
            shutil.copy(f, dest / f.name)
            copied.append(f.name)
    for name in ("eval_result.json",):
        if (out / name).exists():
            shutil.copy(out / name, dest / name)
            copied.append(name)
    log = next(iter(out.glob("*.log")), None)
    print("copied:", sorted(copied))
    s = dest / "summary.json"
    if s.exists():
        d = json.loads(s.read_text())
        print(f"\n== {d.get('run_name')} score={d.get('score'):.3f} dev={d.get('score_dev')} val={d.get('score_val')} "
              f"levels={d.get('levels_completed')}/{d.get('levels_total')} actions={d.get('actions')} wall={d.get('wall_s')}s failures={d.get('failures')}")
        for g in d.get("results", []):
            st = g.get("agent_stats", {})
            print(f"   {g['game_id']:5s} {g['score']:6.2f}  L{g['levels_completed']}/{g['win_levels']} acts={g['actions']:4d} "
                  f"calls={st.get('model_calls')} errors={st.get('model_errors')} p50={st.get('model_latency_p50_s')}s "
                  f"wm={st.get('wm_matched')}/{st.get('wm_checked')} rules_fits={st.get('rules_fits')} {g.get('failure') or ''}")
    else:
        print("no summary.json found under", src_runs)
    if log is not None:
        print("\n--- kernel log tail ---")
        print(log.read_text()[-3000:])


if __name__ == "__main__":
    main()
