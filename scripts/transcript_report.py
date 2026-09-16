"""Summarise the per-game transcripts of a run (``runs/<name>/<game>.transcript.jsonl``) for post-mortems.

Per game: turns, model calls, actions, levels, tool errors, which helpers the model used (auto_rules, plan_rules,
set_model, move_model, plan_to_entity, verify_model), the coverage numbers auto_rules reported, and the last
notes. Use it right after ``kaggle kernels output`` to see what the model did without reading every transcript.

    .venv/bin/python scripts/transcript_report.py runs/kaggle-repl-dev-007 [--game ls20] [--full]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

HELPERS = ("auto_rules", "plan_rules", "rules_predictor", "goal_candidates", "set_model", "set_models", "verify_model",
           "move_model", "plan_to_entity", "plan_to", "ents", "events", "describe_events", "avatar", "act", "click", "note")


def load(path: Path) -> tuple[dict, list[dict]]:
    meta, recs = {}, []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "meta":
                meta = r
            else:
                recs.append(r)
    return meta, recs


def report_game(path: Path, full: bool = False) -> dict:
    meta, recs = load(path)
    game = meta.get("game", path.stem.split(".")[0])
    stats = meta.get("stats", {})
    used: Counter = Counter()
    coverages: list[str] = []
    errors = 0
    levels = set()
    pred_fail = 0
    stagnation = 0
    council_rounds = 0
    summary_effort = None
    for r in recs:
        if r["kind"] == "assistant":
            for code in r.get("code", []):
                for h in HELPERS:
                    if re.search(rf"\b{h}\(", code):
                        used[h] += 1
        elif r["kind"] == "observation":
            if r.get("effort") not in (None, "low"):
                summary_effort = r.get("effort")
            if "STAGNATION" in (r.get("text") or ""):
                stagnation += 1
        elif r["kind"] == "council":
            council_rounds += 1
        elif r["kind"] == "tool":
            out = r.get("output", "") or ""
            if r.get("error"):
                errors += 1
            if r.get("level") is not None:
                levels.add(r["level"])
            m = re.search(r"'coverage': ([0-9.]+)", out)
            if m:
                coverages.append(m.group(1))
            pred_fail += len(re.findall(r"'pred_ok': False", out))
    summary = {
        "game": game, "turns": stats.get("turns"), "calls": stats.get("model_calls"), "actions": stats.get("actions_model"),
        "fallback": stats.get("actions_fallback"), "tool_errors": errors, "levels_seen": sorted(levels)[-1] if levels else None,
        "p50_s": stats.get("model_latency_p50_s"), "used": dict(used), "coverage_reports": coverages[:6], "pred_fail": pred_fail,
        "stagnation_notices": stagnation, "council_rounds": council_rounds, "raised_effort_seen": summary_effort,
        "notes": (meta.get("notes") or [])[-4:],
    }
    if full:
        summary["records"] = recs
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--game", default=None)
    p.add_argument("--full", action="store_true", help="print every assistant code cell and tool output head for one game")
    a = p.parse_args()
    run = Path(a.run_dir)
    files = sorted(run.glob("*.transcript.jsonl"))
    if a.game:
        files = [f for f in files if f.name.startswith(a.game)]
    if not files:
        print(f"no transcripts under {run}")
        return
    rows = []
    for f in files:
        s = report_game(f, full=a.full)
        rows.append(s)
        used = " ".join(f"{k}:{v}" for k, v in sorted(s["used"].items(), key=lambda kv: -kv[1]) if k not in ("act", "click", "ents", "events"))
        print(f"{s['game']}: turns {s['turns']} calls {s['calls']} actions {s['actions']} (+{s['fallback']} fallback) errors {s['tool_errors']} "
              f"level {s['levels_seen']} p50 {s['p50_s']}s pred_fail {s['pred_fail']} stagnation {s['stagnation_notices']}"
              + (f" council_rounds {s['council_rounds']}" if s['council_rounds'] else ""))
        print(f"    helpers: {used or '-'}")
        if s["coverage_reports"]:
            print(f"    coverage: {', '.join(s['coverage_reports'])}")
        for n in s["notes"]:
            print(f"    note: {str(n)[:160]}")
        if a.full:
            for r in s["records"]:
                if r["kind"] == "observation":
                    print("  === turn " + str(r.get("turn")) + " observation (effort " + str(r.get("effort")) + "): " + (r.get("text") or "")[:700].replace("\n", "\n      "))
                elif r["kind"] == "council":
                    print("  === council round (" + str(r.get("reason")) + ", " + str(r.get("round_s")) + " s): " + str(r.get("reports"))[:600])
                elif r["kind"] == "assistant":
                    for code in r.get("code", []):
                        print("  >>> " + code.strip().replace("\n", "\n      ")[:1200])
                    if r.get("content") and not r.get("code"):
                        print("  (no code) " + r["content"][:300].replace("\n", " "))
                else:
                    print("  <<< " + (r.get("output") or "")[:600].replace("\n", "\n      "))
    if not a.game:
        n_auto = sum(1 for s in rows if s["used"].get("auto_rules"))
        n_plan = sum(1 for s in rows if s["used"].get("plan_rules") or s["used"].get("plan_to_entity"))
        n_model = sum(1 for s in rows if s["used"].get("set_model"))
        print(f"\n{len(rows)} games: auto_rules used in {n_auto}, a planner in {n_plan}, set_model in {n_model}")


if __name__ == "__main__":
    main()
