#!/usr/bin/env python
"""Push Kaggle runs in priority order as GPU slots free, then pull and score each one when it finishes.

    .venv/bin/python scripts/kaggle_queue.py <queue.json>          # runs until every item is done or skipped

The queue file is a JSON list of items, in priority order:

    {"name": "exp045", "folder": "<built folder>", "kernel": "scottmahony/arc3-taaf-gate", "run": "exp045-gate",
     "kind": "full" | "stress", "after": "<name>" (optional: push only once that item was pushed),
     "requires_ok": "<run>" (optional: push only if runs/<run>/summary.json is a clean stress test; skipped if not)}

Every 2 minutes it tries to push the first eligible item with scripts/push_eval.py (Kaggle refuses pushes while both
GPU slots are busy, so a failed push just waits); every 5 minutes it checks the pushed items' status, and a finished
one is pulled: full runs with scripts/pull_taaf_run.py and scripts/compare_to_harvest.py, stress tests by
downloading kv_stress.json to runs/<run>/summary.json. Progress lines go to stdout; state is kept next to the queue
file (<queue>.state.json), so a restarted runner resumes where it stopped.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
KAGGLE = str(ROOT / ".venv" / "bin" / "kaggle")


def now() -> str:
    return datetime.now(UTC).strftime("%H:%M")


def run(cmd: list[str], timeout: int = 1800) -> str:
    env = dict(os.environ)
    token = ROOT / ".kaggle" / "access_token"
    if token.exists() and not env.get("KAGGLE_API_TOKEN"):
        env["KAGGLE_API_TOKEN"] = token.read_text().strip()
    try:
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout, check=False)
        return (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return "timeout"


def stress_ok(run_name: str) -> bool | None:
    """True for a clean stress test, False for a failed one, None while it has no summary."""
    path = ROOT / "runs" / run_name / "summary.json"
    if not path.exists():
        return None
    s = json.loads(path.read_text())
    if s.get("status") and s["status"] != "ok":
        return False
    return int(s.get("ok") or 0) > 0 and int(s.get("errors") or 0) == 0


def pull(item: dict) -> str:
    name, kernel, run_name = item["name"], item["kernel"], item["run"]
    if item.get("kind") == "stress":
        out_dir = ROOT / "runs" / run_name / "kernel-output"
        out_dir.mkdir(parents=True, exist_ok=True)
        run([KAGGLE, "kernels", "output", kernel, "-p", str(out_dir), "--file-pattern", r"kv_stress\.json|\.log$",
             "-q", "-o"])
        src = out_dir / "kv_stress.json"
        logs = "\n".join(p.read_text(errors="replace") for p in out_dir.glob("*.log"))
        oom = re.findall(r".*(?:OutOfMemoryError|out of memory|Traceback).*", logs)[:3]
        if src.exists():
            s = json.loads(src.read_text())
            s.pop("samples", None)
            summary = {"run_name": run_name, "kernel": kernel, "status": "ok", **s}
        else:
            summary = {"run_name": run_name, "kernel": kernel, "status": "no_output", "log_errors": oom}
        (ROOT / "runs" / run_name / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        return f"{name}: {summary}" + (f"\n  log: {oom}" if oom else "")
    text = run([PY, "scripts/pull_taaf_run.py", kernel, run_name], timeout=3600)
    lines = text.splitlines()[-34:]
    cmp_text = run([PY, "scripts/compare_to_harvest.py", f"runs/{run_name}/summary.json"])
    return "\n".join(lines + cmp_text.splitlines()[:6])


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    qpath = Path(sys.argv[1])
    spath = qpath.with_suffix(".state.json")
    state = json.loads(spath.read_text()) if spath.exists() else {}
    last_status = 0.0
    while True:
        items = json.loads(qpath.read_text())  # re-read: the queue may be edited while this runs
        for it in items:
            state.setdefault(it["name"], {})
        pending = [it for it in items if not state[it["name"]].get("done") and not state[it["name"]].get("skipped")]
        if not pending:
            print(f"{now()} queue empty", flush=True)
            return
        if time.time() - last_status >= 300:
            last_status = time.time()
            for it in pending:
                st = state[it["name"]]
                if not st.get("pushed"):
                    continue
                status = run([KAGGLE, "kernels", "status", it["kernel"]]).splitlines()[-1:] or [""]
                if any(k in status[0] for k in ("COMPLETE", "ERROR", "CANCEL")):
                    print(f"{now()} {it['name']}: {status[0]}", flush=True)
                    print(pull(it), flush=True)
                    st["done"] = now()
        for it in pending:
            st = state[it["name"]]
            if st.get("pushed"):
                continue
            if it.get("after") and not state.get(it["after"], {}).get("pushed"):
                continue
            if it.get("requires_ok"):
                ok = stress_ok(it["requires_ok"])
                if ok is None:
                    continue
                if ok is False:
                    st["skipped"] = f"{it['requires_ok']} did not pass"
                    print(f"{now()} {it['name']}: skipped ({st['skipped']})", flush=True)
                    continue
            out = run([PY, "scripts/push_eval.py", it["folder"]], timeout=900)
            if "successfully pushed" in out:
                st["pushed"] = now()
                print(f"{now()} {it['name']}: pushed ({out.splitlines()[-2] if len(out.splitlines()) > 1 else out})",
                      flush=True)
            break  # one push attempt per cycle, in priority order
        spath.write_text(json.dumps(state, indent=1))
        time.sleep(120)


if __name__ == "__main__":
    main()
