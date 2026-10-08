"""Push a built evaluation notebook folder to Kaggle on the RTX PRO 6000.

    .venv/bin/python scripts/push_eval.py <folder with eval.ipynb + kernel-metadata.json>

Always passes ``--accelerator NvidiaRtxPro6000``: without it Kaggle places the kernel on a Tesla T4 (15 GB, SM75),
the 27B model cannot load, and the run burns ten minutes of quota before falling back (exp-010b/exp-011 v1,
2026-09-16). Prints the pushed version and the kernel's status right after.

Pushes through the SDK (the CLI's output drops it) to record the push's ``kernel_session_id`` in
``<folder>/sessions.jsonl``: the only handle scripts/kaggle_cancel.py has on a session that sits QUEUED for hours
(2026-10-08: two sessions waited 10 h+ for a machine and blocked the 2-session limit).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCELERATOR = "NvidiaRtxPro6000"


def _last_run(api, ref: str) -> str | None:
    """The kernel's lastRunTime as Kaggle lists it (None if it does not exist yet or the listing fails)."""
    owner, slug = ref.split("/", 1)
    try:
        for k in api.kernels_list(user=owner, search=slug, page_size=20) or []:
            if getattr(k, "ref", "") == ref:
                return str(getattr(k, "last_run_time", None) or getattr(k, "lastRunTime", None))
    except Exception:
        return None
    return None


def _log_attempt(folder: Path, ref: str, resp, error: str) -> None:
    row = {"time": time.time(), "kernel": ref, "error": error,
           "version": getattr(resp, "version_number", None), "session": getattr(resp, "kernel_session_id", None)}
    with (folder / "push_attempts.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    folder = Path(sys.argv[1])
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    nb = json.loads((folder / meta["code_file"]).read_text())
    acc = nb.get("metadata", {}).get("kaggle", {}).get("accelerator")
    if acc != "nvidiaRtxPro6000":
        raise SystemExit(f"notebook metadata accelerator is {acc!r}, expected 'nvidiaRtxPro6000'; rebuild it")
    env = dict(os.environ)
    token = ROOT / ".kaggle" / "access_token"
    if token.exists() and not env.get("KAGGLE_API_TOKEN"):
        env["KAGGLE_API_TOKEN"] = token.read_text().strip()
    kaggle = str(ROOT / ".venv" / "bin" / "kaggle")
    os.environ.update({k: v for k, v in env.items() if k == "KAGGLE_API_TOKEN"})
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    before = _last_run(api, meta["id"])
    try:
        resp = api.kernels_push(str(folder), None, ACCELERATOR)
        error = resp.error if resp is not None else "no response"
    except Exception as exc:  # the CLI prints the same refusal text ("Maximum batch GPU session count ...")
        resp, error = None, str(exc)
    _log_attempt(folder, meta["id"], resp, error)
    if error:
        # 2026-10-08: a push Kaggle accepted came back as a client-side error, and the retry loop never saw it
        # succeed. Trust the kernel's last-run time over the reply: if it moved, the push went through.
        time.sleep(10)
        after = _last_run(api, meta["id"])
        if after and after != before:
            print(f"Kernel push reported an error ({error}) but the kernel's last run moved {before} -> {after}: "
                  "successfully pushed (confirmed on Kaggle; session id unknown)")
            return
        print(f"Kernel push error: {error}")
        sys.exit(1)
    print(f"Kernel version {resp.version_number} successfully pushed.  Please check progress at {resp.url} "
          f"(session {resp.kernel_session_id})")
    with (folder / "sessions.jsonl").open("a") as f:
        f.write(json.dumps({"kernel": meta["id"], "version": resp.version_number,
                            "kernel_session_id": resp.kernel_session_id, "pushed": time.time()}) + "\n")
    time.sleep(5)
    s = subprocess.run([kaggle, "kernels", "status", meta["id"]], env=env, capture_output=True, text=True, check=False)
    print((s.stdout + s.stderr).strip().splitlines()[-1])


if __name__ == "__main__":
    main()
