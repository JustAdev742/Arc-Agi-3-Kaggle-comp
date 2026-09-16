"""Push a built evaluation notebook folder to Kaggle on the RTX PRO 6000.

    .venv/bin/python scripts/push_eval.py <folder with eval.ipynb + kernel-metadata.json>

Always passes ``--accelerator NvidiaRtxPro6000``: without it Kaggle places the kernel on a Tesla T4 (15 GB, SM75),
the 27B model cannot load, and the run burns ten minutes of quota before falling back (exp-010b/exp-011 v1,
2026-09-16). Prints the pushed version and the kernel's status right after.
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
    r = subprocess.run([kaggle, "kernels", "push", "-p", str(folder), "--accelerator", ACCELERATOR],
                       env=env, capture_output=True, text=True, check=False)
    print((r.stdout + r.stderr).strip().splitlines()[-1])
    if "successfully pushed" not in r.stdout:
        sys.exit(1)
    time.sleep(5)
    s = subprocess.run([kaggle, "kernels", "status", meta["id"]], env=env, capture_output=True, text=True, check=False)
    print((s.stdout + s.stderr).strip().splitlines()[-1])


if __name__ == "__main__":
    main()
