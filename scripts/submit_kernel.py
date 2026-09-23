#!/usr/bin/env python
"""Submit one version of one of our notebooks to the competition, after checking it can score.

    .venv/bin/python scripts/submit_kernel.py scottmahony/arc3-taaf-ours-b 1 "exp-035 ours-b (history compression)"

Checks, in order, and refuses on any failure: the daily allowance is not spent (`submission-limits`), the kernel's
latest run is COMPLETE, and its output lists submission.parquet. Then runs
`kaggle competitions submit arc-prize-2026-arc-agi-3 -k <kernel> -v <version> -f submission.parquet -m <message>`
and prints the newest rows of `kaggle competitions submissions`. Record the result in docs/status.md.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = "arc-prize-2026-arc-agi-3"


def kaggle(*args: str) -> str:
    env = dict(os.environ)
    token = ROOT / ".kaggle" / "access_token"
    if token.exists() and not env.get("KAGGLE_API_TOKEN"):
        env["KAGGLE_API_TOKEN"] = token.read_text().strip()
    r = subprocess.run([str(ROOT / ".venv" / "bin" / "kaggle"), *args], env=env, capture_output=True, text=True,
                       check=False, timeout=600)
    return (r.stdout + r.stderr).strip()


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    kernel, version, message = sys.argv[1], sys.argv[2], sys.argv[3]
    limits = kaggle("competitions", "submission-limits", COMP)
    print(limits)
    if "Remaining today: 0" in limits:
        raise SystemExit("refused: no submissions remaining today")
    status = kaggle("kernels", "status", kernel)
    print(status)
    if "COMPLETE" not in status:
        raise SystemExit("refused: the kernel's latest run is not COMPLETE")
    token, found = None, False
    for _ in range(100):  # the listing is paginated (at most 200 per page) and TAAF runs write thousands of files
        args = ["kernels", "files", f"{kernel}/{version}", "--page-size", "200", "-v"]
        page = kaggle(*args, *(["--page-token", token] if token else []))
        if "submission.parquet" in page:
            found = True
            break
        nxt = [ln.split("=", 1)[1].strip() for ln in page.splitlines() if ln.startswith("Next Page Token")]
        if not nxt or not nxt[0]:
            break
        token = nxt[0]
    if not found:
        raise SystemExit("refused: submission.parquet not in the output of that kernel version")
    print(kaggle("competitions", "submit", COMP, "-k", kernel, "-v", str(version), "-f", "submission.parquet",
                 "-m", message))
    print("\n".join(kaggle("competitions", "submissions", COMP).splitlines()[:5]))


if __name__ == "__main__":
    main()
