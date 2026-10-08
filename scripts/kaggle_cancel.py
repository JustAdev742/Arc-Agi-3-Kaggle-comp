#!/usr/bin/env python
"""Cancel one of our Kaggle kernel sessions by its kernel_session_id (kagglesdk CancelKernelSession).

    .venv/bin/python scripts/kaggle_cancel.py --session ID

For a session stuck QUEUED behind Kaggle's 2-session limit, or a run that should stop. Only our own kernels.
Where to get the id is open: the push reply of this SDK version carries none (checked 2026-10-08), so FOLDER mode
only works with a sessions.jsonl row that has one.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", nargs="?", type=Path)
    ap.add_argument("--session", type=int)
    args = ap.parse_args()
    if args.session is None:
        if args.folder is None:
            raise SystemExit("give FOLDER or --session")
        rows = [json.loads(line) for line in (args.folder / "sessions.jsonl").read_text().splitlines() if line.strip()]
        args.session = int(rows[-1]["kernel_session_id"])
        print(f"{rows[-1]['kernel']} v{rows[-1]['version']}: session {args.session}")
    token = ROOT / ".kaggle" / "access_token"
    if token.exists() and not os.environ.get("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_API_TOKEN"] = token.read_text().strip()
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiCancelKernelSessionRequest
    api = KaggleApi()
    api.authenticate()
    req = ApiCancelKernelSessionRequest()
    req.kernel_session_id = args.session
    with api.build_kaggle_client() as kaggle:
        print(kaggle.kernels.kernels_api_client.cancel_kernel_session(req))


if __name__ == "__main__":
    main()
