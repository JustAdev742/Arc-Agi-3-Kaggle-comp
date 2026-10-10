#!/usr/bin/env python
"""Download a kernel's output files matching a pattern, paced so Kaggle's output listing does not answer 429.

    KAGGLE_API_TOKEN=... .venv/bin/python scripts/kaggle_pull.py OWNER/KERNEL OUT_DIR [--pattern REGEX]
        [--page-size 100] [--pause 3] [--list-only]

`kaggle kernels output` lists a kernel's output 20 files per page and asks for each next page at once, so a large
output (MTP session A's) costs dozens of ListKernelSessionOutput calls in a burst; on 2026-10-10 Kaggle answered 429
Too Many Requests after the first page and kept refusing for over 15 minutes. This lists with bigger pages, pauses
between them, retries a refused page with backoff (60, 120, 240, 480 s), then downloads only the files matching
--pattern (re.search on the output-relative path, as the CLI does) and writes the kernel log as <kernel>.log, like
the CLI. --list-only prints the listing (path and size when the listing gives it) and downloads nothing.
Exit status: 0 when every page was listed, 1 when a page was still refused.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

WAITS = (60, 120, 240, 480)


def _is_429(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 429 or "429" in str(exc)


def list_pages(api, owner: str, slug: str, page_size: int, pause: float, sleep=time.sleep, log=print):
    """Yield (files, log) per page; a refused page is retried with backoff and re-raises after the last wait."""
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    token = None
    first = True
    while first or token:
        for attempt, wait in enumerate((0, *WAITS)):
            sleep(wait)
            try:
                with api.build_kaggle_client() as kaggle:
                    request = ApiListKernelSessionOutputRequest()
                    request.user_name, request.kernel_slug = owner, slug
                    api._set_paging(request, page_size, token)
                    response = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)
                break
            except Exception as exc:  # the SDK raises requests' HTTPError
                if not _is_429(exc) or attempt == len(WAITS):
                    raise
                log(f"page listing refused (429); retrying in {WAITS[attempt]} s")
        yield list(response.files or []), (response.log if first else None)
        token, first = response.next_page_token, False
        if token:
            sleep(pause)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kernel", help="OWNER/KERNEL")
    ap.add_argument("out", type=Path)
    ap.add_argument("--pattern", default=None)
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--pause", type=float, default=3.0)
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args(argv)
    owner, slug = args.kernel.split("/", 1)
    pattern = re.compile(args.pattern) if args.pattern else None

    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    args.out.mkdir(parents=True, exist_ok=True)
    listed = downloaded = 0
    try:
        for files, log in list_pages(api, owner, slug, args.page_size, args.pause):
            for item in files:
                listed += 1
                name = item.file_name
                if args.list_only:
                    print(f"{name}\t{getattr(item, 'size', '') or ''}")
                    continue
                if pattern and not pattern.search(name):
                    continue
                target = (args.out / name).resolve()
                if not str(target).startswith(str(args.out.resolve()) + os.sep):
                    print(f"skipped {name!r}: outside the output directory", file=sys.stderr)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with requests.get(item.url, stream=True, timeout=600) as r:
                    r.raise_for_status()
                    with open(target, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                downloaded += 1
                print(f"downloaded {name}")
            if log and not args.list_only:
                (args.out / f"{slug}.log").write_text(log)
                print(f"kernel log -> {args.out / (slug + '.log')}")
    except Exception as exc:
        print(f"stopped after {listed} listed / {downloaded} downloaded: {exc}", file=sys.stderr)
        return 1
    print(f"{listed} files listed, {downloaded} downloaded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
