#!/usr/bin/env python
"""Best public scores of this competition's public notebooks (Kaggle's search API; `kaggle kernels list` has none).

    KAGGLE_API_TOKEN=... .venv/bin/python scripts/kaggle_nb_scores.py [--top 40] [--query ARC-AGI-3 ...]

Lists the competition's notebooks with `kaggle kernels list --competition` (several sort orders, 100 each), searches
Kaggle for each query (kagglesdk SearchApiService.ListEntities, kernel documents, which carry
`best_public_score`), and prints the competition notebooks that have a score, best first. A notebook's best public
score is one leaderboard draw (the best of its owner's submissions), not its mean.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.search.types.search_api_service import ListEntitiesFilters, ListEntitiesRequest
from kagglesdk.search.types.search_enums import DocumentType

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "arc-prize-2026-arc-agi-3"
SORTS = ("hotness", "voteCount", "dateRun", "scoreDescending", "dateCreated")


def competition_refs() -> set[str]:
    refs: set[str] = set()
    for sort in SORTS:
        out = subprocess.run([str(ROOT / ".venv" / "bin" / "kaggle"), "kernels", "list", "--competition", COMPETITION,
                              "--sort-by", sort, "--page-size", "100", "--format", "json"],
                             capture_output=True, text=True, check=False).stdout
        try:
            refs |= {k["ref"].lower() for k in json.loads(out)}
        except (json.JSONDecodeError, TypeError, KeyError):
            print(f"kernels list --sort-by {sort}: unreadable output", file=sys.stderr)
    return refs


def search_scores(queries: list[str], pages: int = 20) -> dict[str, dict]:
    api = KaggleApi()
    api.authenticate()
    found: dict[str, dict] = {}
    with api.build_kaggle_client() as kaggle:
        for q in queries:
            token = ""
            for _ in range(pages):
                req = ListEntitiesRequest()
                filters = ListEntitiesFilters()
                filters.query = q
                filters.document_types = [DocumentType.KERNEL]
                req.filters = filters
                req.page_size = 100
                if token:
                    req.page_token = token
                resp = kaggle.search.search_api_client.list_entities(req)
                for d in resp.documents:
                    owner = d.owner_user.user_name if d.owner_user else ""
                    kd = d.kernel_document
                    found[f"{owner}/{d.slug}".lower()] = {
                        "score": kd.best_public_score if kd else None, "votes": d.votes, "title": d.title,
                        "updated": str(d.update_time)[:16]}
                token = resp.next_page_token
                if not token:
                    break
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--query", action="append", default=None)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()
    refs = competition_refs()
    found = search_scores(args.query or ["ARC-AGI-3", "arc3", "ARC Prize 2026"])
    rows = [(ref, v) for ref, v in found.items() if ref in refs and v["score"]]
    rows.sort(key=lambda kv: -kv[1]["score"])
    print(f"{len(refs)} competition notebooks listed, {len(found)} search hits, {len(rows)} with a public score")
    for ref, v in rows[: args.top]:
        print(f"{v['score']:7.2f}  votes={v['votes']:>4}  updated {v['updated']}  {ref}  | {v['title'][:60]}")


if __name__ == "__main__":
    main()
