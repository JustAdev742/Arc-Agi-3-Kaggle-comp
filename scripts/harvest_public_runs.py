#!/usr/bin/env python
"""Harvest other teams' public-25 runs from their public Kaggle notebooks, free of GPU quota.

    .venv/bin/python scripts/harvest_public_runs.py [--pages 3] [--keep-transcripts REF ...]

A public notebook's output (its own "Save & Run All" on the 25 public games) is downloadable. For every competition
notebook in the score-sorted listing this downloads the output, and when a TAAF ``benchmark.json`` is present it
rescores every game with ``arc3.scoring.game_score`` and records the notebook's configuration (attached datasets and
models, plus knobs grepped from its source) so that repeated runs of one configuration can be averaged. Bulky files
are deleted after summarizing (transcripts are kept only for ``--keep-transcripts``). Results go to
``runs/public-harvest/index.json`` (one row per notebook) and ``runs/public-harvest/<owner>__<slug>.json``.
These are other people's runs: use them to rank configurations, never report them as ours.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arc3.scoring import game_score  # noqa: E402
from arc3.splits import resolve  # noqa: E402

COMP = "arc-prize-2026-arc-agi-3"
OUT = ROOT / "runs" / "public-harvest"
KNOBS = ("MULTIMODAL_UPSCALE", "LOCAL_ANALYZER_YIELD_SECONDS", "LOCAL_ANALYZER_SEED", "LOCAL_ANALYZER_TEMPERATURE",
         "LOCAL_ANALYZER_CONTEXT_WINDOW", "TAAF_VLLM_MAX_NUM_SEQS", "TAAF_VLLM_MTP_TOKENS", "max_runtime_s_per_game",
         "analyzer_timeout", "concurrency", "ARC3_ANIMATION_AWARENESS", "ARC3_HARD_NOOP_GUARD", "reasoning_effort")


def kaggle(*args: str, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    token = ROOT / ".kaggle" / "access_token"
    if token.exists() and not env.get("KAGGLE_API_TOKEN"):
        env["KAGGLE_API_TOKEN"] = token.read_text().strip()
    return subprocess.run([str(ROOT / ".venv" / "bin" / "kaggle"), *args], env=env, capture_output=True, text=True,
                          check=False, timeout=timeout)


def list_refs(pages: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for page in range(1, pages + 1):
        r = kaggle("kernels", "list", "--competition", COMP, "--sort-by", "scoreDescending", "--page-size", "100",
                   "-p", str(page), "-v")
        text = r.stdout[r.stdout.find("ref,"):] if "ref," in r.stdout else ""
        page_rows = list(csv.DictReader(io.StringIO(text)))
        if not page_rows:
            break
        for i, row in enumerate(page_rows):
            row["rank"] = str(len(rows) + i + 1)
        rows.extend(page_rows)
    return rows


def knobs_from_source(nb_path: Path) -> dict[str, list[str]]:
    try:
        nb = json.loads(nb_path.read_text())
        src = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))
    except (OSError, ValueError):
        return {}
    found: dict[str, list[str]] = {}
    for knob in KNOBS:
        vals = re.findall(rf"{knob}['\"]?\s*[:=]\s*['\"]?([\w.\-]+)", src)
        if vals:
            found[knob] = sorted(set(vals))
    return found


def summarize(bench: dict) -> dict:
    runs = bench.get("game_runs") or []
    dev, val = set(resolve("dev")), set(resolve("val"))
    games = []
    for g in runs:
        gid = str(g["game_id"]).split("-")[0]
        n = int(g.get("levels_completed", 0))
        apl = list(g.get("actions_per_level") or [])
        base = list(g.get("base_actions_per_level") or [])
        hist = g.get("history") or []
        score = game_score([apl[i] if i < n else None for i in range(len(base))], base) if base else 0.0
        gen = sum(int(h.get("generated_tokens") or 0) for h in hist)
        games.append({"game_id": gid, "levels": n, "levels_total": len(base), "level_actions": apl[:n],
                      "baselines": base[:n], "score": round(score, 3), "actions": len(hist),
                      "tok_per_action": round(gen / len(hist), 1) if hist else None,
                      "wall_s": g.get("final_wallclock_seconds")})

    def mean(ids: set[str] | None) -> float | None:
        xs = [g["score"] for g in games if ids is None or g["game_id"] in ids]
        return round(sum(xs) / len(xs), 3) if xs else None

    return {"games": len(games), "score_all": mean(None), "score_dev": mean(dev), "score_val": mean(val),
            "levels": sum(g["levels"] for g in games), "actions": sum(g["actions"] for g in games), "results": games}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=3)
    ap.add_argument("--keep-transcripts", nargs="*", default=[])
    ap.add_argument("--refs", nargs="*", default=None, help="harvest only these notebooks")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    index_path = OUT / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    listing = list_refs(args.pages) if args.refs is None else [{"ref": r, "rank": "?"} for r in args.refs]
    for row in listing:
        ref = row["ref"]
        if ref in index and index[ref].get("status") in {"ok", "no-benchmark"}:
            continue
        tmp = Path(tempfile.mkdtemp(prefix="harvest-"))
        try:
            meta_dir = tmp / "meta"
            kaggle("kernels", "pull", ref, "-p", str(meta_dir), "-m", timeout=300)
            meta = {}
            if (meta_dir / "kernel-metadata.json").exists():
                meta = json.loads((meta_dir / "kernel-metadata.json").read_text())
            nb = next(iter(meta_dir.glob("*.ipynb")), None)
            knobs = knobs_from_source(nb) if nb else {}
            out_dir = tmp / "out"
            r = kaggle("kernels", "output", ref, "-p", str(out_dir), "-o", timeout=1800)
            benches = sorted(out_dir.rglob("benchmark.json"), key=lambda p: p.stat().st_size)
            entry = {"ref": ref, "rank": row.get("rank"), "title": row.get("title"), "last_run": row.get("lastRunTime"),
                     "datasets": meta.get("dataset_sources", []), "models": meta.get("model_sources", []),
                     "machine": meta.get("machine_shape"), "knobs": knobs,
                     "anim": any("taaf-kaggle-source-anim" in d for d in meta.get("dataset_sources", []))}
            if not benches:
                entry["status"] = "no-benchmark"
                entry["note"] = (r.stdout + r.stderr).strip().splitlines()[-1:] if (r.stdout + r.stderr).strip() else []
            else:
                summary = summarize(json.loads(benches[-1].read_text()))
                entry.update({"status": "ok", **{k: v for k, v in summary.items() if k != "results"}})
                (OUT / f"{ref.replace('/', '__')}.json").write_text(json.dumps({**entry, **summary}, indent=1))
                if ref in args.keep_transcripts:
                    keep = ROOT / "runs" / f"pub-{ref.replace('/', '_')}" / "kernel-output"
                    keep.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(out_dir, keep, dirs_exist_ok=True)
            index[ref] = entry
            print(f"{row.get('rank'):>4} {ref}: {entry['status']} score {entry.get('score_all')} "
                  f"levels {entry.get('levels')} models {entry['models']}", flush=True)
        except (subprocess.TimeoutExpired, OSError, ValueError, KeyError) as exc:
            index[ref] = {"ref": ref, "status": f"error: {exc!r}"[:300]}
            print(f"{row.get('rank')} {ref}: error {exc!r}", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            index_path.write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
