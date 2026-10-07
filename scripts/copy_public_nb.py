#!/usr/bin/env python
"""Make our private Kaggle copy of a vendored public notebook, unchanged except its kernel metadata.

    .venv/bin/python scripts/copy_public_nb.py kaggle/dprime/affectify-arc-31-54-in-a-single-sub.ipynb \
        --sources kaggle/dprime/upstream-kernel-metadata.json --out DIR --slug arc3-dprime

Cells are copied byte for byte; the notebook's Kaggle accelerator metadata is set to the RTX PRO 6000 (what
scripts/push_eval.py checks; the push also passes it), and kernel-metadata.json is written for scottmahony/<slug>:
private, internet off, GPU on, with the upstream notebook's dataset/model/competition sources.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def copy(notebook: Path, sources: Path, out: Path, slug: str) -> dict:
    nb = json.loads(notebook.read_text())
    upstream = json.loads(sources.read_text())
    if upstream.get("enable_internet"):
        raise SystemExit("the upstream notebook has internet on; a submission must run offline")
    nb.setdefault("metadata", {}).setdefault("kaggle", {})["accelerator"] = "nvidiaRtxPro6000"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta = {"id": f"scottmahony/{slug}", "title": slug.replace("-", " "), "code_file": f"{slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": False, "keywords": [],
            "dataset_sources": upstream.get("dataset_sources", []),
            "competition_sources": upstream.get("competition_sources", []),
            "model_sources": upstream.get("model_sources", []),
            "kernel_sources": upstream.get("kernel_sources", [])}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--sources", type=Path, required=True, help="the upstream kernel-metadata.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slug", required=True)
    args = ap.parse_args()
    meta = copy(args.notebook, args.sources, args.out, args.slug)
    print(f"built {args.out / (args.slug + '.ipynb')} for {meta['id']}: sources {meta['dataset_sources']} "
          f"{meta['model_sources']}")


if __name__ == "__main__":
    main()
