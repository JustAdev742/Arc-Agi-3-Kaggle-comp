#!/usr/bin/env python
"""Build Duck-fork arm notebooks from the registry kaggle/taaf/arms.json.

    .venv/bin/python scripts/build_arms.py --out <dir> [--arms exp039 exp040] [--force]

Each arm gets ``<dir>/<exp>/`` with its notebook and kernel-metadata.json, built by scripts/build_taaf_nb.py from the
registry's patches, knobs (the registry defaults, overridden per arm; a null value drops a default) and flags. Arms
with a recorded ``pushed`` version are history and are skipped unless ``--force``: rebuilding them would change what a
later push of the same kernel runs. Push a built folder with scripts/push_eval.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "kaggle" / "taaf" / "arms.json"


def arm_command(arm: dict, defaults: dict, out: Path) -> list[str]:
    knobs = {**defaults.get("knobs", {}), **arm.get("knobs", {})}
    cmd = [sys.executable, str(ROOT / "scripts" / "build_taaf_nb.py"), "--out", str(out / arm["exp"]),
           "--slug", arm["slug"], "--patches", *arm["patches"]]
    for key, value in knobs.items():
        if value is not None:
            cmd += ["--knob", f"{key}={value}"]
    if arm.get("wavefit", defaults.get("wavefit", False)):
        cmd.append("--wavefit")
    if arm.get("note"):
        cmd += ["--note", arm["note"]]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="*", default=None, help="exp ids (default: every arm not yet pushed)")
    ap.add_argument("--force", action="store_true", help="also rebuild arms that record a pushed version")
    args = ap.parse_args()
    registry = json.loads(REGISTRY.read_text())
    arms = {a["exp"]: a for a in registry["arms"]}
    unknown = sorted(set(args.arms or []) - set(arms))
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")
    selected = [arms[e] for e in (args.arms or list(arms))]
    out = Path(args.out)
    for arm in selected:
        if arm.get("pushed") and not args.force:
            print(f"skip {arm['exp']} ({arm['slug']} v{arm['pushed']} already pushed; --force to rebuild)")
            continue
        subprocess.run(arm_command(arm, registry.get("defaults", {}), out), check=True)


if __name__ == "__main__":
    main()
