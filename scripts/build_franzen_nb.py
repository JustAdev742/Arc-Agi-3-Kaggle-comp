#!/usr/bin/env python
"""Build our private arms of Daniel Franzen's Milestone 2 notebook (public LB 27.89; Apache-2.0, kaggle/franzen/).

    .venv/bin/python scripts/build_franzen_nb.py --out DIR --slug arc3-franzen-m2              # unchanged copy
    .venv/bin/python scripts/build_franzen_nb.py --out DIR --slug arc3-franzen-m2-full25 --full25 121
    .venv/bin/python scripts/build_franzen_nb.py ... --env MULTIMODAL_UPSCALE=8 --env ARC3_MAX_ACTIVE_STREAMS=12

Without options the notebook is byte-for-byte his (only our kernel metadata differs): a Kaggle "Save & Run" of it plays
his 10-game demo subset for 25 minutes per game, and a competition rerun plays the hidden set exactly as his did.
Options change a non-submission run only, or a named environment knob everywhere:

- ``--full25 MIN``: the Save & Run plays all 25 public games (his demo list emptied) with MIN minutes per game. 121
  matches the hidden set's compute per game (110 games share his 10 admission slots for 532 minutes: 48.4
  slot-minutes per game; 25 games x 48.4 / 10 slots = 121 minutes). The competition rerun is untouched.
- ``--env KEY=VALUE`` (repeatable): override one key of his ``setup_env`` / priority-scheduling dictionaries in cell 4
  (they become environment variables for the harness); the key must already exist there.

Every change is anchored on text that must occur exactly once, and listed in the first markdown cell. Writes
``<out>/<slug>.ipynb`` and ``<out>/kernel-metadata.json`` (private, internet off, RTX PRO 6000).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kaggle" / "franzen" / "arc-agi-3-milestone-2-solution.ipynb"
BASE_SHA256 = "7b76c194b478faa0309b01e5fba05840f1314da7f288e05e5f4b09972e4f9b4c"
SOURCES = {  # his kernel-metadata.json, 2026-10-02
    "dataset_sources": ["dfranzen/pennyroyal-v253", "dfranzen/taaf-kaggle-source-bundle-copy"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "model_sources": ["dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1",
                      "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1"],
    "kernel_sources": [],
}
DEMO_ANCHOR = ("demo_excluded_games = [] if TRUE_SUBMISSION else ['bp35', 'cd82', 'cn04', 'dc22', 'g50t', 'ka59', "
               "'lf52', 'ls20', 'm0r0', 's5i5', 'sk48', 'sp80', 'su15', 'tn36', 'wa30']")
BUDGET_ANCHOR = "        bm.solver.max_runtime_s_per_game = 25*60 #532*60 * bm.solver.concurrency // 110"


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"{what}: anchor found {text.count(old)} times in the base notebook (expected once)")
    return text.replace(old, new)


def build(out: Path, slug: str, full25: float | None = None, env: dict[str, str] | None = None,
          note: str = "") -> list[str]:
    import hashlib

    raw = BASE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise SystemExit(f"{BASE} is not the vendored copy (sha256 differs); it must stay unmodified")
    nb = json.loads(raw)
    cells = nb["cells"]
    changes: list[str] = []
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        s = "".join(cell["source"])
        if full25 is not None and DEMO_ANCHOR in s:
            s = _replace_once(s, DEMO_ANCHOR, "demo_excluded_games = []  # ours (--full25): all 25 public games", "demo")
            s = _replace_once(s, BUDGET_ANCHOR,
                              f"        bm.solver.max_runtime_s_per_game = {float(full25)!r}*60  # ours (--full25)",
                              "budget")
            changes.append(f"Save & Run plays all 25 public games, {full25:g} min per game (competition rerun unchanged)")
        for key, value in (env or {}).items():
            pattern = re.compile(rf"^(\s*)'{re.escape(key)}': ([^,\n]+),", re.M)
            hits = pattern.findall(s)
            if not hits:
                continue
            if len(hits) != 1:
                raise SystemExit(f"--env {key}: found {len(hits)} times in cell {i}")
            new_value = value if re.fullmatch(r"-?\d+(\.\d+)?", value) else repr(value)
            s = pattern.sub(lambda m, nv=new_value, k=key: f"{m.group(1)}'{k}': {nv},  # ours (--env)", s)
            changes.append(f"env {key}={value}")
        cell["source"] = s.splitlines(keepends=True)
    expected = (full25 is not None) + len(env or {})
    if len(changes) != expected:
        raise SystemExit(f"not every requested change found its anchor: {changes}")
    if changes or note:
        cells[0]["source"] = ["".join(cells[0]["source"]) + "\n\n**Our arm (scottmahony, built by "
                              "scripts/build_franzen_nb.py from the unmodified notebook):** "
                              + ("; ".join(changes) or "unchanged") + (f". {note}" if note else "") + "\n"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta = {"id": f"scottmahony/{slug}", "title": slug.replace("-", " "), "code_file": f"{slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": False, "keywords": [], **SOURCES}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return changes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--full25", type=float, default=None, metavar="MIN")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    env = dict(e.split("=", 1) for e in args.env)
    changes = build(args.out, args.slug, args.full25, env, args.note)
    print(f"built {args.out / (args.slug + '.ipynb')}: {changes or 'unchanged'}")


if __name__ == "__main__":
    main()
