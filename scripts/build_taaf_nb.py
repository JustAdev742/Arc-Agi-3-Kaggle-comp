#!/usr/bin/env python
"""Build a Kaggle notebook folder for one arm of the animation-aware Duck on the Flash-Next server.

    .venv/bin/python scripts/build_taaf_nb.py --out <dir> --slug arc3-taaf-ours [--patches P1 ...] [--kv-dtype fp8]
        [--wavefit] [--note "what this arm changes"]

Base: kaggle/taaf/base-thui-animfast.ipynb, our private copy of the public notebook
yocybercode/thui-animfast-b71-full25-r1 (Keith Tyser's Flash-Next NVFP4 serving bundle + Jakob Brueggen's
animation-aware TAAF source; Tufa Labs' Duck harness, MIT). Options:

- ``--patches``: our source patches from scripts/taaf_ours_patch.py. The notebook inlines that file, copies the mounted
  anim bundle to /tmp/ours_bundle, applies the patches there and imports the solver from the copy (no dataset of ours).
- ``--kv-dtype`` / ``--max-num-seqs`` / ``--cudagraph``: serving profile overrides (the base profile is kv5-bf16-mtp3-c8-cg32).
- ``--wavefit``: in a real competition rerun only, size the per-game cap to the concurrency waves the hidden list needs so
  the last wave is not cancelled by the notebook's soft deadline (at most 1.25x the stock 7,920 s).

Writes ``<out>/<slug>.ipynb`` and ``<out>/kernel-metadata.json`` (private, RTX PRO 6000); push with scripts/push_eval.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kaggle" / "taaf" / "base-thui-animfast.ipynb"
PATCH_SRC = ROOT / "scripts" / "taaf_ours_patch.py"
BASE_META = {
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
    "enable_internet": False, "keywords": [], "kernel_sources": [],
    "dataset_sources": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                        "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1",
                        "jakobbrggen/taaf-kaggle-source-anim-20260807-anim"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
}
ANIM_ANCHOR = 'ANIM_BUNDLE_DIR = _find_bundle_dir("anim-20260807-anim")     # ours: the solver tree + its pickled benchmark / target'
SOFT_END_ANCHOR = """soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(
    seconds=budget - 600.0
)"""
WAVEFIT = """
# ours (--wavefit): in a real rerun, size the per-game cap to the waves the hidden list needs (110 games / 28 = 4
# waves), so the last wave ends before soft_end instead of being cancelled; never above 1.25x the stock 7,920 s.
if TRUE_SUBMISSION:
    _n_games = len(bm.games)
    _conc = max(1, int(getattr(bm.solver, "concurrency", 28) or 28))
    _waves = max(1, -(-_n_games // _conc))
    _remaining = soft_end.timestamp() - time.time() - 300.0
    _fit = max(1800.0, min(7920.0 * 1.25, _remaining / _waves))
    print(f"ours: wave-fit games={_n_games} concurrency={_conc} waves={_waves} "
          f"remaining_s={_remaining:.0f} per_game_s={_fit:.0f} (was {bm.solver.max_runtime_s_per_game})", flush=True)
    bm.solver.max_runtime_s_per_game = _fit"""


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [src]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--patches", nargs="*", default=[])
    ap.add_argument("--kv-dtype", choices=["auto", "fp8"], default=None)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--cudagraph", type=int, default=None)
    ap.add_argument("--wavefit", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    nb = json.loads(BASE.read_text())
    cells = nb["cells"]
    changes = []
    src0 = "".join(cells[0]["source"])
    assert src0.startswith("# Evaluation fork (taaf-anim-flashnext)"), "unexpected base notebook header"
    header = (f"# {args.slug} (team scottmahony)\n\n"
              "Private arm built by scripts/build_taaf_nb.py from our copy of the public notebook "
              "[yocybercode/thui-animfast-b71-full25-r1](https://www.kaggle.com/code/yocybercode/thui-animfast-b71-full25-r1). "
              "Solver: Tufa Labs' TAAF / Duck harness (MIT, credit to the Tufa Labs team) on Jakob Brüggen's "
              "animation-awareness branch; serving: Keith Tyser's Flash-Next NVFP4 stack; graft: Thuitanium / Knowless Crew.")
    for cell in cells:
        s = "".join(cell["source"])
        orig = s
        if s.startswith("# Evaluation fork (taaf-anim-flashnext)"):
            s = header
        if "PUBLIC25_VLLM_PROFILE_ENV = {" in s:
            if args.kv_dtype:
                s = s.replace('"TAAF_VLLM_KV_CACHE_DTYPE": "auto"', f'"TAAF_VLLM_KV_CACHE_DTYPE": "{args.kv_dtype}"')
                changes.append(f"KV cache dtype {args.kv_dtype}")
            if args.max_num_seqs:
                s = s.replace('"TAAF_VLLM_MAX_NUM_SEQS": "8"', f'"TAAF_VLLM_MAX_NUM_SEQS": "{args.max_num_seqs}"')
                changes.append(f"max_num_seqs {args.max_num_seqs}")
            if args.cudagraph:
                s = s.replace('"TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE": "32"',
                              f'"TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE": "{args.cudagraph}"')
                changes.append(f"cudagraph capture {args.cudagraph}")
            s = s.replace("PUBLIC25_VLLM_PROFILE_NAME = 'kv5-bf16-mtp3-c8-cg32'",
                          f"PUBLIC25_VLLM_PROFILE_NAME = '{args.slug}'")
        if args.patches and ANIM_ANCHOR in s:
            s = s.replace(ANIM_ANCHOR, ANIM_ANCHOR + f"""
# ours: patch a writable copy of the anim bundle with scripts/taaf_ours_patch.py (inlined in the next cell's source)
import shutil as _shutil
_OURS_BUNDLE = Path("/tmp/ours_bundle")
if _OURS_BUNDLE.exists():
    _shutil.rmtree(_OURS_BUNDLE)
_shutil.copytree(ANIM_BUNDLE_DIR, _OURS_BUNDLE, ignore=_shutil.ignore_patterns("__pycache__"))
_ours_ns = {{"__name__": "taaf_ours_patch"}}
exec(compile(_OURS_PATCH_SOURCE, "taaf_ours_patch.py", "exec"), _ours_ns)
print("ours: applied", _ours_ns["apply"](_OURS_BUNDLE, {args.patches!r}), flush=True)
ANIM_BUNDLE_DIR = _OURS_BUNDLE""")
            changes.append(f"source patches {args.patches}")
        if args.wavefit and SOFT_END_ANCHOR in s:
            s = s.replace(SOFT_END_ANCHOR, SOFT_END_ANCHOR + WAVEFIT)
            changes.append("wave-fit per-game cap in real reruns")
        if s != orig:
            cell["source"] = [s]
    if args.patches:
        idx = next(i for i, c in enumerate(cells) if ANIM_ANCHOR in "".join(c["source"]))
        cells.insert(idx, code_cell("# ours: source of scripts/taaf_ours_patch.py, applied in the next cell\n"
                                    f"_OURS_PATCH_SOURCE = {PATCH_SRC.read_text()!r}\n"))
    expected = (bool(args.kv_dtype) + bool(args.max_num_seqs) + bool(args.cudagraph) + bool(args.patches)
                + bool(args.wavefit))
    if len(changes) != expected:
        raise SystemExit(f"not every requested change found its anchor: {changes}")
    cells[0]["source"] = ["".join(cells[0]["source"]) + "\n\n**Changes in this arm:** "
                          + ("; ".join(changes) if changes else "none (control)")
                          + (f". {args.note}" if args.note else "")]
    nb.setdefault("metadata", {})["kaggle"] = {"accelerator": "nvidiaRtxPro6000", "isInternetEnabled": False,
                                               "isGpuEnabled": True, "language": "python", "sourceType": "notebook"}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.slug}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {"id": f"scottmahony/{args.slug}", "title": args.slug.replace("-", " "), "code_file": f"{args.slug}.ipynb",
            **BASE_META}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"built {out / (args.slug + '.ipynb')}: {changes or 'control'}")


if __name__ == "__main__":
    main()
