#!/usr/bin/env python
"""Build a Kaggle notebook folder for one arm of the animation-aware Duck on the Flash-Next server.

    .venv/bin/python scripts/build_taaf_nb.py --out <dir> --slug arc3-taaf-ours [--patches P1 ...] [--kv-dtype fp8]
        [--wavefit] [--note "what this arm changes"]

Base: kaggle/taaf/base-thui-animfast.ipynb, our private copy of the public notebook
yocybercode/thui-animfast-b71-full25-r1 (Keith Tyser's Flash-Next NVFP4 serving bundle + Jakob Brueggen's
animation-aware TAAF source; Tufa Labs' Duck harness, MIT). Options:

- ``--patches``: our source patches from scripts/taaf_ours_patch.py. The notebook inlines that file, copies the mounted
  anim bundle to /tmp/ours_bundle, applies the patches there and imports the solver from the copy (no dataset of ours).
- ``--kv-dtype`` / ``--max-num-seqs`` / ``--cudagraph`` / ``--kv-gib`` / ``--batched-tokens`` / ``--prefix-caching`` /
  ``--mtp-tokens``: serving profile overrides (the base profile is kv5-bf16-mtp3-c8-cg32 with 8,192 batched tokens and no
  prefix caching).
- ``--wavefit``: in a real competition rerun only, size the per-game cap to the concurrency waves the hidden list needs so
  the last wave is not cancelled by the notebook's soft deadline (at most 1.25x the stock 7,920 s).
- ``--model nvidia``: serve NVIDIA's checkpoint (Kaggle model xiaoz259/qwen3-8-flash-next-nvfp4/PyTorch/nvidia-nvfp4/1)
  instead of RadixArk's. The notebook inlines scripts/nvidia_serving_patch.py, copies Keith's mounted serving bundle to
  /tmp, patches the copy's serving_setup.py (model identity, ``--quantization modelopt_mixed``, three vLLM backports on
  the extracted runtime) and points BUNDLE_DIR at it, so setup, watchdog and teardown all use the patched copy.

Writes ``<out>/<slug>.ipynb`` and ``<out>/kernel-metadata.json`` (private, RTX PRO 6000); push with scripts/push_eval.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nvidia_serving_patch as nv  # noqa: E402

BASE = ROOT / "kaggle" / "taaf" / "base-thui-animfast.ipynb"
PATCH_SRC = ROOT / "scripts" / "taaf_ours_patch.py"
NV_PATCH_SRC = ROOT / "scripts" / "nvidia_serving_patch.py"
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
KNOBS_ANCHOR = '_KNOBS = {"LOCAL_ANALYZER_SEED": "20260825", "LOCAL_ANALYZER_YIELD_SECONDS": "180"}'
YIELD_ASSERT = ('assert float(_tool_agent._LOCAL_ANALYZER_YIELD_SECONDS) == float("180"), '
                '_tool_agent._LOCAL_ANALYZER_YIELD_SECONDS')
GRAFT_PRINT = 'print(f"THUI_ANIMFAST_GRAFT ok'
LANES_ANCHOR = "bm.solver.concurrency = 28\n"
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
BUNDLE_ANCHOR = ('BUNDLE_DIR = _find_bundle_dir("duck-harness-kaggle")          '
                 '# his: serving_setup.py, vllm patches, watchdog, teardown')
NV_BUNDLE = "/tmp/ours_nvidia_serving_bundle"
NV_SWAP = f"""
# ours (--model nvidia): serve nvidia/Qwen3.8-Flash-Next-NVFP4 instead of RadixArk's build. Patch a writable copy of his
# serving bundle with scripts/nvidia_serving_patch.py (inlined in the previous cell); setup, watchdog and teardown all
# read BUNDLE_DIR, so every serving step uses the patched copy.
_nv_ns = {{"__name__": "nvidia_serving_patch"}}
exec(compile(_NV_SERVING_PATCH_SOURCE, "nvidia_serving_patch.py", "exec"), _nv_ns)
print("ours: nvidia serving patch", _nv_ns["apply"](BUNDLE_DIR, Path({NV_BUNDLE!r})), flush=True)
BUNDLE_DIR = Path({NV_BUNDLE!r})"""
NV_MARKDOWN = [  # (old, new) in the upstream description, so the page names the weights it actually serves
    ("the pinned `RadixArk/Qwen3.8-Flash-Next-NVFP4` checkpoint (his Kaggle model asset)",
     "the pinned `RadixArk/Qwen3.8-Flash-Next-NVFP4` checkpoint (his Kaggle model asset; **this arm serves "
     "`nvidia/Qwen3.8-Flash-Next-NVFP4` instead**, see cell 0)"),
    ("- **Weights** — RadixArk's NVFP4 quantisation of Qwen/Qwen3.8-Flash-Next (Qwen licence terms apply).",
     "- **Weights** — this arm: NVIDIA's NVFP4 quantisation of Qwen/Qwen3.8-Flash-Next "
     "([nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4), Kaggle mirror "
     "`xiaoz259/qwen3-8-flash-next-nvfp4`). Licensed by NVIDIA Corporation under the NVIDIA Open Model License; "
     "Qwen Community License 1.0 for the base model."),
]


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
    ap.add_argument("--kv-gib", type=float, default=None, help="TAAF_VLLM_KV_CACHE_MEMORY_BYTES in GiB (base 5)")
    ap.add_argument("--prefix-caching", action="store_true", help="TAAF_VLLM_ENABLE_PREFIX_CACHING=1")
    ap.add_argument("--batched-tokens", type=int, default=None, help="TAAF_VLLM_MAX_NUM_BATCHED_TOKENS (base 8192)")
    ap.add_argument("--mtp-tokens", type=int, default=None, choices=range(0, 5),
                    help="TAAF_VLLM_MTP_TOKENS (base 3; 0 = no speculative decoding, the MTP head is not loaded)")
    ap.add_argument("--wavefit", action="store_true")
    ap.add_argument("--lanes", type=int, default=None,
                    help="games played at once (base 28); the public-25 run's per-game cap becomes 7,920 s / its waves so "
                         "the run keeps its length (a real rerun sizes the cap with --wavefit)")
    ap.add_argument("--knob", action="append", default=[], metavar="KEY=VALUE",
                    help="analyzer env override applied with the thui knobs (e.g. LOCAL_ANALYZER_MAX_OUTPUT=6144)")
    ap.add_argument("--model", choices=["radixark", "nvidia"], default="radixark",
                    help="checkpoint to serve (base: RadixArk's, Keith's pin); nvidia patches his serving setup")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    if args.lanes and not args.wavefit:
        raise SystemExit("--lanes needs --wavefit: a rerun would otherwise keep 7,920 s per game over more waves")

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
            serving = [  # (requested, profile key, base value, new value, change note)
                (args.kv_dtype, "TAAF_VLLM_KV_CACHE_DTYPE", "auto", args.kv_dtype, f"KV cache dtype {args.kv_dtype}"),
                (args.max_num_seqs, "TAAF_VLLM_MAX_NUM_SEQS", "8", args.max_num_seqs, f"max_num_seqs {args.max_num_seqs}"),
                (args.kv_gib, "TAAF_VLLM_KV_CACHE_MEMORY_BYTES", "5368709120",
                 int((args.kv_gib or 0) * 1024**3), f"KV cache {args.kv_gib} GiB"),
                (args.prefix_caching, "TAAF_VLLM_ENABLE_PREFIX_CACHING", "0", 1, "prefix caching on"),
                (args.batched_tokens, "TAAF_VLLM_MAX_NUM_BATCHED_TOKENS", "8192", args.batched_tokens,
                 f"max_num_batched_tokens {args.batched_tokens}"),
                (args.mtp_tokens is not None, "TAAF_VLLM_MTP_TOKENS", "3", args.mtp_tokens, f"MTP tokens {args.mtp_tokens}"),
                (args.cudagraph, "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE", "32", args.cudagraph,
                 f"cudagraph capture {args.cudagraph}"),
            ]
            for requested, key, base, value, note in serving:
                if requested:
                    anchor = f'"{key}": "{base}"'
                    if s.count(anchor) != 1:  # a flag whose anchor moved must fail the build, not do nothing
                        raise SystemExit(f"serving anchor {anchor} not found exactly once in the base profile")
                    s = s.replace(anchor, f'"{key}": "{value}"')
                    changes.append(note)
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
        if args.knob and KNOBS_ANCHOR in s:
            knobs = {"LOCAL_ANALYZER_SEED": "20260825", "LOCAL_ANALYZER_YIELD_SECONDS": "180"}
            knobs.update(dict(kv.split("=", 1) for kv in args.knob))
            s = s.replace(KNOBS_ANCHOR, f"_KNOBS = {knobs!r}")
            s = s.replace(YIELD_ASSERT, YIELD_ASSERT.replace('float("180")', f'float({knobs["LOCAL_ANALYZER_YIELD_SECONDS"]!r})'))
            s = s.replace(GRAFT_PRINT, "for _k, _v in _KNOBS.items():\n    assert os.environ[_k] == _v, (_k, os.environ.get(_k))\n"
                          + GRAFT_PRINT)
            changes.append(f"analyzer knobs {knobs}")
        if args.lanes and LANES_ANCHOR in s:
            s = s.replace(LANES_ANCHOR, f"""bm.solver.concurrency = {args.lanes}  # ours (--lanes)
if not TRUE_SUBMISSION:  # the public 25 in waves of {args.lanes}: same total length as 25 games at once for 7,920 s
    bm.solver.max_runtime_s_per_game = 7920.0 / -(-25 // {args.lanes})
""")
            changes.append(f"{args.lanes} lanes (games at once)")
        if args.wavefit and SOFT_END_ANCHOR in s:
            s = s.replace(SOFT_END_ANCHOR, SOFT_END_ANCHOR + WAVEFIT)
            changes.append("wave-fit per-game cap in real reruns")
        if args.model == "nvidia" and BUNDLE_ANCHOR in s:
            if s.count(BUNDLE_ANCHOR) != 1:
                raise SystemExit("bundle anchor found more than once in the base notebook")
            s = s.replace(BUNDLE_ANCHOR, BUNDLE_ANCHOR + NV_SWAP)
            changes.append(f"model {nv.NVIDIA_HF_REPO} (Kaggle model {nv.NVIDIA_MODEL_SOURCE}) served through "
                           "scripts/nvidia_serving_patch.py: --quantization modelopt_mixed and vLLM backports of "
                           "d4d703c (#54882, FP8 PLE) and 60ad959 (#55513, block-FP8 MTP experts) on Keith's runtime")
        if args.model == "nvidia" and cell["cell_type"] == "markdown":
            for old, new in NV_MARKDOWN:
                s = s.replace(old, new)
        if s != orig:
            cell["source"] = [s]
    if args.model == "nvidia":
        text = "\n".join("".join(c["source"]) for c in cells)
        missing = [old for old, new in NV_MARKDOWN if new not in text]
        if missing:
            raise SystemExit(f"--model nvidia: markdown anchors not found: {missing}")
        idx = next(i for i, c in enumerate(cells) if BUNDLE_ANCHOR in "".join(c["source"]))
        cells.insert(idx, code_cell("# ours: source of scripts/nvidia_serving_patch.py, applied in the next cell\n"
                                    f"_NV_SERVING_PATCH_SOURCE = {NV_PATCH_SRC.read_text()!r}\n"))
    if args.patches:
        idx = next(i for i, c in enumerate(cells) if ANIM_ANCHOR in "".join(c["source"]))
        cells.insert(idx, code_cell("# ours: source of scripts/taaf_ours_patch.py, applied in the next cell\n"
                                    f"_OURS_PATCH_SOURCE = {PATCH_SRC.read_text()!r}\n"))
    expected = (bool(args.kv_dtype) + bool(args.max_num_seqs) + bool(args.cudagraph) + bool(args.patches)
                + bool(args.wavefit) + bool(args.knob) + bool(args.kv_gib) + bool(args.prefix_caching)
                + bool(args.batched_tokens) + (args.mtp_tokens is not None) + bool(args.lanes)
                + (args.model == "nvidia"))
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
    if args.model == "nvidia":
        meta["model_sources"] = [nv.NVIDIA_MODEL_SOURCE]
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"built {out / (args.slug + '.ipynb')}: {changes or 'control'}")


if __name__ == "__main__":
    main()
