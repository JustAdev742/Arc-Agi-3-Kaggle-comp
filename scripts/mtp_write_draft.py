#!/usr/bin/env python
"""Write a fine-tuned MTP draft directory that Pennyroyal's launcher accepts, and check one (plan section 4).

    python -I scripts/mtp_write_draft.py write --draft ALBUCINO_DIR --trained TRAIN_OUT/trained-dense.safetensors \\
        --out OUT_DIR [--tokenizer-from draft|target|none] [--target-dir MODEL_DIR] [--overwrite]
    python -I scripts/mtp_write_draft.py check OUT_DIR [--reference ALBUCINO_DIR] [--trained TRAINED]

``write`` copies albucino's draft (``runtime/mtp-int4-g32``: config.json, model.safetensors.index.json,
mtp-dense.safetensors, mtp-routed-experts-int4.safetensors and its small tokenizer/processor files) and rewrites only
the data of the dense tensors named in ``--trained`` (BF16, same names and shapes; scripts/mtp_train.py writes that
file). The dense file keeps its header byte for byte (same names, dtypes, shapes, offsets and order), so every
frozen tensor and the 2.5 GB lm_head/embed copies stay byte-identical; config.json, the index and the INT4 experts are
plain copies. ``--tokenizer-from target`` takes the tokenizer files from the target instead (the launcher links the
target's into its view anyway); ``none`` leaves them out. A manifest ``arc3-draft-manifest.json`` lists every file's
sha256 and which tensors changed.

``check`` validates a directory against the launcher's ``prepare_draft_view`` and ``indexed_shards`` (Franzen's cell
12, kaggle/franzen/arc-agi-3-milestone-2-solution.ipynb):
1. exactly one config.json under the directory (searched recursively) with ``quantization_config.quant_method``
   ``compressed-tensors`` and a ``mtp_routed_experts`` config group;
2. that group's weights are ``num_bits`` 4, ``group_size`` 32, ``symmetric`` true; ``targets`` is ``["RoutedExperts"]``
   or the launcher's EXPERT_TARGET regex; ``ignore`` is ``[]`` or DENSE_IGNORE;
3. model.safetensors.index.json is non-empty, names at least one ``mtp.layers.0.mlp.experts.`` key, and every shard it
   names is a relative path without ``..``, exists and has at least 8 bytes;
and further, what SGLang's loader will read: every indexed tensor is in its shard's header with data inside the file;
the 29 dense ``mtp.*`` tensors have the shapes the config implies, are BF16 and finite; the experts are complete
(``weight_packed`` I32 [out, in/8], ``weight_scale`` BF16 [out, in/group], ``weight_shape`` I64 = [out, in]). With
``--reference`` (albucino's directory): config, index and expert shards are byte-identical, the dense shard has the same
header bytes, and the tensors whose bytes differ are listed (with ``--trained``, they must be among its names).

Stdlib + numpy (runs with ``python -I`` in any environment of ours). Exit status 1 on any problem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

EXPERT_TARGET = r"re:^mtp\.layers\.0\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$"  # cell 12
DENSE_IGNORE = r"re:^(?!mtp\.layers\.0\.mlp\.experts(?:\.|$)).*"  # cell 12
INDEX = "model.safetensors.index.json"
MANIFEST = "arc3-draft-manifest.json"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json",
                   "generation_config.json", "video_preprocessor_config.json", "vocab.json")
DTYPE_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "U8": 1, "I8": 1, "F8_E4M3": 1, "BOOL": 1}
L0 = "mtp.layers.0."


def dense_layout(text_config: dict) -> dict[str, tuple]:
    """The 29 dense mtp.* tensors and their shapes (mirrors scripts/mtp_replica.py dense_shapes; tested equal)."""
    t = text_config
    H, hc, R = int(t["hidden_size"]), int(t.get("hc_count", 4)), int(t.get("hc_lowrank", 320))
    W = hc * H
    heads, kvh = int(t["num_attention_heads"]), int(t["num_key_value_heads"])
    D = int(t.get("head_dim") or H // heads)
    E, inter = int(t["num_experts"]), int(t.get("shared_expert_intermediate_size") or t["moe_intermediate_size"])
    ih, idim = int(t.get("indexer_n_heads", 4)), int(t.get("indexer_head_dim", 128))
    out = {"mtp.fc_embedding.weight": (H, H), "mtp.fc_hidden.weight": (H, H)}
    out.update({"mtp.hyper_connection_mixer.hc_norm.weight": (W,),
                "mtp.hyper_connection_mixer.input_mix_weight_down.weight": (R, W),
                "mtp.hyper_connection_mixer.input_mix_weight_up.weight": (W, R)})
    for part in ("attn_hyper_connection.", "mlp_hyper_connection."):
        out.update({L0 + part + "block_inject_weight.weight": (hc, W), L0 + part + "hc_norm.weight": (W,),
                    L0 + part + "input_mix_weight_down.weight": (R, W), L0 + part + "input_mix_weight_up.weight": (W, R)})
    out.update({L0 + "mlp.gate.weight": (E, H), L0 + "mlp.shared_expert.down_proj.weight": (H, inter),
                L0 + "mlp.shared_expert.gate_proj.weight": (inter, H),
                L0 + "mlp.shared_expert.up_proj.weight": (inter, H),
                L0 + "mlp.shared_expert_gate.weight": (1, H),
                L0 + "self_attn.indexer.index_qk_proj.weight": ((ih + 1) * idim, H),
                L0 + "self_attn.indexer.k_layernorm.weight": (idim,), L0 + "self_attn.indexer.q_layernorm.weight": (idim,),
                L0 + "self_attn.k_norm.weight": (D,), L0 + "self_attn.k_proj.weight": (kvh * D, H),
                L0 + "self_attn.o_proj.weight": (H, heads * D), L0 + "self_attn.q_norm.weight": (D,),
                L0 + "self_attn.q_proj.weight": (2 * heads * D, H), L0 + "self_attn.v_proj.weight": (kvh * D, H),
                "mtp.pre_fc_norm_embedding.weight": (H,), "mtp.pre_fc_norm_hidden.weight": (W,)})
    return out


def expert_layout(text_config: dict, group: int) -> dict[str, tuple]:
    t = text_config
    H, inter, E = int(t["hidden_size"]), int(t["moe_intermediate_size"]), int(t["num_experts"])
    out = {}
    for e in range(E):
        for proj, (o, i) in (("gate_proj", (inter, H)), ("up_proj", (inter, H)), ("down_proj", (H, inter))):
            p = f"{L0}mlp.experts.{e}.{proj}."
            out[p + "weight_packed"] = ("I32", (o, i // 8))
            out[p + "weight_scale"] = ("BF16", (o, i // group))
            out[p + "weight_shape"] = ("I64", (2,), (o, i))
    return out


# ---------------------------------------------------------------------------------------------- safetensors


def read_header(path) -> tuple[dict, int, bytes]:
    """(header with ``__metadata__`` removed, data offset, the raw header bytes)."""
    with open(path, "rb") as f:
        raw_size = f.read(8)
        if len(raw_size) < 8:
            raise ValueError(f"{path}: shorter than a safetensors header")
        size = struct.unpack("<Q", raw_size)[0]
        raw = f.read(size)
    header = json.loads(raw)
    header.pop("__metadata__", None)
    return header, 8 + size, raw


def read_tensor_bytes(path, info: dict, base: int) -> bytes:
    begin, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + begin)
        return f.read(end - begin)


def read_tensors(path) -> dict[str, tuple[str, tuple, bytes]]:
    header, base, _ = read_header(path)
    return {name: (info["dtype"], tuple(info["shape"]), read_tensor_bytes(path, info, base))
            for name, info in header.items()}


def bf16_finite(data: bytes) -> bool:
    bits = np.frombuffer(data, dtype="<u2")
    return not bool(((bits & 0x7F80) == 0x7F80).any())


def sha256(path, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------- the launcher's rules


def draft_candidates(root) -> tuple[list[Path], list[Path]]:
    """(config.json files the launcher would accept, every config.json under root) - prepare_draft_view's search."""
    root = Path(root)
    configs = [root / "config.json"] if (root / "config.json").is_file() else []
    configs += sorted(p for p in root.rglob("config.json") if p not in configs)
    ok = []
    for path in configs:
        try:
            q = json.loads(path.read_text()).get("quantization_config", {})
        except (ValueError, OSError):
            continue
        if q.get("quant_method") == "compressed-tensors" and "mtp_routed_experts" in q.get("config_groups", {}):
            ok.append(path)
    return ok, configs


def launcher_problems(source: Path) -> list[str]:
    """Rules 2 and 3 of the module docstring for the draft whose config.json is in ``source``."""
    problems = []
    cfg = json.loads((source / "config.json").read_text())
    q = cfg["quantization_config"]
    group = q["config_groups"]["mtp_routed_experts"]
    w = group.get("weights") or {}
    if (w.get("num_bits"), w.get("group_size"), w.get("symmetric")) != (4, 32, True):
        problems.append(f"experts are not symmetric INT4 g32: {w.get('num_bits')}, {w.get('group_size')}, "
                        f"{w.get('symmetric')}")
    if group.get("targets") not in (["RoutedExperts"], [EXPERT_TARGET]):
        problems.append(f"unexpected expert targets {group.get('targets')}")
    if q.get("ignore", []) not in ([], [DENSE_IGNORE]):
        problems.append(f"unexpected ignore rules {q.get('ignore')}")
    index_path = source / INDEX
    if not index_path.is_file():
        return [*problems, f"{INDEX} missing"]
    weight_map = json.loads(index_path.read_text()).get("weight_map") or {}
    names = sorted(set(weight_map.values()))
    if not names:
        problems.append("empty weight index")
    if not any(k.startswith(L0 + "mlp.experts.") for k in weight_map):
        problems.append("the index has no mtp.layers.0.mlp.experts. key")
    for name in names:
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            problems.append(f"invalid shard path {name}")
            continue
        p = source / rel
        if not p.is_file() or p.stat().st_size < 8:
            problems.append(f"missing or empty shard {name}")
    return problems


def check_draft_dir(root, *, reference=None, trained_names=None, check_finite: bool = True) -> dict:
    """All checks of the module docstring; ``ok`` is False when any problem was found."""
    root = Path(root)
    problems, notes = [], []
    ok, configs = draft_candidates(root)
    if len(ok) != 1:
        return {"ok": False, "problems": [f"expected exactly one INT4 g32 MTP config.json under {root}, found "
                                          f"{len(ok)}"], "notes": notes}
    if len(configs) > 1:
        notes.append(f"{len(configs) - 1} other config.json file(s) under the directory (the launcher ignores them)")
    source = ok[0].parent
    problems += launcher_problems(source)
    if problems:
        return {"ok": False, "problems": problems, "notes": notes, "source": str(source)}
    cfg = json.loads((source / "config.json").read_text())
    text = cfg.get("text_config") or cfg
    group = int(cfg["quantization_config"]["config_groups"]["mtp_routed_experts"]["weights"]["group_size"])
    weight_map = json.loads((source / INDEX).read_text())["weight_map"]
    headers, bad = {}, set()
    for shard in sorted(set(weight_map.values())):
        try:
            header, base, raw = read_header(source / shard)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            problems.append(f"{shard}: unreadable header ({exc})")
            continue
        size = (source / shard).stat().st_size
        for name, info in header.items():
            begin, end = info["data_offsets"]
            need = int(np.prod(info["shape"], dtype=np.int64)) * DTYPE_SIZE.get(info["dtype"], 0)
            if end - begin != need or base + end > size:
                problems.append(f"{shard}: {name} has a bad data range")
                bad.add(name)
        headers[shard] = (header, base, raw)
    for key, shard in weight_map.items():
        if shard in headers and key not in headers[shard][0]:
            problems.append(f"{key} is indexed in {shard} but not stored there")
    stored = {n for h, _, _ in headers.values() for n in h}
    if stored - set(weight_map):
        notes.append(f"{len(stored - set(weight_map))} stored tensor(s) not in the index (never loaded)")

    def info_of(name):
        shard = weight_map.get(name)
        if shard is None or shard not in headers or name not in headers[shard][0]:
            return None, None, None
        header, base, _ = headers[shard]
        return shard, header[name], base

    for name, shape in dense_layout(text).items():
        shard, info, base = info_of(name)
        if info is None:
            problems.append(f"dense tensor {name} missing")
            continue
        if info["dtype"] != "BF16" or tuple(info["shape"]) != shape:
            problems.append(f"{name}: {info['dtype']} {info['shape']}, expected BF16 {list(shape)}")
        elif check_finite and name not in bad and not bf16_finite(read_tensor_bytes(source / shard, info, base)):
            problems.append(f"{name} has NaN or Inf values")
    missing = 0
    for name, spec in expert_layout(text, group).items():
        shard, info, base = info_of(name)
        if info is None:
            missing += 1
            continue
        if info["dtype"] != spec[0] or tuple(info["shape"]) != spec[1]:
            problems.append(f"{name}: {info['dtype']} {info['shape']}, expected {spec[0]} {list(spec[1])}")
        elif len(spec) == 3 and name not in bad and \
                tuple(np.frombuffer(read_tensor_bytes(source / shard, info, base), "<i8")) != spec[2]:
            problems.append(f"{name} does not hold {list(spec[2])}")
    if missing:
        problems.append(f"{missing} expert tensor(s) missing")
    result = {"source": str(source), "shards": sorted(headers), "tensors": len(weight_map)}
    if reference is not None:
        result.update(compare_with_reference(source, Path(reference), weight_map, headers, trained_names, problems))
    result.update({"ok": not problems, "problems": problems, "notes": notes})
    return result


def compare_with_reference(source: Path, reference: Path, weight_map: dict, headers: dict, trained_names,
                           problems: list) -> dict:
    ref_ok, _ = draft_candidates(reference)
    if len(ref_ok) != 1:
        problems.append(f"the reference {reference} is not a single MTP draft")
        return {}
    ref = ref_ok[0].parent
    for name in ("config.json", INDEX):
        if sha256(source / name) != sha256(ref / name):
            problems.append(f"{name} differs from the reference")
    dense_shard = weight_map.get("mtp.fc_embedding.weight")
    changed = []
    for shard, (header, base, raw) in headers.items():
        if not (ref / shard).is_file():
            problems.append(f"{shard} is not in the reference")
            continue
        if shard != dense_shard:
            if sha256(source / shard) != sha256(ref / shard):
                problems.append(f"{shard} differs from the reference (only the dense shard may change)")
            continue
        ref_header, ref_base, ref_raw = read_header(ref / shard)
        if raw != ref_raw:
            problems.append(f"{shard}: header differs from the reference (names, dtypes, shapes, offsets or order)")
            continue
        for name, info in header.items():
            if read_tensor_bytes(source / shard, info, base) != read_tensor_bytes(ref / shard, ref_header[name], ref_base):
                changed.append(name)
    if trained_names is not None:
        extra = sorted(set(changed) - set(trained_names))
        if extra:
            problems.append(f"tensors changed that were not trained: {extra}")
    return {"changed": changed, "reference": str(ref)}


# ----------------------------------------------------------------------------------------------------- writing


def write_draft(draft_dir, trained_file, out_dir, *, tokenizer_from: str = "draft", target_dir=None,
                overwrite: bool = False) -> dict:
    ok, _ = draft_candidates(draft_dir)
    if len(ok) != 1:
        raise ValueError(f"expected exactly one INT4 g32 MTP config.json under {draft_dir}, found {len(ok)}")
    source = ok[0].parent
    problems = launcher_problems(source)
    if problems:
        raise ValueError(f"the source draft fails the launcher's checks: {problems}")
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} is not empty (use --overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    weight_map = json.loads((source / INDEX).read_text())["weight_map"]
    dense_shard = weight_map.get("mtp.fc_embedding.weight")
    if dense_shard is None:
        raise ValueError("the draft index has no mtp.fc_embedding.weight")
    header, base, _ = read_header(source / dense_shard)
    trained = read_tensors(trained_file)
    for name, (dtype, shape, data) in trained.items():
        info = header.get(name)
        if info is None or weight_map.get(name) != dense_shard:
            raise ValueError(f"{name} is not a tensor of {dense_shard}")
        if dtype != info["dtype"] or list(shape) != info["shape"]:
            raise ValueError(f"{name}: {dtype} {list(shape)}, the draft has {info['dtype']} {info['shape']}")
        if dtype == "BF16" and not bf16_finite(data):
            raise ValueError(f"{name} has NaN or Inf values")
    for name in [*sorted(set(weight_map.values())), "config.json", INDEX]:
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, dest)
    with open(out / dense_shard, "r+b") as f:
        for name, (_, _, data) in trained.items():
            f.seek(base + header[name]["data_offsets"][0])
            f.write(data)
    copied = []
    if tokenizer_from != "none":
        src = source if tokenizer_from == "draft" else Path(target_dir or "")
        if tokenizer_from == "target" and not src.is_dir():
            raise ValueError("--tokenizer-from target needs --target-dir")
        for name in TOKENIZER_FILES + (("compact_sources.json",) if tokenizer_from == "draft" else ()):
            if (src / name).is_file():
                shutil.copyfile(src / name, out / name)
                copied.append(name)
    manifest = {"format": "arc3-mtp-draft", "source": str(source), "trained_file": str(trained_file),
                "trained_sha256": sha256(trained_file), "replaced": sorted(trained), "copied_extra": copied,
                "files": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != MANIFEST}}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--draft", type=Path, required=True, help="albucino's draft directory (or a folder holding it)")
    w.add_argument("--trained", type=Path, required=True, help="trained-dense.safetensors from scripts/mtp_train.py")
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--tokenizer-from", choices=("draft", "target", "none"), default="draft")
    w.add_argument("--target-dir", type=Path, default=None)
    w.add_argument("--overwrite", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("dir", type=Path)
    c.add_argument("--reference", type=Path, default=None, help="albucino's draft directory, for byte identity")
    c.add_argument("--trained", type=Path, default=None, help="only these tensors may differ from the reference")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "write":
            manifest = write_draft(args.draft, args.trained, args.out, tokenizer_from=args.tokenizer_from,
                                   target_dir=args.target_dir, overwrite=args.overwrite)
            print(f"wrote {args.out}: {len(manifest['replaced'])} dense tensors replaced, "
                  f"{len(manifest['files'])} files")
            result = check_draft_dir(args.out, reference=args.draft, trained_names=manifest["replaced"])
        else:
            names = list(read_header(args.trained)[0]) if args.trained else None
            result = check_draft_dir(args.dir, reference=args.reference, trained_names=names)
    except (ValueError, OSError, KeyError, FileExistsError) as exc:
        print(f"mtp_write_draft: FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in result.items() if k != "shards"}, indent=1))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
