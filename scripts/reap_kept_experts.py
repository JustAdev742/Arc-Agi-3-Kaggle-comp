#!/usr/bin/env python
"""Recover which routed experts a public REAP-pruned checkpoint kept, by matching router rows (CPU + network only).

    .venv/bin/python -I scripts/reap_kept_experts.py --out kaggle/franzen/reap448_kept_experts.json

A REAP-pruned checkpoint keeps K of E routed experts per MoE layer, renumbers them 0..K-1 and keeps the router rows
of the kept experts bit for bit. Every row of the pruned router therefore equals exactly one row of the unpruned
router, and its position there is the original expert id. This script reads ONLY the router tensors
(``model.language_model.layers.{i}.mlp.gate.weight``, about 2.5 MB per layer) of both checkpoints with HTTP range
reads (the safetensors header first, then the tensor's byte range) and never downloads a whole shard. It requires
an exact byte match for every pruned row, a unique match, and distinct original ids; anything else is an error.

Writes OUT as ``{"<layer>": [sorted original expert ids], ...}`` and OUT with ``.meta.json`` for ``.json``: the
sources, per-layer sha256 of the unpruned router bytes (the loader patch checks these at load time,
scripts/sglang_reap_patch.py), the order in which the pruned checkpoint lists its kept experts, and a check that
the MTP router is untouched. Defaults: the pruned source is lee-chang-93/Qwen3.8-Flash-Next-NVFP4-REAP-k448 and the
unpruned one is Intel's W4A16 AutoRound checkpoint that Franzen's notebook loads (its BF16 routers), both pinned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PRUNED = "lee-chang-93/Qwen3.8-Flash-Next-NVFP4-REAP-k448@8d565c90add22b39cf3632943f859d7a9a411b48"
BASE = "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound@4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897"
ROUTER = "model.language_model.layers.{}.mlp.gate.weight"
MTP_ROUTER = "mtp.layers.0.mlp.gate.weight"
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}
MAX_HEADER = 64 * 1024 * 1024


class Repo:
    """One pinned Hugging Face repository read with HTTP range requests (or local files for tests)."""

    def __init__(self, spec: str, *, local: Path | None = None, index: Path | None = None):
        self.name, _, self.revision = spec.partition("@")
        if not local and not self.revision:
            raise SystemExit(f"{spec}: pin a revision (repo@sha)")
        self.local = local
        self.bytes_read = 0
        self._headers: dict[str, tuple[int, dict]] = {}
        raw = index.read_bytes() if index else self._read("model.safetensors.index.json", None)
        self.weight_map: dict[str, str] = json.loads(raw)["weight_map"]

    def _read(self, filename: str, span: tuple[int, int] | None) -> bytes:
        if self.local is not None:
            with open(self.local / filename, "rb") as f:
                if span is None:
                    return f.read()
                f.seek(span[0])
                return f.read(span[1] - span[0])
        url = f"https://huggingface.co/{self.name}/resolve/{self.revision}/{filename}"
        request = urllib.request.Request(url)
        if span is not None:
            request.add_header("Range", f"bytes={span[0]}-{span[1] - 1}")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = response.read()
                    if span is not None and response.status != 206:
                        raise SystemExit(f"{url}: expected HTTP 206 for a range read, got {response.status}")
                break
            except (OSError, urllib.error.URLError) as exc:  # transient network errors: retry
                if attempt == 3:
                    raise SystemExit(f"{url}: {exc}") from None
                time.sleep(2 * (attempt + 1))
        if span is not None and len(data) != span[1] - span[0]:
            raise SystemExit(f"{url}: asked for {span[1] - span[0]} bytes, got {len(data)}")
        self.bytes_read += len(data)
        return data

    def header(self, filename: str) -> tuple[int, dict]:
        if filename not in self._headers:
            (size,) = struct.unpack("<Q", self._read(filename, (0, 8)))
            if not 0 < size <= MAX_HEADER:
                raise SystemExit(f"{self.name}/{filename}: implausible safetensors header length {size}")
            self._headers[filename] = (8 + size, json.loads(self._read(filename, (8, 8 + size))))
        return self._headers[filename]

    def tensor(self, name: str) -> tuple[list[int], str, bytes]:
        if name not in self.weight_map:
            raise SystemExit(f"{self.name}: {name} is not in model.safetensors.index.json")
        filename = self.weight_map[name]
        data_start, header = self.header(filename)
        meta = header[name]
        begin, end = meta["data_offsets"]
        shape, dtype = meta["shape"], meta["dtype"]
        if dtype not in DTYPE_BYTES or len(shape) != 2 or end - begin != shape[0] * shape[1] * DTYPE_BYTES[dtype]:
            raise SystemExit(f"{self.name}: {name} has unexpected layout {dtype} {shape} ({end - begin} bytes)")
        return shape, dtype, self._read(filename, (data_start + begin, data_start + end))


def rows(shape: list[int], data: bytes) -> list[bytes]:
    width = len(data) // shape[0]
    return [data[i * width:(i + 1) * width] for i in range(shape[0])]


def match_layer(pruned_rows: list[bytes], base_rows: list[bytes]) -> list[int]:
    """Original id of each pruned row (exact byte match, unique in the base, ids distinct)."""
    where: dict[bytes, list[int]] = {}
    for i, row in enumerate(base_rows):
        where.setdefault(row, []).append(i)
    ids = []
    for j, row in enumerate(pruned_rows):
        hits = where.get(row, [])
        if len(hits) != 1:
            raise ValueError(f"pruned row {j} matches {len(hits)} base rows (expected exactly 1)")
        ids.append(hits[0])
    if len(set(ids)) != len(ids):
        raise ValueError("two pruned rows match the same base row")
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pruned", default=PRUNED, help="repo@revision of the pruned checkpoint")
    ap.add_argument("--base", default=BASE, help="repo@revision of the unpruned checkpoint")
    ap.add_argument("--pruned-index", type=Path, help="a local copy of its model.safetensors.index.json")
    ap.add_argument("--base-index", type=Path, help="a local copy of its model.safetensors.index.json")
    ap.add_argument("--pruned-local", type=Path, help=argparse.SUPPRESS)  # tests: a directory of safetensors files
    ap.add_argument("--base-local", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()

    pruned = Repo(args.pruned, local=args.pruned_local, index=args.pruned_index)
    base = Repo(args.base, local=args.base_local, index=args.base_index)
    layers = sorted(int(n.split(".")[3]) for n in pruned.weight_map
                    if n.startswith("model.language_model.layers.") and n.endswith(".mlp.gate.weight"))
    if not layers or layers != list(range(len(layers))):
        raise SystemExit(f"pruned checkpoint routers: unexpected layer ids {layers[:5]}...")

    kept: dict[str, list[int]] = {}
    meta_layers: dict[str, dict] = {}
    for layer in layers:
        p_shape, p_dtype, p_data = pruned.tensor(ROUTER.format(layer))
        b_shape, b_dtype, b_data = base.tensor(ROUTER.format(layer))
        if p_dtype != b_dtype or p_shape[1] != b_shape[1] or p_shape[0] >= b_shape[0]:
            raise SystemExit(f"layer {layer}: pruned router {p_dtype} {p_shape} vs base {b_dtype} {b_shape}")
        try:
            order = match_layer(rows(p_shape, p_data), rows(b_shape, b_data))
        except ValueError as exc:
            raise SystemExit(f"layer {layer}: {exc}") from None
        kept[str(layer)] = sorted(order)
        meta_layers[str(layer)] = {
            "num_experts": b_shape[0], "kept": len(order), "pruned_order_is_sorted": order == sorted(order),
            "base_router_sha256": hashlib.sha256(b_data).hexdigest(),
            "pruned_router_sha256": hashlib.sha256(p_data).hexdigest(),
        }
        print(f"layer {layer:2d}: {len(order)} of {b_shape[0]} rows matched exactly; "
              f"dropped {sorted(set(range(b_shape[0])) - set(order))[:8]}...", flush=True)

    try:  # informational: the pruned card says its MTP block is untouched
        p_shape, p_dtype, p_data = pruned.tensor(MTP_ROUTER)
        b_shape, b_dtype, b_data = base.tensor(MTP_ROUTER)
        mtp_identical = (p_shape, p_dtype, p_data) == (b_shape, b_dtype, b_data)
    except SystemExit as exc:
        p_shape, mtp_identical = None, f"not compared: {exc}"
    meta = {
        "what": "per MoE layer, the original routed-expert ids the pruned checkpoint keeps (sorted); written by "
                "scripts/reap_kept_experts.py from exact router-row matches",
        "pruned_source": args.pruned, "base_source": args.base,
        "num_layers": len(layers), "router_tensor": ROUTER, "router_dtype": b_dtype,
        "every_row_matched_bitwise": True,
        "mtp_router_identical": mtp_identical, "mtp_router_shape": p_shape,
        "bytes_read": {"pruned": pruned.bytes_read, "base": base.bytes_read},
        "layers": meta_layers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(kept, indent=None, separators=(",", ":")) + "\n")
    meta_path = args.out.with_name(args.out.name.removesuffix(".json") + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {args.out} and {meta_path}; {len(layers)} layers; MTP router identical: {mtp_identical}; "
          f"read {pruned.bytes_read / 1e6:.0f} MB + {base.bytes_read / 1e6:.0f} MB")
    if not mtp_identical:
        print("WARNING: the MTP router differs between the two checkpoints", file=sys.stderr)


if __name__ == "__main__":
    main()
