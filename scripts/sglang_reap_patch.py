"""REAP expert pruning at load time for the Flash-Next target in Pennyroyal SGLang (sglang-0.5.19+gd00d88efc8d6).

One file, two roles:

1. **Runtime module.** ``apply`` copies this file into the installed package as ``sglang/srt/arc3_reap.py``. When
   the server process has ``ARC3_REAP_KEPT_EXPERTS=/path/kept.json`` (``{"<layer>": [sorted original expert ids]}``,
   kaggle/franzen/reap448_kept_experts.json), :func:`wrap_weights` filters the target checkpoint's weight stream:
   expert ``j`` of layer ``i`` becomes slot ``kept[i].index(j)``, pruned experts are dropped, and the router
   ``...layers.{i}.mlp.gate.weight`` keeps only the kept rows, in kept order. Without the variable it returns the
   stream unchanged. It checks everything it relies on and raises (the server fails to start) when a check fails:
   the model must be built with ``num_experts == K`` (``--json-model-override-args
   '{"text_config":{"num_experts":K}}'``), every router must have more rows than the largest kept id, every layer
   must deliver the same tensors for every kept slot and every pruned expert, and when ``<kept>.meta.json`` exists
   each full router must match the sha256 recorded when the list was made (scripts/reap_kept_experts.py), so the
   list can only be applied to the checkpoint it was derived for.
2. **Installer** (``python -I scripts/sglang_reap_patch.py apply --site-packages SP --kept kept.json``): writes the
   module and makes two anchored edits to ``SP/sglang/srt/models/qwen4_exp.py``. It refuses a file whose sha256 is
   not the analysed wheel's, an anchor that does not occur exactly once, or a kept list that does not validate,
   and exits non-zero (the notebook cell raises). Running it again on a patched tree is a no-op.

The two edits, both active only with the variable set:

- ``Qwen4ExpForConditionalGeneration.load_weights`` (the target; the MTP draft is ``Qwen4ExpForCausalLMMTP`` and
  loads through ``Qwen3_5ForCausalLMMTP.load_weights``, untouched) starts with ``weights = wrap_weights(...)``.
- ``get_model_config_for_expert_location`` returns None, so no global expert-location map is built. The draft's
  MoE layer has ``layer_id`` 0 and 512 experts, and its ``FusedMoE.weight_loader`` looks experts up in the
  *target's* map, which would be 448 wide (IndexError for ids 448-511). Without a map both models take the
  loader's ``metadata is None`` path, which gives the same local ids at EP 1. Nothing else reads the map in this
  configuration (no EPLB, no expert-distribution recorder, no dispatch algorithm, no DeepEP).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import os
import re
import sys
from pathlib import Path

ENV = "ARC3_REAP_KEPT_EXPERTS"
MODEL_FILE = Path("sglang/srt/models/qwen4_exp.py")
MODULE_FILE = Path("sglang/srt/arc3_reap.py")
# sglang/srt/models/qwen4_exp.py in dfranzen/pennyroyal-v253 wheels/sglang-0.5.19+gd00d88efc8d6-cp312-cp312-linux_x86_64.whl
# (wheel sha256 d0620216732b82b6efd4f8a78cc9b5f55343a4d3837fdfa3bcbb841b7718a225), the file this patch was written against
MODEL_FILE_SHA256 = "35a1785ce4c287821f00f1518b3880ab6838ef58d42fd51ece86d8098b2ed666"
MARK = "# >>> arc3 REAP (scripts/sglang_reap_patch.py)"
END = "# <<< arc3 REAP"

EXPERT_RE = re.compile(r"^(model\.(?:language_model\.)?layers\.(\d+)\.mlp\.experts\.)(\d+)(\..+)$")
FUSED_EXPERT_RE = re.compile(r"^model\.(?:language_model\.)?layers\.\d+\.mlp\.experts\.(?:gate_up_proj|down_proj)")
ROUTER_RE = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.mlp\.gate\.weight$")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------------------------------- runtime


def active() -> bool:
    return bool(os.environ.get(ENV, "").strip())


def load_kept(path: str | os.PathLike) -> dict[int, list[int]]:
    """The kept list: layers 0..L-1, each a strictly increasing list of the same length K of ids >= 0."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: expected a non-empty JSON object {{layer: [expert ids]}}")
    kept: dict[int, list[int]] = {}
    for key, ids in raw.items():
        if not (isinstance(key, str) and key.isdigit()):
            raise ValueError(f"{path}: layer key {key!r} is not a non-negative integer")
        if not (isinstance(ids, list) and ids and all(isinstance(e, int) and not isinstance(e, bool) for e in ids)):
            raise ValueError(f"{path}: layer {key} must be a non-empty list of integers")
        if ids[0] < 0 or any(b <= a for a, b in itertools.pairwise(ids)):
            raise ValueError(f"{path}: layer {key} ids must be >= 0 and strictly increasing")
        kept[int(key)] = ids
    if sorted(kept) != list(range(len(kept))):
        raise ValueError(f"{path}: layers must be 0..{len(kept) - 1} without gaps")
    if len({len(ids) for ids in kept.values()}) != 1:
        raise ValueError(f"{path}: every layer must keep the same number of experts")
    return kept


def meta_path(path: str | os.PathLike) -> Path:
    path = Path(path)
    return path.with_name(path.name.removesuffix(".json") + ".meta.json")


def load_router_sha256(path: str | os.PathLike) -> dict[int, str]:
    """Per-layer sha256 of the full (unpruned) router bytes from <kept>.meta.json, or {} when there is none."""
    meta = meta_path(path)
    if not meta.is_file():
        return {}
    layers = json.loads(meta.read_text())["layers"]
    return {int(k): v["base_router_sha256"] for k, v in layers.items()}


def slot_maps(kept: dict[int, list[int]]) -> dict[int, dict[int, int]]:
    """{layer: {original expert id: slot}}; slots follow the sorted kept list (as the pruned checkpoint numbers them)."""
    return {layer: {expert: slot for slot, expert in enumerate(ids)} for layer, ids in kept.items()}


def _is_torch(tensor) -> bool:
    return type(tensor).__module__.split(".")[0] == "torch"


def take_rows(tensor, rows: list[int]):
    """tensor[rows] along dim 0 (torch on any device, or numpy for tests), as a new contiguous tensor."""
    if _is_torch(tensor):
        import torch

        index = torch.as_tensor(rows, dtype=torch.long, device=tensor.device)
        return tensor.index_select(0, index).contiguous()
    return tensor[rows].copy()


def tensor_sha256(tensor) -> str:
    """sha256 of the tensor's bytes in row-major order (= its safetensors payload for a contiguous checkpoint tensor)."""
    if _is_torch(tensor):
        import torch

        data = tensor.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    else:
        import numpy as np

        data = np.ascontiguousarray(tensor).tobytes()
    return hashlib.sha256(data).hexdigest()


def wrap_weights(weights, config):
    """The target's weight stream, filtered to the kept experts when ARC3_REAP_KEPT_EXPERTS is set (else unchanged)."""
    if not active():
        return weights
    path = os.environ[ENV].strip()
    kept = load_kept(path)
    num_kept = len(kept[0])
    if getattr(config, "num_experts", None) != num_kept:
        raise RuntimeError(f"ARC3 REAP: the model was built with num_experts={getattr(config, 'num_experts', None)} "
                           f"but {path} keeps {num_kept} per layer; launch with --json-model-override-args "
                           f"'{{\"text_config\": {{\"num_experts\": {num_kept}}}}}'")
    if getattr(config, "num_hidden_layers", None) != len(kept):
        raise RuntimeError(f"ARC3 REAP: {path} lists {len(kept)} layers, the model has "
                           f"{getattr(config, 'num_hidden_layers', None)}")
    return _filter(weights, kept, load_router_sha256(path), path)


def _filter(weights, kept: dict[int, list[int]], router_sha256: dict[int, str], path: str):
    slots = slot_maps(kept)
    per_slot: dict[int, dict[int, set[str]]] = {layer: {} for layer in kept}    # layer -> slot -> tensor suffixes
    per_pruned: dict[int, dict[int, set[str]]] = {layer: {} for layer in kept}  # layer -> expert -> suffixes
    routers: dict[int, int] = {}                                                # layer -> rows in the checkpoint
    for name, tensor in weights:
        if FUSED_EXPERT_RE.match(name):
            raise RuntimeError(f"ARC3 REAP: fused expert tensor {name} is not supported (per-expert tensors only)")
        match = EXPERT_RE.match(name)
        if match:
            prefix, layer, expert, suffix = match.group(1), int(match.group(2)), int(match.group(3)), match.group(4)
            if layer not in kept:
                raise RuntimeError(f"ARC3 REAP: {name} belongs to layer {layer}, not in {path}")
            slot = slots[layer].get(expert)
            if slot is None:
                per_pruned[layer].setdefault(expert, set()).add(suffix)
                continue
            per_slot[layer].setdefault(slot, set()).add(suffix)
            yield f"{prefix}{slot}{suffix}", tensor
            continue
        match = ROUTER_RE.match(name)
        if match:
            layer = int(match.group(1))
            if layer not in kept or layer in routers:
                raise RuntimeError(f"ARC3 REAP: unexpected or repeated router {name}")
            rows = int(tensor.shape[0])
            if rows <= kept[layer][-1]:
                raise RuntimeError(f"ARC3 REAP: {name} has {rows} rows but the kept list names expert "
                                   f"{kept[layer][-1]}")
            if layer in router_sha256 and tensor_sha256(tensor) != router_sha256[layer]:
                raise RuntimeError(f"ARC3 REAP: {name} differs from the router the kept list was derived from "
                                   f"({meta_path(path).name}); this checkpoint needs its own kept list")
            routers[layer] = rows
            yield name, take_rows(tensor, kept[layer])
            continue
        yield name, tensor

    problems = []
    for layer, ids in kept.items():
        if layer not in routers:
            problems.append(f"layer {layer}: no router")
            continue
        expected_pruned = routers[layer] - len(ids)
        suffix_sets = list(per_slot[layer].values()) + list(per_pruned[layer].values())
        if len(per_slot[layer]) != len(ids):
            problems.append(f"layer {layer}: {len(per_slot[layer])} of {len(ids)} kept experts loaded")
        if len(per_pruned[layer]) != expected_pruned:
            problems.append(f"layer {layer}: {len(per_pruned[layer])} pruned experts seen, expected {expected_pruned}")
        if suffix_sets and any(s != suffix_sets[0] for s in suffix_sets):
            problems.append(f"layer {layer}: experts do not all have the same tensors")
    if problems:
        raise RuntimeError("ARC3 REAP: the checkpoint did not match the kept list: " + "; ".join(problems[:6]))
    tensors = sum(len(s) for layer in per_slot.values() for s in layer.values())
    skipped = sum(len(s) for layer in per_pruned.values() for s in layer.values())
    logger.info("ARC3 REAP: kept %d of %d routed experts in each of %d layers (%d expert tensors loaded, %d pruned "
                "tensors skipped); routers sliced to %d rows%s; list %s", len(kept[0]), routers[0], len(kept), tensors,
                skipped, len(kept[0]), ", router sha256 verified" if router_sha256 else "", path)


# -------------------------------------------------------------------------------------------------------- installer

LOAD_ANCHOR = ("    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):\n"
               "        stacked_params_mapping = [\n")
LOAD_NEW = ("    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):\n"
            f"        {MARK}: with {ENV} set, only the listed routed\n"
            "        # experts (renumbered 0..K-1) and their router rows are loaded; unset, weights pass through unchanged.\n"
            "        from sglang.srt import arc3_reap as _arc3_reap\n"
            "\n"
            "        weights = _arc3_reap.wrap_weights(weights, self.config)\n"
            f"        {END}\n"
            "        stacked_params_mapping = [\n")
LOCATION_ANCHOR = ('        text_config = getattr(config, "text_config", config)\n'
                   '        if getattr(text_config, "num_experts", None) is None:\n'
                   "            return None\n")
LOCATION_NEW = (LOCATION_ANCHOR
                + f"        {MARK}: no expert-location map while pruned, so\n"
                "        # the 512-expert MTP draft (layer_id 0) is not looked up in a map sized for the pruned target.\n"
                "        from sglang.srt import arc3_reap as _arc3_reap\n"
                "\n"
                "        if _arc3_reap.active():\n"
                "            return None\n"
                f"        {END}\n")
EDITS = ((LOAD_ANCHOR, LOAD_NEW, "Qwen4ExpForConditionalGeneration.load_weights"),
         (LOCATION_ANCHOR, LOCATION_NEW, "get_model_config_for_expert_location"))


class PatchError(Exception):
    pass


def patch_text(text: str, *, check_hash: bool = True) -> str:
    """qwen4_exp.py with both edits; refuses another file version and anchors that do not occur exactly once."""
    if MARK in text:
        raise PatchError("already patched")
    if check_hash and hashlib.sha256(text.encode()).hexdigest() != MODEL_FILE_SHA256:
        raise PatchError(f"{MODEL_FILE} is not the analysed Pennyroyal file (sha256 "
                         f"{hashlib.sha256(text.encode()).hexdigest()[:12]}..., expected {MODEL_FILE_SHA256[:12]}...); "
                         f"re-check the loader before patching another version")
    for anchor, new, what in EDITS:
        if text.count(anchor) != 1:
            raise PatchError(f"anchor for {what} found {text.count(anchor)} times (expected once)")
        text = text.replace(anchor, new)
    compile(text, str(MODEL_FILE), "exec")  # syntax only; nothing is imported or run
    return text


def apply(site_packages: Path, kept: Path | None = None, *, check_hash: bool = True) -> str:
    """Patch an installed sglang in SITE_PACKAGES (idempotent). Returns what was done."""
    target, module = site_packages / MODEL_FILE, site_packages / MODULE_FILE
    if not target.is_file():
        raise PatchError(f"{target} not found")
    if kept is not None:
        ids = load_kept(kept)
        meta = load_router_sha256(kept)
        if meta and sorted(meta) != sorted(ids):
            raise PatchError(f"{meta_path(kept)} does not cover the same layers as {kept}")
    source = Path(__file__).read_text()
    compile(source, str(MODULE_FILE), "exec")
    text = target.read_text()
    if MARK in text:
        original = text
        for anchor, edit, _ in EDITS:
            original = original.replace(edit, anchor) if text.count(edit) == 1 else ""
        if not original or MARK in original or (
                check_hash and hashlib.sha256(original.encode()).hexdigest() != MODEL_FILE_SHA256):
            raise PatchError(f"{target} carries a different or partial arc3 REAP patch")
        state = "already patched"
    else:
        target.write_text(patch_text(text, check_hash=check_hash))
        state = "patched"
        for stale in (target.parent / "__pycache__").glob("qwen4_exp.*.pyc"):
            stale.unlink()
    if not module.is_file() or module.read_text() != source:
        module.write_text(source)
    return (f"arc3 REAP: {target} {state}; module {module}"
            + (f"; kept list {kept}: {len(ids)} layers x {len(ids[0])} experts"
               f"{', router sha256 for every layer' if meta else ''}" if kept is not None else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("apply", help="patch an installed sglang (site-packages directory)")
    a.add_argument("--site-packages", type=Path, required=True)
    a.add_argument("--kept", type=Path, help="validate this kept list too")
    a.add_argument("--allow-other-version", action="store_true", help="skip the file hash check (anchors still apply)")
    c = sub.add_parser("check-wheel", help="apply the edits to the wheel's qwen4_exp.py in memory (no install)")
    c.add_argument("wheel", type=Path)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "apply":
            print(apply(args.site_packages, args.kept, check_hash=not args.allow_other_version))
        else:
            import zipfile

            with zipfile.ZipFile(args.wheel) as wheel:
                text = wheel.read(str(MODEL_FILE)).decode()
            patched = patch_text(text)
            print(f"arc3 REAP: {args.wheel.name}: {MODEL_FILE} matches sha256 {MODEL_FILE_SHA256[:12]}..., both anchors "
                  f"found once, patched file compiles ({len(patched) - len(text)} bytes added)")
    except (PatchError, ValueError, OSError, KeyError, SyntaxError) as exc:
        print(f"arc3 REAP patch FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
