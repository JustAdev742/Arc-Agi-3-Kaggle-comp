#!/usr/bin/env python
"""Serve nvidia/Qwen3.8-Flash-Next-NVFP4 instead of RadixArk's build on Keith Tyser's pinned vLLM runtime.

    apply(<his mounted bundle>, <writable copy>) -> summary      # what the notebook runs (build_taaf_nb.py --model nvidia)

Keith Tyser's serving bundle (dataset keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1, MIT) pins RadixArk's ModelOpt
NVFP4 checkpoint. NVIDIA's checkpoint (HF nvidia/Qwen3.8-Flash-Next-NVFP4 @ fc694b54, Kaggle model
xiaoz259/qwen3-8-flash-next-nvfp4/PyTorch/nvidia-nvfp4/1; NVIDIA Open Model License over the Qwen Community License 1.0)
has the same main-model tensors (296,444 names, same dtypes and shapes, same FP8 PLE table) but a different
quantization config (ModelOpt MIXED_PRECISION with per-layer ``quantized_layers`` instead of plain NVFP4) and an MTP head
whose routed experts are 128x128 block FP8 (RadixArk's are BF16). His runtime (vLLM 0.1.dev20073+g8e685d198, image
files dated 2026-08-26) predates the two upstream fixes NVIDIA's model card requires, so this module:

1. copies his bundle to a writable directory and rewrites its ``serving_setup.py`` (clearly delimited ``ours`` block):
   NVIDIA model identity, mount path and 25-file manifest (sizes always, sha256 of the small files in fast start, of
   every file in full mode), ``--quantization modelopt_mixed``, and a step that patches the extracted runtime;
2. that step backports into the runtime's vLLM files, each checked against a pinned before/after sha256:
   - ``vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py``: vLLM d4d703c (#54882), FP8 PLE embedding for a
     MIXED_PRECISION ModelOpt config (without it the FP8 PLE table would load through the unquantized path);
   - ``vllm/model_executor/layers/quantization/modelopt.py`` and ``.../nvidia/mtp.py``: vLLM 60ad959 (#55513), block-FP8
     routed experts in MIXED_PRECISION configs and the MTP layer-index remap of ``quantized_layers``;
3. updates the copy's SOURCE_IDENTITY.json (his setup checks its own sha256 and the model identity there).

His RadixArk PLE patch still applies unchanged (its gate is pinned to RadixArk's config hash and never fires here), and
its ``VLLM_RADIXARK_QWEN38_NVFP4_PLE_FP8=1`` marker stays set because his teardown finds the server by it.
Status 2026-09-29: built and unit-tested on CPU against the exact runtime files; never run on a GPU.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

KEITH_SERVING_SETUP_SHA256 = "037c041c9bd9dcffa9084b32f47af9cf1bf35849d5eaf2e1a3422daac098b2e2"
RADIXARK_MODEL_SOURCE = "keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"
NVIDIA_MODEL_SOURCE = "xiaoz259/qwen3-8-flash-next-nvfp4/PyTorch/nvidia-nvfp4/1"
NVIDIA_KAGGLE_PATH = "/kaggle/input/models/xiaoz259/qwen3-8-flash-next-nvfp4/pytorch/nvidia-nvfp4/1"
NVIDIA_HF_REPO = "nvidia/Qwen3.8-Flash-Next-NVFP4"
NVIDIA_HF_REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
NVIDIA_CONFIG_SHA256 = "deef67a61f3311faf051b23dc4192f442c7fee4f9cd2f38cbcbe4da55c763a80"
NVIDIA_QUANT_CONFIG_SHA256 = "331ad11d57c8bc374554579977198084e0d4d0933d5b3558f125f6f670aba0e8"
# (path, bytes, sha256): HF tree API at the revision above (LFS oids; small files hashed after download, 2026-09-29).
# The Kaggle model lists the same 25 names and sizes (132,734,506,847 bytes).
NVIDIA_MODEL_FILES = (
    (".gitattributes", 1635, "fe81d528e7ee055bbcf6a310afe0f2808423a4a16d45225f29c285926abef08d"),
    ("README.md", 12109, "e3ed0cb89950ef0f41e8df344ba1cc15c67e1baf91e1cb5305a7344e58a8d975"),
    ("chat_template.jinja", 8952, "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"),
    ("config.json", 30820, "deef67a61f3311faf051b23dc4192f442c7fee4f9cd2f38cbcbe4da55c763a80"),
    ("generation_config.json", 202, "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"),
    ("hf_quant_config.json", 23001, "331ad11d57c8bc374554579977198084e0d4d0933d5b3558f125f6f670aba0e8"),
    ("merges.txt", 3353259, "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"),
    ("model-00001-of-00010.safetensors", 3115991696, "63fde954be6f08b49b876f4f70a0ad0bcfee71aff7b1faa33779b6b32feca2a2"),
    ("model-00002-of-00010.safetensors", 10005510304, "4dafaef62a908e49e0d92c7c2a3fa99f4d9651fdeeb0ca09f09a9b40095af4e0"),
    ("model-00003-of-00010.safetensors", 10005364728, "3218ddc129258e91a721a8e81329ca8f00588332ec721bd8d2cf7d20e324bf47"),
    ("model-00004-of-00010.safetensors", 10006060112, "55e2bdf6a3a1f6e65270787f65c63b2a2a56b308b54eb1fb318dfabc9b48b141"),
    ("model-00005-of-00010.safetensors", 10005362392, "d3c169d3694bfba846455d01e48f047d770f78d071c5a4d22f129389411d686e"),
    ("model-00006-of-00010.safetensors", 10005375328, "38f65a9d11090428e139cc60d0587e8e5433ea2ce79820883c583d7cfcc53130"),
    ("model-00007-of-00010.safetensors", 10005956544, "8cf3fa05c04cb2e060963b69bb01f7a7b9f43c58435fbe380836a0457a19cd3e"),
    ("model-00008-of-00010.safetensors", 10005375200, "53d1f80746aa0cc0a7c2f4836587002a8a4dd72c827432cd2e43abae3c3899e4"),
    ("model-00009-of-00010.safetensors", 5679113808, "eaff6a87ece6fbfd6ad0fbaa0a5cf9ff542f8076acba38e62410b1ed55e12cc3"),
    ("model-00010-of-00010.safetensors", 128587536, "0d49b0cf7c15bf9b5bd1e17403b36727315860a8f2a2e0f5a53655547084f640"),
    ("model-fp8-mtp-ple.safetensors", 53717551730, "3525520c8602d850003eb1960aec0b64291dae33b83f8b65dd639a451df78823"),
    ("model.safetensors.index.json", 31275518, "660414e8300728ba80062e6a3c81cb76a65e2ab136ac9e9ceb450ba0c51c3c0d"),
    ("preprocessor_config.json", 390, "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"),
    ("processor_config.json", 1191, "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1"),
    ("tokenizer.json", 12809320, "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"),
    ("tokenizer_config.json", 17928, "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"),
    ("video_preprocessor_config.json", 385, "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"),
    ("vocab.json", 6722759, "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"),
)
# Hashed even in fast start (small; the config pins the quantization layout the backports rely on).
NVIDIA_FAST_HASHED_FILES = ("config.json", "hf_quant_config.json", "chat_template.jinja", "generation_config.json",
                            "tokenizer_config.json")


def manifest() -> dict:
    return {"repo": NVIDIA_HF_REPO, "revision": NVIDIA_HF_REVISION,
            "files": [{"path": p, "size": s, "sha256": h} for p, s, h in NVIDIA_MODEL_FILES],
            "total_bytes": sum(s for _, s, _ in NVIDIA_MODEL_FILES)}


def manifest_sha256() -> str:
    return hashlib.sha256(json.dumps(manifest(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# --- runtime backports (files of vllm/vllm-openai:qwen38-flash-next, layer 24 of his runtime dataset) ---------------

PLE_REL = "vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"
MODELOPT_REL = "vllm/model_executor/layers/quantization/modelopt.py"
MTP_REL = "vllm/models/qwen3_8_flash_next/nvidia/mtp.py"

_PLE_OLD = """    if _is_exact_radixark_nvfp4_ple(quant_config, prefix, config):
        return Qwen3_8FlashNextPLEFp8EmbeddingMethod()
    if not isinstance(quant_config, Fp8Config):
"""
_PLE_NEW = """    if _is_exact_radixark_nvfp4_ple(quant_config, prefix, config):
        return Qwen3_8FlashNextPLEFp8EmbeddingMethod()
    # ours (scripts/nvidia_serving_patch.py): backport of vLLM d4d703c (#54882), FP8 PLE in a
    # MIXED_PRECISION ModelOpt checkpoint (nvidia/Qwen3.8-Flash-Next-NVFP4).
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig,
    )

    if isinstance(quant_config, ModelOptMixedPrecisionConfig):
        if quant_config._resolve_quant_algo(prefix) == "FP8":
            print(f"OURS_NVIDIA_PLE_FP8 selected prefix={prefix}", flush=True)
            return Qwen3_8FlashNextPLEFp8EmbeddingMethod()
        return None
    if not isinstance(quant_config, Fp8Config):
"""

_MO_LOGGER_OLD = "logger = init_logger(__name__)\n\nQUANT_ALGOS = [\n"
_MO_LOGGER_NEW = """logger = init_logger(__name__)

# ours (scripts/nvidia_serving_patch.py): backport of vLLM 60ad959 (#55513). ``FP8_PB_WO`` is ModelOpt's
# 2D block-FP8 name; early composed Qwen3.8-Flash-Next checkpoints used ``FP8_BLOCK_SCALES``.
_BLOCK_FP8_MOE_ALGOS = ("FP8_PB_WO", "FP8_BLOCK_SCALES")

QUANT_ALGOS = [
"""
_MO_INIT_OLD = """        self.w4a16_nvfp4_config = w4a16_nvfp4_config
        self.mxfp8_config = mxfp8_config
"""
_MO_INIT_NEW = """        self.w4a16_nvfp4_config = w4a16_nvfp4_config
        self.mxfp8_config = mxfp8_config

        # ours: backport of vLLM 60ad959 (#55513), block-FP8 routed experts (the MTP head).
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        block_sizes = {
            int(layer_info.get("group_size", 128))
            for layer_info in quantized_layers.values()
            if layer_info.get("quant_algo", "").upper() in _BLOCK_FP8_MOE_ALGOS
        }
        if len(block_sizes) > 1:
            raise ValueError(
                "MIXED_PRECISION currently requires all block-FP8 MoE layers "
                f"to use one group_size, got {sorted(block_sizes)}."
            )
        block_size = next(iter(block_sizes), 128)
        self.fp8_block_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[block_size, block_size],
        )
"""
_MO_MOE_OLD = """        if isinstance(layer, RoutedExperts):
            if quant_algo == "FP8":
                return ModelOptFp8MoEMethod(
"""
_MO_MOE_NEW = """        if isinstance(layer, RoutedExperts):
            if quant_algo in _BLOCK_FP8_MOE_ALGOS:  # ours: vLLM 60ad959 (#55513)
                from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

                print(f"OURS_NVIDIA_BLOCK_FP8_MOE prefix={prefix}", flush=True)
                return Fp8MoEMethod(self.fp8_block_config, layer)
            if quant_algo == "FP8":
                return ModelOptFp8MoEMethod(
"""

_MTP_FN_OLD = "def _remap_mtp_weight_name(name: str) -> str | None:\n"
_MTP_FN_NEW = '''def _remap_quantized_layers(
    quantized_layers: dict[str, dict],
    mtp_start_layer_idx: int,
) -> dict[str, dict]:
    """Map checkpoint MTP layer indices to standalone draft indices.

    ours (scripts/nvidia_serving_patch.py): backport of vLLM 60ad959 (#55513).
    """
    return {
        _remap_ignored_layers([name], mtp_start_layer_idx)[0]: layer_info
        for name, layer_info in quantized_layers.items()
    }


def _remap_mtp_weight_name(name: str) -> str | None:
'''
_MTP_CALL_OLD = """                _remap_ignored_layers(exclude_modules, mtp_start_layer_idx),
            )

    draft_vllm_config = replace(
"""
_MTP_CALL_NEW = """                _remap_ignored_layers(exclude_modules, mtp_start_layer_idx),
            )
        quantized_layers = getattr(draft_quant_config, "quantized_layers", None)
        if quantized_layers:  # ours: vLLM 60ad959 (#55513)
            setattr(  # noqa: B010
                draft_quant_config,
                "quantized_layers",
                _remap_quantized_layers(quantized_layers, mtp_start_layer_idx),
            )

    draft_vllm_config = replace(
"""

# ple_layer.py "before" is his RadixArk-patched file (VLLM_PLE_PATCHED_SHA256): his patch step runs first.
RUNTIME_BACKPORTS = (
    (PLE_REL, "a8a064744efc3c99eefff649f50395d77e1c36085266034d17b51722f68176ed",
     "eabc0dbfd8b9b5573f3143b50e259797c8feeeed58cf846758f0ada2e3ab4d5e", ((_PLE_OLD, _PLE_NEW),)),
    (MODELOPT_REL, "3f3ca743fd3c66d72be92b7544591a8632f1aa422b73bb50c83e8a3281196e7d",
     "9bc53bcda1f2a6c3b562f489cd8c119da678627afbe41f48b8b680d34c4d195f",
     ((_MO_LOGGER_OLD, _MO_LOGGER_NEW), (_MO_INIT_OLD, _MO_INIT_NEW), (_MO_MOE_OLD, _MO_MOE_NEW))),
    (MTP_REL, "7735cee47d0d1e4776bebd30d907e4a62160409ce4ef2d65611559f8d58af431",
     "d8da3c847c536d8bbdd5866e34536fc1268097db666c0daa60e9c43701b70cb7",
     ((_MTP_FN_OLD, _MTP_FN_NEW), (_MTP_CALL_OLD, _MTP_CALL_NEW))),
)


def backported_text(text: str, edits) -> str:
    for old, new in edits:
        if text.count(old) != 1:
            raise RuntimeError(f"nvidia backport anchor found {text.count(old)} times, expected once: {old[:80]!r}")
        text = text.replace(old, new, 1)
    return text


# --- serving_setup.py rewrite ----------------------------------------------------------------------------------------

_BLOCK_CODE = '''
def resolve_model_dir() -> Path:
    candidates = [MODEL_KAGGLE_PATH]
    if Path("/kaggle/input/models").is_dir():  # Kaggle mounts models at models/<owner>/<slug>/<framework>/<variation>/<v>
        candidates += sorted(p.parent for p in Path("/kaggle/input/models").glob("*/*/*/nvidia-nvfp4/*/config.json"))
    for candidate in dict.fromkeys(candidates):
        config = candidate / "config.json"
        if config.is_file() and sha256_file(config) == NVIDIA_CONFIG_SHA256:
            return candidate
    raise FileNotFoundError(
        f"Expected mounted {MODEL_HF_REPO}@{MODEL_HF_REVISION} (Kaggle model {NVIDIA_MODEL_SOURCE}) at "
        f"{MODEL_KAGGLE_PATH}; no candidate had its config.json: {[str(c) for c in candidates]}"
    )


def verify_model(model_dir: Path, *, full_file_hashes: bool = True) -> dict[str, Any]:
    started = time.monotonic()
    manifest = {
        "repo": MODEL_HF_REPO,
        "revision": MODEL_HF_REVISION,
        "files": [{"path": p, "size": s, "sha256": h} for p, s, h in NVIDIA_MODEL_FILES],
        "total_bytes": MODEL_TOTAL_BYTES,
    }
    manifest_sha = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest_sha != MODEL_MANIFEST_SHA256:
        raise RuntimeError(f"NVIDIA model manifest hash mismatch: {manifest_sha} != {MODEL_MANIFEST_SHA256}")
    hashed_files = 0
    hashed_bytes = 0
    for index, (relative, size, expected_sha) in enumerate(NVIDIA_MODEL_FILES, start=1):
        path = model_dir / relative
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"Missing or wrong-sized model file: {path}")
        if full_file_hashes or relative in NVIDIA_FAST_HASHED_FILES:
            if sha256_file(path) != expected_sha:
                raise RuntimeError(f"Model file hash mismatch: {relative}")
            hashed_files += 1
            hashed_bytes += size
        if full_file_hashes and (index == len(NVIDIA_MODEL_FILES) or index % 5 == 0):
            print(
                f"MODEL_VERIFY files={index}/{len(NVIDIA_MODEL_FILES)} bytes={hashed_bytes} "
                f"elapsed_s={time.monotonic() - started:.1f}",
                flush=True,
            )
    if not full_file_hashes:
        print(
            f"MODEL_IDENTITY_ONLY files={len(NVIDIA_MODEL_FILES)} bytes={MODEL_TOTAL_BYTES} "
            f"hashed={hashed_files} payload_check=vllm_load",
            flush=True,
        )
    config = read_json(model_dir / "config.json")
    quant = read_json(model_dir / "hf_quant_config.json")
    text_config = config.get("text_config") or {}
    quantization = config.get("quantization_config") or {}
    layers = quantization.get("quantized_layers") or {}
    nvfp4_experts = sorted(
        name for name, info in layers.items()
        if name.startswith("model.language_model.layers.") and name.endswith(".mlp.experts")
        and info == {"quant_algo": "NVFP4", "group_size": 16}
    )
    ple = layers.get("model.language_model.layers.1.ple.ple_embedding.ngram_embedding") or {}
    mtp = layers.get("mtp.layers.0.mlp.experts") or {}
    if config.get("architectures") != ["Qwen4ExpForConditionalGeneration"] or config.get("model_type") != "qwen4_exp":
        raise RuntimeError(f"Wrong model architecture: {config.get('architectures')} {config.get('model_type')}")
    if (
        quantization.get("quant_method") != "modelopt"
        or quantization.get("quant_algo") != "MIXED_PRECISION"
        or (quant.get("quantization") or {}).get("quant_algo") != "MIXED_PRECISION"
        or len(nvfp4_experts) != 48
        or len(layers) != 50
        or ple.get("quant_algo") != "FP8"
        or mtp.get("quant_algo") not in ("FP8_PB_WO", "FP8_BLOCK_SCALES")
        or int(mtp.get("group_size", -1)) != 128
    ):
        raise RuntimeError(f"Wrong NVIDIA ModelOpt MIXED_PRECISION layout: {sorted(layers.items())[:3]}")
    if (
        text_config.get("ple_layer_ids") != [2]
        or int(text_config.get("mtp_num_hidden_layers", -1)) != 1
        or int((text_config.get("mtp") or {}).get("num_hidden_layers", -1)) != 1
    ):
        raise RuntimeError("The NVIDIA checkpoint PLE or native MTP identity does not match.")
    return {
        "checkpoint": "nvidia",
        "manifest_sha256": manifest_sha,
        "config_sha256": NVIDIA_CONFIG_SHA256,
        "quant_config_sha256": NVIDIA_QUANT_CONFIG_SHA256,
        "file_count": len(NVIDIA_MODEL_FILES),
        "total_bytes": MODEL_TOTAL_BYTES,
        "architecture": config["architectures"][0],
        "quant_algo": quantization["quant_algo"],
        "ple_quant_algo": ple["quant_algo"],
        "mtp_experts_quant_algo": mtp["quant_algo"],
        "mtp_num_hidden_layers": text_config["mtp_num_hidden_layers"],
        "payload_sha256_verified": full_file_hashes,
        "hashed_file_count": hashed_files,
        "hashed_bytes": hashed_bytes,
        "payload_check_deferred_to_vllm_load": not full_file_hashes,
        "verification_mode": "full" if full_file_hashes else "identity-only",
        "verify_seconds": time.monotonic() - started,
    }


def apply_nvidia_runtime_backports() -> dict[str, Any]:
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    rows: dict[str, Any] = {}
    for relative, before_sha, after_sha, edits in NVIDIA_RUNTIME_BACKPORTS:
        path = site / relative
        current = path.read_bytes()
        current_sha = hashlib.sha256(current).hexdigest()
        if current_sha == after_sha:
            rows[relative] = "already-patched"
            continue
        if current_sha != before_sha:
            raise RuntimeError(f"NVIDIA backport target changed: {relative} {current_sha} != {before_sha}")
        text = current.decode("utf-8")
        for old, new in edits:
            if text.count(old) != 1:
                raise RuntimeError(f"NVIDIA backport anchor not found once in {relative}: {old[:80]!r}")
            text = text.replace(old, new, 1)
        data = text.encode("utf-8")
        if hashlib.sha256(data).hexdigest() != after_sha:
            raise RuntimeError(f"NVIDIA backport produced an unexpected {relative}")
        temporary = path.with_name(f".{path.name}.nvidia-backport.tmp")
        try:
            temporary.write_bytes(data)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        if sha256_file(path) != after_sha:
            raise RuntimeError(f"NVIDIA backport write mismatch: {relative}")
        rows[relative] = {"before_sha256": before_sha, "after_sha256": after_sha, "edits": len(edits)}
    print("NVIDIA_RUNTIME_BACKPORTS " + json.dumps(rows, sort_keys=True), flush=True)
    return rows
'''

BLOCK_BEGIN = "# ==== ours: scripts/nvidia_serving_patch.py (team scottmahony): serve nvidia/Qwen3.8-Flash-Next-NVFP4 ===="
BLOCK_END = "# ==== end ours: scripts/nvidia_serving_patch.py ===="


def setup_block() -> str:
    """The code appended after his verify_model(): later definitions override the RadixArk model identity."""
    rows = ",\n".join(f"    {row!r}" for row in NVIDIA_MODEL_FILES)
    return (
        f"{BLOCK_BEGIN}\n"
        "# Overrides the RadixArk model identity above. MODEL_CONFIG_SHA256 keeps the RadixArk value on purpose: his PLE\n"
        "# patch identity, its gate environment and his full-mode import probe are pinned to it (the gate never fires\n"
        "# for this checkpoint, whose quantization config is ModelOpt MIXED_PRECISION).\n"
        f"MODEL_HF_REPO = {NVIDIA_HF_REPO!r}\n"
        f"MODEL_HF_REVISION = {NVIDIA_HF_REVISION!r}\n"
        f"MODEL_KAGGLE_PATH = Path({NVIDIA_KAGGLE_PATH!r})\n"
        f"NVIDIA_MODEL_SOURCE = {NVIDIA_MODEL_SOURCE!r}\n"
        f"NVIDIA_CONFIG_SHA256 = {NVIDIA_CONFIG_SHA256!r}\n"
        f"NVIDIA_QUANT_CONFIG_SHA256 = {NVIDIA_QUANT_CONFIG_SHA256!r}\n"
        f"NVIDIA_MODEL_FILES = (\n{rows},\n)\n"
        f"NVIDIA_FAST_HASHED_FILES = {NVIDIA_FAST_HASHED_FILES!r}\n"
        "MODEL_FILE_COUNT = len(NVIDIA_MODEL_FILES)\n"
        "MODEL_TOTAL_BYTES = sum(size for _, size, _ in NVIDIA_MODEL_FILES)\n"
        f"MODEL_MANIFEST_SHA256 = {manifest_sha256()!r}\n"
        f"NVIDIA_RUNTIME_BACKPORTS = {RUNTIME_BACKPORTS!r}\n"
        f"NVIDIA_PLE_FINAL_SHA256 = {RUNTIME_BACKPORTS[0][2]!r}\n"
        f"{_BLOCK_CODE}"
        f"{BLOCK_END}\n"
    )


def setup_edits() -> list[tuple[str, str, str]]:
    """(label, old, new) text edits of his serving_setup.py; each old text must occur exactly once."""
    return [
        ("nvidia identity, model checks and runtime backports",
         "\n\ndef _apply_whiteouts(", "\n\n" + setup_block() + "\n\ndef _apply_whiteouts("),
        ("runtime backports run after his PLE patch",
         "    ple_patch = patch_ple_layer()\n",
         "    ple_patch = patch_ple_layer()\n"
         "    ple_patch = {**ple_patch, \"nvidia_runtime_backports\": apply_nvidia_runtime_backports()}  # ours\n"),
        ("pre-import PLE hash is the backported one",
         "    if sha256_file(ple_path) != VLLM_PLE_PATCHED_SHA256:\n",
         "    if sha256_file(ple_path) != NVIDIA_PLE_FINAL_SHA256:  # ours: his patch + the d4d703c backport\n"),
        ("quantization method",
         '        "--quantization",\n        "modelopt_fp4",\n',
         '        "--quantization",\n        "modelopt_mixed",  # ours: NVIDIA ModelOpt MIXED_PRECISION checkpoint\n'),
    ]


def patched_setup(text: str) -> str:
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != KEITH_SERVING_SETUP_SHA256:
        raise RuntimeError("his serving_setup.py is not the pinned version this patch was written against")
    for label, old, new in setup_edits():
        if text.count(old) != 1:
            raise RuntimeError(f"nvidia serving patch '{label}': expected exactly one match, found {text.count(old)}")
        text = text.replace(old, new, 1)
    return text


def apply(bundle_dir: Path, dest: Path) -> dict:
    """Copy his bundle to ``dest`` (replaced if present) and patch the copy; returns what changed."""
    bundle_dir, dest = Path(bundle_dir), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(bundle_dir, dest, ignore=shutil.ignore_patterns("__pycache__"))
    setup_path = dest / "serving_setup.py"
    new_setup = patched_setup(setup_path.read_text(encoding="utf-8"))
    setup_path.write_text(new_setup, encoding="utf-8")
    setup_sha = hashlib.sha256(new_setup.encode("utf-8")).hexdigest()
    identity_path = dest / "SOURCE_IDENTITY.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["serving_setup_sha256"] = setup_sha
    identity["model_manifest_sha256"] = manifest_sha256()
    identity["model"] = {
        "hf_repo": NVIDIA_HF_REPO, "hf_revision": NVIDIA_HF_REVISION, "kaggle_model_source": NVIDIA_MODEL_SOURCE,
        "config_sha256": NVIDIA_CONFIG_SHA256, "file_count": len(NVIDIA_MODEL_FILES),
        "total_bytes": manifest()["total_bytes"], "manifest_sha256": manifest_sha256(),
    }
    identity["ours_nvidia_serving_patch"] = {
        "replaces_model": RADIXARK_MODEL_SOURCE, "his_serving_setup_sha256": KEITH_SERVING_SETUP_SHA256,
        "runtime_backports": {rel: after for rel, _, after, _ in RUNTIME_BACKPORTS},
    }
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"bundle": str(dest), "serving_setup_sha256": setup_sha, "model": NVIDIA_MODEL_SOURCE,
            "edits": [label for label, _, _ in setup_edits()]}
