#!/usr/bin/env python3
"""Apply the exact RadixArk NVFP4 PLE loader compatibility patch."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


TARGET_RELATIVE_PATH = Path(
    "vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"
)
EXPECTED_STOCK_SHA256 = (
    "a71144c1d36e06f22a2da1b1ada900076597fe5e824a911e7ada86249a0993e7"
)
EXPECTED_PATCHED_SHA256 = (
    "a8a064744efc3c99eefff649f50395d77e1c36085266034d17b51722f68176ed"
)

_STOCK_IMPORT = """import math
from collections.abc import Iterable, Sequence
"""
_PATCHED_IMPORT = """import math
import os
from collections.abc import Iterable, Sequence
"""

_STOCK_FUNCTION = '''def _get_ple_embedding_quant_method(
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> QuantizeMethodBase | None:
    """Select global-scale FP8 only for quantized PLE checkpoint shards."""

    if not isinstance(quant_config, Fp8Config):
        return None
    if not quant_config.is_checkpoint_fp8_serialized:
        return None

    ignored_layers = quant_config.ignored_layers
    if is_layer_skipped(
        prefix,
        ignored_layers,
        quant_config.packed_modules_mapping,
        match_mode=quant_config.ignored_layers_match_mode,
    ):
        return None
    # PLE checkpoint shards form one runtime embedding parameter.
    shard_prefix = f"{prefix}.shard_"
    if any(name.startswith(shard_prefix) for name in ignored_layers):
        return None
    return Qwen3_8FlashNextPLEFp8EmbeddingMethod()
'''

_PATCHED_FUNCTION = '''_RADIXARK_NVFP4_PLE_FP8_ENV = "VLLM_RADIXARK_QWEN38_NVFP4_PLE_FP8"
_RADIXARK_NVFP4_CONFIG_SHA256_ENV = (
    "VLLM_RADIXARK_QWEN38_NVFP4_CONFIG_SHA256"
)
_RADIXARK_NVFP4_CONFIG_SHA256 = (
    "e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624"
)
_RADIXARK_NVFP4_PLE_PREFIX = (
    "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
)
_RADIXARK_NVFP4_PLE_EXCLUDED_PREFIX = (
    "model.language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
)
_RADIXARK_NVFP4_LAYER_TYPES = tuple(
    "full_attention" if (layer_idx + 1) % 4 == 0 else "linear_attention"
    for layer_idx in range(48)
)
_RADIXARK_NVFP4_TEXT_CONFIG_FIELDS = {
    "model_type": "qwen4_exp_text",
    "hidden_size": 2560,
    "num_hidden_layers": 48,
    "num_experts": 512,
    "ple_embed_dim": 2560,
    "ple_embedding_dtype": "float8_e4m3fn",
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "ngram_vocab_size_base": 20_000_000,
    "split_ngram_parts": 128,
}


def _is_exact_radixark_nvfp4_ple(
    quant_config: QuantizationConfig | None,
    prefix: str,
    config: Qwen3_8FlashNextTextConfig,
) -> bool:
    if os.environ.get(_RADIXARK_NVFP4_PLE_FP8_ENV) != "1":
        return False
    if (
        os.environ.get(_RADIXARK_NVFP4_CONFIG_SHA256_ENV)
        != _RADIXARK_NVFP4_CONFIG_SHA256
    ):
        return False
    try:
        if type(quant_config).__name__ != "ModelOptNvFp4Config":
            return False
        get_name = getattr(quant_config, "get_name", None)
        if not callable(get_name) or get_name() != "modelopt_fp4":
            return False
        if getattr(quant_config, "quant_method", None) != "NVFP4":
            return False
        if getattr(quant_config, "is_checkpoint_nvfp4_serialized", None) is not True:
            return False
        if getattr(quant_config, "group_size", None) != 16:
            return False
        is_layer_excluded = getattr(quant_config, "is_layer_excluded", None)
        if (
            not callable(is_layer_excluded)
            or is_layer_excluded(_RADIXARK_NVFP4_PLE_EXCLUDED_PREFIX) is not True
        ):
            return False
        if prefix != _RADIXARK_NVFP4_PLE_PREFIX:
            return False
        if any(
            getattr(config, field, None) != expected
            for field, expected in _RADIXARK_NVFP4_TEXT_CONFIG_FIELDS.items()
        ):
            return False
        if list(getattr(config, "ple_layer_ids", ())) != [2]:
            return False
        if tuple(getattr(config, "layer_types", ())) != _RADIXARK_NVFP4_LAYER_TYPES:
            return False
    except Exception:
        return False
    return True


def _get_ple_embedding_quant_method(
    quant_config: QuantizationConfig | None,
    prefix: str,
    config: Qwen3_8FlashNextTextConfig,
) -> QuantizeMethodBase | None:
    """Select global-scale FP8 only for quantized PLE checkpoint shards."""

    if _is_exact_radixark_nvfp4_ple(quant_config, prefix, config):
        return Qwen3_8FlashNextPLEFp8EmbeddingMethod()
    if not isinstance(quant_config, Fp8Config):
        return None
    if not quant_config.is_checkpoint_fp8_serialized:
        return None

    ignored_layers = quant_config.ignored_layers
    if is_layer_skipped(
        prefix,
        ignored_layers,
        quant_config.packed_modules_mapping,
        match_mode=quant_config.ignored_layers_match_mode,
    ):
        return None
    # PLE checkpoint shards form one runtime embedding parameter.
    shard_prefix = f"{prefix}.shard_"
    if any(name.startswith(shard_prefix) for name in ignored_layers):
        return None
    return Qwen3_8FlashNextPLEFp8EmbeddingMethod()
'''

_STOCK_CALL = """            quant_method=_get_ple_embedding_quant_method(
                quant_config, f"{prefix}.ngram_embedding"
            ),
"""
_PATCHED_CALL = """            quant_method=_get_ple_embedding_quant_method(
                quant_config, f"{prefix}.ngram_embedding", config
            ),
"""


class PatchError(RuntimeError):
    """The target did not match the pinned stock or patched source."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patched_bytes(stock: bytes) -> bytes:
    if sha256_bytes(stock) != EXPECTED_STOCK_SHA256:
        raise PatchError("stock ple_layer.py SHA-256 mismatch")
    text = stock.decode("utf-8")
    replacements = (
        (_STOCK_IMPORT, _PATCHED_IMPORT, "import block"),
        (_STOCK_FUNCTION, _PATCHED_FUNCTION, "PLE loader selector"),
        (_STOCK_CALL, _PATCHED_CALL, "PLE loader call"),
    )
    for old, new, label in replacements:
        if text.count(old) != 1:
            raise PatchError(f"expected one stock {label}")
        text = text.replace(old, new, 1)
    result = text.encode("utf-8")
    if (
        EXPECTED_PATCHED_SHA256 != "__PATCHED_SHA256_PENDING__"
        and sha256_bytes(result) != EXPECTED_PATCHED_SHA256
    ):
        raise PatchError("generated patched ple_layer.py SHA-256 mismatch")
    return result


def apply_patch(site_packages: Path) -> str:
    target = site_packages / TARGET_RELATIVE_PATH
    current = target.read_bytes()
    current_sha256 = sha256_bytes(current)
    if current_sha256 == EXPECTED_PATCHED_SHA256:
        return "already-patched"
    result = patched_bytes(current)
    temporary = target.with_name(f".{target.name}.radixark-patch.tmp")
    try:
        temporary.write_bytes(result)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    if sha256_bytes(target.read_bytes()) != EXPECTED_PATCHED_SHA256:
        raise PatchError("post-write ple_layer.py SHA-256 mismatch")
    return "patched"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "site_packages",
        type=Path,
        nargs="?",
        help="directory that directly contains the vllm package",
    )
    parser.add_argument(
        "--print-generated-sha256",
        action="store_true",
        help="print the deterministic patched hash without writing",
    )
    parser.add_argument(
        "--stock-file",
        type=Path,
        help="stock ple_layer.py used with --print-generated-sha256",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_generated_sha256:
        if args.stock_file is None:
            raise SystemExit("--stock-file is required")
        print(sha256_bytes(patched_bytes(args.stock_file.read_bytes())))
        return
    if args.site_packages is None:
        raise SystemExit("site_packages is required")
    print(apply_patch(args.site_packages))


if __name__ == "__main__":
    main()
