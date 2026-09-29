"""Prepare the pinned Qwen3.8-Flash-Next vLLM server on Kaggle.

This file changes serving only. The Duck policy, game, model, and notebook stay
frozen. Every runtime and model input is offline and pinned. Fast start trusts
the immutable Kaggle payload after checking its small identity files; vLLM's
load is the payload-integrity test.
"""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import math
import os
import re
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import sysconfig
import time
import traceback
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any


SOURCE_DATASET = "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1"
RUNTIME_DATASET = "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"
MODEL_HF_REPO = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
MODEL_HF_REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
MODEL_MANIFEST_SHA256 = "a09bdad3fe3240c73332c0f99f4388a547205cb488c127f9b9059c9267dd9a5b"
MODEL_CONFIG_SHA256 = "e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624"
MODEL_QUANT_CONFIG_SHA256 = "7e69ef4b94302ae5b6f453b913621f698d5631a1d023d8b3e9e3b829721b98e8"
MODEL_FILE_COUNT = 419
MODEL_TOTAL_BYTES = 135_253_622_894
MODEL_KAGGLE_PATH = Path(
    "/kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/"
    "pytorch/radixark-modelopt-fp4/1"
)

VLLM_IMAGE = "vllm/vllm-openai:qwen38-flash-next"
VLLM_INDEX_DIGEST = "sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8"
VLLM_AMD64_MANIFEST_DIGEST = (
    "sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3"
)
VLLM_CONFIG_DIGEST = "sha256:bd995759b5b8ac51062e04c9e4d7c91c382d1ba377bb787e24dca2ccb39925e9"
VLLM_RUNTIME_MANIFEST_SHA256 = (
    "e9453f8d0e9c5eb2e14712e0f8563aaa96752ddc1705f245cac327537502baad"
)
VLLM_VERSION = "0.1.dev20073+g8e685d198"
VLLM_PLE_STOCK_SHA256 = "a71144c1d36e06f22a2da1b1ada900076597fe5e824a911e7ada86249a0993e7"
VLLM_PLE_PATCHED_SHA256 = "a8a064744efc3c99eefff649f50395d77e1c36085266034d17b51722f68176ed"
PLE_PATCH_DIRNAME = "vllm-patches"
PLE_PATCH_APPLICATOR = "apply_radixark_nvfp4_ple_fp8_patch.py"
PLE_PATCH_APPLICATOR_SHA256 = "a9f88c019d04d7c8804ca6532e1c689d2ea97cf085cfec37dd4a779cb2c2c3fd"
PLE_PATCH_APPLICATOR_BYTES = 8_230
PLE_PATCH_DIFF = "radixark_nvfp4_ple_fp8.patch"
PLE_PATCH_DIFF_SHA256 = "e3b83f9a65e436d21d3e757ddc8dd2699f3c59de9f73967107b3b20c101d0a9f"
PLE_PATCH_DIFF_BYTES = 3_940
PLE_PATCH_IDENTITY = "PATCH_IDENTITY.json"
PLE_PATCH_IDENTITY_SHA256 = "f0bc8a948567e551d1bdf63f1163503ded897c3eba070d149d44c0c08fccbd35"
PLE_PATCH_IDENTITY_BYTES = 1_349
PLE_PATCH_README = "README.md"
PLE_PATCH_README_SHA256 = "18e73020660bfe46b66c0ec1dfc3e702e848b51bc3d4773f58f6ae06ef126f9d"
PLE_PATCH_README_BYTES = 1_581
PLE_EXACT_PREFIX = (
    "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
)
PLE_EXCLUDED_PREFIX = (
    "model.language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
)
CUTLASS_PTH_SHA256 = "cc710b99ddb3d76361a70461461d274212a979d00c381d821fce5bb3ad9cf246"
CUTLASS_PTH_TEXT = (
    "import sys, os, nvidia_cutlass_dsl; sys.path.insert(0, "
    "os.path.join(nvidia_cutlass_dsl.__path__[0], 'dsl_packages'))"
)
CUTLASS_DSL_VERSION = "4.6.2"
TORCH_SHM_MANAGER_CMDLINE_SHA256 = (
    "fd3a41c6faf7553a76d60949dea5632c423de414c9c0e9494171ea942543f428"
)
EXPECTED_RUNTIME_LAYERS = {
    8: (
        "f4f325a97c3e4739640b0e472a818bb299e7ce8e585bdbfe5586e9d3b5a2c475",
        1_374_796_713,
    ),
    17: (
        "a5881cf32fc46f24318a6fc7015ccae863706075d4434d186b8b1b4e5bafb36e",
        4_634_843_373,
    ),
    18: (
        "692ea0d28a87e9cea6db254c830f650f29804522045c9201f8ea63f30471b33b",
        1_511_319_261,
    ),
    21: (
        "651e9c6638c77d15bf671d9264badcba80ea551f84109de486427b87d2ddbf75",
        61_571_140,
    ),
    24: (
        "10dce885652a53a2e644a75d256b90830663330988b8599c3c0e56cd6c8a4675",
        309_686_893,
    ),
    27: (
        "49a63ac9d116f92e5a8b006ae5665b940fcbc3d22d53aa7c01a3753ef341c53e",
        14_479_724,
    ),
}
FAST_RUNTIME_WHITEOUTS = {
    8: (
        "usr/local/cuda-13.0/targets/x86_64-linux/lib/.wh.libcudart.so.13.0.88",
    ),
    17: (
        "tmp/.wh.requirements-cuda.txt",
        "tmp/.wh.common.txt",
    ),
}

SERVED_MODEL_NAME = "Qwen/Qwen3.8-Flash-Next-NVFP4"
HOST = "127.0.0.1"
PORT = 1234
BASE_URL = f"http://{HOST}:{PORT}/v1"
ANALYZER_CONTEXT = 32_768
# Preserve the frozen 28-lane Duck solver. The v9 OOM came from the 32K-token
# dummy warmup, so the primary profile cuts that batch without reducing lanes.
MAX_NUM_SEQS = 28
# Keep the model author's validated single-GPU setting. Raising this to 0.98
# would trade the known-safe GPU reserve for KV cache that vLLM can size itself.
GPU_MEMORY_UTILIZATION = "0.92"
# The primary profile is the v8 fit: chunked prefill with an 8K batch cap.
MAX_NUM_BATCHED_TOKENS = 8_192
# Keep the explicit no-chunk API valid for controlled diagnostics. Pinned vLLM
# rejects disabled chunked prefill when this is below max_model_len.
NO_CHUNKED_MAX_NUM_BATCHED_TOKENS = ANALYZER_CONTEXT
SERVER_READY_TIMEOUT = 1_500
SETUP_TOTAL_TIMEOUT = 1_800
FAST_START_ENV = "TAAF_KAGGLE_FAST_START"
MAX_NUM_SEQS_ENV = "TAAF_VLLM_MAX_NUM_SEQS"
MAX_NUM_BATCHED_TOKENS_ENV = "TAAF_VLLM_MAX_NUM_BATCHED_TOKENS"
OMP_THREADS_ENV = "TAAF_VLLM_OMP_THREADS"
MTP_TOKENS_ENV = "TAAF_VLLM_MTP_TOKENS"
KV_CACHE_MEMORY_BYTES_ENV = "TAAF_VLLM_KV_CACHE_MEMORY_BYTES"
KV_CACHE_DTYPE_ENV = "TAAF_VLLM_KV_CACHE_DTYPE"
ENABLE_PREFIX_CACHING_ENV = "TAAF_VLLM_ENABLE_PREFIX_CACHING"
MAX_CUDAGRAPH_CAPTURE_SIZE_ENV = "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE"
MOE_BACKEND_ENV = "TAAF_VLLM_MOE_BACKEND"
MTP_INDEX_SHARE_FOR_ITERATION_ENV = (
    "TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION"
)
MTP_DYNAMIC_BATCH_SCHEDULE_ENV = "TAAF_VLLM_MTP_DYNAMIC_BATCH_SCHEDULE"
DEFAULT_OMP_THREADS = 4
DEFAULT_MTP_TOKENS = 3
MAX_EXPLICIT_KV_CACHE_MEMORY_BYTES = 96 * 1024**3
MIN_EXPLICIT_KV_CACHE_MEMORY_BYTES = 64 * 1024**2
VLLM_TUNING_FIELDS = {
    "schema_version",
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "kv_cache_dtype",
    "kv_cache_memory_bytes",
    "max_cudagraph_capture_size",
    "max_num_batched_tokens",
    "max_num_batched_tokens_requested",
    "max_num_seqs",
    "moe_backend",
    "mtp_dynamic_batch_schedule",
    "mtp_index_share_for_iteration",
    "mtp_speculative_tokens",
    "omp_num_threads",
    "overridden",
}
VLLM_TUNING_OVERRIDE_FIELDS = {
    MAX_NUM_SEQS_ENV,
    MAX_NUM_BATCHED_TOKENS_ENV,
    OMP_THREADS_ENV,
    MTP_TOKENS_ENV,
    KV_CACHE_MEMORY_BYTES_ENV,
    KV_CACHE_DTYPE_ENV,
    ENABLE_PREFIX_CACHING_ENV,
    MAX_CUDAGRAPH_CAPTURE_SIZE_ENV,
    MOE_BACKEND_ENV,
    MTP_INDEX_SHARE_FOR_ITERATION_ENV,
    MTP_DYNAMIC_BATCH_SCHEDULE_ENV,
}
MIN_HOST_AVAILABLE_BYTES = 64 * 1024**3
MIN_GPU_FREE_MIB = 4_096
MIN_RUNTIME_FILESYSTEM_FREE_BYTES = 36_507_222_016
RUNTIME_EXTRACTED_BOUND_BYTES = 18_359_349_248
RUNTIME_CACHE_RESERVE_BYTES = 16 * 1024**3
RUNTIME_STORAGE_AUDIT_SHA256 = (
    "5791fbfae9b53d4b8f4000bab769f13e1b1d78c763e889663199e3ee76f5252c"
)

BUNDLE_DIR = Path(os.environ["TAAF_KAGGLE_BUNDLE_DIR"])
WORKING_DIR = Path(os.environ["TAAF_KAGGLE_WORKING_DIR"])
SETUP_ENV_PATH = Path(os.environ["TAAF_KAGGLE_SETUP_ENV"])
RUNTIME_ROOT = Path("/tmp/qwen38-flash-next-vllm-runtime")
CACHE_ROOT = Path("/tmp/qwen38-flash-next-vllm-cache")
COMPILE_CACHE_ROOT = Path("/tmp/qwen38-flash-next-vllm-compile-cache")
TMP_ROOT = Path("/tmp/qwen38-flash-next-vllm-tmp")
SERVER_LOG = WORKING_DIR / "vllm-openai-server.log"
SERVER_IDENTITY = WORKING_DIR / "vllm-server-identity.json"
SETUP_PROVENANCE = WORKING_DIR / "vllm-setup-provenance.json"
PREFLIGHT_PATH = WORKING_DIR / "vllm-server-preflight.json"
GPU_PATH = WORKING_DIR / "loaded-gpu-compute.json"
METRICS_PATH = WORKING_DIR / "vllm-metrics-after-preflight.prom"
FAILURE_PATH = WORKING_DIR / "vllm-setup-failure.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fast_start_enabled() -> bool:
    raw = os.environ.get(FAST_START_ENV, "1").strip()
    if raw not in {"0", "1"}:
        raise RuntimeError(f"{FAST_START_ENV} must be 0 or 1, found {raw!r}.")
    return raw == "1"


def _bounded_integer_environment(
    name: str, *, default: int, minimum: int, maximum: int
) -> tuple[int, bool]:
    raw = os.environ.get(name)
    if raw is None:
        return default, False
    stripped = raw.strip()
    if re.fullmatch(r"\d+", stripped) is None:
        raise RuntimeError(
            f"{name} must be an integer from {minimum} through {maximum}, "
            f"found {raw!r}."
        )
    value = int(stripped)
    if not minimum <= value <= maximum:
        raise RuntimeError(
            f"{name} must be an integer from {minimum} through {maximum}, "
            f"found {raw!r}."
        )
    return value, True


def _binary_environment(name: str, *, default: bool = False) -> tuple[bool, bool]:
    raw = os.environ.get(name)
    if raw is None:
        return default, False
    stripped = raw.strip()
    if stripped not in {"0", "1"}:
        raise RuntimeError(f"{name} must be 0 or 1, found {raw!r}.")
    return stripped == "1", True


def _validate_mtp_dynamic_batch_schedule(
    value: Any, *, max_speculative_tokens: int
) -> list[list[int]]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(
            f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV} must be a non-empty JSON list."
        )
    resolved: list[list[int]] = []
    previous_maximum = 0
    for index, row in enumerate(value):
        if (
            not isinstance(row, list)
            or len(row) != 3
            or any(type(item) is not int for item in row)
        ):
            raise RuntimeError(
                f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV}[{index}] must contain exactly "
                "three integers: [minimum_batch_size, maximum_batch_size, "
                "num_speculative_tokens]."
            )
        minimum_batch_size, maximum_batch_size, speculative_tokens = row
        if index == 0 and minimum_batch_size != 1:
            raise RuntimeError(
                f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV} must start at batch size 1."
            )
        if minimum_batch_size < 1 or maximum_batch_size < minimum_batch_size:
            raise RuntimeError(
                f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV}[{index}] has an invalid batch "
                f"range: {row!r}."
            )
        if minimum_batch_size <= previous_maximum:
            raise RuntimeError(
                f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV} batch ranges must be strictly "
                "increasing and non-overlapping."
            )
        if not 1 <= speculative_tokens <= max_speculative_tokens:
            raise RuntimeError(
                f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV}[{index}] speculative tokens "
                f"must be from 1 through {max_speculative_tokens}, found "
                f"{speculative_tokens}."
            )
        resolved.append(
            [minimum_batch_size, maximum_batch_size, speculative_tokens]
        )
        previous_maximum = maximum_batch_size
    return resolved


def _mtp_dynamic_batch_schedule_environment(
    *, max_speculative_tokens: int
) -> tuple[list[list[int]] | None, bool]:
    raw = os.environ.get(MTP_DYNAMIC_BATCH_SCHEDULE_ENV)
    if raw is None:
        return None, False
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV} must contain valid JSON, "
            f"found {raw!r}."
        ) from exc
    return (
        _validate_mtp_dynamic_batch_schedule(
            value, max_speculative_tokens=max_speculative_tokens
        ),
        True,
    )


def resolve_vllm_tuning(
    *, enable_chunked_prefill: bool = True
) -> dict[str, Any]:
    """Resolve and validate the small serving-only benchmark surface."""
    if not isinstance(enable_chunked_prefill, bool):
        raise TypeError("enable_chunked_prefill must be a bool.")
    max_num_seqs, max_num_seqs_overridden = _bounded_integer_environment(
        MAX_NUM_SEQS_ENV,
        default=MAX_NUM_SEQS,
        minimum=1,
        maximum=256,
    )
    requested_batch_tokens, batch_tokens_overridden = _bounded_integer_environment(
        MAX_NUM_BATCHED_TOKENS_ENV,
        default=MAX_NUM_BATCHED_TOKENS,
        minimum=1,
        maximum=1_048_576,
    )
    if enable_chunked_prefill:
        max_num_batched_tokens = requested_batch_tokens
    elif batch_tokens_overridden:
        if requested_batch_tokens < ANALYZER_CONTEXT:
            raise RuntimeError(
                f"{MAX_NUM_BATCHED_TOKENS_ENV} must be at least {ANALYZER_CONTEXT} "
                "when chunked prefill is disabled."
            )
        max_num_batched_tokens = requested_batch_tokens
    else:
        max_num_batched_tokens = NO_CHUNKED_MAX_NUM_BATCHED_TOKENS

    omp_threads, omp_threads_overridden = _bounded_integer_environment(
        OMP_THREADS_ENV,
        default=DEFAULT_OMP_THREADS,
        minimum=1,
        maximum=64,
    )
    mtp_tokens, mtp_tokens_overridden = _bounded_integer_environment(
        MTP_TOKENS_ENV,
        default=DEFAULT_MTP_TOKENS,
        minimum=0,
        maximum=4,
    )
    mtp_index_share_for_iteration, mtp_index_share_overridden = (
        _binary_environment(MTP_INDEX_SHARE_FOR_ITERATION_ENV)
    )
    mtp_dynamic_batch_schedule, mtp_dynamic_schedule_overridden = (
        _mtp_dynamic_batch_schedule_environment(
            max_speculative_tokens=mtp_tokens
        )
        if mtp_tokens > 0
        else (None, MTP_DYNAMIC_BATCH_SCHEDULE_ENV in os.environ)
    )
    if mtp_tokens == 0 and mtp_index_share_for_iteration:
        raise RuntimeError(
            f"{MTP_INDEX_SHARE_FOR_ITERATION_ENV}=1 requires "
            f"{MTP_TOKENS_ENV} to be greater than 0."
        )
    if mtp_tokens == 0 and MTP_DYNAMIC_BATCH_SCHEDULE_ENV in os.environ:
        raise RuntimeError(
            f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV} requires {MTP_TOKENS_ENV} "
            "to be greater than 0."
        )

    moe_backend_raw = os.environ.get(MOE_BACKEND_ENV)
    moe_backend = None if moe_backend_raw is None else moe_backend_raw.strip()
    if moe_backend not in {None, "flashinfer_b12x"}:
        raise RuntimeError(
            f"{MOE_BACKEND_ENV} must be 'flashinfer_b12x' when set, "
            f"found {moe_backend_raw!r}."
        )
    kv_cache_memory_bytes, kv_cache_memory_overridden = (
        _bounded_integer_environment(
            KV_CACHE_MEMORY_BYTES_ENV,
            default=0,
            minimum=0,
            maximum=MAX_EXPLICIT_KV_CACHE_MEMORY_BYTES,
        )
    )
    if 0 < kv_cache_memory_bytes < MIN_EXPLICIT_KV_CACHE_MEMORY_BYTES:
        raise RuntimeError(
            f"{KV_CACHE_MEMORY_BYTES_ENV} must be 0 or at least "
            f"{MIN_EXPLICIT_KV_CACHE_MEMORY_BYTES}, found {kv_cache_memory_bytes}."
        )

    kv_cache_dtype_raw = os.environ.get(KV_CACHE_DTYPE_ENV)
    kv_cache_dtype = (
        "auto" if kv_cache_dtype_raw is None else kv_cache_dtype_raw.strip().lower()
    )
    if kv_cache_dtype not in {"auto", "fp8"}:
        raise RuntimeError(
            f"{KV_CACHE_DTYPE_ENV} must be 'auto' or 'fp8', "
            f"found {kv_cache_dtype_raw!r}."
        )

    prefix_raw = os.environ.get(ENABLE_PREFIX_CACHING_ENV)
    if prefix_raw is None:
        enable_prefix_caching = False
    else:
        stripped_prefix = prefix_raw.strip()
        if stripped_prefix not in {"0", "1"}:
            raise RuntimeError(
                f"{ENABLE_PREFIX_CACHING_ENV} must be 0 or 1, found {prefix_raw!r}."
            )
        enable_prefix_caching = stripped_prefix == "1"

    max_cudagraph_capture_size, cudagraph_overridden = (
        _bounded_integer_environment(
            MAX_CUDAGRAPH_CAPTURE_SIZE_ENV,
            default=0,
            minimum=0,
            maximum=256,
        )
    )

    return {
        "schema_version": 1,
        "enable_chunked_prefill": enable_chunked_prefill,
        "enable_prefix_caching": enable_prefix_caching,
        "kv_cache_dtype": kv_cache_dtype,
        "kv_cache_memory_bytes": kv_cache_memory_bytes,
        "max_cudagraph_capture_size": max_cudagraph_capture_size,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_batched_tokens_requested": requested_batch_tokens,
        "max_num_seqs": max_num_seqs,
        "moe_backend": moe_backend,
        "mtp_dynamic_batch_schedule": mtp_dynamic_batch_schedule,
        "mtp_index_share_for_iteration": mtp_index_share_for_iteration,
        "mtp_speculative_tokens": mtp_tokens,
        "omp_num_threads": omp_threads,
        "overridden": {
            MAX_NUM_SEQS_ENV: max_num_seqs_overridden,
            MAX_NUM_BATCHED_TOKENS_ENV: batch_tokens_overridden,
            OMP_THREADS_ENV: omp_threads_overridden,
            MTP_TOKENS_ENV: mtp_tokens_overridden,
            KV_CACHE_MEMORY_BYTES_ENV: kv_cache_memory_overridden,
            KV_CACHE_DTYPE_ENV: kv_cache_dtype_raw is not None,
            ENABLE_PREFIX_CACHING_ENV: prefix_raw is not None,
            MAX_CUDAGRAPH_CAPTURE_SIZE_ENV: cudagraph_overridden,
            MOE_BACKEND_ENV: moe_backend_raw is not None,
            MTP_INDEX_SHARE_FOR_ITERATION_ENV: mtp_index_share_overridden,
            MTP_DYNAMIC_BATCH_SCHEDULE_ENV: mtp_dynamic_schedule_overridden,
        },
    }


def validate_vllm_tuning_document(value: Any) -> list[str]:
    """Validate persisted tuning without trusting it for process ownership."""
    if not isinstance(value, dict):
        return ["vllm_tuning_is_not_an_object"]
    errors: list[str] = []
    missing = sorted(VLLM_TUNING_FIELDS - set(value))
    unknown = sorted(set(value) - VLLM_TUNING_FIELDS)
    if missing:
        errors.append(f"vllm_tuning_missing_fields:{','.join(missing)}")
    if unknown:
        errors.append(f"vllm_tuning_unknown_fields:{','.join(unknown)}")
    if errors:
        return errors

    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        errors.append("vllm_tuning_wrong_schema_version")
    for field in (
        "enable_chunked_prefill",
        "enable_prefix_caching",
        "mtp_index_share_for_iteration",
    ):
        if type(value[field]) is not bool:
            errors.append(f"vllm_tuning_invalid_{field}")
    if not isinstance(value["kv_cache_dtype"], str) or value[
        "kv_cache_dtype"
    ] not in {"auto", "fp8"}:
        errors.append("vllm_tuning_invalid_kv_cache_dtype")
    if value["moe_backend"] not in {None, "flashinfer_b12x"}:
        errors.append("vllm_tuning_invalid_moe_backend")

    integer_bounds = {
        "kv_cache_memory_bytes": (0, MAX_EXPLICIT_KV_CACHE_MEMORY_BYTES),
        "max_cudagraph_capture_size": (0, 256),
        "max_num_batched_tokens": (1, 1_048_576),
        "max_num_batched_tokens_requested": (1, 1_048_576),
        "max_num_seqs": (1, 256),
        "mtp_speculative_tokens": (0, 4),
        "omp_num_threads": (1, 64),
    }
    for field, (minimum, maximum) in integer_bounds.items():
        item = value[field]
        if type(item) is not int or not minimum <= item <= maximum:
            errors.append(f"vllm_tuning_invalid_{field}")
    kv_bytes = value["kv_cache_memory_bytes"]
    if type(kv_bytes) is int and 0 < kv_bytes < MIN_EXPLICIT_KV_CACHE_MEMORY_BYTES:
        errors.append("vllm_tuning_invalid_kv_cache_memory_bytes")

    dynamic_schedule = value["mtp_dynamic_batch_schedule"]
    mtp_tokens = value["mtp_speculative_tokens"]
    if dynamic_schedule is not None:
        try:
            _validate_mtp_dynamic_batch_schedule(
                dynamic_schedule,
                max_speculative_tokens=(
                    mtp_tokens if type(mtp_tokens) is int and mtp_tokens > 0 else 0
                ),
            )
        except RuntimeError:
            errors.append("vllm_tuning_invalid_mtp_dynamic_batch_schedule")
    if (
        value["mtp_index_share_for_iteration"] is True
        and (type(mtp_tokens) is not int or mtp_tokens <= 0)
    ):
        errors.append("vllm_tuning_mtp_index_share_without_mtp")

    if (
        type(value["enable_chunked_prefill"]) is bool
        and type(value["max_num_batched_tokens"]) is int
        and type(value["max_num_batched_tokens_requested"]) is int
    ):
        if (
            value["enable_chunked_prefill"]
            and value["max_num_batched_tokens"]
            != value["max_num_batched_tokens_requested"]
        ):
            errors.append("vllm_tuning_chunked_batch_tokens_mismatch")
        if (
            not value["enable_chunked_prefill"]
            and value["max_num_batched_tokens"]
            < value["max_num_batched_tokens_requested"]
        ):
            errors.append("vllm_tuning_no_chunk_batch_tokens_mismatch")

    overridden = value["overridden"]
    if not isinstance(overridden, dict):
        errors.append("vllm_tuning_overridden_is_not_an_object")
    else:
        missing_overrides = sorted(VLLM_TUNING_OVERRIDE_FIELDS - set(overridden))
        unknown_overrides = sorted(set(overridden) - VLLM_TUNING_OVERRIDE_FIELDS)
        if missing_overrides:
            errors.append(
                "vllm_tuning_overridden_missing_fields:"
                + ",".join(missing_overrides)
            )
        if unknown_overrides:
            errors.append(
                "vllm_tuning_overridden_unknown_fields:"
                + ",".join(unknown_overrides)
            )
        if any(type(item) is not bool for item in overridden.values()):
            errors.append("vllm_tuning_overridden_values_are_not_boolean")
    return errors


def input_paths() -> dict[str, Path]:
    raw = os.environ.get("TAAF_KAGGLE_INPUT_PATHS", "").strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError("TAAF_KAGGLE_INPUT_PATHS must contain a JSON object.")
    return {str(key): Path(str(path)) for key, path in value.items()}


def source_identity() -> dict[str, Any]:
    path = BUNDLE_DIR / "SOURCE_IDENTITY.json"
    value = read_json(path)
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid source identity: {path}")
    expected_setup = str(value.get("serving_setup_sha256", ""))
    actual_setup = sha256_file(Path(__file__).resolve())
    if not expected_setup or expected_setup != actual_setup:
        raise RuntimeError(
            f"Serving setup identity mismatch: {actual_setup} != {expected_setup or '<missing>'}"
        )
    if value.get("model_manifest_sha256") != MODEL_MANIFEST_SHA256:
        raise RuntimeError("Source identity has the wrong model manifest hash.")
    if value.get("runtime_manifest_sha256") != VLLM_RUNTIME_MANIFEST_SHA256:
        raise RuntimeError("Source identity has the wrong vLLM runtime manifest hash.")
    model = value.get("model") or {}
    if model.get("hf_repo") != MODEL_HF_REPO or model.get("hf_revision") != MODEL_HF_REVISION:
        raise RuntimeError("Source identity has the wrong model revision.")
    runtime = value.get("runtime") or {}
    if runtime.get("image") != VLLM_IMAGE or runtime.get("kaggle_dataset") != RUNTIME_DATASET:
        raise RuntimeError("Source identity has the wrong vLLM runtime.")
    return value


def resolve_runtime_dir() -> Path:
    mapped = input_paths().get(RUNTIME_DATASET)
    candidates = [
        mapped,
        Path("/kaggle/input") / RUNTIME_DATASET.split("/", 1)[1],
        Path("/kaggle/input/datasets") / RUNTIME_DATASET,
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "runtime-manifest.json").is_file():
            return candidate
    matches = []
    for path in Path("/kaggle/input").rglob("runtime-manifest.json"):
        try:
            manifest = read_json(path)
        except Exception:
            continue
        if manifest.get("index_digest") == VLLM_INDEX_DIGEST:
            matches.append(path.parent)
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not resolve one pinned {RUNTIME_DATASET}; matches={matches}")
    return matches[0]


def resolve_model_dir() -> Path:
    manifest_path = MODEL_KAGGLE_PATH / "MODEL_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Expected mounted {MODEL_HF_REPO}@{MODEL_HF_REVISION} at "
            f"{MODEL_KAGGLE_PATH}."
        )
    value = read_json(manifest_path)
    if (
        not isinstance(value, dict)
        or value.get("repo") != MODEL_HF_REPO
        or value.get("revision") != MODEL_HF_REVISION
    ):
        raise RuntimeError(f"Mounted model identity is wrong at {MODEL_KAGGLE_PATH}.")
    return MODEL_KAGGLE_PATH


def verify_model(model_dir: Path, *, full_file_hashes: bool = True) -> dict[str, Any]:
    manifest_path = model_dir / "MODEL_MANIFEST.json"
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != MODEL_MANIFEST_SHA256:
        raise RuntimeError(
            f"Model manifest hash mismatch: {actual_manifest_sha} != {MODEL_MANIFEST_SHA256}"
        )
    manifest = read_json(manifest_path)
    if manifest.get("repo") != MODEL_HF_REPO or manifest.get("revision") != MODEL_HF_REVISION:
        raise RuntimeError("Mounted model provenance does not match the pinned checkpoint.")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != MODEL_FILE_COUNT:
        raise RuntimeError(f"Expected {MODEL_FILE_COUNT} model files, found {len(files or [])}.")
    try:
        total = sum(int(row["size"]) for row in files)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Model manifest file metadata is invalid.") from exc
    if total != MODEL_TOTAL_BYTES or total != int(manifest.get("total_bytes", -1)):
        raise RuntimeError(f"Model byte total mismatch: {total} != {MODEL_TOTAL_BYTES}")

    hashed_files = 0
    hashed_bytes = 0
    stat_only_weight_files = 0
    stat_only_weight_bytes = 0
    payload_files_checked = 0
    started = time.monotonic()
    if full_file_hashes:
        for index, row in enumerate(files, start=1):
            relative = str(row["path"])
            path = model_dir / relative
            expected_size = int(row["size"])
            if not path.is_file() or path.stat().st_size != expected_size:
                raise RuntimeError(f"Missing or wrong-sized model file: {path}")
            actual_sha = sha256_file(path)
            if actual_sha != str(row["sha256"]):
                raise RuntimeError(f"Model file hash mismatch: {relative}")
            hashed_files += 1
            hashed_bytes += expected_size
            payload_files_checked += 1
            if index == len(files) or index % 25 == 0:
                print(
                    f"MODEL_VERIFY files={index}/{len(files)} bytes={hashed_bytes} "
                    f"elapsed_s={time.monotonic() - started:.1f}",
                    flush=True,
                )
    else:
        print(
            f"MODEL_IDENTITY_ONLY files={len(files)} bytes={total} "
            "payload_check=vllm_load",
            flush=True,
        )

    config_path = model_dir / "config.json"
    quant_path = model_dir / "hf_quant_config.json"
    if sha256_file(config_path) != MODEL_CONFIG_SHA256:
        raise RuntimeError("Pinned model config hash does not match.")
    if sha256_file(quant_path) != MODEL_QUANT_CONFIG_SHA256:
        raise RuntimeError("Pinned ModelOpt quantization config hash does not match.")
    config = read_json(config_path)
    quant = read_json(quant_path)
    text_config = config.get("text_config") or {}
    quantization = config.get("quantization_config") or {}
    if config.get("architectures") != ["Qwen4ExpForConditionalGeneration"]:
        raise RuntimeError(f"Wrong model architecture: {config.get('architectures')}")
    if config.get("model_type") != "qwen4_exp":
        raise RuntimeError(f"Wrong model type: {config.get('model_type')}")
    if (
        quantization.get("quant_method") != "modelopt"
        or quantization.get("quant_algo") != "NVFP4"
        or (quantization.get("producer") or {}).get("version") != "0.46.0"
    ):
        raise RuntimeError(f"Wrong in-model quantization identity: {quantization}")
    quant_details = quant.get("quantization") or {}
    if quant_details.get("quant_algo") != "NVFP4" or int(quant_details.get("group_size", -1)) != 16:
        raise RuntimeError(f"Wrong ModelOpt NVFP4 details: {quant_details}")
    if (
        text_config.get("ple_embedding_dtype") != "float8_e4m3fn"
        or text_config.get("ple_layer_ids") != [2]
        or int(text_config.get("mtp_num_hidden_layers", -1)) != 1
        or int((text_config.get("mtp") or {}).get("num_hidden_layers", -1)) != 1
    ):
        raise RuntimeError("The pinned FP8 PLE or native MTP identity does not match.")
    for name in ("model.safetensors.index.json", "chat_template.jinja"):
        if not (model_dir / name).is_file():
            raise RuntimeError(f"Incomplete checkpoint; missing {name}")
    return {
        "manifest_sha256": actual_manifest_sha,
        "config_sha256": MODEL_CONFIG_SHA256,
        "quant_config_sha256": MODEL_QUANT_CONFIG_SHA256,
        "file_count": len(files),
        "total_bytes": total,
        "architecture": config["architectures"][0],
        "quant_algo": quantization["quant_algo"],
        "ple_embedding_dtype": text_config["ple_embedding_dtype"],
        "mtp_num_hidden_layers": text_config["mtp_num_hidden_layers"],
        "payload_sha256_verified": full_file_hashes,
        "hashed_file_count": hashed_files,
        "hashed_bytes": hashed_bytes,
        "stat_only_weight_file_count": stat_only_weight_files,
        "stat_only_weight_bytes": stat_only_weight_bytes,
        "payload_files_checked": payload_files_checked,
        "payload_check_deferred_to_vllm_load": not full_file_hashes,
        "verification_mode": "full" if full_file_hashes else "identity-only",
        "verify_seconds": time.monotonic() - started,
    }


def _apply_whiteouts(expected_markers: tuple[str, ...] | None = None) -> list[str]:
    applied: list[str] = []
    if expected_markers is None:
        whiteouts = sorted(
            RUNTIME_ROOT.rglob(".wh.*"),
            key=lambda path: len(path.parts),
            reverse=True,
        )
    else:
        whiteouts = [RUNTIME_ROOT / relative for relative in expected_markers]
        missing = [str(path) for path in whiteouts if not os.path.lexists(path)]
        if missing:
            raise RuntimeError(f"Expected pinned runtime whiteouts are missing: {missing}")
    for marker in whiteouts:
        if marker.name == ".wh..wh..opq":
            raise RuntimeError(f"Unsupported opaque runtime whiteout: {marker}")
        target = marker.with_name(marker.name[4:])
        if target.is_symlink() or target.is_file():
            target.unlink(missing_ok=True)
        elif target.is_dir():
            shutil.rmtree(target)
        applied.append(str(marker.relative_to(RUNTIME_ROOT)))
        marker.unlink(missing_ok=True)
    return applied


def validate_runtime_storage() -> dict[str, Any]:
    roots = (RUNTIME_ROOT, CACHE_ROOT, COMPILE_CACHE_ROOT, TMP_ROOT)
    rows: list[dict[str, Any]] = []
    devices: set[int] = set()
    for root in roots:
        if root.is_symlink():
            raise RuntimeError(f"A runtime or cache root must not be a symlink: {root}")
        probe = root.parent
        if not probe.is_dir() or probe.is_symlink():
            raise RuntimeError(f"The runtime or cache parent is not a regular directory: {probe}")
        stat = probe.stat()
        if root.exists() and root.stat().st_dev != stat.st_dev:
            raise RuntimeError(f"A runtime or cache root crossed its audited filesystem: {root}")
        filesystem = os.statvfs(probe)
        free_bytes = int(filesystem.f_bavail * filesystem.f_frsize)
        devices.add(int(stat.st_dev))
        rows.append(
            {
                "root": str(root),
                "statvfs_path": str(probe.resolve()),
                "device": int(stat.st_dev),
                "free_bytes": free_bytes,
                "threshold_bytes": MIN_RUNTIME_FILESYSTEM_FREE_BYTES,
            }
        )
        if free_bytes < MIN_RUNTIME_FILESYSTEM_FREE_BYTES:
            raise RuntimeError(
                f"The filesystem for {root} has too little free space: "
                f"{free_bytes} < {MIN_RUNTIME_FILESYSTEM_FREE_BYTES}."
            )
    if len(devices) != 1:
        raise RuntimeError(
            "The pinned runtime and JIT/cache roots do not share one audited filesystem: "
            f"{rows}"
        )
    return {
        "roots": rows,
        "shared_device": next(iter(devices)),
        "minimum_free_bytes": MIN_RUNTIME_FILESYSTEM_FREE_BYTES,
        "conservative_extracted_bound_bytes": RUNTIME_EXTRACTED_BOUND_BYTES,
        "jit_cache_reserve_bytes": RUNTIME_CACHE_RESERVE_BYTES,
        "audit_sha256": RUNTIME_STORAGE_AUDIT_SHA256,
    }


def verify_and_extract_runtime(
    runtime_dir: Path,
    *,
    full_layer_hashes: bool = True,
    scan_extracted_caches: bool = True,
) -> dict[str, Any]:
    manifest_path = runtime_dir / "runtime-manifest.json"
    actual_sha = sha256_file(manifest_path)
    if actual_sha != VLLM_RUNTIME_MANIFEST_SHA256:
        raise RuntimeError(
            f"Runtime manifest hash mismatch: {actual_sha} != {VLLM_RUNTIME_MANIFEST_SHA256}"
        )
    manifest = read_json(manifest_path)
    expected_header = {
        "image": VLLM_IMAGE,
        "index_digest": VLLM_INDEX_DIGEST,
        "amd64_manifest_digest": VLLM_AMD64_MANIFEST_DIGEST,
        "config_digest": VLLM_CONFIG_DIGEST,
        "selected_compressed_bytes": 7_906_697_104,
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"Wrong vLLM runtime {key}: {manifest.get(key)!r} != {expected!r}")
    layers = manifest.get("selected_layers")
    if not isinstance(layers, list) or len(layers) != len(EXPECTED_RUNTIME_LAYERS):
        raise RuntimeError("Runtime manifest has the wrong selected layer count.")
    layer_map = {int(row["index"]): row for row in layers}
    if set(layer_map) != set(EXPECTED_RUNTIME_LAYERS):
        raise RuntimeError(f"Wrong runtime layer indexes: {sorted(layer_map)}")

    shutil.rmtree(RUNTIME_ROOT, ignore_errors=True)
    RUNTIME_ROOT.mkdir(parents=True)
    started = time.monotonic()
    verified: list[dict[str, Any]] = []
    whiteouts: list[str] = []
    for index in sorted(EXPECTED_RUNTIME_LAYERS):
        expected_hash, expected_size = EXPECTED_RUNTIME_LAYERS[index]
        row = layer_map[index]
        if (
            row.get("sha256") != expected_hash
            or row.get("digest") != f"sha256:{expected_hash}"
            or int(row.get("size", -1)) != expected_size
        ):
            raise RuntimeError(f"Runtime layer {index} identity does not match: {row}")
        layer_path = runtime_dir / str(row["file"])
        if not layer_path.is_file():
            layer_path = runtime_dir / f"{row['file']}.blob"
        if not layer_path.is_file() or layer_path.stat().st_size != expected_size:
            raise RuntimeError(f"Missing or wrong-sized runtime layer: {layer_path}")
        actual_layer_sha = None
        if full_layer_hashes:
            actual_layer_sha = sha256_file(layer_path)
            if actual_layer_sha != expected_hash:
                raise RuntimeError(f"Runtime layer hash mismatch: {layer_path.name}")
            print(
                f"RUNTIME_LAYER_VERIFY index={index} bytes={expected_size} "
                f"sha256={actual_layer_sha}",
                flush=True,
            )
        else:
            print(
                f"RUNTIME_LAYER_FAST_CHECK index={index} bytes={expected_size} "
                f"manifest_sha256={expected_hash}",
                flush=True,
            )
        subprocess.run(
            [
                "tar",
                "-xzf",
                str(layer_path),
                "-C",
                str(RUNTIME_ROOT),
                "--no-same-owner",
                "--exclude=*/__pycache__",
                "--exclude=*/__pycache__/*",
                "--exclude=*.pyc",
                "--exclude=*.pyo",
            ],
            check=True,
        )
        expected_whiteouts = (
            None if full_layer_hashes else FAST_RUNTIME_WHITEOUTS.get(index, ())
        )
        whiteouts.extend(_apply_whiteouts(expected_whiteouts))
        verified.append(
            {
                "index": index,
                "file": str(row["file"]),
                "stored_file": layer_path.name,
                "sha256": expected_hash,
                "manifest_sha256": expected_hash,
                "actual_sha256": actual_layer_sha,
                "payload_sha256_verified": full_layer_hashes,
                "size": expected_size,
            }
        )
    if scan_extracted_caches:
        cache_artifacts = [
            path
            for path in RUNTIME_ROOT.rglob("*")
            if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
        ]
        if cache_artifacts:
            raise RuntimeError(
                f"Runtime extraction produced Python cache files: {cache_artifacts[:20]}"
            )
    return {
        "manifest_sha256": actual_sha,
        "image": VLLM_IMAGE,
        "index_digest": VLLM_INDEX_DIGEST,
        "amd64_manifest_digest": VLLM_AMD64_MANIFEST_DIGEST,
        "layers": verified,
        "whiteouts_applied": whiteouts,
        "payload_sha256_verified": full_layer_hashes,
        "cache_scan_performed": scan_extracted_caches,
        "verification_mode": "full" if full_layer_hashes else "fast",
        "extract_seconds": time.monotonic() - started,
    }


def patch_ple_layer() -> dict[str, Any]:
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    path = site / "vllm" / "models" / "qwen3_8_flash_next" / "nvidia" / "ple_layer.py"
    patch_dir = BUNDLE_DIR / PLE_PATCH_DIRNAME
    expected_files = {
        PLE_PATCH_APPLICATOR: (PLE_PATCH_APPLICATOR_BYTES, PLE_PATCH_APPLICATOR_SHA256),
        PLE_PATCH_IDENTITY: (PLE_PATCH_IDENTITY_BYTES, PLE_PATCH_IDENTITY_SHA256),
        PLE_PATCH_DIFF: (PLE_PATCH_DIFF_BYTES, PLE_PATCH_DIFF_SHA256),
        PLE_PATCH_README: (PLE_PATCH_README_BYTES, PLE_PATCH_README_SHA256),
    }
    if not patch_dir.is_dir() or patch_dir.is_symlink():
        raise RuntimeError(f"Missing regular PLE patch directory: {patch_dir}")
    actual_names = {entry.name for entry in patch_dir.iterdir()}
    if actual_names != set(expected_files):
        raise RuntimeError(
            "PLE patch bundle allowlist mismatch: "
            f"actual={sorted(actual_names)} expected={sorted(expected_files)}"
        )
    artifact_rows: dict[str, dict[str, Any]] = {}
    for name, (expected_size, expected_sha) in expected_files.items():
        artifact = patch_dir / name
        if not artifact.is_file() or artifact.is_symlink():
            raise RuntimeError(f"PLE patch payload is not a regular file: {artifact}")
        actual_size = artifact.stat().st_size
        actual_sha = sha256_file(artifact)
        if actual_size != expected_size or actual_sha != expected_sha:
            raise RuntimeError(
                f"PLE patch payload mismatch for {name}: "
                f"size={actual_size}/{expected_size} sha256={actual_sha}/{expected_sha}"
            )
        artifact_rows[name] = {"bytes": actual_size, "sha256": actual_sha}

    patch_identity = read_json(patch_dir / PLE_PATCH_IDENTITY)
    required_identity = {
        "artifact": PLE_PATCH_DIFF,
        "artifact_bytes": PLE_PATCH_DIFF_BYTES,
        "artifact_sha256": PLE_PATCH_DIFF_SHA256,
        "exact_model_config": "model/RadixArk-Qwen3.8-Flash-Next-NVFP4/config.json",
        "exact_model_config_bytes": 4_983,
        "exact_model_config_producer": {"name": "modelopt", "version": "0.46.0"},
        "exact_model_config_sha256": MODEL_CONFIG_SHA256,
        "modelopt_internal_gate": {
            "checkpoint_excluded_prefix": PLE_EXCLUDED_PREFIX,
            "group_size": 16,
            "is_checkpoint_nvfp4_serialized": True,
            "is_layer_excluded_checkpoint_prefix": True,
            "quant_method": "NVFP4",
            "runtime_prefix": PLE_EXACT_PREFIX,
        },
        "patched_target_sha256": VLLM_PLE_PATCHED_SHA256,
        "stock_target_sha256": VLLM_PLE_STOCK_SHA256,
        "target": "vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py",
        "vllm_image_amd64_manifest": VLLM_AMD64_MANIFEST_DIGEST,
        "vllm_image_index_digest": VLLM_INDEX_DIGEST,
        "vllm_version": VLLM_VERSION,
    }
    if patch_identity != required_identity:
        raise RuntimeError("PATCH_IDENTITY.json did not match the exact approved patch identity.")

    before_sha = sha256_file(path)
    if before_sha != VLLM_PLE_STOCK_SHA256:
        raise RuntimeError(f"Stock vLLM PLE source hash mismatch: {before_sha}")
    completed = subprocess.run(
        [sys.executable, "-B", str(patch_dir / PLE_PATCH_APPLICATOR), str(site)],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0 or completed.stdout.strip() != "patched":
        raise RuntimeError(
            "Pinned PLE patch applicator failed: "
            f"returncode={completed.returncode} stdout={completed.stdout!r} "
            f"stderr={completed.stderr!r}"
        )
    after_sha = sha256_file(path)
    if after_sha != VLLM_PLE_PATCHED_SHA256:
        raise RuntimeError(f"Written vLLM PLE patch hash mismatch: {after_sha}")
    cache_artifacts = [
        child
        for root in (patch_dir, site / "vllm" / "models" / "qwen3_8_flash_next")
        for child in root.rglob("*")
        if child.name == "__pycache__" or child.suffix in {".pyc", ".pyo"}
    ]
    if cache_artifacts:
        raise RuntimeError(f"PLE patching produced forbidden Python cache files: {cache_artifacts}")
    return {
        "path": str(path),
        "stock_sha256": before_sha,
        "patched_sha256": after_sha,
        "artifact_directory": str(patch_dir),
        "artifacts": artifact_rows,
        "patch_identity": patch_identity,
        "applicator_result": completed.stdout.strip(),
        "gate_environment": "VLLM_RADIXARK_QWEN38_NVFP4_PLE_FP8=1",
        "gate_model_config_sha256": MODEL_CONFIG_SHA256,
        "gate_runtime_prefix": PLE_EXACT_PREFIX,
        "gate_checkpoint_excluded_prefix": PLE_EXCLUDED_PREFIX,
    }


def _find_real_cuda_driver() -> Path:
    candidates: list[Path] = []
    completed = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, check=True)
    for line in completed.stdout.splitlines():
        if "libcuda.so.1" not in line or "=>" not in line:
            continue
        candidate = Path(line.split("=>", 1)[1].strip())
        if candidate.is_file() and "stubs" not in str(candidate).lower():
            candidates.append(candidate.resolve())
    search_dirs = [
        Path(entry)
        for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if entry
    ]
    search_dirs.extend([Path("/usr/local/nvidia/lib64"), Path("/usr/lib/x86_64-linux-gnu")])
    for directory in search_dirs:
        candidate = directory / "libcuda.so.1"
        if candidate.is_file() and "stubs" not in str(candidate.resolve()).lower():
            candidates.append(candidate.resolve())
    if not candidates:
        raise RuntimeError("No real non-stub libcuda.so.1 was found.")
    return max(set(candidates), key=lambda path: path.stat().st_size)


def validate_jit_toolchain(
    env: dict[str, str], cuda_home: Path, cuda_lib: Path, nvcc: Path
) -> dict[str, Any]:
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"The pinned vLLM runtime requires CPython 3.12, found {sys.implementation.name} "
            f"{sys.version_info.major}.{sys.version_info.minor}."
        )
    glibc_raw = os.confstr("CS_GNU_LIBC_VERSION")
    glibc_match = re.fullmatch(r"glibc\s+(\d+)\.(\d+)", glibc_raw or "")
    if glibc_match is None or tuple(map(int, glibc_match.groups())) < (2, 28):
        raise RuntimeError(f"The vLLM runtime requires glibc >=2.28, found {glibc_raw!r}.")
    resolved: dict[str, Path] = {}
    for name in ("gcc", "g++", "ld", "ninja"):
        value = shutil.which(name, path=env["PATH"])
        if not value:
            raise RuntimeError(f"Required vLLM JIT tool was absent from final PATH: {name}")
        resolved[name] = Path(value).resolve()
    if nvcc.resolve() != Path(shutil.which("nvcc", path=env["PATH"]) or "").resolve():
        raise RuntimeError("Final PATH did not resolve nvcc to the pinned CUDA 13 toolkit.")
    if any(str(resolved[name]).startswith(str(RUNTIME_ROOT)) for name in ("gcc", "g++", "ld")):
        raise RuntimeError("Relocated image GCC was selected without validated absolute spec paths.")
    python_h = Path(sysconfig.get_path("include")) / "Python.h"
    if not python_h.is_file():
        raise RuntimeError(f"The Kaggle runtime Python development header is absent: {python_h}")
    if not (cuda_lib / "libcudart.so").is_file():
        raise RuntimeError(f"The pinned CUDA runtime linker name is absent: {cuda_lib}")

    version_commands = {
        "gcc": [str(resolved["gcc"]), "--version"],
        "g++": [str(resolved["g++"]), "--version"],
        "ld": [str(resolved["ld"]), "--version"],
        "ninja": [str(resolved["ninja"]), "--version"],
        "nvcc": [str(nvcc), "--version"],
        "python": [sys.executable, "--version"],
    }
    versions: dict[str, str] = {}
    for name, command in version_commands.items():
        completed = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=60
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"JIT tool version probe failed for {name}: "
                f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
            )
        versions[name] = (completed.stdout + completed.stderr).strip()
    if "release 13.0" not in versions["nvcc"]:
        raise RuntimeError(f"The pinned nvcc was not CUDA 13.0: {versions['nvcc']}")
    if not versions["python"].startswith("Python 3.12"):
        raise RuntimeError(f"The vLLM runtime requires Python 3.12: {versions['python']}")

    libstdcpp_query = subprocess.run(
        [str(resolved["g++"]), "-print-file-name=libstdc++.so.6"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    libstdcpp_raw = libstdcpp_query.stdout.strip()
    libstdcpp = Path(libstdcpp_raw).resolve() if libstdcpp_raw else Path()
    if (
        libstdcpp_query.returncode != 0
        or not libstdcpp_raw
        or libstdcpp_raw == "libstdc++.so.6"
        or not libstdcpp.is_file()
    ):
        raise RuntimeError(
            "The host g++ did not resolve a real libstdc++.so.6: "
            f"returncode={libstdcpp_query.returncode} stdout={libstdcpp_query.stdout!r} "
            f"stderr={libstdcpp_query.stderr!r}"
        )
    libstdcpp_bytes = libstdcpp.read_bytes()
    required_abi_symbols = (b"GLIBCXX_3.4.22", b"CXXABI_1.3.11")
    missing_abi_symbols = [
        symbol.decode("ascii") for symbol in required_abi_symbols if symbol not in libstdcpp_bytes
    ]
    if missing_abi_symbols:
        raise RuntimeError(
            f"The host libstdc++ is missing required ABI symbols: {missing_abi_symbols}"
        )

    probe_root = TMP_ROOT / "jit-toolchain-probe"
    shutil.rmtree(probe_root, ignore_errors=True)
    probe_root.mkdir(parents=True)
    source = (
        "#include <Python.h>\n"
        "#include <cuda_runtime_api.h>\n"
        "__global__ void duck_probe_kernel(int* value) { *value = 13000; }\n"
        'extern "C" int duck_vllm_jit_probe() {\n'
        "  int version = 0, host_value = 0;\n"
        "  int* device_value = nullptr;\n"
        "  if (cudaRuntimeGetVersion(&version) != cudaSuccess || version != 13000) return -1;\n"
        "  if (cudaMalloc(&device_value, sizeof(int)) != cudaSuccess) return -2;\n"
        "  duck_probe_kernel<<<1, 1>>>(device_value);\n"
        "  if (cudaGetLastError() != cudaSuccess) { cudaFree(device_value); return -3; }\n"
        "  if (cudaDeviceSynchronize() != cudaSuccess) { cudaFree(device_value); return -4; }\n"
        "  if (cudaMemcpy(&host_value, device_value, sizeof(int), cudaMemcpyDeviceToHost) "
        "!= cudaSuccess) { cudaFree(device_value); return -5; }\n"
        "  if (cudaFree(device_value) != cudaSuccess) return -6;\n"
        "  return host_value;\n"
        "}\n"
    )
    source_path = probe_root / "probe.cu"
    object_path = probe_root / "probe.o"
    library_path = probe_root / "probe.so"
    source_path.write_text(source, encoding="utf-8", newline="\n")
    compile_command = [
        str(nvcc),
        "-std=c++17",
        "-arch=sm_120",
        "-Xcompiler",
        "-fPIC",
        "-ccbin",
        str(resolved["g++"]),
        f"-I{python_h.parent}",
        "-c",
        str(source_path),
        "-o",
        str(object_path),
    ]
    link_command = [
        str(resolved["g++"]),
        "-shared",
        str(object_path),
        f"-L{cuda_lib}",
        f"-Wl,-rpath,{cuda_lib}",
        "-lcudart",
        "-o",
        str(library_path),
    ]

    def run_checked(command: list[str], timeout: int) -> dict[str, Any]:
        completed = subprocess.run(
            command,
            env=env,
            cwd=probe_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "JIT-equivalent toolchain probe failed: "
                f"command={command} returncode={completed.returncode} "
                f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
            )
        return {
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    try:
        compiled = run_checked(compile_command, 180)
        if not object_path.is_file() or object_path.stat().st_size <= 0:
            raise RuntimeError("nvcc did not create the JIT-equivalent CUDA object.")
        linked = run_checked(link_command, 120)
        if not library_path.is_file() or library_path.stat().st_size <= 0:
            raise RuntimeError("g++ did not create the JIT-equivalent shared library.")
        library_sha256 = sha256_file(library_path)
        load_script = """
import ctypes, json, sys
library = ctypes.CDLL(sys.argv[1])
function = library.duck_vllm_jit_probe
function.argtypes = []
function.restype = ctypes.c_int
print(json.dumps({"cuda_runtime_version": function()}))
"""
        loaded = subprocess.run(
            [sys.executable, "-c", load_script, str(library_path)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if loaded.returncode != 0:
            raise RuntimeError(
                "ctypes could not load the JIT-equivalent shared library: "
                f"stdout={loaded.stdout!r} stderr={loaded.stderr!r}"
            )
        load_result = json.loads(loaded.stdout.strip().splitlines()[-1])
        if int(load_result.get("cuda_runtime_version", -1)) != 13_000:
            raise RuntimeError(f"Wrong CUDA runtime linked into JIT probe: {load_result}")
        return {
            "compiler_source": "Kaggle host compilers with pinned CUDA 13.0 toolkit",
            "python_runtime": {
                "implementation": sys.implementation.name,
                "version": sys.version,
                "glibc": glibc_raw,
            },
            "resolved_paths": {
                **{name: str(path) for name, path in resolved.items()},
                "nvcc": str(nvcc.resolve()),
                "python": str(Path(sys.executable).resolve()),
                "Python.h": str(python_h.resolve()),
                "libcudart.so": str((cuda_lib / "libcudart.so").resolve()),
                "cuda_home": str(cuda_home.resolve()),
            },
            "versions": versions,
            "host_cxx_abi": {
                "libstdcpp": str(libstdcpp),
                "libstdcpp_sha256": sha256_file(libstdcpp),
                "required_symbols": [
                    symbol.decode("ascii") for symbol in required_abi_symbols
                ],
            },
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "compile": compiled,
            "link": linked,
            "shared_library_sha256": library_sha256,
            "ctypes_load": load_result,
        }
    finally:
        shutil.rmtree(probe_root, ignore_errors=True)


def runtime_environment(
    *,
    deep_preload_validation: bool = True,
    tuning: dict[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    resolved_tuning = tuning or resolve_vllm_tuning()
    allocator_input = {
        key: os.environ.get(key)
        for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
    }
    expandable_true = re.compile(
        r"\bexpandable_segments\s*:\s*(?:true|1|yes)\b",
        re.IGNORECASE,
    )
    bad_allocator = {
        key: value
        for key, value in allocator_input.items()
        if value is not None and expandable_true.search(value)
    }
    if bad_allocator:
        raise RuntimeError(
            "Expandable CUDA allocator segments are forbidden for PLE CPU-offload IPC: "
            f"{bad_allocator}"
        )
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    cutlass_pth = site / "nvidia_cutlass_dsl_packages.pth"
    cutlass_nested = site / "nvidia_cutlass_dsl" / "dsl_packages"
    cuda_home = RUNTIME_ROOT / "usr" / "local" / "cuda-13.0"
    cuda_lib = cuda_home / "targets" / "x86_64-linux" / "lib"
    nvcc = cuda_home / "bin" / "nvcc"
    ple_path = (
        site
        / "vllm"
        / "models"
        / "qwen3_8_flash_next"
        / "nvidia"
        / "ple_layer.py"
    )
    required = [
        site / "torch" / "__init__.py",
        site / "vllm" / "__init__.py",
        site / "flashinfer" / "__init__.py",
        ple_path,
        nvcc,
        cuda_lib / "libcudart.so",
        cutlass_pth,
        cutlass_nested / "cutlass" / "__init__.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Incomplete pinned vLLM runtime: {missing}")
    if sha256_file(ple_path) != VLLM_PLE_PATCHED_SHA256:
        raise RuntimeError("The patched PLE source hash changed before import.")
    if sha256_file(cutlass_pth) != CUTLASS_PTH_SHA256:
        raise RuntimeError("The pinned nvidia-cutlass-dsl .pth hash changed.")
    if cutlass_pth.read_text(encoding="utf-8").strip() != CUTLASS_PTH_TEXT:
        raise RuntimeError("The pinned nvidia-cutlass-dsl .pth instruction changed.")
    real_driver = _find_real_cuda_driver()

    pinned_nvidia_libraries = sorted(
        path for path in (site / "nvidia").glob("*/lib") if path.is_dir()
    )
    library_dirs = [
        cuda_lib,
        cuda_home / "lib64",
        site / "torch" / "lib",
        *pinned_nvidia_libraries,
        real_driver.parent,
        Path("/usr/local/nvidia/lib64"),
        Path("/usr/lib/x86_64-linux-gnu"),
    ]
    for root in (CACHE_ROOT, COMPILE_CACHE_ROOT, TMP_ROOT):
        root.mkdir(parents=True, exist_ok=True)
    cache_paths = {
        "HF_HOME": CACHE_ROOT / "huggingface",
        "XDG_CACHE_HOME": CACHE_ROOT,
        "TORCH_HOME": CACHE_ROOT / "torch",
        "TORCHINDUCTOR_CACHE_DIR": COMPILE_CACHE_ROOT / "torchinductor",
        "TRITON_CACHE_DIR": COMPILE_CACHE_ROOT / "triton",
        "CUDA_CACHE_PATH": COMPILE_CACHE_ROOT / "cuda",
        "FLASHINFER_WORKSPACE_BASE": COMPILE_CACHE_ROOT / "flashinfer",
    }
    for path in cache_paths.values():
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.pop("PYTORCH_ALLOC_CONF", None)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(cutlass_nested), str(site)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["PATH"] = os.pathsep.join(
        [str(cuda_home / "bin"), str(RUNTIME_ROOT / "usr" / "local" / "bin"), env.get("PATH", "")]
    )
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(path) for path in library_dirs if path.is_dir()]
        + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
    )
    for key, path in cache_paths.items():
        env[key] = str(path)
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_HOME": str(cuda_home),
            "CUDACXX": str(nvcc),
            "TMPDIR": str(TMP_ROOT),
            "VLLM_PLE_CPU_OFFLOAD": "1",
            "VLLM_PLE_OFFLOAD_READY_TIMEOUT": str(SERVER_READY_TIMEOUT),
            "VLLM_ENABLE_CUDA_COMPATIBILITY": "0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
            "VLLM_RADIXARK_QWEN38_NVFP4_PLE_FP8": "1",
            "VLLM_RADIXARK_QWEN38_NVFP4_CONFIG_SHA256": MODEL_CONFIG_SHA256,
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "TORCH_CUDA_ARCH_LIST": "12.0",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": str(resolved_tuning["omp_num_threads"]),
            "MKL_NUM_THREADS": str(resolved_tuning["omp_num_threads"]),
        }
    )
    if not deep_preload_validation:
        if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
            raise RuntimeError(
                "The pinned vLLM runtime requires CPython 3.12, found "
                f"{sys.implementation.name} {sys.version_info.major}.{sys.version_info.minor}."
            )
        glibc_raw = os.confstr("CS_GNU_LIBC_VERSION")
        glibc_match = re.fullmatch(r"glibc\s+(\d+)\.(\d+)", glibc_raw or "")
        if glibc_match is None or tuple(map(int, glibc_match.groups())) < (2, 28):
            raise RuntimeError(f"The vLLM runtime requires glibc >=2.28, found {glibc_raw!r}.")
        tools = {
            name: shutil.which(name, path=env["PATH"])
            for name in ("gcc", "g++", "ld", "ninja", "nvcc")
        }
        missing_tools = [name for name, path in tools.items() if not path]
        if missing_tools:
            raise RuntimeError(f"Required vLLM tools are absent: {missing_tools}")
        return env, {
            "validation_mode": "fast",
            "deep_preload_validation": False,
            "skipped_duplicate_probes": [
                "CUDA compile/link/device probe",
                "torch/flashinfer/vllm import subprocess",
                "NCCL metadata and runtime probe",
            ],
            "allocator_input": allocator_input,
            "allocator_effective": "expandable_segments:False",
            "allocator_alias_removed": True,
            "cuda_compatibility": "0",
            "torch_cuda_arch_list": "12.0",
            "python": sys.version,
            "glibc": glibc_raw,
            "site_packages": str(site),
            "cutlass_pth": str(cutlass_pth),
            "cutlass_pth_sha256": CUTLASS_PTH_SHA256,
            "cutlass_nested_path": str(cutlass_nested),
            "cuda_home": str(cuda_home),
            "cuda_library_dir": str(cuda_lib),
            "real_cuda_driver": str(real_driver),
            "required_tools": tools,
            "ple_patch_sha256": VLLM_PLE_PATCHED_SHA256,
        }
    jit_toolchain = validate_jit_toolchain(env, cuda_home, cuda_lib, nvcc)
    nvcc_result = subprocess.run(
        [str(nvcc), "--version"], env=env, capture_output=True, text=True, check=True, timeout=60
    )
    import_script = """
from importlib import metadata
import inspect, json, os
from types import SimpleNamespace
import cutlass, torch, flashinfer, vllm
import vllm.models.qwen3_8_flash_next.nvidia.ple_layer as ple_module
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.models.qwen3_8_flash_next.nvidia.ple_layer import (
    Qwen3_8FlashNextPLEFp8EmbeddingMethod,
    _get_ple_embedding_quant_method,
    _is_exact_radixark_nvfp4_ple,
)
runtime_prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
excluded_prefix = "model.language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
config = ModelOptNvFp4Config(
    quant_method="NVFP4",
    is_checkpoint_nvfp4_serialized=True,
    exclude_modules=["*.ple.*"],
    group_size=16,
)
text_config = SimpleNamespace(
    model_type="qwen4_exp_text",
    hidden_size=2560,
    num_hidden_layers=48,
    num_experts=512,
    ple_embed_dim=2560,
    ple_embedding_dtype="float8_e4m3fn",
    ngram_size=3,
    heads_per_ngram=8,
    ngram_vocab_size_base=20_000_000,
    split_ngram_parts=128,
    ple_layer_ids=[2],
    layer_types=[
        "full_attention" if (layer_idx + 1) % 4 == 0 else "linear_attention"
        for layer_idx in range(48)
    ],
)
method = _get_ple_embedding_quant_method(
    config,
    runtime_prefix,
    text_config,
)
checkpoint_method = _get_ple_embedding_quant_method(
    config,
    excluded_prefix,
    text_config,
)
print(json.dumps({
    "allocator_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    "allocator_alias": os.environ.get("PYTORCH_ALLOC_CONF"),
    "cuda_compatibility": os.environ.get("VLLM_ENABLE_CUDA_COMPATIBILITY"),
    "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_nccl": list(torch.cuda.nccl.version()),
    "torch_nccl_requirements": [
        item for item in (metadata.requires("torch") or [])
        if item.lower().startswith("nvidia-nccl-cu13")
    ],
    "installed_nccl_dist": metadata.version("nvidia-nccl-cu13"),
    "flashinfer": getattr(flashinfer, "__version__", "unknown"),
    "cutlass": metadata.version("nvidia-cutlass-dsl"),
    "cutlass_file": cutlass.__file__,
    "vllm": metadata.version("vllm"),
    "vllm_file": vllm.__file__,
    "ple_method": type(method).__name__,
    "ple_gate_source": inspect.getsource(_get_ple_embedding_quant_method),
    "ple_exact_gate_source": inspect.getsource(_is_exact_radixark_nvfp4_ple),
    "quant_config_name": config.get_name(),
    "quant_method": config.quant_method,
    "quant_serialized": config.is_checkpoint_nvfp4_serialized,
    "quant_group_size": config.group_size,
    "checkpoint_prefix_excluded": config.is_layer_excluded(excluded_prefix),
    "checkpoint_prefix_selected": checkpoint_method is not None,
    "patch_env_name": ple_module._RADIXARK_NVFP4_PLE_FP8_ENV,
    "patch_config_env_name": ple_module._RADIXARK_NVFP4_CONFIG_SHA256_ENV,
    "patch_config_sha256": ple_module._RADIXARK_NVFP4_CONFIG_SHA256,
    "patch_runtime_prefix": ple_module._RADIXARK_NVFP4_PLE_PREFIX,
    "patch_excluded_prefix": ple_module._RADIXARK_NVFP4_PLE_EXCLUDED_PREFIX,
}, sort_keys=True))
"""
    imported = subprocess.run(
        [sys.executable, "-c", import_script],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    import_info = json.loads(imported.stdout.strip().splitlines()[-1])
    if import_info.get("allocator_conf") != "expandable_segments:False":
        raise RuntimeError(f"Pinned allocator mode was not active before torch import: {import_info}")
    if import_info.get("allocator_alias") is not None:
        raise RuntimeError(f"Conflicting allocator alias reached torch import: {import_info}")
    if import_info.get("cuda_compatibility") != "0":
        raise RuntimeError(f"CUDA compatibility mode was not disabled before import: {import_info}")
    if import_info.get("torch_cuda_arch_list") != "12.0":
        raise RuntimeError(f"Wrong CUDA architecture list reached torch import: {import_info}")
    if import_info.get("vllm") != VLLM_VERSION:
        raise RuntimeError(f"Wrong vLLM version: {import_info}")
    if import_info.get("torch") != "2.13.0+cu130" or import_info.get("torch_cuda") != "13.0":
        raise RuntimeError(f"Wrong pinned PyTorch/CUDA build: {import_info}")
    if import_info.get("installed_nccl_dist") != "2.30.7":
        raise RuntimeError(f"Wrong installed nvidia-nccl-cu13 distribution: {import_info}")
    if import_info.get("torch_nccl") != [2, 29, 7]:
        raise RuntimeError(f"PyTorch's pinned NCCL build version changed: {import_info}")
    nccl_requirements = import_info.get("torch_nccl_requirements") or []
    if len(nccl_requirements) != 1 or not str(nccl_requirements[0]).startswith(
        "nvidia-nccl-cu13==2.29.7;"
    ):
        raise RuntimeError(f"PyTorch's pinned NCCL dependency record changed: {import_info}")
    if import_info.get("cutlass") != CUTLASS_DSL_VERSION:
        raise RuntimeError(f"Wrong nvidia-cutlass-dsl version: {import_info}")
    if not str(import_info.get("cutlass_file", "")).startswith(str(cutlass_nested)):
        raise RuntimeError(f"cutlass did not import from its pinned nested path: {import_info}")
    if not str(import_info.get("vllm_file", "")).startswith(str(site)):
        raise RuntimeError(f"vLLM did not import from the pinned runtime: {import_info}")
    if import_info.get("ple_method") != "Qwen3_8FlashNextPLEFp8EmbeddingMethod":
        raise RuntimeError(f"The exact NVFP4-to-FP8 PLE gate did not select its loader: {import_info}")
    gate_source = str(import_info.pop("ple_gate_source", ""))
    exact_gate_source = str(import_info.pop("ple_exact_gate_source", ""))
    expected_quant_gate = {
        "quant_config_name": "modelopt_fp4",
        "quant_method": "NVFP4",
        "quant_serialized": True,
        "quant_group_size": 16,
        "checkpoint_prefix_excluded": True,
        "checkpoint_prefix_selected": False,
    }
    if any(import_info.get(key) != value for key, value in expected_quant_gate.items()):
        raise RuntimeError(f"ModelOpt NVFP4 internals did not pass the exact PLE gate: {import_info}")
    expected_patch_constants = {
        "patch_env_name": "VLLM_RADIXARK_QWEN38_NVFP4_PLE_FP8",
        "patch_config_env_name": "VLLM_RADIXARK_QWEN38_NVFP4_CONFIG_SHA256",
        "patch_config_sha256": MODEL_CONFIG_SHA256,
        "patch_runtime_prefix": PLE_EXACT_PREFIX,
        "patch_excluded_prefix": PLE_EXCLUDED_PREFIX,
    }
    if any(import_info.get(key) != value for key, value in expected_patch_constants.items()):
        raise RuntimeError(f"Patched PLE constants did not match their pins: {import_info}")
    for marker in ("get_name", "group_size", "is_layer_excluded", "prefix !="):
        if marker not in exact_gate_source:
            raise RuntimeError(f"Patched PLE function is missing gate marker {marker!r}.")
    return env, {
        "validation_mode": "full",
        "deep_preload_validation": True,
        "allocator_input": allocator_input,
        "allocator_effective": "expandable_segments:False",
        "allocator_alias_removed": True,
        "cuda_compatibility": "0",
        "torch_cuda_arch_list": "12.0",
        "site_packages": str(site),
        "cutlass_pth": str(cutlass_pth),
        "cutlass_pth_sha256": sha256_file(cutlass_pth),
        "cutlass_nested_path": str(cutlass_nested),
        "cuda_home": str(cuda_home),
        "cuda_library_dir": str(cuda_lib),
        "real_cuda_driver": str(real_driver),
        "real_cuda_driver_sha256": sha256_file(real_driver),
        "nvcc": nvcc_result.stdout.strip(),
        "imports": import_info,
        "nccl_dependency_mismatch": {
            "torch_metadata_requires": nccl_requirements[0],
            "installed_distribution": import_info["installed_nccl_dist"],
            "loaded_runtime": import_info["torch_nccl"],
            "tp1_no_nccl_expected": True,
        },
        "jit_toolchain": jit_toolchain,
        "ple_patch_sha256": sha256_file(ple_path),
        "ple_gate_source_sha256": hashlib.sha256(gate_source.encode("utf-8")).hexdigest(),
        "ple_exact_gate_source_sha256": hashlib.sha256(
            exact_gate_source.encode("utf-8")
        ).hexdigest(),
    }


def _pidfd_probe_once(env: dict[str, str]) -> dict[str, Any]:
    holder_script = """
import json, os, sys
fd = os.open('/dev/null', os.O_RDONLY)
print(json.dumps({'pid': os.getpid(), 'fd': fd}), flush=True)
sys.stdin.buffer.read(1)
os.close(fd)
"""
    reader_script = """
import ctypes, json, os, sys
pid, target_fd = map(int, sys.argv[1:3])
result = {'pid': pid, 'target_fd': target_fd}
pidfd = os.pidfd_open(pid, 0)
try:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    copied = int(libc.syscall(438, pidfd, target_fd, 0))
    if copied < 0:
        error = ctypes.get_errno()
        result.update({'ok': False, 'errno': error, 'error': os.strerror(error)})
    else:
        os.close(copied)
        result.update({'ok': True, 'copied_fd': copied})
finally:
    os.close(pidfd)
print(json.dumps(result, sort_keys=True))
"""
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_script],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        readable, _, _ = select.select([holder.stdout], [], [], 10)
        if not readable:
            raise TimeoutError("pidfd diagnostic holder produced no identity within 10s")
        line = holder.stdout.readline()
        if not line:
            stderr = holder.stderr.read() if holder.stderr is not None else ""
            raise RuntimeError(f"pidfd holder failed before identity: {stderr}")
        target = json.loads(line)
        reader = subprocess.run(
            [sys.executable, "-c", reader_script, str(target["pid"]), str(target["fd"])],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if reader.returncode != 0:
            return {
                "ok": False,
                "reader_returncode": reader.returncode,
                "reader_stdout": reader.stdout,
                "reader_stderr": reader.stderr,
            }
        value = json.loads(reader.stdout.strip().splitlines()[-1])
        value["holder_pid"] = int(target["pid"])
        return value
    finally:
        if holder.stdin is not None:
            try:
                holder.stdin.write("x")
                holder.stdin.flush()
                holder.stdin.close()
            except OSError:
                pass
        try:
            holder.wait(timeout=10)
        except subprocess.TimeoutExpired:
            holder.kill()
            holder.wait(timeout=10)


def diagnose_pidfd_ipc(env: dict[str, str]) -> dict[str, Any]:
    """Record sibling pidfd_getfd behavior without changing process permissions.

    The pinned PyTorch allocator uses classic CUDA IPC because expandable
    segments are disabled. This CPU probe is diagnostic only.
    """
    if not hasattr(os, "pidfd_open"):
        return {"diagnostic_only": True, "supported": False, "result": None}
    try:
        result = _pidfd_probe_once(env)
    except Exception as exc:
        result = {
            "ok": False,
            "diagnostic_exception_type": type(exc).__name__,
            "diagnostic_exception": str(exc),
        }
    return {
        "diagnostic_only": True,
        "supported": True,
        "allocator": "classic_cuda_ipc",
        "permissions_changed": False,
        "result": result,
    }


def validate_cli_contract(env: dict[str, str]) -> dict[str, Any]:
    full_help_option = "--help=all"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            full_help_option,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    help_text = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"Pinned vLLM CLI help failed: {help_text[-4000:]}")
    required_flags = [
        "--distributed-executor-backend",
        "--gpu-memory-utilization",
        "--kv-cache-memory-bytes",
        "--kv-cache-dtype",
        "--enable-prefix-caching",
        "--no-enable-prefix-caching",
        "--max-model-len",
        "--max-num-seqs",
        "--max-num-batched-tokens",
        "--async-scheduling",
        "--enable-chunked-prefill",
        "--no-enable-chunked-prefill",
        "--speculative-config",
        "--no-enable-log-requests",
        "--disable-uvicorn-access-log",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "--reasoning-parser",
        "--max-cudagraph-capture-size",
    ]
    missing = [flag for flag in required_flags if flag not in help_text]
    if missing:
        raise RuntimeError(f"Pinned vLLM CLI is missing required flags: {missing}")
    return {
        "module": "vllm.entrypoints.cli.main",
        "subcommand": "serve",
        "help_option": full_help_option,
        "required_flags": required_flags,
        "help_sha256": hashlib.sha256(help_text.encode("utf-8")).hexdigest(),
    }


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        match = re.search(r"(\d+)", raw)
        if match:
            values[key] = int(match.group(1)) * 1024
    return values


def gpu_inventory() -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or "rtx pro 6000" not in rows[0].lower():
        raise RuntimeError(f"Expected one RTX PRO 6000, found: {rows}")
    meminfo = _meminfo()
    available = int(meminfo.get("MemAvailable", 0))
    if available < MIN_HOST_AVAILABLE_BYTES:
        raise RuntimeError(
            f"Host memory is too small for FP8 PLE CPU offload: {available} < {MIN_HOST_AVAILABLE_BYTES}"
        )
    shm = os.statvfs("/dev/shm")
    shm_free = int(shm.f_bavail * shm.f_frsize)
    probe = Path("/dev/shm") / f"duck-vllm-ipc-{os.getpid()}"
    try:
        probe.write_bytes(b"duck-vllm-ipc")
        if probe.read_bytes() != b"duck-vllm-ipc":
            raise RuntimeError("/dev/shm readback did not match.")
    finally:
        probe.unlink(missing_ok=True)
    disk = subprocess.run(
        ["df", "-B1", str(WORKING_DIR), "/tmp", "/dev/shm"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return {
        "gpu_rows": rows,
        "meminfo_bytes": meminfo,
        "host_mem_available_bytes": available,
        "shm_free_bytes": shm_free,
        "disk": disk.stdout,
    }


def proc_record(pid: int) -> dict[str, Any] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        close = raw.rfind(")")
        fields = raw[close + 2 :].split()
        cmdline_bytes = Path(f"/proc/{pid}/cmdline").read_bytes()
        argv = [
            part.decode("utf-8", errors="replace")
            for part in cmdline_bytes.split(b"\x00")
            if part
        ]
        cmdline = cmdline_bytes.replace(b"\x00", b" ").decode(
            "utf-8", errors="replace"
        ).strip()
        return {
            "pid": pid,
            "state": fields[0],
            "ppid": int(fields[1]),
            "pgid": int(fields[2]),
            "sid": int(fields[3]),
            "start_ticks": int(fields[19]),
            "cmdline": cmdline,
            "cmdline_sha256": hashlib.sha256(cmdline_bytes).hexdigest(),
            "argv_sha256": hashlib.sha256(
                json.dumps(argv, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
        }
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def process_table() -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for item in Path("/proc").iterdir():
        if item.name.isdigit():
            record = proc_record(int(item.name))
            if record is not None:
                records[int(item.name)] = record
    return records


def descendant_ids(root_pid: int, records: dict[int, dict[str, Any]]) -> set[int]:
    found = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, record in records.items():
            if record["ppid"] in found and pid not in found:
                found.add(pid)
                changed = True
    return found


def allowed_detached_worker(
    record: dict[str, Any],
    workers_by_pid: dict[int, dict[str, Any]],
    root_pgid: int,
    root_sid: int,
) -> bool:
    pid = int(record.get("pid", -1))
    parent = workers_by_pid.get(int(record.get("ppid", -1)))
    return (
        pid > 1
        and parent is not None
        and int(parent.get("pgid", -1)) == root_pgid
        and int(parent.get("sid", -1)) == root_sid
        and int(record.get("pgid", -1)) == pid
        and int(record.get("sid", -1)) == pid
        and record.get("cmdline_sha256") == TORCH_SHM_MANAGER_CMDLINE_SHA256
    )


def start_ticks(pid: int) -> int:
    record = proc_record(pid)
    if record is None:
        raise ProcessLookupError(pid)
    return int(record["start_ticks"])


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()


def tail_log(lines: int = 200) -> str:
    if not SERVER_LOG.exists():
        return ""
    return "\n".join(
        SERVER_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def alive(pid: int, ticks: int | None = None) -> bool:
    record = proc_record(pid)
    return (
        record is not None
        and record["state"] != "Z"
        and (ticks is None or int(record["start_ticks"]) == ticks)
    )


def stop_owned_server() -> dict[str, Any]:
    result: dict[str, Any] = {"identity_exists": SERVER_IDENTITY.exists(), "stopped": True}
    if not SERVER_IDENTITY.exists():
        return result
    identity = read_json(SERVER_IDENTITY)
    if not isinstance(identity, dict):
        result.update(
            {
                "identity_matches": False,
                "identity_rejection_reasons": ["identity is not a JSON object"],
                "survivors": None,
                "port_open": port_open(),
                "stopped": False,
            }
        )
        return result
    identity_keys = {
        "schema_version",
        "backend",
        "boot_id",
        "host",
        "port",
        "pid",
        "start_ticks",
        "pgid",
        "sid",
        "argv",
        "argv_sha256",
        "vllm_tuning",
        "started_epoch",
        "phase",
        "snapshot_epoch",
        "ready_epoch",
        "workers",
    }
    worker_keys = {"pid", "start_ticks", "ppid", "pgid", "sid", "cmdline_sha256"}
    reasons: list[str] = []
    if set(identity) != identity_keys:
        reasons.append("identity key set mismatch")
    reasons.extend(validate_vllm_tuning_document(identity.get("vllm_tuning")))
    try:
        pid = int(identity["pid"])
        ticks = int(identity["start_ticks"])
        pgid = int(identity["pgid"])
        sid = int(identity["sid"])
    except (KeyError, TypeError, ValueError):
        pid = ticks = pgid = sid = -1
        reasons.append("invalid root numeric identity")
    argv = identity.get("argv")
    expected_argv_sha = (
        hashlib.sha256(
            json.dumps(argv, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if isinstance(argv, list) and argv and all(isinstance(item, str) for item in argv)
        else None
    )
    if expected_argv_sha is None or identity.get("argv_sha256") != expected_argv_sha:
        reasons.append("canonical argv hash mismatch")
    if (
        identity.get("schema_version") != 1
        or identity.get("backend") != "vllm"
        or identity.get("boot_id") != boot_id()
        or identity.get("host") != HOST
        or identity.get("port") != PORT
        or pid <= 1
        or pgid != pid
        or sid != pid
    ):
        reasons.append("root ownership header mismatch")
    phase = identity.get("phase")
    workers = identity.get("workers")
    started_epoch = identity.get("started_epoch")
    snapshot_epoch = identity.get("snapshot_epoch")
    ready_epoch = identity.get("ready_epoch")
    if (
        phase not in {"starting", "ready"}
        or not isinstance(started_epoch, (int, float))
        or not isinstance(snapshot_epoch, (int, float))
        or snapshot_epoch < started_epoch
        or not isinstance(workers, list)
        or not workers
    ):
        reasons.append("invalid phase or snapshot semantics")
    elif phase == "starting" and (ready_epoch is not None or len(workers) != 1):
        reasons.append("invalid starting-phase snapshot")
    elif phase == "ready" and (
        not isinstance(ready_epoch, (int, float))
        or ready_epoch < started_epoch
        or ready_epoch > snapshot_epoch
    ):
        reasons.append("invalid ready-phase snapshot")
    if isinstance(workers, list):
        workers_by_pid = {
            int(worker["pid"]): worker
            for worker in workers
            if isinstance(worker, dict) and type(worker.get("pid")) is int
        }
        for worker in workers:
            if not isinstance(worker, dict) or set(worker) != worker_keys:
                reasons.append("worker key set mismatch")
                break
            if int(worker.get("sid", -1)) != sid and not allowed_detached_worker(
                worker, workers_by_pid, pgid, sid
            ):
                reasons.append(
                    f"unrecognized detached worker for pid {worker.get('pid')}"
                )
                break
        root_workers = [
            worker
            for worker in workers
            if isinstance(worker, dict) and worker.get("pid") == pid
        ]
        if (
            len(root_workers) != 1
            or int(root_workers[0].get("start_ticks", -1)) != ticks
            or int(root_workers[0].get("pgid", -1)) != pgid
            or int(root_workers[0].get("sid", -1)) != sid
        ):
            reasons.append("root worker snapshot mismatch")

    current_root = proc_record(pid) if pid > 1 else None
    if current_root is not None and (
        int(current_root["start_ticks"]) != ticks
        or int(current_root["pgid"]) != pgid
        or int(current_root["sid"]) != sid
        or current_root["argv_sha256"] != identity.get("argv_sha256")
    ):
        reasons.append("live root fingerprint mismatch")
    if current_root is not None and isinstance(workers, list):
        root_worker = next(
            (worker for worker in workers if isinstance(worker, dict) and worker.get("pid") == pid),
            None,
        )
        if root_worker is None or current_root["cmdline_sha256"] != root_worker.get(
            "cmdline_sha256"
        ):
            reasons.append("live root cmdline fingerprint mismatch")
    table = process_table()
    group_records = [record for record in table.values() if int(record["pgid"]) == pgid]
    if current_root is not None and (
        not group_records or any(int(record["sid"]) != sid for record in group_records)
    ):
        reasons.append("live process group is not wholly inside the owned session")
    if isinstance(workers, list):
        for worker in workers:
            if not isinstance(worker, dict):
                continue
            live = table.get(int(worker.get("pid", -1)))
            if live is None:
                continue
            if (
                int(live["start_ticks"]) != int(worker.get("start_ticks", -1))
                or int(live["pgid"]) != int(worker.get("pgid", -1))
                or int(live["sid"]) != int(worker.get("sid", -1))
                or live["cmdline_sha256"] != worker.get("cmdline_sha256")
            ):
                reasons.append(f"live worker fingerprint mismatch for pid {worker.get('pid')}")
                break

    matches = current_root is not None and not reasons
    result.update(
        {
            "pid": pid,
            "start_ticks": ticks,
            "pgid": pgid,
            "sid": sid,
            "identity_matches": matches,
            "identity_rejection_reasons": reasons,
        }
    )

    def live_owned_records() -> list[dict[str, Any]]:
        if sid <= 1:
            return []
        table = process_table()
        owned = {
            int(record["pid"]): record
            for record in table.values()
            if int(record["sid"]) == sid and record["state"] != "Z"
        }
        if isinstance(workers, list):
            workers_by_pid = {
                int(worker["pid"]): worker
                for worker in workers
                if isinstance(worker, dict) and type(worker.get("pid")) is int
            }
            for worker in workers:
                if (
                    not isinstance(worker, dict)
                    or int(worker.get("sid", -1)) == sid
                    or not allowed_detached_worker(
                        worker, workers_by_pid, pgid, sid
                    )
                ):
                    continue
                live = table.get(int(worker["pid"]))
                if live is not None and (
                    int(live["start_ticks"]) == int(worker["start_ticks"])
                    and int(live["pgid"]) == int(worker["pgid"])
                    and int(live["sid"]) == int(worker["sid"])
                    and live["cmdline_sha256"] == worker["cmdline_sha256"]
                ):
                    owned[int(live["pid"])] = live
        return list(owned.values())

    if matches:
        try:
            os.killpg(pgid, signal.SIGTERM)
            result["sigterm_sent"] = True
        except ProcessLookupError:
            pass
        separately_signaled: list[int] = []
        for record in live_owned_records():
            if int(record["pgid"]) == pgid:
                continue
            try:
                os.kill(int(record["pid"]), signal.SIGTERM)
                separately_signaled.append(int(record["pid"]))
            except ProcessLookupError:
                pass
        if separately_signaled:
            result["separate_sigterm_pids"] = separately_signaled
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and live_owned_records():
            time.sleep(1)
        survivors = live_owned_records()
        if survivors:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            separately_killed: list[int] = []
            for record in survivors:
                if int(record["pgid"]) == pgid:
                    continue
                current = proc_record(int(record["pid"]))
                if (
                    current is None
                    or int(current["start_ticks"]) != int(record["start_ticks"])
                    or int(current["pgid"]) != int(record["pgid"])
                    or int(current["sid"]) != int(record["sid"])
                    or current["cmdline_sha256"] != record["cmdline_sha256"]
                ):
                    continue
                try:
                    os.kill(int(record["pid"]), signal.SIGKILL)
                    separately_killed.append(int(record["pid"]))
                except ProcessLookupError:
                    pass
            result["sigkill_sent"] = True
            if separately_killed:
                result["separate_sigkill_pids"] = separately_killed
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and live_owned_records():
                time.sleep(0.5)
    survivors = live_owned_records()
    result["survivors"] = [
        {
            "pid": int(record["pid"]),
            "start_ticks": int(record["start_ticks"]),
            "ppid": int(record["ppid"]),
            "pgid": int(record["pgid"]),
            "sid": int(record["sid"]),
            "cmdline_sha256": str(record["cmdline_sha256"]),
        }
        for record in sorted(survivors, key=lambda item: int(item["pid"]))
    ]
    result["port_open"] = port_open()
    result["stopped"] = not survivors and not result["port_open"]
    return result


def port_open(timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=timeout):
            return True
    except OSError:
        return False


def request_bytes(url: str, payload: dict[str, Any] | None = None, timeout: int = 60) -> bytes:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {error_body[:4000]}") from exc


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
    value = json.loads(request_bytes(url, payload=payload, timeout=timeout).decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object from {url}, found {type(value).__name__}.")
    return value


def server_command(
    model_dir: Path,
    *,
    enable_chunked_prefill: bool = True,
    tuning: dict[str, Any] | None = None,
) -> list[str]:
    if not isinstance(enable_chunked_prefill, bool):
        raise TypeError("enable_chunked_prefill must be a bool.")
    resolved_tuning = tuning or resolve_vllm_tuning(
        enable_chunked_prefill=enable_chunked_prefill
    )
    if resolved_tuning.get("enable_chunked_prefill") is not enable_chunked_prefill:
        raise RuntimeError("Resolved vLLM tuning has the wrong chunked-prefill mode.")
    chunked_prefill_flag = (
        "--enable-chunked-prefill"
        if enable_chunked_prefill
        else "--no-enable-chunked-prefill"
    )
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model_dir),
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--load-format",
        "safetensors",
        "--dtype",
        "bfloat16",
        "--quantization",
        "modelopt_fp4",
        "--tensor-parallel-size",
        "1",
        "--distributed-executor-backend",
        "mp",
    ]
    if resolved_tuning["moe_backend"] is not None:
        command.extend(["--moe-backend", str(resolved_tuning["moe_backend"])])
    if int(resolved_tuning["kv_cache_memory_bytes"]) > 0:
        command.extend(
            [
                "--kv-cache-memory-bytes",
                str(resolved_tuning["kv_cache_memory_bytes"]),
            ]
        )
    else:
        command.extend(
            ["--gpu-memory-utilization", GPU_MEMORY_UTILIZATION]
        )
    command.extend(
        [
            "--max-model-len",
            str(ANALYZER_CONTEXT),
            "--max-num-seqs",
            str(resolved_tuning["max_num_seqs"]),
            "--max-num-batched-tokens",
            str(resolved_tuning["max_num_batched_tokens"]),
            "--async-scheduling",
            chunked_prefill_flag,
        ]
    )
    if resolved_tuning["kv_cache_dtype"] != "auto":
        command.extend(
            ["--kv-cache-dtype", str(resolved_tuning["kv_cache_dtype"])]
        )
    if int(resolved_tuning["max_cudagraph_capture_size"]) > 0:
        command.extend(
            [
                "--max-cudagraph-capture-size",
                str(resolved_tuning["max_cudagraph_capture_size"]),
            ]
        )
    command.extend(
        [
            (
                "--enable-prefix-caching"
                if resolved_tuning["enable_prefix_caching"]
                else "--no-enable-prefix-caching"
            ),
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
            "--reasoning-parser",
            "qwen3",
            "--chat-template",
            str(model_dir / "chat_template.jinja"),
        ]
    )
    if int(resolved_tuning["mtp_speculative_tokens"]) > 0:
        speculative_config: dict[str, Any] = {
            "method": "mtp",
            "num_speculative_tokens": resolved_tuning[
                "mtp_speculative_tokens"
            ],
        }
        if resolved_tuning["mtp_index_share_for_iteration"]:
            speculative_config["index_share_for_mtp_iteration"] = True
        if resolved_tuning["mtp_dynamic_batch_schedule"] is not None:
            speculative_config["num_speculative_tokens_per_batch_size"] = (
                resolved_tuning["mtp_dynamic_batch_schedule"]
            )
        command.extend(
            [
                "--speculative-config",
                json.dumps(
                    speculative_config,
                    separators=(",", ":"),
                ),
            ]
        )
    command.extend(
        [
            "--no-enable-log-requests",
            "--disable-uvicorn-access-log",
            "--uvicorn-log-level",
            "info",
        ]
    )
    return command


def start_server(
    model_dir: Path,
    env: dict[str, str],
    *,
    enable_chunked_prefill: bool = True,
    tuning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cleanup = stop_owned_server()
    if not cleanup.get("stopped", False):
        raise RuntimeError(f"A prior owned vLLM server did not stop: {cleanup}")
    if port_open():
        raise RuntimeError(f"Port {HOST}:{PORT} is already in use by an unowned process.")
    SERVER_LOG.unlink(missing_ok=True)
    resolved_tuning = tuning or resolve_vllm_tuning(
        enable_chunked_prefill=enable_chunked_prefill
    )
    command = server_command(
        model_dir,
        enable_chunked_prefill=enable_chunked_prefill,
        tuning=resolved_tuning,
    )
    print("VLLM_START_COMMAND", json.dumps(command), flush=True)
    log_handle = SERVER_LOG.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    deadline = time.monotonic() + 10
    root_record = None
    while time.monotonic() < deadline:
        root_record = proc_record(process.pid)
        if (
            root_record is not None
            and root_record["pgid"] == process.pid
            and root_record["sid"] == process.pid
        ):
            break
        time.sleep(0.05)
    if root_record is None or root_record["pgid"] != process.pid or root_record["sid"] != process.pid:
        process.terminate()
        raise RuntimeError(
            f"vLLM root did not enter its isolated process session: {root_record}"
        )
    argv_sha256 = hashlib.sha256(
        json.dumps(command, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if root_record["argv_sha256"] != argv_sha256:
        process.terminate()
        raise RuntimeError(
            "The live vLLM root argv did not match the exact launched command: "
            f"{root_record['argv_sha256']} != {argv_sha256}"
        )
    started_epoch = time.time()
    root_worker = {
        "pid": int(root_record["pid"]),
        "start_ticks": int(root_record["start_ticks"]),
        "ppid": int(root_record["ppid"]),
        "pgid": int(root_record["pgid"]),
        "sid": int(root_record["sid"]),
        "cmdline_sha256": str(root_record["cmdline_sha256"]),
    }
    identity = {
        "schema_version": 1,
        "backend": "vllm",
        "boot_id": boot_id(),
        "host": HOST,
        "port": PORT,
        "pid": process.pid,
        "start_ticks": int(root_record["start_ticks"]),
        "pgid": int(root_record["pgid"]),
        "sid": int(root_record["sid"]),
        "argv": command,
        "argv_sha256": argv_sha256,
        "vllm_tuning": resolved_tuning,
        "started_epoch": started_epoch,
        "phase": "starting",
        "snapshot_epoch": started_epoch,
        "ready_epoch": None,
        "workers": [root_worker],
    }
    write_json(SERVER_IDENTITY, identity)
    return identity


def update_ready_identity(identity: dict[str, Any]) -> dict[str, Any]:
    if identity.get("boot_id") != boot_id():
        raise RuntimeError("vLLM identity boot ID changed before worker snapshot.")
    root = proc_record(int(identity["pid"]))
    if (
        root is None
        or int(root["start_ticks"]) != int(identity["start_ticks"])
        or int(root["pgid"]) != int(identity["pgid"])
        or int(root["sid"]) != int(identity["sid"])
    ):
        raise RuntimeError(f"vLLM root identity changed before worker snapshot: {root}")
    table = process_table()
    descendants = descendant_ids(int(identity["pid"]), table)
    detached = [
        table[pid]
        for pid in sorted(descendants)
        if pid in table
        and int(table[pid]["sid"]) != int(identity["sid"])
        and allowed_detached_worker(
            table[pid], table, int(identity["pgid"]), int(identity["sid"])
        )
    ]
    escaped = [
        table[pid]
        for pid in sorted(descendants)
        if pid in table
        and int(table[pid]["sid"]) != int(identity["sid"])
        and not allowed_detached_worker(
            table[pid], table, int(identity["pgid"]), int(identity["sid"])
        )
    ]
    if escaped:
        raise RuntimeError(f"Owned vLLM descendants escaped the isolated session: {escaped}")
    owned_by_pid = {
        int(record["pid"]): record
        for record in table.values()
        if int(record["sid"]) == int(identity["sid"])
    }
    owned_by_pid.update({int(record["pid"]): record for record in detached})
    owned = list(owned_by_pid.values())
    if not any(int(record["pid"]) == int(identity["pid"]) for record in owned):
        raise RuntimeError("The vLLM root was absent from its post-readiness session snapshot.")
    workers = [
        {
            "pid": int(record["pid"]),
            "start_ticks": int(record["start_ticks"]),
            "ppid": int(record["ppid"]),
            "pgid": int(record["pgid"]),
            "sid": int(record["sid"]),
            "cmdline_sha256": str(record["cmdline_sha256"]),
        }
        for record in sorted(owned, key=lambda item: int(item["pid"]))
    ]
    updated = dict(identity)
    now = time.time()
    updated["phase"] = "ready"
    updated["snapshot_epoch"] = now
    updated["ready_epoch"] = identity.get("ready_epoch") or now
    updated["workers"] = workers
    write_json(SERVER_IDENTITY, updated)
    return updated


def _bounded_timeout(setup_deadline: float, requested: int) -> int:
    remaining = setup_deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"The {SETUP_TOTAL_TIMEOUT}s setup/preflight ceiling expired.")
    return max(1, min(requested, math.ceil(remaining)))


def wait_for_server(identity: dict[str, Any], setup_deadline: float) -> dict[str, Any]:
    deadline = min(time.monotonic() + SERVER_READY_TIMEOUT, setup_deadline)
    started = time.monotonic()
    last_notice = -1
    while time.monotonic() < deadline:
        if not alive(int(identity["pid"]), int(identity["start_ticks"])):
            raise RuntimeError(f"vLLM exited before readiness.\n{tail_log(300)}")
        try:
            models = request_json(
                f"{BASE_URL}/models", timeout=_bounded_timeout(setup_deadline, 10)
            )
            rows = models.get("data") or []
            ids = [row.get("id") for row in rows if isinstance(row, dict)]
            if ids != [SERVED_MODEL_NAME]:
                raise RuntimeError(f"vLLM served the wrong model identity: {ids}")
            return {"models": models, "ready_seconds": time.monotonic() - started}
        except Exception:
            elapsed = int(time.monotonic() - started)
            if elapsed // 60 != last_notice:
                last_notice = elapsed // 60
                print(f"VLLM_WAIT elapsed_s={elapsed}\n{tail_log(30)}", flush=True)
            time.sleep(5)
    raise TimeoutError(f"Timed out waiting for vLLM after {SERVER_READY_TIMEOUT}s.\n{tail_log(300)}")


def message_text(response: dict[str, Any]) -> str:
    message = response["choices"][0]["message"]
    return "\n".join(
        str(value or "")
        for value in (
            message.get("reasoning_content"),
            message.get("reasoning"),
            message.get("content"),
        )
    ).strip()


def completion_tokens(response: dict[str, Any]) -> int:
    return int((response.get("usage") or {}).get("completion_tokens") or 0)


def chat(
    payload: dict[str, Any], setup_deadline: float, timeout: int = 900
) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    response = request_json(
        f"{BASE_URL}/chat/completions",
        payload=payload,
        timeout=_bounded_timeout(setup_deadline, timeout),
    )
    return response, time.monotonic() - started


def base_payload(prompt: Any, max_tokens: int, enable_thinking: bool) -> dict[str, Any]:
    return {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": 0,
        "chat_template_kwargs": {
            "enable_thinking": enable_thinking,
            "preserve_thinking": True,
            "reasoning_effort": "xhigh",
        },
    }


def red_blue_png_data_url() -> str:
    width = height = 8
    rows = []
    for _ in range(height):
        rows.append(b"\x00" + b"\xff\x00\x00" * 4 + b"\x00\x00\xff" * 4)
    raw = b"".join(rows)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _parse_metric_sum(metrics: str, metric: str) -> float:
    total = 0.0
    found = False
    pattern = re.compile(rf"^{re.escape(metric)}(?:\{{[^}}]*\}})?\s+([^\s]+)\s*$")
    for line in metrics.splitlines():
        match = pattern.match(line.strip())
        if match:
            value = float(match.group(1))
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite metric value for {metric}: {value}")
            total += value
            found = True
    if not found:
        raise RuntimeError(f"Required vLLM metric was absent: {metric}")
    return total


def run_preflights(setup_deadline: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metric_names = (
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
    )
    baseline_metrics = request_bytes(
        f"http://{HOST}:{PORT}/metrics",
        timeout=_bounded_timeout(setup_deadline, 30),
    ).decode("utf-8", errors="replace")
    baseline_counters = {
        metric: _parse_metric_sum(baseline_metrics, metric) for metric in metric_names
    }
    text_response, text_seconds = chat(
        base_payload("Answer with only: 4", 32, False), setup_deadline
    )
    text_content = str(text_response["choices"][0]["message"].get("content") or "").strip()
    if completion_tokens(text_response) <= 0 or text_content != "4":
        raise RuntimeError(f"Text preflight was not exact: {text_response}")
    result["text"] = {
        "seconds": text_seconds,
        "completion_tokens": completion_tokens(text_response),
        "text": text_content,
    }

    tool_payload = base_payload("Call submit_number with value 4 now. Do not answer in prose.", 128, False)
    tool_payload["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "submit_number",
                "description": "Submit one integer.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
            },
        }
    ]
    tool_payload["tool_choice"] = "required"
    tool_response, tool_seconds = chat(tool_payload, setup_deadline)
    tool_message = tool_response["choices"][0]["message"]
    calls = tool_message.get("tool_calls") or []
    if not calls:
        raise RuntimeError(f"Tool preflight produced no native structured call: {tool_response}")
    function = calls[0].get("function") or {}
    raw_arguments = function.get("arguments") or {}
    arguments = raw_arguments if isinstance(raw_arguments, dict) else json.loads(str(raw_arguments))
    if function.get("name") != "submit_number" or int(arguments.get("value", -1)) != 4:
        raise RuntimeError(f"Wrong native tool preflight call: {calls}")
    if tool_response["choices"][0].get("finish_reason") != "tool_calls":
        raise RuntimeError(f"Tool preflight had the wrong finish reason: {tool_response}")
    result["tool"] = {
        "seconds": tool_seconds,
        "completion_tokens": completion_tokens(tool_response),
        "structured": True,
        "calls": calls,
        "finish_reason": "tool_calls",
    }

    vision_prompt = [
        {"type": "text", "text": "Name one color visible in this image. Answer with one word."},
        {"type": "image_url", "image_url": {"url": red_blue_png_data_url()}},
    ]
    vision_response, vision_seconds = chat(
        base_payload(vision_prompt, 32, False), setup_deadline
    )
    vision_text = str(vision_response["choices"][0]["message"].get("content") or "").strip()
    if completion_tokens(vision_response) <= 0 or not any(
        color in vision_text.lower() for color in ("red", "blue")
    ):
        raise RuntimeError(f"Vision preflight did not identify the known image: {vision_response}")
    result["vision"] = {
        "seconds": vision_seconds,
        "completion_tokens": completion_tokens(vision_response),
        "text": vision_text[:500],
        "image": "generated 8x8 half-red half-blue PNG data URL",
    }

    warm_response, warm_seconds = chat(
        base_payload("Print the word velocity followed by a space repeatedly until stopped.", 512, False),
        setup_deadline,
    )
    warm_tokens = completion_tokens(warm_response)
    if warm_tokens < 128:
        raise RuntimeError(f"C1 throughput preflight ended too early: {warm_tokens} tokens")
    c1_rate = warm_tokens / max(warm_seconds, 1e-6)
    result["c1_throughput"] = {
        "seconds": warm_seconds,
        "completion_tokens": warm_tokens,
        "completion_tokens_per_second": c1_rate,
        "finish_reason": warm_response["choices"][0].get("finish_reason"),
    }

    def one_concurrent(index: int) -> dict[str, Any]:
        response, seconds = chat(
            base_payload(
                f"Stream {index}: print the word arc followed by a space repeatedly until stopped.",
                256,
                False,
            ),
            setup_deadline,
        )
        tokens = completion_tokens(response)
        if tokens < 64:
            raise RuntimeError(f"C4 preflight {index} ended too early: {tokens}")
        return {
            "index": index,
            "seconds": seconds,
            "completion_tokens": tokens,
            "completion_tokens_per_second": tokens / max(seconds, 1e-6),
        }

    c4_started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        c4_rows = list(executor.map(one_concurrent, range(4)))
    c4_seconds = time.monotonic() - c4_started
    c4_tokens = sum(row["completion_tokens"] for row in c4_rows)
    c4_rate = c4_tokens / max(c4_seconds, 1e-6)
    if c4_rate < 1.5 * c1_rate:
        raise RuntimeError(f"C4 aggregate rate did not reach 1.5x C1: {c4_rate} < {1.5*c1_rate}")
    weak = [row for row in c4_rows if row["completion_tokens_per_second"] < 0.35 * c1_rate]
    if weak:
        raise RuntimeError(f"One or more C4 lanes fell below 0.35x C1: {weak}")
    result["c4_throughput"] = {
        "seconds": c4_seconds,
        "completion_tokens": c4_tokens,
        "aggregate_completion_tokens_per_second": c4_rate,
        "aggregate_over_c1": c4_rate / c1_rate,
        "requests": c4_rows,
    }

    metrics = request_bytes(
        f"http://{HOST}:{PORT}/metrics",
        timeout=_bounded_timeout(setup_deadline, 30),
    ).decode("utf-8", errors="replace")
    METRICS_PATH.write_text(metrics, encoding="utf-8")
    after_counters = {metric: _parse_metric_sum(metrics, metric) for metric in metric_names}
    deltas = {
        metric: after_counters[metric] - baseline_counters[metric] for metric in metric_names
    }
    drafts = deltas["vllm:spec_decode_num_drafts_total"]
    draft_tokens = deltas["vllm:spec_decode_num_draft_tokens_total"]
    accepted_tokens = deltas["vllm:spec_decode_num_accepted_tokens_total"]
    if drafts <= 0 or draft_tokens <= 0 or accepted_tokens <= 0:
        raise RuntimeError(
            "Native MTP telemetry deltas were not positive: "
            f"drafts={drafts} draft_tokens={draft_tokens} accepted={accepted_tokens}"
        )
    result["native_mtp"] = {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted_tokens,
        "acceptance_rate": accepted_tokens / draft_tokens,
        "baseline_counters": baseline_counters,
        "after_counters": after_counters,
        "counter_deltas": deltas,
        "baseline_metrics_sha256": hashlib.sha256(
            baseline_metrics.encode("utf-8")
        ).hexdigest(),
        "metrics_sha256": sha256_file(METRICS_PATH),
    }
    return result


def capture_process_and_gpu(identity: dict[str, Any]) -> dict[str, Any]:
    table = process_table()
    owned_ids = {
        pid for pid, record in table.items() if int(record["sid"]) == int(identity["sid"])
    }
    records = [table[pid] for pid in sorted(owned_ids)]
    if not records:
        raise RuntimeError("The owned vLLM process session disappeared after preflight.")
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    gpu_rows: list[dict[str, Any]] = []
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) == 4 and parts[0].isdigit():
            gpu_rows.append(
                {
                    "pid": int(parts[0]),
                    "process_name": parts[1],
                    "used_memory_mib": int(parts[2].split()[0]),
                    "gpu_uuid": parts[3],
                }
            )
    if len(gpu_rows) != 1 or gpu_rows[0]["pid"] not in owned_ids:
        raise RuntimeError(f"Expected one owned vLLM GPU process, found: {gpu_rows}")
    memory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    memory_parts = [int(part.strip().split()[0]) for part in memory.stdout.strip().split(",")]
    if len(memory_parts) != 3 or memory_parts[2] < MIN_GPU_FREE_MIB:
        raise RuntimeError(f"vLLM left too little free GPU memory: {memory.stdout.strip()}")
    log_text = SERVER_LOG.read_text(encoding="utf-8", errors="replace")
    ple_log_patterns = {
        "found": r"Found\s+\d+\s+PleOffloadLayer\(s\):",
        "gpu_registration": (
            r"GPU worker\s+0\s+registered\s+\(dp_rank=0,\s*tp_rank=0,\s*layers="
        ),
        "ready": r"Worker ready\s+-\s+\d+\s+PleOffloadLayer\(s\):",
        "busy_loop": r"Busy-loop started\.",
    }
    ple_log_matches = {
        name: (match.group(0) if (match := re.search(pattern, log_text)) else None)
        for name, pattern in ple_log_patterns.items()
    }
    if any(value is None for value in ple_log_matches.values()):
        raise RuntimeError(
            f"vLLM did not prove native PLE CPU-offload readiness: {ple_log_matches}"
        )
    value = {
        "server_identity": identity,
        "process_tree": records,
        "gpu_rows": gpu_rows,
        "gpu_memory_mib": {
            "total": memory_parts[0],
            "used": memory_parts[1],
            "free": memory_parts[2],
        },
        "ple_offload_log_matches": ple_log_matches,
        "captured_epoch": time.time(),
    }
    write_json(GPU_PATH, value)
    return value


def persist_analyzer_environment(env: dict[str, str]) -> dict[str, str]:
    setup_env = read_json(SETUP_ENV_PATH) if SETUP_ENV_PATH.exists() else {}
    if not isinstance(setup_env, dict):
        raise RuntimeError("TAAF_KAGGLE_SETUP_ENV must contain a JSON object.")
    runtime_keys = {
        "PYTHONPATH": env["PYTHONPATH"],
        "PATH": env["PATH"],
        "LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"],
        "CUDA_HOME": env["CUDA_HOME"],
        "CUDACXX": env["CUDACXX"],
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_ENABLE_CUDA_COMPATIBILITY": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        "TORCH_CUDA_ARCH_LIST": "12.0",
        "VLLM_RADIXARK_QWEN38_NVFP4_PLE_FP8": "1",
        "VLLM_RADIXARK_QWEN38_NVFP4_CONFIG_SHA256": MODEL_CONFIG_SHA256,
    }
    analyzer = {
        "LOCAL_ANALYZER_BASE_URL": BASE_URL,
        "OPENAI_BASE_URL": BASE_URL,
        "LOCAL_ANALYZER_PROVIDER": "vllm",
        "OPENAI_PROVIDER": "vllm",
        "LOCAL_ANALYZER_MODEL_ID": SERVED_MODEL_NAME,
        "INFERENCE_ANALYZER_MODEL": SERVED_MODEL_NAME,
        "OPENAI_API_KEY": "offline-kaggle-local-server",
        "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness",
        "LOCAL_ANALYZER_CONTEXT_WINDOW": str(ANALYZER_CONTEXT),
        "LOCAL_ANALYZER_MAX_OUTPUT": "0",
        "LOCAL_ANALYZER_TOOL_STEPS": "0",
        "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
        "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
        "LOCAL_ANALYZER_YIELD_SECONDS": "60",
        "LOCAL_ANALYZER_TEMPERATURE": "0.6",
        "LOCAL_ANALYZER_TOP_P": "0.95",
        "LOCAL_ANALYZER_TOP_K": "20",
        "LOCAL_ANALYZER_ENABLE_THINKING": "true",
        "MULTIMODAL_CONTEXT": "current_grid",
        "MULTIMODAL_UPSCALE": "4",
    }
    setup_env.update(runtime_keys)
    setup_env.update(analyzer)
    write_json(SETUP_ENV_PATH, setup_env)
    return {**runtime_keys, **analyzer}


def main() -> None:
    setup_started = time.monotonic()
    setup_started_epoch = time.time()
    setup_deadline = setup_started + SETUP_TOTAL_TIMEOUT
    fast_start = fast_start_enabled()
    tuning = resolve_vllm_tuning()
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    FAILURE_PATH.unlink(missing_ok=True)
    print(
        f"VLLM_SETUP_MODE {'fast' if fast_start else 'full'} "
        f"{FAST_START_ENV}={int(fast_start)}",
        flush=True,
    )
    source_identity()
    inventory = gpu_inventory()
    runtime_dir = resolve_runtime_dir()
    model_dir = resolve_model_dir()
    print(f"PINNED_VLLM_RUNTIME_PATH {runtime_dir}", flush=True)
    print(f"PINNED_MODEL_PATH {model_dir}", flush=True)
    storage_check = validate_runtime_storage()
    runtime_check = verify_and_extract_runtime(
        runtime_dir,
        full_layer_hashes=not fast_start,
        scan_extracted_caches=not fast_start,
    )
    ple_patch = patch_ple_layer()
    env, environment_check = runtime_environment(
        deep_preload_validation=not fast_start,
        tuning=tuning,
    )
    if fast_start:
        ipc_check = {
            "skipped": True,
            "reason": "diagnostic-only duplicate of live PLE startup",
        }
        cli_check = {
            "skipped": True,
            "reason": "the exact pinned server command is exercised directly",
        }
    else:
        ipc_check = diagnose_pidfd_ipc(env)
        cli_check = validate_cli_contract(env)
    model_check = verify_model(model_dir, full_file_hashes=not fast_start)
    server_identity = start_server(model_dir, env, tuning=tuning)
    readiness = wait_for_server(server_identity, setup_deadline)
    server_identity = update_ready_identity(server_identity)
    if fast_start:
        preflight = {
            "mode": "fast",
            "skipped": True,
            "reason": "the authorized one-game smoke is the integrated serving test",
            "served_model": SERVED_MODEL_NAME,
            "models_endpoint": readiness["models"],
            "skipped_requests": [
                "synthetic text",
                "synthetic tool call",
                "synthetic vision",
                "C1 throughput",
                "C4 throughput",
                "pre-game MTP telemetry",
            ],
            "final_mtp_telemetry_source": "vllm-metrics-final.prom from teardown",
        }
    else:
        preflight = run_preflights(setup_deadline)
    write_json(PREFLIGHT_PATH, preflight)
    if fast_start:
        process_gpu = {
            "deferred": True,
            "reason": "teardown records final process ownership and GPU state",
        }
    else:
        server_identity = update_ready_identity(server_identity)
        process_gpu = capture_process_and_gpu(server_identity)
    persisted = persist_analyzer_environment(env)
    provenance = {
        "source_dataset": SOURCE_DATASET,
        "source_identity_sha256": sha256_file(BUNDLE_DIR / "SOURCE_IDENTITY.json"),
        "runtime_dataset": RUNTIME_DATASET,
        "model_hf_repo": MODEL_HF_REPO,
        "model_hf_revision": MODEL_HF_REVISION,
        "vllm_image": VLLM_IMAGE,
        "vllm_index_digest": VLLM_INDEX_DIGEST,
        "vllm_version": VLLM_VERSION,
        "vllm_tuning": tuning,
        "fast_start": fast_start,
        "fast_start_environment": FAST_START_ENV,
        "setup_started_epoch": setup_started_epoch,
        "setup_total_timeout_seconds": SETUP_TOTAL_TIMEOUT,
        "server_ready_timeout_seconds": SERVER_READY_TIMEOUT,
        "inventory": inventory,
        "runtime_storage": storage_check,
        "runtime": runtime_check,
        "ple_patch": ple_patch,
        "model": model_check,
        "environment": environment_check,
        "ipc_permission": ipc_check,
        "cli_contract": cli_check,
        "server_identity": server_identity,
        "readiness": readiness,
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "process_gpu": process_gpu,
        "persisted_analyzer_environment": persisted,
        "server_log_sha256_at_preflight": sha256_file(SERVER_LOG),
        "server_log_tail": tail_log(300),
        "setup_elapsed_seconds": time.monotonic() - setup_started,
        "completed_epoch": time.time(),
    }
    write_json(SETUP_PROVENANCE, provenance)
    summary = {
        "mode": "fast" if fast_start else "full",
        "ready_seconds": readiness["ready_seconds"],
        "pid": server_identity["pid"],
    }
    if not fast_start:
        summary.update(
            {
                "c1_tokens_per_second": preflight["c1_throughput"][
                    "completion_tokens_per_second"
                ],
                "c4_tokens_per_second": preflight["c4_throughput"][
                    "aggregate_completion_tokens_per_second"
                ],
                "mtp_draft_tokens": preflight["native_mtp"]["draft_tokens"],
                "mtp_accepted_tokens": preflight["native_mtp"]["accepted_tokens"],
            }
        )
    print("VLLM_SETUP_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)


def _setup_timeout_handler(_signum: int, _frame: Any) -> None:
    raise TimeoutError(f"The {SETUP_TOTAL_TIMEOUT}s setup/preflight ceiling expired.")


if __name__ == "__main__":
    try:
        if not hasattr(signal, "SIGALRM"):
            raise RuntimeError("The pinned Kaggle setup requires Linux SIGALRM support.")
        signal.signal(signal.SIGALRM, _setup_timeout_handler)
        signal.alarm(SETUP_TOTAL_TIMEOUT)
        main()
        signal.alarm(0)
    except BaseException as exc:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "server_log_tail": tail_log(400),
            "cleanup": stop_owned_server(),
            "failed_epoch": time.time(),
        }
        write_json(FAILURE_PATH, failure)
        print("VLLM_SETUP_FAILED", json.dumps(failure, sort_keys=True), flush=True)
        raise
