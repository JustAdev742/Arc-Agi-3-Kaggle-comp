"""vLLM server helpers for the Kaggle notebook and the local box.

The flag set below is the *starting point* from CLAUDE.md (FP8 weights, MTP speculative
decoding, prefix caching, FP8 KV cache). Speculative-decoding syntax differs between vLLM
versions; ``VLLM_EXTRA_ARGS`` overrides everything after the model path.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from typing import Optional


def build_vllm_command(model_dir: str, *, port: int = 8000, served_name: str = "arc3-model",
                       max_model_len: int = 32768, gpu_mem: float = 0.90, mtp_tokens: int = 2,
                       kv_cache_dtype: str = "fp8", extra: Optional[str] = None) -> list[str]:
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model_dir,
           "--served-model-name", served_name, "--port", str(port), "--host", "127.0.0.1",
           "--max-model-len", str(max_model_len), "--gpu-memory-utilization", str(gpu_mem),
           "--enable-prefix-caching", "--trust-remote-code", "--enable-auto-tool-choice",
           "--tool-call-parser", "hermes", "--reasoning-parser", "qwen3"]
    if kv_cache_dtype:
        cmd += ["--kv-cache-dtype", kv_cache_dtype]
    if mtp_tokens > 0:
        cmd += ["--speculative-config", '{"method": "mtp", "num_speculative_tokens": %d}' % mtp_tokens]
    extra = os.environ.get("VLLM_EXTRA_ARGS", "") if extra is None else extra
    if extra:
        cmd += shlex.split(extra)
    return cmd


def start_vllm(model_dir: str, *, log_path: str = "vllm.log", **kw) -> subprocess.Popen:
    cmd = build_vllm_command(model_dir, **kw)
    logf = open(log_path, "a")
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)


def wait_for_server(base_url: str = "http://127.0.0.1:8000/v1", *, timeout_s: float = 1800, proc: Optional[subprocess.Popen] = None) -> bool:
    import requests

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            r = requests.get(f"{base_url}/models", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False
