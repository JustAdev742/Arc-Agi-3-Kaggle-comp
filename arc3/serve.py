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
                       kv_cache_dtype: str = "fp8", max_num_seqs: int = 32, images_per_prompt: int = 16,
                       tool_parser: Optional[str] = None, reasoning_parser: Optional[str] = None,
                       extra: Optional[str] = None) -> list[str]:
    """Env overrides: VLLM_TOOL_PARSER, VLLM_REASONING_PARSER, VLLM_MTP_TOKENS, VLLM_KV_CACHE_DTYPE,
    VLLM_MAX_MODEL_LEN, VLLM_EXTRA_ARGS. ``images_per_prompt`` matters: the REPL agent attaches one
    image per user turn and vLLM's default limit is one image per prompt."""
    tool_parser = tool_parser or os.environ.get("VLLM_TOOL_PARSER", "qwen3_coder")  # Qwen3.8 chat template uses <function=...><parameter=...> XML
    reasoning_parser = reasoning_parser or os.environ.get("VLLM_REASONING_PARSER", "qwen3")
    mtp_tokens = int(os.environ.get("VLLM_MTP_TOKENS", mtp_tokens))
    kv_cache_dtype = os.environ.get("VLLM_KV_CACHE_DTYPE", kv_cache_dtype)
    max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", max_model_len))
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model_dir,
           "--served-model-name", served_name, "--port", str(port), "--host", "127.0.0.1",
           "--max-model-len", str(max_model_len), "--gpu-memory-utilization", str(gpu_mem),
           "--max-num-seqs", str(max_num_seqs), "--limit-mm-per-prompt", '{"image": %d}' % images_per_prompt,
           "--enable-prefix-caching", "--trust-remote-code", "--enable-auto-tool-choice",
           "--tool-call-parser", tool_parser]
    if reasoning_parser and reasoning_parser != "none":
        cmd += ["--reasoning-parser", reasoning_parser]
    if kv_cache_dtype and kv_cache_dtype != "auto":
        cmd += ["--kv-cache-dtype", kv_cache_dtype]
    if mtp_tokens > 0:
        cmd += ["--speculative-config", '{"method": "mtp", "num_speculative_tokens": %d}' % mtp_tokens]
    extra = os.environ.get("VLLM_EXTRA_ARGS", "") if extra is None else extra
    if extra:
        cmd += shlex.split(extra)
    return cmd


def kaggle_env() -> dict[str, str]:
    """Environment for vLLM on Kaggle GPU images: the CUDA driver libs are off the linker path
    (the Milestone-1 winner's notebook prepends /usr/local/nvidia/lib64), and the flashinfer
    sampler is disabled by the 2nd/3rd place notebooks (VLLM_USE_FLASHINFER_SAMPLER=0)."""
    env = dict(os.environ)
    cuda = "/usr/local/nvidia/lib64"
    if os.path.isdir(cuda):
        for key in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
            env[key] = os.pathsep.join(p for p in [cuda, env.get(key, "")] if p)
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    env.setdefault("VLLM_STARTUP_TIMEOUT", "1800")
    return env


def start_vllm(model_dir: str, *, log_path: str = "vllm.log", **kw) -> subprocess.Popen:
    cmd = build_vllm_command(model_dir, **kw)
    logf = open(log_path, "a")
    logf.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
    logf.flush()
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=kaggle_env())


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
