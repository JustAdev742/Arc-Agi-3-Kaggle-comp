"""vLLM server helpers for the Kaggle notebook and the local box.

The flag set below is the *starting point* from CLAUDE.md (FP8 weights, MTP speculative
decoding, prefix caching, FP8 KV cache). Speculative-decoding syntax differs between vLLM
versions; ``VLLM_EXTRA_ARGS`` overrides everything after the model path.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from typing import Optional


def gpu_info() -> list[dict[str, str]]:
    """[{name, compute_cap, memory_mib}] from nvidia-smi, or [] when unavailable."""
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
                                       "--format=csv,noheader,nounits"], text=True, timeout=20)
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "compute_cap": parts[1], "memory_mib": parts[2]})
    return gpus


def default_kv_cache_dtype() -> str:
    """FP8 KV cache needs SM89+ (Ada, Hopper, Blackwell); older parts (T4 = SM75) must use auto."""
    gpus = gpu_info()
    if not gpus:
        return "auto"
    try:
        cc = min(float(g["compute_cap"]) for g in gpus)
    except ValueError:
        return "auto"
    return "fp8" if cc >= 8.9 else "auto"


def build_vllm_command(model_dir: str, *, port: int = 8000, served_name: str = "arc3-model",
                       max_model_len: int = 32768, gpu_mem: float = 0.90, mtp_tokens: int = 2,
                       kv_cache_dtype: Optional[str] = None, max_num_seqs: int = 32, images_per_prompt: int = 16,
                       tool_parser: Optional[str] = None, reasoning_parser: Optional[str] = None,
                       tensor_parallel: Optional[int] = None, extra: Optional[str] = None) -> list[str]:
    """Env overrides: VLLM_TOOL_PARSER, VLLM_REASONING_PARSER, VLLM_MTP_TOKENS, VLLM_KV_CACHE_DTYPE,
    VLLM_MAX_MODEL_LEN, VLLM_EXTRA_ARGS. ``images_per_prompt`` matters: the REPL agent attaches one
    image per user turn and vLLM's default limit is one image per prompt."""
    tool_parser = tool_parser or os.environ.get("VLLM_TOOL_PARSER", "qwen3_coder")  # Qwen3.8 chat template uses <function=...><parameter=...> XML
    # vLLM >= 0.27 ignores VLLM_ATTENTION_BACKEND; the backend is a CLI arg. On the RTX PRO 6000 (SM120)
    # the auto choice is FlashInfer, which needs cubins from NVIDIA's artifactory (no internet on Kaggle)
    # and dies on the first request (diag runs v2/v3, 2026-09-16). Triton attention was the other candidate.
    attention_backend = os.environ.get("ARC3_ATTENTION_BACKEND", "TRITON_ATTN")
    reasoning_parser = reasoning_parser or os.environ.get("VLLM_REASONING_PARSER", "qwen3")
    mtp_tokens = int(os.environ.get("VLLM_MTP_TOKENS", mtp_tokens))
    kv_cache_dtype = os.environ.get("VLLM_KV_CACHE_DTYPE") or kv_cache_dtype or default_kv_cache_dtype()
    max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", max_model_len))
    tensor_parallel = int(os.environ.get("VLLM_TP", tensor_parallel or 1))
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model_dir,
           "--served-model-name", served_name, "--port", str(port), "--host", "127.0.0.1",
           "--max-model-len", str(max_model_len), "--gpu-memory-utilization", str(gpu_mem),
           "--max-num-seqs", str(max_num_seqs), "--limit-mm-per-prompt", '{"image": %d}' % images_per_prompt,
           "--enable-prefix-caching", "--trust-remote-code", "--enable-auto-tool-choice",
           "--tool-call-parser", tool_parser]
    if attention_backend and attention_backend.lower() != "auto":
        cmd += ["--attention-backend", attention_backend]
    if tensor_parallel > 1:
        cmd += ["--tensor-parallel-size", str(tensor_parallel)]
    if reasoning_parser and reasoning_parser != "none":
        cmd += ["--reasoning-parser", reasoning_parser]
    if kv_cache_dtype and kv_cache_dtype != "auto":
        cmd += ["--kv-cache-dtype", kv_cache_dtype]
    if mtp_tokens > 0:
        # The MTP draft model does NOT inherit the main attention backend (vLLM resets it and auto-selects
        # FlashInfer, which fails offline on SM120; diag v4 2026-09-16). SpeculativeConfig has its own
        # `attention_backend` field, so pin it there as well.
        spec = {"method": "mtp", "num_speculative_tokens": int(mtp_tokens)}
        if attention_backend and attention_backend.lower() != "auto":
            spec["attention_backend"] = attention_backend
        cmd += ["--speculative-config", json.dumps(spec)]
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


def probe_completion(base_url: str, model: str, *, timeout_s: float = 300.0) -> tuple[bool, str]:
    """One tiny chat completion. Readiness is not enough: on Kaggle the server can come up and then
    fail on the first request (FlashInfer inside the MTP draft model, diag v4 2026-09-16)."""
    import requests

    try:
        r = requests.post(f"{base_url}/chat/completions", json={
            "model": model, "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
            "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False}}, timeout=timeout_s)
        if r.status_code != 200:
            return False, f"{r.status_code}: {r.text[:300]}"
        return True, (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:300]


def start_vllm_with_fallback(model_dir: str, *, log_path: str = "vllm.log", timeout_s: float = 1500.0,
                             base_url: Optional[str] = None, **kw) -> tuple[Optional[subprocess.Popen], bool]:
    """Start vLLM with the tuned flags and prove it with one real completion; on failure retry once
    with the conservative set (no speculative decoding, auto KV cache). Returns (process, ready)."""
    port = int(kw.get("port", 8000))
    served = str(kw.get("served_name", "arc3-model"))
    base_url = base_url or f"http://127.0.0.1:{port}/v1"
    attempts = [dict(kw), {**kw, "mtp_tokens": 0, "kv_cache_dtype": "auto"}]
    t_end = time.time() + timeout_s
    for i, attempt in enumerate(attempts):
        remaining = t_end - time.time()
        if remaining < 120:
            break
        proc = start_vllm(model_dir, log_path=log_path, **attempt)
        # First attempt gets at most ~60% of the budget so the fallback still has a chance.
        budget = remaining * (0.6 if i == 0 and len(attempts) > 1 else 1.0)
        ok = wait_for_server(base_url, timeout_s=budget, proc=proc)
        if ok:
            ok, detail = probe_completion(base_url, served, timeout_s=min(300.0, max(30.0, t_end - time.time())))
            with open(log_path, "a") as f:
                f.write(f"\n# probe completion attempt {i + 1}: {'OK' if ok else 'FAILED'} {detail[:200]}\n")
        if ok:
            return proc, True
        try:
            proc.terminate()
            proc.wait(timeout=30)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        with open(log_path, "a") as f:
            f.write(f"\n# attempt {i + 1} failed; {'retrying with conservative flags' if i + 1 < len(attempts) else 'giving up'}\n")
    return None, False
