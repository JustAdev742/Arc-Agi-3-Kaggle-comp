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


def gpu_memory_mib() -> Optional[tuple[int, int]]:
    """(free, total) MiB of the first GPU from nvidia-smi, or None when unavailable."""
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free,memory.total",
                                       "--format=csv,noheader,nounits"], text=True, timeout=20)
        free, total = [int(float(x.strip())) for x in out.strip().splitlines()[0].split(",")[:2]]
        return free, total
    except Exception:
        return None


def gpu_fraction_available(reserve_mib: int = 1536) -> Optional[float]:
    """Fraction of the first GPU a *new* vLLM process may claim: free memory minus a reserve for its own CUDA
    context, over the total. vLLM refuses to start when free memory is below gpu_memory_utilization * total,
    so a second server (the council's specialist) has to be sized from what the first one left."""
    m = gpu_memory_mib()
    if not m or m[1] <= 0:
        return None
    return max(0.0, (m[0] - reserve_mib) / m[1])


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
                       tensor_parallel: Optional[int] = None, enforce_eager: bool = False,
                       extra: Optional[str] = None) -> list[str]:
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
    if enforce_eager:  # no CUDA graphs / torch.compile: slower per token, but the most likely start on an untested kernel path
        cmd += ["--enforce-eager"]
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


def start_vllm(model_dir: str, *, log_path: str = "vllm.log", env_extra: Optional[dict] = None, **kw) -> subprocess.Popen:
    cmd = build_vllm_command(model_dir, **kw)
    env = kaggle_env()
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})
    logf = open(log_path, "a")
    if env_extra:
        logf.write("# env " + " ".join(f"{k}={v}" for k, v in env_extra.items()) + "\n")
    logf.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
    logf.flush()
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)


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


def _tiny_png() -> str:
    """A 16x16 grey PNG (base64) for the image probe; no PIL needed."""
    import base64
    import struct
    import zlib

    raw = b"".join(b"\x00" + b"\x80" * 48 for _ in range(16))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return base64.b64encode(png).decode()


def probe_completion(base_url: str, model: str, *, timeout_s: float = 300.0, with_image: bool = False) -> tuple[bool, str]:
    """One tiny chat completion. Readiness is not enough: on Kaggle the server can come up and then
    fail on the first request (FlashInfer inside the MTP draft model, diag v4 2026-09-16). ``with_image``
    sends a small PNG too, so a vision model whose image path is broken fails here, not in the first game."""
    import requests

    content: object = "Reply with the single word: ready"
    if with_image:
        content = [{"type": "text", "text": "Reply with the single word: ready"},
                   {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _tiny_png()}}]
    try:
        r = requests.post(f"{base_url}/chat/completions", json={
            "model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False}}, timeout=timeout_s)
        if r.status_code != 200:
            return False, f"{r.status_code}: {r.text[:300]}"
        return True, (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:300]


# Flags for the council's specialist server (Qwen3-VL-8B class: no MTP head, no thinking, hermes tool format).
# Sized for 6 roles x 8 concurrent games of ~3-5k-token prompts with one small image each.
SPECIALIST_TUNED: dict = {"max_model_len": 16384, "mtp_tokens": 0, "max_num_seqs": 48, "images_per_prompt": 4,
                          "reasoning_parser": "none", "tool_parser": "hermes", "fit_gpu_mem": True, "probe_image": True}
SPECIALIST_CONSERVATIVE: dict = {**SPECIALIST_TUNED, "kv_cache_dtype": "auto", "enforce_eager": True,
                                 "max_num_seqs": 16, "max_model_len": 8192}

# exp-010 (2026-09-16, runs/_kaggle_output/arc3-eval-dev-council/vllm-specialist.log): vLLM picked
# FlashInferCutlassNvFp4LinearKernel for the NVFP4 checkpoint and flashinfer's JIT raised "No supported CUDA
# architectures found for major versions [12]" (the RTX PRO 6000 is SM120). vLLM's NVFP4 GEMM backend is selectable;
# Marlin runs on every SM80+ part. UNVERIFIED on this wheelhouse: the FP8 rungs come first, this only helps the NVFP4 ones.
NVFP4_ENV: dict = {"VLLM_NVFP4_GEMM_BACKEND": "marlin", "VLLM_USE_FLASHINFER_MOE_FP4": "0"}
# exp-010b v2 (2026-09-16, runs/_kaggle_output/arc3-eval-dev-council-b/vllm-specialist.log): the official FP8 Qwen3-VL-8B
# build went through DeepGEMM and died at load with "Assertion error (deepgemm .../layout.hpp:60): Unknown SF
# transformation" on SM120, twice. Disabling DeepGEMM sends FP8 linear layers to the CUTLASS/Triton path.
FP8_ENV: dict = {"VLLM_USE_DEEP_GEMM": "0"}

# What the last start_vllm_with_fallback call ended with: {"ready", "attempt", "label", "model_dir", "gpu_mem", "elapsed_s"}.
LAST_START: dict = {}


def specialist_attempts(model_dirs: list, *, gpu_mem: float = 0.30, port: int = 8001,
                        served_name: str = "arc3-specialist") -> list[dict]:
    """Start ladder for the specialist server: for each candidate checkpoint, in order of preference, the tuned
    flags and then the conservative ones (bf16 KV cache, eager mode, fewer sequences, a smaller share of the
    GPU). Each attempt is a kwargs dict for ``start_vllm`` plus ``model_dir``, ``label`` and the ladder-only
    keys ``fit_gpu_mem`` / ``probe_image``. Exp-010 (2026-09-16): the NVFP4 specialist failed engine-core
    initialisation twice in 150 s and the council silently fell back to the coordinator model; a ladder
    over an official FP8 build and the NVFP4 build makes that fallback the last resort, not the first."""
    attempts: list[dict] = []
    for d in model_dirs:
        if not d:
            continue
        name = os.path.basename(os.path.normpath(str(d)))
        base = {"model_dir": str(d), "port": port, "served_name": served_name, "gpu_mem": gpu_mem}
        if "nvfp4" in name.lower() or "fp4" in name.lower():
            base["env_extra"] = dict(NVFP4_ENV)
        elif "fp8" in name.lower():
            base["env_extra"] = dict(FP8_ENV)
        attempts.append({**base, **SPECIALIST_TUNED, "label": f"{name} tuned"})
        attempts.append({**base, **SPECIALIST_CONSERVATIVE, "gpu_mem": round(gpu_mem * 0.85, 3), "label": f"{name} conservative"})
    return attempts


def _stop(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def start_vllm_with_fallback(model_dir: str, *, log_path: str = "vllm.log", timeout_s: float = 1500.0,
                             base_url: Optional[str] = None, attempts: Optional[list[dict]] = None,
                             **kw) -> tuple[Optional[subprocess.Popen], bool]:
    """Start vLLM and prove it with one real completion, walking a ladder of attempts until one works.

    Default ladder: the tuned flags in ``kw``, then the same without speculative decoding and with an auto KV
    cache. ``attempts`` replaces the ladder; each entry is a kwargs dict for ``start_vllm`` that may also carry
    ``model_dir`` (a different checkpoint), ``label`` (for the log), ``fit_gpu_mem`` (clamp gpu_mem to what
    nvidia-smi says is still free, for a second server on the same GPU) and ``probe_image`` (probe with an
    image). Returns (process, ready); ``LAST_START`` records which attempt won or that all failed."""
    port = int(kw.get("port", 8000))
    served = str(kw.get("served_name", "arc3-model"))
    if attempts is None:
        attempts = [{**kw, "label": "tuned"}, {**kw, "mtp_tokens": 0, "kv_cache_dtype": "auto", "label": "conservative"}]
    t0 = time.time()
    t_end = t0 + timeout_s
    n = len(attempts)
    LAST_START.clear()
    LAST_START.update({"ready": False, "attempts": n})
    for i, spec in enumerate(attempts):
        attempt = dict(spec)
        label = str(attempt.pop("label", f"attempt {i + 1}"))
        mdir = str(attempt.pop("model_dir", model_dir))
        fit = bool(attempt.pop("fit_gpu_mem", False))
        probe_image = bool(attempt.pop("probe_image", False))
        env_extra = attempt.pop("env_extra", None)
        a_port = int(attempt.get("port", port))
        a_served = str(attempt.get("served_name", served))
        url = base_url or f"http://127.0.0.1:{a_port}/v1"
        remaining = t_end - time.time()
        if remaining < 120:
            with open(log_path, "a") as f:
                f.write(f"\n# no time left for attempt {i + 1} ({label}); giving up\n")
            break
        if fit:
            avail = gpu_fraction_available()
            want = float(attempt.get("gpu_mem", 0.9))
            if avail is not None and avail < want:
                got = round(max(0.0, avail - 0.02), 3)
                with open(log_path, "a") as f:
                    f.write(f"\n# attempt {i + 1} ({label}): only {avail:.3f} of the GPU is free; gpu_mem {want} -> {got}\n")
                if got < 0.08:
                    continue
                attempt["gpu_mem"] = got
        with open(log_path, "a") as f:
            f.write(f"\n# attempt {i + 1}/{n} ({label}) model_dir={mdir} gpu_mem={attempt.get('gpu_mem')}\n")
        proc = start_vllm(mdir, log_path=log_path, env_extra=env_extra, **attempt)
        # Earlier attempts get a share of the budget so the later ones still have a chance; a process that
        # dies early returns at once from wait_for_server, so fast failures do not spend it.
        left = n - i
        budget = remaining if left == 1 else max(120.0, remaining * min(1.0, 1.2 / left))
        ok = wait_for_server(url, timeout_s=budget, proc=proc)
        if ok:
            ok, detail = probe_completion(url, a_served, timeout_s=min(300.0, max(30.0, t_end - time.time())), with_image=probe_image)
            with open(log_path, "a") as f:
                f.write(f"\n# probe completion attempt {i + 1} ({label}): {'OK' if ok else 'FAILED'} {detail[:200]}\n")
        if ok:
            LAST_START.update({"ready": True, "attempt": i + 1, "label": label, "model_dir": mdir,
                               "gpu_mem": attempt.get("gpu_mem"), "elapsed_s": round(time.time() - t0, 1)})
            return proc, True
        _stop(proc)
        with open(log_path, "a") as f:
            f.write(f"\n# attempt {i + 1} ({label}) failed; {'trying the next attempt' if i + 1 < n else 'giving up'}\n")
    LAST_START.update({"elapsed_s": round(time.time() - t0, 1)})
    return None, False
