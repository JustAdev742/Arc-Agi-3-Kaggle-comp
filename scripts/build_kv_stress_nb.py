#!/usr/bin/env python
"""Build a short serving stress notebook for the Flash-Next vLLM server at a chosen KV-cache size.

    .venv/bin/python scripts/build_kv_stress_nb.py --out <dir> --slug arc3-kv-stress-8g --kv-gib 8 [--minutes 12]

It keeps the setup cells of kaggle/taaf/base-thui-animfast.ipynb (serving setup, analyzer env, solver import checks)
and replaces the benchmark with a synthetic load shaped like the Duck's: the Duck's own system prompt, user turns padded
to about 20k prompt tokens with a 256x256 board image, thinking on, 28 concurrent clients, for ``--minutes``. It
records completions, errors, generated tokens and, every 10 s, the server's running/waiting counts, KV usage and
preemptions, then writes /kaggle/working/kv_stress.json and tears the server down. Use it to check that a larger
``TAAF_VLLM_KV_CACHE_MEMORY_BYTES`` neither runs out of GPU memory nor slows decoding before a full run uses it.

``--engine sglang --sglang-running R [--sglang-hicache-gb G] [--sglang-mamba-cache N]`` builds the SGLang version
(docs/research/sglang-serving-plan.md section 5): the same setup cells with Keith's setup command replaced by
scripts/sglang_serving.py (as ``build_taaf_nb.py --engine sglang`` does), then the plan's seven-check functional gate
(its harness-path checks use a copy of the anim bundle patched with P30 and P30b), then the identical 12-minute
28-client load, sampling SGLang's metrics, GPU memory and host RAM, and finally the go/no-go of plan section 6, all
written to /kaggle/working/sgl_stress.json (``SGL_STRESS_GO`` in the log). The two notebooks of the plan:

    ... --engine sglang --slug arc3-sgl-stress-r12 --sglang-running 12
    ... --engine sglang --slug arc3-sgl-stress-r16-hic32 --sglang-running 16 --sglang-mamba-cache 96 --sglang-hicache-gb 32
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sglang_serving as sgl  # noqa: E402

BASE = ROOT / "kaggle" / "taaf" / "base-thui-animfast.ipynb"
KV_ANCHOR = '"TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "5368709120"'
ENV_END = '"TAAF_VLLM_OMP_THREADS": "1"\n}'
BUNDLE_ANCHOR = ('BUNDLE_DIR = _find_bundle_dir("duck-harness-kaggle")          '
                 '# his: serving_setup.py, vllm patches, watchdog, teardown')
PATCH_SRC = ROOT / "scripts" / "taaf_ours_patch.py"
BASELINE_TOK_S = 152.9  # kvstress-7g75-b2k (runs/kvstress-7g75-b2k/summary.json): the vLLM profile of exp-054

STRESS = r'''
# ours: synthetic load shaped like the Duck's requests, then a JSON summary (no games are played)
import base64
import io
import random
import re
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

MINUTES = __MINUTES__
CLIENTS = 28
BASE_URL = os.environ["LOCAL_ANALYZER_BASE_URL"].rstrip("/")
MODEL = os.environ["LOCAL_ANALYZER_MODEL_ID"]
SYSTEM = _tool_agent._build_system_prompt(tool_output_tokens=1024)
rng = random.Random(0)
WORDS = ("grid object color shape row column move left right up down click level goal wall key door path agent "
         "blue red green yellow gray block target score frame change probe search plan").split()


def board_image() -> dict:
    img = Image.new("RGB", (256, 256))
    px = img.load()
    for r in range(64):
        for c in range(64):
            v = rng.randrange(16)
            color = ((v * 53) % 256, (v * 97) % 256, (v * 151) % 256)
            for dr in range(4):
                for dc in range(4):
                    px[c * 4 + dc, r * 4 + dr] = color
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


def user_text(n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


stop = threading.Event()
stats = {"ok": 0, "errors": 0, "error_samples": [], "completion_tokens": 0, "prompt_tokens": 0, "latency_s": []}
lock = threading.Lock()


def client(i: int) -> None:
    while not stop.is_set():
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [{"type": "text", "text": user_text(15000)
                                                  + "\nThink about the board, then write a short plan."},
                                                 board_image()]}]
        t0 = _time.time()
        try:
            r = requests.post(f"{BASE_URL}/chat/completions", timeout=900, json={
                "model": MODEL, "messages": messages, "max_tokens": 1500, "temperature": 0.6, "top_p": 0.95,
                "top_k": 20, "chat_template_kwargs": {"enable_thinking": True}})
            r.raise_for_status()
            usage = r.json().get("usage") or {}
            with lock:
                stats["ok"] += 1
                stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                stats["latency_s"].append(round(_time.time() - t0, 1))
        except Exception as exc:  # noqa: BLE001
            with lock:
                stats["errors"] += 1
                if len(stats["error_samples"]) < 5:
                    stats["error_samples"].append(repr(exc)[:300])
            _time.sleep(5)


def scrape() -> dict:
    text = requests.get(BASE_URL.rsplit("/v1", 1)[0] + "/metrics", timeout=10).text
    out = {}
    for key in ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
                "vllm:gpu_cache_usage_perc", "vllm:num_preemptions_total", "vllm:generation_tokens_total",
                "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"):
        m = re.search(rf"^{re.escape(key)}(?:{{[^}}]*}})? ([0-9.eE+-]+)$", text, re.M)
        if m:
            out[key] = float(m.group(1))
    return out


samples = []
t_start = _time.time()
with ThreadPoolExecutor(CLIENTS) as pool:
    for i in range(CLIENTS):
        pool.submit(client, i)
    while _time.time() - t_start < MINUTES * 60:
        _time.sleep(10)
        try:
            samples.append({"t": round(_time.time() - t_start), **scrape()})
        except Exception as exc:  # noqa: BLE001
            samples.append({"t": round(_time.time() - t_start), "scrape_error": repr(exc)[:200]})
        print(samples[-1], flush=True)
    stop.set()
elapsed = _time.time() - t_start
run = [s.get("vllm:num_requests_running") for s in samples if "vllm:num_requests_running" in s]
summary = {"kv_bytes": os.environ.get("TAAF_VLLM_KV_CACHE_MEMORY_BYTES"), "minutes": MINUTES, "clients": CLIENTS,
           "elapsed_s": round(elapsed), "ok": stats["ok"], "errors": stats["errors"],
           "error_samples": stats["error_samples"],
           "generated_tokens_per_s": round(stats["completion_tokens"] / elapsed, 1),
           "mean_prompt_tokens": round(stats["prompt_tokens"] / max(1, stats["ok"])),
           "mean_latency_s": round(sum(stats["latency_s"]) / max(1, len(stats["latency_s"])), 1),
           "running_mean": round(sum(run) / len(run), 2) if run else None, "running_max": max(run) if run else None,
           "samples": samples}
(WORKING_DIR / "kv_stress.json").write_text(json.dumps(summary, indent=1))
print("KV_STRESS", json.dumps({k: v for k, v in summary.items() if k != "samples"}), flush=True)
'''

TEARDOWN = '''
for command in json.loads((BUNDLE_DIR / "teardown_commands.json").read_text()):
    print(f"taaf.kaggle: teardown command: {command}", flush=True)
    subprocess.run(command, shell=True, check=False, cwd=WORKING_DIR, env=_command_env(), timeout=30.0)
import pandas as pd
pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(
    WORKING_DIR / "submission.parquet", index=False)
'''

SGL_GATE = r'''
# ours: SGLang functional gate (docs/research/sglang-serving-plan.md section 5); every check goes to sgl_stress.json
import base64
import io
import random
import re
import shutil
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

BASE_URL = os.environ["LOCAL_ANALYZER_BASE_URL"].rstrip("/")
ROOT_URL = BASE_URL.rsplit("/v1", 1)[0]
MODEL = os.environ["LOCAL_ANALYZER_MODEL_ID"]
ENGINE = os.environ.get("OURS_SERVING", "")
RUNNING = int(os.environ.get("OURS_SGLANG_RUNNING") or __RUNNING__)
SYSTEM = _tool_agent._build_system_prompt(tool_output_tokens=1024)
SGL_JSON = WORKING_DIR / "sgl_stress.json"
REPORT = {"slug": __SLUG__, "engine": ENGINE, "profile": __PROFILE__, "gate": {}}
rng = random.Random(0)
WORDS = ("grid object color shape row column move left right up down click level goal wall key door path agent "
         "blue red green yellow gray block target score frame change probe search plan").split()
print(f"SGL_GATE engine={ENGINE} running={RUNNING}", flush=True)


def board_image() -> dict:
    img = Image.new("RGB", (256, 256))
    px = img.load()
    for r in range(64):
        for c in range(64):
            v = rng.randrange(16)
            color = ((v * 53) % 256, (v * 97) % 256, (v * 151) % 256)
            for dr in range(4):
                for dc in range(4):
                    px[c * 4 + dc, r * 4 + dr] = color
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


def user_text(n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def chat(payload: dict, timeout: float = 900.0) -> dict:
    r = requests.post(f"{BASE_URL}/chat/completions", json={"model": MODEL, **payload}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def save_report() -> None:
    SGL_JSON.write_text(json.dumps(REPORT, indent=1, default=str))


def check(name: str, fn) -> None:
    t0 = _time.time()
    try:
        out = fn()
    except Exception as exc:  # noqa: BLE001
        out = {"ok": False, "error": repr(exc)[:600]}
    out["seconds"] = round(_time.time() - t0, 1)
    REPORT["gate"][name] = out
    save_report()
    print("SGL_GATE", name, json.dumps(out, default=str)[:800], flush=True)


NO_THINK = {"temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
PY_TOOL = [{"type": "function", "function": {"name": "python", "description": "Python code to run.", "parameters": {
    "type": "object", "properties": {"code": {"type": "string", "description": "Python code to run."}},
    "required": ["code"]}}}]
REASONING_1K = user_text(1000)


def history(key=None) -> list:
    past = {"role": "assistant", "content": "The answer is 4."}
    if key:
        past[key] = REASONING_1K
    return [{"role": "user", "content": "What is 2 + 2?"}, past,
            {"role": "user", "content": "And 3 + 3? Reply with the number only."}]


def prompt_tokens(messages: list) -> int:
    return chat({"messages": messages, "max_tokens": 8, **NO_THINK})["usage"]["prompt_tokens"]


# The patched harness path (checks 4 and 6): a copy of the anim bundle with P30 and P30b only, driven in a subprocess
# so its `inference` package does not clash with the one this kernel imported.
GATE_BUNDLE = Path("/tmp/ours_gate_bundle")
if GATE_BUNDLE.exists():
    shutil.rmtree(GATE_BUNDLE)
shutil.copytree(ANIM_BUNDLE_DIR, GATE_BUNDLE, ignore=shutil.ignore_patterns("__pycache__"))
_gate_ns = {"__name__": "taaf_ours_patch"}
exec(compile(_OURS_PATCH_SOURCE, "taaf_ours_patch.py", "exec"), _gate_ns)
print("ours: gate bundle", _gate_ns["apply"](GATE_BUNDLE, ["P30", "P30B"]), flush=True)
HARNESS_PROBE = """
import json, os, sys
os.environ["LOCAL_ANALYZER_MAX_OUTPUT"] = "16"
sys.path.insert(0, sys.argv[1])
import requests
from inference.agent import tool_agent as ta
past = {"role": "assistant", "content": "The answer is 4.", "reasoning": sys.argv[2]}
messages = [{"role": "user", "content": "What is 2 + 2?"}, past,
            {"role": "user", "content": "And 3 + 3? Reply with the number only."}]
agent = ta.ToolAgent(model=os.environ["LOCAL_ANALYZER_MODEL_ID"])
out = {}
for engine in ("sglang", ""):
    os.environ["OURS_SERVING"] = engine
    out["prompt_tokens_" + (engine or "p30_off")] = (agent._chat_completion(json.loads(json.dumps(messages)),
                                                                          tools=None).usage or {}).get("prompt_tokens")
try:
    agent._chat_completion([{"role": "user", "content": "x " * 45000}], tools=None)
    out["over_length"] = "accepted"
except requests.RequestException as exc:
    out["over_length"] = str(exc)[:600]
    out["over_length_matched"] = ta._is_context_length_error(exc)
print("HARNESS_PROBE " + json.dumps(out))
"""
_probe = subprocess.run([sys.executable, "-c", HARNESS_PROBE, str(GATE_BUNDLE / "src" / "ARC3-Inference"), REASONING_1K],
                        capture_output=True, text=True, timeout=1800, env=_command_env())
_line = next((x for x in _probe.stdout.splitlines() if x.startswith("HARNESS_PROBE ")), None)
HARNESS = json.loads(_line.split(" ", 1)[1]) if _line else {"error": (_probe.stderr or _probe.stdout)[-1500:]}
REPORT["harness_probe"] = HARNESS
print("SGL_GATE harness probe", json.dumps(HARNESS)[:800], flush=True)


def c_models() -> dict:
    ids = [m.get("id") for m in requests.get(f"{BASE_URL}/models", timeout=60).json().get("data", [])]
    return {"ok": ids == [MODEL], "ids": ids}


def c_vision() -> dict:
    text = "Describe the colours in the board image in one sentence."
    out = {}
    for label, content in (("text_only", text), ("with_image", [{"type": "text", "text": text}, board_image()])):
        body = chat({"messages": [{"role": "user", "content": content}], "max_tokens": 64, **NO_THINK})
        out[label] = {"prompt_tokens": body["usage"]["prompt_tokens"],
                      "reply": (body["choices"][0]["message"].get("content") or "")[:160]}
    out["image_tokens"] = out["with_image"]["prompt_tokens"] - out["text_only"]["prompt_tokens"]
    out["ok"] = bool(out["with_image"]["reply"].strip()) and out["image_tokens"] > 0
    return out


def c_tool_call() -> dict:
    body = chat({"messages": [{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": "Use the python tool to print the sum of 2 and 3. Call it now."}],
                 "tools": PY_TOOL, "tool_choice": "auto", "max_tokens": 4096, "temperature": 0.6, "top_p": 0.95,
                 "top_k": 20, "chat_template_kwargs": {"enable_thinking": True}})
    msg = body["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    content = msg.get("content") or ""
    try:
        args_ok = "code" in json.loads(calls[0]["function"]["arguments"])
    except Exception:  # noqa: BLE001
        args_ok = False
    return {"ok": bool(calls) and calls[0]["function"]["name"] == "python" and args_ok
            and "<tool_call>" not in content and "<function=" not in content,
            "calls": len(calls), "arguments": calls[0]["function"]["arguments"][:200] if calls else None,
            "content_head": content[:200], "finish_reason": body["choices"][0].get("finish_reason")}


def c_reasoning_round_trip() -> dict:
    base = prompt_tokens(history())
    with_rc = prompt_tokens(history("reasoning_content")) - base
    with_r = prompt_tokens(history("reasoning")) - base
    on, off = HARNESS.get("prompt_tokens_sglang"), HARNESS.get("prompt_tokens_p30_off")
    harness_delta = on - off if isinstance(on, int) and isinstance(off, int) else None
    return {"ok": with_rc >= 800 and abs(with_r) <= 20 and harness_delta is not None and harness_delta >= 800,
            "reasoning_content_delta": with_rc, "reasoning_delta": with_r, "p30_harness_delta": harness_delta}


def c_response_split() -> dict:
    body = chat({"messages": [{"role": "user", "content": "What is 17 * 23? Think, then give the number."}],
                 "max_tokens": 4096, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                 "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}})
    msg = body["choices"][0]["message"]
    reasoning, content = msg.get("reasoning_content") or msg.get("reasoning") or "", msg.get("content") or ""
    return {"ok": bool(reasoning.strip()) and "</think>" not in content and "<think>" not in content,
            "reasoning_chars": len(reasoning), "content_head": content[:160],
            "finish_reason": body["choices"][0].get("finish_reason")}


def c_over_length() -> dict:
    text = str(HARNESS.get("over_length", ""))
    return {"ok": HARNESS.get("over_length_matched") is True and "longer than the model's context length" in text,
            "error_head": text[:300]}


def c_concurrency() -> dict:
    def one(i: int):
        t0 = _time.time()
        body = chat({"messages": [{"role": "user", "content": f"[{i}] " + user_text(24000) + "\nReply with one word."}],
                     "max_tokens": 16, **NO_THINK}, timeout=1800)
        return body["usage"]["prompt_tokens"], round(_time.time() - t0, 1)

    with ThreadPoolExecutor(RUNNING) as pool:
        done = list(pool.map(one, range(RUNNING)))
    return {"ok": len(done) == RUNNING and min(p for p, _ in done) >= 20000, "requests": len(done),
            "prompt_tokens_min": min(p for p, _ in done), "latency_max_s": max(s for _, s in done)}


for _name, _fn in (("1_models", c_models), ("2_vision", c_vision), ("3_tool_call", c_tool_call),
                   ("4_reasoning_round_trip", c_reasoning_round_trip), ("5_response_split", c_response_split),
                   ("6_over_length", c_over_length), ("7_concurrency", c_concurrency)):
    check(_name, _fn)
REPORT["gate_passed"] = all(v.get("ok") for v in REPORT["gate"].values()) and len(REPORT["gate"]) == 7
save_report()
print("SGL_GATE_PASSED", REPORT["gate_passed"], flush=True)
'''

SGL_LOAD = r'''
# ours: the 12-minute load of the vLLM stress tests (28 clients, Duck system prompt, 15,000 words + a 256x256 image,
# max_tokens 1500, thinking on, top_k 20), sampling SGLang's metrics, GPU memory and host RAM, then plan section 6
MINUTES = __MINUTES__
CLIENTS = 28
BASELINE_TOK_S = __BASELINE__
KEYS = ("sglang:num_running_reqs", "sglang:num_queue_reqs", "sglang:token_usage", "sglang:num_retracted_reqs",
        "sglang:num_retracted_requests_total", "sglang:spec_accept_length", "sglang:cache_hit_rate",
        "sglang:gen_throughput", "sglang:generation_tokens_total", "sglang:kv_available_tokens", "sglang:mamba_usage",
        "sglang:max_total_num_tokens", "sglang:startup_time_seconds", "sglang:weight_load_duration_seconds",
        "sglang:startup_cuda_graph_time_seconds", "vllm:num_requests_running", "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc", "vllm:num_preemptions_total", "vllm:generation_tokens_total")
stop = threading.Event()
stats = {"ok": 0, "errors": 0, "error_samples": [], "completion_tokens": 0, "prompt_tokens": 0, "latency_s": []}
lock = threading.Lock()


def client(i: int) -> None:
    while not stop.is_set():
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [{"type": "text", "text": user_text(15000)
                                                  + "\nThink about the board, then write a short plan."},
                                                 board_image()]}]
        t0 = _time.time()
        try:
            usage = chat({"messages": messages, "max_tokens": 1500, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                          "chat_template_kwargs": {"enable_thinking": True}}).get("usage") or {}
            with lock:
                stats["ok"] += 1
                stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                stats["latency_s"].append(round(_time.time() - t0, 1))
        except Exception as exc:  # noqa: BLE001
            with lock:
                stats["errors"] += 1
                if len(stats["error_samples"]) < 5:
                    stats["error_samples"].append(repr(exc)[:300])
            _time.sleep(5)


def scrape() -> dict:
    text = requests.get(ROOT_URL + "/metrics", timeout=10).text
    out = {}
    for key in KEYS:
        values = [float(m.group(1)) for m in re.finditer(rf"^{re.escape(key)}(?:{{[^}}]*}})? ([0-9.eE+-]+)$", text, re.M)]
        if values:
            out[key] = sum(values)
    return out


def machine() -> dict:
    out = {}
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        out["gpu_used_mib"], out["gpu_total_mib"] = (float(x) for x in smi.split(","))
    except Exception as exc:  # noqa: BLE001
        out["gpu_error"] = repr(exc)[:100]
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
    kib = {k: float(v.split()[0]) for k, v in info.items() if k in ("MemTotal", "MemAvailable")}
    out["host_used_gib"] = round((kib["MemTotal"] - kib["MemAvailable"]) / 2**20, 1)
    return out


try:
    first = scrape()
except Exception as exc:  # noqa: BLE001
    first = {"scrape_error": repr(exc)[:200]}
samples = []
t_start = _time.time()
with ThreadPoolExecutor(CLIENTS) as pool:
    for i in range(CLIENTS):
        pool.submit(client, i)
    while _time.time() - t_start < MINUTES * 60:
        _time.sleep(10)
        try:
            samples.append({"t": round(_time.time() - t_start), **scrape(), **machine()})
        except Exception as exc:  # noqa: BLE001
            samples.append({"t": round(_time.time() - t_start), "scrape_error": repr(exc)[:200], **machine()})
        print(samples[-1], flush=True)
    stop.set()
elapsed = _time.time() - t_start
try:
    last = scrape()
except Exception as exc:  # noqa: BLE001
    last = {"scrape_error": repr(exc)[:200]}
run_key = "sglang:num_running_reqs" if ENGINE == "sglang" else "vllm:num_requests_running"
run = [s[run_key] for s in samples if run_key in s]
lat = sorted(stats["latency_s"])
tok_s = round(stats["completion_tokens"] / elapsed, 1)
retracted = (last.get("sglang:num_retracted_requests_total", 0.0) - first.get("sglang:num_retracted_requests_total", 0.0))
setup_report = json.loads((WORKING_DIR / "sglang-setup.json").read_text()) if (WORKING_DIR / "sglang-setup.json").exists() else {}
rung = setup_report.get("rung") or {}
events_path = WORKING_DIR / "sglang-watchdog.jsonl"
events = [json.loads(x) for x in events_path.read_text().splitlines() if x.strip()] if events_path.exists() else []
restarts = sum(1 for e in events if e.get("event") == "restart_launched")
oom_lines = [line for log in WORKING_DIR.glob("sglang-server-rung*.log")
             for line in log.read_text(errors="replace").splitlines()
             if "out of memory" in line.lower() or "OutOfMemoryError" in line]
peak_host = max((s.get("host_used_gib", 0.0) for s in samples), default=None)
summary = {"minutes": MINUTES, "clients": CLIENTS, "elapsed_s": round(elapsed), "ok": stats["ok"],
           "errors": stats["errors"], "error_samples": stats["error_samples"], "generated_tokens_per_s": tok_s,
           "mean_prompt_tokens": round(stats["prompt_tokens"] / max(1, stats["ok"])),
           "mean_latency_s": round(sum(lat) / max(1, len(lat)), 1),
           "latency_p50_s": lat[len(lat) // 2] if lat else None,
           "latency_p90_s": lat[min(len(lat) - 1, int(0.9 * len(lat)))] if lat else None,
           "running_mean": round(sum(run) / len(run), 2) if run else None, "running_max": max(run) if run else None,
           "retracted": retracted, "watchdog_restarts": restarts, "peak_host_used_gib": peak_host,
           "peak_gpu_used_mib": max((s.get("gpu_used_mib", 0.0) for s in samples), default=None),
           "gpu_oom_lines": oom_lines[:5], "metrics_at_start": first, "metrics_at_end": last,
           "rung": rung, "setup_phases": setup_report.get("phases"), "versions": setup_report.get("versions"),
           "ready_after_notebook_start_s": setup_report.get("ready_after_notebook_start_s"),
           "server_argv": setup_report.get("argv"), "setup_attempts": setup_report.get("attempts"),
           "notebook_elapsed_s": round(_time.time() - NOTEBOOK_START_EPOCH)}
ready_s = setup_report.get("ready_after_notebook_start_s")
go = {
    "1_up_within_45_min_on_fp8_kv": ENGINE == "sglang" and ready_s is not None and ready_s <= 2700
                                     and rung.get("kv") == "fp8_e4m3",
    "2_all_seven_checks": bool(REPORT.get("gate_passed")),
    "3_no_errors_no_restarts": stats["errors"] == 0 and restarts == 0,
    "4_tok_s_at_least_1.25x_baseline_and_8_running": tok_s >= round(1.25 * BASELINE_TOK_S, 1)
                                                     and bool(run) and sum(run) / len(run) >= 8,
    "5_retracted_under_1pct": retracted < 0.01 * max(1, stats["ok"]),
    "6_host_ram_150_gib_no_gpu_oom": peak_host is not None and peak_host <= 150 and not oom_lines,
}
REPORT.update(load=summary, samples=samples, go_rule=go, go=all(go.values()),
              baseline={"run": "kvstress-7g75-b2k", "generated_tokens_per_s": BASELINE_TOK_S, "ok": 421,
                        "running_mean": 5.65, "mean_prompt_tokens": 18335, "mean_latency_s": 49.8})
save_report()
print("SGL_STRESS", json.dumps({k: v for k, v in summary.items() if k not in ("metrics_at_start", "metrics_at_end")},
                               default=str), flush=True)
print("SGL_STRESS_GO", str(all(go.values())).lower(), json.dumps(go), flush=True)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--kv-gib", type=float, default=None, help="vLLM KV cache in GiB (required with --engine vllm)")
    ap.add_argument("--minutes", type=int, default=12)
    ap.add_argument("--prefix-caching", action="store_true", help="TAAF_VLLM_ENABLE_PREFIX_CACHING=1 (MTP kept)")
    ap.add_argument("--batched-tokens", type=int, default=None,
                    help="TAAF_VLLM_MAX_NUM_BATCHED_TOKENS (base 8192); smaller prefill chunks use less activation memory")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="extra serving env for the launcher, e.g. TAAF_VLLM_MOE_BACKEND=flashinfer_b12x")
    ap.add_argument("--engine", choices=["vllm", "sglang"], default="vllm")
    ap.add_argument("--sglang-running", type=int, default=None, help="SGLang --max-running-requests R (default 12)")
    ap.add_argument("--sglang-hicache-gb", type=int, default=0, help="SGLang hierarchical host cache in GB (default off)")
    ap.add_argument("--sglang-mamba-cache", type=int, default=None, help="SGLang --max-mamba-cache-size (default 6R)")
    args = ap.parse_args()
    if args.engine == "vllm" and args.kv_gib is None:
        ap.error("--kv-gib is required with --engine vllm")
    sgl_profile = sgl.resolve_profile({k: v for k, v in (("running", args.sglang_running),
                                                         ("hicache_gb", args.sglang_hicache_gb),
                                                         ("mamba_cache", args.sglang_mamba_cache)) if v})
    nb = json.loads(BASE.read_text())
    cells = nb["cells"][:11]
    kv_bytes = int(args.kv_gib * 1024**3) if args.kv_gib is not None else None
    hits = swaps = 0
    for cell in cells:
        s = "".join(cell["source"])
        if KV_ANCHOR in s:
            if kv_bytes is not None:  # with --engine sglang the vLLM profile matters only for the fallback
                s = s.replace(KV_ANCHOR, f'"TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "{kv_bytes}"')
            if args.prefix_caching:
                assert '"TAAF_VLLM_ENABLE_PREFIX_CACHING": "0"' in s
                s = s.replace('"TAAF_VLLM_ENABLE_PREFIX_CACHING": "0"', '"TAAF_VLLM_ENABLE_PREFIX_CACHING": "1"')
            if args.batched_tokens:
                assert '"TAAF_VLLM_MAX_NUM_BATCHED_TOKENS": "8192"' in s
                s = s.replace('"TAAF_VLLM_MAX_NUM_BATCHED_TOKENS": "8192"',
                              f'"TAAF_VLLM_MAX_NUM_BATCHED_TOKENS": "{args.batched_tokens}"')
            for kv in args.env:
                key, value = kv.split("=", 1)
                if f'"{key}": ' in s:
                    s = re.sub(rf'"{re.escape(key)}": "[^"]*"', f'"{key}": "{value}"', s)
                else:
                    assert ENV_END in s
                    s = s.replace(ENV_END, f'"TAAF_VLLM_OMP_THREADS": "1",\n    "{key}": "{value}"\n}}')
            s = s.replace("PUBLIC25_VLLM_PROFILE_NAME = 'kv5-bf16-mtp3-c8-cg32'",
                          f"PUBLIC25_VLLM_PROFILE_NAME = '{args.slug}'")
            cell["source"] = [s]
            hits += 1
        if args.engine == "sglang" and BUNDLE_ANCHOR in s:
            cell["source"] = [s.replace(BUNDLE_ANCHOR, BUNDLE_ANCHOR + sgl.notebook_swap(sgl_profile))]
            swaps += 1
    if hits != 1:
        raise SystemExit("KV anchor not found exactly once")
    if args.engine == "sglang":
        if swaps != 1:
            raise SystemExit("bundle anchor not found exactly once")
        build_sglang(args, nb, cells, sgl_profile)
        return
    cells[0]["source"] = [f"# {args.slug}: serving stress test (team scottmahony)\n\nFlash-Next NVFP4 vLLM server "
                          f"(Keith Tyser's bundle) with a {args.kv_gib} GiB KV cache"
                          f"{' and prefix caching on' if args.prefix_caching else ''}"
                          f"{f' and {args.batched_tokens}-token prefill chunks' if args.batched_tokens else ''}"
                          f"{f' and {args.env}' if args.env else ''} under a synthetic Duck-shaped "
                          f"load for {args.minutes} minutes; no games are played. Built by scripts/build_kv_stress_nb.py."]
    code = lambda src: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [src]}  # noqa: E731
    cells += [code(STRESS.replace("__MINUTES__", str(args.minutes))), code(TEARDOWN)]
    nb["cells"] = cells
    nb.setdefault("metadata", {})["kaggle"] = {"accelerator": "nvidiaRtxPro6000", "isInternetEnabled": False,
                                               "isGpuEnabled": True, "language": "python", "sourceType": "notebook"}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.slug}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {"id": f"scottmahony/{args.slug}", "title": args.slug.replace("-", " "), "code_file": f"{args.slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": False, "keywords": [], "kernel_sources": [],
            "dataset_sources": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1",
                                "jakobbrggen/taaf-kaggle-source-anim-20260807-anim"],
            "competition_sources": ["arc-prize-2026-arc-agi-3"],
            "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"]}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"built {out / (args.slug + '.ipynb')} with KV {kv_bytes} bytes, {args.minutes} min")


def build_sglang(args: argparse.Namespace, nb: dict, cells: list, profile: dict) -> None:
    """The SGLang stress notebook: the setup cells (bundle swapped), the functional gate, the load, the teardown."""
    def code(src: str) -> dict:
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [src]}

    cells[0]["source"] = [
        f"# {args.slug}: SGLang serving stress test (team scottmahony)\n\nSGLang 0.5.20 (public wheel datasets, "
        "installed offline by scripts/sglang_serving.py) serving RadixArk's Flash-Next NVFP4 checkpoint with an FP8 KV "
        f"cache, R={profile['running']}, mamba cache {profile['mamba_cache']}, host cache {profile['hicache_gb']} GB; "
        "Keith Tyser's vLLM bundle is the fallback. The functional gate and go/no-go of "
        f"docs/research/sglang-serving-plan.md sections 5-6, then the {args.minutes}-minute 28-client Duck-shaped load "
        "of the vLLM stress tests; no games are played. Built by scripts/build_kv_stress_nb.py --engine sglang."]
    idx = next(i for i, c in enumerate(cells) if BUNDLE_ANCHOR in "".join(c["source"]))
    source = (ROOT / "scripts" / "sglang_serving.py").read_text()
    cells.insert(idx, code(sgl.SOURCE_CELL_HEADER + f"_SGLANG_SERVING_SOURCE = {source!r}\n"))
    gate = (SGL_GATE.replace("__RUNNING__", str(profile["running"])).replace("__SLUG__", repr(args.slug))
            .replace("__PROFILE__", repr(profile)))
    load = SGL_LOAD.replace("__MINUTES__", str(args.minutes)).replace("__BASELINE__", repr(BASELINE_TOK_S))
    cells += [code("# ours: source of scripts/taaf_ours_patch.py, for the gate's P30/P30b harness checks\n"
                   f"_OURS_PATCH_SOURCE = {PATCH_SRC.read_text()!r}\n"),
              code(gate), code(load), code(TEARDOWN)]
    nb["cells"] = cells
    nb.setdefault("metadata", {})["kaggle"] = {"accelerator": "nvidiaRtxPro6000", "isInternetEnabled": False,
                                               "isGpuEnabled": True, "language": "python", "sourceType": "notebook"}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.slug}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {"id": f"scottmahony/{args.slug}", "title": args.slug.replace("-", " "), "code_file": f"{args.slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": False, "keywords": [], "kernel_sources": [],
            "dataset_sources": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1",
                                "jakobbrggen/taaf-kaggle-source-anim-20260807-anim", *sgl.EXTRA_DATASET_SOURCES],
            "competition_sources": ["arc-prize-2026-arc-agi-3"],
            "model_sources": [sgl.MODEL_SOURCE]}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"built {out / (args.slug + '.ipynb')}: SGLang profile {profile}, {args.minutes} min")


if __name__ == "__main__":
    main()
