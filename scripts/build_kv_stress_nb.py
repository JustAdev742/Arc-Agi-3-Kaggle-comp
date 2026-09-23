#!/usr/bin/env python
"""Build a short serving stress notebook for the Flash-Next vLLM server at a chosen KV-cache size.

    .venv/bin/python scripts/build_kv_stress_nb.py --out <dir> --slug arc3-kv-stress-8g --kv-gib 8 [--minutes 12]

It keeps the setup cells of kaggle/taaf/base-thui-animfast.ipynb (serving setup, analyzer env, solver import checks)
and replaces the benchmark with a synthetic load shaped like the Duck's: the Duck's own system prompt, user turns padded
to about 20k prompt tokens with a 256x256 board image, thinking on, 28 concurrent clients, for ``--minutes``. It
records completions, errors, generated tokens and, every 10 s, the server's running/waiting counts, KV usage and
preemptions, then writes /kaggle/working/kv_stress.json and tears the server down. Use it to check that a larger
``TAAF_VLLM_KV_CACHE_MEMORY_BYTES`` neither runs out of GPU memory nor slows decoding before a full run uses it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kaggle" / "taaf" / "base-thui-animfast.ipynb"
KV_ANCHOR = '"TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "5368709120"'

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
                "vllm:gpu_cache_usage_perc", "vllm:num_preemptions_total", "vllm:generation_tokens_total"):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--kv-gib", type=float, required=True)
    ap.add_argument("--minutes", type=int, default=12)
    args = ap.parse_args()
    nb = json.loads(BASE.read_text())
    cells = nb["cells"][:11]
    kv_bytes = int(args.kv_gib * 1024**3)
    hits = 0
    for cell in cells:
        s = "".join(cell["source"])
        if KV_ANCHOR in s:
            s = s.replace(KV_ANCHOR, f'"TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "{kv_bytes}"')
            s = s.replace("PUBLIC25_VLLM_PROFILE_NAME = 'kv5-bf16-mtp3-c8-cg32'",
                          f"PUBLIC25_VLLM_PROFILE_NAME = '{args.slug}'")
            cell["source"] = [s]
            hits += 1
    if hits != 1:
        raise SystemExit("KV anchor not found exactly once")
    cells[0]["source"] = [f"# {args.slug}: serving stress test (team scottmahony)\n\nFlash-Next NVFP4 vLLM server "
                          f"(Keith Tyser's bundle) with a {args.kv_gib} GiB KV cache under a synthetic Duck-shaped "
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


if __name__ == "__main__":
    main()
