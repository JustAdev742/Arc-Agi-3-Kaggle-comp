#!/usr/bin/env python
"""Fidelity probe: replay logged ARC agent requests through an OpenAI-compatible server, greedy, with logprobs.

Runs in place of the benchmark in our probe notebooks (scripts/build_franzen_nb.py --probe, which writes this file to
/kaggle/arc3-fidelity-probe.py and loads it into the notebook's kernel), and locally against any server:

    python -I scripts/fidelity_probe.py --data requests.jsonl --base-url http://127.0.0.1:8001 --model flashnext \
        --out fidelity.json [--arm base] [--passes seq,conc] [--concurrency 8] [--max-tokens 192] [--top-logprobs 5]

The data is scripts/fidelity_sample.py's ``requests.jsonl``: each line holds one logged request (messages exactly as
the harness sent them, images included, its tools, tool_choice and chat_template_kwargs). Every request is sent to
``POST {base}/v1/chat/completions`` the way the harness sends it (inference/agent/tool_agent.py ``_chat_completion``:
same model name, messages, tools, tool_choice, top_p 0.95, top_k 20, ``chat_template_kwargs`` = ``{"enable_thinking":
true}`` + the logged ones, no ``separate_reasoning``, not streamed) except: ``temperature`` 0 (greedy; SGLang then
samples with top_k 1 and returns the plain log-softmax of the logits), ``max_tokens`` 192, ``logprobs`` with
``top_logprobs`` 5, and, when the server accepts them, ``return_token_ids`` + ``return_meta_info`` (Pennyroyal SGLang:
exact token ids for every logprob, and per-request speculative accept counts).

Order: wait for ``/health`` (raise if the server process dies or the deadline passes), record ``/server_info`` (and
refuse a server whose speculative acceptance thresholds are not 1.0, or whose expert count is not the arm's), one
short warm-up request (which also picks the response extras the server accepts), then each pass after flushing the
prefix cache: ``seq`` one request at a time, ``conc`` the same requests ``--concurrency`` in flight, in the same
order. Results are written to ``--out`` after each pass (a crash keeps the finished passes); the run raises at the end
if a pass produced no usable response. scripts/fidelity_compare.py reads the output.

Stdlib only: the notebook kernel imports it, and ``python -I`` runs it.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

PROBE_VERSION = 1
CONTROL_KEY = "_arc3_control"  # stripped by the harness before sending; the sampler already did, this is a guard
HARNESS_SAMPLING = {"top_p": 0.95, "top_k": 20}  # Franzen's cell 4; irrelevant under greedy, sent as the harness does
# Response extras, tried in order by the warm-up: Pennyroyal SGLang's ChatCompletionRequest has both fields
# (entrypoints/openai/protocol.py); a server that rejects one falls back to plain OpenAI logprobs.
EXTRAS_LADDER = ({"return_token_ids": True, "return_meta_info": True}, {"return_token_ids": True}, {})
SPEC_KEYS = ("spec_accept_length", "spec_verify_ct", "spec_num_correct_drafts", "spec_num_proposed_drafts")
SERVER_KEYS = (
    "version", "model_path", "served_model_name", "json_model_override_args", "quantization", "dtype",
    "kv_cache_dtype", "context_length", "mem_fraction_static", "max_running_requests", "max_total_num_tokens",
    "chunked_prefill_size", "max_prefill_tokens", "page_size", "schedule_policy", "disable_radix_cache",
    "max_mamba_cache_size", "mamba_radix_cache_strategy", "cuda_graph_bs_decode", "attention_backend",
    "sampling_backend", "enable_deterministic_inference", "random_seed", "reasoning_parser", "tool_call_parser",
    "default_chat_template_kwargs", "speculative_algorithm", "speculative_num_steps", "speculative_eagle_topk",
    "speculative_num_draft_tokens", "speculative_accept_threshold_single", "speculative_accept_threshold_acc",
    "speculative_draft_model_override_args", "speculative_token_map",
)
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # localhost: never through a proxy


def log(message: str) -> None:
    print(f"[probe {time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mount_alternatives(path: str) -> list[str]:
    """PATH and its other Kaggle mount layout (lesson 0030): /kaggle/input/datasets/<owner>/<slug>/... is also
    mounted as /kaggle/input/<slug>/..., and the reverse cannot be derived (the owner is not in the short form)."""
    parts = str(path).split("/")  # '', 'kaggle', 'input', 'datasets', owner, slug, ...
    if len(parts) > 5 and parts[:4] == ["", "kaggle", "input", "datasets"]:
        return [str(path), "/".join(parts[:3] + parts[5:])]
    return [str(path)]


def find_dataset(candidates: list, files: dict[str, str], wait_s: float = 120.0, poll_s: float = 5.0,
                 logger=log) -> list[Path]:
    """The data files from the first candidate directory holding all of them, each checked against its sha256.

    Each candidate is tried in both Kaggle mount layouts. Waits up to ``wait_s`` for a mount; raises (listing
    /kaggle/input) when none appears or a hash differs."""
    candidates = [alt for c in candidates for alt in mount_alternatives(str(c))]
    t0 = time.time()
    while True:
        for candidate in candidates:
            folder = Path(candidate)
            if all((folder / name).is_file() for name in files):
                for name, expected in files.items():
                    got = sha256_file(folder / name)
                    if got != expected:
                        raise RuntimeError(f"probe data {folder / name}: sha256 {got[:16]} is not the manifest's "
                                           f"{expected[:16]} (a different dataset version?)")
                logger(f"data: {folder} ({', '.join(files)}), sha256 verified")
                return [folder / name for name in files]
        if time.time() - t0 >= wait_s:
            listing = []
            root = Path("/kaggle/input")
            if root.is_dir():
                for path, dirs, _ in os.walk(root):
                    if path.count(os.sep) <= 4:
                        listing.append(f"{path} {sorted(dirs)[:8]}")
            raise RuntimeError(f"probe data {sorted(files)} not found under {[str(c) for c in candidates]} after "
                               f"{wait_s:g} s; " + "; ".join(listing[:20]))
        time.sleep(poll_s)


def load_samples(paths: list) -> list[dict]:
    samples = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    if not isinstance(sample, dict) or "id" not in sample or "request" not in sample:
                        raise ValueError(f"{path}: not a fidelity sample line")
                    samples.append(sample)
    ids = [s["id"] for s in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample ids")
    return samples


def build_body(sample: dict, *, model: str, max_tokens: int, top_logprobs: int, extras: dict) -> dict:
    """The harness's request for this sample, made greedy and short, with logprobs."""
    request = sample["request"]
    messages = [{k: v for k, v in m.items() if k != CONTROL_KEY} for m in request["messages"]]
    body: dict = {"model": model, "messages": messages, "stream": False, "temperature": 0.0, **HARNESS_SAMPLING,
                  "max_tokens": int(max_tokens),
                  "chat_template_kwargs": {"enable_thinking": True, **(request.get("chat_template_kwargs") or {})},
                  "logprobs": True, "top_logprobs": int(top_logprobs)}
    if request.get("tools"):
        body["tools"] = request["tools"]
        if request.get("tool_choice"):
            body["tool_choice"] = request["tool_choice"]
    body.update(extras)
    return body


def _http(url: str, data: bytes | None = None, timeout: float = 30.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        with _OPENER.open(req, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def wait_healthy(base: str, *, deadline: float, alive=None, on_failure=None, poll_s: float = 5.0,
                 logger=log) -> float:
    """Seconds waited until ``GET {base}/health`` answered 200. Raises when ``alive()`` turns False or at ``deadline``
    (epoch seconds), after calling ``on_failure()`` (the notebook prints serve.log's tail)."""
    t0 = last = time.time()
    while True:
        if alive is not None and not alive():
            if on_failure is not None:
                on_failure()
            raise RuntimeError("the model server exited before it became healthy (serve.log has the reason)")
        try:
            status, _ = _http(base + "/health", timeout=5)
            if status == 200:
                logger(f"server healthy ({time.time() - t0:.0f} s of waiting here)")
                return time.time() - t0
        except (OSError, http.client.HTTPException):
            pass
        if time.time() >= deadline:
            if on_failure is not None:
                on_failure()
            raise RuntimeError(f"the model server was not healthy by the deadline ({time.time() - t0:.0f} s of "
                               "waiting here); stopping instead of spending the session")
        if time.time() - last >= 60:
            logger(f"waiting for {base}/health ({time.time() - t0:.0f} s)")
            last = time.time()
        time.sleep(poll_s)


def server_info(base: str) -> dict:
    try:
        status, raw = _http(base + "/server_info", timeout=60)
    except (OSError, http.client.HTTPException) as exc:
        return {"error": repr(exc)}
    if status != 200:
        return {"error": f"HTTP {status}: {raw[:200].decode(errors='replace')}"}
    try:
        info = json.loads(raw)
    except ValueError:
        return {"error": "not JSON"}
    return {k: info[k] for k in SERVER_KEYS if k in info}


def num_experts_override(info: dict) -> int | None:
    """``text_config.num_experts`` from the server's ``json_model_override_args`` (None when not overridden)."""
    raw = info.get("json_model_override_args")
    if not raw:
        return None
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return None
    value = (args.get("text_config") or {}).get("num_experts") if isinstance(args, dict) else None
    return int(value) if value is not None else None


def check_server(info: dict, expect_num_experts: int | None) -> list[str]:
    """Refuse a server that is not the arm this notebook was built as. Missing keys are noted, not fatal."""
    notes = []
    for key in ("speculative_accept_threshold_single", "speculative_accept_threshold_acc"):
        if key not in info:
            notes.append(f"{key} not reported")
        elif float(info[key]) != 1.0:
            raise RuntimeError(f"{key} = {info[key]}: the probe needs lossless speculative decoding (1.0)")
    if "json_model_override_args" not in info:
        notes.append("json_model_override_args not reported; the arm is not verified")
    elif num_experts_override(info) != expect_num_experts:
        raise RuntimeError(f"server json_model_override_args {info['json_model_override_args']!r} does not match this "
                           f"arm (expected num_experts override {expect_num_experts})")
    return notes


def parse_response(obj: dict) -> dict:
    """Tokens, ids, logprobs, top-k, texts, usage and speculative counts from one chat completion."""
    choice = (obj.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = (choice.get("logprobs") or {}).get("content") or []
    rec = {
        "finish_reason": choice.get("finish_reason"), "matched_stop": choice.get("matched_stop"),
        "reasoning": message.get("reasoning_content"), "content": message.get("content"),
        "tool_calls": message.get("tool_calls"), "usage": obj.get("usage"),
        "tokens": [e.get("token") for e in content],
        "logprobs": [e.get("logprob") for e in content],
        "top": [[[None, t.get("token"), t.get("logprob")] for t in (e.get("top_logprobs") or [])] for e in content],
        "token_ids": None,
    }
    meta = choice.get("meta_info") or {}
    out = meta.get("output_token_logprobs")
    if out and len(out) == len(content):  # [(logprob, token id, text)], the source of the OpenAI fields
        rec["token_ids"] = [int(x[1]) for x in out]
        tops = meta.get("output_top_logprobs")
        if tops and len(tops) == len(out):
            rec["top"] = [[[int(t[1]), t[2], t[0]] for t in (row or [])] for row in tops]
    elif choice.get("response_token_ids") is not None and len(choice["response_token_ids"]) == len(content):
        rec["token_ids"] = [int(x) for x in choice["response_token_ids"]]
    spec = {k: meta[k] for k in SPEC_KEYS if k in meta}
    rec["spec"] = spec or None
    rec["server_e2e_s"] = meta.get("e2e_latency")
    details = (obj.get("usage") or {}).get("prompt_tokens_details") or {}
    rec["cached_tokens"] = details.get("cached_tokens")
    return rec


def call(base: str, data: bytes, timeout: float) -> dict:
    t0 = time.time()
    try:
        status, raw = _http(base + "/v1/chat/completions", data, timeout=timeout)
    except Exception as exc:  # connection refused/reset, timeout: recorded, the pass goes on
        return {"ok": False, "status": None, "error": repr(exc)[:500], "latency_s": round(time.time() - t0, 3)}
    rec: dict = {"status": status, "latency_s": round(time.time() - t0, 3)}
    if status != 200:
        rec.update(ok=False, error=raw[:500].decode(errors="replace"))
        return rec
    try:
        rec.update(parse_response(json.loads(raw)))
    except (ValueError, TypeError, AttributeError, IndexError) as exc:
        rec.update(ok=False, error=f"unparseable response: {exc!r}"[:500])
        return rec
    rec["ok"] = bool(rec["tokens"]) and all(isinstance(x, (int, float)) for x in rec["logprobs"] if x is not None)
    if not rec["tokens"]:
        rec["error"] = "no logprobs in the response"
    return rec


def flush_cache(base: str, attempts: int = 15, logger=log) -> bool:
    """Empty the prefix cache (SGLang refuses while requests run; retried)."""
    for _ in range(attempts):
        try:
            status, raw = _http(base + "/flush_cache", timeout=60)
        except (OSError, http.client.HTTPException) as exc:
            status, raw = None, repr(exc).encode()
        if status == 200:
            return True
        time.sleep(2)
    logger(f"prefix cache NOT flushed ({status}: {raw[:120]!r})")
    return False


def choose_extras(base: str, sample: dict, *, model: str, top_logprobs: int, timeout: float, retries: int = 2,
                  retry_s: float = 10.0, logger=log) -> tuple[dict, list[dict]]:
    """Warm-up: the first ``EXTRAS_LADDER`` entry the server answers with logprobs (8 tokens of the first sample).

    A 4xx moves down the ladder at once; a connection error or a 5xx is retried ``retries`` times first."""
    tried = []
    for extras in EXTRAS_LADDER:
        body = json.dumps(build_body(sample, model=model, max_tokens=8, top_logprobs=top_logprobs, extras=extras))
        for attempt in range(retries + 1):  # a connection error or a 5xx may be transient: retry the same extras
            rec = call(base, body.encode(), timeout)
            tried.append({"extras": extras, **{k: rec.get(k) for k in ("ok", "status", "error", "latency_s")},
                          "token_ids": bool(rec.get("token_ids")), "spec": rec.get("spec")})
            if rec.get("ok"):
                logger(f"warm-up ok in {rec['latency_s']:.1f} s with extras {sorted(extras) or 'none'} "
                       f"(token ids: {bool(rec.get('token_ids'))})")
                return extras, tried
            logger(f"warm-up with extras {sorted(extras) or 'none'} failed: {rec.get('status')} {rec.get('error')}")
            if (rec.get("status") is not None and rec["status"] < 500) or attempt == retries:
                break  # a 4xx (or a 200 without logprobs) is an answer about these extras
            time.sleep(retry_s)
    raise RuntimeError("the server returned no logprobs for the warm-up request: " + json.dumps(tried)[:1000])


def replay(base: str, samples: list[dict], *, body_for, concurrency: int, timeout: float, alive=None,
           deadline: float | None = None, logger=log, label: str = "") -> tuple[list[dict], int]:
    """One record per sample, in sample order; ``concurrency`` requests in flight. Returns (records, max in flight)."""
    records: list = [None] * len(samples)
    lock = threading.Lock()
    state = {"in_flight": 0, "max_in_flight": 0, "done": 0}
    t0 = time.time()

    def one(i: int) -> dict:
        rec: dict = {"id": samples[i]["id"]}
        if deadline is not None and time.time() > deadline:
            return {**rec, "ok": False, "error": "skipped: the probe's time budget is spent"}
        if alive is not None and not alive():
            return {**rec, "ok": False, "error": "skipped: the server is not running"}
        data = json.dumps(body_for(samples[i])).encode()
        with lock:
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        rec["t_submit"] = round(time.time() - t0, 3)
        try:
            rec.update(call(base, data, timeout))
        finally:
            with lock:
                state["in_flight"] -= 1
        return rec

    def progress(rec: dict) -> None:
        state["done"] += 1
        if not rec.get("ok"):
            logger(f"{label} {rec['id']}: FAILED {rec.get('status')} {str(rec.get('error'))[:200]}")
        if state["done"] % 20 == 0 or state["done"] == len(samples):
            logger(f"{label}: {state['done']}/{len(samples)} done, {time.time() - t0:.0f} s")

    if concurrency <= 1:
        for i in range(len(samples)):
            records[i] = one(i)
            progress(records[i])
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(one, i): i for i in range(len(samples))}
            for future in concurrent.futures.as_completed(futures):
                records[futures[future]] = future.result()
                progress(records[futures[future]])
    return records, state["max_in_flight"]


def _gpu() -> str | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _write(out: Path, result: dict) -> None:
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=True))
    tmp.replace(out)


def _pass_summary(records: list[dict]) -> dict:
    ok = [r for r in records if r and r.get("ok")]
    gen = sum(len(r["tokens"]) for r in ok)
    accept = [r["spec"]["spec_accept_length"] for r in ok if r.get("spec") and "spec_accept_length" in r["spec"]]
    lat = sorted(r["latency_s"] for r in ok)
    return {"ok": len(ok), "failed": len(records) - len(ok), "generated_tokens": gen,
            "median_latency_s": lat[len(lat) // 2] if lat else None,
            "mean_accept_length": round(sum(accept) / len(accept), 3) if accept else None}


def run(files: list, out, *, base_url: str, model: str, arm: str, build: dict | None = None,
        passes: tuple = ("seq", "conc"), concurrency: int = 8, max_tokens: int = 192, top_logprobs: int = 5,
        timeout: float = 900.0, alive=None, on_failure=None, health_deadline: float | None = None,
        max_minutes: float = 150.0, expect_num_experts: int | None = None, flush: bool = True,
        health_poll_s: float = 5.0, logger=log) -> dict:
    """The probe (see the module docstring). Writes ``out`` after each pass; returns the result."""
    t_start = time.time()
    out = Path(out)
    base = base_url.rstrip("/")
    samples = load_samples(files)
    if not samples:
        raise RuntimeError("no samples")
    result: dict = {
        "probe": PROBE_VERSION, "arm": arm, "build": build or {}, "model": model,
        "params": {"passes": list(passes), "concurrency": concurrency, "max_tokens": max_tokens,
                   "top_logprobs": top_logprobs, "temperature": 0.0, "timeout_s": timeout, "flush": flush,
                   "files": {Path(p).name: sha256_file(p) for p in files}},
        "samples": [{"id": s["id"], "game": s.get("game"), **(s.get("stats") or {}), **(s.get("stratum") or {})}
                    for s in samples],
        "timing": {"started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t_start))},
        "gpu": {"start": _gpu()}, "passes": {},
    }
    logger(f"{len(samples)} requests, arm {arm}, passes {list(passes)}, {concurrency} in flight for conc")
    result["timing"]["waited_for_health_s"] = round(wait_healthy(
        base, deadline=health_deadline if health_deadline is not None else time.time() + 1800, alive=alive,
        on_failure=on_failure, poll_s=health_poll_s, logger=logger), 1)
    info = server_info(base)
    result["server_info"] = info
    result["server_notes"] = check_server(info, expect_num_experts)
    for note in result["server_notes"]:
        logger(f"note: {note}")
    extras, result["warmup"] = choose_extras(base, samples[0], model=model, top_logprobs=top_logprobs,
                                             timeout=timeout, logger=logger)
    result["params"]["extras"] = extras
    deadline = t_start + max_minutes * 60

    def body_for(sample: dict) -> dict:
        return build_body(sample, model=model, max_tokens=max_tokens, top_logprobs=top_logprobs, extras=extras)

    for name in passes:
        width = 1 if name == "seq" else concurrency
        flushed = flush_cache(base, logger=logger) if flush else None
        t0 = time.time()
        logger(f"pass {name}: {len(samples)} requests, {width} in flight, prefix cache flushed: {flushed}")
        records, max_in_flight = replay(base, samples, body_for=body_for, concurrency=width, timeout=timeout,
                                        alive=alive, deadline=deadline, logger=logger, label=name)
        summary = _pass_summary(records)
        result["passes"][name] = {"concurrency": width, "max_in_flight": max_in_flight, "flushed": flushed,
                                  "wall_s": round(time.time() - t0, 1), **summary, "records": records}
        result["timing"][f"{name}_wall_s"] = round(time.time() - t0, 1)
        _write(out, result)
        logger(f"pass {name}: {summary['ok']} ok, {summary['failed']} failed, {summary['generated_tokens']} tokens "
               f"in {time.time() - t0:.0f} s, median latency {summary['median_latency_s']} s, mean accept length "
               f"{summary['mean_accept_length']}; written to {out}")
        if alive is not None and not alive():
            if on_failure is not None:
                on_failure()
            raise RuntimeError(f"the model server died during pass {name}; finished passes are in {out}")
    result["gpu"]["end"] = _gpu()
    result["timing"]["total_s"] = round(time.time() - t_start, 1)
    if "seq" in result["passes"] and "conc" in result["passes"]:
        a = {r["id"]: r for r in result["passes"]["seq"]["records"] if r and r.get("ok")}
        b = {r["id"]: r for r in result["passes"]["conc"]["records"] if r and r.get("ok")}
        both = sorted(set(a) & set(b))
        key = "token_ids" if all(a[i].get("token_ids") and b[i].get("token_ids") for i in both) else "tokens"
        same = sum(1 for i in both if a[i][key] == b[i][key])
        result["within_run_identical"] = {"compared": len(both), "identical": same, "by": key}
        logger(f"seq vs conc: {same} of {len(both)} greedy outputs identical (by {key})")
    _write(out, result)
    empty = [name for name in passes if not result["passes"][name]["ok"]]
    if empty:
        raise RuntimeError(f"no usable response in pass(es) {empty}; see {out}")
    logger(f"done in {result['timing']['total_s']:.0f} s -> {out}")
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, type=Path, action="append", help="requests.jsonl (repeatable)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8001", help="server root (without /v1)")
    ap.add_argument("--model", default="flashnext")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--arm", default="local")
    ap.add_argument("--passes", default="seq,conc")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--top-logprobs", type=int, default=5)
    ap.add_argument("--health-wait", type=float, default=60.0, help="seconds to wait for /health")
    ap.add_argument("--expect-num-experts", type=int, default=None)
    args = ap.parse_args(argv)
    run(args.data, args.out, base_url=args.base_url, model=args.model, arm=args.arm,
        passes=tuple(p for p in args.passes.split(",") if p), concurrency=args.concurrency,
        max_tokens=args.max_tokens, top_logprobs=args.top_logprobs, health_deadline=time.time() + args.health_wait,
        expect_num_experts=args.expect_num_experts)


if __name__ == "__main__":
    main()
