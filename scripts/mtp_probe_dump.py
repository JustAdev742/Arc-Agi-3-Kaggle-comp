#!/usr/bin/env python
"""Dump the target's hidden states over held-out probe requests plus the greedy output a reference run recorded,
for the MTP replica check (docs/research/beat-tufa/mtp-drafter-finetune.md 5.2 and 10; scripts/mtp_replica.py check).

    python -I scripts/mtp_probe_dump.py --data requests.jsonl --reference fidelity.json --dry-run
    python -I scripts/mtp_probe_dump.py --data requests.jsonl --reference fidelity.json --out OUT --dump-dir DUMP \\
        [--base-url http://127.0.0.1:8001] [--model flashnext] [--count 16] [--max-rows 400000] [--ids a#1,b#2]

``--data`` is the fidelity probe's ``requests.jsonl`` (scripts/fidelity_sample.py; dataset
scottmahony/arc3-fidelity-prompts) and ``--reference`` the ``fidelity.json`` of a greedy, lossless probe run on the
serving configuration to be mirrored (runs/fidelity-base: D' with the albucino draft and Pennyroyal's
hot_tokens_64k.pt). Its ``--pass`` (default ``seq``: one request at a time) gives, per request, the greedy output ids
and SGLang's ``spec_accept_length``, ``spec_verify_ct`` and ``spec_num_correct_drafts`` (and
``spec_correct_drafts_histogram`` when the probe kept it).

**Why a continuation.** SGLang returns no draft proposals, only those per-request counts, so the replica check replays
the chain along the same greedy path, which needs the target's ``H`` at every prompt row (draft KV and QSA keys) and at
every output row (the chain's inputs). The dump patch writes only prefill (EXTEND) batches, so each request is sent as
its logged messages plus a final assistant message holding the decoded output (``/v1/detokenize``, special tokens
kept) with ``continue_final_message`` true: SGLang then renders the prompt with its generation prompt and appends the
encoded text (serving_chat.py ``_handle_last_assistant_message``, ``_append_assistant_prefix_to_prompt_ids``), i.e.
prompt ids + output ids, and prefills it with ``max_tokens`` 1. On the 154 outputs of runs/fidelity-base, decoding and
re-encoding after ``<think>\\n`` reproduced every id sequence exactly (served tokenizer, sha256 06b95093);
scripts/mtp_replica.py still compares the dumped output ids with the recorded ones and stops at a mismatch.

**Server** (the dump server of section 9.3 with two changes): ``ARC3_HC_DUMP_KEEP=all`` (every row) and
``ARC3_HC_DUMP_DTYPE=bf16`` (FP8 rows would add ~3% noise to the draft's inputs). BF16 rows take 20 KiB each, so
``--max-rows`` (default 400,000, ~8.2 GB) bounds the selection: one request per held-out game per round, the one
closest to ``--target-tokens`` prompt tokens, until ``--count`` requests or the budget.

Writes OUT/probe-dump.jsonl (per request: rid, id, game, prompt_tokens, the recorded token_ids and spec counts, the
reference's cached_tokens, the server's prompt_tokens and the dump check), OUT/probe-plan.json and a summary.
Stdlib only; loads scripts/fidelity_probe.py and scripts/hc_dump_driver.py from its own folder.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

BF16_ROW_BYTES = 10240 * 2 + 4 + 1 + 20  # a kept BF16 row (+ position, role) and its prompt-token arrays


def _sibling(name: str):
    key = f"arc3_probe_{name}"
    if key in sys.modules:
        return sys.modules[key]
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(key, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} not found (keep it next to {Path(__file__).name})")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


fp = _sibling("fidelity_probe")
dd = _sibling("hc_dump_driver")


def log(message: str) -> None:
    print(f"[probe-dump {time.strftime('%H:%M:%S')}] {message}", flush=True)


def reference_records(reference: dict, pass_name: str = "seq") -> dict:
    """Usable records of one pass by id: answered, with greedy token ids and speculative counts."""
    records = reference["passes"][pass_name]["records"]
    return {r["id"]: r for r in records if r.get("ok") and r.get("token_ids") and r.get("spec")
            and (r.get("usage") or {}).get("prompt_tokens")}


def select(samples: list, records: dict, *, count: int = 16, max_rows: int = 400_000, target_tokens: int = 24_000,
           ids=None, games=None, min_prompt: int = 4096) -> list:
    """The probe samples to dump (module docstring), in a deterministic order."""
    by_id = {s["id"]: s for s in samples}
    if ids:
        missing = [i for i in ids if i not in by_id or i not in records]
        if missing:
            raise ValueError(f"no sample or no usable reference record for {missing}")
        return [by_id[i] for i in ids]
    pool: dict = {}
    for s in samples:
        r = records.get(s["id"])
        if r is None or (games and s["game"] not in games):
            continue
        if r["usage"]["prompt_tokens"] < min_prompt:
            continue
        pool.setdefault(s["game"], []).append(s)
    for game in pool:
        pool[game].sort(key=lambda s: (abs(records[s["id"]]["usage"]["prompt_tokens"] - target_tokens), s["id"]))
    chosen, rows = [], 0
    while len(chosen) < count and any(pool.values()):
        progressed = False
        for game in sorted(pool):
            if len(chosen) >= count or not pool[game]:
                continue
            s = pool[game].pop(0)
            r = records[s["id"]]
            need = r["usage"]["prompt_tokens"] + len(r["token_ids"])
            if rows + need > max_rows:
                continue
            chosen.append(s)
            rows += need
            progressed = True
        if not progressed:
            break
    return chosen


def rid_for(sample_id: str) -> str:
    return "probe-" + sample_id.replace("#", "-")


def continuation_body(sample: dict, text: str, *, model: str, rid: str) -> dict:
    """The logged request plus the output as a final assistant message to continue, greedy, one new token."""
    request = dict(sample["request"])
    request["messages"] = list(request["messages"]) + [{"role": "assistant", "content": text}]
    body = fp.build_body({"request": request}, model=model, max_tokens=1, top_logprobs=1,
                         extras={"rid": rid, "continue_final_message": True})
    return body


def detokenize(base: str, ids: list, timeout: float = 120.0) -> str:
    data = json.dumps({"tokens": [int(i) for i in ids], "skip_special_tokens": False}).encode()
    status, raw = fp._http(base + "/v1/detokenize", data, timeout=timeout)
    if status != 200:
        raise RuntimeError(f"/v1/detokenize: HTTP {status}: {raw[:200]!r}")
    text = json.loads(raw).get("text")
    if not isinstance(text, str):
        raise RuntimeError("/v1/detokenize returned no text")
    return text


def run(samples: list, records: dict, out: Path, *, base_url: str, dump_dir: Path | None, model: str = "flashnext",
        timeout: float = 900.0, health_wait: float = 1800.0, logger=log) -> dict:
    base = base_url.rstrip("/")
    out.mkdir(parents=True, exist_ok=True)
    fp.wait_healthy(base, deadline=time.time() + health_wait, logger=logger)
    info = fp.server_info(base)
    flush_each = info.get("disable_radix_cache") is not True
    tail = dd.IndexTail(dump_dir) if dump_dir else None
    if dump_dir:
        (Path(dump_dir) / "plans").mkdir(parents=True, exist_ok=True)
    summary = {"sent": 0, "ok": 0, "dump_ok": 0, "server_info": info, "dump_dir": str(dump_dir) if dump_dir else None,
               "started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    with open(out / "probe-dump.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            r = records[s["id"]]
            rid = rid_for(s["id"])
            P, n = int(r["usage"]["prompt_tokens"]), len(r["token_ids"])
            text = detokenize(base, r["token_ids"])
            if dump_dir:
                (Path(dump_dir) / "plans" / f"{rid}.json").write_text(json.dumps({"rid": rid, "loss_spans": None}))
            if flush_each:
                fp.flush_cache(base, logger=logger)
            rec = fp.call(base, json.dumps(continuation_body(s, text, model=model, rid=rid)).encode(), timeout)
            usage = rec.get("usage") or {}
            line = {"rid": rid, "id": s["id"], "game": s["game"], "prompt_tokens": P, "out_tokens": n,
                    "token_ids": r["token_ids"], "spec": r.get("spec"), "cached_tokens": r.get("cached_tokens"),
                    "ok": bool(rec.get("ok")), "server_prompt_tokens": usage.get("prompt_tokens"),
                    "error": rec.get("error")}
            summary["sent"] += 1
            summary["ok"] += line["ok"]
            problems = [] if line["ok"] else ["the request failed"]
            if line["ok"] and usage.get("prompt_tokens") != P + n:
                problems.append(f"the server prefilled {usage.get('prompt_tokens')} tokens, expected {P} + {n}")
            if tail is not None and line["ok"]:
                tail.poll()
                n_assistant = sum(1 for m in s["request"]["messages"] if m.get("role") == "assistant")
                more, stats = dd.check_dump(tail.by_rid.get(rid, []), int(usage.get("prompt_tokens") or 0),
                                            n_assistant)
                problems += more
                line["dump"] = stats
            line["dump_ok"] = not problems
            line["problems"] = problems
            summary["dump_ok"] += line["dump_ok"]
            f.write(json.dumps(line) + "\n")
            f.flush()
            logger(f"{rid}: P={P} out={n} -> {usage.get('prompt_tokens')} prompt tokens in {rec.get('latency_s')} s; "
                   + ("ok" if not problems else "BAD: " + "; ".join(problems)))
            if problems and summary["sent"] == 1 and tail is not None:
                summary["stopped"] = "the first request was not dumped completely: " + "; ".join(problems)
                break
    summary["finished_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    (out / "probe-dump-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, required=True, help="the fidelity probe's requests.jsonl")
    ap.add_argument("--reference", type=Path, required=True, help="fidelity.json of the reference probe run")
    ap.add_argument("--pass", dest="pass_name", default="seq")
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=400_000)
    ap.add_argument("--target-tokens", type=int, default=24_000)
    ap.add_argument("--min-prompt", type=int, default=4096, help="skip shorter prompts (QSA is dense up to 2,048)")
    ap.add_argument("--ids", default=None, help="comma-separated sample ids (overrides the selection)")
    ap.add_argument("--games", default=None, help="comma-separated games to draw from")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dump-dir", type=Path, default=None, help="the server's ARC3_HC_DUMP (plans + checks)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="flashnext")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--health-wait", type=float, default=1800.0)
    args = ap.parse_args(argv)
    samples = fp.load_samples([args.data])
    records = reference_records(json.loads(args.reference.read_text()), args.pass_name)
    chosen = select(samples, records, count=args.count, max_rows=args.max_rows, target_tokens=args.target_tokens,
                    ids=[i for i in args.ids.split(",") if i] if args.ids else None,
                    games={g for g in args.games.split(",") if g} if args.games else None, min_prompt=args.min_prompt)
    rows = sum(records[s["id"]]["usage"]["prompt_tokens"] + len(records[s["id"]]["token_ids"]) for s in chosen)
    plan = {"selected": [{"id": s["id"], "game": s["game"], "rid": rid_for(s["id"]),
                          "prompt_tokens": records[s["id"]]["usage"]["prompt_tokens"],
                          "out_tokens": len(records[s["id"]]["token_ids"]),
                          "spec_accept_length": records[s["id"]]["spec"].get("spec_accept_length")} for s in chosen],
            "rows": rows, "dump_gb_bf16_est": round(rows * BF16_ROW_BYTES / 1e9, 2),
            "reference": str(args.reference), "pass": args.pass_name}
    for p in plan["selected"]:
        print(f"{p['id']:>10} {p['game']} P={p['prompt_tokens']:>7,} out={p['out_tokens']} "
              f"sglang accept {p['spec_accept_length']:.3f}")
    print(f"{len(chosen)} requests, {rows:,} rows, ~{plan['dump_gb_bf16_est']} GB of BF16 rows", flush=True)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "probe-plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    if args.dry_run:
        return 0
    if args.out is None:
        ap.error("--out is required unless --dry-run")
    if not chosen:
        print("mtp_probe_dump: nothing selected", file=sys.stderr)
        return 1
    summary = run(chosen, records, args.out, base_url=args.base_url, dump_dir=args.dump_dir, model=args.model,
                  timeout=args.timeout, health_wait=args.health_wait)
    log(f"done: {summary['ok']} of {summary['sent']} ok, {summary['dump_ok']} dumped completely")
    return 0 if summary["dump_ok"] == summary["sent"] and summary["sent"] else 1


if __name__ == "__main__":
    sys.exit(main())
