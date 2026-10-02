#!/usr/bin/env python
"""Report on one run of Daniel Franzen's notebook (or of scripts/franzen_bed.py): per game and for the server.

Reads what his notebook leaves in /kaggle/working:

- ``benchmark.json`` (TAAF): per game the levels completed, actions per level, the level baselines and TAAF's score;
  the score is recomputed with our scorer (arc3/scoring.py game_score) and both are shown.
- ``*_requests.jsonl`` (``save_request_logs``; the Save & Run sets it, a competition rerun does not): one ``request``
  snapshot per model call and one ``response`` with the server's usage. Gives requests, prompt/generated/cached
  tokens and the prefix-cache hit share per game, and "prefix breaks": requests whose messages do not extend the
  previous request's, i.e. history trims. Each break is where the priority gate hands the game's slot over and
  re-admits it, so gate admissions per game = 1 + prefix breaks. ``requests.jsonl`` holds calls made while only one
  game had a runtime-state file (the harness falls back to the shared name); shown as "(unattributed)".
- ``transcripts/*.txt``: analyzer turns, read timeouts, context-overflow recoveries, yields, calibration notes.
- ``serve.log`` (SGLang): decode throughput (``Decode batch`` lines), running/queued requests, MTP accept length,
  server-side prefix cache share and output tokens from ``ReqTimeStats`` lines, HTTP statuses.
- ``summary.txt`` (TAAF), and optionally the notebook log (``--log``: Kaggle's JSON stream or plain text) for the
  priority gate lines (``priority gate active``, ``tail fade phase``, ``DIAG ... gate:`` counters when
  ARC3_DIAG_CONCURRENCY is on), read timeouts and per-game ``[finished]`` lines.

    .venv/bin/python scripts/franzen_report.py RUN_DIR [--log RUN_DIR/<slug>.log] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arc3.scoring import game_score  # noqa: E402

UNATTRIBUTED = "(unattributed)"
_TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] ")
def _fields(text: str) -> dict:
    """``#running-req: 9, accept len: 3.04, cuda graph: True, ...`` -> {name: number or bool}."""
    out: dict = {}
    for part in text.split(","):
        key, sep, value = part.rpartition(":")
        if not sep:
            continue
        value = value.strip()
        try:
            out[key.strip()] = (value == "True") if value in ("True", "False") else float(value)
        except ValueError:
            continue
    return out


def _message_key(message: dict) -> str:
    return hashlib.sha1(json.dumps(message, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def request_stats(path: Path) -> dict:
    """Counters from one ``*_requests.jsonl`` (request/response snapshots in call order)."""
    out = {"requests": 0, "responses": 0, "prompt_tokens": 0, "generated_tokens": 0, "cached_tokens": 0,
           "reasoning_tokens": 0, "image_tokens": 0, "max_prompt_tokens": 0, "prefix_breaks": 0,
           "finish_reasons": Counter(), "analysis_steps": set()}
    previous: list[str] | None = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("analysis_step") is not None:
                out["analysis_steps"].add(row["analysis_step"])
            if row.get("event") == "request":
                out["requests"] += 1
                keys = [_message_key(m) for m in row.get("messages") or []]
                if previous is not None and keys[:len(previous)] != previous:
                    out["prefix_breaks"] += 1
                previous = keys
            elif row.get("event") == "response":
                out["responses"] += 1
                usage = row.get("usage") or {}
                details = usage.get("prompt_tokens_details") or {}
                prompt = int(usage.get("prompt_tokens") or 0)
                out["prompt_tokens"] += prompt
                out["generated_tokens"] += int(usage.get("completion_tokens") or 0)
                out["cached_tokens"] += int(details.get("cached_tokens") or 0)
                out["image_tokens"] += int(details.get("image_tokens") or 0)
                out["reasoning_tokens"] += int(usage.get("reasoning_tokens") or 0)
                out["max_prompt_tokens"] = max(out["max_prompt_tokens"], prompt)
                out["finish_reasons"][str(row.get("finish_reason"))] += 1
    out["turns"] = len(out.pop("analysis_steps"))
    out["finish_reasons"] = dict(out["finish_reasons"])
    return out


def transcript_stats(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    statuses = re.findall(r"^\[ANALYZER STATUS\]\n([^\n:]+)", text, re.M)
    status = Counter(s.strip() for s in statuses if s.strip() not in ("model", "base_url"))
    return {"turn_headers": len(re.findall(r"^--- analysis_step=\d+ \| action=", text, re.M)),
            "read_timeouts": status.get("request_timeout (yielding, history preserved)", 0),
            "overflow_recovered": status.get("context_overflow_recovered", 0),
            "statuses": dict(status)}


def _quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    k = (len(values) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def serve_stats(path: Path) -> dict:
    """SGLang serve.log: decode throughput, concurrency, MTP acceptance, prefix cache share, HTTP statuses."""
    decode: list[dict] = []
    prefill_new = prefill_cached = 0
    req = {"n": 0, "input": 0, "cached": 0, "output": 0, "prefill_ms": 0.0, "forward_ms": 0.0}
    statuses: Counter = Counter()
    first = last = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _TS.match(line)
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") if m else None
            if "Decode batch" in line:
                decode.append(_fields(line.split("Decode batch,", 1)[1]))
            elif "Prefill batch" in line:
                fields = _fields(line.split("Prefill batch,", 1)[1])
                prefill_new += int(fields.get("#new-token", 0))
                prefill_cached += int(fields.get("#cached-token", 0))
            elif "ReqTimeStats(" in line:
                vals = dict(re.findall(r"(\w+)=([\d.]+)", line))
                req["n"] += 1
                req["input"] += int(vals.get("input_len", 0))
                req["cached"] += int(vals.get("cached_input_len", 0))
                req["output"] += int(vals.get("output_len", 0))
                req["prefill_ms"] += float(vals.get("initial_prefill_elapsed", 0))
                req["forward_ms"] += float(vals.get("forward_duration", 0))
                if ts:
                    first = first or ts
                    last = ts
            else:
                hm = re.search(r'"(?:POST|GET) (\S+) HTTP/[\d.]+" (\d{3})', line)
                if hm:
                    statuses[f"{hm.group(1)} {hm.group(2)}"] += 1
    busy = [d for d in decode if d.get("#running-req", 0) > 0]
    tput = [d["gen throughput (token/s)"] for d in busy if "gen throughput (token/s)" in d]
    span = (last - first).total_seconds() if first and last and last > first else 0.0
    return {
        "decode_lines": len(decode),
        "decode_tok_s_mean": round(statistics.fmean(tput), 1) if tput else 0.0,
        "decode_tok_s_median": round(statistics.median(tput), 1) if tput else 0.0,
        "decode_tok_s_p90": round(_quantile(tput, 0.9), 1) if tput else 0.0,
        "running_req_mean": round(statistics.fmean(d["#running-req"] for d in busy), 2) if busy else 0.0,
        "queue_req_max": int(max((d.get("#queue-req", 0) for d in decode), default=0)),
        "accept_len_mean": round(statistics.fmean(d["accept len"] for d in busy if "accept len" in d), 2)
        if any("accept len" in d for d in busy) else None,
        "requests_finished": req["n"], "input_tokens": req["input"], "cached_input_tokens": req["cached"],
        "output_tokens": req["output"],
        "server_cache_share": round(req["cached"] / req["input"], 4) if req["input"] else 0.0,
        "prefill_new_tokens": prefill_new, "prefill_cached_tokens": prefill_cached,
        "output_tok_s_over_span": round(req["output"] / span, 1) if span else 0.0, "span_s": span,
        "prefill_time_share": round(req["prefill_ms"] / req["forward_ms"], 4) if req["forward_ms"] else 0.0,
        "http": dict(statuses),
    }


def _log_lines(path: Path) -> list[str]:
    """Lines of a notebook log: Kaggle's JSON stream (list of {stream_name, time, data}) or plain text."""
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        return text.splitlines()
    lines: list[str] = []
    for row in rows if isinstance(rows, list) else []:
        lines += str(row.get("data", "")).splitlines()
    return lines


def log_stats(path: Path) -> dict:
    lines = _log_lines(path)
    gate = [ln for ln in lines if "priority gate active" in ln]
    diag = [ln for ln in lines if re.search(r"gate: \d+ handovers", ln)]
    last = re.search(r"gate: (\d+) handovers, (\d+) blocked, (\d+)ms blocked, (\d+)ms total", diag[-1]) if diag else None
    finished = {}
    for ln in lines:
        m = re.match(r"\[finished\] (\S+) state=(\S+) level=(\d+)/(\d+) score=([\d.]+) actions=(\d+)", ln)
        if m:
            finished[m.group(1)] = {"state": m.group(2), "levels": int(m.group(3)), "score": float(m.group(5)),
                                    "actions": int(m.group(6))}
    slots = re.search(r"priority gate active: (\d+)", gate[0]) if gate else None
    return {
        "gate_slots": int(slots.group(1)) if slots else None,
        "tail_fade": next((ln.strip() for ln in lines if "tail fade phase" in ln), None),
        "gate_diag": {"enqueued": int(last.group(1)), "blocked": int(last.group(2)), "blocked_ms": int(last.group(3)),
                      "handover_ms": int(last.group(4))} if last else None,
        "read_timeouts": sum("analyzer request timed out" in ln for ln in lines) // (2 if _doubled(lines) else 1),
        "warmup_resets": sum("warmup RESET issued" in ln for ln in lines) // (2 if _doubled(lines) else 1),
        "finished": finished,
    }


def _doubled(lines: list[str]) -> bool:
    """Kaggle's log repeats each harness warning (handler and lastResort); count such lines once."""
    gate = [ln for ln in lines if "priority gate active" in ln]
    return len(gate) == 2 and gate[0] == gate[1]


def analyze(run_dir: Path, log: Path | None = None) -> dict:
    run_dir = Path(run_dir)
    bench = json.loads((run_dir / "benchmark.json").read_text())
    games: dict[str, dict] = {}
    for run in bench.get("game_runs", []):
        gid = run["game_id"]
        baselines = [int(b) for b in run.get("base_actions_per_level") or []]
        per_level = [int(a) for a in run.get("actions_per_level") or []]
        done = int(run.get("levels_completed") or 0)
        history = run.get("history")
        ids = Counter((h.get("action") or {}).get("id", "?") for h in history) if history is not None else Counter()
        ours = game_score([*per_level[:done], *([None] * (len(baselines) - done))], baselines)
        taaf = run.get("final_score")
        games[gid] = {
            "state": run.get("state"), "levels": done, "total_levels": len(baselines),
            "actions": len(history) if history is not None else sum(per_level),
            "actions_completed_levels": sum(per_level[:done]), "baseline_completed_levels": sum(baselines[:done]),
            "actions_current_level": per_level[done] if done < len(per_level) else 0,
            "per_level": [f"{a}/{b}" for a, b in zip(per_level[:done + 1], baselines)],
            "score_ours": round(ours, 2), "score_taaf": round(taaf, 2) if taaf is not None else None,
            "score_match": taaf is not None and abs(ours - taaf) < 1e-6,
            "undo_actions": ids.get("ACTION7", 0), "resets": ids.get("RESET", 0),
            "wallclock_s": round(float(run.get("final_wallclock_seconds") or 0), 1),
        }
    requests: dict[str, dict] = {}
    for path in sorted(run_dir.glob("*requests.jsonl")):
        name = path.name[:-len("_requests.jsonl")] if path.name.endswith("_requests.jsonl") else UNATTRIBUTED
        gid = re.sub(r"_p\d+$", "", name)
        requests[gid] = request_stats(path)
    for gid, stats in requests.items():
        stats["gate_admissions"] = 1 + stats["prefix_breaks"] if gid != UNATTRIBUTED else None
        stats["cache_share"] = round(stats["cached_tokens"] / stats["prompt_tokens"], 4) if stats["prompt_tokens"] else 0.0
    transcripts = {re.sub(r"_p\d+$", "", p.stem): transcript_stats(p)
                   for p in sorted((run_dir / "transcripts").glob("*.txt"))} if (run_dir / "transcripts").is_dir() else {}
    totals = {k: sum(s[k] for s in requests.values()) for k in
              ("requests", "responses", "prompt_tokens", "generated_tokens", "cached_tokens", "prefix_breaks")}
    totals["cache_share"] = round(totals["cached_tokens"] / totals["prompt_tokens"], 4) if totals["prompt_tokens"] else 0.0
    scores = [g["score_ours"] for g in games.values()]
    report = {
        "run_dir": str(run_dir), "label": bench.get("label"), "start": bench.get("start_time"),
        "end": bench.get("end_time"), "games": games, "requests": requests, "transcripts": transcripts,
        "totals": {**totals, "games": len(games), "mean_score_ours": round(sum(scores) / len(scores), 2) if scores else 0.0,
                   "levels": sum(g["levels"] for g in games.values()),
                   "actions": sum(g["actions"] for g in games.values())},
        "serve": serve_stats(run_dir / "serve.log") if (run_dir / "serve.log").is_file() else None,
        "summary": (run_dir / "summary.txt").read_text() if (run_dir / "summary.txt").is_file() else None,
    }
    if log is None:
        candidates = [p for p in run_dir.glob("*.log") if p.name not in ("serve.log", "bed.log")]
        log = candidates[0] if len(candidates) == 1 else (run_dir / "bed.log" if (run_dir / "bed.log").is_file() else None)
    report["log"] = log_stats(Path(log)) if log else None
    return report


def render(report: dict) -> str:
    out = [f"run {report['run_dir']} ({report['label']}, {report['start']} -> {report['end']})"]
    head = (f"{'game':<15} {'state':<9} {'lvl':>5} {'acts':>5} {'done/base':>10} {'score':>6} {'taaf':>6} "
            f"{'req':>4} {'prompt':>9} {'gen':>7} {'cache':>6} {'adm':>4} {'undo':>4}")
    out += [head, "-" * len(head)]
    for gid, g in sorted(report["games"].items()):
        r = report["requests"].get(gid, {})
        out.append(
            f"{gid:<15} {g['state'] or '?':<9} {g['levels']:>2}/{g['total_levels']:<2} {g['actions']:>5} "
            f"{g['actions_completed_levels']:>4}/{g['baseline_completed_levels']:<5} {g['score_ours']:>6.2f} "
            f"{(g['score_taaf'] if g['score_taaf'] is not None else float('nan')):>6.2f} {r.get('requests', 0):>4} "
            f"{r.get('prompt_tokens', 0):>9} {r.get('generated_tokens', 0):>7} {r.get('cache_share', 0):>6.1%} "
            f"{r.get('gate_admissions') or 0:>4} {g['undo_actions']:>4}")
    if UNATTRIBUTED in report["requests"]:
        r = report["requests"][UNATTRIBUTED]
        out.append(f"{UNATTRIBUTED:<15} {'':<9} {'':>5} {'':>5} {'':>10} {'':>6} {'':>6} {r['requests']:>4} "
                   f"{r['prompt_tokens']:>9} {r['generated_tokens']:>7} {r['cache_share']:>6.1%}")
    t = report["totals"]
    out.append(f"total: {t['games']} games, mean score (ours) {t['mean_score_ours']:.2f}, {t['levels']} levels, "
               f"{t['actions']} actions; {t['requests']} requests ({t['responses']} answered), prompt "
               f"{t['prompt_tokens']} / generated {t['generated_tokens']} tokens, prefix-cache hit share "
               f"{t['cache_share']:.1%}, {t['prefix_breaks']} prefix breaks")
    mismatched = [gid for gid, g in report["games"].items() if g["score_taaf"] is not None and not g["score_match"]]
    if mismatched:
        out.append(f"WARNING: our scorer and TAAF's disagree on {mismatched}")
    s = report.get("serve")
    if s:
        out.append(f"server: decode {s['decode_tok_s_mean']} tok/s mean (median {s['decode_tok_s_median']}, p90 "
                   f"{s['decode_tok_s_p90']}) over {s['decode_lines']} decode lines; {s['running_req_mean']} running on "
                   f"average, max queue {s['queue_req_max']}; MTP accept length {s['accept_len_mean']}; "
                   f"{s['requests_finished']} finished requests, cache share {s['server_cache_share']:.1%} of "
                   f"{s['input_tokens']} input tokens; {s['output_tokens']} output tokens = "
                   f"{s['output_tok_s_over_span']} tok/s over {s['span_s']:.0f} s; prefill {s['prefill_time_share']:.1%} "
                   f"of forward time; HTTP {s['http']}")
    lg = report.get("log")
    if lg:
        out.append(f"log: gate slots {lg['gate_slots']}, gate counters {lg['gate_diag']}, tail fade: "
                   f"{lg['tail_fade'] or 'no'}, read timeouts {lg['read_timeouts']}, warmup resets {lg['warmup_resets']}")
    tr = report.get("transcripts") or {}
    if tr:
        statuses = Counter()
        for t_ in tr.values():
            statuses.update(t_["statuses"])
        out.append(f"transcripts: {sum(t_['turn_headers'] for t_ in tr.values())} analyzer turns; statuses "
                   f"{dict(statuses.most_common(6))}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--log", type=Path, default=None, help="the notebook log (default: the one *.log in RUN_DIR)")
    ap.add_argument("--json", type=Path, default=None, help="also write the full report as JSON")
    args = ap.parse_args()
    report = analyze(args.run_dir, args.log)
    print(render(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
