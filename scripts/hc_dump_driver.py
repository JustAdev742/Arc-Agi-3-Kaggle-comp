#!/usr/bin/env python
"""Replay maximal snapshots of logged ARC agent conversations through the hyper-connection dump server, for the MTP
draft fine-tune (docs/research/beat-tufa/mtp-drafter-finetune.md, sections 2.4-2.6 and 9).

    python -I scripts/hc_dump_driver.py --logs RUN_DIR [--logs RUN_DIR2] --dry-run [--split train]
    python -I scripts/hc_dump_driver.py --logs RUN_DIR --out OUT --dump-dir /dev/shm/hc \\
        [--base-url http://127.0.0.1:8001] [--model flashnext] [--split train] [--max-dump-gb 30]

Inputs are Daniel Franzen's harness request logs, ``<game>-<id>_p<k>_requests.jsonl`` at any depth under each
``--logs`` folder (written on Save & Run with ``save_request_logs=True``). Each line is a ``request`` (messages, tools,
tool_choice, chat_template_kwargs, written right before the call) or the ``response`` that followed (usage, finish
reason). They are untrusted data: parsed as JSON, nothing in them is run.

Per log (one game pass):

- **Snapshots.** The maximal requests: each request that the next one does not extend (same tools and template
  kwargs, and its messages a prefix of the next one's), plus the last. Past reasoning is rendered
  (preserve_thinking), so their prompts hold every logged assistant turn except the game's last reply.
- **Turns.** Assistant messages, identified by their tool-call ids (unique per generation) or else their JSON, in
  order of first appearance. A turn's length is the completion tokens of the request before its first appearance.
- **Hygiene** (section 2.6). A turn is excluded from the loss if its text (reasoning, content, tool calls) repeats
  an earlier turn of the pass exactly ("repeat"), if it contains "stuck in a loop" ("stuck"), or, unless
  ``--keep-duplicates``, if an earlier snapshot already has it ("covered": a history trim keeps recent turns, which
  then appear in two snapshots; the first one has the context they were generated in). The remaining turns are the
  snapshot's loss spans: its assistant message k is span k of the dump (scripts/sglang_hc_dump_patch.py counts
  assistant headers from 0). A snapshot without a loss turn is dropped.
- **Split** by game: ``--split train`` (default) leaves out the 11 games of the fidelity probe sample (HOLDOUT),
  ``holdout`` keeps only them, ``all`` keeps both.

Replay, snapshot by snapshot in log order: write ``<dump-dir>/plans/<rid>.json`` (its loss spans, read by the dump
server), then send the logged request to ``POST {base}/v1/chat/completions`` the way scripts/fidelity_probe.py does
(``build_body``: messages without ``_arc3_control``, tools, tool_choice, ``chat_template_kwargs`` =
``{"enable_thinking": true}`` + the logged ones), with ``max_tokens`` 1, temperature 0, one logprob and the ``rid``.
With ``--dump-dir`` each request is checked against the dump's index: its chunks must cover the prompt from position 0
to the server's ``prompt_tokens``, and the dump must have opened one span per assistant message plus the generation
prompt. The first request failing that check stops the run (the server is not dumping), and so do three failures in
a row. The run stops before ``--max-dump-gb``, ``--max-prefill-tokens`` or ``--max-minutes`` would be passed. The
prefix cache is flushed before every request unless the server reports ``disable_radix_cache``.

Writes OUT/plan.json (summary, parameters, sources with sha256), OUT/snapshots.jsonl (one line per snapshot with its
turns and loss spans: what the trainer needs to map rids to games and spans to turns), OUT/replay.jsonl (one line per
request sent) and OUT/replay-summary.json. ``--dry-run`` only prints the plan (and writes plan.json and
snapshots.jsonl when ``--out`` is given).

Stdlib only. It loads scripts/fidelity_probe.py and scripts/fidelity_sample.py from its own folder: keep the three
files together.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import re
import statistics
import sys
import time
from pathlib import Path

HOLDOUT = frozenset({"ar25", "ft09", "lp85", "r11l", "re86", "sb26", "sc25", "tn36", "tr87", "tu93", "vc33"})
LOOP_PHRASE = "stuck in a loop"
LOG_RE = re.compile(r"^(?P<game>[a-z0-9]+)-(?P<gid>[0-9a-z]+)_p(?P<pass>\d+)_requests\.jsonl$")
CONTEXT = 256                  # the dump server's ARC3_HC_DUMP_CONTEXT (only used for estimates here)
ROW_BYTES = 10240 + 2 + 4 + 1  # a kept FP8 row: codes, BF16 scale, position, role
TOKEN_BYTES = 4 + 4 + 12       # per prefill token: id, position, M-RoPE positions
PREFILL_TOK_S = 10_700         # measured prefill rate (serving.md, Franzen's serve.log, requests > 4k new tokens)
DEFAULT_TOK_PER_CHAR, DEFAULT_TOK_PER_IMAGE = 1 / 3.3, 400  # estimates when a log has no usage to calibrate on


def _sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"arc3_hc_driver_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} not found (keep it next to {Path(__file__).name})")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fp = _sibling("fidelity_probe")
fs = _sibling("fidelity_sample")


def log(message: str) -> None:
    print(f"[hc-dump {time.strftime('%H:%M:%S')}] {message}", flush=True)


# ------------------------------------------------------------------------------------------------------- planning


def discover(folders) -> list[Path]:
    """Every ``<game>-<id>_p<k>_requests.jsonl`` under the folders (or the files given), each content once."""
    seen, out = set(), []
    for folder in folders:
        folder = Path(folder)
        found = [folder] if folder.is_file() else sorted(folder.rglob("*_requests.jsonl"))
        for path in found:
            if LOG_RE.match(path.name):
                digest = fs.sha256_file(path)
                if digest not in seen:
                    seen.add(digest)
                    out.append(path)
    return out


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def turn_identity(message: dict) -> str:
    ids = [tc["id"] for tc in message.get("tool_calls") or [] if isinstance(tc, dict) and tc.get("id")]
    if ids:
        return "ids:" + "|".join(ids)
    wire = {k: v for k, v in message.items() if k != fs.CONTROL_KEY}
    return "json:" + hashlib.sha256(json.dumps(wire, sort_keys=True).encode()).hexdigest()


def turn_text(message: dict) -> str:
    """What the model generated: reasoning (trimmed, as the template renders it), content, tool calls."""
    parts = [str(message.get("reasoning_content") or "").strip(), _content_text(message.get("content"))]
    for tc in message.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        args = fn.get("arguments")
        parts += [str(fn.get("name") or ""), args if isinstance(args, str) else json.dumps(args, sort_keys=True)]
    return "\x00".join(parts)


def extends(row: dict, nxt: dict) -> bool:
    """``nxt`` renders ``row``'s prompt as its prefix: same tools and kwargs, ``row``'s messages first (or equal),
    compared as sent (without the harness's private ``_arc3_control`` key)."""
    a, b = fs.wire_messages(row["messages"]), fs.wire_messages(nxt["messages"])
    return (len(b) >= len(a) and b[:len(a)] == a and (row.get("tools") or []) == (nxt.get("tools") or [])
            and (row.get("chat_template_kwargs") or {}) == (nxt.get("chat_template_kwargs") or {}))


def _text_chars(row: dict) -> int:
    chars = len(json.dumps(row.get("tools") or []))
    for m in row["messages"]:
        chars += len(_content_text(m.get("content"))) + len(str(m.get("reasoning_content") or ""))
        for tc in m.get("tool_calls") or []:
            chars += len(json.dumps(((tc or {}).get("function") or {}).get("arguments") or ""))
    return chars


def _n_images(row: dict) -> int:
    return sum(fs._images(m) for m in row["messages"])


def _usage(req: dict) -> dict:
    return dict((req.get("response") or {}).get("usage") or {})


def calibrate(requests: list[dict]) -> tuple[float, float]:
    """Tokens per text character and per image, from the answered requests' usage (medians)."""
    per_image, per_char = [], []
    for req in requests:
        usage = _usage(req)
        prompt = int(usage.get("prompt_tokens") or 0)
        if not prompt:
            continue
        images = _n_images(req["row"])
        image_tokens = int((usage.get("prompt_tokens_details") or {}).get("image_tokens") or 0)
        if images and image_tokens:
            per_image.append(image_tokens / images)
        chars = _text_chars(req["row"])
        if chars:
            per_char.append((prompt - image_tokens) / chars)
    return (statistics.median(per_char) if per_char else DEFAULT_TOK_PER_CHAR,
            statistics.median(per_image) if per_image else DEFAULT_TOK_PER_IMAGE)


def plan_log(path: Path, *, keep_duplicates: bool = False, context: int = CONTEXT) -> dict:
    """The snapshots of one game pass, their turns, and estimates (see the module docstring)."""
    match = LOG_RE.match(path.name)
    if not match:
        raise ValueError(f"{path.name}: not a <game>-<id>_p<k>_requests.jsonl log")
    game, game_pass = match["game"], int(match["pass"])
    digest = fs.sha256_file(path)
    requests = fs.read_game(path)
    per_char, per_image = calibrate(requests)

    turns: dict[str, dict] = {}
    first_text: dict[str, int] = {}
    for j, req in enumerate(requests):
        for message in req["row"]["messages"]:
            if message.get("role") != "assistant":
                continue
            ident = turn_identity(message)
            if ident in turns:
                continue
            text = turn_text(message)
            usage = _usage(requests[j - 1]) if j else {}
            reasons = []
            if LOOP_PHRASE in text.lower():
                reasons.append("stuck")
            key = hashlib.sha256(text.encode()).hexdigest()
            if key in first_text:
                reasons.append("repeat")
            first_text.setdefault(key, len(turns))
            turns[ident] = {"turn": len(turns), "first_request": j, "excluded": reasons,
                            "repeat_of": first_text[key] if "repeat" in reasons else None,
                            "tokens": int(usage.get("completion_tokens") or 0) or round(len(text) * per_char),
                            "tokens_source": "logged" if usage.get("completion_tokens") else "estimate"}

    snapshots, covered = [], set()
    for i, req in enumerate(requests):
        if i + 1 < len(requests) and extends(req["row"], requests[i + 1]["row"]):
            continue
        row = req["row"]
        listed = []
        for message in row["messages"]:
            if message.get("role") != "assistant":
                continue
            ident = turn_identity(message)
            turn = turns[ident]
            reasons = list(turn["excluded"]) or (["covered"] if ident in covered and not keep_duplicates else [])
            listed.append({"span": len(listed), "turn": turn["turn"], "loss": not reasons, "reasons": reasons,
                           "tokens": turn["tokens"]})
        covered.update(turn_identity(m) for m in row["messages"] if m.get("role") == "assistant")
        usage = _usage(req)
        prompt = int(usage.get("prompt_tokens") or 0)
        estimate = round(_text_chars(row) * per_char + _n_images(row) * per_image)
        loss = [t["span"] for t in listed if t["loss"]]
        loss_tokens = sum(t["tokens"] for t in listed if t["loss"])
        kept = loss_tokens + context * len(loss)  # each loss span plus its context rows (overlaps ignored)
        snapshots.append({
            "rid": f"hc-{digest[:8]}-{game}-p{game_pass}-r{i:04d}", "game": game, "pass": game_pass,
            "split": "holdout" if game in HOLDOUT else "train",
            "source": {"file": path.name, "sha256": digest, "line": req["line"], "request_ordinal": i,
                       "messages_sha256": hashlib.sha256(json.dumps(row["messages"], sort_keys=True).encode()
                                                         ).hexdigest()},
            "n_messages": len(row["messages"]), "n_assistant": len(listed), "n_images": _n_images(row),
            "loss_spans": loss, "turns": listed,
            "prompt_tokens": prompt or estimate, "prompt_tokens_source": "logged" if prompt else "estimate",
            "loss_tokens_est": loss_tokens, "kept_rows_est": kept,
            "dropped": None if loss else "no loss turn",
        })
    excluded = collections.Counter(r for t in turns.values() for r in t["excluded"])
    return {"path": str(path), "file": path.name, "sha256": digest, "game": game, "pass": game_pass,
            "split": "holdout" if game in HOLDOUT else "train", "requests": len(requests), "turns": len(turns),
            "excluded_turns": dict(excluded), "tokens_per_char": round(per_char, 4),
            "tokens_per_image": round(per_image, 1), "snapshots": snapshots}


def plan(paths, *, split: str = "train", games=None, keep_duplicates: bool = False, context: int = CONTEXT,
         max_snapshots: int | None = None) -> dict:
    """Plans for every log in the split; ``replay`` is the snapshots to send, in order (by file, then turn)."""
    logs, replay = [], []
    for path in paths:
        match = LOG_RE.match(Path(path).name)
        if not match:
            continue
        game = match["game"]
        if (split == "train" and game in HOLDOUT) or (split == "holdout" and game not in HOLDOUT):
            continue
        if games and game not in games:
            continue
        entry = plan_log(Path(path), keep_duplicates=keep_duplicates, context=context)
        logs.append(entry)
        replay += [s for s in entry["snapshots"] if not s["dropped"]]
    if max_snapshots is not None:
        replay = replay[:max_snapshots]
    totals = {
        "logs": len(logs), "games": sorted({e["game"] for e in logs}), "requests": sum(e["requests"] for e in logs),
        "turns": sum(e["turns"] for e in logs), "snapshots": sum(len(e["snapshots"]) for e in logs),
        "snapshots_replayed": len(replay),
        "excluded_turns": dict(sum((collections.Counter(e["excluded_turns"]) for e in logs), collections.Counter())),
        "loss_turns": sum(len(s["loss_spans"]) for s in replay),
        "covered_turns": sum(1 for s in replay for t in s["turns"] if "covered" in t["reasons"]),
        "prefill_tokens": sum(s["prompt_tokens"] for s in replay),
        "loss_tokens_est": sum(s["loss_tokens_est"] for s in replay),
        "kept_rows_est": sum(s["kept_rows_est"] for s in replay),
    }
    totals["dump_gb_est"] = round((totals["kept_rows_est"] * ROW_BYTES + totals["prefill_tokens"] * TOKEN_BYTES) / 1e9,
                                  2)
    totals["prefill_minutes_est"] = round(totals["prefill_tokens"] / PREFILL_TOK_S / 60, 1)
    params = {"split": split, "games": sorted(games) if games else None, "keep_duplicates": keep_duplicates,
              "context": context, "max_snapshots": max_snapshots, "holdout": sorted(HOLDOUT)}
    return {"params": params, "totals": totals, "logs": logs, "replay": replay}


def format_plan(result: dict) -> str:
    lines = [f"{'log':<36} {'split':<7} {'req':>4} {'turns':>5} {'snap':>4} {'used':>4} {'loss':>5} {'excl':>4} "
             f"{'cov':>4} {'prefill tok':>12} {'loss tok':>10}"]
    for e in result["logs"]:
        used = [s for s in e["snapshots"] if not s["dropped"]]
        lines.append(f"{e['file'][:36]:<36} {e['split']:<7} {e['requests']:>4} {e['turns']:>5} "
                     f"{len(e['snapshots']):>4} {len(used):>4} {sum(len(s['loss_spans']) for s in used):>5} "
                     f"{sum(e['excluded_turns'].values()):>4} "
                     f"{sum(1 for s in used for t in s['turns'] if 'covered' in t['reasons']):>4} "
                     f"{sum(s['prompt_tokens'] for s in used):>12,} {sum(s['loss_tokens_est'] for s in used):>10,}")
    t = result["totals"]
    lines.append(f"total: {t['logs']} logs ({len(t['games'])} games), {t['snapshots_replayed']} of {t['snapshots']} "
                 f"snapshots to replay, {t['loss_turns']} loss turns (excluded {t['excluded_turns'] or 'none'}, "
                 f"{t['covered_turns']} covered by an earlier snapshot); prefill {t['prefill_tokens']:,} tokens "
                 f"(~{t['prefill_minutes_est']} min at {PREFILL_TOK_S:,} tok/s), loss rows ~{t['loss_tokens_est']:,}, "
                 f"kept rows ~{t['kept_rows_est']:,}, dump ~{t['dump_gb_est']} GB (FP8)")
    return "\n".join(lines)


def write_plan(result: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in result.items() if k not in ("logs", "replay")}
    summary["logs"] = [{k: v for k, v in e.items() if k != "snapshots"} for e in result["logs"]]
    summary["driver"] = {"script": "scripts/hc_dump_driver.py", "sha256": fs.sha256_file(Path(__file__))}
    (out / "plan.json").write_text(json.dumps(summary, indent=1) + "\n")
    with open(out / "snapshots.jsonl", "w", encoding="utf-8") as f:
        for snapshot in result["replay"]:
            f.write(json.dumps(snapshot) + "\n")


# --------------------------------------------------------------------------------------------------------- replay


def iter_rows(result: dict):
    """(snapshot, logged request row) in replay order; each log is read once, line by line."""
    by_path = {e["file"] + e["sha256"]: e["path"] for e in result["logs"]}
    current, rows = None, {}
    for snapshot in result["replay"]:
        key = snapshot["source"]["file"] + snapshot["source"]["sha256"]
        if key != current:
            current = key
            wanted = {s["source"]["line"] for s in result["replay"] if s["source"]["file"] + s["source"]["sha256"] == key}
            rows = {}
            with open(by_path[key], encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    if line_no in wanted:
                        rows[line_no] = json.loads(line)
        row = rows.pop(snapshot["source"]["line"])
        digest = hashlib.sha256(json.dumps(row["messages"], sort_keys=True).encode()).hexdigest()
        if digest != snapshot["source"]["messages_sha256"]:
            raise RuntimeError(f"{snapshot['source']['file']}:{snapshot['source']['line']} changed since it was planned")
        yield snapshot, row


class IndexTail:
    """Reads the dump's index.jsonl as it grows: chunk lines by request id, and the bytes written so far."""

    def __init__(self, dump_dir: Path):
        self.path = Path(dump_dir) / "index.jsonl"
        self.offset = 0
        self.by_rid: dict[str, list[dict]] = collections.defaultdict(list)
        self.bytes_total = 0
        self.stopped: dict | None = None

    def poll(self) -> None:
        if not self.path.is_file():
            return
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = f.read()
        end = data.rfind(b"\n") + 1  # a line still being written stays for the next poll
        self.offset += end
        for line in data[:end].decode("utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("rid") is not None:
                self.by_rid[record["rid"]].append(record)
            if record.get("event") == "stopped":
                self.stopped = record
            if "bytes_total" in record:
                self.bytes_total = max(self.bytes_total, int(record["bytes_total"]))


def check_dump(records: list[dict], prompt_tokens: int, n_assistant: int) -> tuple[list[str], dict]:
    """Problems with one request's dump (empty when it is complete) and its counts."""
    problems = [f"dump error: {r.get('error')}" for r in records if r.get("event") == "error"]
    chunks = [r for r in records if "file" in r]
    if chunks:
        attempt = max(r["attempt"] for r in chunks)
        chunks = sorted((r for r in chunks if r["attempt"] == attempt), key=lambda r: r["start"])
    stats = {"chunks": len(chunks), "kept_rows": sum(r["kept"] for r in chunks),
             "span_rows": sum(r["span_rows"] for r in chunks), "bytes": sum(r["bytes"] for r in chunks),
             "qerr_max": max((r["qerr_max"] for r in chunks), default=None)}
    if not chunks:
        return [*problems, "no dump lines for this request"], stats
    covered = 0
    for r in chunks:
        if r["start"] != covered:
            problems.append(f"chunk at {r['start']}, expected {covered} (rows missing or repeated)")
            break
        covered += r["n"]
    if covered != prompt_tokens:
        problems.append(f"dump covers {covered} positions, the server counted {prompt_tokens} prompt tokens")
    opened = chunks[-1]["spans_opened_total"]
    if opened != n_assistant + 1:
        problems.append(f"{opened} assistant headers in the prompt, expected {n_assistant + 1} (a message renders "
                        "an extra or missing header, so spans do not map to turns)")
    return problems, stats


def replay(result: dict, out: Path, *, base_url: str, model: str = "flashnext", dump_dir: Path | None = None,
           timeout: float = 900.0, health_wait: float = 1800.0, max_minutes: float | None = None,
           max_prefill_tokens: int | None = None, max_dump_gb: float | None = None, logger=log) -> dict:
    """Send every planned snapshot (see the module docstring); returns the summary also written to OUT."""
    t0 = time.time()
    base = base_url.rstrip("/")
    out.mkdir(parents=True, exist_ok=True)
    write_plan(result, out)
    fp.wait_healthy(base, deadline=time.time() + health_wait, logger=logger)
    info = fp.server_info(base)
    notes = []
    flush_each = info.get("disable_radix_cache") is not True
    if flush_each:
        notes.append("the server keeps a prefix cache (no --disable-radix-cache): flushing before every request")
    if info.get("max_running_requests") not in (None, 1):
        notes.append(f"max_running_requests is {info.get('max_running_requests')}, not 1")
    if info.get("speculative_algorithm"):
        notes.append(f"speculative decoding is on ({info.get('speculative_algorithm')}); the dump needs none")
    for note in notes:
        logger(f"note: {note}")
    tail = IndexTail(dump_dir) if dump_dir else None
    if dump_dir:
        (Path(dump_dir) / "plans").mkdir(parents=True, exist_ok=True)
    summary = {"started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t0)), "server_info": info,
               "notes": notes, "planned": len(result["replay"]), "sent": 0, "ok": 0, "dump_ok": 0,
               "prefill_tokens": 0, "stopped": None, "failed": False, "dump_dir": str(dump_dir) if dump_dir else None}
    failures_in_a_row = 0
    with open(out / "replay.jsonl", "w", encoding="utf-8") as results:
        for snapshot, row in iter_rows(result):
            elapsed_min = (time.time() - t0) / 60
            if max_minutes is not None and elapsed_min >= max_minutes:
                summary["stopped"] = f"--max-minutes {max_minutes} reached"
            elif max_prefill_tokens is not None and (summary["prefill_tokens"] + snapshot["prompt_tokens"]
                                                     > max_prefill_tokens):
                summary["stopped"] = f"the next snapshot would pass --max-prefill-tokens {max_prefill_tokens}"
            elif tail is not None and max_dump_gb is not None and tail.bytes_total + snapshot["kept_rows_est"] * \
                    ROW_BYTES > max_dump_gb * 1e9:
                summary["stopped"] = f"the next snapshot would pass --max-dump-gb {max_dump_gb}"
            elif tail is not None and tail.stopped:
                summary["stopped"] = f"the dump server stopped writing: {tail.stopped.get('reason')}"
            if summary["stopped"]:
                logger(f"stopping: {summary['stopped']}")
                break
            rid = snapshot["rid"]
            if dump_dir:
                plan_file = Path(dump_dir) / "plans" / f"{rid}.json"
                plan_file.write_text(json.dumps({"rid": rid, "loss_spans": snapshot["loss_spans"],
                                                 "n_assistant": snapshot["n_assistant"], "game": snapshot["game"],
                                                 "split": snapshot["split"], "source": snapshot["source"]}))
            if flush_each:
                fp.flush_cache(base, logger=logger)
            body = fp.build_body({"request": {"messages": row["messages"], "tools": row.get("tools") or [],
                                              "tool_choice": row.get("tool_choice"),
                                              "chat_template_kwargs": row.get("chat_template_kwargs") or {}}},
                                 model=model, max_tokens=1, top_logprobs=1, extras={"rid": rid})
            rec = fp.call(base, json.dumps(body).encode(), timeout)
            usage = rec.get("usage") or {}
            line = {"rid": rid, "game": snapshot["game"], "ok": bool(rec.get("ok")), "status": rec.get("status"),
                    "latency_s": rec.get("latency_s"), "prompt_tokens": usage.get("prompt_tokens"),
                    "cached_tokens": rec.get("cached_tokens"), "planned_prompt_tokens": snapshot["prompt_tokens"],
                    "error": rec.get("error")}
            summary["sent"] += 1
            summary["ok"] += line["ok"]
            summary["prefill_tokens"] += int(usage.get("prompt_tokens") or 0)
            if tail is not None:
                tail.poll()
                if line["ok"]:
                    problems, stats = check_dump(tail.by_rid.get(rid, []), int(usage.get("prompt_tokens") or 0),
                                                 snapshot["n_assistant"])
                else:
                    problems, stats = ["the request failed"], {}
                line["dump"] = {"ok": not problems, "problems": problems, **stats}
                summary["dump_ok"] += not problems
                failures_in_a_row = failures_in_a_row + 1 if problems else 0
            results.write(json.dumps(line) + "\n")
            results.flush()
            logger(f"{rid}: {'ok' if line['ok'] else 'FAILED ' + str(line['error'])[:200]} "
                   f"{line['prompt_tokens']} prompt tokens in {line['latency_s']} s"
                   + (f"; dump {'ok' if line['dump']['ok'] else 'BAD: ' + '; '.join(line['dump']['problems'])}"
                      if tail is not None else ""))
            if tail is not None and line["dump"]["problems"]:
                if summary["sent"] == 1:
                    summary["stopped"] = ("the first request was not dumped completely; is ARC3_HC_DUMP set for the "
                                          "server, the patch applied, and the prefill CUDA graph off? "
                                          + "; ".join(line["dump"]["problems"]))
                elif failures_in_a_row >= 3:
                    summary["stopped"] = "three requests in a row were not dumped completely"
                if summary["stopped"]:
                    summary["failed"] = True
                    logger(f"stopping: {summary['stopped']}")
                    break
    summary["elapsed_s"] = round(time.time() - t0, 1)
    if tail is not None:
        tail.poll()
        summary["dump_bytes"] = tail.bytes_total
    summary["prefill_tok_s"] = round(summary["prefill_tokens"] / max(summary["elapsed_s"], 1e-9), 1)
    (out / "replay-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    logger(f"done: {summary['ok']} of {summary['sent']} requests ok"
           + (f", {summary['dump_ok']} dumped completely ({summary.get('dump_bytes', 0) / 1e9:.2f} GB)" if tail else "")
           + f", {summary['prefill_tokens']:,} prompt tokens in {summary['elapsed_s']:.0f} s -> {out}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--logs", required=True, type=Path, action="append",
                    help="folder with <game>-<id>_p<k>_requests.jsonl files at any depth, or one such file (repeatable)")
    ap.add_argument("--out", type=Path, default=None, help="where plan.json, snapshots.jsonl and replay.jsonl go")
    ap.add_argument("--split", choices=("train", "holdout", "all"), default="train")
    ap.add_argument("--games", default=None, help="comma-separated game ids to keep (within the split)")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="keep a turn as loss in every snapshot that has it, not only the first")
    ap.add_argument("--context", type=int, default=CONTEXT, help="the server's ARC3_HC_DUMP_CONTEXT (for estimates)")
    ap.add_argument("--max-snapshots", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop (no server)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8001", help="server root (without /v1)")
    ap.add_argument("--model", default="flashnext")
    ap.add_argument("--dump-dir", type=Path, default=None, help="the server's ARC3_HC_DUMP (plans + checks)")
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds per request")
    ap.add_argument("--health-wait", type=float, default=1800.0, help="seconds to wait for /health")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--max-prefill-tokens", type=int, default=None)
    ap.add_argument("--max-dump-gb", type=float, default=None)
    args = ap.parse_args(argv)
    paths = discover(args.logs)
    if not paths:
        print(f"hc_dump_driver: no <game>-<id>_p<k>_requests.jsonl under {[str(p) for p in args.logs]}",
              file=sys.stderr)
        return 1
    games = {g.strip() for g in args.games.split(",") if g.strip()} if args.games else None
    result = plan(paths, split=args.split, games=games, keep_duplicates=args.keep_duplicates, context=args.context,
                  max_snapshots=args.max_snapshots)
    print(format_plan(result), flush=True)
    if args.dry_run:
        if args.out:
            write_plan(result, args.out)
        return 0
    if args.out is None:
        ap.error("--out is required unless --dry-run")
    if not result["replay"]:
        print("hc_dump_driver: nothing to replay", file=sys.stderr)
        return 1
    summary = replay(result, args.out, base_url=args.base_url, model=args.model, dump_dir=args.dump_dir,
                     timeout=args.timeout, health_wait=args.health_wait, max_minutes=args.max_minutes,
                     max_prefill_tokens=args.max_prefill_tokens, max_dump_gb=args.max_dump_gb)
    return 1 if summary["failed"] or not summary["ok"] else 0


if __name__ == "__main__":
    sys.exit(main())
