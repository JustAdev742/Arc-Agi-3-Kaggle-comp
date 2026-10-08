#!/usr/bin/env python
"""Sample logged ARC agent requests for the fidelity probe (docs/research/beat-tufa/fidelity-probe.md).

    python -I scripts/fidelity_sample.py --logs RUN_DIR [--logs RUN_DIR2] --out DATASET_DIR [--per-game 14] \
        [--seed 20261008] [--note TEXT] [--dataset-id scottmahony/arc3-fidelity-prompts] [--copy-meta kaggle/fidelity]

RUN_DIR holds a Save & Run output of Daniel Franzen's notebook: one ``<game>_p0_requests.jsonl`` per game (one log
per game over all ``--logs`` folders), each line a
``request`` snapshot (the harness's messages, tools, tool_choice and its chat_template_kwargs, written right before the
call) or a ``response`` record (the server's usage and finish reason; same messages again). The logs are untrusted
data: this script only parses JSON (run it with ``python -I``) and executes nothing found in them.

Per game it keeps the answered requests (a response record follows) and draws ``--per-game`` of them, stratified
over turns and context lengths: the game's requests are cut into four consecutive quarters (early to late turns;
the harness trims history at ~116k tokens, so late quarters hold both long contexts and post-trim ~55k ones), and
within a quarter the draw is systematic over the requests sorted by prompt tokens (a seeded offset), so each quarter
spans its range of context lengths. Draws are deterministic for a given seed.

Each sampled request is written as it went over the wire: the harness's ``_strip_control_keys`` drops the private
``_arc3_control`` key from messages (``_apply_summary_visibility`` is a no-op with his settings), and everything else
is kept byte for byte (image parts are base64 PNG data URLs). The harness's other request fields are not logged; they
are reconstructed in the manifest (``harness_payload``) from his notebook's cell 4 and the harness code. For reference
each sample also carries the logged usage and finish reason, and the reply as it appears in the game's next request's
history (when that request extends this one).

Writes DATASET_DIR/requests.jsonl (replay order: by game, then by turn), DATASET_DIR/manifest.json (counts, sizes,
sha256 of every input and output file, parameters) and DATASET_DIR/dataset-metadata.json (private Kaggle dataset).
``--copy-meta DIR`` also copies the manifest and the metadata there (the repo keeps them; the data stays out of git).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

CONTROL_KEY = "_arc3_control"  # inference/agent/tool_agent.py _CONTROL_MESSAGE_KEY, stripped before sending
DATASET_ID = "scottmahony/arc3-fidelity-prompts"
OUT_NAME = "requests.jsonl"
QUARTERS = 4
# What the harness sends besides the logged fields (Franzen's notebook cell 4 setup_env, and
# ARC3-Inference/inference/agent/tool_agent.py _chat_completion + inference/utils/openai_compat.py build_chat_payload).
HARNESS_PAYLOAD = {
    "endpoint": "POST {LOCAL_ANALYZER_BASE_URL}/chat/completions = http://127.0.0.1:8001/v1/chat/completions",
    "model": "flashnext (LOCAL_ANALYZER_MODEL_ID = SERVED_MODEL_NAME)",
    "stream": False,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 12288,
    "seed": "not sent (LOCAL_ANALYZER_SEED unset)",
    "chat_template_kwargs": "{'enable_thinking': True} (build_chat_payload, LOCAL_ANALYZER_ENABLE_THINKING=true) merged "
                            "with the logged kwargs ({'preserve_thinking': True})",
    "separate_reasoning": "not sent (server default True; the server runs --reasoning-parser qwen3)",
    "messages": "the logged messages with '_arc3_control' removed (_strip_control_keys); "
                "_apply_summary_visibility is a no-op (ARC3_HIDE_FOLLOWUP_SUMMARIES unset)",
    "tools / tool_choice": "as logged (one 'python' function; 'auto')",
    "reasoning history": "assistant messages carry 'reasoning_content' (ARC3_REASONING_HISTORY_KEY)",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def wire_messages(messages: list[dict]) -> list[dict]:
    """The messages as the harness sends them (its _strip_control_keys)."""
    return [{k: v for k, v in m.items() if k != CONTROL_KEY} if CONTROL_KEY in m else m for m in messages]


def _images(message: dict) -> int:
    content = message.get("content")
    if not isinstance(content, list):
        return 0
    return sum(1 for part in content if isinstance(part, dict) and part.get("type") == "image_url")


def read_game(path: Path) -> list[dict]:
    """The request snapshots of one log, in call order, each with the usage of the response that followed it."""
    requests: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{line_no}: not a JSON object")
            event = row.get("event")
            if event == "request":
                if not isinstance(row.get("messages"), list) or not row["messages"]:
                    raise ValueError(f"{path.name}:{line_no}: request without messages")
                requests.append({"line": line_no, "row": row, "response": None})
            elif event == "response" and requests and requests[-1]["response"] is None:
                keys = ("analysis_step", "action", "request_index_within_turn")
                if all(row.get(k) == requests[-1]["row"].get(k) for k in keys):
                    requests[-1]["response"] = row
    for i, req in enumerate(requests):
        req["ordinal"] = i
        nxt = requests[i + 1]["row"]["messages"] if i + 1 < len(requests) else None
        msgs = req["row"]["messages"]
        req["reply"] = None
        if nxt is not None and len(nxt) > len(msgs) and nxt[:len(msgs)] == msgs and nxt[len(msgs)].get("role") == "assistant":
            req["reply"] = wire_messages([nxt[len(msgs)]])[0]
    return requests


def _usage(req: dict) -> dict:
    return dict((req["response"] or {}).get("usage") or {})


def draw(requests: list[dict], per_game: int, rng: random.Random) -> list[dict]:
    """``per_game`` answered requests: QUARTERS consecutive quarters by turn, systematic over prompt tokens within each."""
    answered = [r for r in requests if int(_usage(r).get("prompt_tokens") or 0) > 0]
    if len(answered) < per_game:
        raise ValueError(f"only {len(answered)} answered requests, fewer than --per-game {per_game}")
    quarters: list[list[dict]] = [[] for _ in range(QUARTERS)]
    for rank, req in enumerate(answered):
        req["quarter"] = rank * QUARTERS // len(answered)
        req["position"] = round(rank / max(1, len(answered) - 1), 4)
        quarters[req["quarter"]].append(req)
    chosen = []
    for q, members in enumerate(quarters):
        k = per_game // QUARTERS + (1 if q < per_game % QUARTERS else 0)
        if k > len(members):
            raise ValueError(f"quarter {q} has {len(members)} requests, fewer than {k}")
        ranked = sorted(members, key=lambda r: (int(_usage(r)["prompt_tokens"]), r["ordinal"]))
        step = len(ranked) / k
        offset = rng.random() * step
        chosen += [ranked[int(offset + j * step)] for j in range(k)]
    return sorted(chosen, key=lambda r: r["ordinal"])


def sample_line(game: str, game_id: str, source: str, req: dict, previous: dict | None) -> dict:
    row, usage = req["row"], _usage(req)
    details = usage.get("prompt_tokens_details") or {}
    messages = wire_messages(row["messages"])
    extends = previous is not None and row["messages"][:len(previous["row"]["messages"])] == previous["row"]["messages"]
    request = {"messages": messages, "tools": row.get("tools") or [], "tool_choice": row.get("tool_choice"),
               "chat_template_kwargs": row.get("chat_template_kwargs") or {}}
    return {
        "id": f"{game}#{req['ordinal']:03d}",
        "game": game,
        "game_id": game_id,
        "source": {"file": source, "line": req["line"], "request_ordinal": req["ordinal"],
                   "analysis_step": row.get("analysis_step"), "action": row.get("action"),
                   "request_index_within_turn": row.get("request_index_within_turn")},
        "stratum": {"quarter": req["quarter"], "position": req["position"]},
        "stats": {"prompt_tokens": int(usage.get("prompt_tokens") or 0),
                  "cached_tokens": int(details.get("cached_tokens") or 0),
                  "image_tokens": int(details.get("image_tokens") or 0),
                  "n_messages": len(messages), "n_images": sum(_images(m) for m in messages),
                  "last_role": messages[-1].get("role"), "last_has_image": _images(messages[-1]) > 0,
                  "extends_previous_sample": bool(extends)},
        "request": request,
        "logged": {"finish_reason": (req["response"] or {}).get("finish_reason"), "usage": usage, "reply": req["reply"]},
    }


def sample(logs: Path | list[Path], out: Path, per_game: int = 14, seed: int = 20261008,
           dataset_id: str = DATASET_ID, notes: list[str] | tuple = ()) -> dict:
    folders = [logs] if isinstance(logs, Path) else list(logs)
    sources = sorted((p for folder in folders for p in folder.glob("*_requests.jsonl") if p.name != "requests.jsonl"),
                     key=lambda p: p.name)
    if not sources:
        raise SystemExit(f"no <game>_requests.jsonl in {[str(f) for f in folders]}")
    games = [p.name.split("-")[0] for p in sources]
    if len(set(games)) != len(games):
        raise ValueError(f"a game has more than one log: {sorted(g for g in games if games.count(g) > 1)}")
    out.mkdir(parents=True, exist_ok=True)
    lines, source_meta = [], []
    for path in sources:
        game_id = path.name[:-len("_requests.jsonl")].removesuffix("_p0")
        game = game_id.split("-")[0]
        requests = read_game(path)
        chosen = draw(requests, per_game, random.Random(f"{seed}:{game}"))
        previous = None
        for req in chosen:
            lines.append(sample_line(game, game_id, path.name, req, previous))
            previous = req
        source_meta.append({"dir": path.parent.name, "name": path.name, "sha256": sha256_file(path),
                            "bytes": path.stat().st_size, "requests": len(requests),
                            "answered": sum(1 for r in requests if int(_usage(r).get("prompt_tokens") or 0) > 0)})
    target = out / OUT_NAME
    with open(target, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=True) + "\n")
    tokens = [s["stats"]["prompt_tokens"] for s in lines]
    images = [s["stats"]["n_images"] for s in lines]
    new_tokens = 0
    for i, s in enumerate(lines):  # the probe's sequential pass: what the prefix cache cannot serve, roughly
        prev = lines[i - 1] if i and lines[i - 1]["game"] == s["game"] else None
        shared = prev["stats"]["prompt_tokens"] if prev and s["stats"]["extends_previous_sample"] else 0
        new_tokens += s["stats"]["prompt_tokens"] - shared
    per_game_counts: dict[str, int] = {}
    per_quarter: dict[str, int] = {}
    for s in lines:
        per_game_counts[s["game"]] = per_game_counts.get(s["game"], 0) + 1
        per_quarter[str(s["stratum"]["quarter"])] = per_quarter.get(str(s["stratum"]["quarter"]), 0) + 1
    manifest = {
        "dataset_id": dataset_id,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        "files": [{"name": OUT_NAME, "sha256": sha256_file(target), "bytes": target.stat().st_size,
                   "requests": len(lines)}],
        "requests": len(lines),
        "per_game": per_game_counts,
        "per_quarter": per_quarter,
        "prompt_tokens": {"min": min(tokens), "median": statistics.median(tokens), "max": max(tokens),
                          "sum": sum(tokens)},
        "images_per_request": {"min": min(images), "median": statistics.median(images), "max": max(images)},
        "last_message_has_image": sum(1 for s in lines if s["stats"]["last_has_image"]),
        "estimate_sequential_prefill_tokens": new_tokens,
        "params": {"per_game": per_game, "seed": seed, "quarters": QUARTERS,
                   "order": "by game (file name), then by turn"},
        "source": {"files": source_meta, "notes": list(notes)},
        "harness_payload": HARNESS_PAYLOAD,
        "sampler": {"script": "scripts/fidelity_sample.py", "sha256": sha256_file(Path(__file__))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    metadata = {"title": dataset_id.split("/")[1], "id": dataset_id, "licenses": [{"name": "other"}],
                "subtitle": "Logged ARC-AGI-3 agent requests for a serving fidelity probe (private)",
                "description": "Requests sampled by scripts/fidelity_sample.py from the request logs of Daniel "
                               "Franzen's public Milestone 2 notebook run (Apache-2.0). Private; see manifest.json."}
    (out / "dataset-metadata.json").write_text(json.dumps(metadata, indent=1) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--logs", required=True, type=Path, action="append",
                    help="run output dir with <game>_p0_requests.jsonl files (repeatable; one log per game)")
    ap.add_argument("--out", required=True, type=Path, help="dataset folder to write (kept out of git)")
    ap.add_argument("--per-game", type=int, default=14)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--dataset-id", default=DATASET_ID)
    ap.add_argument("--note", action="append", default=[], help="where the logs come from (kept in the manifest)")
    ap.add_argument("--copy-meta", type=Path, default=None, help="also copy manifest.json + dataset-metadata.json here")
    args = ap.parse_args()
    try:
        manifest = sample(args.logs, args.out, args.per_game, args.seed, args.dataset_id, args.note)
    except ValueError as exc:
        raise SystemExit(f"fidelity_sample: {exc}") from None
    if args.copy_meta:
        args.copy_meta.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.json", "dataset-metadata.json"):
            (args.copy_meta / name).write_text((args.out / name).read_text())
    f = manifest["files"][0]
    print(f"wrote {args.out / f['name']}: {f['requests']} requests, {f['bytes'] / 1e6:.1f} MB, sha256 {f['sha256'][:12]}; "
          f"per game {manifest['per_game']}; prompt tokens {manifest['prompt_tokens']}; "
          f"sequential-pass prefill estimate {manifest['estimate_sequential_prefill_tokens']:,} tokens")


if __name__ == "__main__":
    sys.exit(main())
