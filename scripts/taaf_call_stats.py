#!/usr/bin/env python
"""Per-call statistics of Duck (TAAF) runs from their transcripts, to compare arms on more than the score.

    .venv/bin/python scripts/taaf_call_stats.py runs/exp032-anim-flashnext runs/exp037-effort-medium [...]

For each run directory (the folder holding ``kernel-output/transcripts`` or the transcripts folder itself), counts the
model responses (``[MODEL RESPONSE META]`` blocks) per game and reports the reasoning and visible-content length per
response, the finish reasons (``length`` means the output cap cut the response) and the analysis turns per game. The
server-side view (queue and decode seconds per request, running requests) is in ``summary.json["server"]`` from
scripts/pull_taaf_run.py.
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import Counter
from pathlib import Path

META_RE = re.compile(r"^\[MODEL RESPONSE META\]\n(.*?)(?=^raw_tool_calls:|^\[|\Z)", re.M | re.S)
TURN_RE = re.compile(r"^--- analysis_step=\d+ \| action=\d+ \| \d\d:\d\d:\d\d \| tool-agent ---$", re.M)


def transcripts_dir(path: Path) -> Path:
    for cand in (path / "kernel-output" / "transcripts", path / "transcripts", path):
        if cand.is_dir() and any(cand.glob("*.txt")):
            return cand
    raise SystemExit(f"no transcripts under {path}")


def pct(xs: list[int], q: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def run_stats(path: Path) -> dict:
    reasoning, content, finish, calls_per_game, turns_per_game = [], [], Counter(), [], []
    for f in sorted(transcripts_dir(path).glob("*.txt")):
        text = f.read_text(errors="replace")
        metas = META_RE.findall(text)
        calls_per_game.append(len(metas))
        turns_per_game.append(len(TURN_RE.findall(text)))
        for block in metas:
            fields = dict(ln.split(": ", 1) for ln in block.splitlines() if ": " in ln)
            finish[fields.get("finish_reason", "?")] += 1
            if fields.get("reasoning_chars", "").isdigit():
                reasoning.append(int(fields["reasoning_chars"]))
            if fields.get("content_chars", "").isdigit():
                content.append(int(fields["content_chars"]))
    return {"run": str(path), "games": len(calls_per_game), "responses": sum(calls_per_game),
            "responses_per_game_median": statistics.median(calls_per_game) if calls_per_game else 0,
            "turns_per_game_median": statistics.median(turns_per_game) if turns_per_game else 0,
            "reasoning_chars_mean": round(statistics.mean(reasoning)) if reasoning else 0,
            "reasoning_chars_p50": pct(reasoning, 0.5), "reasoning_chars_p90": pct(reasoning, 0.9),
            "reasoning_chars_max": max(reasoning, default=0),
            "content_chars_mean": round(statistics.mean(content)) if content else 0,
            "finish_reasons": dict(finish.most_common())}


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    for arg in sys.argv[1:]:
        s = run_stats(Path(arg))
        print(f"{s['run']}: {s['games']} games, {s['responses']} responses "
              f"(median {s['responses_per_game_median']} per game over {s['turns_per_game_median']} turns); "
              f"reasoning chars mean {s['reasoning_chars_mean']} p50 {s['reasoning_chars_p50']} "
              f"p90 {s['reasoning_chars_p90']} max {s['reasoning_chars_max']}; content chars mean "
              f"{s['content_chars_mean']}; finish {s['finish_reasons']}")


if __name__ == "__main__":
    main()
