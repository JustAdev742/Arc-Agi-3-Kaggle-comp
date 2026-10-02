"""scripts/franzen_report.py on an extract of the real output of Daniel Franzen's notebook (tests/fixtures/franzen/
run-extract, see its NOTICE.md): his Kaggle run of 2026-09-30, 10 demo games for 25 minutes each.

The per-game numbers asserted for ft09 are the same as on the full downloaded output (2026-10-02): the extract keeps
ft09's request log with each message replaced by a stub derived from its hash, so counts, usage and prefix breaks are
unchanged. The server numbers are those of the extract's first 240 serve.log lines.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import franzen_report as fr  # noqa: E402

RUN = ROOT / "tests" / "fixtures" / "franzen" / "run-extract"


@pytest.fixture(scope="module")
def report():
    return fr.analyze(RUN)


def test_scores_levels_and_actions_match_taaf_and_his_summary(report):
    games = report["games"]
    assert len(games) == 10 and all(g["score_match"] for g in games.values())
    assert report["totals"]["mean_score_ours"] == 36.56 and "mean score:    36.56" in report["summary"]
    assert report["totals"]["levels"] == 40 and report["totals"]["actions"] == 1327
    sb26 = games["sb26-7fbdac44"]
    assert (sb26["state"], sb26["levels"], sb26["total_levels"], sb26["score_ours"]) == ("won", 8, 8, 93.34)
    tu93 = games["tu93-0768757b"]  # completed levels 48/19, 15/16, 19/34 -> 15.7, 113.8, 115 (capped), weights 1-3
    assert tu93["score_ours"] == 13.07 and tu93["per_level"][:4] == ["48/19", "15/16", "19/34", "25/42"]
    assert tu93["actions_completed_levels"] == 82 and tu93["resets"] == 3


def test_requests_tokens_cache_and_gate_admissions(report):
    ft09 = report["requests"]["ft09-0d8bbf25"]
    assert (ft09["requests"], ft09["responses"], ft09["turns"]) == (35, 34, 15)  # one call timed out
    assert (ft09["prompt_tokens"], ft09["generated_tokens"], ft09["cached_tokens"]) == (1689100, 95626, 1524416)
    assert ft09["cache_share"] == 0.9025 and ft09["finish_reasons"] == {"tool_calls": 33, "length": 1}
    assert ft09["prefix_breaks"] == 1 and ft09["gate_admissions"] == 2


def test_server_log(report):
    s = report["serve"]
    assert (s["decode_lines"], s["decode_tok_s_mean"], s["decode_tok_s_p90"]) == (62, 713.4, 848.7)
    assert (s["running_req_mean"], s["accept_len_mean"], s["queue_req_max"]) == (9.84, 2.6, 1)
    assert (s["requests_finished"], s["input_tokens"], s["cached_input_tokens"]) == (58, 550142, 432384)
    assert s["http"]["/v1/chat/completions 200"] == 56 and s["http"]["/health 503"] == 2


def test_notebook_log(report):
    lg = report["log"]
    assert lg["gate_slots"] == 10 and lg["tail_fade"].startswith("tail fade phase after 20.5 minutes")
    assert (lg["read_timeouts"], lg["warmup_resets"], lg["gate_diag"]) == (5, 1, None)
    assert len(lg["finished"]) == 10 and lg["finished"]["sb26-7fbdac44"]["score"] == 93.34


def _write_requests(path: Path, conversations: list[list[str]]) -> None:
    rows = []
    for i, contents in enumerate(conversations, start=1):
        messages = [{"role": "user", "content": c} for c in contents]
        rows.append({"event": "request", "messages": messages, "analysis_step": i})
        rows.append({"event": "response", "messages": messages, "analysis_step": i, "finish_reason": "tool_calls",
                     "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                               "prompt_tokens_details": {"cached_tokens": 50 * (i > 1)}}})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_prefix_breaks_count_trims_only(tmp_path):
    path = tmp_path / "g_p0_requests.jsonl"
    _write_requests(path, [["s", "u1"], ["s", "u1", "a1", "t1"], ["s", "u1", "a1", "t1", "u2"],
                           ["s", "u2"], ["s", "u2", "a2"], ["s", "u3"]])
    stats = fr.request_stats(path)
    assert stats["requests"] == 6 and stats["prefix_breaks"] == 2 and stats["cached_tokens"] == 250


def test_serve_log_parsing():
    import tempfile
    lines = [
        "[2026-09-30 08:30:35] Decode batch, #running-req: 9, #full token: 11584, full token usage: 0.01, mamba num: 36, "
        "mamba usage: 0.60, accept len: 3.04, accept rate: 0.68, cuda graph: True, gen throughput (token/s): 22.40, "
        "#queue-req: 1",
        "[2026-09-30 08:30:15] Prefill batch, #new-seq: 1, #new-token: 64, #cached-token: 128, full token usage: 0.00, "
        "mamba usage: 0.05, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, "
        "input throughput (token/s): 7.34",
        "[2026-09-30 08:54:56] ReqTimeStats(rid=bde3, input_len=1000, cached_input_len=900, output_len=50, attempts=0, "
        "type=unified): queue_duration=1.56ms, initial_prefill_elapsed=5.0ms, post_prefill_elapsed=95.0ms, "
        "forward_duration=100.0ms, entry_time=1790758489.264",
        "[2026-09-30 08:54:58] ReqTimeStats(rid=bde4, input_len=1000, cached_input_len=0, output_len=150, attempts=0, "
        "type=unified): queue_duration=1.56ms, initial_prefill_elapsed=20.0ms, post_prefill_elapsed=80.0ms, "
        "forward_duration=100.0ms, entry_time=1790758489.264",
        '[2026-09-30 08:54:58] INFO:     127.0.0.1:53640 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request',
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "serve.log"
        path.write_text("\n".join(lines) + "\n")
        s = fr.serve_stats(path)
    assert s["decode_tok_s_mean"] == 22.4 and s["running_req_mean"] == 9 and s["accept_len_mean"] == 3.04
    assert (s["prefill_new_tokens"], s["prefill_cached_tokens"]) == (64, 128)
    assert s["server_cache_share"] == 0.45 and s["output_tok_s_over_span"] == 100.0 and s["prefill_time_share"] == 0.125
    assert s["http"] == {"/v1/chat/completions 400": 1}


def test_cli_prints_the_table_and_writes_json(tmp_path):
    out = tmp_path / "r.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "franzen_report.py"), str(RUN), "--json", str(out)],
                       capture_output=True, text=True, check=True)
    assert "ft09-0d8bbf25" in r.stdout and "prefix-cache hit share 90.2%" in r.stdout
    assert json.loads(out.read_text())["totals"]["games"] == 10
