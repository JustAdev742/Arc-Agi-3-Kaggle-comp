"""scripts/hc_dump_driver.py: snapshot planning (maximal requests, loop hygiene, duplicates, the game split) on
samples of real harness request logs (tests/fixtures/hc_dump_driver, from the bed) and synthetic ones, and the
replay against a fake SGLang-like server, including one that dumps through scripts/sglang_hc_dump_patch.py's Dumper.
CPU only."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hc_dump_driver as dd  # noqa: E402
import sglang_hc_dump_patch as hd  # noqa: E402

from tests.fidelity_fakes import FakeServer, write_fake_logs  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "hc_dump_driver"


def _requests(path: Path) -> list[dict]:
    return [r for r in map(json.loads, path.read_text().splitlines()) if r["event"] == "request"]


def _assistants(row: dict) -> list[dict]:
    return [m for m in row["messages"] if m["role"] == "assistant"]


# --- planning ---------------------------------------------------------------------------------------------------------


def test_plan_of_the_bed_logs():
    result = dd.plan(dd.discover([FIXTURES]), split="all")
    logs = {e["game"]: e for e in result["logs"]}
    assert sorted(logs) == ["ls20", "sb26"] and logs["ls20"]["split"] == "train" and logs["sb26"]["split"] == "holdout"
    for game, ordinals, turns in (("ls20", [5, 8, 13], 13), ("sb26", [5, 9], 8)):
        entry = logs[game]
        rows = _requests(FIXTURES / entry["file"])
        snaps = entry["snapshots"]
        assert [s["source"]["request_ordinal"] for s in snaps] == ordinals  # not extended by the next, and the last
        assert entry["turns"] == turns and entry["excluded_turns"] == {}
        seen = set()
        for s in snaps:
            row = rows[s["source"]["request_ordinal"]]
            assert s["n_assistant"] == len(_assistants(row)) == len(s["turns"])
            assert s["source"]["line"] == 2 * s["source"]["request_ordinal"] + 1  # request lines alternate with responses
            assert s["prompt_tokens_source"] == "logged" and s["prompt_tokens"] > 1000
            for t in s["turns"]:  # a turn is a loss span in the first snapshot that has it, and only there
                assert t["loss"] == (t["turn"] not in seen) and t["reasons"] == ([] if t["loss"] else ["covered"])
                assert t["span"] == s["turns"].index(t)
            seen |= {t["turn"] for t in s["turns"]}
        assert seen == set(range(turns))  # every logged turn is in some snapshot (only the last reply is not logged)
        assert sum(len(s["loss_spans"]) for s in snaps) == turns
    rids = [s["rid"] for s in result["replay"]]
    assert len(set(rids)) == len(rids) == 5 and all(hd.safe_rid(r) == r for r in rids)
    assert rids[0].startswith("hc-" + logs["ls20"]["sha256"][:8] + "-ls20-p0-r0005")
    assert result["totals"]["covered_turns"] == 2 and result["totals"]["loss_turns"] == 21


def test_split_games_and_limits():
    paths = dd.discover([FIXTURES, FIXTURES / "ls20-9607627b_p0_requests.jsonl"])
    assert len(paths) == 2  # a file listed twice counts once
    assert [e["game"] for e in dd.plan(paths)["logs"]] == ["ls20"]  # train: the fidelity-probe games are held out
    assert [e["game"] for e in dd.plan(paths, split="holdout")["logs"]] == ["sb26"]
    assert [e["game"] for e in dd.plan(paths, split="all", games={"sb26"})["logs"]] == ["sb26"]
    assert len(dd.plan(paths, split="all", max_snapshots=2)["replay"]) == 2
    keep = dd.plan(paths, split="all", keep_duplicates=True)
    assert keep["totals"]["loss_turns"] == 23 and keep["totals"]["covered_turns"] == 0
    assert sorted(dd.HOLDOUT) == ["ar25", "ft09", "lp85", "r11l", "re86", "sb26", "sc25", "tn36", "tr87", "tu93", "vc33"]


def _edit_turns(path: Path, edits: dict[str, dict]) -> None:
    """Rewrite every occurrence of the assistant messages whose first tool-call id is a key of EDITS."""
    lines = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        for m in row["messages"]:
            if m["role"] == "assistant" and m.get("tool_calls") and m["tool_calls"][0]["id"] in edits:
                m.update(edits[m["tool_calls"][0]["id"]])
        lines.append(json.dumps(row))
    path.write_text("\n".join(lines) + "\n")


def test_repeats_and_loop_mentions_leave_the_loss_and_an_empty_snapshot_is_dropped(tmp_path):
    logs = tmp_path / "logs"
    path = write_fake_logs(logs, games=("aa01",), n=16, trim_at=11)[0]
    twin = {"reasoning_content": "thinking about aa01 turn 2",
            "tool_calls": [{"id": "call7", "type": "function", "function": {"name": "python",
                                                                          "arguments": json.dumps({"code": "a(2)"})}}]}
    edits = {"call7": twin, "call9": {"reasoning_content": "Hmm. I'm Stuck in a loop here."}}
    edits.update({f"call{k}": {"reasoning_content": f"stuck in a loop again ({k})"} for k in (11, 12, 13, 14)})
    _edit_turns(path, edits)
    result = dd.plan([path], split="all")
    entry = result["logs"][0]
    assert entry["excluded_turns"] == {"repeat": 1, "stuck": 5}
    snaps = entry["snapshots"]
    assert [s["source"]["request_ordinal"] for s in snaps] == [10, 15]  # before the trim, and the last request
    by_turn = {t["turn"]: t for t in snaps[0]["turns"]}
    assert by_turn[7]["reasons"] == ["repeat"] and by_turn[9]["reasons"] == ["stuck"] and by_turn[2]["loss"]
    assert snaps[0]["loss_spans"] == [k for k in range(10) if k not in (7, 9)]
    assert snaps[1]["dropped"] == "no loss turn" and snaps[1]["loss_spans"] == []  # turns 11-14, all stuck
    assert [s["rid"] for s in result["replay"]] == [snaps[0]["rid"]]


def test_unanswered_requests_get_a_calibrated_estimate(tmp_path):
    logs = tmp_path / "logs"
    path = write_fake_logs(logs, games=("aa01",), n=16, trim_at=11, unanswered=15)[0]
    snap = dd.plan([path], split="all")["logs"][0]["snapshots"][-1]
    assert snap["source"]["request_ordinal"] == 15 and snap["prompt_tokens_source"] == "estimate"
    assert 0 < snap["prompt_tokens"] < 100_000


def test_dry_run_cli_prints_the_plan_and_writes_only_with_out(tmp_path):
    cmd = [sys.executable, "-I", str(ROOT / "scripts" / "hc_dump_driver.py"), "--logs", str(FIXTURES), "--dry-run"]
    run = subprocess.run([*cmd, "--split", "all"], capture_output=True, text=True, check=False, cwd=tmp_path)
    assert run.returncode == 0, run.stderr
    assert "ls20-9607627b_p0_requests.jsonl" in run.stdout and "5 of 5 snapshots to replay" in run.stdout
    assert list(tmp_path.iterdir()) == []
    run = subprocess.run([*cmd, "--out", str(tmp_path / "out")], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    summary = json.loads((tmp_path / "out" / "plan.json").read_text())
    assert summary["totals"]["games"] == ["ls20"] and summary["params"]["split"] == "train"
    lines = (tmp_path / "out" / "snapshots.jsonl").read_text().splitlines()
    assert [json.loads(x)["source"]["request_ordinal"] for x in lines] == [5, 8, 13]
    run = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "hc_dump_driver.py"), "--logs",
                          str(tmp_path / "empty"), "--dry-run"], capture_output=True, text=True, check=False)
    assert run.returncode == 1 and "no <game>-<id>_p<k>_requests.jsonl" in run.stderr


# --- replay -------------------------------------------------------------------------------------------------------------


def test_replay_sends_each_snapshot_once_as_the_harness_did_with_max_tokens_one(tmp_path):
    result = dd.plan(dd.discover([FIXTURES]), split="all")
    with FakeServer(delay=0.0) as server:
        summary = dd.replay(result, tmp_path / "out", base_url=server.url, health_wait=10, logger=lambda m: None)
        bodies, flushes = server.bodies, server.flushes
    assert summary["sent"] == summary["ok"] == len(bodies) == 5 and not summary["failed"]
    assert flushes == 5 and "prefix cache" in summary["notes"][0]  # the fake keeps its cache: flush every time
    for body, snap in zip(bodies, result["replay"]):
        row = _requests(FIXTURES / snap["source"]["file"])[snap["source"]["request_ordinal"]]
        assert body["rid"] == snap["rid"] and body["max_tokens"] == 1 and body["temperature"] == 0.0
        assert body["messages"] == [{k: v for k, v in m.items() if k != "_arc3_control"} for m in row["messages"]]
        assert body["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True}
        assert body["tools"] == row["tools"] and body["tool_choice"] == "auto" and body["model"] == "flashnext"
        assert body["logprobs"] is True and body["top_logprobs"] == 1 and body["stream"] is False
    assert any("_arc3_control" in m for b in result["replay"] for m in
               _requests(FIXTURES / b["source"]["file"])[b["source"]["request_ordinal"]]["messages"])
    lines = [json.loads(x) for x in (tmp_path / "out" / "replay.jsonl").read_text().splitlines()]
    assert [x["rid"] for x in lines] == [s["rid"] for s in result["replay"]] and all(x["ok"] for x in lines)
    assert (tmp_path / "out" / "plan.json").is_file() and (tmp_path / "out" / "replay-summary.json").is_file()


def fake_tokens(messages: list[dict]) -> list[int]:
    """A stand-in for the chat template + tokenizer: the headers the dump looks for, a few tokens of body."""
    ids = []
    for m in messages:
        body = [1000 + (len(json.dumps(m)) + j) % 997 for j in range(4 + len(json.dumps(m)) % 7)]
        header = {"assistant": [hd.IM_START, hd.ASSISTANT, hd.NEWLINE, hd.THINK, hd.NEWLINE],
                  "system": [hd.IM_START, 8678, hd.NEWLINE]}.get(m["role"], [hd.IM_START, 846, hd.NEWLINE])
        ids += header + body + [hd.IM_END, hd.NEWLINE]
    return ids + [hd.IM_START, hd.ASSISTANT, hd.NEWLINE, hd.THINK, hd.NEWLINE]


class DumpingServer(FakeServer):
    """A fake server that dumps every request through the real Dumper, as the patched SGLang would."""

    def __init__(self, dump_dir: Path, *, chunk: int = 16, gap: tuple = (), silent: tuple = (), **kwargs):
        super().__init__(delay=0.0, info={"disable_radix_cache": True, "max_running_requests": 1}, **kwargs)
        self.dumper = hd.Dumper(dump_dir, context=6)
        self.chunk, self.gap, self.silent = chunk, set(gap), set(silent)
        self.dump_lock = threading.Lock()

    def completion(self, body: dict) -> dict:
        response = super().completion(body)
        ids = fake_tokens(body["messages"])
        rid = body.get("rid")
        hc = np.random.default_rng(len(ids)).standard_normal((len(ids), 8)).astype(np.float32)
        with self.dump_lock:
            for lo in range(0, len(ids), self.chunk):
                if rid in self.silent or (rid in self.gap and lo == self.chunk):
                    continue
                self.dumper.process_chunk(rid, lo, np.array(ids[lo:lo + self.chunk]), hc[lo:lo + self.chunk])
        response["usage"]["prompt_tokens"] = len(ids)
        return response


def test_replay_against_a_dumping_server_keeps_exactly_the_loss_spans(tmp_path):
    result = dd.plan(dd.discover([FIXTURES]), split="all")
    dump = tmp_path / "dump"
    with DumpingServer(dump) as server:
        summary = dd.replay(result, tmp_path / "out", base_url=server.url, dump_dir=dump, health_wait=10,
                            logger=lambda m: None)
        flushes = server.flushes
    assert summary["sent"] == summary["ok"] == summary["dump_ok"] == 5 and not summary["failed"] and flushes == 0
    assert summary["dump_bytes"] == sum(r["bytes"] for r in hd.read_index(dump) if "bytes" in r)
    for snap in result["replay"]:
        plan = json.loads((dump / "plans" / f"{snap['rid']}.json").read_text())
        assert plan["loss_spans"] == snap["loss_spans"] and plan["n_assistant"] == snap["n_assistant"]
        got = hd.load_request(dump, snap["rid"])
        assert len(got["spans"]) == snap["n_assistant"] + 1
        with_rows = {k for k, (_, s, e, _, _) in got["spans"].items()
                     for p in got["hc_pos"][got["hc_role"] == 1].tolist() if s <= p <= e}
        assert with_rows == set(snap["loss_spans"])  # covered turns are not dumped again as span rows
    lines = [json.loads(x) for x in (tmp_path / "out" / "replay.jsonl").read_text().splitlines()]
    assert all(x["dump"]["ok"] and x["dump"]["chunks"] >= 2 for x in lines)
    assert sum(1 for s in result["replay"] for t in s["turns"] if not t["loss"]) == 2


def test_replay_stops_when_the_server_is_not_dumping(tmp_path):
    result = dd.plan(dd.discover([FIXTURES]), split="all")
    with FakeServer(delay=0.0) as server:
        summary = dd.replay(result, tmp_path / "out", base_url=server.url, dump_dir=tmp_path / "dump", health_wait=10,
                            logger=lambda m: None)
    assert summary["sent"] == 1 and summary["failed"] and "first request was not dumped" in summary["stopped"]
    assert (tmp_path / "dump" / "plans" / f"{result['replay'][0]['rid']}.json").is_file()
    line = json.loads((tmp_path / "out" / "replay.jsonl").read_text())
    assert line["ok"] and line["dump"]["problems"] == ["no dump lines for this request"]


def test_replay_records_incomplete_dumps_and_stops_after_three_in_a_row(tmp_path):
    result = dd.plan(dd.discover([FIXTURES]), split="all")
    rids = [s["rid"] for s in result["replay"]]
    with DumpingServer(tmp_path / "d1", gap=(rids[1],)) as server:
        summary = dd.replay(result, tmp_path / "o1", base_url=server.url, dump_dir=tmp_path / "d1", health_wait=10,
                            logger=lambda m: None)
    assert summary["sent"] == 5 and summary["dump_ok"] == 4 and not summary["failed"]
    bad = [json.loads(x) for x in (tmp_path / "o1" / "replay.jsonl").read_text().splitlines()][1]
    assert bad["rid"] == rids[1] and not bad["dump"]["ok"] and "expected 0" in bad["dump"]["problems"][0]
    with DumpingServer(tmp_path / "d2", silent=tuple(rids[1:])) as server:
        summary = dd.replay(result, tmp_path / "o2", base_url=server.url, dump_dir=tmp_path / "d2", health_wait=10,
                            logger=lambda m: None)
    assert summary["sent"] == 4 and summary["failed"] and summary["stopped"].startswith("three requests in a row")


def test_replay_stops_before_a_budget_would_be_passed(tmp_path):
    result = dd.plan(dd.discover([FIXTURES]), split="all")
    first = result["replay"][0]["prompt_tokens"]
    with DumpingServer(tmp_path / "d") as server:
        summary = dd.replay(result, tmp_path / "o", base_url=server.url, dump_dir=tmp_path / "d", health_wait=10,
                            max_dump_gb=1e-9, logger=lambda m: None)
        assert summary["sent"] == 0 and "--max-dump-gb" in summary["stopped"] and not summary["failed"]
        summary = dd.replay(result, tmp_path / "o2", base_url=server.url, max_prefill_tokens=first + 1,
                            logger=lambda m: None)
        assert summary["sent"] == 1 and "--max-prefill-tokens" in summary["stopped"]
        summary = dd.replay(result, tmp_path / "o3", base_url=server.url, max_minutes=0, logger=lambda m: None)
        assert summary["sent"] == 0 and "--max-minutes" in summary["stopped"]


def test_check_dump_finds_gaps_wrong_lengths_and_header_mismatches():
    def chunk(start, n, opened, attempt=0):
        return {"file": "f", "attempt": attempt, "start": start, "n": n, "kept": 1, "span_rows": 1, "bytes": 10,
                "qerr_max": 0.02, "spans_opened_total": opened}

    ok, stats = dd.check_dump([chunk(0, 10, 1), chunk(10, 5, 3)], 15, 2)
    assert ok == [] and stats["chunks"] == 2 and stats["bytes"] == 20
    assert "expected 10" in dd.check_dump([chunk(0, 10, 1), chunk(12, 3, 3)], 15, 2)[0][0]
    assert "server counted 16" in dd.check_dump([chunk(0, 10, 1), chunk(10, 5, 3)], 16, 2)[0][0]
    assert "3 assistant headers" in dd.check_dump([chunk(0, 15, 3)], 15, 1)[0][0]
    assert dd.check_dump([chunk(0, 4, 0), chunk(0, 15, 3, attempt=1)], 15, 2)[0] == []  # the latest attempt counts
    problems, _ = dd.check_dump([{"event": "error", "error": "boom"}], 15, 2)
    assert problems == ["dump error: boom", "no dump lines for this request"]


def test_a_log_changed_after_planning_is_refused(tmp_path):
    path = tmp_path / "ls20-9607627b_p0_requests.jsonl"
    path.write_text((FIXTURES / path.name).read_text())
    result = dd.plan([path], split="all")
    lines = path.read_text().splitlines()
    row = json.loads(lines[10])
    row["messages"][0]["content"] = "changed"
    lines[10] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(RuntimeError, match="changed since it was planned"):
        list(dd.iter_rows(result))


def test_the_cli_replays_and_exits_non_zero_when_nothing_is_dumped(tmp_path):
    with DumpingServer(tmp_path / "dump") as server:
        code = dd.main(["--logs", str(FIXTURES), "--split", "holdout", "--out", str(tmp_path / "out"), "--base-url",
                        server.url, "--dump-dir", str(tmp_path / "dump"), "--health-wait", "10"])
        assert code == 0 and len(server.bodies) == 2
    t0 = time.time()
    with FakeServer(delay=0.0) as server:
        code = dd.main(["--logs", str(FIXTURES), "--out", str(tmp_path / "o2"), "--base-url", server.url,
                        "--dump-dir", str(tmp_path / "nodump"), "--health-wait", "10"])
    assert code == 1 and time.time() - t0 < 60


def test_the_dump_cap_counts_what_an_earlier_run_wrote_to_the_same_dump(tmp_path):
    # session A dumps the held-out games and then the train split into one directory, one driver run each, with a
    # cumulative --max-dump-gb: the cap must count the earlier runs' bytes before this run's first snapshot
    result = dd.plan(dd.discover([FIXTURES]), split="all")
    one = dd.plan(dd.discover([FIXTURES]), split="all", max_snapshots=1)
    dump = tmp_path / "dump"
    with DumpingServer(dump) as server:
        first = dd.replay(one, tmp_path / "o1", base_url=server.url, dump_dir=dump, health_wait=10,
                          logger=lambda m: None)
        written = first["dump_bytes"]
        est = result["replay"][0]["kept_rows_est"] * dd.ROW_BYTES
        cap = (written + est - 1) / 1e9  # the next snapshot alone fits; with what is already there it does not
        second = dd.replay(result, tmp_path / "o2", base_url=server.url, dump_dir=dump, health_wait=10,
                           max_dump_gb=cap, logger=lambda m: None)
    assert first["sent"] == first["dump_ok"] == 1 and written > 0
    assert second["sent"] == 0 and "--max-dump-gb" in second["stopped"] and not second["failed"]
