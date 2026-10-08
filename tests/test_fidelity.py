"""The fidelity probe (docs/research/beat-tufa/fidelity-probe.md): sampling logged requests, replaying them against a
server, and comparing runs. All on CPU, against synthetic logs and a fake SGLang-like server (tests/fidelity_fakes.py)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fidelity_compare as fc  # noqa: E402
import fidelity_probe as fp  # noqa: E402
import fidelity_sample as fs  # noqa: E402

from tests.fidelity_fakes import FakeServer, write_fake_logs  # noqa: E402


def _dataset(tmp_path: Path, per_game: int = 8) -> tuple[Path, dict]:
    logs = tmp_path / "logs"
    write_fake_logs(logs)
    out = tmp_path / "dataset"
    return out, fs.sample(logs, out, per_game=per_game, seed=7)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- sampling ------------------------------------------------------------------------------------------------------


def test_sampler_stratifies_by_turn_and_keeps_requests_as_sent(tmp_path):
    out, manifest = _dataset(tmp_path)
    rows = _rows(out / "requests.jsonl")
    assert manifest["requests"] == len(rows) == 16 and manifest["per_game"] == {"aa01": 8, "bb02": 8}
    assert manifest["per_quarter"] == {"0": 4, "1": 4, "2": 4, "3": 4}  # 2 per quarter per game
    assert [r["id"] for r in rows] == sorted(r["id"] for r in rows)  # replay order: by game, then by turn
    assert all(r["source"]["request_ordinal"] != 5 for r in rows)  # the unanswered request is never drawn
    logs = {p.name: [json.loads(x) for x in p.read_text().splitlines()] for p in (tmp_path / "logs").glob("*_p0_*")}
    for r in rows:
        logged = [x for x in logs[r["source"]["file"]] if x["event"] == "request"][r["source"]["request_ordinal"]]
        # the wire messages: the logged ones without the harness's private key, nothing else changed
        assert r["request"]["messages"] == fs.wire_messages(logged["messages"])
        assert not any("_arc3_control" in m for m in r["request"]["messages"])
        assert r["request"]["tools"] == logged["tools"] and r["request"]["tool_choice"] == "auto"
        assert r["request"]["chat_template_kwargs"] == {"preserve_thinking": True}
        assert r["stats"]["n_images"] == sum(1 for m in logged["messages"] if isinstance(m.get("content"), list))
        reply = r["logged"]["reply"]
        if r["source"]["request_ordinal"] in (10, 15):  # before the trim, and the last request: no next request extends
            assert reply is None
        else:
            assert reply["role"] == "assistant" and f"turn {r['source']['request_ordinal']}" in reply["reasoning_content"]
    assert any("_arc3_control" in json.dumps(x) for rows_ in logs.values() for x in rows_)  # the fixture has some
    # sha256 and size in the manifest are the file's; a second draw with the same seed is byte-identical
    assert manifest["files"][0]["sha256"] == fs.sha256_file(out / "requests.jsonl")
    again = fs.sample(tmp_path / "logs", tmp_path / "again", per_game=8, seed=7)
    assert again["files"][0]["sha256"] == manifest["files"][0]["sha256"]
    meta = json.loads((out / "dataset-metadata.json").read_text())
    assert meta["id"] == "scottmahony/arc3-fidelity-prompts" and meta["licenses"] == [{"name": "other"}]
    with pytest.raises(ValueError, match="fewer than --per-game"):
        fs.sample(tmp_path / "logs", tmp_path / "x", per_game=40)


def test_sampler_takes_several_log_folders_but_one_log_per_game(tmp_path):
    write_fake_logs(tmp_path / "a", games=("aa01",))
    write_fake_logs(tmp_path / "b", games=("cc03",))
    manifest = fs.sample([tmp_path / "a", tmp_path / "b"], tmp_path / "ds", per_game=4, notes=["a: one run", "b"])
    assert manifest["per_game"] == {"aa01": 4, "cc03": 4} and manifest["source"]["notes"] == ["a: one run", "b"]
    assert [(f["dir"], f["name"][:4]) for f in manifest["source"]["files"]] == [("a", "aa01"), ("b", "cc03")]
    write_fake_logs(tmp_path / "c", games=("aa01",))
    with pytest.raises(ValueError, match="more than one log"):
        fs.sample([tmp_path / "a", tmp_path / "c"], tmp_path / "ds2", per_game=4)


def test_sampler_cli_runs_isolated(tmp_path):
    logs = tmp_path / "logs"
    write_fake_logs(logs)
    run = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "fidelity_sample.py"), "--logs", str(logs),
                          "--out", str(tmp_path / "ds"), "--per-game", "4", "--copy-meta", str(tmp_path / "meta")],
                         capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    assert "8 requests" in run.stdout
    assert sorted(p.name for p in (tmp_path / "meta").iterdir()) == ["dataset-metadata.json", "manifest.json"]


# --- the probe -----------------------------------------------------------------------------------------------------


def test_probe_replays_every_request_greedy_with_logprobs_one_pass_after_another(tmp_path):
    out, _ = _dataset(tmp_path)
    data = out / "requests.jsonl"
    with FakeServer(delay=0.05) as server:
        result = fp.run([data], tmp_path / "fidelity.json", base_url=server.url, model="flashnext", arm="base",
                        concurrency=4, health_deadline=time.time() + 10, logger=lambda m: None)
        bodies, flushes, max_in_flight = server.bodies, server.flushes, server.max_in_flight
    saved = json.loads((tmp_path / "fidelity.json").read_text())
    assert saved["arm"] == "base" and saved["params"]["extras"] == {"return_token_ids": True, "return_meta_info": True}
    samples = _rows(data)
    assert len(bodies) == 1 + 2 * len(samples)  # warm-up + two passes
    for body in bodies[1:]:
        assert body["temperature"] == 0.0 and body["max_tokens"] == 192 and body["logprobs"] is True
        assert body["top_logprobs"] == 5 and body["model"] == "flashnext" and body["stream"] is False
        assert body["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True}
        assert body["tool_choice"] == "auto" and body["tools"][0]["function"]["name"] == "python"
        assert body["top_p"] == 0.95 and body["top_k"] == 20 and "separate_reasoning" not in body
    assert bodies[0]["max_tokens"] == 8  # the warm-up
    assert [b["messages"] for b in bodies[1:1 + len(samples)]] == [s["request"]["messages"] for s in samples]
    assert flushes == 2  # the prefix cache is emptied before each pass
    seq, conc = saved["passes"]["seq"], saved["passes"]["conc"]
    assert seq["max_in_flight"] == 1 and 1 < conc["max_in_flight"] <= 4 and 1 < max_in_flight <= 4
    for p in (seq, conc):
        assert p["ok"] == len(samples) and [r["id"] for r in p["records"]] == [s["id"] for s in samples]
        rec = p["records"][0]
        assert len(rec["token_ids"]) == len(rec["tokens"]) == len(rec["logprobs"]) == 24
        assert rec["top"][0][0][0] == rec["token_ids"][0] and len(rec["top"][0]) == 5  # [id, text, logprob]
        assert rec["spec"]["spec_accept_length"] == 2.5 and rec["finish_reason"] == "length"
    assert saved["within_run_identical"] == {"compared": len(samples), "identical": len(samples), "by": "token_ids"}
    assert [s["id"] for s in saved["samples"]] == [s["id"] for s in samples]
    assert saved["samples"][0]["prompt_tokens"] > 0 and "quarter" in saved["samples"][0]
    assert result["timing"]["total_s"] >= 0


def test_probe_falls_back_to_plain_logprobs_when_the_server_rejects_the_extras(tmp_path):
    out, _ = _dataset(tmp_path)
    with FakeServer(reject=("return_meta_info",)) as server:
        fp.run([out / "requests.jsonl"], tmp_path / "f.json", base_url=server.url, model="m", arm="base",
               passes=("seq",), health_deadline=time.time() + 10, logger=lambda m: None)
    saved = json.loads((tmp_path / "f.json").read_text())
    assert saved["params"]["extras"] == {"return_token_ids": True}
    assert [w["ok"] for w in saved["warmup"]] == [False, True]
    rec = saved["passes"]["seq"]["records"][0]
    assert rec["token_ids"] and rec["top"][0][0][0] is None and rec["spec"] is None  # ids, but no ids in the top-k
    with FakeServer(reject=("return_meta_info", "return_token_ids", "logprobs")) as server, \
            pytest.raises(RuntimeError, match="no logprobs"):
        fp.run([out / "requests.jsonl"], tmp_path / "g.json", base_url=server.url, model="m", arm="base",
               health_deadline=time.time() + 10, logger=lambda m: None)


def test_warm_up_retries_a_transient_error_before_giving_up_the_extras(tmp_path):
    out, _ = _dataset(tmp_path)
    sample = _rows(out / "requests.jsonl")[0]
    with FakeServer(fail_first=2) as server:
        extras, tried = fp.choose_extras(server.url, sample, model="m", top_logprobs=5, timeout=10, retry_s=0.05,
                                         logger=lambda m: None)
    assert extras == fp.EXTRAS_LADDER[0] and [t["status"] for t in tried] == [503, 503, 200]
    with FakeServer(fail_first=3) as server:  # still failing after the retries: the next extras are tried
        extras, tried = fp.choose_extras(server.url, sample, model="m", top_logprobs=5, timeout=10, retry_s=0.05,
                                         logger=lambda m: None)
    assert extras == fp.EXTRAS_LADDER[1] and [t["status"] for t in tried] == [503, 503, 503, 200]


def test_probe_stops_when_the_server_dies_or_never_becomes_healthy(tmp_path):
    out, _ = _dataset(tmp_path)
    called = []
    with FakeServer() as server, pytest.raises(RuntimeError, match="exited before it became healthy"):
        fp.run([out / "requests.jsonl"], tmp_path / "f.json", base_url=server.url, model="m", arm="base",
               alive=lambda: False, on_failure=lambda: called.append(1), logger=lambda m: None)
    assert called == [1]
    with FakeServer(healthy_after=60) as server, pytest.raises(RuntimeError, match="not healthy by the deadline"):
        fp.run([out / "requests.jsonl"], tmp_path / "f.json", base_url=server.url, model="m", arm="base",
               health_deadline=time.time() + 1, health_poll_s=0.2, on_failure=lambda: called.append(2),
               logger=lambda m: None)
    assert called == [1, 2] and not (tmp_path / "f.json").exists()
    with FakeServer(healthy_after=1.0) as server:  # a slow start is waited for
        assert fp.wait_healthy(server.url, deadline=time.time() + 20, poll_s=0.2, logger=lambda m: None) >= 0.5


def test_probe_refuses_a_server_that_is_not_its_arm(tmp_path):
    out, _ = _dataset(tmp_path)
    reap_info = {"json_model_override_args": '{"text_config": {"num_experts": 448}}'}
    for info, expect, message in [({}, 448, "does not match this arm"), (reap_info, None, "does not match this arm"),
                                  ({"speculative_accept_threshold_acc": 0.5}, None, "lossless")]:
        with FakeServer(info=info) as server, pytest.raises(RuntimeError, match=message):
            fp.run([out / "requests.jsonl"], tmp_path / "f.json", base_url=server.url, model="m", arm="x",
                   expect_num_experts=expect, health_deadline=time.time() + 10, logger=lambda m: None)
    with FakeServer(info=reap_info) as server:
        fp.run([out / "requests.jsonl"], tmp_path / "f.json", base_url=server.url, model="m", arm="reap448",
               passes=("seq",), expect_num_experts=448, health_deadline=time.time() + 10, logger=lambda m: None)
    assert json.loads((tmp_path / "f.json").read_text())["server_info"]["json_model_override_args"] == reap_info[
        "json_model_override_args"]
    assert fp.check_server({}, None) == ["speculative_accept_threshold_single not reported",
                                         "speculative_accept_threshold_acc not reported",
                                         "json_model_override_args not reported; the arm is not verified"]


def test_find_dataset_checks_the_hash_in_either_mount_layout(tmp_path):
    new = tmp_path / "kaggle/input/datasets/scottmahony/arc3-fidelity-prompts"
    old = tmp_path / "kaggle/input/arc3-fidelity-prompts"
    assert fp.mount_alternatives("/kaggle/input/datasets/scottmahony/arc3-fidelity-prompts") == [
        "/kaggle/input/datasets/scottmahony/arc3-fidelity-prompts", "/kaggle/input/arc3-fidelity-prompts"]
    old.mkdir(parents=True)
    (old / "requests.jsonl").write_text("{}\n")
    sha = fp.sha256_file(old / "requests.jsonl")
    assert fp.find_dataset([old], {"requests.jsonl": sha}, wait_s=0, logger=lambda m: None) == [old / "requests.jsonl"]
    with pytest.raises(RuntimeError, match="not the manifest's"):
        fp.find_dataset([old], {"requests.jsonl": "0" * 64}, wait_s=0, logger=lambda m: None)
    with pytest.raises(RuntimeError, match="not found"):
        fp.find_dataset([new], {"requests.jsonl": sha}, wait_s=0, logger=lambda m: None)


# --- comparing runs ------------------------------------------------------------------------------------------------


def _record(rid: str, ids: list[int], lps: list[float], second: list[int] | None = None, gap: float = 0.03) -> dict:
    """A greedy completion: chosen ids with logprobs; the runner-up at each position is ``second`` (default id+1)."""
    second = second or [i + 1 for i in ids]
    top = [[[i, f"t{i}", lp], [s, f"t{s}", lp - gap], [9000 + j, f"t{9000 + j}", -5.0]]
           for j, (i, s, lp) in enumerate(zip(ids, second, lps))]
    return {"id": rid, "ok": True, "token_ids": ids, "tokens": [f"t{i}" for i in ids], "logprobs": lps, "top": top,
            "finish_reason": "length"}


def test_compare_records_measures_the_agreed_prefix_and_the_divergence():
    x = _record("a", [1, 2, 3, 4], [-0.1, -0.2, -0.3, -0.4])
    y = _record("a", [1, 2, 5, 6], [-0.12, -0.2, -0.31, -0.4], second=[2, 7, 3, 7])
    row = fc.compare_records(x, y)
    assert row["prefix"] == 2 and not row["identical"] and row["prefix_share"] == 0.5 and row["by_ids"]
    assert row["dlp"] == pytest.approx([0.02, 0.0])
    assert row["rank2_flips"] == 1 and row["topset_changes"] == 1  # position 1: runner-up 3 vs 7
    # at position 2, x chose 3 and had 5 nowhere in its top-k; y chose 5 and had 3 at -0.34: margin 0.03
    assert row["margin_x"] is None and row["margin_y"] == pytest.approx(0.03)
    same = fc.compare_records(x, x)
    assert same["identical"] and same["prefix_share"] == 1.0 and same["margin_x"] is None
    shorter = fc.compare_records(x, _record("a", [1, 2, 3], [-0.1, -0.2, -0.3]))
    assert not shorter["identical"] and shorter["prefix"] == 3
    by_text = fc.compare_records({**x, "token_ids": None}, y)
    assert not by_text["by_ids"] and by_text["prefix"] == 2


def _run(arm: str, passes: dict[str, list[dict]], prompt_tokens: int = 30000) -> dict:
    ids = sorted({r["id"] for recs in passes.values() for r in recs})
    return {"probe": 1, "arm": arm, "params": {"files": {"requests.jsonl": "abc"}},
            "samples": [{"id": i, "game": i.split("#")[0], "prompt_tokens": prompt_tokens * (k % 4 + 1),
                         "n_images": 5 * k, "last_has_image": k % 2 == 0, "quarter": k % 4}
                        for k, i in enumerate(ids)],
            "server_info": {"json_model_override_args": "{}"},
            "passes": {name: {"ok": len(recs), "failed": 0, "records": recs} for name, recs in passes.items()}}


def _runs(n: int = 30, reap_prefix: int = 4, reap_shift: float = 0.2, floor_prefix: int = 15) -> list[dict]:
    """A base run whose conc pass diverges late with tiny logprob noise, and a REAP run that diverges at
    ``reap_prefix`` with logprobs shifted by ``reap_shift``."""
    base_seq, base_conc, reap_seq, reap_conc = [], [], [], []
    for k in range(n):
        rid = f"{'g1' if k % 2 else 'g2'}#{k:03d}"
        ids = [100 + k * 50 + j for j in range(20)]
        lps = [-0.1 - 0.01 * j for j in range(20)]
        base_seq.append(_record(rid, ids, lps))
        alt = ids[:floor_prefix] + [7 + j for j in range(20 - floor_prefix)]
        base_conc.append(_record(rid, alt, [lp + 0.001 for lp in lps]))
        moved = ids[:reap_prefix] + [3000 + j for j in range(20 - reap_prefix)]
        reap_seq.append(_record(rid, moved, [lp - reap_shift for lp in lps]))
        reap_conc.append(_record(rid, moved, [lp - reap_shift for lp in lps]))
    return [_run("base", {"seq": base_seq, "conc": base_conc}), _run("reap448", {"seq": reap_seq, "conc": reap_conc})]


def test_compare_reads_a_pruned_server_that_moves_against_the_floor(tmp_path):
    paths = []
    for run in _runs():
        paths.append(tmp_path / f"{run['arm']}.json")
        paths[-1].write_text(json.dumps(run))
    report = fc.analyze(paths, n_boot=300)
    kinds = {(p["a"], p["b"]): p["kind"] for p in report["pairs"]}
    assert kinds[("base.seq", "base.conc")] == "floor" and kinds[("base.seq", "reap448.seq")] == "reap"
    assert kinds[("reap448.seq", "reap448.conc")] == "reap-own" and kinds[("base.seq", "reap448.conc")] == "reap-mixed"
    h = report["headline"]
    assert h["floor_pair"] == "base.seq vs base.conc" and h["reap_pair"] == "base.seq vs reap448.seq"
    assert h["floor"]["mean_prefix_share"] == pytest.approx(0.75) and h["reap"]["mean_prefix_share"] == pytest.approx(0.2)
    assert h["floor"]["mean_abs_dlp"] == pytest.approx(0.001) and h["reap"]["mean_abs_dlp"] == pytest.approx(0.2)
    assert h["prefix_share_diff_ci"]["mean"] == pytest.approx(-0.55) and h["prefix_share_diff_ci"]["hi"] < 0
    assert h["verdict"].startswith("REAP SHIFTS THE MODEL")
    assert set(h["breakdowns"]) == {"game", "context", "images", "last message", "turn quarter"}
    assert set(h["breakdowns"]["game"]) == {"g1", "g2"} and h["breakdowns"]["game"]["g1"]["reap"]["n"] == 15
    text = fc.render(report)
    assert "HEADLINE: base.seq vs reap448.seq" in text and "REAP SHIFTS THE MODEL" in text
    # a third file (a second base run) gives the across-run floor, used by the headline
    second = _runs()[0]
    paths.append(tmp_path / "base2.json")
    paths[-1].write_text(json.dumps(second))
    h = fc.analyze(paths, n_boot=100)["headline"]
    assert h["floor_pair"] == "base.seq vs base#2.seq" and h["floor"]["identical_share"] == 1.0


def test_compare_reads_noise_level_changes_as_close_to_the_floor(tmp_path):
    runs = _runs(reap_prefix=15, reap_shift=0.0015)
    paths = []
    for run in runs:
        paths.append(tmp_path / f"{run['arm']}.json")
        paths[-1].write_text(json.dumps(run))
    h = fc.analyze(paths, n_boot=300)["headline"]
    assert h["verdict"].startswith("REAP is CLOSE TO THE FLOOR")
    assert fc.verdict({"n": 5, "mean_abs_dlp": 0.0025, "mean_prefix_share": 0.6},
                      {"n": 5, "mean_abs_dlp": 0.001, "mean_prefix_share": 0.7}, None).startswith("a SMALL")


def test_compare_cli_writes_json(tmp_path):
    paths = []
    for run in _runs(n=6):
        paths.append(tmp_path / f"{run['arm']}.json")
        paths[-1].write_text(json.dumps(run))
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "fidelity_compare.py"), *map(str, paths),
                          "--json", str(tmp_path / "cmp.json"), "--bootstrap", "50"],
                         capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    assert "READING:" in run.stdout
    assert json.loads((tmp_path / "cmp.json").read_text())["headline"]["reap"]["n"] == 6


def test_probe_output_feeds_the_comparison_end_to_end(tmp_path):
    """Two fake servers, one choosing its runner-up from token 3 on, through the probe and the comparison."""
    out, _ = _dataset(tmp_path)
    results = []
    for arm, kwargs in (("base", {}), ("reap448", {"diverge_at": 3, "shift": -0.05,
                                                   "info": {"json_model_override_args":
                                                            '{"text_config": {"num_experts": 448}}'}})):
        with FakeServer(**kwargs) as server:
            fp.run([out / "requests.jsonl"], tmp_path / f"{arm}.json", base_url=server.url, model="m", arm=arm,
                   concurrency=3, expect_num_experts=448 if arm != "base" else None,
                   health_deadline=time.time() + 10, logger=lambda m: None)
        results.append(tmp_path / f"{arm}.json")
    report = fc.analyze(results, n_boot=100)
    h = report["headline"]
    assert h["floor"]["identical_share"] == 1.0  # the fake server has no batching noise
    assert h["reap"]["median_first_divergence"] == 3 and h["reap"]["mean_abs_dlp"] == pytest.approx(0.05)
    assert h["reap"]["near_tie_share"] == 1.0  # the fake's runner-up is within 0.09 nats
    assert h["verdict"].startswith("REAP SHIFTS THE MODEL")
