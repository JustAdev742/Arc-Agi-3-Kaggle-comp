"""scripts/mtp_session_a.py: the steps A0-A11 of the MTP draft fine-tune's session A (plan section 10.4).

Pure functions (the dump servers' arguments and environment, the storage plan, input lookup in both mount layouts,
the training watch), then whole sessions on CPU: the real module drives stub scripts (which record their command
lines and write what the real ones write) and a stub ``sglang serve`` (an HTTP server that answers /health and
/server_info from its own command line), through GO, NO-GO and a failed step. Every command line the module builds
is checked against the real script's argparse options. No GPU, no network."""
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mtp_session_a as sa  # noqa: E402

REAP_OVERRIDE = '{"text_config": {"num_experts": 448}}'


def launcher_args(sglang: str = "/tmp/sgl-intel/venv/bin/sglang", port: int = 8001, reap: bool = True) -> list[str]:
    """The shape of D' cell 12's arguments (scripts/build_mtp_session.py's test executes the real cell for them)."""
    args = [sglang, "serve", "--model-path", "/kaggle/input/models/x/1", "--load-format", "safetensors",
            "--model-loader-extra-config", '{"enable_multithread_load":false}', "--served-model-name", "flashnext",
            "--host", "127.0.0.1", "--port", str(port), "--max-running-requests", "10", "--chunked-prefill-size",
            "8192", "--max-prefill-tokens", "16384", "--cuda-graph-max-bs-decode", "10", "--cuda-graph-bs-decode",
            "1", "2", "4", "7", "8", "9", "10", "--mamba-radix-cache-strategy", "extra_buffer",
            "--mamba-track-interval", "64", "--ple-offload-embedding", "--trust-remote-code",
            "--default-chat-template-kwargs", '{"preserve_thinking":true}', "--schedule-policy", "lpm",
            "--enable-cache-report"]
    if reap:
        args += ["--json-model-override-args", REAP_OVERRIDE, "--speculative-draft-model-override-args", "{}"]
    args += ["--weight-loader-prefetch-checkpoints", "--gdn-mtp-cache-mode", "none",
             "--speculative-algorithm", "NEXTN", "--speculative-num-steps", "3", "--speculative-eagle-topk", "1",
             "--speculative-num-draft-tokens", "4", "--speculative-draft-model-path", "/tmp/sgl-intel/draft-view-x",
             "--speculative-draft-kv-cache-dtype", "fp8_e4m3", "--speculative-accept-threshold-single", "1.0",
             "--speculative-token-map", "/kaggle/input/pennyroyal/hot_tokens_64k.pt"]
    return args


# ------------------------------------------------------------------------------------------- the dump servers


def test_dump_server_args_are_the_launchers_minus_speculation_plus_the_dump_flags():
    launcher = launcher_args()
    probe = sa.dump_server_args(launcher, reap=False)
    train = sa.dump_server_args(launcher, reap=True)
    _, items = sa.split_flags(launcher)
    expected_probe = []
    for flag, values in items:
        if flag.startswith("--speculative-") or flag == "--json-model-override-args":
            continue
        expected_probe += [flag, *({"--max-running-requests": ["1"], "--chunked-prefill-size": ["8192"]}
                                   .get(flag, values))]
    assert probe == launcher[:2] + expected_probe + ["--disable-cuda-graph", "--disable-radix-cache"]
    assert sa.args_diff(launcher, probe) == {
        "removed": [[f, *v] for f, v in items if f.startswith("--speculative-") or f == "--json-model-override-args"],
        "added": [["--disable-cuda-graph"], ["--disable-radix-cache"]],
        "changed": [["--max-running-requests", "10", "->", "1"]]}
    # the training dump keeps REAP's override (the served target) and only that
    diff = sa.args_diff(probe, train)
    assert diff == {"removed": [], "added": [["--json-model-override-args", REAP_OVERRIDE]], "changed": []}
    assert train.index("--json-model-override-args") < train.index("--weight-loader-prefetch-checkpoints")
    # multi-valued flags and JSON values survive untouched
    i = probe.index("--cuda-graph-bs-decode")
    assert probe[i + 1:i + 8] == ["1", "2", "4", "7", "8", "9", "10"]
    assert '{"preserve_thinking":true}' in probe and "--mamba-radix-cache-strategy" in probe
    assert not any(a.startswith("--speculative-") for a in probe + train)


def test_dump_server_args_refuse_a_launcher_they_do_not_understand():
    with pytest.raises(ValueError, match="already passes --disable-radix-cache"):
        sa.dump_server_args(launcher_args() + ["--disable-radix-cache"], reap=False)
    no_chunk = launcher_args()
    i = no_chunk.index("--chunked-prefill-size")
    del no_chunk[i:i + 2]
    with pytest.raises(ValueError, match="--chunked-prefill-size 0 times"):
        sa.dump_server_args(no_chunk, reap=False)
    with pytest.raises(ValueError, match="should serve REAP"):
        sa.dump_server_args(launcher_args(reap=False), reap=True)
    assert "--json-model-override-args" not in sa.dump_server_args(launcher_args(reap=False), reap=False)


def test_dump_server_env_sets_the_dump_and_keeps_reap_only_for_the_training_dump():
    launcher = {"PATH": "/bin", "CUDA_HOME": "/cuda", "ARC3_REAP_KEPT_EXPERTS": "/kaggle/arc3-reap-kept.json",
                "ARC3_HC_DUMP_KEEP": "stale"}
    probe = sa.dump_server_env(launcher, {"ARC3_HC_DUMP": Path("/tmp/p"), "ARC3_HC_DUMP_KEEP": "all",
                                          "ARC3_HC_DUMP_DTYPE": "bf16", "ARC3_HC_DUMP_MAX_GB": 8.5}, reap=False)
    assert probe == {"PATH": "/bin", "CUDA_HOME": "/cuda", "ARC3_HC_DUMP": "/tmp/p", "ARC3_HC_DUMP_KEEP": "all",
                     "ARC3_HC_DUMP_DTYPE": "bf16", "ARC3_HC_DUMP_MAX_GB": "8.5"}
    train = sa.dump_server_env(launcher, {"ARC3_HC_DUMP": "/tmp/t", "ARC3_HC_DUMP_MAX_GB": 30}, reap=True)
    assert train["ARC3_REAP_KEPT_EXPERTS"] == "/kaggle/arc3-reap-kept.json" and "ARC3_HC_DUMP_KEEP" not in train
    with pytest.raises(ValueError, match="no ARC3_REAP_KEPT_EXPERTS"):
        sa.dump_server_env({"PATH": "/bin"}, {"ARC3_HC_DUMP": "/tmp/t"}, reap=True)
    with pytest.raises(ValueError, match="not dump variables"):
        sa.dump_server_env(launcher, {"CUDA_VISIBLE_DEVICES": "1"}, reap=False)


def test_server_checks_split_what_breaks_the_dump_from_what_does_not():
    good = {"disable_radix_cache": True, "max_running_requests": 1, "chunked_prefill_size": 8192,
            "speculative_algorithm": None, "json_model_override_args": "{}"}
    assert sa.server_problems(good, num_experts=None) == [] and sa.server_notes(good) == []
    reap = dict(good, json_model_override_args=REAP_OVERRIDE)
    assert sa.num_experts_override(reap) == 448 and sa.server_problems(reap, num_experts=448) == []
    assert sa.server_problems(reap, num_experts=None) and sa.server_problems(good, num_experts=448)
    bad = dict(good, disable_radix_cache=False, speculative_algorithm="NEXTN", max_running_requests=10,
               chunked_prefill_size=4096)
    assert len(sa.server_problems(bad, num_experts=None)) == 2 and len(sa.server_notes(bad)) == 2
    assert sa.server_problems({"error": "HTTP 500"}, num_experts=None) == ["/server_info: HTTP 500"]


# ---------------------------------------------------------------------------------------------- storage (A0)


def _test(directory, *, fs="ext4", written=12.0, free=500.0, error=None):
    return {"dir": directory, "fs": fs, "written_gb": written, "free_gb_before": free, "error": error}


PLAN = {"mem_total_gb": 176.9, "probe_gb": 8.5, "train_gb": 34.0, "holdout_gb": 6.0, "min_train_gb": 6.0,
        "server_ram_gb": 125.0, "margin_gb": 3.0}


def test_plan_storage_puts_both_dumps_on_a_disk_that_passed():
    plan = sa.plan_storage([_test("/tmp"), _test("/dev/shm", fs="tmpfs", free=88.0)], **PLAN)
    assert plan["error"] is None and plan["probe_dir"] == plan["train_dir"] == "/tmp"
    assert plan["train_gb"] == 34.0 and plan["holdout_gb"] == 6.0 and plan["ram_budget_gb"] == pytest.approx(48.9)


def test_plan_storage_falls_back_to_tmpfs_within_the_ram_a_server_leaves():
    # /tmp hit a quota after 5 GB: it holds 2 GB; /dev/shm is RAM: 88 GB free but only 48.9 GB beside a server
    plan = sa.plan_storage([_test("/tmp", written=5.0, error="OSError: [Errno 122] Disk quota exceeded"),
                            _test("/dev/shm", fs="tmpfs", free=88.0)], **PLAN)
    assert plan["error"] is None and plan["probe_dir"] == plan["train_dir"] == "/dev/shm"
    assert plan["room"]["/tmp"]["gb"] == 2.0
    assert plan["train_gb"] == 34.0
    # with a bigger server the RAM beside it is the limit, and the probe dump stays in RAM until A10
    plan = sa.plan_storage([_test("/tmp", written=5.0, error="quota"), _test("/dev/shm", fs="tmpfs", free=88.0)],
                           **dict(PLAN, server_ram_gb=140.0))
    assert plan["ram_budget_gb"] == pytest.approx(33.9) and plan["train_gb"] == pytest.approx(33.9 - 8.5)
    # a disk too small for the training cap still takes it when it has the most room
    plan = sa.plan_storage([_test("/tmp", free=20.0), _test("/dev/shm", fs="tmpfs", free=10.0)], **PLAN)
    assert plan["probe_dir"] == "/tmp" and plan["train_dir"] == "/tmp" and plan["train_gb"] == pytest.approx(8.5)
    assert plan["holdout_gb"] == pytest.approx(0.3 * 8.5)


def test_plan_storage_reports_what_does_not_fit():
    plan = sa.plan_storage([_test("/tmp", written=0.0, error="PermissionError"), _test("/dev/shm", fs="tmpfs",
                                                                                       free=0.06)], **PLAN)
    assert plan["probe_dir"] is None and "no directory holds the probe dump" in plan["error"]
    plan = sa.plan_storage([_test("/tmp", free=15.0)], **PLAN)
    assert plan["train_gb"] == pytest.approx(3.5) and "the training dump would get 3.5 GB" in plan["error"]


def test_write_test_writes_measures_and_cleans_up(tmp_path):
    rec = sa.write_test(tmp_path, gb=0.02, seconds=30, chunk_mb=4, echo=lambda m: None)
    assert rec["complete"] and rec["error"] is None and rec["written_gb"] >= 0.02 and rec["free_gb_before"] > 0
    assert list(tmp_path.iterdir()) == []
    rec = sa.write_test(tmp_path / "missing" / "deeper", gb=0.001, seconds=5, chunk_mb=1, echo=lambda m: None)
    assert rec["complete"]  # created on the way, like a dump directory
    blocked = tmp_path / "file"
    blocked.write_text("x")
    rec = sa.write_test(blocked, gb=0.001, seconds=5, echo=lambda m: None)
    assert rec["error"] and not rec.get("complete")


def test_fs_type_takes_the_longest_mount_point(tmp_path):
    mounts = tmp_path / "mounts"
    mounts.write_text("overlay / overlay rw 0 0\ntmpfs /dev/shm tmpfs rw 0 0\n/dev/sdb /kaggle/working ext4 rw 0 0\n")
    assert sa.fs_type("/dev/shm/x", str(mounts)) == "tmpfs"
    assert sa.fs_type("/kaggle/working/a/b", str(mounts)) == "ext4"
    assert sa.fs_type("/tmp", str(mounts)) == "overlay"


# --------------------------------------------------------------------------------------------------- inputs


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_inputs_are_found_in_either_mount_layout_and_checked(tmp_path):
    root = tmp_path / "input"
    data = b'{"x": 1}\n'
    spec = {"kind": "dataset", "id": "me/prompts", "files": {"requests.jsonl": _sha(data)}}
    assert sa.locate(spec, str(root)) is None
    new = root / "datasets" / "me" / "prompts"
    new.mkdir(parents=True)
    (new / "requests.jsonl").write_bytes(data)
    hit = sa.locate(spec, str(root))
    assert hit["dir"] == str(new) and hit["requests.jsonl.sha256"] == _sha(data)
    old = tmp_path / "input2" / "prompts"
    old.mkdir(parents=True)
    (old / "requests.jsonl").write_bytes(data)
    assert sa.locate(spec, str(tmp_path / "input2"))["dir"] == str(old)
    (old / "requests.jsonl").write_bytes(b"other\n")
    with pytest.raises(sa.InputError, match="not the pinned one"):
        sa.locate(spec, str(tmp_path / "input2"))
    # kernel outputs: <slug>, notebooks/<owner>/<slug>, or files a level or two further down; sizes or several sha256
    ref = {"kind": "kernel", "id": "me/probe-run", "files": {"fidelity.json": [_sha(b"v1"), _sha(b"v2")]},
           "labels": {_sha(b"v2"): "v2"}}
    logs = {"kind": "kernel", "id": "me/full-run", "files": {"ab12-0f_p0_requests.jsonl": 3}}
    for n, layout in enumerate(["{slug}", "notebooks/me/{slug}", "notebooks/me/{slug}/output", "x/y/{slug}"]):
        r = tmp_path / f"k{n}"
        for s, files in (("probe-run", {"fidelity.json": b"v2"}), ("full-run", {"ab12-0f_p0_requests.jsonl": b"abc"})):
            folder = r / layout.format(slug=s)
            folder.mkdir(parents=True)
            for name, content in files.items():
                (folder / name).write_bytes(content)
        assert sa.locate(ref, str(r))["dir"] == str(r / layout.format(slug="probe-run")), layout
        assert sa.locate(logs, str(r))["ab12-0f_p0_requests.jsonl"].endswith("_requests.jsonl")
    (r / layout.format(slug="full-run") / "ab12-0f_p0_requests.jsonl").write_bytes(b"abcd")
    with pytest.raises(sa.InputError, match="4 bytes, not the pinned 3"):
        sa.locate(logs, str(r))


def test_draft_source_is_found_as_the_launcher_finds_it(tmp_path):
    q = {"quantization_config": {"quant_method": "compressed-tensors", "config_groups": {"mtp_routed_experts": {}}}}
    with pytest.raises(ValueError, match="found 0"):
        sa.draft_source(tmp_path)
    (tmp_path / "runtime" / "mtp-int4-g32").mkdir(parents=True)
    (tmp_path / "runtime" / "mtp-int4-g32" / "config.json").write_text(json.dumps(q))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "other"}))
    assert sa.draft_source(tmp_path) == tmp_path / "runtime" / "mtp-int4-g32"
    (tmp_path / "copy").mkdir()
    (tmp_path / "copy" / "config.json").write_text(json.dumps(q))
    with pytest.raises(ValueError, match="found 2"):
        sa.draft_source(tmp_path)


# ------------------------------------------------------------------------------------------- training budget


def test_train_watch_restarts_once_with_the_steps_that_fit(tmp_path):
    (tmp_path / "eval-original.json").write_text(json.dumps({"seconds": 120.0}))
    watch = sa.TrainWatch(tmp_path, total_steps=400, budget_s=3600, probe_steps=4, min_steps=10)
    log = tmp_path / "train-log.jsonl"
    log.write_text("".join(json.dumps({"step": i, "seconds": 20.0}) + "\n" for i in range(1, 3)))
    assert watch(300.0) is None and not watch.decided  # too few steps to judge
    with open(log, "a") as f:
        f.write("not json\n" + "".join(json.dumps({"step": i, "seconds": 20.0}) + "\n" for i in range(3, 5)))
    reason = watch(340.0)
    # 396 steps x 20 s x 1.1 ~ 2.4 h: past the hour. Overhead (load + first evaluation): first seen at 300 s with a
    # 20 s first step. A fresh run fits (3600 - 340 - 280 - 120 eval - 60 save) / 22 = 127 steps
    assert reason and "restarting with --max-steps 127" in reason and watch.restart_steps == 127
    assert watch.info["overhead_s"] == 280.0 and watch(400.0) is None  # decides once
    fits = sa.TrainWatch(tmp_path, total_steps=40, budget_s=3600, probe_steps=4)
    assert fits(340.0) is None and fits.decided and fits.restart_steps is None
    assert sa.steps_that_fit(1000, 100, 100, 10.0, slack=1.0) == 74 and sa.steps_that_fit(10, 100, 100, 1.0) == 0


def test_last_json_object_and_index_stats(tmp_path):
    text = 'loading\n{\n "a": 1\n}\nnoise\n{\n "train": {\n  "windows": 3\n },\n "total_steps": 9\n}\ntail'
    assert sa.last_json_object(text) == {"train": {"windows": 3}, "total_steps": 9}
    assert sa.last_json_object("no json here") is None
    index = tmp_path / "index.jsonl"
    lines = [{"rid": "a", "file": "f", "kept": 10, "span_rows": 6, "bytes": 1000, "qerr_stream_max": 0.02,
              "qerr_max": 0.01, "scale_groups": 1},
             {"rid": "b", "file": "g", "kept": 5, "span_rows": 5, "bytes": 500, "qerr_stream_max": 0.09,
              "qerr_max": 0.03, "nonfinite_rows": 1, "scale_groups": 1},
             {"event": "error", "rid": "c", "error": "x"}]
    index.write_text("".join(json.dumps(x) + "\n" for x in lines) + "{broken\n")
    q = sa.index_stats(tmp_path)
    assert (q["requests"], q["kept_rows"], q["span_rows"], q["bytes"], q["errors"], q["nonfinite_rows"]) == \
        (2, 15, 11, 1500, 1, 1)
    assert q["qerr_stream_max"]["max"] == 0.09 and q["scale_groups"] == [1]
    assert sa.index_stats(tmp_path / "none")["requests"] == 0


# ---------------------------------------------------------------------------------- whole sessions (stubs, CPU)

STUB_COMMON = '''
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
with open(HERE / "calls.jsonl", "a") as f:
    f.write(json.dumps({"script": Path(__file__).name, "argv": sys.argv[1:]}) + "\\n")
def opt(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default
'''

STUBS = {
    "sglang_hc_dump_patch.py": '''
sp = opt("--site-packages")
assert sys.argv[1] == "apply" and (Path(sp) / "sglang/srt/models/qwen4_exp.py").is_file()
print(f"arc3 HC dump: {sp}/sglang/srt/models/qwen4_exp.py patched (base: the wheel's file + "
      f"scripts/sglang_reap_patch.py); module {sp}/sglang/srt/arc3_hc_dump.py")
''',
    "mtp_probe_dump.py": '''
out, dump = Path(opt("--out")), Path(opt("--dump-dir"))
assert Path(opt("--data")).is_file() and Path(opt("--reference")).is_file() and dump.is_dir()
out.mkdir(parents=True, exist_ok=True)
ok = int((HERE / "probe_ok.txt").read_text()) if (HERE / "probe_ok.txt").exists() else 16
lines = [{"rid": f"probe-g{i}", "id": f"g#{i}", "dump_ok": i < ok} for i in range(16)]
(out / "probe-dump.jsonl").write_text("".join(json.dumps(x) + "\\n" for x in lines))
(out / "probe-dump-summary.json").write_text(json.dumps({"sent": 16, "ok": 16, "dump_ok": ok, "stopped": None}))
(dump / "index.jsonl").write_text(json.dumps({"rid": "probe-g0", "file": "f", "kept": 3, "bytes": 30}) + "\\n")
sys.exit(0 if ok == 16 else 1)
''',
    "mtp_replica.py": '''
out = Path(opt("--out"))
assert sys.argv[1] == "check" and Path(opt("--probe")).is_file() and Path(opt("--dump")).is_dir()
go = (HERE / "verdict.txt").read_text().strip() == "go" or opt("--trained") is not None
summary = {"full": {"requests": 16, "replica_mean": 2.7, "sglang_mean": 2.74, "mean_diff": -0.04, "pearson": 0.95}}
reasons = [] if go else ["mean difference 0.3 beyond +-0.05"]
out.write_text(json.dumps({"verdict": {"go": go, "reasons": reasons}, "summary": summary, "gate_variant": "full",
                           "embed_mode": "prefill", "requests": []}))
print("replica check:", "GO" if go else "NO-GO", reasons)
sys.exit(0 if go else 2)
''',
    "hc_dump_driver.py": '''
out, dump = Path(opt("--out")), Path(opt("--dump-dir"))
out.mkdir(parents=True, exist_ok=True)
game = opt("--games") or "bp35"
rid = f"hc-0000-{game}-p0-r0001"
(out / "snapshots.jsonl").write_text(json.dumps({"rid": rid, "game": game, "split": opt("--split")}) + "\\n")
(out / "replay-summary.json").write_text(json.dumps({"planned": 1, "sent": 1, "ok": 1, "dump_ok": 1,
                                                     "prefill_tokens": 1000, "stopped": None, "failed": False}))
with open(dump / "index.jsonl", "a") as f:
    f.write(json.dumps({"rid": rid, "file": "x", "kept": 100, "span_rows": 60, "bytes": 2_000_000_000,
                        "qerr_stream_max": 0.02, "qerr_max": 0.01, "scale_groups": 1}) + "\\n")
''',
    "mtp_train.py": '''
cmd = sys.argv[1]
if cmd == "plan":
    print("planning")
    print(json.dumps({"requests": 3, "train": {"windows": 12}, "holdout": {"windows": 2}, "total_steps": 8}, indent=1))
    sys.exit(0)
out = Path(opt("--out"))
out.mkdir(parents=True, exist_ok=True)
assert Path(opt("--token-map")).is_file() and opt("--target-dir") and opt("--draft")
steps = int(opt("--max-steps", "8"))
(out / "eval-original.json").write_text(json.dumps({"seconds": 1.0}))
with open(out / "train-log.jsonl", "w") as f:
    for i in range(1, steps + 1):
        f.write(json.dumps({"step": i, "seconds": 0.01, "loss": 1.0 / i}) + "\\n")
(out / "trained-dense.safetensors").write_bytes(b"dense")
(out / "train-report.json").write_text(json.dumps({"windows": {}, "total_steps": steps, "steps_done": steps,
                                                    "train_seconds": 1.0, "comparison": {"accept_expected": [2.8, 2.9]}}))
''',
    "mtp_write_draft.py": '''
out = Path(opt("--out"))
assert sys.argv[1] == "write" and Path(opt("--trained")).is_file()
out.mkdir(parents=True)
(out / "mtp-dense.safetensors").write_bytes(b"x" * 100)
(out / "arc3-draft-manifest.json").write_text(json.dumps({"replaced": ["a", "b"], "files": {"x": 1}}))
print("check: ok")
''',
}

SERVER = '''
import http.server, json, os, signal, sys
from pathlib import Path
argv = sys.argv[1:]
def opt(name):
    return argv[argv.index(name) + 1] if name in argv else None
info = {"disable_radix_cache": "--disable-radix-cache" in argv,
        "max_running_requests": int(opt("--max-running-requests") or 0) or None,
        "chunked_prefill_size": int(opt("--chunked-prefill-size") or 0) or None,
        "speculative_algorithm": opt("--speculative-algorithm"),
        "json_model_override_args": opt("--json-model-override-args") or "{}"}
with open(Path(__file__).with_name("servers.jsonl"), "a") as f:
    f.write(json.dumps({"argv": argv, "env": {k: v for k, v in os.environ.items() if k.startswith("ARC3_")}}) + "\\n")
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        body = json.dumps(info if self.path == "/server_info" else {}).encode()
        self.send_response(200 if self.path in ("/health", "/server_info") else 404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
signal.signal(signal.SIGTERM, lambda *a: os._exit(0))
boot = Path(__file__).with_name("boot.json")  # a slow boot: shard progress lines for a while, then silence
if boot.exists():
    import time
    plan, t0, k = json.loads(boot.read_text()), time.time(), 0
    while time.time() - t0 < plan.get("progress_s", 0):
        k = min(k + 1, 38)
        print(f"\\rLoading safetensors checkpoint shards: {k * 100 // 38:3d}% Completed | {k}/38 [00:01<00:01]",
              end="", flush=True)
        time.sleep(0.1)
    time.sleep(plan.get("silent_s", 0))
print("stub sglang serving", flush=True)
http.server.HTTPServer(("127.0.0.1", int(opt("--port"))), H).serve_forever()
'''


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _real_options(name: str) -> set[str]:
    text = (ROOT / "scripts" / name).read_text()
    subcommands = set(re.findall(r'add_parser\("(\w+)"', text))
    for group in re.findall(r'for \w+ in \(([^)]*)\):\s*\n\s*\w+ = \w+\.add_parser\(\w+\)', text):
        subcommands |= set(re.findall(r'"(\w+)"', group))  # mtp_train.py: for name in ("plan", "train", "eval")
    return set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', text)) | subcommands


class Bed:
    """A Kaggle-shaped folder tree for one session: inputs (both layouts), stub scripts, a stub server, a venv."""

    def __init__(self, tmp: Path, *, verdict: str = "go", probe_ok: int = 16):
        self.tmp = tmp
        self.scripts = tmp / "arc3-mtp"
        self.scripts.mkdir(parents=True)
        for name, body in STUBS.items():
            (self.scripts / name).write_text(STUB_COMMON + textwrap.dedent(body))
        (self.scripts / "verdict.txt").write_text(verdict)
        (self.scripts / "probe_ok.txt").write_text(str(probe_ok))
        (self.scripts / "sglang.py").write_text(SERVER)
        self.input = tmp / "input"
        prompts = self.input / "datasets" / "me" / "arc3-fidelity-prompts"
        prompts.mkdir(parents=True)
        (prompts / "requests.jsonl").write_bytes(b'{"id": "g#0"}\n')
        ref = self.input / "arc3-fidelity-base"  # the older layout for kernel outputs
        ref.mkdir(parents=True)
        (ref / "fidelity.json").write_bytes(b'{"passes": {}}')
        self.logs = self.input / "notebooks" / "me" / "arc3-full-run"
        self.logs.mkdir(parents=True)
        sizes = {}
        for name in ("ar25-0c556536_p0_requests.jsonl", "bp35-0a0ad940_p0_requests.jsonl"):
            (self.logs / name).write_bytes(b"{}\n" * 7)
            sizes[name] = 21
        self.venv = tmp / "venv"
        sp = self.venv / "lib" / "python3.12" / "site-packages" / "sglang" / "srt" / "models"
        sp.mkdir(parents=True)
        (sp / "qwen4_exp.py").write_text("# the model file\n")
        self.draft = tmp / "albucino" / "runtime" / "mtp-int4-g32"
        self.draft.mkdir(parents=True)
        (self.draft / "config.json").write_text(json.dumps({"quantization_config": {
            "quant_method": "compressed-tensors", "config_groups": {"mtp_routed_experts": {}}}}))
        self.generic = tmp / "hot_tokens_64k.pt"
        self.generic.write_bytes(b"generic map")
        self.arc_map = tmp / "arc3-hot-tokens.pt"
        self.arc_map.write_bytes(b"arc map")
        self.working = tmp / "working"
        self.working.mkdir()
        self.port = _free_port()
        self.config = {
            "session_hours": 1.0, "scripts_dir": str(self.scripts), "wait_inputs_s": 5, "working_limit_gb": 19.5,
            "census_paths": [str(tmp)],
            "inputs": {
                "prompts": {"kind": "dataset", "id": "me/arc3-fidelity-prompts",
                            "files": {"requests.jsonl": _sha(b'{"id": "g#0"}\n')}},
                "reference": {"kind": "kernel", "id": "me/arc3-fidelity-base",
                              "files": {"fidelity.json": [_sha(b"other"), _sha(b'{"passes": {}}')]},
                              "labels": {_sha(b'{"passes": {}}'): "runs/fidelity-base2"}},
                "logs": {"kind": "kernel", "id": "me/arc3-full-run", "files": sizes}},
            "storage": {"candidates": [str(tmp / "disk"), str(tmp / "shm")], "test_gb": 0.002, "test_seconds": 10,
                        "probe_gb": 0.5, "train_gb": 3.0, "holdout_gb": 1.0, "min_train_gb": 0.1,
                        "server_ram_gb": 0.0, "margin_gb": 0.0},
            "server": {"reap": True, "num_experts": 448, "gpu_free_mib": 3000},
            "probe": {"pass": "seq", "count": 16, "max_rows": 400_000, "min_requests": 10,
                      "generic_map_sha256": _sha(b"generic map")},
            "dump": {"holdout_games": ["ar25", "ft09", "lp85"], "holdout_snapshots_per_game": 1, "holdout_minutes": 5,
                     "minutes": 5, "scale_groups": 1, "qerr_warn": 0.05},
            "train": {"minutes": 5, "min_minutes": 1, "reserve_minutes": 5, "grace_minutes": 1, "probe_steps": 3,
                      "min_steps": 2, "checkpoint_every": 50, "args": [], "plan_args": [],
                      "map": str(self.arc_map), "map_sha256": _sha(b"arc map")},
            "budgets_min": {"A1": 1, "boot": 1, "boot_max": 1, "boot_stall": 1, "A3": 1, "A5": 1, "A8": 1,
                            "A10": 1, "A11": 1},
            "build": {"builder": "test"},
        }

    def session(self, **kw) -> sa.Session:
        s = sa.Session(self.working, self.config, echo=lambda m: None, **kw)
        s.poll_s = 0.05
        s.census = lambda: "census"
        return s

    def launcher(self, s: sa.Session) -> None:
        args = launcher_args(sglang=sys.executable, port=self.port)
        args[1:2] = [str(self.scripts / "sglang.py"), "serve"]
        env = dict(os.environ, ARC3_REAP_KEPT_EXPERTS="/kaggle/arc3-reap-kept.json")
        s.set_launcher(args=args, env=env, log=self.working / "serve.log", port=self.port, model_name="flashnext",
                       python_torch=sys.executable, venv=self.venv, model_dir=self.tmp / "target",
                       draft_dir=self.tmp / "albucino", generic_map=self.generic, precache_thread=None)

    def calls(self) -> list[dict]:
        path = self.scripts / "calls.jsonl"
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

    def servers(self) -> list[dict]:
        path = self.scripts / "servers.jsonl"
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def _until_a5(bed: Bed, s: sa.Session) -> None:
    s.find_inputs(root=str(bed.input), poll_s=0.1)
    s.storage()
    bed.launcher(s)
    s.apply_dump_patch()
    s.start_server("A2", "probe")
    s.probe_dump()
    s.stop("A4")


def _check_calls_against_the_real_scripts(bed: Bed) -> None:
    for call in bed.calls():
        real = _real_options(call["script"])
        used = {a for a in call["argv"] if a.startswith("--")} | ({call["argv"][0]} if not call["argv"][0]
                                                                  .startswith("-") else set())
        assert used <= real, (call["script"], used - real)


def test_a_go_session_runs_every_step_and_leaves_its_reports(tmp_path):
    bed = Bed(tmp_path)
    s = bed.session()
    _until_a5(bed, s)
    assert s.go("A5") and s.replica_gate()
    for step in ("A6", "A7", "A8", "A9", "A10", "A11"):
        assert s.go(step)
    s.start_server("A6", "train")
    s.dump_training()
    s.train_plan()
    s.train()
    s.replica_check("A10", trained=True)
    s.write_draft()
    s.finish()
    state = json.loads((bed.working / "session-a.json").read_text())
    assert state["verdict"] == "complete" and state["exit_code"] is None and s.server is None
    assert [x["step"] for x in state["steps"]] == ["inputs", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
                                                    "A9", "A10", "A11"]
    assert all(x["status"] == "ok" for x in state["steps"])
    assert state["inputs"]["reference"]["versions"] == ["runs/fidelity-base2"]
    assert state["inputs"]["logs"]["dir"] == str(bed.logs)
    plan = state["storage"]["plan"]
    assert plan["probe_dir"] == plan["train_dir"] == str(tmp_path / "disk") and plan["error"] is None
    # the two servers: exactly the launcher's arguments minus/plus the flags of plan 10.4, REAP only for A6
    servers = bed.servers()
    assert len(servers) == 2
    launcher = s.launcher["args"]
    assert servers[0]["argv"] == sa.dump_server_args(launcher, reap=False)[2:]
    assert servers[1]["argv"] == sa.dump_server_args(launcher, reap=True)[2:]
    assert servers[0]["env"] == {"ARC3_HC_DUMP": plan["probe_dump"], "ARC3_HC_DUMP_KEEP": "all",
                                 "ARC3_HC_DUMP_DTYPE": "bf16", "ARC3_HC_DUMP_MAX_GB": "0.5"}
    assert servers[1]["env"] == {"ARC3_HC_DUMP": plan["train_dump"], "ARC3_HC_DUMP_MAX_GB": "3.0",
                                 "ARC3_REAP_KEPT_EXPERTS": "/kaggle/arc3-reap-kept.json"}
    assert state["servers"]["A2"]["problems"] == [] and state["servers"]["A6"]["problems"] == []
    assert state["servers"]["A2"]["diff"]["added"] == [["--disable-cuda-graph"], ["--disable-radix-cache"]]
    # the scripts, in order, each with the options its real counterpart has
    calls = bed.calls()
    assert [c["script"] for c in calls] == ["sglang_hc_dump_patch.py", "mtp_probe_dump.py", "mtp_replica.py",
                                            "hc_dump_driver.py", "hc_dump_driver.py", "mtp_train.py", "mtp_train.py",
                                            "mtp_replica.py", "mtp_write_draft.py"]
    _check_calls_against_the_real_scripts(bed)
    holdout = next(c["argv"] for c in calls if c["script"] == "hc_dump_driver.py")
    assert holdout[holdout.index("--logs") + 1] == str(bed.logs / "ar25-0c556536_p0_requests.jsonl")
    # ar25 wrote 2 GB of the 1 GB held-out cap: ft09 and lp85 are skipped, the train split still runs
    assert list(state["dump"]["holdout"]) == ["ar25"] and any("held-out games from ft09" in w for w in state["warnings"])
    train_call = [c["argv"] for c in calls if c["script"] == "mtp_train.py"][1]
    assert train_call[0] == "train" and train_call[train_call.index("--token-map") + 1] == str(bed.arc_map)
    assert train_call.count("--snapshots") == 2
    replica = [c["argv"] for c in calls if c["script"] == "mtp_replica.py"]
    assert all(r[r.index("--token-map") + 1] == str(bed.generic) for r in replica)  # the reference's map
    assert "--trained" not in replica[0] and "--trained" in replica[1]
    for name in ("replica-check.json", "replica-check-trained.json", "session-a.json", "mtp-draft", "logs",
                 "mtp-train"):
        assert (bed.working / name).exists(), name
    assert (bed.working / "mtp-train" / "train-report.json").is_file()
    assert state["train"]["completed"] and state["draft"]["replaced"] == 2
    assert state["working_total_gb"] < 19.5


def test_a_no_go_stops_cleanly_at_a5_before_any_training(tmp_path):
    bed = Bed(tmp_path, verdict="no-go")
    s = bed.session()
    _until_a5(bed, s)
    assert s.go("A5") and s.replica_gate() is False
    skipped = [step for step in ("A6", "A7", "A8", "A9", "A10", "A11") if not s.go(step)]
    s.finish()
    assert skipped == ["A6", "A7", "A8", "A9", "A10", "A11"]
    state = json.loads((bed.working / "session-a.json").read_text())
    assert (state["verdict"], state["stopped_at"], state["exit_code"]) == ("no-go", "A5", 2)
    assert "replica check NO-GO" in state["reason"] and state["replica_check"]["exit"] == 2
    assert [c["script"] for c in bed.calls()] == ["sglang_hc_dump_patch.py", "mtp_probe_dump.py", "mtp_replica.py"]
    assert len(bed.servers()) == 1 and s.server is None
    assert (bed.working / "replica-check.json").is_file() and not (bed.working / "mtp-train").exists()
    assert "working_bytes" in state and state["steps"][-1]["step"] == "A5"
    # a step called by hand after the NO-GO refuses, and the recorded stop stays the first one
    with pytest.raises(sa.Stop) as stop:
        s.start_server("A6", "train")
    assert stop.value.code == 2 and len(bed.servers()) == 1
    state = json.loads((bed.working / "session-a.json").read_text())
    assert state["stopped_at"] == "A5" and any(w.startswith("A6 after the stop at A5") for w in state["warnings"])


def test_a_failed_step_stops_its_server_and_the_session(tmp_path):
    bed = Bed(tmp_path, probe_ok=4)
    s = bed.session()
    s.find_inputs(root=str(bed.input), poll_s=0.1)
    s.storage()
    bed.launcher(s)
    s.apply_dump_patch()
    s.start_server("A2", "probe")
    with pytest.raises(sa.Stop) as stop:
        s.probe_dump()
    assert stop.value.code == 1 and s.server is None
    state = json.loads((bed.working / "session-a.json").read_text())
    assert (state["verdict"], state["stopped_at"]) == ("failed", "A3") and "only 4 of 16" in state["reason"]
    assert state["servers"]["A2"].get("stopped_utc") and not sa.port_open(bed.port)
    with pytest.raises(sa.Stop):
        s.go("A4")


def _boot(tmp_path, *, progress_s: float, silent_s: float = 0.0, **budgets) -> tuple[Bed, sa.Session]:
    bed = Bed(tmp_path)
    (bed.scripts / "boot.json").write_text(json.dumps({"progress_s": progress_s, "silent_s": silent_s}))
    bed.config["budgets_min"].update(budgets)
    s = bed.session()
    s.find_inputs(root=str(bed.input), poll_s=0.1)
    s.storage()
    bed.launcher(s)
    s.apply_dump_patch()
    return bed, s


def test_a_server_still_loading_weights_is_waited_for_past_the_boot_budget(tmp_path):
    # session A v1: weights loaded 4-5x slower than usual and a fixed 20-minute limit stopped it at 29 of 38 shards
    bed, s = _boot(tmp_path, progress_s=2.0, boot=0.01, boot_max=0.5, boot_stall=0.01)
    s.start_server("A2", "probe")
    state = json.loads((bed.working / "session-a.json").read_text())
    assert state["verdict"] == "running" and state["servers"]["A2"]["healthy_after_s"] >= 1.5
    s.stop("A4")


def test_a_server_whose_log_stops_growing_is_stopped_after_the_boot_budget(tmp_path):
    bed, s = _boot(tmp_path, progress_s=0.5, silent_s=60, boot=0.01, boot_max=0.5, boot_stall=0.02)
    with pytest.raises(sa.Stop):
        s.start_server("A2", "probe")
    state = json.loads((bed.working / "session-a.json").read_text())
    assert (state["verdict"], state["stopped_at"]) == ("failed", "A2") and s.server is None
    assert "its log had not grown for" in state["reason"] and re.search(r"weights \d+/38 shards", state["reason"])
    assert not sa.port_open(bed.port)


def test_boot_max_stops_a_server_that_keeps_loading(tmp_path):
    bed, s = _boot(tmp_path, progress_s=60, boot=0.01, boot_max=0.03, boot_stall=1)
    with pytest.raises(sa.Stop):
        s.start_server("A2", "probe")
    state = json.loads((bed.working / "session-a.json").read_text())
    assert state["stopped_at"] == "A2" and "still loading at the 0.03-minute limit" in state["reason"]


def test_inputs_that_are_missing_or_differ_stop_before_anything_is_installed(tmp_path):
    bed = Bed(tmp_path)
    (bed.logs / "bp35-0a0ad940_p0_requests.jsonl").write_bytes(b"{}\n")  # another version of the run
    s = bed.session()
    with pytest.raises(sa.Stop, match="21"):
        s.find_inputs(root=str(bed.input), poll_s=0.1)
    bed = Bed(tmp_path / "b")
    (bed.input / "arc3-fidelity-base" / "fidelity.json").unlink()
    s = bed.session()
    with pytest.raises(sa.Stop, match="not mounted after"):
        s.find_inputs(root=str(bed.input), wait_s=0.3, poll_s=0.1)
    state = json.loads((bed.working / "session-a.json").read_text())
    assert state["stopped_at"] == "inputs" and "reference (kernel me/arc3-fidelity-base)" in state["reason"]
