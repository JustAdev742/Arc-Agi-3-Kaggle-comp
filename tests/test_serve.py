"""Server start-up policy: a tuned attempt that fails its probe must fall back to conservative flags."""
import arc3.serve as serve


class FakeProc:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def test_fallback_after_failed_probe(monkeypatch, tmp_path):
    started = []
    probes = iter([(False, "500: EngineCore encountered an issue"), (True, "ready")])
    monkeypatch.setattr(serve, "start_vllm", lambda model_dir, log_path="", **kw: (started.append(kw), FakeProc())[1])
    monkeypatch.setattr(serve, "wait_for_server", lambda *a, **k: True)
    monkeypatch.setattr(serve, "probe_completion", lambda *a, **k: next(probes))
    proc, ok = serve.start_vllm_with_fallback("/m", log_path=str(tmp_path / "vllm.log"), timeout_s=1000, port=8000, mtp_tokens=2)
    assert ok and proc is not None
    assert len(started) == 2
    assert started[0]["mtp_tokens"] == 2
    assert started[1]["mtp_tokens"] == 0 and started[1]["kv_cache_dtype"] == "auto"
    log = (tmp_path / "vllm.log").read_text()
    assert "probe completion attempt 1: FAILED" in log and "attempt 1 failed" in log


def test_gives_up_when_both_attempts_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(serve, "start_vllm", lambda model_dir, log_path="", **kw: FakeProc())
    monkeypatch.setattr(serve, "wait_for_server", lambda *a, **k: False)
    proc, ok = serve.start_vllm_with_fallback("/m", log_path=str(tmp_path / "vllm.log"), timeout_s=1000)
    assert not ok and proc is None


def test_command_flags():
    cmd = serve.build_vllm_command("/m", mtp_tokens=2, kv_cache_dtype="fp8")
    assert cmd[cmd.index("--attention-backend") + 1] == "TRITON_ATTN"
    assert "--speculative-config" in cmd and cmd[cmd.index("--kv-cache-dtype") + 1] == "fp8"
    import json as _json
    spec = _json.loads(cmd[cmd.index("--speculative-config") + 1])
    assert spec == {"method": "mtp", "num_speculative_tokens": 2, "attention_backend": "TRITON_ATTN"}
    cmd = serve.build_vllm_command("/m", mtp_tokens=0, kv_cache_dtype="auto")
    assert "--speculative-config" not in cmd and "--kv-cache-dtype" not in cmd
