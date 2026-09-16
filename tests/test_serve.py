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
    assert "probe completion attempt 1 (tuned): FAILED" in log and "attempt 1 (tuned) failed" in log


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


def test_specialist_ladder_flags():
    """Exp-010: the NVFP4 specialist failed engine-core init twice and the council silently shared the coordinator.
    The ladder tries each checkpoint tuned then conservative, and the conservative flags really reach the command."""
    ladder = serve.specialist_attempts(["/in/qwen3-vl-8b-instruct-fp8", "/in/qwen3-vl-8b-instruct-nvfp4"], gpu_mem=0.30)
    assert [a["label"] for a in ladder] == ["qwen3-vl-8b-instruct-fp8 tuned", "qwen3-vl-8b-instruct-fp8 conservative",
                                            "qwen3-vl-8b-instruct-nvfp4 tuned", "qwen3-vl-8b-instruct-nvfp4 conservative"]
    assert [a["model_dir"] for a in ladder][:2] == ["/in/qwen3-vl-8b-instruct-fp8"] * 2
    assert all(a["port"] == 8001 and a["served_name"] == "arc3-specialist" and a["mtp_tokens"] == 0 for a in ladder)
    assert ladder[1]["gpu_mem"] < ladder[0]["gpu_mem"] == 0.30
    ladder_keys = {"model_dir", "label", "fit_gpu_mem", "probe_image", "env_extra"}
    tuned = serve.build_vllm_command(ladder[0]["model_dir"], **{k: v for k, v in ladder[0].items() if k not in ladder_keys})
    cons = serve.build_vllm_command(ladder[1]["model_dir"], **{k: v for k, v in ladder[1].items() if k not in ladder_keys})
    assert "--enforce-eager" not in tuned and "--enforce-eager" in cons
    assert "--speculative-config" not in tuned and "--reasoning-parser" not in tuned
    assert tuned[tuned.index("--tool-call-parser") + 1] == "hermes"
    assert cons[cons.index("--max-model-len") + 1] == "8192" and "--kv-cache-dtype" not in cons
    assert serve.specialist_attempts([None, ""]) == []
    # NVFP4 rungs carry the Marlin GEMM backend (flashinfer's cutlass FP4 JIT has no SM120 kernels, exp-010); FP8 rungs do not
    assert ladder[2]["env_extra"]["VLLM_NVFP4_GEMM_BACKEND"] == "marlin"
    # FP8 rungs disable DeepGEMM (exp-010b v2: "Unknown SF transformation" at load on SM120)
    assert ladder[0]["env_extra"] == {"VLLM_USE_DEEP_GEMM": "0"} and ladder[1]["env_extra"] == {"VLLM_USE_DEEP_GEMM": "0"}


def test_ladder_walks_attempts_and_fits_gpu_memory(monkeypatch, tmp_path):
    started = []
    probes = iter([(False, "500: EngineCore"), (True, "ready")])

    class DeadProc(FakeProc):
        def poll(self):
            return 1

    procs = iter([DeadProc(), FakeProc(), FakeProc()])
    monkeypatch.setattr(serve, "start_vllm", lambda model_dir, log_path="", env_extra=None, **kw: (started.append((model_dir, kw)), next(procs))[1])
    monkeypatch.setattr(serve, "wait_for_server", lambda url, timeout_s, proc=None: proc.poll() is None)
    monkeypatch.setattr(serve, "probe_completion", lambda *a, **k: next(probes))
    monkeypatch.setattr(serve, "gpu_fraction_available", lambda reserve_mib=1536: 0.20)
    ladder = serve.specialist_attempts(["/in/fp8", "/in/nvfp4"], gpu_mem=0.30)
    proc, ok = serve.start_vllm_with_fallback("/in/fp8", log_path=str(tmp_path / "s.log"), timeout_s=1200, attempts=ladder)
    assert ok and proc is not None
    # attempt 1 died at start, attempt 2 came up but failed its probe, attempt 3 (nvfp4 tuned) won
    assert [m for m, _ in started] == ["/in/fp8", "/in/fp8", "/in/nvfp4"]
    assert all(kw["gpu_mem"] == 0.18 for _, kw in started)  # clamped to the 0.20 the coordinator left, minus 0.02
    assert not any(k in kw for _, kw in started for k in ("label", "model_dir", "fit_gpu_mem", "probe_image", "env_extra"))
    assert serve.LAST_START["ready"] and serve.LAST_START["attempt"] == 3 and serve.LAST_START["label"] == "nvfp4 tuned"
    log = (tmp_path / "s.log").read_text()
    assert "attempt 1 (fp8 tuned) failed" in log and "probe completion attempt 2 (fp8 conservative): FAILED" in log
    assert "gpu_mem 0.3 -> 0.18" in log


def test_ladder_gives_up_and_records_it(monkeypatch, tmp_path):
    monkeypatch.setattr(serve, "start_vllm", lambda model_dir, log_path="", **kw: FakeProc())
    monkeypatch.setattr(serve, "wait_for_server", lambda *a, **k: False)
    monkeypatch.setattr(serve, "gpu_fraction_available", lambda reserve_mib=1536: None)
    proc, ok = serve.start_vllm_with_fallback("/in/fp8", log_path=str(tmp_path / "s.log"), timeout_s=1000,
                                              attempts=serve.specialist_attempts(["/in/fp8"]))
    assert not ok and proc is None and serve.LAST_START["ready"] is False and serve.LAST_START["attempts"] == 2


def test_probe_image_payload(monkeypatch):
    sent = {}

    class R:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "ready"}}]}

    import types
    fake_requests = types.SimpleNamespace(post=lambda url, json, timeout: (sent.update(json), R())[1])
    monkeypatch.setitem(__import__("sys").modules, "requests", fake_requests)
    ok, text = serve.probe_completion("http://x/v1", "m", with_image=True)
    assert ok and text == "ready"
    content = sent["messages"][0]["content"]
    assert isinstance(content, list) and content[1]["image_url"]["url"].startswith("data:image/png;base64,iVBOR")
