"""scripts/sglang_reap_patch.py: REAP expert pruning at load time (expert id remapping, router slicing, the anchored
edit of the installed sglang) and the kept list it applies (kaggle/franzen/reap448_kept_experts.json). CPU only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sglang_reap_patch as rp  # noqa: E402

KEPT = ROOT / "kaggle" / "franzen" / "reap448_kept_experts.json"
# The Pennyroyal sglang wheel (dataset dfranzen/pennyroyal-v253, wheels/sglang-0.5.19+gd00d88efc8d6-...whl, 24.8 MB);
# the tests that read its qwen4_exp.py skip without it.
WHEEL = Path(os.environ.get("PENNYROYAL_WHEEL", "/tmp/claude-0/-home-user-Arc-Agi-3-Kaggle-comp/"
                            "d342458e-03bd-545b-8a6d-06bca061963e/scratchpad/reap/dl-wheel/"
                            "sglang-0.5.19+gd00d88efc8d6-cp312-cp312-linux_x86_64.whl"))
needs_wheel = pytest.mark.skipif(not WHEEL.is_file(), reason="Pennyroyal sglang wheel not available (PENNYROYAL_WHEEL)")
SUFFIXES = [".gate_proj.qweight", ".gate_proj.scales", ".up_proj.qweight", ".up_proj.scales", ".down_proj.qweight",
            ".down_proj.scales"]


class Cfg:
    def __init__(self, num_experts: int, num_hidden_layers: int):
        self.num_experts, self.num_hidden_layers = num_experts, num_hidden_layers


def _routers(layers: int, experts: int, width: int = 4) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(0)
    return {i: rng.standard_normal((experts, width)).astype(np.float16) for i in range(layers)}


def _stream(routers, experts: int, *, drop: tuple[int, int] | None = None, prefix="model.language_model."):
    """Checkpoint-like (name, tensor) pairs: per expert one tiny tensor per suffix whose value is the expert id."""
    for layer, router in routers.items():
        yield f"{prefix}layers.{layer}.linear_attn.out_proj.weight", np.zeros(1)
        for expert in range(experts):
            for suffix in SUFFIXES:
                if (layer, expert) == drop and suffix.startswith(".down"):
                    continue
                yield f"{prefix}layers.{layer}.mlp.experts.{expert}{suffix}", np.array([expert])
        yield f"{prefix}layers.{layer}.mlp.gate.weight", router
        yield f"{prefix}layers.{layer}.mlp.shared_expert_gate.weight", np.ones(1)
    yield "mtp.layers.0.mlp.experts.7.gate_proj.weight", np.array([7])
    yield "mtp.layers.0.mlp.gate.weight", np.ones((8, 4))


def _write_kept(tmp_path: Path, kept: dict[int, list[int]], routers=None) -> Path:
    path = tmp_path / "kept.json"
    path.write_text(json.dumps({str(k): v for k, v in kept.items()}))
    if routers is not None:
        layers = {str(i): {"base_router_sha256": hashlib.sha256(r.tobytes()).hexdigest()} for i, r in routers.items()}
        (tmp_path / "kept.meta.json").write_text(json.dumps({"layers": layers}))
    return path


KEPT_SMALL = {0: [0, 2, 3, 5, 6, 7], 1: [1, 2, 3, 4, 6, 7]}


# --- runtime: remapping and router slicing -------------------------------------------------------------------------


def test_unset_variable_leaves_the_stream_alone(monkeypatch):
    monkeypatch.delenv(rp.ENV, raising=False)
    stream = iter([("a", 1)])
    assert rp.wrap_weights(stream, Cfg(512, 48)) is stream and not rp.active()


def test_kept_experts_are_renumbered_pruned_ones_dropped_and_router_rows_sliced(tmp_path, monkeypatch, caplog):
    routers = _routers(2, 8)
    monkeypatch.setenv(rp.ENV, str(_write_kept(tmp_path, KEPT_SMALL, routers)))
    caplog.set_level("INFO")
    out = list(rp.wrap_weights(_stream(routers, 8), Cfg(6, 2)))
    experts = [(n, t) for n, t in out if ".mlp.experts." in n and n.startswith("model.")]
    assert len(experts) == 2 * 6 * len(SUFFIXES)
    for name, tensor in experts:  # slot s of layer i holds original expert kept[i][s]
        layer, slot = int(name.split(".")[3]), int(name.split(".")[6])
        assert KEPT_SMALL[layer][slot] == int(tensor[0]) and 0 <= slot < 6
    assert "model.language_model.layers.1.mlp.experts.0.gate_proj.qweight" in dict(out)  # slot 0 of layer 1 = expert 1
    assert int(dict(out)["model.language_model.layers.1.mlp.experts.0.gate_proj.qweight"][0]) == 1
    for layer, router in routers.items():
        sliced = dict(out)[f"model.language_model.layers.{layer}.mlp.gate.weight"]
        assert sliced.shape == (6, 4) and np.array_equal(sliced, router[KEPT_SMALL[layer]])
    # everything else (attention, shared expert gate, the MTP block) passes through untouched and in order
    names = [n for n, _ in out]
    assert names[0] == "model.language_model.layers.0.linear_attn.out_proj.weight"
    assert names[-2:] == ["mtp.layers.0.mlp.experts.7.gate_proj.weight", "mtp.layers.0.mlp.gate.weight"]
    assert dict(out)["mtp.layers.0.mlp.gate.weight"].shape == (8, 4)
    assert "kept 6 of 8 routed experts in each of 2 layers (72 expert tensors loaded, 24 pruned tensors skipped)" \
        in caplog.text and "router sha256 verified" in caplog.text


def test_older_checkpoint_names_without_language_model_prefix(tmp_path, monkeypatch):
    routers = _routers(2, 8)
    monkeypatch.setenv(rp.ENV, str(_write_kept(tmp_path, KEPT_SMALL)))
    out = dict(rp.wrap_weights(_stream(routers, 8, prefix="model."), Cfg(6, 2)))
    assert np.array_equal(out["model.layers.0.mlp.gate.weight"], routers[0][[0, 2, 3, 5, 6, 7]])
    assert int(out["model.layers.0.mlp.experts.1.up_proj.scales"][0]) == 2


def test_wrong_model_size_or_layer_count_fails_before_loading(tmp_path, monkeypatch):
    monkeypatch.setenv(rp.ENV, str(_write_kept(tmp_path, KEPT_SMALL)))
    with pytest.raises(RuntimeError, match="json-model-override-args"):
        rp.wrap_weights(iter(()), Cfg(8, 2))
    with pytest.raises(RuntimeError, match="lists 2 layers, the model has 3"):
        rp.wrap_weights(iter(()), Cfg(6, 3))


@pytest.mark.parametrize("case, message", [
    ("missing tensor", "experts do not all have the same tensors"),
    ("missing router", "layer 1: no router"),
    ("other router", "differs from the router the kept list was derived from"),
    ("too few rows", "has 7 rows but the kept list names expert 7"),
    ("fused", "fused expert tensor"),
])
def test_a_checkpoint_that_does_not_match_the_list_is_refused(tmp_path, monkeypatch, case, message):
    routers = _routers(2, 8)
    monkeypatch.setenv(rp.ENV, str(_write_kept(tmp_path, KEPT_SMALL, routers)))
    stream = _stream(routers, 8, drop=(1, 4) if case == "missing tensor" else None)
    if case == "missing router":
        stream = ((n, t) for n, t in stream if n != "model.language_model.layers.1.mlp.gate.weight")
    elif case == "other router":
        stream = ((n, t + 1 if n.endswith("layers.0.mlp.gate.weight") else t) for n, t in stream)
    elif case == "too few rows":
        stream = ((n, t[:7] if n.endswith("layers.0.mlp.gate.weight") else t) for n, t in stream)
    elif case == "fused":
        stream = iter([("model.language_model.layers.0.mlp.experts.gate_up_proj", np.zeros((8, 2, 2)))])
    with pytest.raises(RuntimeError, match=message):
        list(rp.wrap_weights(stream, Cfg(6, 2)))


@pytest.mark.parametrize("raw, message", [
    ({"0": [1, 0]}, "strictly increasing"), ({"0": [0, 0]}, "strictly increasing"), ({"0": [-1, 2]}, ">= 0"),
    ({"1": [0, 1]}, "without gaps"), ({"0": [0, 1], "1": [0]}, "same number"), ({"0": [True, 2]}, "integers"),
    ({"x": [0]}, "not a non-negative integer"), ([], "non-empty JSON object"),
])
def test_kept_list_validation(tmp_path, raw, message):
    path = tmp_path / "k.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=message):
        rp.load_kept(path)


def test_torch_tensors_are_sliced_on_their_device_and_hashed_as_their_bytes():
    torch = pytest.importorskip("torch")
    router = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4).to(torch.bfloat16)
    sliced = rp.take_rows(router, [1, 5, 6])
    assert sliced.dtype == torch.bfloat16 and sliced.is_contiguous() and torch.equal(sliced, router[[1, 5, 6]])
    assert rp.tensor_sha256(router) == hashlib.sha256(router.view(torch.int16).numpy().tobytes()).hexdigest()


# --- the shipped kept list -------------------------------------------------------------------------------------------


def test_the_reap448_list_and_its_provenance():
    kept = rp.load_kept(KEPT)
    meta = json.loads(rp.meta_path(KEPT).read_text())
    assert len(kept) == 48 and {len(v) for v in kept.values()} == {448} and all(v[-1] < 512 for v in kept.values())
    assert meta["every_row_matched_bitwise"] is True and meta["mtp_router_identical"] is True
    assert meta["pruned_source"].startswith("lee-chang-93/Qwen3.8-Flash-Next-NVFP4-REAP-k448@8d565c9")
    assert meta["base_source"].startswith("Intel/Qwen3.8-Flash-Next-W4A16-AutoRound@4c67bf6")
    assert set(rp.load_router_sha256(KEPT)) == set(kept)
    assert all(m["num_experts"] == 512 and m["kept"] == 448 for m in meta["layers"].values())


# --- the installer: anchored edits of the installed sglang -------------------------------------------------------------


def _fake_model_file() -> str:
    return ("class Qwen4ExpForConditionalGeneration:\n" + rp.LOAD_ANCHOR + "        ]\n\n"
            "    @classmethod\n    def get_model_config_for_expert_location(cls, config):\n" + rp.LOCATION_ANCHOR
            + "        return ModelConfigForExpertLocation(\n        )\n")


def test_patch_text_needs_each_anchor_exactly_once():
    text = _fake_model_file()
    patched = rp.patch_text(text, check_hash=False)
    assert patched.count(rp.MARK) == 2 and "weights = _arc3_reap.wrap_weights(weights, self.config)" in patched
    with pytest.raises(rp.PatchError, match="not the analysed Pennyroyal file"):
        rp.patch_text(text)
    with pytest.raises(rp.PatchError, match="found 0 times"):
        rp.patch_text(text.replace(rp.LOAD_ANCHOR, ""), check_hash=False)
    with pytest.raises(rp.PatchError, match="found 2 times"):
        rp.patch_text(text + rp.LOCATION_ANCHOR, check_hash=False)
    with pytest.raises(rp.PatchError, match="already patched"):
        rp.patch_text(patched, check_hash=False)


def _site_packages(tmp_path: Path, text: str) -> Path:
    site = tmp_path / "site-packages"
    (site / rp.MODEL_FILE).parent.mkdir(parents=True)
    (site / rp.MODEL_FILE).write_text(text)
    cache = site / rp.MODEL_FILE.parent / "__pycache__"
    cache.mkdir()
    (cache / "qwen4_exp.cpython-312.pyc").write_bytes(b"stale")
    return site


@needs_wheel
def test_the_real_wheel_file_patches_installs_once_and_refuses_other_versions(tmp_path):
    with zipfile.ZipFile(WHEEL) as wheel:
        original = wheel.read(str(rp.MODEL_FILE)).decode()
    assert hashlib.sha256(original.encode()).hexdigest() == rp.MODEL_FILE_SHA256
    site = _site_packages(tmp_path, original)
    assert rp.main(["apply", "--site-packages", str(site), "--kept", str(KEPT)]) == 0
    patched = (site / rp.MODEL_FILE).read_text()
    assert patched == rp.patch_text(original) and patched.count(rp.MARK) == 2
    assert (site / rp.MODULE_FILE).read_text() == Path(rp.__file__).read_text()
    assert not list((site / rp.MODEL_FILE.parent / "__pycache__").glob("*.pyc"))
    # both edits sit in the right functions of the real file
    load = patched.index("    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):")
    assert patched.index("wrap_weights(weights, self.config)") - load < 600
    location = patched.index("def get_model_config_for_expert_location(cls, config):")
    assert 0 < patched.index("if _arc3_reap.active():", location) - location < 700
    # a second run is a no-op; a different or half-patched file is refused
    assert rp.main(["apply", "--site-packages", str(site)]) == 0 and (site / rp.MODEL_FILE).read_text() == patched
    (site / rp.MODEL_FILE).write_text(patched.replace("            return None\n        # <<< arc3 REAP", "        # <<< arc3 REAP"))
    assert rp.main(["apply", "--site-packages", str(site)]) == 1
    other = _site_packages(tmp_path / "other", original.replace("Qwen4-Exp VL weights", "Qwen4-Exp weights"))
    assert rp.main(["apply", "--site-packages", str(other)]) == 1
    assert rp.main(["apply", "--site-packages", str(other), "--allow-other-version"]) == 0
    assert rp.main(["check-wheel", str(WHEEL)]) == 0


def test_apply_refuses_a_bad_kept_list_before_touching_anything(tmp_path):
    site = _site_packages(tmp_path, _fake_model_file())
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"0": [3, 1]}))
    assert rp.main(["apply", "--site-packages", str(site), "--kept", str(bad), "--allow-other-version"]) == 1
    assert (site / rp.MODEL_FILE).read_text() == _fake_model_file() and not (site / rp.MODULE_FILE).exists()
    assert rp.main(["apply", "--site-packages", str(tmp_path / "nowhere")]) == 1
