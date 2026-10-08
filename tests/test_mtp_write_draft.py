"""scripts/mtp_write_draft.py: writing a fine-tuned MTP draft directory from albucino's and checking it against the
launcher's rules, on a tiny albucino-style draft (tests/mtp_fakes.py). The launcher's own ``prepare_draft_view`` and
``indexed_shards`` (Franzen's cell 12) run on the result. numpy only: runs without torch."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests import mtp_fakes as mf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mtp_write_draft as wd  # noqa: E402

NOTEBOOK = ROOT / "kaggle" / "franzen" / "arc-agi-3-milestone-2-solution.ipynb"
TRAINABLE = [n for n in wd.dense_layout(mf.TINY_TEXT) if ".indexer." not in n]


def launcher() -> dict:
    """``indexed_shards``, ``prepare_draft_view`` and the two regexes from the notebook's launcher cell, executed
    alone (the cell itself installs and starts a server)."""
    nb = json.loads(NOTEBOOK.read_text())
    src = next("".join(c["source"]) for c in nb["cells"] if "def prepare_draft_view" in "".join(c["source"]))
    tree = ast.parse(src)
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in ("indexed_shards", "prepare_draft_view"))
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("EXPERT_TARGET", "DENSE_IGNORE")
                                                  for t in n.targets))]
    ns = {"json": json, "hashlib": hashlib, "copy": copy, "Path": Path}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(NOTEBOOK), "exec"), ns)  # our own notebook's code
    return ns


@pytest.fixture
def draft(tmp_path):
    path = tmp_path / "albucino" / "runtime" / "mtp-int4-g32"
    arrays = mf.write_tiny_draft(path)
    return path, arrays


def test_the_constants_and_layout_match_the_launcher_and_albucino():
    ns = launcher()
    assert ns["EXPERT_TARGET"] == wd.EXPERT_TARGET and ns["DENSE_IGNORE"] == wd.DENSE_IGNORE
    assert wd.dense_layout(mf.REAL_TEXT) == mf.ALBUCINO_DENSE  # the 29 dense tensors of albucino's header
    experts = wd.expert_layout(mf.REAL_TEXT, 32)
    assert len(experts) == 4608  # albucino's mtp-routed-experts-int4.safetensors
    assert experts["mtp.layers.0.mlp.experts.511.down_proj.weight_packed"] == ("I32", (2560, 80))
    assert experts["mtp.layers.0.mlp.experts.0.gate_proj.weight_scale"] == ("BF16", (640, 80))
    assert experts["mtp.layers.0.mlp.experts.7.up_proj.weight_shape"] == ("I64", (2,), (640, 2560))


def test_write_replaces_only_the_trained_tensors_and_keeps_everything_else_byte_identical(draft, tmp_path):
    src, _ = draft
    trained_file = tmp_path / "trained-dense.safetensors"
    trained = mf.write_trained(trained_file, mf.TINY_TEXT, TRAINABLE)
    out = tmp_path / "out"
    manifest = wd.write_draft(tmp_path / "albucino", trained_file, out)
    assert manifest["replaced"] == sorted(TRAINABLE) and Path(manifest["source"]) == src
    for name in ("config.json", wd.INDEX, "mtp-routed-experts-int4.safetensors", "tokenizer.json", "vocab.json"):
        assert (out / name).read_bytes() == (src / name).read_bytes(), name
    new, old = wd.read_tensors(out / "mtp-dense.safetensors"), wd.read_tensors(src / "mtp-dense.safetensors")
    assert list(new) == list(old) and wd.read_header(out / "mtp-dense.safetensors")[2] == \
        wd.read_header(src / "mtp-dense.safetensors")[2]  # same header bytes: names, dtypes, shapes, offsets, order
    for name in old:
        if name in trained:
            assert new[name][2] == mf.bf16_bits(trained[name]).tobytes() != old[name][2]
        else:
            assert new[name][2] == old[name][2], name  # frozen: indexer, lm_head, embed_tokens
    result = wd.check_draft_dir(out, reference=src, trained_names=TRAINABLE)
    assert result["ok"], result["problems"]
    assert sorted(result["changed"]) == sorted(TRAINABLE) and result["tensors"] == len(
        json.loads((src / wd.INDEX).read_text())["weight_map"])
    files = json.loads((out / wd.MANIFEST).read_text())["files"]
    assert files["mtp-dense.safetensors"] == wd.sha256(out / "mtp-dense.safetensors")


def test_the_launchers_own_draft_view_accepts_the_written_directory(draft, tmp_path):
    src, _ = draft
    trained_file = tmp_path / "t.safetensors"
    mf.write_trained(trained_file, mf.TINY_TEXT, TRAINABLE[:3])
    out = tmp_path / "out"
    wd.write_draft(src, trained_file, out)
    target = tmp_path / "target"
    target.mkdir()
    (target / "tokenizer.json").write_text('{"stub": "target"}')
    ns = launcher()
    view = ns["prepare_draft_view"](target, out, tmp_path / "work")
    assert (view / "mtp-dense.safetensors").resolve() == (out / "mtp-dense.safetensors").resolve()
    assert (view / "tokenizer.json").resolve() == (target / "tokenizer.json").resolve()  # linked from the target
    cfg = json.loads((view / "config.json").read_text())["quantization_config"]
    assert cfg["config_groups"]["mtp_routed_experts"]["targets"] == [wd.EXPERT_TARGET]
    assert cfg["ignore"] == [wd.DENSE_IGNORE]
    _, shards = ns["indexed_shards"](out)
    assert shards == ["mtp-dense.safetensors", "mtp-routed-experts-int4.safetensors"]
    # the launcher's rewritten view config passes our checker too (its regex forms of targets and ignore)
    assert wd.check_draft_dir(view)["ok"]


def _broken(src: Path, tmp_path: Path, edit) -> Path:
    out = tmp_path / "broken"
    trained_file = tmp_path / "tb.safetensors"
    mf.write_trained(trained_file, mf.TINY_TEXT, TRAINABLE[:2])
    wd.write_draft(src, trained_file, out, overwrite=True)
    edit(out)
    return out


def _edit_json(path: Path, fn) -> None:
    obj = json.loads(path.read_text())
    fn(obj)
    path.write_text(json.dumps(obj))


@pytest.mark.parametrize("case, expected", [
    ("group_size", "symmetric INT4 g32"), ("bits", "symmetric INT4 g32"), ("targets", "unexpected expert targets"),
    ("ignore", "unexpected ignore rules"), ("dotdot", "invalid shard path"), ("missing", "missing or empty shard"),
    ("empty", "missing or empty shard"), ("no_experts_key", "no mtp.layers.0.mlp.experts. key"),
    ("second_draft", "found 2"), ("not_ct", "found 0"), ("nan", "NaN or Inf"), ("bad_shape", "does not hold [32, 32]"),
    ("unindexed", "indexed in"), ("short_data", "bad data range"),
])
def test_the_checker_refuses_what_the_launcher_or_the_loader_would(draft, tmp_path, case, expected):
    src, _ = draft
    q = "quantization_config"

    def edit(out: Path):
        cfg, index = out / "config.json", out / wd.INDEX
        if case == "group_size":
            _edit_json(cfg, lambda c: c[q]["config_groups"]["mtp_routed_experts"]["weights"].update(group_size=128))
        elif case == "bits":
            _edit_json(cfg, lambda c: c[q]["config_groups"]["mtp_routed_experts"]["weights"].update(num_bits=8))
        elif case == "targets":
            _edit_json(cfg, lambda c: c[q]["config_groups"]["mtp_routed_experts"].update(targets=["Linear"]))
        elif case == "ignore":
            _edit_json(cfg, lambda c: c[q].update(ignore=["lm_head"]))
        elif case == "dotdot":
            _edit_json(index, lambda i: i["weight_map"].update({"mtp.fc_hidden.weight": "../mtp-dense.safetensors"}))
        elif case == "missing":
            (out / "mtp-routed-experts-int4.safetensors").unlink()
        elif case == "empty":
            (out / "mtp-routed-experts-int4.safetensors").write_bytes(b"1234")
        elif case == "no_experts_key":
            _edit_json(index, lambda i: i.update(weight_map={k: v for k, v in i["weight_map"].items()
                                                             if ".experts." not in k}))
        elif case == "second_draft":
            mf.write_tiny_draft(out / "copy")
        elif case == "not_ct":
            _edit_json(cfg, lambda c: c[q].update(quant_method="auto-round"))
        elif case == "nan":
            header, base, _ = wd.read_header(out / "mtp-dense.safetensors")
            begin = header["mtp.fc_hidden.weight"]["data_offsets"][0]
            with open(out / "mtp-dense.safetensors", "r+b") as f:
                f.seek(base + begin)
                f.write(np.array([0x7FC0], "<u2").tobytes())
        elif case == "bad_shape":
            header, base, _ = wd.read_header(out / "mtp-routed-experts-int4.safetensors")
            begin = header["mtp.layers.0.mlp.experts.3.down_proj.weight_shape"]["data_offsets"][0]
            with open(out / "mtp-routed-experts-int4.safetensors", "r+b") as f:
                f.seek(base + begin)
                f.write(np.array([5, 5], "<i8").tobytes())
        elif case == "unindexed":
            _edit_json(index, lambda i: i["weight_map"].update({"mtp.not_there.weight": "mtp-dense.safetensors"}))
        elif case == "short_data":
            data = (out / "mtp-routed-experts-int4.safetensors").read_bytes()
            (out / "mtp-routed-experts-int4.safetensors").write_bytes(data[:-10])

    result = wd.check_draft_dir(_broken(src, tmp_path, edit))
    assert not result["ok"] and any(expected in p for p in result["problems"]), result["problems"]


def test_the_reference_comparison_catches_changed_frozen_tensors_and_files(draft, tmp_path):
    src, _ = draft
    out = _broken(src, tmp_path, lambda out: None)
    assert wd.check_draft_dir(out, reference=src, trained_names=TRAINABLE[:2])["ok"]
    result = wd.check_draft_dir(out, reference=src, trained_names=TRAINABLE[:1])
    assert not result["ok"] and "not trained" in result["problems"][0]
    # a frozen tensor (the indexer) rewritten, and a changed expert file
    header, base, _ = wd.read_header(out / "mtp-dense.safetensors")
    begin = header["mtp.layers.0.self_attn.indexer.k_layernorm.weight"]["data_offsets"][0]
    with open(out / "mtp-dense.safetensors", "r+b") as f:
        f.seek(base + begin)
        f.write(b"\x00\x3f")
    data = bytearray((out / "mtp-routed-experts-int4.safetensors").read_bytes())
    data[-1] ^= 1
    (out / "mtp-routed-experts-int4.safetensors").write_bytes(bytes(data))
    result = wd.check_draft_dir(out, reference=src, trained_names=TRAINABLE[:2])
    problems = " | ".join(result["problems"])
    assert "indexer.k_layernorm" in problems and "mtp-routed-experts-int4.safetensors differs" in problems


def test_write_refuses_bad_trained_files_and_a_used_output(draft, tmp_path):
    src, _ = draft
    shapes = wd.dense_layout(mf.TINY_TEXT)
    bad = tmp_path / "bad.safetensors"
    name = "mtp.fc_hidden.weight"
    for items, message in (
            ([(name, "BF16", mf.bf16_bits(np.zeros((3, 3))))], "the draft has BF16"),
            ([(name, "F32", np.zeros(shapes[name], np.float32))], "the draft has BF16"),
            ([("lm_head.weight2", "BF16", mf.bf16_bits(np.zeros(4)))], "not a tensor of"),
            ([(name, "BF16", np.full(shapes[name], 0x7F80, np.uint16))], "NaN or Inf")):
        mf.hd.write_safetensors(bad, mf.hd.encode_safetensors(items, {}))
        with pytest.raises(ValueError, match=re.escape(message)):
            wd.write_draft(src, bad, tmp_path / "o1", overwrite=True)
    good = tmp_path / "good.safetensors"
    mf.write_trained(good, mf.TINY_TEXT, TRAINABLE[:1])
    wd.write_draft(src, good, tmp_path / "o2")
    with pytest.raises(FileExistsError):
        wd.write_draft(src, good, tmp_path / "o2")


def test_tokenizer_files_from_the_target_or_none(draft, tmp_path):
    src, _ = draft
    trained = tmp_path / "t.safetensors"
    mf.write_trained(trained, mf.TINY_TEXT, TRAINABLE[:1])
    target = tmp_path / "target"
    target.mkdir()
    (target / "tokenizer.json").write_text('{"stub": "target"}')
    (target / "chat_template.jinja").write_text("target template")
    wd.write_draft(src, trained, tmp_path / "a", tokenizer_from="target", target_dir=target)
    assert (tmp_path / "a" / "tokenizer.json").read_text() == '{"stub": "target"}'
    assert not (tmp_path / "a" / "vocab.json").exists() and not (tmp_path / "a" / "compact_sources.json").exists()
    wd.write_draft(src, trained, tmp_path / "b", tokenizer_from="none")
    assert sorted(p.name for p in (tmp_path / "b").iterdir()) == sorted(
        ["config.json", wd.INDEX, "mtp-dense.safetensors", "mtp-routed-experts-int4.safetensors", wd.MANIFEST])
    assert wd.check_draft_dir(tmp_path / "b", reference=src)["ok"]


def test_the_cli_writes_checks_and_exits_non_zero_on_problems(draft, tmp_path):
    src, _ = draft
    trained = tmp_path / "t.safetensors"
    mf.write_trained(trained, mf.TINY_TEXT, TRAINABLE)
    script = str(ROOT / "scripts" / "mtp_write_draft.py")
    out = tmp_path / "cli"
    run = subprocess.run([sys.executable, "-I", script, "write", "--draft", str(src), "--trained", str(trained),
                          "--out", str(out)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout[run.stdout.index("{"):])["ok"]
    run = subprocess.run([sys.executable, "-I", script, "check", str(out), "--reference", str(src), "--trained",
                          str(trained)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    (out / "mtp-routed-experts-int4.safetensors").unlink()
    run = subprocess.run([sys.executable, "-I", script, "check", str(out)], capture_output=True, text=True)
    assert run.returncode == 1 and "missing or empty shard" in run.stdout
