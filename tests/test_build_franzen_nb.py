"""scripts/build_franzen_nb.py: our arms of Daniel Franzen's Milestone 2 notebook (kaggle/franzen/, Apache-2.0).
The competition rerun must stay exactly his; only the Save & Run demo settings and named knobs may change."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_franzen_nb as bf  # noqa: E402


def _cells(path: Path) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(path.read_text())["cells"]]


def test_unchanged_copy_is_his_notebook_with_our_private_offline_metadata(tmp_path):
    assert bf.build(tmp_path, "arc3-franzen-m2") == []
    assert _cells(tmp_path / "arc3-franzen-m2.ipynb") == _cells(bf.BASE)
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == "scottmahony/arc3-franzen-m2" and meta["is_private"] is True
    assert meta["enable_internet"] is False and meta["enable_gpu"] is True
    assert meta["model_sources"] == bf.SOURCES["model_sources"] and meta["dataset_sources"] == bf.SOURCES["dataset_sources"]
    nb = json.loads((tmp_path / "arc3-franzen-m2.ipynb").read_text())
    assert nb["metadata"]["kaggle"]["accelerator"] == "nvidiaRtxPro6000"  # scripts/push_eval.py requires it


def test_full25_changes_only_the_save_and_run_demo(tmp_path):
    changes = bf.build(tmp_path, "full", full25=121)
    assert len(changes) == 1
    base, ours = _cells(bf.BASE), _cells(tmp_path / "full.ipynb")
    differing = [i for i, (a, b) in enumerate(zip(base, ours)) if a != b]
    assert differing == [0, 16]  # the note in the first markdown cell, and the customization cell
    cell = ours[16]
    assert "demo_excluded_games = []  # ours" in cell and "max_runtime_s_per_game = 121.0*60" in cell
    # the competition rerun branch is his, untouched
    assert "bm.solver.concurrency = 120" in cell and "bm.solver.max_runtime_s_per_game = 532*60" in cell


def test_env_overrides_existing_knobs_only(tmp_path):
    changes = bf.build(tmp_path, "env", env={"MULTIMODAL_UPSCALE": "8", "ARC3_MAX_ACTIVE_STREAMS": "12"})
    assert changes == ["env MULTIMODAL_UPSCALE=8", "env ARC3_MAX_ACTIVE_STREAMS=12"]
    cell = _cells(tmp_path / "env.ipynb")[4]
    assert "'MULTIMODAL_UPSCALE': 8,  # ours (--env)" in cell and "'MULTIMODAL_UPSCALE': '10'" not in cell
    with pytest.raises(SystemExit, match="anchor"):
        bf.build(tmp_path / "x", "x", env={"NOT_A_KNOB": "1"})


def test_the_vendored_notebook_is_guarded(tmp_path, monkeypatch):
    copy = tmp_path / "base.ipynb"
    copy.write_bytes(bf.BASE.read_bytes() + b" ")
    monkeypatch.setattr(bf, "BASE", copy)
    with pytest.raises(SystemExit, match="unmodified"):
        bf.build(tmp_path / "o", "o")
