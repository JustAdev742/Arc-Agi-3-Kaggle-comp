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


# --- --env-add, --cfg, --server-env ------------------------------------------------------------------------------


def test_env_add_adds_only_new_keys_and_combines_with_full25(tmp_path):
    changes = bf.build(tmp_path, "add", full25=25, env_add={"EXPOSE_RESET": "on", "ARC3_DEATH_LEDGER": "1"})
    assert changes[1:] == ["env added EXPOSE_RESET=on", "env added ARC3_DEATH_LEDGER=1"]
    base, ours = _cells(bf.BASE), _cells(tmp_path / "add.ipynb")
    assert [i for i, (a, b) in enumerate(zip(base, ours)) if a != b] == [0, 4, 16]
    cell = ours[4]
    added = cell.index("'EXPOSE_RESET': 'on',  # ours (--env-add)")
    assert cell.index("'EXPOSE_UNDO': 'on',") < added < cell.index("\n}\n") < cell.index("if USE_PRIORITY_SCHEDULING")
    assert "'ARC3_DEATH_LEDGER': 1,  # ours (--env-add)" in cell
    assert "bm.solver.max_runtime_s_per_game = 25.0*60  # ours (--full25)" in ours[16]
    assert "bm.solver.max_runtime_s_per_game = 532*60" in ours[16]  # the rerun branch is his
    for key in ("EXPOSE_UNDO", "ARC3_MAX_ACTIVE_STREAMS", "ARC3_WARMUP_ACTION_GAMES"):  # setup_env, update, os.environ
        with pytest.raises(SystemExit, match="already sets it"):
            bf.build(tmp_path / key, "x", env_add={key: "1"})


def test_cfg_keeps_each_entry_type(tmp_path):
    changes = bf.build(tmp_path, "cfg", cfg={"MAXREQ": "12", "CUDAGRAPH_MAXBS": "12", "MAMBA_CACHE": "72",
                                             "MEMFRAC": "0.975", "SPEC_ACCEPT_ACC": "1", "AUTOTUNE": "false",
                                             "KVDTYPE": "bf16", "CTX": "(116+12+16)*1024"})
    assert changes[0] == "SGLang CFG MAXREQ=12" and len(changes) == 8
    base, ours = _cells(bf.BASE), _cells(tmp_path / "cfg.ipynb")
    assert [i for i, (a, b) in enumerate(zip(base, ours)) if a != b] == [0, 12]
    cell = ours[12]
    for line in ("    MAXREQ=12,  # ours (--cfg)", "    CUDAGRAPH_MAXBS=12,  # ours (--cfg)", "    MAMBA_CACHE=72,  # ours",
                 "    MEMFRAC=0.975,  # ours", "    SPEC_ACCEPT_ACC=1.0,  # ours", "    AUTOTUNE=False,  # ours",
                 '    KVDTYPE="bf16",  # ours', "    CTX=(116+12+16)*1024,  # ours"):
        assert line in cell
    assert "    SPEC=True,\n" in cell and "    SPEC_STEPS=3,\n" in cell  # neighbours with a shared prefix untouched
    namespace = {"SERVED_MODEL_NAME": "flashnext"}
    exec(cell[cell.index("CFG = dict("):cell.index("\n)\n") + 2], namespace)  # still valid Python, intended types
    assert namespace["CFG"]["MAXREQ"] == 12 and namespace["CFG"]["MEMFRAC"] == 0.975
    assert namespace["CFG"]["AUTOTUNE"] is False and namespace["CFG"]["CTX"] == 144 * 1024
    for key, value, message in [("MAXREQ", "12.5", "expected an int"), ("MEMFRAC", "high", "expected a number"),
                                ("SPEC", "maybe", "expected a bool"), ("SERVED_NAME", "x", "not a literal"),
                                ("NOPE", "1", "found 0 times"), ("SPEC", "1.5", "expected a bool")]:
        with pytest.raises(SystemExit, match=message):
            bf.build(tmp_path / "bad", "x", cfg={key: value})


def test_server_env_changes_the_value_the_server_is_started_with(tmp_path):
    changes = bf.build(tmp_path, "srv", server_env={"SGLANG_SM120_ONLINE_MXFP8": "true",
                                                    "SGLANG_MM_PREPROCESS_DEVICE": "cuda"})
    assert changes == ["SGLang server env SGLANG_SM120_ONLINE_MXFP8=true",
                       "SGLang server env SGLANG_MM_PREPROCESS_DEVICE=cuda"]
    cell = _cells(tmp_path / "srv.ipynb")[12]
    assert '"SGLANG_SM120_ONLINE_MXFP8": "true",' in cell and '"SGLANG_SM120_ONLINE_MXFP8": "0"' not in cell
    update = cell[cell.index("env.update({"):cell.index("\n})\n") + 3]
    env: dict[str, str] = {}
    exec(update, {"env": env, "CUDA_HOME": "/c", "VENV": "/v", "CC": "gcc", "CXX": "g++", "cache": "/k"})
    assert env["SGLANG_SM120_ONLINE_MXFP8"] == "true" and env["SGLANG_MM_PREPROCESS_DEVICE"] == "cuda"
    assert env["SGLANG_SM120_LOWM_FP8_WEIGHT"] == "0"  # its neighbour on the same line
    # the dictionary handed to subprocess.Popen is that env, and nothing sets the key after env.update
    after = cell[cell.index("\n})\n"):]
    assert "Popen(args, env=env" in after and "SGLANG_SM120_ONLINE_MXFP8" not in after
    for key, message in [("CUDA_HOME", "not a plain string"), ("SGLANG_NOT_THERE", "named 0 times"),
                         ("PATH", "named 2 times")]:  # PATH is also read with env.get("PATH")
        with pytest.raises(SystemExit, match=message):
            bf.build(tmp_path / "bad", "x", server_env={key: "1"})


# --- --patch -------------------------------------------------------------------------------------------------------

SAMPLE = ROOT / "tests" / "fixtures" / "franzen" / "sample-ours.patch"
needs_repo = pytest.mark.skipif(not (bf.franzen_tree.his_repo_path() / "ARC3-Inference").is_dir(),
                                reason="his repo (da-fr/arc-agi-3-solution) not available")


def test_patch_becomes_a_writefile_cell_after_his_and_an_apply_step_after_his(tmp_path):
    changes = bf.build(tmp_path, "p", patches=[SAMPLE], apply_check=False)
    assert changes[0].startswith("harness patch ours-01-sample-ours.patch (1 file(s): "
                                 "ARC3-Inference/inference/agent/tool_agent.py; sha256 ")
    base, ours = _cells(bf.BASE), _cells(tmp_path / "p.ipynb")
    assert len(ours) == len(base) + 1
    assert bf.franzen_tree.writefile_body(ours[3]) == ("/kaggle/ours-01-sample-ours.patch", SAMPLE.read_text())
    assert ours[2] == base[2]  # his patch cell
    assert [i for i, (a, b) in enumerate(zip(base, ours[:3] + ours[4:])) if a != b] == [0, 4]
    cell = ours[5]
    assert (cell.index(bf.APPLY_ANCHOR) < cell.index(bf.OURS_BEGIN) < cell.index(bf.OURS_END)
            < cell.index("TRUE_SUBMISSION ="))
    assert "['/kaggle/ours-01-sample-ours.patch']" in cell
    meta = json.loads((tmp_path / "p.ipynb").read_text())["cells"][3]
    assert meta["cell_type"] == "code" and meta["outputs"] == [] and meta["execution_count"] is None


def test_patch_validation(tmp_path):
    bad = tmp_path / "x.patch"
    for text, message in [("--- a/x\n+++ b/x\n", "must be `git diff` output"),
                          ("diff --git a/notes.txt b/notes.txt\n", "outside"),
                          ("diff --git a/ARC3-Inference/x b/ARC3-Inference/x\nGIT binary patch\n", "binary")]:
        bad.write_text(text)
        with pytest.raises(SystemExit, match=message):
            bf.build(tmp_path / "o", "x", patches=[bad], apply_check=False)


def _exec_apply_steps(cell4: str, bundle_dir: Path, kaggle_dir: Path) -> None:
    """Run cell 4's patch-application lines, his and ours, against BUNDLE_DIR (paths /kaggle/ -> KAGGLE_DIR)."""
    import os
    import subprocess as sp
    start = cell4.index('subprocess.run(["git", "apply", "--include=ARC3-Inference/*"')
    stop = cell4.index(bf.OURS_END) if bf.OURS_END in cell4 else cell4.index(bf.APPLY_ANCHOR) + len(bf.APPLY_ANCHOR)
    code = cell4[start:stop].replace("/kaggle/", f"{kaggle_dir}/")
    exec(code, {"subprocess": sp, "os": os, "Path": Path, "BUNDLE_DIR": bundle_dir})


def _write_patch_cells(cells: list[str], kaggle_dir: Path) -> None:
    for source in cells:
        if source.startswith("%%writefile /kaggle/"):
            path, text = bf.franzen_tree.writefile_body(source)
            (kaggle_dir / Path(path).name).write_text(text)


@needs_repo
def test_the_built_notebook_applies_his_patch_then_ours_to_the_bundle(tmp_path, capsys):
    """Cells 2, 3 and 4 of a built arm, executed against Tufa's bundle, give the tree the builder checked."""
    bf.build(tmp_path / "nb", "p", patches=[SAMPLE])
    cells = _cells(tmp_path / "nb" / "p.ipynb")
    kaggle = tmp_path / "kaggle"
    kaggle.mkdir()
    _write_patch_cells(cells, kaggle)
    share = bf.franzen_tree.build_bundle(tmp_path / "share")
    _exec_apply_steps(cells[5], share, kaggle)
    assert "our harness patches applied successfully: 1" in capsys.readouterr().out
    expected = tmp_path / "expected"
    bf.franzen_tree.notebook_bundle(expected, [SAMPLE])
    assert bf.franzen_tree.tree_hashes(share) == bf.franzen_tree.tree_hashes(expected)
    assert "OURS_SYSTEM_PROMPT_SUFFIX" in (share / "src/ARC3-Inference/inference/agent/tool_agent.py").read_text()


@needs_repo
def test_a_patch_that_does_not_apply_is_refused_at_build_time_and_stops_the_notebook(tmp_path):
    bad = tmp_path / "bad.patch"
    bad.write_text(SAMPLE.read_text().replace("    return prompt\n", "    return prompt + ''\n", 1))
    with pytest.raises(SystemExit, match="could not be patched"):
        bf.build(tmp_path / "nb", "bad", patches=[bad])
    bf.build(tmp_path / "nb", "bad", patches=[bad], apply_check=False)
    cells = _cells(tmp_path / "nb" / "bad.ipynb")
    kaggle = tmp_path / "kaggle"
    kaggle.mkdir()
    _write_patch_cells(cells, kaggle)
    share = bf.franzen_tree.build_bundle(tmp_path / "share")
    with pytest.raises(RuntimeError, match="did not apply: exit 1, 0 of 1 files"):
        _exec_apply_steps(cells[5], share, kaggle)


# --- --base dprime ---------------------------------------------------------------------------------------------------


def test_dprime_base_copies_its_cells_and_takes_the_same_options(tmp_path):
    assert bf.build(tmp_path / "plain", "d", base="dprime") == []
    dprime = _cells(bf.DPRIME)
    assert _cells(tmp_path / "plain" / "d.ipynb") == dprime
    nb = json.loads((tmp_path / "plain" / "d.ipynb").read_text())
    assert nb["metadata"]["kaggle"]["accelerator"] == "nvidiaRtxPro6000"  # the upstream notebook lacks it
    changes = bf.build(tmp_path / "srv", "d2", base="dprime", full25=25, cfg={"MAXREQ": "12"},
                       server_env={"SGLANG_SM120_ONLINE_MXFP8": "true"}, env={"ARC3_MAX_ACTIVE_STREAMS": "12"})
    assert len(changes) == 4
    ours = _cells(tmp_path / "srv" / "d2.ipynb")
    assert "#OURS_FORM" in "".join(ours)  # their priority module is still there
    demo = next(c for c in ours if "demo_excluded_games = []  # ours" in c)
    assert "max_runtime_s_per_game = 25.0*60  # ours (--full25)" in demo
    assert "bm.solver.max_runtime_s_per_game = 532*60" in demo
    assert any("    MAXREQ=12,  # ours (--cfg)" in c for c in ours)
    assert any('"SGLANG_SM120_ONLINE_MXFP8": "true",' in c for c in ours)
