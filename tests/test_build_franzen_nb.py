"""scripts/build_franzen_nb.py: our arms of Daniel Franzen's Milestone 2 notebook (kaggle/franzen/, Apache-2.0).
The competition rerun must stay exactly his; only the Save & Run demo settings and named knobs may change."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_franzen_nb as bf  # noqa: E402
import sglang_reap_patch as rp  # noqa: E402


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
    # his Kaggle image, pinned: Kaggle's latest image is Python 3.13 and his wheels are cp312 (exp-070 failed)
    upstream = json.loads((ROOT / "kaggle" / "franzen" / "upstream-kernel-metadata.json").read_text())
    assert meta["docker_image"] == upstream["docker_image"] == bf.IMAGE
    assert meta["docker_image_pinning_type"] == "original" and meta["machine_shape"] == "NvidiaRtxPro6000"


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
    meta = json.loads((tmp_path / "plain" / "kernel-metadata.json").read_text())
    # his GPU image: the one D''s page lists is Kaggle's CPU image (no CUDA in exp-070d v2)
    assert meta["docker_image"] == bf.IMAGE and "kaggle-images/python" not in meta["docker_image"]
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


def test_copy_public_nb_pins_the_upstream_image_and_refuses_metadata_without_one(tmp_path):
    import copy_public_nb as cp
    up = ROOT / "kaggle" / "dprime" / "upstream-kernel-metadata.json"
    meta = cp.copy(bf.DPRIME, up, tmp_path / "out", "arc3-x")
    assert meta["docker_image"] == json.loads(up.read_text())["docker_image"]
    assert meta["docker_image_pinning_type"] == "original" and meta["machine_shape"] == "NvidiaRtxPro6000"
    assert meta["is_private"] is True and meta["enable_internet"] is False
    assert _cells(tmp_path / "out" / "arc3-x.ipynb") == _cells(bf.DPRIME)
    bare = json.loads(up.read_text())
    del bare["docker_image"]
    (tmp_path / "bare.json").write_text(json.dumps(bare))
    with pytest.raises(SystemExit, match="docker_image"):
        cp.copy(bf.DPRIME, tmp_path / "bare.json", tmp_path / "out2", "arc3-y")
    assert cp.copy(bf.DPRIME, tmp_path / "bare.json", tmp_path / "out3", "arc3-z", bf.IMAGE)["docker_image"] == bf.IMAGE


# --- --reap-kept -----------------------------------------------------------------------------------------------------

REAP_KEPT = ROOT / "kaggle" / "franzen" / "reap448_kept_experts.json"
REAP_META = ROOT / "kaggle" / "franzen" / "reap448_kept_experts.meta.json"
# The Pennyroyal sglang wheel (dataset dfranzen/pennyroyal-v253); the test that patches its real file skips without it.
WHEEL = Path(os.environ.get("PENNYROYAL_WHEEL", "/tmp/claude-0/-home-user-Arc-Agi-3-Kaggle-comp/"
                            "d342458e-03bd-545b-8a6d-06bca061963e/scratchpad/reap/dl-wheel/"
                            "sglang-0.5.19+gd00d88efc8d6-cp312-cp312-linux_x86_64.whl"))


def _reap_blocks(cell: str) -> list[str]:
    blocks, start = [], 0
    while (start := cell.find(bf.REAP_BEGIN, start)) >= 0:
        stop = cell.index(bf.REAP_END, start) + len(bf.REAP_END) + 1
        blocks.append(cell[start:stop])
        start = stop
    return blocks


def test_reap_kept_writes_its_files_before_the_launcher_and_edits_only_cell_12(tmp_path):
    changes = bf.build(tmp_path, "r", reap_kept=REAP_KEPT, cfg={"MAXREQ": "16"})
    assert changes[-1].startswith("REAP expert pruning at load: reap448_kept_experts.json")
    assert "keeps 448 of 512 routed experts in each of 48 layers" in changes[-1]
    base, ours = _cells(bf.BASE), _cells(tmp_path / "r.ipynb")
    assert len(ours) == len(base) + 3
    assert [bf.franzen_tree.writefile_body(c) for c in ours[12:15]] == [
        (bf.REAP_FILES["script"], bf.REAP_SCRIPT.read_text()), (bf.REAP_FILES["kept"], REAP_KEPT.read_text()),
        (bf.REAP_FILES["meta"], REAP_META.read_text())]
    assert bf.REAP_FILES["meta"] == str(rp.meta_path(bf.REAP_FILES["kept"]))
    assert [i for i, (a, b) in enumerate(zip(base, ours[:12] + ours[15:])) if a != b] == [0, 12]
    cell = ours[15]
    apply_block, args_block = _reap_blocks(cell)
    # without our two blocks (and the --cfg line) the launcher is his, byte for byte
    stripped = cell.replace(apply_block, "").replace(args_block, "")
    assert stripped == base[12].replace("    MAXREQ=10,\n", "    MAXREQ=16,  # ours (--cfg)\n")
    # the patch runs after the install and env.update; the flags join args before the server is started
    assert (cell.index('run(install + ["--reinstall", "--no-deps", str(wheel)]') < cell.index("\n})\n")
            < cell.index(apply_block) < cell.index(bf.REAP_APPLY_ANCHOR) < cell.index(args_block)
            < cell.index(bf.REAP_ARGS_ANCHOR) < cell.index("subprocess.Popen(args, env=env"))
    namespace: dict = {"args": []}
    exec(args_block, namespace)
    assert namespace["args"] == ["--json-model-override-args", '{"text_config": {"num_experts": 448}}',
                                 "--speculative-draft-model-override-args", "{}"]
    # the dprime base takes it the same way
    bf.build(tmp_path / "d", "d", base="dprime", reap_kept=REAP_KEPT)
    d = _cells(tmp_path / "d" / "d.ipynb")
    launch = next(i for i, c in enumerate(d) if bf.LAUNCH_ANCHOR in c)
    assert d[launch - 3].startswith("%%writefile /kaggle/arc3-reap-patch.py\n") and len(_reap_blocks(d[launch])) == 2


def test_reap_kept_refuses_a_bad_list(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"0": [2, 1]}))
    with pytest.raises(SystemExit, match=r"--reap-kept: .*strictly increasing"):
        bf.build(tmp_path / "o", "o", reap_kept=bad)
    short = tmp_path / "short.json"
    short.write_text(json.dumps({"0": [1, 2]}))
    (tmp_path / "short.meta.json").write_text(REAP_META.read_text())
    with pytest.raises(SystemExit, match="does not cover the layers"):
        bf.build(tmp_path / "o", "o", reap_kept=short)


def _run_reap_apply(cell: str, cells: list[str], venv: Path, kaggle: Path) -> dict:
    """Execute the launcher's own run() and our apply block, with /kaggle/ -> KAGGLE and VENV -> VENV."""
    import shlex
    import subprocess as sp
    for source in cells:
        if source.startswith("%%writefile /kaggle/arc3-reap"):
            path, text = bf.franzen_tree.writefile_body(source)
            (kaggle / Path(path).name).write_text(text)
    namespace = {"subprocess": sp, "shlex": shlex, "sys": sys, "Path": Path, "VENV": str(venv), "env": {}}
    exec(cell[cell.index("def run(cmd"):cell.index("\n\n\ndef find_unique")], namespace)
    exec(_reap_blocks(cell)[0].replace("/kaggle/", f"{kaggle}/"), namespace)
    return namespace["env"]


def _fake_venv(tmp_path: Path, text: str) -> tuple[Path, Path]:
    model = tmp_path / "venv/lib/python3.12/site-packages" / rp.MODEL_FILE
    model.parent.mkdir(parents=True)
    model.write_text(text)
    kaggle = tmp_path / "kaggle"
    kaggle.mkdir()
    return tmp_path / "venv", kaggle


def test_reap_apply_step_stops_the_notebook_on_a_file_it_was_not_written_for(tmp_path, capsys):
    bf.build(tmp_path / "nb", "r", reap_kept=REAP_KEPT)
    cells = _cells(tmp_path / "nb" / "r.ipynb")
    venv, kaggle = _fake_venv(tmp_path, "class Qwen4ExpForConditionalGeneration:\n    pass\n")
    with pytest.raises(RuntimeError, match="Command failed"):
        _run_reap_apply(cells[15], cells, venv, kaggle)
    assert "arc3 REAP patch FAILED" in capsys.readouterr().out


@pytest.mark.skipif(not WHEEL.is_file(), reason="Pennyroyal sglang wheel not available (PENNYROYAL_WHEEL)")
def test_reap_apply_step_patches_the_real_installed_file_and_sets_the_server_env(tmp_path, capsys):
    import zipfile
    with zipfile.ZipFile(WHEEL) as wheel:
        original = wheel.read(str(rp.MODEL_FILE)).decode()
    bf.build(tmp_path / "nb", "r", reap_kept=REAP_KEPT)
    cells = _cells(tmp_path / "nb" / "r.ipynb")
    venv, kaggle = _fake_venv(tmp_path, original)
    env = _run_reap_apply(cells[15], cells, venv, kaggle)
    assert env == {"ARC3_REAP_KEPT_EXPERTS": f"{kaggle}/arc3-reap-kept.json"}
    site = venv / "lib/python3.12/site-packages"
    assert (site / rp.MODEL_FILE).read_text() == rp.patch_text(original)
    assert "48 layers x 448 experts, router sha256 for every layer" in capsys.readouterr().out
    _run_reap_apply(cells[15], cells, venv, kaggle)  # a rerun of the cell (install skipped) is a no-op
    assert "already patched" in capsys.readouterr().out


def test_wait_inputs_waits_in_cell_4_before_the_bundle_copy_and_reports_missing_inputs(tmp_path, capsys):
    changes = bf.build(tmp_path / "w", "w", base="dprime", wait_inputs=0)
    assert any("waits up to 0 s" in c for c in changes)
    cell = next(c for c in _cells(tmp_path / "w" / "w.ipynb") if bf.WAIT_ANCHOR in c)
    i, j = cell.index("# >>> ours (--wait-inputs)"), cell.index(bf.WAIT_ANCHOR)
    assert i < j and re.search(r"ORIG_BUNDLE_DIR\s+=", cell[:i])  # after his path constants, before his rm/cp
    code = cell[i:j]
    import os
    import time
    present = {k: tmp_path / k for k in ("bundle", "wheels", "model", "draft")}
    (present["bundle"] / "src").mkdir(parents=True)
    (present["wheels"] / "wheels").mkdir(parents=True)
    present["model"].mkdir()
    present["draft"].mkdir()
    ns = {"os": os, "time": time, "ORIG_BUNDLE_DIR": str(present["bundle"]), "WHEELHOUSE_DIR": str(present["wheels"]),
          "MODEL_DIR": str(present["model"]), "DRAFT_MODEL_DIR": str(present["draft"])}
    exec(code, ns)
    assert "inputs mounted after 0 s" in capsys.readouterr().out
    ns["DRAFT_MODEL_DIR"] = str(tmp_path / "missing")
    with pytest.raises(RuntimeError, match="inputs not mounted after 0 s"):
        exec(code, ns)
