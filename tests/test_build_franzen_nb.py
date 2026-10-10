"""scripts/build_franzen_nb.py: our arms of Daniel Franzen's Milestone 2 notebook (kaggle/franzen/, Apache-2.0).
The competition rerun must stay exactly his; only the Save & Run demo settings and named knobs may change."""
from __future__ import annotations

import ast
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


@pytest.mark.parametrize("base", ["franzen", "dprime"])
def test_env_also_changes_a_plain_os_environ_line_of_cell_4(tmp_path, base):
    # the first-request grace (lesson 0038) is set as os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '900'
    changes = bf.build(tmp_path, "g", base=base, env={"ARC3_HTTP_RETRY_INITIAL_SECONDS": "2400"})
    assert changes == ["env ARC3_HTTP_RETRY_INITIAL_SECONDS=2400"]
    setup = next(c for c in _code_cells(tmp_path / "g.ipynb") if bf.SETUP_ANCHOR in c)
    assert "os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '2400'  # ours (--env)" in setup
    assert "= '900'" not in setup
    ns: dict = {}
    line = next(x for x in setup.splitlines() if "ARC3_HTTP_RETRY_INITIAL_SECONDS" in x)
    exec("import os\n" + line, ns)
    assert ns["os"].environ.pop("ARC3_HTTP_RETRY_INITIAL_SECONDS") == "2400"
    with pytest.raises(SystemExit, match="--env-add adds"):
        bf.build(tmp_path / "x", "x", base=base, env={"ARC3_NOT_SET_ANYWHERE": "1"})


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


@pytest.mark.parametrize("base", ["franzen", "dprime"])
def test_input_fallback_wraps_every_input_path_and_resolves_both_mount_layouts(tmp_path, base):
    changes = bf.build(tmp_path / "f", "f", base=base, input_fallback=True, wait_inputs=0)
    assert any("resolves its 6 /kaggle/input paths" in c for c in changes)
    nb = json.loads((tmp_path / "f" / "f.ipynb").read_text())
    codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    unwrapped = [m.group(0) for src in codes for m in re.finditer(r"(?<!_ours_input\()(['\"])/kaggle/input/\S+?\1", src)]
    assert unwrapped == []  # every literal input path goes through the helper
    setup = next(src for src in codes if bf.SETUP_ANCHOR in src)
    assert setup.startswith(bf.INPUT_HELPER)
    for src in (c for c in codes if not c.startswith("%%")):  # valid Python apart from IPython's ! and % lines
        compile("\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("!", "%"))), "cell", "exec",
                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)

    existing = set()

    class FakePath:
        @staticmethod
        def exists(p):
            return p in existing

    class FakeOs:
        path = FakePath

    ns: dict = {}
    exec(bf.INPUT_HELPER, ns)
    ns["_ours_os"] = FakeOs
    new = "/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy"
    comp = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"
    model = "/kaggle/input/models/dfranzen/x/transformers/default/1"
    existing |= {new, comp}
    assert ns["_ours_input"](new) == new and ns["_ours_input"](comp) == comp  # new layout: unchanged
    existing.clear()
    existing |= {"/kaggle/input/taaf-kaggle-source-bundle-copy", "/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"}
    assert ns["_ours_input"](new) == "/kaggle/input/taaf-kaggle-source-bundle-copy"
    assert ns["_ours_input"](comp) == "/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"
    assert ns["_ours_input"](model) == model  # no alternative: the original path, so the wait or the cell reports it


# --- --probe (docs/research/beat-tufa/fidelity-probe.md) -------------------------------------------------------------

PROBE_DIR = ROOT / "kaggle" / "fidelity"  # the repo's copy of the probe dataset's metadata (the data is not in git)


def _unwrapped_inputs(sources: list[str]) -> list[str]:
    return [m.group(0) for src in sources
            for m in re.finditer(r"(?<!_ours_input\()(['\"])/kaggle/input/\S+?\1", src)]


@pytest.mark.parametrize("base", ["franzen", "dprime"])
def test_probe_replaces_the_benchmark_and_checks_its_prompts_before_the_server_starts(tmp_path, base):
    changes = bf.build(tmp_path / "p", "p", base=base, input_fallback=True, probe=PROBE_DIR)
    manifest = json.loads((PROBE_DIR / "manifest.json").read_text())
    data_sha = manifest["files"][0]["sha256"]
    assert changes[-1].startswith(f"fidelity probe instead of the benchmark (base arm): replays the "
                                  f"{manifest['requests']} requests of scottmahony/arc3-fidelity-prompts "
                                  f"(requests.jsonl sha256 {data_sha[:12]}; checked right after cell 4)")
    bf.build(tmp_path / "ref", "ref", base=base, input_fallback=True)  # the same notebook without the probe
    ours, ref = _cells(tmp_path / "p" / "p.ipynb"), _cells(tmp_path / "ref" / "ref.ipynb")
    kinds = [c["cell_type"] for c in json.loads((tmp_path / "ref" / "ref.ipynb").read_text())["cells"]]
    setup = next(i for i, c in enumerate(ref) if kinds[i] == "code" and bf.SETUP_ANCHOR in c)
    run = next(i for i, c in enumerate(ref) if kinds[i] == "code" and bf.PROBE_RUN_ANCHOR in c)
    header = next(i for i, c in enumerate(ref) if kinds[i] == "markdown" and c.startswith(bf.PROBE_MD_ANCHOR))
    # up to and including cell 4: unchanged; then the probe module and the data check
    assert ours[1:setup + 1] == ref[1:setup + 1]
    assert bf.franzen_tree.writefile_body(ours[setup + 1]) == (bf.PROBE_FILE, bf.PROBE_SCRIPT.read_text())
    preflight = ours[setup + 2]
    assert preflight.startswith(bf.PROBE_BEGIN) and preflight.rstrip().endswith(bf.PROBE_END)
    assert "find_dataset([_ours_input('/kaggle/input/datasets/scottmahony/arc3-fidelity-prompts')]" in preflight
    assert f"{{'requests.jsonl': '{data_sha}'}}" in preflight
    # everything from cell 4 to the benchmark cell is theirs (the server launch byte for byte), the section header
    # is ours, the benchmark cell is the probe, and nothing follows it
    middle = [i for i in range(setup + 1, run) if i != header]
    assert [ours[i + 2] for i in middle] == [ref[i] for i in middle]
    assert ours[header + 2].startswith("## 9. Fidelity probe (ours, replaces the benchmark)")
    assert len(ours) == run + 3
    probe = ours[-1]
    assert probe.startswith(bf.PROBE_BEGIN) and "arc3_probe.run(" in probe and "TRUE_SUBMISSION" in probe
    assert "arm='base', expect_num_experts=None," in probe
    assert "max_tokens=192, top_logprobs=5, concurrency=8, max_minutes=150," in probe
    assert "health_deadline=NOTEBOOK_START_TIME + 55 * 60)" in probe and "alive=lambda: proc.poll() is None" in probe
    nb = json.loads((tmp_path / "p" / "p.ipynb").read_text())
    codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert not any(bf.PROBE_RUN_ANCHOR in c for c in codes)
    assert _unwrapped_inputs(codes) == []
    for src in (c for c in codes if not c.startswith("%%")):  # valid Python apart from IPython's ! and % lines
        compile("\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("!", "%"))), "cell",
                "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    meta = json.loads((tmp_path / "p" / "kernel-metadata.json").read_text())
    ref_meta = json.loads((tmp_path / "ref" / "kernel-metadata.json").read_text())
    assert meta["dataset_sources"] == bf.SOURCES["dataset_sources"] + ["scottmahony/arc3-fidelity-prompts"]
    assert {k: v for k, v in meta.items() if k not in ("id", "title", "code_file", "dataset_sources")} == \
        {k: v for k, v in ref_meta.items() if k not in ("id", "title", "code_file", "dataset_sources")}
    assert meta["is_private"] is True and meta["docker_image"] == bf.IMAGE


def test_probe_reap_arm_differs_from_the_base_arm_only_by_the_pruning(tmp_path):
    common = {"base": "dprime", "input_fallback": True, "wait_inputs": 120, "probe": PROBE_DIR}
    bf.build(tmp_path / "b", "arc3-fidelity-base", **common)
    changes = bf.build(tmp_path / "r", "arc3-fidelity-reap448", reap_kept=REAP_KEPT, **common)
    assert changes[-1].startswith("fidelity probe instead of the benchmark (reap448 arm)")
    b, r = _cells(tmp_path / "b" / "arc3-fidelity-base.ipynb"), _cells(tmp_path / "r" / "arc3-fidelity-reap448.ipynb")
    launch = next(i for i, c in enumerate(b) if bf.LAUNCH_ANCHOR in c)
    assert [bf.franzen_tree.writefile_body(c)[0] for c in r[launch:launch + 3]] == list(bf.REAP_FILES.values())
    r = r[:launch] + r[launch + 3:]
    assert [i for i, (x, y) in enumerate(zip(b, r)) if x != y] == [0, launch, len(b) - 1]
    assert len(_reap_blocks(r[launch])) == 2  # the launcher differs by the two REAP blocks only
    assert r[launch].replace(_reap_blocks(r[launch])[0], "").replace(_reap_blocks(r[launch])[1], "") == b[launch]
    differing = [(x, y) for x, y in zip(b[-1].splitlines(), r[-1].splitlines()) if x != y]
    assert [x.strip()[:20] for x, _ in differing] == ["arm='base', expect_n", "build={'builder': 's"]
    assert "arm='reap448', expect_num_experts=448," in r[-1] and "'reap_kept': 'reap448_kept_experts.json" in r[-1]
    meta_b = json.loads((tmp_path / "b" / "kernel-metadata.json").read_text())
    meta_r = json.loads((tmp_path / "r" / "kernel-metadata.json").read_text())
    assert [k for k in meta_b if meta_b[k] != meta_r[k]] == ["id", "title", "code_file"]


def test_probe_refuses_what_would_make_it_unsafe_or_meaningless(tmp_path):
    for kwargs, message in [({}, "needs --input-fallback"),
                            ({"input_fallback": True, "full25": 25}, "replaces the benchmark run"),
                            ({"input_fallback": True, "patches": [SAMPLE], "apply_check": False},
                             "replaces the benchmark run"),
                            ({"input_fallback": True, "cfg": {"SPEC_ACCEPT_ACC": "0.5"}}, "must stay 1.0"),
                            ({"input_fallback": True, "cfg": {"SPEC_ACCEPT_SINGLE": "true"}}, "must stay 1.0")]:
        with pytest.raises(SystemExit, match=message):
            bf.build(tmp_path / "x", "x", probe=PROBE_DIR, **kwargs)
    assert bf.build(tmp_path / "ok", "ok", probe=PROBE_DIR, input_fallback=True, cfg={"SPEC_ACCEPT_ACC": "1"})
    stale = tmp_path / "stale"
    stale.mkdir()
    for name in ("dataset-metadata.json", "manifest.json"):
        (stale / name).write_text((PROBE_DIR / name).read_text())
    (stale / "requests.jsonl").write_text("{}\n")  # not the data the manifest describes
    with pytest.raises(SystemExit, match=r"does not match manifest\.json"):
        bf.build(tmp_path / "x", "x", probe=stale, input_fallback=True)
    (stale / "requests.jsonl").unlink()
    (stale / "dataset-metadata.json").write_text(json.dumps({"id": "Scott Mahony/x"}))
    with pytest.raises(SystemExit, match="is not owner/slug"):
        bf.build(tmp_path / "x", "x", probe=stale, input_fallback=True)
    with pytest.raises(SystemExit, match=r"needs dataset-metadata\.json and manifest\.json"):
        bf.build(tmp_path / "x", "x", probe=tmp_path / "nowhere", input_fallback=True)


def test_the_probe_cells_run_against_a_server(tmp_path):
    """The built notebook's probe module cell, data check and probe cell, executed against a fake SGLang server."""
    import time
    import types

    import fidelity_sample

    from tests.fidelity_fakes import FakeServer, write_fake_logs
    write_fake_logs(tmp_path / "logs")
    kaggle = tmp_path / "kaggle"
    data = kaggle / "input/datasets/scottmahony/arc3-fidelity-prompts"
    fidelity_sample.sample(tmp_path / "logs", data, per_game=4)
    bf.build(tmp_path / "nb", "p", base="dprime", input_fallback=True, probe=data)
    cells = _cells(tmp_path / "nb" / "p.ipynb")
    path, text = bf.franzen_tree.writefile_body(next(c for c in cells if c.startswith(f"%%writefile {bf.PROBE_FILE}")))
    (kaggle / Path(path).name).write_text(text)
    preflight = next(c for c in cells if c.startswith(bf.PROBE_BEGIN) and "find_dataset" in c)
    ns: dict = {}
    exec(bf.INPUT_HELPER, ns)
    with FakeServer() as server:
        ns.update(TRUE_SUBMISSION=False, WORKING_DIR=tmp_path, SERVED_MODEL_HOST="127.0.0.1",
                  SERVED_MODEL_PORT=server.port, SERVED_MODEL_NAME="flashnext", NOTEBOOK_START_TIME=time.time(),
                  proc=types.SimpleNamespace(poll=lambda: None), show_log_tail=lambda: None)
        exec(preflight.replace("/kaggle/", f"{kaggle}/"), ns)
        exec(cells[-1], ns)
    out = json.loads((tmp_path / "fidelity.json").read_text())
    assert out["arm"] == "base" and out["model"] == "flashnext"
    assert out["passes"]["seq"]["ok"] == out["passes"]["conc"]["ok"] == 8
    assert out["build"]["dataset"] == "scottmahony/arc3-fidelity-prompts" and out["build"]["base"] == "dprime"
    ns["TRUE_SUBMISSION"] = True
    with pytest.raises(RuntimeError, match="not a submission"):
        exec(cells[-1], ns)
    (data / "requests.jsonl").write_text("{}\n")  # a dataset version that is not the one built for: stops early
    with pytest.raises(RuntimeError, match="not the manifest's"):
        exec(preflight.replace("/kaggle/", f"{kaggle}/"), ns)


# --- --model, --reap-no-verify, --fail-fast ----------------------------------------------------------------------------

def _code_cells(path: Path) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(path.read_text())["cells"] if c["cell_type"] == "code"]


def _compiles(src: str) -> None:
    compile("\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("!", "%"))), "cell", "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


@pytest.mark.parametrize("base", ["franzen", "dprime"])
def test_model_swift_links_both_instances_into_one_view_and_swaps_the_model_source(tmp_path, base):
    changes = bf.build(tmp_path / "m", "m", base=base, model="swift", input_fallback=True, wait_inputs=0)
    assert any(c.startswith("serves Swift-1.5") for c in changes)
    codes = _code_cells(tmp_path / "m" / "m.ipynb")
    setup = next(src for src in codes if bf.SETUP_ANCHOR in src)
    assert bf.MODEL_LINE not in setup and setup.count(bf.MODEL_BEGIN) == 1
    assert setup.index(bf.MODEL_END) < setup.index("# >>> ours (--wait-inputs)")  # the view exists before the wait
    assert "intel-qwen3.8-flash-next" not in "".join(codes)
    for src in (c for c in codes if not c.startswith("%%")):
        _compiles(src)
    meta = json.loads((tmp_path / "m" / "kernel-metadata.json").read_text())
    assert meta["model_sources"] == ["dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1",
                                     *bf.MODELS["swift"]["sources"]]
    assert bf.INTEL_SOURCE not in meta["model_sources"] and meta["dataset_sources"] == bf.SOURCES["dataset_sources"]


def test_the_model_view_links_every_file_once_and_keeps_the_skipped_ones_out(tmp_path, capsys):
    import time
    code = bf._model_code("swift")
    ns = {"os": os, "time": time}
    exec(code.split("MODEL_DIR         =")[0], ns)  # the function only
    a = tmp_path / "models/o/s/pytorch/w4a16-a/1"
    b = tmp_path / "models/o/s/pytorch/w4a16-b/1"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "model-00001-of-00002.safetensors").write_text("a")
    (b / "model-00002-of-00002.safetensors").write_text("b")
    (b / "model.safetensors.index.json").write_text("{}")
    (b / "model_mtp.safetensors").write_text("mtp")
    pats = [str(tmp_path / "models/o/s/*/w4a16-a/1"), str(tmp_path / "models/o/s/*/w4a16-b/1")]
    view = tmp_path / "view"
    assert ns["_ours_model_view"](pats, str(view), ["model_mtp.safetensors"], timeout_s=0) == str(view)
    assert sorted(p.name for p in view.iterdir()) == ["model-00001-of-00002.safetensors",
                                                     "model-00002-of-00002.safetensors", "model.safetensors.index.json"]
    assert (view / "model-00001-of-00002.safetensors").read_text() == "a"
    assert "3 files from" in capsys.readouterr().out
    assert ns["_ours_model_view"](pats, str(view), ["model_mtp.safetensors"], timeout_s=0) == str(view)  # idempotent
    (a / "model.safetensors.index.json").write_text("{}")  # the same name in both instances
    with pytest.raises(RuntimeError, match="more than one instance"):
        ns["_ours_model_view"](pats, str(tmp_path / "view2"), [], timeout_s=0)
    with pytest.raises(RuntimeError, match="not mounted"):
        ns["_ours_model_view"]([str(tmp_path / "missing/*")], str(tmp_path / "view3"), [], timeout_s=0)


def test_model_with_reap_needs_reap_no_verify_which_leaves_the_router_check_out(tmp_path):
    kept = ROOT / "kaggle" / "franzen" / "reap448_kept_experts.json"
    with pytest.raises(SystemExit, match="--reap-no-verify"):
        bf.build(tmp_path / "x", "x", base="dprime", model="swift", reap_kept=kept, cfg={"MAXREQ": "14"})
    with pytest.raises(SystemExit, match="only applies to --reap-kept"):
        bf.build(tmp_path / "y", "y", base="dprime", reap_verify=False)
    changes = bf.build(tmp_path / "z", "z", base="dprime", model="swift", reap_kept=kept, reap_verify=False,
                       cfg={"MAXREQ": "14"})
    assert any("router sha256 check is off" in c for c in changes)
    written = [c.splitlines()[0] for c in _code_cells(tmp_path / "z" / "z.ipynb") if c.startswith("%%writefile")]
    assert f"%%writefile {bf.REAP_FILES['kept']}" in written and f"%%writefile {bf.REAP_FILES['meta']}" not in written
    verified = bf.build(tmp_path / "v", "v", base="dprime", reap_kept=kept, cfg={"MAXREQ": "14"})  # Intel: checked
    assert not any("router sha256 check is off" in c for c in verified)
    assert any(c.startswith(f"%%writefile {bf.REAP_FILES['meta']}") for c in _code_cells(tmp_path / "v" / "v.ipynb"))


def test_fail_fast_watches_the_server_from_right_before_the_benchmark(tmp_path):
    with pytest.raises(SystemExit, match="needs --full25"):
        bf.build(tmp_path / "x", "x", base="dprime", fail_fast=True)
    changes = bf.build(tmp_path / "f", "f", base="dprime", full25=25, fail_fast=True)
    assert any("fails fast" in c for c in changes)
    run = next(src for src in _code_cells(tmp_path / "f" / "f.ipynb") if "await bm.run(" in src)
    i, j = run.index(bf.FAIL_FAST_BEGIN), run.index(bf.FAIL_FAST_END)
    assert i < j < run.index("await bm.run(")
    _compiles(run)
    watch = run[i:j]

    class Exit(Exception):
        pass

    class FakeOs:
        @staticmethod
        def _exit(code):
            raise Exit(code)

    class Proc:
        def __init__(self, code):
            self.returncode = code

        def poll(self):
            return self.returncode

    class Clock:
        def __init__(self, sleeps):
            self.t, self.left = 0.0, sleeps

        def time(self):
            return self.t

        def sleep(self, s):
            if self.left == 0:
                raise StopIteration
            self.left -= 1
            self.t += s

    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Urllib:
        class request:
            healthy = True

            @classmethod
            def urlopen(cls, url, timeout):
                if not cls.healthy:
                    raise OSError("refused")
                return Resp()

    def run_watch(proc, healthy, sleeps):
        Urllib.request.healthy = healthy
        ns = {"os": FakeOs, "time": Clock(sleeps), "urllib": Urllib, "proc": proc, "show_log_tail": lambda: None,
              "SERVED_MODEL_PORT": 8001, "NOTEBOOK_START_TIME": 0.0, "TRUE_SUBMISSION": True}  # no thread here
        exec(watch, ns)
        ns["_ours_server_watch"]()

    with pytest.raises(Exit):  # the server process is gone
        run_watch(Proc(1), True, 5)
    with pytest.raises(StopIteration):  # alive and healthy: keeps watching
        run_watch(Proc(None), True, 200)
    with pytest.raises(Exit):  # alive but never healthy past the deadline
        run_watch(Proc(None), False, 200)
    assert "if not TRUE_SUBMISSION:" in watch  # never started in a competition rerun


def test_hot_tokens_writes_our_map_before_the_launcher_and_his_assert_checks_it(tmp_path):
    import hashlib
    hot = ROOT / "kaggle" / "franzen" / "hot_tokens_64k_arc.pt"
    meta = json.loads((ROOT / "kaggle" / "franzen" / "hot_tokens_64k_arc.meta.json").read_text())
    assert hashlib.sha256(hot.read_bytes()).hexdigest() == meta["sha256"]
    changes = bf.build(tmp_path / "h", "h", base="dprime", hot_tokens=hot, cfg={"MAXREQ": "14"})
    assert any(c.startswith("MTP draft FR-Spec map hot_tokens_64k_arc.pt") for c in changes)
    nb = tmp_path / "h" / "h.ipynb"
    assert nb.stat().st_size < 900_000  # Kaggle refuses a notebook near 1 MB (HTTP 400): the map must stay small
    codes = _code_cells(nb)
    launch = next(i for i, src in enumerate(codes) if bf.LAUNCH_ANCHOR in src)
    sha = re.search(r'TOKEN_MAP_SHA = "([0-9a-f]{64})"', codes[launch]).group(1)
    assert "becfa41d" not in codes[launch]
    assert f"tok = Path({bf.HOT_MAP_FILE!r})" in codes[launch] and "hot_tokens_64k.pt\", required" not in codes[launch]
    assert "assert sha256(tok) == TOKEN_MAP_SHA" in codes[launch]  # his check, now of the file the cell writes
    writer = codes[launch - 1]
    assert writer.startswith(bf.HOT_BEGIN) and len(writer) < 10_000
    _compiles(writer)
    _compiles(codes[launch])
    out = tmp_path / "arc3-hot-tokens.pt"
    for _ in range(2):  # the same bytes every time: his assert can hold a literal sha256
        exec(writer.replace(bf.HOT_MAP_FILE, str(out)), {})
        assert hashlib.sha256(out.read_bytes()).hexdigest() == sha
    assert bf._read_hot_ids(out) == bf._read_hot_ids(hot)  # the ids his torch.load will see
    with pytest.raises(SystemExit, match="FRSPEC is off"):
        bf.build(tmp_path / "x", "x", base="dprime", hot_tokens=hot, cfg={"FRSPEC": "False"})


def test_compact_ships_the_same_files_compressed_and_the_bed_reads_them_back(tmp_path):
    import hashlib
    patches = sorted((ROOT / "kaggle" / "franzen" / "patches").glob("ours-0[2-5]-*.patch"))[:3] + [SAMPLE]
    plain = bf.build(tmp_path / "p", "p", patches=patches, apply_check=False, reap_kept=REAP_KEPT)
    small = bf.build(tmp_path / "c", "c", patches=patches, apply_check=False, reap_kept=REAP_KEPT, compact=True)
    assert small[:-1] == plain and small[-1].startswith(f"the {len(patches) + 3} file(s) our cells write are shipped")
    p, c = _cells(tmp_path / "p" / "p.ipynb"), _cells(tmp_path / "c" / "c.ipynb")
    assert len(p) == len(c)
    differ = [i for i, (a, b) in enumerate(zip(p, c)) if a != b]
    assert differ[0] == 0 and len(differ) == 1 + len(patches) + 3  # the markdown summary and the written files
    for i in differ[1:]:
        assert p[i].startswith("%%writefile ") and c[i].startswith(bf.COMPACT_BEGIN)
        path, text = bf.franzen_tree.writefile_body(p[i])
        assert bf.franzen_tree.written_file(c[i]) == (path, text)
        _compiles(c[i])
        out = tmp_path / "w" / Path(path).name
        out.parent.mkdir(exist_ok=True)
        exec(c[i].replace(repr(path), repr(str(out))), {})  # what the notebook writes, byte for byte
        assert out.read_bytes() == text.encode("utf-8")
        broken = c[i].replace(hashlib.sha256(text.encode()).hexdigest(), "0" * 64)
        with pytest.raises(RuntimeError, match="corrupted in the notebook"):
            exec(broken.replace(repr(path), repr(str(out))), {})
    # scripts/franzen_bed.py takes our patches from either notebook
    assert bf.franzen_tree.our_patch_cells(tmp_path / "c" / "c.ipynb") == \
        bf.franzen_tree.our_patch_cells(tmp_path / "p" / "p.ipynb")
    assert len(bf.franzen_tree.our_patch_cells(tmp_path / "c" / "c.ipynb")) == len(patches)
    # the point: these files (188 KB here) cost the notebook about a quarter of their size (Kaggle refuses ~1 MB)
    written = sum(len(bf.franzen_tree.writefile_body(p[i])[1].encode()) for i in differ[1:])
    saved = (tmp_path / "p" / "p.ipynb").stat().st_size - (tmp_path / "c" / "c.ipynb").stat().st_size
    assert saved > 0.7 * written


def _draft_output(root: Path, dense: bytes = b"dense") -> tuple[Path, Path]:
    """A session-A output's mtp-draft/ with a manifest, and the manifest as pulled (a copy outside it)."""
    import hashlib
    d = root / "mtp-draft"
    d.mkdir(parents=True)
    (d / bf.DRAFT_DENSE).write_bytes(dense)
    (d / "config.json").write_text("{}")
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}
    (d / bf.DRAFT_MANIFEST).write_text(json.dumps({"replaced": ["mtp.fc.weight"], "files": files}, indent=1) + "\n")
    pulled = root.parent / f"{root.name}-pulled-manifest.json"
    pulled.write_bytes((d / bf.DRAFT_MANIFEST).read_bytes())
    return d, pulled


@pytest.mark.parametrize("base", ["franzen", "dprime"])
def test_draft_serves_session_as_output_instead_of_albucino_and_pins_its_version(tmp_path, base):
    _, manifest = _draft_output(tmp_path / "out")
    changes = bf.build(tmp_path / "d", "d", base=base, input_fallback=True, wait_inputs=0,
                       draft="scottmahony/arc3-mtp-session-a", draft_manifest=manifest)
    assert any(c.startswith("MTP draft: scottmahony/arc3-mtp-session-a's output mtp-draft/") for c in changes)
    codes = _code_cells(tmp_path / "d" / "d.ipynb")
    setup = next(src for src in codes if bf.SETUP_ANCHOR in src)
    assert bf.DRAFT_LINE not in setup and setup.count(bf.DRAFT_BEGIN) == 1
    assert setup.index(bf.DRAFT_END) < setup.index("# >>> ours (--wait-inputs)")  # resolved before the wait
    assert "albucino-qwen3-8-flash-next-drafter" not in "".join(codes)
    assert "'/kaggle/input/notebooks/scottmahony/arc3-mtp-session-a/mtp-draft'" in setup
    assert "'/kaggle/input/arc3-mtp-session-a/mtp-draft'" in setup
    for src in (c for c in codes if not c.startswith("%%")):
        _compiles(src)
    meta = json.loads((tmp_path / "d" / "kernel-metadata.json").read_text())
    assert bf.ALBUCINO_SOURCE not in meta["model_sources"] and meta["model_sources"] == [bf.INTEL_SOURCE]
    assert meta["kernel_sources"] == ["scottmahony/arc3-mtp-session-a"]
    with pytest.raises(SystemExit, match="go together"):
        bf.build(tmp_path / "x", "x", base=base, draft="scottmahony/arc3-mtp-session-a")
    with pytest.raises(SystemExit, match="OWNER/KERNEL"):
        bf.build(tmp_path / "x", "x", base=base, draft="arc3-mtp-session-a", draft_manifest=manifest)
    (tmp_path / "bad.json").write_text(json.dumps({"files": {}}))
    with pytest.raises(SystemExit, match="--draft-manifest"):
        bf.build(tmp_path / "x", "x", base=base, draft="o/k", draft_manifest=tmp_path / "bad.json")


def test_the_draft_dir_is_found_in_either_layout_and_refused_when_it_is_another_version(tmp_path, capsys):
    import time
    d, manifest = _draft_output(tmp_path / "input" / "arc3-mtp-session-a")  # the older layout
    spec = bf._draft_spec("me/arc3-mtp-session-a", manifest)
    code = bf._draft_code(spec)
    ns = {"os": os, "time": time}
    exec(code.split("DRAFT_MODEL_DIR   =")[0], ns)  # the function only
    cands = [str(tmp_path / "input" / "notebooks" / "me" / "arc3-mtp-session-a" / "mtp-draft"), str(d)]
    assert ns["_ours_draft_dir"](cands, spec["manifest_sha"], spec["dense_sha"], timeout_s=0) == str(d)
    assert "ours: MTP draft" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="another version"):
        ns["_ours_draft_dir"](cands, "0" * 64, spec["dense_sha"], timeout_s=0)
    (d / bf.DRAFT_DENSE).write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="its manifest says"):
        ns["_ours_draft_dir"](cands, spec["manifest_sha"], spec["dense_sha"], timeout_s=0)
    with pytest.raises(RuntimeError, match="not mounted"):
        ns["_ours_draft_dir"]([str(tmp_path / "missing")], spec["manifest_sha"], spec["dense_sha"], timeout_s=0)


def test_a_draft_probe_is_labelled_by_its_map_and_draft_and_records_both(tmp_path):
    _, manifest = _draft_output(tmp_path / "out")
    kept = ROOT / "kaggle" / "franzen" / "reap448_kept_experts.json"
    hot = ROOT / "kaggle" / "franzen" / "hot_tokens_64k_arc.pt"
    bf.build(tmp_path / "p", "p", base="dprime", input_fallback=True, probe=ROOT / "kaggle" / "fidelity",
             reap_kept=kept, hot_tokens=hot, draft="me/arc3-mtp-session-a", draft_manifest=manifest)
    probe = _code_cells(tmp_path / "p" / "p.ipynb")[-1]
    assert "arm='reap448-arc-draft', expect_num_experts=448," in probe
    assert "'hot_tokens': 'hot_tokens_64k_arc.pt sha256 " in probe and "'kernel': 'me/arc3-mtp-session-a'" in probe
