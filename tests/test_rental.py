"""The rental runner (scripts/rental.py, scripts/rental_box.py): offer filter, job packing, the create-instance body,
the spend guard, and the box runner end to end against a fake Kaggle CLI (no network, no GPU, no money)."""
from __future__ import annotations

import json
import os
import stat
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rental  # noqa: E402
import rental_box  # noqa: E402


def _offer(i, name="RTX PRO 6000 S", dph=1.4, driver="580.173.02", gpus=1):
    return {"id": i, "gpu_name": name, "dph_total": dph, "driver_version": driver, "num_gpus": gpus,
            "cpu_ram": 200 * 1024, "cpu_cores_effective": 48, "disk_space": 800, "inet_down": 900}


def test_offers_need_driver_580_and_prefer_the_server_edition():
    offers = [_offer(1, "RTX PRO 6000 WS", 1.1), _offer(2, dph=1.5), _offer(3, dph=1.3, driver="570.172.08"),
              _offer(4, dph=1.45), _offer(5, gpus=2, dph=2.8)]
    assert [o["id"] for o in rental.good_offers(offers)] == [4, 2, 1]
    assert rental.OFFER_FILTER["num_gpus"] == {"eq": 1}  # Keith's setup wants exactly one RTX PRO 6000 in nvidia-smi


def test_input_paths_follow_both_kaggle_layouts(tmp_path):
    assert rental_box.input_paths("dataset", "keithtyser/duck-harness", tmp_path) == (
        tmp_path / "input" / "duck-harness", [tmp_path / "input" / "datasets" / "keithtyser" / "duck-harness"])
    model = "keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"
    path, links = rental_box.input_paths("model", model, tmp_path)
    # the path Keith's serving_setup.MODEL_KAGGLE_PATH and scripts/sglang_serving.py expect (framework lower-cased)
    assert path == tmp_path / "input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
    assert links == []
    comp = rental_box.input_paths("competition", "arc-prize-2026-arc-agi-3", tmp_path)
    assert comp == (tmp_path / "input" / "arc-prize-2026-arc-agi-3",
                    [tmp_path / "input" / "competitions" / "arc-prize-2026-arc-agi-3"])


def _built_folder(root: Path, name: str, internet: bool = False) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    meta = {"id": f"scottmahony/arc3-{name}", "code_file": f"arc3-{name}.ipynb", "enable_internet": internet,
            "dataset_sources": ["keithtyser/duck-harness", "jakobbrggen/anim"],
            "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
            "competition_sources": ["arc-prize-2026-arc-agi-3"], "kernel_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta))
    (folder / meta["code_file"]).write_text(json.dumps({"cells": [], "metadata": {}, "nbformat": 4,
                                                        "nbformat_minor": 5}))
    return folder


def test_pack_writes_flat_files_a_manifest_and_the_box_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(rental, "RENTAL_DIR", tmp_path / "rental")
    folders = [_built_folder(tmp_path / "built", "exp054"), _built_folder(tmp_path / "built", "exp064")]
    out = rental.pack("oct-a", folders)
    manifest = json.loads((out / "manifest.json").read_text())
    assert [r["name"] for r in manifest["runs"]] == ["exp054", "exp064"]
    assert manifest["results_prefix"] == "arc3-rental-oct-a" and manifest["owner"] == "scottmahony"
    for r in manifest["runs"]:
        assert (out / r["notebook"]).is_file() and (out / r["metadata"]).is_file()
    assert (out / "rental_box.py").read_text() == (ROOT / "scripts" / "rental_box.py").read_text()
    assert json.loads((out / "dataset-metadata.json").read_text())["id"] == "scottmahony/arc3-rental-oct-a"
    assert not [p for p in out.iterdir() if p.is_dir()]  # a dataset download returns sub-folders as zips


def test_pack_refuses_an_online_notebook_and_bad_job_names(tmp_path, monkeypatch):
    monkeypatch.setattr(rental, "RENTAL_DIR", tmp_path / "rental")
    with pytest.raises(SystemExit, match="internet"):
        rental.pack("ok", [_built_folder(tmp_path / "b", "x", internet=True)])
    with pytest.raises(SystemExit, match="job names"):
        rental.pack("bad name!", [_built_folder(tmp_path / "c", "y")])


def test_create_body_runs_the_boot_command_as_the_entrypoint_and_keeps_the_token_in_env():
    body = rental.create_body("oct-a", "tok'en")
    assert body["runtype"] == "args" and body["image"] == rental.IMAGE and body["disk"] == rental.DISK_GB
    assert body["args"][:2] == ["bash", "-c"]
    boot = body["args"][2]
    assert "tok" not in boot and "rental_box.py /root/job/manifest.json" in boot
    assert f"kaggle=={rental.KAGGLE_CLI_VERSION}" in boot and "--target /opt/kcli" in boot  # not the notebook's Python
    assert "-e KAGGLE_API_TOKEN='tok'\"'\"'en'" in body["env"]
    assert "-e RENTAL_JOB=scottmahony/arc3-rental-oct-a" in body["env"]
    assert len(boot) < 4048  # vast.ai's onstart limit, kept in case the boot moves there


def test_launch_refuses_without_the_owners_approval(tmp_path, monkeypatch):
    monkeypatch.setattr(rental, "RENTAL_DIR", tmp_path / "rental")
    called = []
    monkeypatch.setattr(rental, "vast", lambda *a, **k: called.append(a))
    with pytest.raises(SystemExit, match="spends money"):
        rental.launch("oct-a", 123, approved="  ")
    assert called == []


def test_quote_counts_setup_and_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(rental, "RENTAL_DIR", tmp_path / "rental")
    rental.pack("q", [_built_folder(tmp_path / "b", "a"), _built_folder(tmp_path / "b", "b")])
    q = rental.quote("q", _offer(9, dph=1.4))
    assert q["hours_estimate"] == rental.HOURS_SETUP + 2 * rental.HOURS_PER_RUN
    assert q["cost_estimate_usd"] == round(1.4 * q["hours_estimate"], 2)


FAKE_KAGGLE = r'''#!/usr/bin/env python3
import json, os, sys, tarfile, zipfile, io
from pathlib import Path
log = Path(os.environ["FAKE_KAGGLE_LOG"])
with log.open("a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
a = sys.argv[1:]
dest = Path(a[a.index("-p") + 1]) if "-p" in a else None
if a[:2] == ["datasets", "download"]:
    (dest / "file.txt").write_text(a[2])
elif a[:2] == ["models", "instances"]:
    inner = dest / "wrapper"
    inner.mkdir()
    (inner / "config.json").write_text("{}")
    (inner / "MODEL_MANIFEST.json").write_text("{}")
elif a[:2] == ["competitions", "download"]:
    with zipfile.ZipFile(dest / "c.zip", "w") as z:
        z.writestr("arc_agi_3_wheels/arc_agi-0-py3-none-any.whl", "w")
elif a[:2] == ["datasets", "create"]:
    assert "--public" not in a
    assert json.loads((dest / "dataset-metadata.json").read_text())["id"].startswith("scottmahony/")
else:
    sys.exit(3)
'''


def test_box_runner_end_to_end_with_a_fake_kaggle_cli(tmp_path, monkeypatch):
    job = tmp_path / "job"
    monkeypatch.setattr(rental, "RENTAL_DIR", tmp_path / "rental")
    packed = rental.pack("e2e", [_built_folder(tmp_path / "b", "exp054"), _built_folder(tmp_path / "b", "exp064")])
    packed.rename(job)
    cli = tmp_path / "kaggle"
    cli.write_text(FAKE_KAGGLE)
    cli.chmod(cli.stat().st_mode | stat.S_IEXEC)
    root, results, calls = tmp_path / "kroot", tmp_path / "results", tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_KAGGLE_LOG", str(calls))
    monkeypatch.setenv("KAGGLE_API_TOKEN", "secret-token")
    monkeypatch.setattr(rental_box, "KAGGLE_ROOT", root)
    monkeypatch.setattr(rental_box, "RESULTS", results)
    monkeypatch.setattr(rental_box, "KAGGLE_CLI", str(cli))
    monkeypatch.setattr(rental_box.time, "sleep", lambda s: None)
    seen_env = []

    def fake_execute(notebook, working, log_path, timeout_s):
        seen_env.append(rental_box.notebook_env())
        (working / "benchmark.json").write_text(json.dumps({"game_runs": [], "notebook": notebook.name}))
        log_path.write_text("ran\n")
        return {"rc": 0, "timed_out": False, "seconds": 1.0, "executor": "fake"}

    monkeypatch.setattr(rental_box, "execute", fake_execute)
    assert rental_box.main([str(job / "manifest.json")]) == 0

    model_dir = root / "input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
    assert (model_dir / "MODEL_MANIFEST.json").is_file()  # the wrapper folder was flattened
    assert (root / "input" / "datasets" / "keithtyser" / "duck-harness" / "file.txt").is_file()  # both layouts
    assert (root / "input" / "competitions" / "arc-prize-2026-arc-agi-3" / "arc_agi_3_wheels").is_dir()
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    downloads = [c for c in commands if c[1] == "download" or c[:2] == ["models", "instances"]]
    assert len(downloads) == 4  # each input once, although both runs list all of them
    creates = [c for c in commands if c[:2] == ["datasets", "create"]]
    assert len(creates) == 2
    for n, name in ((1, "exp054"), (2, "exp064")):
        folder = results / f"arc3-rental-e2e-{n}"
        run = json.loads((folder / "run.json").read_text())
        assert run["run"] == name and run["execution"]["rc"] == 0
        with tarfile.open(folder / "output.tar.gz") as tar:
            assert {"benchmark.json", f"arc3-{name}.log"} <= set(tar.getnames())
        assert "secret-token" not in (folder / "run.json").read_text()
    assert all("KAGGLE_API_TOKEN" not in env and "KAGGLE_IS_COMPETITION_RERUN" not in env for env in seen_env)
    assert json.loads((results / "job.json").read_text())[1]["results"]["dataset"] == "scottmahony/arc3-rental-e2e-2"


def test_box_runner_caps_a_silent_notebook(tmp_path, monkeypatch):
    """The wall-clock cap holds even when the executor prints nothing (a long silent cell)."""
    monkeypatch.setattr(rental_box, "has_papermill", lambda: False)
    working = tmp_path / "w"
    working.mkdir()
    fake = tmp_path / "fakepy"
    fake.write_text("#!/bin/sh\nexec sleep 60\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(rental_box.sys, "executable", str(fake))
    out = rental_box.execute(tmp_path / "nb.ipynb", working, working / "x.log", timeout_s=1.0)
    assert out["timed_out"] is True and out["seconds"] < 45


def test_boot_script_is_valid_bash(tmp_path):
    script = tmp_path / "boot.sh"
    script.write_text(rental.boot_command())
    assert os.system(f"bash -n {script}") == 0
