"""The rental runner (scripts/rental.py, scripts/rental_box.py): offer filter and prices, job packing, the
create-instance body, the spend guards, and the box runner end to end against a fake Kaggle CLI (no network, no GPU,
no money). Each guard here answers a finding of the 2026-09-29 review (docs/research/rental-runner.md)."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rental  # noqa: E402
import rental_box  # noqa: E402


def _offer(i, name="RTX PRO 6000 S", dph=1.4, driver="580.173.02", gpus=1, ram_gb=200, frac=1.0, days=30,
           storage=0.2, down_tb=2.7):
    return {"id": i, "gpu_name": name, "dph_total": dph, "driver_version": driver, "num_gpus": gpus,
            "cpu_ram": ram_gb * 1024, "gpu_frac": frac, "cpu_cores_effective": 48, "disk_space": 800,
            "inet_down": 900, "duration": days * 86400, "storage_cost": storage,
            "storage_total_cost": storage * 8 / 720, "internet_down_cost_per_tb": down_tb}


# --- offers and prices ------------------------------------------------------------------------------------------


def test_offers_filter_on_the_rentals_ram_share_driver_and_contract():
    offers = [_offer(1, "RTX PRO 6000 WS", 1.1), _offer(2, dph=1.5), _offer(3, dph=1.3, driver="570.172.08"),
              _offer(4, dph=1.45), _offer(5, gpus=2, dph=2.8),
              _offer(6, dph=1.2, ram_gb=252, frac=0.25),  # 63 GB share: cannot hold the pinned PLE table
              _offer(7, dph=1.2, days=1)]  # contract ends tomorrow
    assert [o["id"] for o in rental.good_offers(offers)] == [1, 4, 2]  # price first; the S card wins within 10%
    assert [o["id"] for o in rental.good_offers([_offer(1, "RTX PRO 6000 WS", 1.3), _offer(2, dph=1.4)])] == [2, 1]
    assert rental.ram_share_gb(_offer(8, ram_gb=377, frac=0.5)) == pytest.approx(188.5)


def test_prices_include_our_disk_and_rank_by_it():
    cheap_gpu_dear_disk = _offer(1, dph=1.41, storage=0.92)  # the reviewer's example host
    assert rental.disk_dph(cheap_gpu_dear_disk) == pytest.approx(0.92 * 500 / 720)
    assert rental.run_dph(cheap_gpu_dear_disk) == pytest.approx(1.41 - 0.92 * 8 / 720 + 0.92 * 500 / 720)
    other = _offer(2, dph=1.50, storage=0.1)
    assert [o["id"] for o in rental.good_offers([cheap_gpu_dear_disk, other])] == [2, 1]
    dear_download = _offer(3, dph=1.45, storage=0.1, down_tb=29.3)  # ~$4.7 for the ~160 GB of inputs
    assert rental.download_cost(dear_download) == pytest.approx(29.3 * rental.SETUP_DOWNLOAD_TB)
    assert [o["id"] for o in rental.good_offers([dear_download, other])] == [2, 3]


# --- packing ------------------------------------------------------------------------------------------------------


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


@pytest.fixture
def rental_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(rental, "RENTAL_DIR", tmp_path / "rental")
    return tmp_path


def test_pack_writes_flat_files_a_manifest_the_box_runner_and_the_job_sha(rental_dir):
    arm = _built_folder(rental_dir / "built", "exp054")
    out = rental.pack("oct-a", [arm, arm])  # one arm twice: two runs
    manifest = json.loads((out / "manifest.json").read_text())
    assert [r["name"] for r in manifest["runs"]] == ["exp054", "exp054"]
    assert manifest["results_prefix"] == "arc3-rental-oct-a" and manifest["owner"] == "scottmahony"
    for r in manifest["runs"]:
        assert (out / r["notebook"]).is_file() and (out / r["metadata"]).is_file()
    assert (out / "rental_box.py").read_text() == (ROOT / "scripts" / "rental_box.py").read_text()
    assert not [p for p in out.iterdir() if p.is_dir()]  # a dataset download returns sub-folders as zips
    assert rental.packed_sha("oct-a") == rental_box.job_sha(out)
    (out / "run1.ipynb").write_text("changed")
    assert rental.packed_sha("oct-a") != rental_box.job_sha(out)  # any change to the job changes its sha


def test_pack_refuses_online_notebooks_bad_names_and_launched_jobs(rental_dir):
    with pytest.raises(SystemExit, match="internet"):
        rental.pack("okay", [_built_folder(rental_dir / "b", "x", internet=True)])
    ok = _built_folder(rental_dir / "c", "y")
    for bad in ("bad name", "Upper", "-lead", "a--b", "ab", "x" * 25):
        with pytest.raises(SystemExit, match="job names"):
            rental.pack(bad, [ok])
    rental.pack("done-1", [ok])
    (rental.job_dir("done-1") / "launch.json").write_text("{}")
    with pytest.raises(SystemExit, match="was launched"):
        rental.pack("done-1", [ok])


# --- launch -------------------------------------------------------------------------------------------------------


def test_create_body_matches_the_cli_args_mode_and_keeps_secrets_in_env():
    body = rental.create_body("oct-a", "tok'en", "abc123", 7.25)
    assert body["runtype"] == "args" and body["onstart"] == "bash" and body["args"][0] == "-c"
    assert body["image"] == rental.IMAGE and body["disk"] == rental.DISK_GB and body["cancel_unavail"] is True
    assert body["env"] == {"KAGGLE_API_TOKEN": "tok'en", "RENTAL_JOB": "scottmahony/arc3-rental-oct-a",
                           "RENTAL_JOB_SHA": "abc123", "RENTAL_CAP_HOURS": "7.25"}
    boot = body["args"][1]
    assert "tok" not in boot and "rental_box.py /root/job/manifest.json" in boot
    assert f"kaggle=={rental.KAGGLE_CLI_VERSION}" in boot and "--target /opt/kcli" in boot  # not the notebook's Python
    assert 'timeout --kill-after=300 "${RENTAL_CAP_HOURS}h"' in boot and "trap" in boot and '"stopped"' in boot
    assert len(boot) < 4048


def test_boot_script_is_valid_bash(tmp_path):
    script = tmp_path / "boot.sh"
    script.write_text(rental.boot_command())
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0


def test_launch_refuses_without_the_owners_approval(rental_dir, monkeypatch):
    called = []
    monkeypatch.setattr(rental, "vast", lambda *a, **k: called.append(a))
    with pytest.raises(SystemExit, match="spends money"):
        rental.launch("oct-a", 123, approved="  ")
    assert called == []


def _launch_ready(rental_dir, monkeypatch, uploaded_sha=None, put=None, listed=()):
    rental.pack("oct-b", [_built_folder(rental_dir / "b", "exp054")])
    calls, state = [], {"put": False}

    def fake_vast(method, path, body=None, key=None, timeout=60):
        calls.append((method, path))
        if method == "GET" and path == "/instances/":  # the instance shows up once a create was attempted
            return {"instances": [*listed, *([{"id": 77, "label": "arc3-oct-b"}] if state["put"] else [])]}
        if method == "PUT" and path.startswith("/asks/"):
            state["put"] = True
            return put(body) if put else {"success": True, "new_contract": 555}
        raise AssertionError((method, path))

    monkeypatch.setattr(rental, "vast", fake_vast)
    monkeypatch.setattr(rental, "vast_key", lambda: "k")
    monkeypatch.setattr(rental, "find_offer", lambda i: _offer(i))
    monkeypatch.setattr(rental, "uploaded_sha", lambda job: uploaded_sha or rental.packed_sha(job))
    monkeypatch.setattr(rental, "kaggle_env", lambda: {"KAGGLE_API_TOKEN": "t"})
    return calls


def test_launch_refuses_a_stale_uploaded_job_before_renting(rental_dir, monkeypatch):
    calls = _launch_ready(rental_dir, monkeypatch, uploaded_sha="0" * 64)
    with pytest.raises(SystemExit, match="differs from the local pack"):
        rental.launch("oct-b", 9, approved="owner: yes, calib054 for $11")
    assert not any(m == "PUT" for m, _ in calls)
    assert not (rental.job_dir("oct-b") / "launch.json").exists()


def test_launch_records_the_instance(rental_dir, monkeypatch):
    calls = _launch_ready(rental_dir, monkeypatch)
    rec = rental.launch("oct-b", 9, approved="owner: yes")
    assert rec["instance"] == 555 and rec["status"] == "created" and rec["job_sha"] == rental.packed_sha("oct-b")
    assert ("PUT", "/asks/9/") in calls
    assert json.loads((rental.job_dir("oct-b") / "launch.json").read_text())["instance"] == 555
    with pytest.raises(SystemExit, match="launch record"):  # never a second box for the same job
        rental.launch("oct-b", 9, approved="owner: yes")


def test_launch_recovers_the_instance_by_label_after_a_lost_response(rental_dir, monkeypatch):
    def lost(body):
        raise OSError("timed out")

    _launch_ready(rental_dir, monkeypatch, put=lost)
    rec = rental.launch("oct-b", 9, approved="owner: yes")
    assert rec["instance"] == 77 and rec["status"] == "created (recovered by label)"


def test_launch_refuses_when_an_instance_with_the_label_exists(rental_dir, monkeypatch):
    _launch_ready(rental_dir, monkeypatch, listed=[{"id": 5, "label": "arc3-oct-b"}])
    with pytest.raises(SystemExit, match="exists already"):
        rental.launch("oct-b", 9, approved="owner: yes")


def test_quote_counts_setup_runs_disk_and_the_idle_disk(rental_dir):
    arm = _built_folder(rental_dir / "b", "a")
    rental.pack("q-1", [arm, arm])
    offer = _offer(9, dph=1.4, storage=0.9)
    q = rental.quote("q-1", offer)
    assert q["hours_estimate"] == rental.HOURS_SETUP + 2 * rental.HOURS_PER_RUN
    assert q["cost_estimate_usd"] == round(rental.run_dph(offer) * q["hours_estimate"] + rental.download_cost(offer), 2)
    assert q["disk_per_day_after_exit_usd"] == round(0.9 * 500 / 30, 2)


# --- the box runner -------------------------------------------------------------------------------------------------


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


def test_finish_model_extracts_leftover_tars_and_flattens_only_a_model_wrapper(tmp_path):
    dest = tmp_path / "m"
    (dest / "wrap" / "wrap").mkdir(parents=True)  # a child with the wrapper's own name
    (dest / "wrap" / "config.json").write_text("{}")
    rental_box._finish_model(dest)
    assert (dest / "config.json").is_file() and (dest / "wrap").is_dir() and not (dest / "wrap" / "config.json").exists()

    keep = tmp_path / "k"
    (keep / "sub").mkdir(parents=True)
    (keep / "sub" / "x.bin").write_text("x")  # no model marker inside: not a wrapper, left alone
    rental_box._finish_model(keep)
    assert (keep / "sub" / "x.bin").is_file()

    tarred = tmp_path / "t"
    tarred.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    (src / "MODEL_MANIFEST.json").write_text("{}")
    with tarfile.open(tarred / "model.tar.gz", "w:gz") as tar:
        tar.add(src / "MODEL_MANIFEST.json", arcname="MODEL_MANIFEST.json")
    rental_box._finish_model(tarred)  # the CLI's retry skips extracting a complete download
    assert (tarred / "MODEL_MANIFEST.json").is_file() and not (tarred / "model.tar.gz").exists()


def test_preflight_gates_the_gpu_driver_ram_and_shm():
    good = {"gpus": ["NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 580.173.02, 12.0"],
            "ram_bytes": 180 * 1024**3, "shm_bytes": 90 * 1024**3}
    assert rental_box.preflight_problems(good) == []
    assert rental_box.preflight_problems({**good, "gpus": good["gpus"] * 2})
    assert rental_box.preflight_problems({**good, "gpus": ["NVIDIA RTX PRO 6000 ..., 97887 MiB, 570.172.08, 12.0"]})
    assert rental_box.preflight_problems({**good, "ram_bytes": 63 * 1024**3})
    assert rental_box.preflight_problems({**good, "shm_bytes": 64 * 1024**2})


FAKE_KAGGLE = r'''#!/usr/bin/env python3
import json, os, sys, zipfile
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
    meta = json.loads((dest / "dataset-metadata.json").read_text())
    assert meta["id"].startswith("scottmahony/") and 6 <= len(meta["title"]) <= 50
    flag = Path(os.environ.get("FAKE_FAIL_ONCE", "/nonexistent")) / meta["id"].split("/")[1]
    if flag.exists():  # holds how many more creates of this slug fail
        left = int(flag.read_text() or "1") - 1
        flag.write_text(str(left)) if left > 0 else flag.unlink()
        print("Dataset creation error: something went wrong")  # the real CLI exits 0 here
        sys.exit(0)
    print("Your private Dataset is being created. Please check progress at https://www.kaggle.com/x")
else:
    sys.exit(3)
'''


@pytest.fixture
def box(tmp_path, monkeypatch):
    monkeypatch.setattr(rental, "RENTAL_DIR", tmp_path / "rental")
    cli = tmp_path / "kaggle"
    cli.write_text(FAKE_KAGGLE)
    cli.chmod(cli.stat().st_mode | stat.S_IEXEC)
    env = {"root": tmp_path / "kroot", "results": tmp_path / "results", "calls": tmp_path / "calls.jsonl",
           "fail": tmp_path / "fail", "executed": []}
    env["fail"].mkdir()
    monkeypatch.setenv("FAKE_KAGGLE_LOG", str(env["calls"]))
    monkeypatch.setenv("FAKE_FAIL_ONCE", str(env["fail"]))
    monkeypatch.setenv("KAGGLE_API_TOKEN", "secret-token")
    monkeypatch.setenv("CONTAINER_API_KEY", "instance-key")
    monkeypatch.setenv("RENTAL_SKIP_PREFLIGHT", "1")  # no GPU here; the gate itself is tested above
    monkeypatch.setattr(rental_box, "KAGGLE_ROOT", env["root"])
    monkeypatch.setattr(rental_box, "RESULTS", env["results"])
    monkeypatch.setattr(rental_box, "KAGGLE_CLI", str(cli))
    monkeypatch.setattr(rental_box.time, "sleep", lambda s: None)
    monkeypatch.setattr(rental_box, "cleanup_after_run", lambda before, scratch: {"fake": True})

    def fake_execute(notebook, working, log_path, timeout_s):
        env["executed"].append((notebook.name, rental_box.notebook_env()))
        (working / "benchmark.json").write_text(json.dumps({"game_runs": [], "notebook": notebook.name}))
        log_path.write_text("ran\n")
        return {"rc": 0, "timed_out": False, "seconds": 1.0, "executor": "fake"}

    monkeypatch.setattr(rental_box, "execute", fake_execute)

    def make_job(name="e2e", arms=("exp054", "exp064")):
        packed = rental.pack(name, [_built_folder(tmp_path / f"b-{name}", a) for a in arms])
        monkeypatch.setenv("RENTAL_JOB_SHA", rental.packed_sha(name))
        return packed

    env["make_job"] = make_job
    return env


def _calls(box):
    return [json.loads(line) for line in box["calls"].read_text().splitlines()] if box["calls"].exists() else []


def test_box_runner_end_to_end_with_a_fake_kaggle_cli(box):
    job = box["make_job"]()
    assert rental_box.main([str(job / "manifest.json")]) == 0
    root, results = box["root"], box["results"]
    model_dir = root / "input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
    assert (model_dir / "MODEL_MANIFEST.json").is_file()  # the wrapper folder was flattened
    assert (root / "input" / "datasets" / "keithtyser" / "duck-harness" / "file.txt").is_file()  # both layouts
    assert (root / "input" / "competitions" / "arc-prize-2026-arc-agi-3" / "arc_agi_3_wheels").is_dir()
    commands = _calls(box)
    assert len([c for c in commands if c[1] == "download" or c[:2] == ["models", "instances"]]) == 4  # inputs once
    assert len([c for c in commands if c[:2] == ["datasets", "create"]]) == 2
    for n, name in ((1, "exp054"), (2, "exp064")):
        folder = results / f"arc3-rental-e2e-{n}"
        run = json.loads((folder / "run.json").read_text())
        assert run["run"] == name and run["execution"]["rc"] == 0 and run["job_sha"] == os.environ["RENTAL_JOB_SHA"]
        assert not (folder / "output.tar.gz").exists()  # Kaggle would unpack it; the .blob name keeps it whole
        with tarfile.open(folder / rental_box.ARCHIVE) as tar:
            assert {"benchmark.json", f"arc3-{name}.log"} <= set(tar.getnames())
        assert "secret-token" not in (folder / "run.json").read_text()
        rental._unpack(folder, folder / "unpacked")  # what collect does with the download
        assert (folder / "unpacked" / "benchmark.json").is_file()
    for _, env in box["executed"]:
        assert not {"KAGGLE_API_TOKEN", "CONTAINER_API_KEY", "KAGGLE_IS_COMPETITION_RERUN"} & set(env)
    job_json = json.loads((results / "job.json").read_text())
    assert [r["results"]["uploaded"] for r in job_json] == [True, True]


def test_box_runner_refuses_a_job_that_is_not_the_packed_one(box, monkeypatch):
    job = box["make_job"]()
    monkeypatch.setenv("RENTAL_JOB_SHA", "f" * 64)
    assert rental_box.main([str(job / "manifest.json")]) == 2
    assert _calls(box) == [] and box["executed"] == []  # nothing downloaded, nothing run


def test_a_failed_upload_neither_looks_successful_nor_stops_the_next_run(box, monkeypatch):
    job = box["make_job"]()
    (box["fail"] / "arc3-rental-e2e-1").write_text("3")  # all three tries of run 1's upload fail, with exit code 0
    retried = []
    real_retry = rental_box.retry_uploads
    monkeypatch.setattr(rental_box, "retry_uploads",
                        lambda records, owner, **kw: (retried.append([r["results"]["uploaded"] for r in records]),
                                                      real_retry(records, owner, **kw)))
    assert rental_box.main([str(job / "manifest.json")]) == 0
    assert [name for name, _ in box["executed"]] == ["run1.ipynb", "run2.ipynb"]  # run 2 still ran
    assert retried == [[False, True]]
    assert [r["results"]["uploaded"] for r in json.loads((box["results"] / "job.json").read_text())] == [True, True]


def test_a_restarted_box_replays_nothing_that_was_uploaded(box):
    job = box["make_job"]()
    assert rental_box.main([str(job / "manifest.json")]) == 0
    box["executed"].clear()
    assert rental_box.main([str(job / "manifest.json")]) == 0  # restart after the end: uploads only
    assert box["executed"] == []

    partial = box["results"] / "job.partial.json"
    records = json.loads((box["results"] / "job.json").read_text())
    partial.write_text(json.dumps(records[:1]))  # a restart in the middle of run 2
    (box["results"] / "job.json").unlink()
    assert rental_box.main([str(job / "manifest.json")]) == 0
    assert [name for name, _ in box["executed"]] == ["run2.ipynb"]


def test_cleanup_kills_what_the_run_started_and_removes_its_scratch(tmp_path, monkeypatch):
    """Scoped to this test's own child: the pid list is injected, so nothing else in the container is touched."""
    child = subprocess.Popen(["sleep", "60"], start_new_session=True)  # like the kernel or vLLM: its own session
    alive = lambda: child.poll() is None  # noqa: E731
    monkeypatch.setattr(rental_box, "_pids", lambda: {child.pid} if alive() else set())
    monkeypatch.setattr(rental_box, "_ancestors", set)
    monkeypatch.setattr(rental_box, "_port_open", lambda port=1234: False)
    monkeypatch.setattr(rental_box, "_gpu_apps", list)
    site = tmp_path / "site"
    site.mkdir()
    (site / "taaf_kaggle_sources.pth").write_text("x")
    monkeypatch.setattr(rental_box, "SITE_PACKAGES", site)
    scratch = tmp_path / "scratch"
    (scratch / "old").mkdir(parents=True)
    (scratch / "qwen38-flash-next-vllm-cache").mkdir()
    (scratch / "shm-segment").write_text("x")
    out = rental_box.cleanup_after_run(set(), {scratch: {"old"}}, wait_s=1)
    child.wait(timeout=10)
    assert out["killed_pids"] == 1 and not alive()
    assert sorted(p.name for p in scratch.iterdir()) == ["old"]
    assert not (site / "taaf_kaggle_sources.pth").exists()


def test_box_runner_caps_a_silent_notebook(tmp_path, monkeypatch):
    """The wall-clock cap holds even when the executor prints nothing (a long silent cell)."""
    monkeypatch.setattr(rental_box, "has_papermill", lambda: False)
    working = tmp_path / "w"
    working.mkdir()
    fake = tmp_path / "fakepy"
    fake.write_text("#!/bin/sh\nexec sleep 60\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(rental_box.sys, "executable", str(fake))
    started = time.time()
    out = rental_box.execute(tmp_path / "nb.ipynb", working, working / "x.log", timeout_s=1.0)
    assert out["timed_out"] is True and time.time() - started < 45


def test_collect_takes_the_tree_when_kaggle_unpacked_the_archive(tmp_path):
    dl = tmp_path / "dl"
    (dl / "transcripts").mkdir(parents=True)
    (dl / "benchmark.json").write_text("{}")
    (dl / "transcripts" / "g.txt").write_text("t")
    (dl / "run.json").write_text("{}")
    dest = tmp_path / "out"
    dest.mkdir()
    rental._unpack(dl, dest)
    assert (dest / "benchmark.json").is_file() and (dest / "transcripts" / "g.txt").is_file()
    assert not (dest / "run.json").exists()
