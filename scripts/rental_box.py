#!/usr/bin/env python
"""Run built Kaggle notebooks on a rented single-GPU box, the way Kaggle runs them, and upload each run's output.

    python rental_box.py /root/job/manifest.json        # on the box (started by the boot command of scripts/rental.py)

This file runs on the rented machine only (the Kaggle GPU image, CPython 3.12, stdlib only). It is shipped inside the
job dataset that ``scripts/rental.py pack`` builds, next to ``manifest.json`` and, per run, the notebook and its
``kernel-metadata.json`` exactly as ``build_arms.py`` / ``build_kv_stress_nb.py`` wrote them (flat files
``runN.ipynb`` / ``runN.kernel-metadata.json``: a Kaggle dataset download returns sub-folders as zips).

What it does, in order:
1. **Preflight**: nvidia-smi, host RAM, CPUs, disk, /dev/shm and Python, written to ``/root/results/preflight.json``.
2. **Inputs**: every dataset, model and competition source of every run is downloaded once with the Kaggle CLI into the
   layout Kaggle mounts (``/kaggle/input/<slug>`` plus a ``/kaggle/input/datasets/<owner>/<slug>`` link,
   ``/kaggle/input/models/<owner>/<model>/<framework>/<instance>/<version>``, ``/kaggle/input/<competition>`` plus a
   ``/kaggle/input/competitions/<competition>`` link). Our notebooks and Keith's serving setup find both layouts.
3. **Runs**, one after another: ``/kaggle/working`` is emptied, the notebook is executed there with papermill (cell
   output streamed to the log, so the vast.ai log shows progress) or, without papermill, nbconvert, under a wall-clock
   cap. ``KAGGLE_IS_COMPETITION_RERUN`` is never set, so the notebooks play the public games as in a Kaggle save.
4. **Upload**: ``/kaggle/working`` is packed as ``output.tar.gz`` with ``run.json`` (exit code, times, preflight) and
   created as a new private Kaggle dataset ``<owner>/<results_prefix>-<n>``. A failed run is uploaded too.
5. **Exit**: the container's command ends, the instance stops and its GPU is released (``scripts/rental.py`` destroys
   it after collecting).

The Kaggle credential arrives as ``KAGGLE_API_TOKEN`` in the container environment and is never written to disk,
printed or uploaded (the environment of each notebook run drops it).
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import Any

KAGGLE_ROOT = Path(os.environ.get("RENTAL_KAGGLE_ROOT", "/kaggle"))  # tests point this at a temporary directory
RESULTS = Path(os.environ.get("RENTAL_RESULTS_DIR", "/root/results"))
KAGGLE_CLI = os.environ.get("RENTAL_KAGGLE_CLI", "kaggle")  # the boot command installs it apart from the notebook's Python
SECRET_ENV = ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "KAGGLE_USERNAME", "VAST_API_KEY")


def log(msg: str) -> None:
    print(f"[rental-box {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=False, text=True, capture_output=True, **kw)


def kaggle(*args: str, retries: int = 3) -> str:
    """One Kaggle CLI call with retries; raises with the CLI's last lines on failure."""
    last = ""
    for attempt in range(1, retries + 1):
        r = run([KAGGLE_CLI, *args])
        if r.returncode == 0:
            return r.stdout
        last = (r.stdout + r.stderr).strip()[-800:]
        log(f"kaggle {args[0]} {args[1] if len(args) > 1 else ''} failed (attempt {attempt}): {last}")
        time.sleep(10 * attempt)
    raise RuntimeError(f"kaggle {' '.join(args[:3])} failed: {last}")


# --- inputs -------------------------------------------------------------------------------------------------------


def input_paths(kind: str, ref: str, root: Path | None = None) -> tuple[Path, list[Path]]:
    """Where Kaggle mounts a source, and the alternative paths linked to it (both layouts Kaggle uses)."""
    base = (root or KAGGLE_ROOT) / "input"
    if kind == "dataset":
        owner, slug = ref.split("/", 1)
        return base / slug, [base / "datasets" / owner / slug]
    if kind == "model":
        owner, model, framework, instance, version = ref.split("/")
        return base / "models" / owner / model / framework.lower() / instance / version, []
    if kind == "competition":
        return base / ref, [base / "competitions" / ref]
    raise ValueError(f"unknown source kind {kind!r}")


def sources_of(meta: dict) -> list[tuple[str, str]]:
    out = [("dataset", r) for r in meta.get("dataset_sources", [])]
    out += [("model", r) for r in meta.get("model_sources", [])]
    out += [("competition", r) for r in meta.get("competition_sources", [])]
    if meta.get("kernel_sources"):
        raise RuntimeError(f"kernel sources are not supported on the rental box: {meta['kernel_sources']}")
    return out


def _flatten_single_subdir(dest: Path) -> None:
    """A download that extracted into one wrapper folder is moved up, so files sit where Kaggle mounts them."""
    entries = [p for p in dest.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir() and not any(dest.glob("*.json")):
        inner = entries[0]
        for child in inner.iterdir():
            shutil.move(str(child), str(dest / child.name))
        inner.rmdir()


def fetch(kind: str, ref: str, root: Path | None = None) -> dict:
    dest, links = input_paths(kind, ref, root)
    done = dest / ".rental_ready"
    started = time.time()
    if not done.exists():
        if dest.exists():
            shutil.rmtree(dest)  # a half-finished earlier attempt
        dest.mkdir(parents=True)
        if kind == "dataset":
            kaggle("datasets", "download", ref, "-p", str(dest), "--unzip")
        elif kind == "model":
            kaggle("models", "instances", "versions", "download", ref, "-p", str(dest), "--untar")
            _flatten_single_subdir(dest)
        else:
            kaggle("competitions", "download", ref, "-p", str(dest))
            for archive in dest.glob("*.zip"):
                shutil.unpack_archive(str(archive), str(dest))
                archive.unlink()
        done.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    for link in links:
        if not link.exists():
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(dest, target_is_directory=True)
    size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file() and not p.is_symlink())
    return {"kind": kind, "ref": ref, "path": str(dest), "bytes": size, "seconds": round(time.time() - started, 1)}


# --- preflight ----------------------------------------------------------------------------------------------------


def preflight() -> dict:
    def out(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60).stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"{type(exc).__name__}: {exc}"

    meminfo = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            meminfo[key] = rest.strip()
    except OSError:
        pass
    shm = shutil.disk_usage("/dev/shm") if Path("/dev/shm").exists() else None
    return {
        "gpu": out(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"]),
        "mem_total": meminfo.get("MemTotal"), "mem_available": meminfo.get("MemAvailable"),
        "cpus": os.cpu_count(), "python": sys.version.split()[0], "platform": platform.platform(),
        "shm_free_bytes": shm.free if shm else None,
        "disk_free_bytes": shutil.disk_usage("/").free,
        "cpu_model": next((ln.split(":", 1)[1].strip() for ln in out(["cat", "/proc/cpuinfo"]).splitlines()
                           if ln.startswith("model name")), None),
    }


# --- one run ------------------------------------------------------------------------------------------------------


def notebook_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k not in SECRET_ENV}
    env.update({"KAGGLE_KERNEL_RUN_TYPE": "Batch", "PYTHONUNBUFFERED": "1"})
    env.pop("KAGGLE_IS_COMPETITION_RERUN", None)
    return env


def has_papermill() -> bool:
    return subprocess.run([sys.executable, "-c", "import papermill"], capture_output=True, check=False).returncode == 0


def execute(notebook: Path, working: Path, log_path: Path, timeout_s: float) -> dict:
    executed = working / "__notebook__.ipynb"
    if has_papermill():
        cmd = [sys.executable, "-m", "papermill", "--log-output", "--no-progress-bar", "--request-save-on-cell-execute",
               "-k", "python3", str(notebook), str(executed)]
    else:
        cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
               "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=python3",
               "--output", str(executed), str(notebook)]
    started = time.time()
    log(f"executing {notebook.name} (cap {timeout_s:.0f} s): {' '.join(cmd[2:4])} ...")
    with open(log_path, "w", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, cwd=str(working), env=notebook_env(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, start_new_session=True)

        def pump() -> None:  # mirrored to the container log, so `rental.py logs` shows progress
            assert proc.stdout is not None
            for line in proc.stdout:
                handle.write(line)
                handle.flush()
                sys.stdout.write(line)
                sys.stdout.flush()

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        timed_out = False
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:  # the cap holds even when a cell prints nothing for hours
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            rc = proc.wait()
        reader.join(timeout=30)
    return {"rc": rc, "timed_out": timed_out, "seconds": round(time.time() - started, 1),
            "executor": "papermill" if "papermill" in cmd else "nbconvert"}


def pack_and_upload(owner: str, dataset_slug: str, title: str, working: Path, record: dict, upload: bool) -> dict:
    folder = RESULTS / dataset_slug
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    with tarfile.open(folder / "output.tar.gz", "w:gz") as tar:
        for path in sorted(working.iterdir()):
            tar.add(str(path), arcname=path.name)
    (folder / "run.json").write_text(json.dumps(record, indent=1, default=str))
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": title[:50], "id": f"{owner}/{dataset_slug}", "licenses": [{"name": "CC0-1.0"}]}, indent=1))
    info = {"dataset": f"{owner}/{dataset_slug}", "bytes": (folder / "output.tar.gz").stat().st_size}
    if upload:
        kaggle("datasets", "create", "-p", str(folder), "--dir-mode", "skip")  # private unless --public is passed
        info["uploaded"] = True
    return info


def run_one(index: int, spec: dict, manifest: dict, job_dir: Path, pre: dict) -> dict:
    working = KAGGLE_ROOT / "working"
    if working.exists():
        shutil.rmtree(working)
    working.mkdir(parents=True)
    notebook = job_dir / spec["notebook"]
    record: dict[str, Any] = {"job": manifest["job"], "run": spec["name"], "slug": spec["slug"], "index": index,
                              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "preflight": pre}
    try:
        record["execution"] = execute(notebook, working, working / f"{spec['slug']}.log",
                                      float(spec.get("timeout_s", manifest.get("timeout_s", 5 * 3600))))
    except Exception as exc:  # a crash of the executor itself: still upload what the run wrote
        record["execution"] = {"rc": None, "error": f"{type(exc).__name__}: {exc}"}
    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    slug = f"{manifest['results_prefix']}-{index}"
    record["results"] = pack_and_upload(manifest["owner"], slug, f"{manifest['results_prefix']} {spec['name']}",
                                        working, record, upload=manifest.get("upload", True))
    log(f"RENTAL_RUN_DONE {json.dumps({'run': spec['name'], **record['execution'], **record['results']})}")
    return record


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    manifest_path = Path(args[0] if args else "/root/job/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    job_dir = manifest_path.parent
    RESULTS.mkdir(parents=True, exist_ok=True)
    pre = preflight()
    (RESULTS / "preflight.json").write_text(json.dumps(pre, indent=1))
    log(f"RENTAL_PREFLIGHT {json.dumps(pre)}")
    wanted: list[tuple[str, str]] = []
    for spec in manifest["runs"]:
        meta = json.loads((job_dir / spec["metadata"]).read_text())
        wanted += [s for s in sources_of(meta) if s not in wanted]
    fetched = []
    for kind, ref in wanted:
        fetched.append(fetch(kind, ref))
        log(f"RENTAL_INPUT {json.dumps(fetched[-1])}")
    pre["inputs"] = fetched
    records = [run_one(i, spec, manifest, job_dir, pre) for i, spec in enumerate(manifest["runs"], start=1)]
    (RESULTS / "job.json").write_text(json.dumps(records, indent=1, default=str))
    log("RENTAL_JOB_DONE " + json.dumps([{"run": r["run"], "rc": r["execution"].get("rc"),
                                          "dataset": r["results"]["dataset"]} for r in records]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
