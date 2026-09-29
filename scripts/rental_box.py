#!/usr/bin/env python
"""Run built Kaggle notebooks on a rented single-GPU box, the way Kaggle runs them, and upload each run's output.

    python rental_box.py /root/job/manifest.json        # on the box (started by the boot command of scripts/rental.py)
    python rental_box.py --stop-self                    # the boot command's exit trap: stop this vast.ai instance

This file runs on the rented machine only (the Kaggle GPU image, CPython 3.12, stdlib only). It is shipped inside the
job dataset that ``scripts/rental.py pack`` builds, next to ``manifest.json`` and, per run, the notebook and its
``kernel-metadata.json`` exactly as ``build_arms.py`` / ``build_kv_stress_nb.py`` wrote them (flat files
``runN.ipynb`` / ``runN.kernel-metadata.json``: a Kaggle dataset download returns sub-folders as zips).

What it does, in order:
0. **Job check**: the sha256 of the job files must equal ``RENTAL_JOB_SHA`` (set by ``rental.py launch`` from the
   local pack), so a stale job dataset never runs under a new name. If ``/root/results/job.json`` exists the box was
   restarted after finishing: it only retries uploads that failed, then exits.
1. **Preflight gate**: one RTX PRO 6000, driver >= 580, RAM (the smaller of /proc/meminfo and the cgroup limit)
   >= 150 GiB, /dev/shm >= 8 GiB, else it stops before the hour of downloads.
2. **Inputs**: every dataset, model and competition source of every run is downloaded once with the Kaggle CLI into the
   layout Kaggle mounts (``/kaggle/input/<slug>`` plus a ``/kaggle/input/datasets/<owner>/<slug>`` link,
   ``/kaggle/input/models/<owner>/<model>/<framework>/<instance>/<version>``, ``/kaggle/input/<competition>`` plus a
   ``/kaggle/input/competitions/<competition>`` link). Our notebooks and Keith's serving setup find both layouts.
3. **Runs**, one after another: ``/kaggle/working`` is emptied, the notebook is executed there with papermill (cell
   output streamed to the log) or, without papermill, nbconvert, under a wall-clock cap. Afterwards every process
   the run started is killed (the kernel and vLLM run in their own sessions), the GPU and port 1234 must be free, and
   what the run left in /tmp, /dev/shm and site-packages is removed, so each run starts cold, as on Kaggle.
   ``KAGGLE_IS_COMPETITION_RERUN`` is never set, so the notebooks play the public games as in a Kaggle save.
4. **Upload**: ``/kaggle/working`` is packed as ``output.tar.gz.blob`` (Kaggle unpacks ``.tar.gz`` uploads; ``.blob``
   stays whole) with ``run.json`` and created as the private dataset ``<owner>/<results_prefix>-<n>``. A failed run
   is uploaded too; a failed upload is retried at the end and never stops the next run.
5. **Exit**: the boot command's trap stops the instance through vast.ai's API (``CONTAINER_ID`` /
   ``CONTAINER_API_KEY``, set in every instance), so the GPU is released; ``rental.py collect --destroy`` deletes it
   once every result is downloaded.

The Kaggle credential arrives as ``KAGGLE_API_TOKEN`` in the container environment and is never written to disk,
printed or uploaded (the environment of each notebook run drops it and the instance key).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import sysconfig
import tarfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

KAGGLE_ROOT = Path(os.environ.get("RENTAL_KAGGLE_ROOT", "/kaggle"))  # tests point this at a temporary directory
RESULTS = Path(os.environ.get("RENTAL_RESULTS_DIR", "/root/results"))
KAGGLE_CLI = os.environ.get("RENTAL_KAGGLE_CLI", "kaggle")  # the boot command installs it apart from the notebook's Python
SECRET_ENV = ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "KAGGLE_USERNAME", "VAST_API_KEY", "CONTAINER_API_KEY")
ARCHIVE = "output.tar.gz.blob"  # Kaggle extracts uploaded archives; this name keeps the tarball whole
SCRATCH_ROOTS = (Path("/tmp"), Path("/dev/shm"))  # what a run leaves here is removed before the next run
SERVER_PORT = 1234
MIN_RAM_GIB = 150  # Keith's setup: >= 64 GiB free beside the 48 GiB pinned PLE table; Kaggle's box has ~180 GB
MIN_SHM_GIB = 8
MIN_DRIVER = 580  # the CUDA 13.0 runtime
SITE_PACKAGES = Path(sysconfig.get_paths()["purelib"])  # the notebook's cell 10 writes taaf_kaggle_sources.pth here
DOWNLOAD_TIMEOUT_S = 3 * 3600
UPLOAD_TIMEOUT_S = 3600


def log(msg: str) -> None:
    print(f"[rental-box {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def kaggle(*args: str, retries: int = 3, timeout: float = DOWNLOAD_TIMEOUT_S, expect: str | None = None) -> str:
    """One Kaggle CLI call with retries and a timeout. ``expect``: text the CLI prints on success (it exits 0 on
    some failures, e.g. 'Dataset creation error: ...')."""
    last = ""
    for attempt in range(1, retries + 1):
        log(f"$ kaggle {' '.join(args)}")
        try:
            r = subprocess.run([KAGGLE_CLI, *args], check=False, text=True, capture_output=True, timeout=timeout)
            out = (r.stdout + r.stderr).strip()
            if r.returncode == 0 and (expect is None or expect in out):
                return out
            last = out[-800:]
        except subprocess.TimeoutExpired:
            last = f"timed out after {timeout:.0f} s"
        log(f"kaggle {args[0]} {args[1] if len(args) > 1 else ''} failed (attempt {attempt}): {last}")
        time.sleep(10 * attempt)
    raise RuntimeError(f"kaggle {' '.join(args[:3])} failed: {last}")


# --- job check ----------------------------------------------------------------------------------------------------


def job_sha(job_dir: Path) -> str:
    """sha256 over the job's files (names and contents), as rental.py pack computes it."""
    h = hashlib.sha256()
    for path in sorted(p for p in job_dir.iterdir() if p.is_file() and p.name != "dataset-metadata.json"):
        h.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


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


MODEL_MARKERS = ("config.json", "MODEL_MANIFEST.json")


def _finish_model(dest: Path) -> None:
    """Extract a model tarball the CLI left behind (its retry skips extraction of a complete download), then move the
    files out of a single wrapper folder when the top level has no model files but that folder does."""
    for archive in sorted(dest.glob("*.tar.gz")):
        with tarfile.open(archive, mode="r:gz") as tar:
            tar.extractall(dest, filter="data")
        archive.unlink()
    entries = [p for p in dest.iterdir() if not p.name.startswith(".")]
    if any((dest / m).exists() for m in MODEL_MARKERS) or len(entries) != 1 or not entries[0].is_dir():
        return
    if not any((entries[0] / m).exists() for m in MODEL_MARKERS):
        return
    inner = entries[0].rename(dest / f".wrapper-{os.getpid()}")  # a child may share the wrapper's name
    for child in inner.iterdir():
        child.rename(dest / child.name)
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
            _finish_model(dest)
            if not any((dest / m).exists() for m in MODEL_MARKERS):
                raise RuntimeError(f"model {ref} has no {MODEL_MARKERS} at {dest}: {sorted(os.listdir(dest))[:10]}")
        else:
            kaggle("competitions", "download", ref, "-p", str(dest))
            for archive in dest.glob("*.zip"):
                shutil.unpack_archive(str(archive), str(dest))
                archive.unlink()
        if not any(p for p in dest.iterdir() if p.name != ".rental_ready"):
            raise RuntimeError(f"{kind} {ref} downloaded nothing into {dest}")
        done.write_text(utc())
    for link in links:
        if not link.exists():
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(dest, target_is_directory=True)
    size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file() and not p.is_symlink())
    return {"kind": kind, "ref": ref, "path": str(dest), "bytes": size, "seconds": round(time.time() - started, 1)}


# --- preflight ----------------------------------------------------------------------------------------------------


def _out(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"{type(exc).__name__}: {exc}"


def cgroup_memory_limit() -> int | None:
    for path in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            raw = path.read_text().strip()
        except OSError:
            continue
        if raw.isdigit() and int(raw) < 1 << 60:
            return int(raw)
    return None


def preflight() -> dict:
    meminfo = {}
    with contextlib.suppress(OSError):
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            meminfo[key] = int(rest.split()[0]) * 1024 if rest.split() else 0
    limit = cgroup_memory_limit()
    ram = min(x for x in (meminfo.get("MemTotal"), limit) if x) if (meminfo.get("MemTotal") or limit) else 0
    shm = shutil.disk_usage("/dev/shm") if Path("/dev/shm").exists() else None
    gpus = [g for g in _out(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                             "--format=csv,noheader"]).splitlines() if g.strip()]
    return {
        "gpus": gpus, "ram_bytes": ram, "mem_total_bytes": meminfo.get("MemTotal"), "cgroup_limit_bytes": limit,
        "cpus": os.cpu_count(), "python": sys.version.split()[0], "platform": platform.platform(),
        "shm_bytes": shm.total if shm else None, "disk_free_bytes": shutil.disk_usage("/").free,
        "cpu_model": next((ln.split(":", 1)[1].strip() for ln in _out(["cat", "/proc/cpuinfo"]).splitlines()
                           if ln.startswith("model name")), None),
    }


def preflight_problems(pre: dict) -> list[str]:
    problems = []
    gpus = pre.get("gpus") or []
    if len(gpus) != 1 or "rtx pro 6000" not in gpus[0].lower():
        problems.append(f"need exactly one RTX PRO 6000, nvidia-smi shows {gpus}")
    else:
        try:
            driver = int(gpus[0].split(",")[2].strip().split(".")[0])
        except (IndexError, ValueError):
            driver = 0
        if driver < MIN_DRIVER:
            problems.append(f"driver {gpus[0]} is older than {MIN_DRIVER} (CUDA 13.0)")
    if (pre.get("ram_bytes") or 0) < MIN_RAM_GIB * 1024**3:
        problems.append(f"RAM {(pre.get('ram_bytes') or 0) / 1024**3:.0f} GiB < {MIN_RAM_GIB} GiB")
    if (pre.get("shm_bytes") or 0) < MIN_SHM_GIB * 1024**3:
        problems.append(f"/dev/shm {(pre.get('shm_bytes') or 0) / 1024**3:.1f} GiB < {MIN_SHM_GIB} GiB")
    return problems


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
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
            rc = proc.wait()
        reader.join(timeout=5)  # an orphan holding the pipe is killed by cleanup_after_run, not waited for
    return {"rc": rc, "timed_out": timed_out, "seconds": round(time.time() - started, 1),
            "executor": "papermill" if "papermill" in cmd else "nbconvert"}


def _pids() -> set[int]:
    return {int(p.name) for p in Path("/proc").iterdir() if p.name.isdigit()}


def _ancestors() -> set[int]:
    out, pid = set(), os.getpid()
    while pid > 1:
        out.add(pid)
        try:
            stat = (Path("/proc") / str(pid) / "stat").read_text()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
    return out


def _port_open(port: int = SERVER_PORT) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _gpu_apps() -> list[str]:
    raw = _out(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"])
    return [ln for ln in raw.splitlines() if ln.strip() and "Error" not in ln and "No running" not in ln]


def cleanup_after_run(before_pids: set[int], scratch_before: dict[Path, set[str]], wait_s: float = 120.0) -> dict:
    """Kill every process the run started (the Jupyter kernel and Keith's vLLM server run in their own sessions, so
    killing papermill's group misses them), wait for the GPU and port 1234 to be free, and remove what the run left in
    /tmp, /dev/shm and site-packages, so the next run starts cold as on Kaggle."""
    keep = before_pids | _ancestors()
    killed = sorted(_pids() - keep)
    if os.environ.get("RENTAL_DRY_CLEANUP"):  # local simulations: report, never kill or delete outside a rented box
        return {"dry_run": True, "would_kill": len(killed)}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in sorted(_pids() - keep):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, sig)
        end = time.time() + (20 if sig == signal.SIGTERM else 10)
        while time.time() < end and (_pids() - keep):
            time.sleep(0.5)
    end = time.time() + wait_s
    while time.time() < end and (_port_open() or _gpu_apps()):
        time.sleep(2)
    removed = []
    for root, names in scratch_before.items():
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if path.name not in names:
                with contextlib.suppress(OSError):
                    if path.is_dir() and not path.is_symlink():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                removed.append(str(path))
    pth = SITE_PACKAGES / "taaf_kaggle_sources.pth"
    if pth.exists():
        pth.unlink()
        removed.append(str(pth))
    return {"killed_pids": len(killed), "port_open": _port_open(), "gpu_apps": _gpu_apps(), "removed": removed}


def upload(owner: str, folder: Path) -> dict:
    slug = json.loads((folder / "dataset-metadata.json").read_text())["id"].split("/", 1)[1]
    try:
        out = kaggle("datasets", "create", "-p", str(folder), "--dir-mode", "skip", timeout=UPLOAD_TIMEOUT_S,
                     expect="Your private Dataset is being created")  # private: --public is never passed
        return {"dataset": f"{owner}/{slug}", "uploaded": True, "cli": out[-200:]}
    except RuntimeError as exc:
        return {"dataset": f"{owner}/{slug}", "uploaded": False, "error": str(exc)[-500:]}


def pack(owner: str, dataset_slug: str, title: str, working: Path, record: dict) -> Path:
    folder = RESULTS / dataset_slug
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    with tarfile.open(folder / ARCHIVE, "w:gz") as tar:
        for path in sorted(working.iterdir()):
            tar.add(str(path), arcname=path.name)
    (folder / "run.json").write_text(json.dumps(record, indent=1, default=str))
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": title[:50].ljust(6, "-"), "id": f"{owner}/{dataset_slug}", "licenses": [{"name": "CC0-1.0"}]},
        indent=1))
    return folder


def run_one(index: int, spec: dict, manifest: dict, job_dir: Path, pre: dict,
            scratch_before: dict[Path, set[str]]) -> dict:
    working = KAGGLE_ROOT / "working"
    if working.exists():
        shutil.rmtree(working)
    working.mkdir(parents=True)
    before = _pids()
    record: dict[str, Any] = {"job": manifest["job"], "run": spec["name"], "slug": spec["slug"], "index": index,
                              "job_sha": manifest.get("job_sha"), "started_utc": utc(), "preflight": pre}
    try:
        record["execution"] = execute(job_dir / spec["notebook"], working, working / f"{spec['slug']}.log",
                                      float(spec.get("timeout_s", manifest.get("timeout_s", 5 * 3600))))
    except Exception as exc:  # a crash of the executor itself: still upload what the run wrote
        record["execution"] = {"rc": None, "error": f"{type(exc).__name__}: {exc}"}
    record["cleanup"] = cleanup_after_run(before, scratch_before)
    record["finished_utc"] = utc()
    slug = f"{manifest['results_prefix']}-{index}"
    folder = pack(manifest["owner"], slug, f"{manifest['results_prefix']} {index} {spec['name']}", working, record)
    record["results"] = {"folder": str(folder), **upload(manifest["owner"], folder)}
    log(f"RENTAL_RUN_DONE {json.dumps({'run': spec['name'], **record['execution'], **record['results']})}")
    return record


def retry_uploads(records: list[dict], owner: str, rounds: int = 3, wait_s: float = 300.0) -> None:
    for attempt in range(rounds):
        pending = [r for r in records if not r["results"].get("uploaded")]
        if not pending:
            return
        if attempt:
            time.sleep(wait_s)
        for r in pending:
            r["results"].update(upload(owner, Path(r["results"]["folder"])))


# --- vast.ai self-stop --------------------------------------------------------------------------------------------


def stop_self() -> int:
    """Stop this instance through vast.ai's API (instance-scoped key vast.ai sets in every container): the GPU is
    released and the disk kept, so results that failed to upload survive for `rental.py` to retry."""
    cid, key = os.environ.get("CONTAINER_ID"), os.environ.get("CONTAINER_API_KEY")
    if not cid or not key:
        log("RENTAL_STOP_SELF skipped: CONTAINER_ID / CONTAINER_API_KEY not set")
        return 1
    req = urllib.request.Request(f"https://console.vast.ai/api/v0/instances/{cid}/", method="PUT",
                                 data=json.dumps({"state": "stopped"}).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            log(f"RENTAL_STOP_SELF {response.status}")
            return 0
    except OSError as exc:
        log(f"RENTAL_STOP_SELF failed: {exc}")
        return 1


# --- main ---------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args[:1] == ["--stop-self"]:
        return stop_self()
    manifest_path = Path(args[0] if args else "/root/job/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    job_dir = manifest_path.parent
    RESULTS.mkdir(parents=True, exist_ok=True)
    job_json = RESULTS / "job.json"
    if job_json.exists():  # restarted after finishing: only uploads that failed are retried
        records = json.loads(job_json.read_text())
        retry_uploads(records, manifest["owner"], rounds=2, wait_s=60)
        job_json.write_text(json.dumps(records, indent=1, default=str))
        log("RENTAL_JOB_DONE (restart) " + json.dumps([r["results"] for r in records]))
        return 0 if all(r["results"].get("uploaded") for r in records) else 4
    want_sha = os.environ.get("RENTAL_JOB_SHA", "")
    have_sha = job_sha(job_dir)
    if want_sha != have_sha:
        log(f"RENTAL_JOB_MISMATCH the job dataset is not the packed job (want {want_sha[:12]}, have {have_sha[:12]})")
        return 2
    manifest["job_sha"] = have_sha
    pre = preflight()
    (RESULTS / "preflight.json").write_text(json.dumps(pre, indent=1))
    log(f"RENTAL_PREFLIGHT {json.dumps(pre)}")
    problems = preflight_problems(pre)
    if problems and not os.environ.get("RENTAL_SKIP_PREFLIGHT"):
        log(f"RENTAL_PREFLIGHT_FAIL {json.dumps(problems)}")
        return 3
    wanted: list[tuple[str, str]] = []
    for spec in manifest["runs"]:
        meta = json.loads((job_dir / spec["metadata"]).read_text())
        wanted += [s for s in sources_of(meta) if s not in wanted]
    fetched = []
    for kind, ref in wanted:
        fetched.append(fetch(kind, ref))
        log(f"RENTAL_INPUT {json.dumps(fetched[-1])}")
    pre["inputs"] = fetched
    scratch_before = {root: {p.name for p in root.iterdir()} if root.is_dir() else set() for root in SCRATCH_ROOTS}
    partial = job_json.with_suffix(".partial.json")
    done = {r["index"]: r for r in (json.loads(partial.read_text()) if partial.exists() else [])
            if r.get("results", {}).get("uploaded")}  # a restart mid-job does not replay uploaded runs
    records = []
    for i, spec in enumerate(manifest["runs"], start=1):
        records.append(done.get(i) or run_one(i, spec, manifest, job_dir, pre, scratch_before))
        partial.write_text(json.dumps(records, indent=1, default=str))
    retry_uploads(records, manifest["owner"])
    job_json.write_text(json.dumps(records, indent=1, default=str))
    ok = all(r["results"].get("uploaded") for r in records)
    log("RENTAL_JOB_DONE " + json.dumps([{"run": r["run"], "rc": r["execution"].get("rc"),
                                          "dataset": r["results"]["dataset"], "uploaded": r["results"].get("uploaded")}
                                         for r in records]))
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
