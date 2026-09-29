#!/usr/bin/env python
"""Run our built Kaggle notebooks on rented vast.ai GPUs and bring the output back like a Kaggle run.

    .venv/bin/python scripts/rental.py offers                                  # read-only, no account needed
    .venv/bin/python scripts/rental.py pack JOB DIR [DIR ...] [--upload]         # built notebook folders -> a job
    .venv/bin/python scripts/rental.py quote JOB --offer ID                      # what the job will cost on that offer
    .venv/bin/python scripts/rental.py launch JOB --offer ID --approved "<the owner's OK for this batch>"
    .venv/bin/python scripts/rental.py status JOB | logs JOB [--tail N] | destroy JOB
    .venv/bin/python scripts/rental.py collect JOB [--destroy]                  # results -> runs/<name>/kernel-output

One job = the runs one box plays one after another (a folder may be listed twice for two runs of one arm). A box is
one RTX PRO 6000: Keith's serving setup requires exactly one such GPU in nvidia-smi, so two notebooks cannot share a
2-GPU box unmodified; two 1-GPU boxes cost the same.

How a job travels (docs/research/rental-runner.md): ``pack`` copies each built notebook and its kernel-metadata.json
plus scripts/rental_box.py and a manifest into ``runs/rental/<JOB>/job/`` and records the job's sha256;
``--upload`` creates (or versions) the private Kaggle dataset ``<owner>/arc3-rental-<JOB>``. ``launch`` checks that
the uploaded job is byte-identical to the local pack, then rents the offer with the Kaggle GPU image (pinned tag), a
500 GB disk and a boot command as the container's command (bash entrypoint): install the Kaggle CLI apart from the
image's Python, download the job, run rental_box.py under a wall-clock cap, and on exit stop the instance through
vast.ai's API. rental_box.py downloads the notebooks' inputs into the /kaggle/input layout, runs each notebook and
uploads each run's /kaggle/working as the private dataset ``<owner>/arc3-rental-<JOB>-<n>``. ``collect`` downloads and
scores the results (pull_taaf_run.py --local); ``--destroy`` then deletes the instance, whose disk is billed until
deleted.

Money: ``launch`` is the only call that rents. It refuses without ``--approved`` (the owner's words for this batch,
kept in ``runs/rental/<JOB>/launch.json``); CLAUDE.md: stop and ask before spending money. Prices shown include the
500 GB disk, which some hosts charge heavily for. Secrets: the vast.ai key is read from ``.vast/api_key``
(git-ignored) and used only here; the Kaggle token (``.kaggle/access_token``) is passed to the instance as
KAGGLE_API_TOKEN, so vast.ai and the host can read it: rotate it on kaggle.com after ``collect``, not before (the
box needs it until its last upload).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from rental_box import ARCHIVE, job_sha  # noqa: E402

RENTAL_DIR = ROOT / "runs" / "rental"
VAST = "https://console.vast.ai/api/v0"
VAST_KEY = ROOT / ".vast" / "api_key"
KAGGLE_TOKEN = ROOT / ".kaggle" / "access_token"
OWNER = "scottmahony"
IMAGE = "gcr.io/kaggle-gpu-images/python:v170"  # Kaggle's GPU image, the tag `latest` pointed at on 2026-09-29
DISK_GB = 500  # image ~60 GB unpacked + the model (111 GB archive, 135 GB extracted, both at once) + runtimes, wheels
KAGGLE_CLI_VERSION = "2.2.4"  # the version this repo uses
HOURS_PER_RUN = 2.75  # public-25 notebook: ~16 min serving setup + 132 min play + teardown (exp-054's logs)
HOURS_SETUP = 1.0  # image pull + ~130 GB of inputs on a fresh box (estimate until the first rental measures it)
CAP_SLACK_HOURS = 1.5  # the boot command's wall-clock cap = estimate + this
MIN_RAM_SHARE_GB = 170  # the instance's share of host RAM (cpu_ram x gpu_frac); Kaggle's box has ~180 GB
MIN_CONTRACT_HOURS = 48

# What a box must have to run the notebook like Kaggle's g4-standard-48 (1 RTX PRO 6000, 48 vCPU, ~180 GB RAM):
# Keith's setup needs >= 64 GiB free RAM beside the 48 GiB pinned PLE table, CUDA 13.0 needs driver >= 580 (checked
# client-side with the RAM share and the contract length, which the search API cannot express).
OFFER_FILTER = {
    "rentable": {"eq": True},
    "verified": {"eq": True},
    "num_gpus": {"eq": 1},
    "gpu_name": {"in": ["RTX PRO 6000 S", "RTX PRO 6000 WS"]},
    "cpu_ram": {"gte": MIN_RAM_SHARE_GB * 1024},
    "cpu_cores_effective": {"gte": 32},
    "disk_space": {"gte": DISK_GB},
    "reliability": {"gte": 0.97},
    "inet_down": {"gte": 500},
}


def log(msg: str) -> None:
    print(msg, flush=True)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --- vast.ai --------------------------------------------------------------------------------------------------------


def vast(method: str, path: str, body: dict | None = None, key: str | None = None, timeout: float = 60) -> Any:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"{VAST}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode() or "null")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"vast.ai {method} {path}: HTTP {exc.code} {exc.read().decode(errors='replace')[:500]}") \
            from exc


def vast_key() -> str:
    if not VAST_KEY.is_file():
        raise SystemExit(f"no vast.ai API key: put it in {VAST_KEY.relative_to(ROOT)} (git-ignored)")
    return VAST_KEY.read_text().strip()


def driver_major(offer: dict) -> int:
    try:
        return int(str(offer.get("driver_version") or "0").split(".")[0])
    except ValueError:
        return 0


def ram_share_gb(offer: dict) -> float:
    """The instance's RAM: vast.ai reports the host's cpu_ram and gives a rental its gpu_frac share of it."""
    return float(offer.get("cpu_ram") or 0) * float(offer.get("gpu_frac") or 1.0) / 1024


def disk_dph(offer: dict) -> float:
    return float(offer.get("storage_cost") or 0) * DISK_GB / (30 * 24)  # storage_cost is $/GB/month


def run_dph(offer: dict) -> float:
    """$/h while running with our disk (dph_total carries only the default ~8 GB of storage unless the search passed
    allocated_storage; either way this is GPU + CPU + our 500 GB)."""
    return float(offer.get("dph_total") or 0) - float(offer.get("storage_total_cost") or 0) + disk_dph(offer)


def good_offers(offers: list[dict]) -> list[dict]:
    """What the search API cannot express: driver >= 580, the RAM share, the contract length. Ranked by the real
    price with our disk; the server edition (Kaggle's card; the workstation card is the same GB202 chip) wins ties
    up to 10%."""
    keep = [o for o in offers if driver_major(o) >= 580 and o.get("num_gpus") == 1
            and ram_share_gb(o) >= MIN_RAM_SHARE_GB and float(o.get("duration") or 0) >= MIN_CONTRACT_HOURS * 3600]
    return sorted(keep, key=lambda o: run_dph(o) * (0.9 if o.get("gpu_name") == "RTX PRO 6000 S" else 1.0))


def search(extra: dict | None = None, limit: int = 100) -> list[dict]:
    body = {"limit": limit, "type": "ondemand", "order": [["dph_total", "asc"]], "allocated_storage": DISK_GB,
            **OFFER_FILTER, **(extra or {})}
    return good_offers((vast("POST", "/bundles/", body) or {}).get("offers", []))


def find_offer(offer_id: int) -> dict | None:
    """One offer by id through the public search (no key needed), or None if it is gone or fails the filters."""
    return next(iter(search({"id": {"eq": int(offer_id)}}, limit=5)), None)


def offer_line(o: dict) -> str:
    return (f"{o['id']:>10}  {o.get('gpu_name', '?'):<16} ${run_dph(o):.2f}/h with {DISK_GB} GB disk "
            f"(disk ${disk_dph(o) * 24:.2f}/day, billed until destroyed)  RAM {ram_share_gb(o):.0f} GB  "
            f"{float(o.get('cpu_cores_effective') or 0):.0f} cores  down {round(o.get('inet_down') or 0)} Mb/s  "
            f"driver {o.get('driver_version')}  rel {float(o.get('reliability2') or o.get('reliability') or 0):.3f}  "
            f"contract {float(o.get('duration') or 0) / 86400:.0f} d  {o.get('geolocation', '')}")


# --- jobs -----------------------------------------------------------------------------------------------------------


def job_dir(job: str) -> Path:
    return RENTAL_DIR / job


def dataset_slug(job: str) -> str:
    return f"arc3-rental-{job}"


def check_job_name(job: str) -> None:
    if not (3 <= len(job) <= 24 and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", job)):
        raise SystemExit("job names: 3-24 characters, lower-case letters and digits, single dashes between them "
                         "(they become Kaggle dataset slugs)")


def pack(job: str, folders: list[Path], owner: str = OWNER, timeout_s: float = 5 * 3600) -> Path:
    """Copy built notebook folders into runs/rental/<job>/job/ as flat files with a manifest and the box runner."""
    check_job_name(job)
    out = job_dir(job) / "job"
    if (job_dir(job) / "launch.json").exists():
        raise SystemExit(f"{job} was launched; pack a new job name instead of changing a launched one")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    runs = []
    for n, folder in enumerate(folders, start=1):
        meta = json.loads((folder / "kernel-metadata.json").read_text())
        if meta.get("enable_internet"):
            raise SystemExit(f"{folder}: internet is on; rental runs must match the offline Kaggle run")
        shutil.copy2(folder / meta["code_file"], out / f"run{n}.ipynb")
        (out / f"run{n}.kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        runs.append({"name": folder.name, "slug": meta["id"].split("/", 1)[1], "notebook": f"run{n}.ipynb",
                     "metadata": f"run{n}.kernel-metadata.json", "timeout_s": timeout_s})
    shutil.copy2(ROOT / "scripts" / "rental_box.py", out / "rental_box.py")
    manifest = {"job": job, "owner": owner, "results_prefix": dataset_slug(job), "runs": runs, "packed_utc": utc()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": f"arc3 rental {job}", "id": f"{owner}/{dataset_slug(job)}", "licenses": [{"name": "CC0-1.0"}]}))
    (job_dir(job) / "pack.json").write_text(json.dumps({"job_sha": job_sha(out), "packed_utc": manifest["packed_utc"],
                                                        "folders": [str(f) for f in folders]}, indent=1))
    return out


def packed_sha(job: str) -> str:
    return json.loads((job_dir(job) / "pack.json").read_text())["job_sha"]


def kaggle_env() -> dict:
    env = dict(os.environ)
    if KAGGLE_TOKEN.exists() and not env.get("KAGGLE_API_TOKEN"):
        env["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN.read_text().strip()
    return env


def kaggle(*args: str, expect: str | None = None, check: bool = True) -> str:
    r = subprocess.run([str(ROOT / ".venv" / "bin" / "kaggle"), *args], env=kaggle_env(), capture_output=True,
                       text=True, check=False, timeout=3600)
    out = (r.stdout + r.stderr).strip()
    if check and (r.returncode != 0 or (expect is not None and expect not in out)):
        # the CLI exits 0 on some failures ("Dataset creation error: ..."), hence `expect`
        raise SystemExit(f"kaggle {' '.join(args[:3])} failed: {out[-600:]}")
    return out


def upload_job(job: str) -> str:
    """Create the private job dataset, or add a version when it exists; then the uploaded files are checked in launch."""
    folder = job_dir(job) / "job"
    ref = f"{OWNER}/{dataset_slug(job)}"
    exists = "ready" in kaggle("datasets", "status", ref, check=False).lower()
    if exists:
        kaggle("datasets", "version", "-p", str(folder), "-m", f"pack {packed_sha(job)[:12]}", "--dir-mode", "skip",
               expect="Dataset version is being created")
    else:
        kaggle("datasets", "create", "-p", str(folder), "--dir-mode", "skip",
               expect="Your private Dataset is being created")  # private: --public is never passed
    return ref


def uploaded_sha(job: str) -> str:
    """sha256 of the job as Kaggle serves it now (a ~1 MB download), to compare with the local pack."""
    with tempfile.TemporaryDirectory() as tmp:
        kaggle("datasets", "download", f"{OWNER}/{dataset_slug(job)}", "-p", tmp, "--unzip", "--force")
        return job_sha(Path(tmp))


BOOT = r"""set -u
mkdir -p /root/job /root/results
exec > >(tee -a /root/results/boot.log) 2>&1
echo "RENTAL_BOOT $(date -u +%FT%TZ) job=$RENTAL_JOB"
stop_self() {
  python3 - <<'PY' || true
import json, os, urllib.request
cid, key = os.environ.get("CONTAINER_ID"), os.environ.get("CONTAINER_API_KEY")
if cid and key:
    req = urllib.request.Request(f"https://console.vast.ai/api/v0/instances/{cid}/", method="PUT",
        data=json.dumps({"state": "stopped"}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    print("RENTAL_STOP_SELF", urllib.request.urlopen(req, timeout=60).status, flush=True)
else:
    print("RENTAL_STOP_SELF skipped: no CONTAINER_ID / CONTAINER_API_KEY", flush=True)
PY
}
trap 'echo "RENTAL_BOOT_END $(date -u +%FT%TZ)"; stop_self' EXIT
if [ ! -x /usr/local/bin/kcli ]; then
  python3 -m pip install -q --no-warn-script-location --target /opt/kcli "kaggle==__KAGGLE_CLI_VERSION__" || exit 10
  printf '#!/bin/sh\nPYTHONPATH=/opt/kcli exec python3 -m kaggle.cli "$@"\n' > /usr/local/bin/kcli
  chmod +x /usr/local/bin/kcli
fi
export RENTAL_KAGGLE_CLI=/usr/local/bin/kcli
if [ ! -f /root/job/manifest.json ]; then
  /usr/local/bin/kcli datasets download "$RENTAL_JOB" -p /root/job --unzip || exit 11
fi
cd /root/job
timeout --kill-after=300 "${RENTAL_CAP_HOURS}h" python3 rental_box.py /root/job/manifest.json
echo "RENTAL_BOX_EXIT rc=$?"
"""


def boot_command() -> str:
    return BOOT.replace("__KAGGLE_CLI_VERSION__", KAGGLE_CLI_VERSION)


def create_body(job: str, token: str, sha: str, cap_hours: float, owner: str = OWNER) -> dict:
    """The vast.ai create-instance body, shaped as the official CLI sends `--entrypoint bash --args -c <boot>`
    (runtype 'args': no ssh/jupyter injected; `onstart` carries the entrypoint; env as an object)."""
    return {"client_id": "me", "image": IMAGE, "disk": DISK_GB, "label": f"arc3-{job}", "runtype": "args",
            "onstart": "bash", "args": ["-c", boot_command()], "cancel_unavail": True,
            "env": {"KAGGLE_API_TOKEN": token, "RENTAL_JOB": f"{owner}/{dataset_slug(job)}", "RENTAL_JOB_SHA": sha,
                    "RENTAL_CAP_HOURS": f"{cap_hours:.2f}"}}


def quote(job: str, offer: dict) -> dict:
    manifest = json.loads((job_dir(job) / "job" / "manifest.json").read_text())
    hours = HOURS_SETUP + HOURS_PER_RUN * len(manifest["runs"])
    dph = run_dph(offer)
    return {"job": job, "runs": [r["name"] for r in manifest["runs"]], "offer": offer.get("id"),
            "gpu": offer.get("gpu_name"), "ram_share_gb": round(ram_share_gb(offer)), "dph_with_disk": round(dph, 3),
            "hours_estimate": round(hours, 2), "cost_estimate_usd": round(dph * hours, 2),
            "cap_hours": round(hours + CAP_SLACK_HOURS, 2), "cost_cap_usd": round(dph * (hours + CAP_SLACK_HOURS), 2),
            "disk_per_day_after_exit_usd": round(disk_dph(offer) * 24, 2),
            "contract_hours": round(float(offer.get("duration") or 0) / 3600)}


def instances(key: str) -> list[dict]:
    res = vast("GET", "/instances/", key=key) or {}
    return res.get("instances", []) if isinstance(res, dict) else []


def launch(job: str, offer_id: int, approved: str) -> dict:
    if not approved.strip():
        raise SystemExit("launch spends money: pass --approved with the owner's OK for this batch (CLAUDE.md)")
    state_file = job_dir(job) / "launch.json"
    if state_file.exists():
        raise SystemExit(f"{job} has a launch record ({state_file}); check `status`, or pack a new job name")
    key = vast_key()
    offer = find_offer(offer_id)
    if offer is None:
        raise SystemExit(f"offer {offer_id} is gone or no longer meets the filters; run `offers` again")
    q = quote(job, offer)
    if q["contract_hours"] < 1.5 * q["cap_hours"]:
        raise SystemExit(f"offer {offer_id}'s contract ends in {q['contract_hours']} h; the job may need {q['cap_hours']}")
    sha = packed_sha(job)
    if uploaded_sha(job) != sha:
        raise SystemExit("the uploaded job dataset differs from the local pack: `pack ... --upload` again, wait a "
                         "minute for Kaggle to process it, then launch")
    label = f"arc3-{job}"
    if any(i.get("label") == label for i in instances(key)):
        raise SystemExit(f"an instance labelled {label} exists already; `destroy` it or pack a new job name")
    token = kaggle_env().get("KAGGLE_API_TOKEN")
    if not token:
        raise SystemExit("no Kaggle token (.kaggle/access_token)")
    record: dict[str, Any] = {"job": job, "offer": offer_id, "instance": None, "status": "creating",
                              "approved": approved, "quote": q, "job_sha": sha, "image": IMAGE, "label": label,
                              "launched_utc": utc()}
    state_file.write_text(json.dumps(record, indent=1))  # before the PUT: a lost response still leaves a trace
    try:
        result = vast("PUT", f"/asks/{offer_id}/", create_body(job, token, sha, q["cap_hours"]), key=key)
        if not (isinstance(result, dict) and result.get("success") and isinstance(result.get("new_contract"), int)):
            raise RuntimeError(f"unexpected create response: {result}")
        record.update(instance=result["new_contract"], status="created")
    except (RuntimeError, OSError) as exc:
        found = [i for i in instances(key) if i.get("label") == label]
        record.update(status="created (recovered by label)" if found else "failed", error=str(exc)[:500],
                      instance=found[0].get("id") if found else None)
    state_file.write_text(json.dumps(record, indent=1))
    return record


def launched(job: str) -> dict:
    path = job_dir(job) / "launch.json"
    if not path.exists():
        raise SystemExit(f"{job} has no launch.json")
    rec = json.loads(path.read_text())
    if rec.get("instance") is None:
        found = [i for i in instances(vast_key()) if i.get("label") == rec.get("label")]
        if found:
            rec["instance"] = found[0].get("id")
            path.write_text(json.dumps(rec, indent=1))
    if rec.get("instance") is None:
        raise SystemExit(f"{job}: no instance id (launch status {rec.get('status')})")
    return rec


def status(job: str) -> dict:
    rec = launched(job)
    info = vast("GET", f"/instances/{rec['instance']}/", key=vast_key())
    inst = (info or {}).get("instances", info) or {}
    dph = float(inst.get("dph_total") or 0)
    age_h = (time.time() - float(inst.get("start_date") or time.time())) / 3600
    return {"instance": rec["instance"], "actual_status": inst.get("actual_status"),
            "intended_status": inst.get("intended_status"), "status_msg": inst.get("status_msg"),
            "dph": dph, "hours_since_start": round(age_h, 2), "cost_so_far_usd": round(dph * age_h, 2),
            "cap_hours": rec["quote"].get("cap_hours")}


def logs(job: str, tail: int = 200) -> str:
    rec = launched(job)
    res = vast("PUT", f"/instances/request_logs/{rec['instance']}/", {"tail": str(tail)}, key=vast_key())
    url = (res or {}).get("result_url")
    if not url:
        return json.dumps(res)
    for _ in range(12):  # the log is uploaded to S3 a few seconds after the request
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read().decode(errors="replace")
        except urllib.error.HTTPError:
            time.sleep(5)
    return f"log not ready at {url}"


def destroy(job: str) -> Any:
    rec = launched(job)
    return vast("DELETE", f"/instances/{rec['instance']}/", key=vast_key())


def _unpack(dl: Path, dest: Path) -> None:
    archive = dl / ARCHIVE
    if archive.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(dest, filter="data")
        return
    extracted = [p for p in dl.iterdir() if p.name not in {"run.json", "dataset-metadata.json"}]
    if not extracted:  # Kaggle may have unpacked the archive anyway: take the tree as it came
        raise FileNotFoundError(f"no {ARCHIVE} and no output files in {dl}")
    for path in extracted:
        target = dest / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)


def collect(job: str, score: bool = True, destroy_after: bool = False) -> list[dict]:
    manifest = json.loads((job_dir(job) / "job" / "manifest.json").read_text())
    out = []
    for n, spec in enumerate(manifest["runs"], start=1):
        name = f"{spec['name']}-r{n}-rental-{job}"
        row: dict[str, Any] = {"run": name, "dataset": f"{manifest['owner']}/{manifest['results_prefix']}-{n}"}
        try:
            dl = ROOT / "runs" / name / "rental"
            if dl.exists():
                shutil.rmtree(dl)
            dl.mkdir(parents=True)
            kaggle("datasets", "download", row["dataset"], "-p", str(dl), "--unzip", "--force")
            dest = ROOT / "runs" / name / "kernel-output"
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True)
            _unpack(dl, dest)
            row["execution"] = json.loads((dl / "run.json").read_text()).get("execution")
            row["ok"] = True
        except (SystemExit, OSError, ValueError, tarfile.TarError) as exc:
            row.update(ok=False, error=str(exc)[:400])
        if row.get("ok") and score:
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "pull_taaf_run.py"), "--local", name, name],
                               capture_output=True, text=True, check=False)
            row["score_log"] = (r.stdout + r.stderr).strip().splitlines()[-3:]
        out.append(row)
    if destroy_after:
        if all(r.get("ok") for r in out):
            out.append({"destroyed": destroy(job)})
        else:
            out.append({"destroyed": False, "why": "some results are missing: the instance keeps its disk"})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("offers")
    p = sub.add_parser("pack")
    p.add_argument("job")
    p.add_argument("folders", nargs="+", type=Path)
    p.add_argument("--upload", action="store_true")
    p = sub.add_parser("quote")
    p.add_argument("job")
    p.add_argument("--offer", type=int, required=True)
    p = sub.add_parser("launch")
    p.add_argument("job")
    p.add_argument("--offer", type=int, required=True)
    p.add_argument("--approved", default="")
    for name in ("status", "destroy"):
        sub.add_parser(name).add_argument("job")
    p = sub.add_parser("collect")
    p.add_argument("job")
    p.add_argument("--destroy", action="store_true")
    p = sub.add_parser("logs")
    p.add_argument("job")
    p.add_argument("--tail", type=int, default=200)
    args = ap.parse_args(argv)
    if args.cmd == "offers":
        for o in search():
            log(offer_line(o))
    elif args.cmd == "pack":
        out = pack(args.job, args.folders)
        log(f"packed {out} (sha {packed_sha(args.job)[:12]}): "
            f"{[r['name'] for r in json.loads((out / 'manifest.json').read_text())['runs']]}")
        if args.upload:
            log(f"uploaded private dataset {upload_job(args.job)}; launch checks it once Kaggle has processed it")
    elif args.cmd == "quote":
        offer = find_offer(args.offer)
        if offer is None:
            raise SystemExit(f"offer {args.offer} is gone or no longer meets the filters")
        log(json.dumps(quote(args.job, offer), indent=1))
    elif args.cmd == "launch":
        log(json.dumps(launch(args.job, args.offer, args.approved), indent=1))
    elif args.cmd == "status":
        log(json.dumps(status(args.job), indent=1))
    elif args.cmd == "logs":
        log(logs(args.job, args.tail))
    elif args.cmd == "destroy":
        log(json.dumps(destroy(args.job)))
    elif args.cmd == "collect":
        log(json.dumps(collect(args.job, destroy_after=args.destroy), indent=1))


if __name__ == "__main__":
    main()
