#!/usr/bin/env python
"""Run our built Kaggle notebooks on rented vast.ai GPUs and bring the output back like a Kaggle run.

    .venv/bin/python scripts/rental.py offers                                  # read-only, no account needed
    .venv/bin/python scripts/rental.py pack JOB DIR [DIR ...] [--upload]         # built notebook folders -> a job
    .venv/bin/python scripts/rental.py quote JOB --offer ID                      # what the job will cost on that offer
    .venv/bin/python scripts/rental.py launch JOB --offer ID --approved "<the owner's OK for this batch>"
    .venv/bin/python scripts/rental.py status JOB | logs JOB [--tail N] | destroy JOB
    .venv/bin/python scripts/rental.py collect JOB                              # results -> runs/<name>/kernel-output

One job = the runs one box plays one after another. A box is one RTX PRO 6000: Keith's serving setup requires exactly
one such GPU in nvidia-smi, so two notebooks cannot share a 2-GPU box unmodified; two 1-GPU boxes cost the same.

How a job travels (docs/research/rental-runner.md): ``pack`` copies each built notebook and its kernel-metadata.json
plus scripts/rental_box.py and a manifest into ``runs/rental/<JOB>/job/`` and, with ``--upload``, creates the private
Kaggle dataset ``<owner>/arc3-rental-<JOB>``. ``launch`` rents the offer with the Kaggle GPU image (pinned tag), a
500 GB disk and a boot command that installs the Kaggle CLI apart from the image's Python, downloads the job dataset
and runs rental_box.py, which downloads the notebooks' inputs into the /kaggle/input layout, runs each notebook and
uploads each run's /kaggle/working as a private dataset ``<owner>/arc3-rental-<JOB>-<n>``. When the command ends the
container exits and the GPU is released; ``collect`` downloads the results and scores them with pull_taaf_run.py, and
``destroy`` deletes the stopped instance (its disk is billed until then).

Money: ``launch`` is the only call that spends. It refuses without ``--approved`` (the owner's words for this batch,
kept in ``runs/rental/<JOB>/launch.json``); CLAUDE.md: stop and ask before spending money. Secrets: the vast.ai key is
read from ``.vast/api_key`` (git-ignored) and used only here; the Kaggle token (``.kaggle/access_token``) is passed to
the instance as KAGGLE_API_TOKEN, so vast.ai and the host can read it: rotate it on kaggle.com after each rental.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RENTAL_DIR = ROOT / "runs" / "rental"
VAST = "https://console.vast.ai/api/v0"
VAST_KEY = ROOT / ".vast" / "api_key"
KAGGLE_TOKEN = ROOT / ".kaggle" / "access_token"
OWNER = "scottmahony"
IMAGE = "gcr.io/kaggle-gpu-images/python:v170"  # Kaggle's GPU image, the tag `latest` pointed at on 2026-09-29
DISK_GB = 500  # image ~60 GB unpacked + the 135 GB model twice while its tar extracts + runtimes and wheels
KAGGLE_CLI_VERSION = "2.2.4"  # the version this repo uses
HOURS_PER_RUN = 2.75  # public-25 notebook: ~16 min serving setup + 132 min play + teardown (exp-054's logs)
HOURS_SETUP = 1.0  # image pull + ~170 GB of inputs on a fresh box (estimate until the first rental measures it)

# What a box must have to run the notebook like Kaggle's g4-standard-48 (1 RTX PRO 6000, 48 vCPU, ~180 GB RAM):
# Keith's setup needs >= 64 GiB free RAM beside the 48 GiB pinned PLE table, CUDA 13.0 needs driver >= 580.
OFFER_FILTER = {
    "rentable": {"eq": True},
    "num_gpus": {"eq": 1},
    "gpu_name": {"in": ["RTX PRO 6000 S", "RTX PRO 6000 WS"]},
    "cpu_ram": {"gte": 170 * 1024},
    "cpu_cores_effective": {"gte": 32},
    "disk_space": {"gte": DISK_GB},
    "reliability": {"gte": 0.97},
    "inet_down": {"gte": 500},
}


def log(msg: str) -> None:
    print(msg, flush=True)


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
        raise SystemExit(f"vast.ai {method} {path}: HTTP {exc.code} {exc.read().decode(errors='replace')[:500]}") from exc


def vast_key() -> str:
    if not VAST_KEY.is_file():
        raise SystemExit(f"no vast.ai API key: put it in {VAST_KEY.relative_to(ROOT)} (git-ignored)")
    return VAST_KEY.read_text().strip()


def driver_major(offer: dict) -> int:
    try:
        return int(str(offer.get("driver_version") or "0").split(".")[0])
    except ValueError:
        return 0


def good_offers(offers: list[dict]) -> list[dict]:
    """The API filter plus what it cannot express: driver >= 580 (CUDA 13.0); server edition first, then price."""
    keep = [o for o in offers if driver_major(o) >= 580 and o.get("num_gpus") == 1]
    return sorted(keep, key=lambda o: (o.get("gpu_name") != "RTX PRO 6000 S", float(o.get("dph_total") or 99)))


def search_offers(limit: int = 60) -> list[dict]:
    body = {"limit": limit, "type": "ondemand", "order": [["dph_total", "asc"]], **OFFER_FILTER}
    return good_offers((vast("POST", "/bundles/", body) or {}).get("offers", []))


def find_offer(offer_id: int) -> dict | None:
    """One offer by id through the public search (no key needed), or None if it is gone or fails the filter."""
    body = {"limit": 5, "type": "ondemand", **OFFER_FILTER, "id": {"eq": int(offer_id)}}
    return next(iter(good_offers((vast("POST", "/bundles/", body) or {}).get("offers", []))), None)


def offer_line(o: dict) -> str:
    return (f"{o['id']:>10}  {o.get('gpu_name', '?'):<16} ${float(o.get('dph_total') or 0):.2f}/h  "
            f"RAM {round((o.get('cpu_ram') or 0) / 1024)} GB  {o.get('cpu_cores_effective', 0):.0f} cores  "
            f"disk {round(o.get('disk_space') or 0)} GB  down {round(o.get('inet_down') or 0)} Mb/s  "
            f"driver {o.get('driver_version')}  rel {float(o.get('reliability2') or o.get('reliability') or 0):.3f}  "
            f"{o.get('verification', '?')}  {o.get('geolocation', '')}")


# --- jobs -----------------------------------------------------------------------------------------------------------


def job_dir(job: str) -> Path:
    return RENTAL_DIR / job


def dataset_slug(job: str) -> str:
    return f"arc3-rental-{job}"


def pack(job: str, folders: list[Path], owner: str = OWNER, timeout_s: float = 5 * 3600) -> Path:
    """Copy built notebook folders into runs/rental/<job>/job/ as flat files with a manifest and the box runner."""
    if not job.replace("-", "").isalnum() or len(job) > 24:
        raise SystemExit("job names are letters, digits and dashes, at most 24 characters (they end up in slugs)")
    out = job_dir(job) / "job"
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
    manifest = {"job": job, "owner": owner, "results_prefix": dataset_slug(job), "upload": True, "runs": runs,
                "packed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": f"arc3 rental {job}", "id": f"{owner}/{dataset_slug(job)}", "licenses": [{"name": "CC0-1.0"}]}))
    return out


def kaggle_env() -> dict:
    env = dict(os.environ)
    if KAGGLE_TOKEN.exists() and not env.get("KAGGLE_API_TOKEN"):
        env["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN.read_text().strip()
    return env


def kaggle(*args: str) -> str:
    r = subprocess.run([str(ROOT / ".venv" / "bin" / "kaggle"), *args], env=kaggle_env(), capture_output=True,
                       text=True, check=False)
    if r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args[:3])} failed: {(r.stdout + r.stderr).strip()[-600:]}")
    return r.stdout


def upload_job(job: str) -> str:
    folder = job_dir(job) / "job"
    kaggle("datasets", "create", "-p", str(folder), "--dir-mode", "skip")  # private: no --public
    return f"{OWNER}/{dataset_slug(job)}"


BOOT = r"""set -eu
mkdir -p /root/job /root/results
exec > >(tee -a /root/results/boot.log) 2>&1
echo "RENTAL_BOOT $(date -u +%FT%TZ) job=$RENTAL_JOB"
python3 -m pip install -q --no-warn-script-location --target /opt/kcli "kaggle==__KAGGLE_CLI_VERSION__"
printf '#!/bin/sh\nPYTHONPATH=/opt/kcli exec python3 -m kaggle.cli "$@"\n' > /usr/local/bin/kcli && chmod +x /usr/local/bin/kcli
export RENTAL_KAGGLE_CLI=/usr/local/bin/kcli
kcli datasets download "$RENTAL_JOB" -p /root/job --unzip
cd /root/job && python3 rental_box.py /root/job/manifest.json
echo "RENTAL_BOOT_END $(date -u +%FT%TZ)"
"""


def boot_command() -> str:
    return BOOT.replace("__KAGGLE_CLI_VERSION__", KAGGLE_CLI_VERSION)


def create_body(job: str, token: str, owner: str = OWNER) -> dict:
    """The vast.ai create-instance body: entrypoint mode ('args'), so the container ends (and the GPU is released)
    when the boot command ends."""
    env = f"-e KAGGLE_API_TOKEN={shlex.quote(token)} -e RENTAL_JOB={owner}/{dataset_slug(job)}"
    return {"client_id": "me", "image": IMAGE, "disk": DISK_GB, "label": f"arc3-{job}", "runtype": "args",
            "args": ["bash", "-c", boot_command()], "env": env}


def quote(job: str, offer: dict) -> dict:
    manifest = json.loads((job_dir(job) / "job" / "manifest.json").read_text())
    hours = HOURS_SETUP + HOURS_PER_RUN * len(manifest["runs"])
    dph = float(offer.get("dph_total") or 0)
    return {"job": job, "runs": [r["name"] for r in manifest["runs"]], "offer": offer.get("id"),
            "gpu": offer.get("gpu_name"), "dph": round(dph, 3), "hours_estimate": round(hours, 2),
            "cost_estimate_usd": round(dph * hours, 2), "cap_hours": round(hours + 1.5, 2)}


def launch(job: str, offer_id: int, approved: str) -> dict:
    if not approved.strip():
        raise SystemExit("launch spends money: pass --approved with the owner's OK for this batch (CLAUDE.md)")
    state_file = job_dir(job) / "launch.json"
    if state_file.exists():
        raise SystemExit(f"{job} was launched already ({state_file}); destroy it or use a new job name")
    key = vast_key()
    offer = find_offer(offer_id)
    if offer is None:
        raise SystemExit(f"offer {offer_id} is gone or no longer meets OFFER_FILTER; run `offers` again")
    q = quote(job, offer)
    token = kaggle_env().get("KAGGLE_API_TOKEN")
    if not token:
        raise SystemExit("no Kaggle token (.kaggle/access_token)")
    result = vast("PUT", f"/asks/{offer_id}/", create_body(job, token), key=key)
    record = {"job": job, "offer": offer_id, "instance": result.get("new_contract"), "approved": approved,
              "quote": q, "launched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "image": IMAGE}
    state_file.write_text(json.dumps(record, indent=1))
    return record


def launched(job: str) -> dict:
    path = job_dir(job) / "launch.json"
    if not path.exists():
        raise SystemExit(f"{job} has no launch.json")
    return json.loads(path.read_text())


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
    key = vast_key()
    res = vast("PUT", f"/instances/request_logs/{rec['instance']}/", {"tail": str(tail)}, key=key)
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


def collect(job: str, score: bool = True) -> list[dict]:
    manifest = json.loads((job_dir(job) / "job" / "manifest.json").read_text())
    out = []
    for n, spec in enumerate(manifest["runs"], start=1):
        name = f"{spec['name']}-rental-{job}"
        dl = ROOT / "runs" / name / "rental"
        dl.mkdir(parents=True, exist_ok=True)
        kaggle("datasets", "download", f"{manifest['owner']}/{manifest['results_prefix']}-{n}", "-p", str(dl),
               "--unzip", "--force")
        dest = ROOT / "runs" / name / "kernel-output"
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        with tarfile.open(dl / "output.tar.gz") as tar:
            tar.extractall(dest, filter="data")
        row = {"run": name, "record": json.loads((dl / "run.json").read_text()).get("execution")}
        if score:
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "pull_taaf_run.py"), "--local", name, name],
                               capture_output=True, text=True, check=False)
            row["score_log"] = (r.stdout + r.stderr).strip().splitlines()[-3:]
        out.append(row)
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
    for name in ("status", "destroy", "collect"):
        sub.add_parser(name).add_argument("job")
    p = sub.add_parser("logs")
    p.add_argument("job")
    p.add_argument("--tail", type=int, default=200)
    args = ap.parse_args(argv)
    if args.cmd == "offers":
        for o in search_offers():
            log(offer_line(o))
    elif args.cmd == "pack":
        out = pack(args.job, args.folders)
        log(f"packed {out}: {[r['name'] for r in json.loads((out / 'manifest.json').read_text())['runs']]}")
        if args.upload:
            log(f"uploaded private dataset {upload_job(args.job)}")
    elif args.cmd == "quote":
        offer = find_offer(args.offer)
        if offer is None:
            raise SystemExit(f"offer {args.offer} is gone or no longer meets OFFER_FILTER")
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
        log(json.dumps(collect(args.job), indent=1))


if __name__ == "__main__":
    main()
