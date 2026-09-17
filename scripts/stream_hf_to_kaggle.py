"""Stream a Hugging Face model repo into a private Kaggle dataset without staging it on disk.

    KAGGLE_API_TOKEN=... .venv/bin/python scripts/stream_hf_to_kaggle.py nvidia/Qwen3.8-Flash-Next-NVFP4 \
        scottmahony/qwen3-8-flash-next-nvfp4 --title "Qwen3.8-Flash-Next-NVFP4 (nvidia, HF snapshot)" \
        --subtitle "Mirror of HF nvidia/Qwen3.8-Flash-Next-NVFP4 for offline vLLM on Kaggle" --license other \
        --state /path/to/state.json [--only small] [--extra-file path ...]

Why: the Claude Code container has ~18-28 GB of writable disk and the checkpoint is 132.7 GB with a 53.7 GB shard, while
the Kaggle CLI needs the whole folder on disk. Kaggle's own upload is a resumable PUT to a signed URL obtained from
start_blob_upload, and Hugging Face serves files with byte ranges, so each file is piped GET -> PUT with resume on
either side. Tokens of completed files are kept in the state file, so a crash resumes with the missing files only;
the dataset is created once every file has a token.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pkgutil
import time
from pathlib import Path

import requests

HF = "https://huggingface.co"
CHUNK = 8 << 20  # 8 MiB


def _find(name: str):
    """Import a kagglesdk type by class name (module layout differs between client versions)."""
    import kagglesdk

    for mod in pkgutil.walk_packages(kagglesdk.__path__, "kagglesdk."):
        try:
            m = importlib.import_module(mod.name)
        except Exception:  # noqa: S112
            continue
        if hasattr(m, name):
            return getattr(m, name)
    raise ImportError(name)


def hf_tree(repo: str, revision: str = "main") -> tuple[str, list[dict]]:
    info = requests.get(f"{HF}/api/models/{repo}", timeout=60, headers={"User-Agent": "arc3"}).json()
    tree = requests.get(f"{HF}/api/models/{repo}/tree/{revision}", timeout=60, headers={"User-Agent": "arc3"}).json()
    files = [{"path": e["path"], "size": (e.get("lfs") or {}).get("size", e.get("size", 0))} for e in tree if e.get("type") == "file"]
    return info.get("sha", revision), files


def hf_open(repo: str, path: str, start: int = 0) -> requests.Response:
    """Streaming GET of one file, optionally from a byte offset (follows the CDN redirect each time)."""
    headers = {"User-Agent": "arc3"}
    if start:
        headers["Range"] = f"bytes={start}-"
    r = requests.get(f"{HF}/{repo}/resolve/main/{path}", headers=headers, stream=True, timeout=(30, 300), allow_redirects=True)
    if r.status_code not in (200, 206):
        raise RuntimeError(f"HF GET {path} from {start}: {r.status_code}")
    return r


class Source:
    """A file-like reader over the HF stream (so requests can send it with a known Content-Length)."""

    def __init__(self, repo: str, path: str, start: int, total: int, log):
        self.resp = hf_open(repo, path, start)
        self.it = self.resp.iter_content(chunk_size=CHUNK)
        self.sent = start
        self.start = start
        self.total = total
        self.log = log
        self.t0 = time.time()
        self.last = self.t0

    def __len__(self) -> int:  # requests needs a length, or it switches to chunked encoding, which the signed URL rejects (400)
        return self.total - self.start

    def read(self, n: int = -1) -> bytes:
        try:
            chunk = next(self.it)
        except StopIteration:
            return b""
        self.sent += len(chunk)
        now = time.time()
        if now - self.last > 30:
            rate = (self.sent / max(1e-9, now - self.t0)) / 1e6
            self.log(f"    {self.sent / 1e9:.2f}/{self.total / 1e9:.2f} GB  {rate:.0f} MB/s")
            self.last = now
        return chunk

    def __iter__(self):
        while True:
            c = self.read()
            if not c:
                return
            yield c


def query_offset(url: str, total: int) -> int | None:
    """Ask the resumable upload how much it has (308 + Range), or None when it is complete / expired."""
    r = requests.put(url, headers={"Content-Length": "0", "Content-Range": f"bytes */{total}"}, timeout=60)
    if r.status_code in (200, 201):
        return -1  # complete
    if r.status_code == 308:
        rng = r.headers.get("Range")
        return int(rng.split("-")[-1]) + 1 if rng else 0
    return None


def upload_one(api, blob_type, repo: str, path: str, size: int, log, attempts: int = 8) -> str:
    StartReq = _find("ApiStartBlobUploadRequest")
    req = StartReq()
    req.type = blob_type
    req.name = os.path.basename(path)
    req.content_length = size
    req.last_modified_epoch_seconds = int(time.time())
    with api.build_kaggle_client() as kaggle:
        resp = api.with_retry(kaggle.blobs.blob_api_client.start_blob_upload)(req)
    url, token = resp.create_url, resp.token
    start = 0
    for attempt in range(attempts):
        try:
            src = Source(repo, path, start, size, log)
            headers = {"Content-Length": str(size - start)}
            if start:
                headers["Content-Range"] = f"bytes {start}-{size - 1}/{size}"
            r = requests.put(url, data=src, headers=headers, timeout=(30, 600))
            if r.status_code in (200, 201):
                return token
            log(f"    PUT returned {r.status_code}: {r.text[:200]}")
        except Exception as e:
            log(f"    transfer error ({type(e).__name__}: {str(e)[:160]})")
        off = query_offset(url, size)
        if off == -1:
            return token
        if off is None:
            raise RuntimeError(f"upload of {path} expired or failed; restart the file")
        start = off
        log(f"    resuming {path} at {start / 1e9:.2f} GB (attempt {attempt + 2})")
        time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"{path}: gave up after {attempts} attempts")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo")
    p.add_argument("dataset")  # owner/slug
    p.add_argument("--title", required=True)
    p.add_argument("--subtitle", required=True)
    p.add_argument("--license", default="other")
    p.add_argument("--state", required=True)
    p.add_argument("--only", choices=["all", "small"], default="all", help="small: files under 1 GB only (dry run of the path)")
    p.add_argument("--extra-file", action="append", default=[], help="local files to add (provenance, metadata notes)")
    p.add_argument("--max-files", type=int, default=0)
    a = p.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    blob_type = _find("ApiBlobType").DATASET
    state_path = Path(a.state)
    state = json.loads(state_path.read_text()) if state_path.exists() else {"tokens": {}}
    logf = open(str(state_path) + ".log", "a")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    sha, files = hf_tree(a.repo)
    if a.only == "small":
        files = [f for f in files if f["size"] < 1 << 30]
    if a.max_files:
        files = files[: a.max_files]
    files.sort(key=lambda f: f["size"])
    log(f"{a.repo} @ {sha}: {len(files)} files, {sum(f['size'] for f in files) / 1e9:.1f} GB; {len(state['tokens'])} already uploaded")
    for f in files:
        if f["path"] in state["tokens"]:
            continue
        log(f"uploading {f['path']} ({f['size'] / 1e9:.2f} GB)")
        t0 = time.time()
        token = upload_one(api, blob_type, a.repo, f["path"], f["size"], log)
        state["tokens"][f["path"]] = token
        state_path.write_text(json.dumps(state, indent=1))
        log(f"  done {f['path']} in {time.time() - t0:.0f} s")
    for extra in a.extra_file:
        name = os.path.basename(extra)
        if name in state["tokens"]:
            continue
        log(f"uploading local {name}")
        with api.build_kaggle_client() as kaggle:
            StartReq = _find("ApiStartBlobUploadRequest")
            req = StartReq()
            req.type = blob_type
            req.name = name
            req.content_length = os.path.getsize(extra)
            req.last_modified_epoch_seconds = int(os.path.getmtime(extra))
            resp = api.with_retry(kaggle.blobs.blob_api_client.start_blob_upload)(req)
        with open(extra, "rb") as fh:
            r = requests.put(resp.create_url, data=fh, headers={"Content-Length": str(os.path.getsize(extra))}, timeout=300)
        if r.status_code not in (200, 201):
            raise RuntimeError(f"{name}: {r.status_code}")
        state["tokens"][name] = resp.token
        state_path.write_text(json.dumps(state, indent=1))
    # create the dataset
    Req = _find("ApiCreateDatasetRequest")
    NewFile = _find("ApiDatasetNewFile")

    owner, slug = a.dataset.split("/", 1)
    req = Req()
    req.title = a.title
    req.slug = slug
    req.owner_slug = owner
    req.license_name = a.license
    req.subtitle = a.subtitle
    req.is_private = True
    req.category_ids = []
    req.files = []
    for token in state["tokens"].values():
        nf = NewFile()
        nf.token = token
        req.files.append(nf)
    with api.build_kaggle_client() as kaggle:
        result = api.with_retry(kaggle.datasets.dataset_api_client.create_dataset)(req)
    log(f"create_dataset: status={getattr(result, 'status', None)} error={getattr(result, 'error', None)} url={getattr(result, 'url', None)}")
    state["created"] = {"status": str(getattr(result, "status", None)), "error": str(getattr(result, "error", None)), "sha": sha}
    state_path.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    main()
