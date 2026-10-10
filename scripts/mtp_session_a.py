#!/usr/bin/env python
"""Session A of the MTP draft fine-tune on Kaggle: steps A0-A11 of docs/research/beat-tufa/mtp-drafter-finetune.md
section 10.4, as the functions the session-A notebook calls (scripts/build_mtp_session.py builds it; section 11).

The notebook is D''s own notebook up to its SGLang launcher: cell 4's paths and setup, the wheelhouse install, and
cell 12's environment and launch arguments. Cell 12 stops right before it would start the server; the cells after it
call this module with the launcher's ``args`` and ``env``:

- A0  write test of the candidate dump directories (``/tmp``, ``/dev/shm``); where the probe and training dumps go,
      and their caps (:func:`plan_storage`).
- A1  the hidden-state dump patch on the installed sglang (scripts/sglang_hc_dump_patch.py), after REAP, which
      cell 12 applies when the served target is pruned.
- A2  probe dump server: the launcher's arguments without ``--speculative-*`` and without REAP's override, plus
      ``--disable-cuda-graph --disable-radix-cache``, ``--max-running-requests 1`` and ``--chunked-prefill-size 8192``
      (:func:`dump_server_args`); ``ARC3_HC_DUMP_KEEP=all``, ``ARC3_HC_DUMP_DTYPE=bf16``.
- A3  scripts/mtp_probe_dump.py: the held-out probe requests plus the greedy outputs the reference run recorded
      (at least ``probe.min_requests`` dumped completely).
- A4  stop the server.
- A5  scripts/mtp_replica.py check. Exit 2 (NO-GO) ends the session here: :meth:`Session.replica_gate` records the
      verdict (``code`` 2), writes the reports and returns False, and every later cell is skipped by
      :meth:`Session.go`, so the notebook ends normally without training.
- A6  training dump server: as A2 but with the served target (REAP's override kept when it is served), dump rows of
      assistant spans with context, FP8.
- A7  scripts/hc_dump_driver.py: the first snapshot of each held-out game (the evaluation set; one shared time budget
      and a cumulative size cap), then the train split, under the size and time caps.
- A8  stop the server; scripts/mtp_train.py plan.
- A9  scripts/mtp_train.py train with a time budget (:class:`TrainWatch`: when the planned steps cannot fit, one
      restart with the number of steps that does).
- A10 scripts/mtp_replica.py check --trained: the replica's forecast of the probe gate (exit 0 or 2).
- A11 scripts/mtp_write_draft.py write: ``<working>/mtp-draft`` (albucino's files, dense tensors replaced).

Every command's output goes to the notebook log and to ``<working>/logs/<name>.log``. ``<working>/session-a.json``
records the inputs, the storage decision, each step (commands, exit codes, times, a census line), the servers'
arguments and settings, and the verdicts; it is rewritten after every step, so a session that stops early still says
where and why. A failed step stops a running server and raises :class:`Stop` (code 1), which ends the notebook (the
Kaggle version then shows an error). Stdlib only: the notebook kernel's Python runs it; the torch steps run in the
Pennyroyal venv's python (``python_torch``) with the launcher's environment.

Inputs (``config["inputs"]``, :func:`locate`): each is found in either mount layout (lesson 0030; kernel outputs under
``<slug>`` or ``notebooks/<owner>/<slug>``, or any directory of that name up to four levels down, with the files up to
two levels below it) and checked against pinned sha256 values (one or several accepted versions) or byte sizes.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

GB = 1e9
BF16_ROW_BYTES = 10240 * 2 + 4 + 1 + 20  # scripts/mtp_probe_dump.py: a kept BF16 row and its prompt-token arrays
DUMP_FLAGS = ("--disable-cuda-graph", "--disable-radix-cache")          # added (plan 10.4, A2/A6)
SET_FLAGS = {"--max-running-requests": "1", "--chunked-prefill-size": "8192"}  # values replaced
DROP_PREFIX = "--speculative-"                                         # dropped with their values
REAP_FLAG = "--json-model-override-args"                               # kept only when the dump serves REAP
REAP_ENV = "ARC3_REAP_KEPT_EXPERTS"                                    # scripts/sglang_reap_patch.py
DUMP_ENV = "ARC3_HC_DUMP"                                              # scripts/sglang_hc_dump_patch.py
TMPFS = ("tmpfs", "ramfs")
INPUT_ROOT = "/kaggle/input"
SHARDS_RE = re.compile(r"checkpoint shards:\s*\d+% Completed \| (\d+)/(\d+)")  # the weight loader's tqdm line
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # localhost: never through a proxy


class Stop(RuntimeError):
    """Session A ends here; the reports are written. ``code`` 2 is the replica check's NO-GO, 1 anything else."""

    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


class InputError(RuntimeError):
    """An input is mounted but is not the one this notebook was built for."""


def _utc(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() if t is None else t))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------------------------- the dump servers' arguments


def split_flags(args) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """The leading words (program, subcommand), then each ``--flag`` with the words up to the next ``--flag``."""
    args = [str(a) for a in args]
    i = 0
    while i < len(args) and not args[i].startswith("--"):
        i += 1
    head, items = args[:i], []
    while i < len(args):
        j = i + 1
        while j < len(args) and not args[j].startswith("--"):
            j += 1
        items.append((args[i], args[i + 1:j]))
        i = j
    return head, items


def dump_server_args(launcher_args, *, reap: bool) -> list[str]:
    """A dump server's arguments: the launcher's, without ``--speculative-*`` flags (and their values), without
    REAP's ``--json-model-override-args`` unless ``reap``, with ``--max-running-requests 1`` and
    ``--chunked-prefill-size 8192``, plus ``--disable-cuda-graph --disable-radix-cache`` (plan 10.4, A2 and A6)."""
    head, items = split_flags(launcher_args)
    flags = [flag for flag, _ in items]
    for flag in DUMP_FLAGS:
        if flag in flags:
            raise ValueError(f"the launcher already passes {flag}")
    for flag in SET_FLAGS:
        if flags.count(flag) != 1:
            raise ValueError(f"the launcher passes {flag} {flags.count(flag)} times (expected once)")
    if reap and flags.count(REAP_FLAG) != 1:
        raise ValueError(f"the dump should serve REAP, but the launcher passes {REAP_FLAG} {flags.count(REAP_FLAG)} "
                         "times (expected once)")
    out = list(head)
    for flag, values in items:
        if flag.startswith(DROP_PREFIX) or (flag == REAP_FLAG and not reap):
            continue
        out += [flag, *([SET_FLAGS[flag]] if flag in SET_FLAGS else values)]
    return out + list(DUMP_FLAGS)


def dump_server_env(launcher_env: dict, dump_env: dict, *, reap: bool) -> dict:
    """The launcher's server environment with this server's ``ARC3_HC_DUMP*`` variables, and without REAP's
    variable unless the dump serves REAP."""
    env = {k: str(v) for k, v in launcher_env.items() if not k.startswith(DUMP_ENV)}
    if reap and REAP_ENV not in env:
        raise ValueError(f"the dump should serve REAP, but the launcher's environment has no {REAP_ENV}")
    if not reap:
        env.pop(REAP_ENV, None)
    bad = [k for k in dump_env if not k.startswith(DUMP_ENV)]
    if bad:
        raise ValueError(f"not dump variables: {bad}")
    env.update({k: str(v) for k, v in dump_env.items()})
    return env


def args_diff(launcher_args, server_args) -> dict:
    """What a server's arguments drop, add or change relative to the launcher's (by flag)."""
    _, a = split_flags(launcher_args)
    _, b = split_flags(server_args)
    da, db = dict(a), dict(b)
    return {"removed": [[f, *v] for f, v in a if f not in db],
            "added": [[f, *v] for f, v in b if f not in da],
            "changed": [[f, *da[f], "->", *v] for f, v in b if f in da and da[f] != v]}


def num_experts_override(info: dict) -> int | None:
    """``text_config.num_experts`` in the server's ``json_model_override_args`` (None when not overridden)."""
    raw = info.get("json_model_override_args")
    if not raw:
        return None
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return None
    n = (value.get("text_config") or {}).get("num_experts") if isinstance(value, dict) else None
    return int(n) if n is not None else None


def server_problems(info: dict, *, num_experts: int | None) -> list[str]:
    """Settings of a running dump server (its /server_info) that make the dump wrong: a prefix cache (cached rows are
    never recomputed, so never dumped), speculative decoding, or another expert count than the target it must
    mirror. The session stops on any of them."""
    if "error" in info:
        return [f"/server_info: {info['error']}"]
    problems = []
    if info.get("disable_radix_cache") is not True:
        problems.append(f"disable_radix_cache is {info.get('disable_radix_cache')!r}, not true")
    if info.get("speculative_algorithm"):
        problems.append(f"speculative decoding is on ({info['speculative_algorithm']})")
    if num_experts_override(info) != num_experts:
        problems.append(f"json_model_override_args {info.get('json_model_override_args')!r}: expected a "
                        f"num_experts override of {num_experts}")
    return problems


def server_notes(info: dict) -> list[str]:
    """Settings that differ from what was asked but do not change what is dumped (warnings): the drivers send one
    request at a time, and the dump does not depend on the server's chunk grid (the replica check rebuilds the
    reference run's grid from its own cache hits)."""
    if "error" in info:
        return []
    notes = []
    if info.get("max_running_requests") != 1:
        notes.append(f"max_running_requests is {info.get('max_running_requests')!r}, not 1")
    if info.get("chunked_prefill_size") != 8192:
        notes.append(f"chunked_prefill_size is {info.get('chunked_prefill_size')!r}, not 8192")
    return notes


# ----------------------------------------------------------------------------------------------- storage (A0)


def fs_type(path, mounts: str = "/proc/mounts") -> str | None:
    """The type of the filesystem holding PATH (the longest mount point above it)."""
    path = os.path.realpath(path)
    best, kind = "", None
    try:
        lines = Path(mounts).read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        point = parts[1].replace("\\040", " ")
        if (path == point or path.startswith(point.rstrip("/") + "/")) and len(point) >= len(best):
            best, kind = point, parts[2]
    return kind


def meminfo() -> dict:
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts:
                out[key] = int(parts[0]) * 1024
    except OSError:
        pass
    return out


def write_test(directory, *, gb: float, seconds: float, chunk_mb: int = 256, echo=print) -> dict:
    """Write up to GB gigabytes (or for up to SECONDS) to DIRECTORY, fsync, measure, delete. Random bytes, so no
    filesystem can compress or deduplicate them."""
    directory = Path(directory)
    rec = {"dir": str(directory), "fs": fs_type(directory), "target_gb": gb, "written_gb": 0.0, "error": None}
    target = directory / f"arc3-session-a-write-test-{os.getpid()}"
    try:
        target.mkdir(parents=True, exist_ok=True)
        st = os.statvfs(directory)
        rec.update(total_gb=round(st.f_blocks * st.f_frsize / GB, 2), free_gb_before=round(st.f_bavail * st.f_frsize / GB, 2))
    except OSError as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec
    block = (os.urandom(16 << 20) * max(1, -(-chunk_mb // 16)))[:max(1, chunk_mb) << 20]
    written, t0 = 0, time.time()
    try:
        with open(target / "fill.bin", "wb") as f:
            while written < gb * GB and time.time() - t0 < seconds:
                f.write(block)
                written += len(block)
            f.flush()
            os.fsync(f.fileno())
    except OSError as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        elapsed = time.time() - t0
        shutil.rmtree(target, ignore_errors=True)
    rec.update(written_gb=round(written / GB, 2), seconds=round(elapsed, 1),
               mb_s=round(written / 1e6 / max(elapsed, 1e-9), 1),
               complete=rec["error"] is None and written >= gb * GB)
    try:
        st = os.statvfs(directory)
        rec["free_gb_after"] = round(st.f_bavail * st.f_frsize / GB, 2)
    except OSError:
        pass
    echo(f"[session A A0] {directory} ({rec['fs']}): wrote {rec['written_gb']} GB in {rec['seconds']} s "
         f"({rec['mb_s']} MB/s), free {rec.get('free_gb_before')} GB of {rec.get('total_gb')}"
         + (f"; ERROR {rec['error']}" if rec["error"] else ""))
    return rec


def plan_storage(tests: list[dict], *, mem_total_gb: float, probe_gb: float, train_gb: float, holdout_gb: float,
                 min_train_gb: float, server_ram_gb: float, margin_gb: float) -> dict:
    """Where the probe dump (BF16, kept until A10) and the training dump (FP8, kept until A9) go, and their caps (GB).

    A directory holds up to its free space before the test, less a margin, when the test wrote without an error
    (also when it stopped at its time limit); up to what it wrote before the error, less the margin, when it failed
    (a full disk, a quota); nothing when nothing could be written. A tmpfs is RAM: the dumps placed on tmpfs together
    must also fit in ``mem_total_gb - server_ram_gb - margin_gb``, the RAM a dump server leaves free (106-119 of
    176.9 GB in use while serving, exp-073's census). The probe dump goes to the first directory (in the tests'
    order) that holds ``probe_gb``; the training dump to the first that then holds ``train_gb``, else to the one with
    the most room, capped at ``train_gb``. ``holdout_gb`` (the evaluation snapshots, dumped first into the training
    dump's directory) is at most 30% of the training cap. ``error`` is set when the probe dump does not fit or the
    training cap is below ``min_train_gb``."""
    room: dict[str, dict] = {}
    for t in tests:
        written = float(t.get("written_gb") or 0.0)
        if t.get("error"):
            gb = written - margin_gb
        elif written > 0:
            gb = float(t.get("free_gb_before") or 0.0) - margin_gb
        else:
            gb = 0.0
        if gb > 0:
            room[t["dir"]] = {"gb": round(gb, 2), "tmpfs": t.get("fs") in TMPFS}
    ram = mem_total_gb - server_ram_gb - margin_gb
    out: dict = {"room": room, "ram_budget_gb": round(ram, 2), "probe_dir": None, "train_dir": None, "error": None}
    order = [t["dir"] for t in tests if t["dir"] in room]

    def avail(d: str, used_tmpfs: float, minus: float = 0.0) -> float:
        gb = room[d]["gb"] - minus
        return min(gb, ram - used_tmpfs) if room[d]["tmpfs"] else gb

    probe = next((d for d in order if avail(d, 0.0) >= probe_gb), None)
    if probe is None:
        out["error"] = (f"no directory holds the probe dump ({probe_gb:g} GB of BF16 rows): "
                        f"{json.dumps(room)}, RAM budget for tmpfs {ram:.1f} GB")
        return out
    used = probe_gb if room[probe]["tmpfs"] else 0.0
    left = {d: avail(d, used, probe_gb if d == probe else 0.0) for d in order}
    train = next((d for d in order if left[d] >= train_gb), None) or max(order, key=lambda d: (left[d], -order.index(d)))
    cap = min(train_gb, left[train])
    out.update(probe_dir=probe, probe_gb=probe_gb, train_dir=train, train_gb=round(cap, 2),
               holdout_gb=round(min(holdout_gb, 0.3 * cap), 2))
    if cap < min_train_gb:
        out["error"] = (f"the training dump would get {cap:.1f} GB (< {min_train_gb:g}): {json.dumps(room)}, RAM budget "
                        f"for tmpfs {ram:.1f} GB")
    return out


# ------------------------------------------------------------------------------------------------------ inputs


def _walk_dirs(root: str, max_depth: int):
    root = os.path.normpath(root)
    base = root.count(os.sep)
    for path, dirs, files in os.walk(root):
        depth = path.count(os.sep) - base
        if depth >= max_depth:
            dirs[:] = []
        yield path, dirs, files


def candidate_dirs(spec: dict, root: str = INPUT_ROOT) -> list[str]:
    """Where Kaggle may mount the source (lesson 0030: datasets at datasets/<owner>/<slug> or <slug>; kernel outputs,
    by analogy, at notebooks/<owner>/<slug> or <slug>; also kernels/ and code/), then any directory named <slug>."""
    owner, slug = spec["id"].split("/")
    if spec["kind"] == "dataset":
        layouts = ["{root}/datasets/{owner}/{slug}", "{root}/{slug}"]
    else:
        layouts = ["{root}/{slug}", "{root}/notebooks/{owner}/{slug}", "{root}/kernels/{owner}/{slug}",
                   "{root}/code/{owner}/{slug}"]
    out = [x.format(root=root, owner=owner, slug=slug) for x in layouts]
    if os.path.isdir(root):
        for path, dirs, _ in _walk_dirs(root, 4):
            out += [os.path.join(path, d) for d in sorted(dirs) if d == slug and os.path.join(path, d) not in out]
    return out


def check_source(spec: dict, folder: Path) -> dict | None:
    """The source's files in FOLDER, checked (None when they are not all there; InputError when one differs)."""
    found = {}
    for name, expected in (spec.get("files") or {}).items():
        path = folder / name
        if not path.is_file():
            return None
        if isinstance(expected, int):  # a pinned size
            size = path.stat().st_size
            if size != expected:
                raise InputError(f"{path}: {size} bytes, not the pinned {expected} (another version of "
                                 f"{spec['id']}?)")
        else:
            got = sha256_file(path)
            if got not in (expected if isinstance(expected, (list, tuple)) else [expected]):
                raise InputError(f"{path}: sha256 {got[:16]}... is not the pinned one (another version of "
                                 f"{spec['id']}?)")
            found[name + ".sha256"] = got
        found[name] = str(path)
    return found


def locate(spec: dict, root: str = INPUT_ROOT, depth: int = 2) -> dict | None:
    """{"dir", "files"...} of the first candidate directory holding the source's files, else None. The files may
    also sit up to ``depth`` levels below a candidate (a layout that adds a version or ``output`` folder)."""
    seen = set()
    for folder in candidate_dirs(spec, root):
        if not os.path.isdir(folder):
            continue
        for path, _, _ in _walk_dirs(folder, depth):
            if path in seen:
                continue
            seen.add(path)
            found = check_source(spec, Path(path))
            if found is not None:
                return {"dir": path, **found}
    return None


def input_listing(root: str = INPUT_ROOT, depth: int = 3) -> list[str]:
    if not os.path.isdir(root):
        return [f"{root} does not exist"]
    return [f"{path} {sorted(dirs)[:8]} {len(files)} files" for path, dirs, files in _walk_dirs(root, depth)][:40]


def draft_source(root) -> Path:
    """albucino's draft directory under DRAFT_MODEL_DIR, found as cell 12's prepare_draft_view finds it: the one
    config.json whose quantization is compressed-tensors with an ``mtp_routed_experts`` group."""
    root = Path(root)
    configs = [root / "config.json"] if (root / "config.json").is_file() else []
    configs += sorted(p for p in root.rglob("config.json") if p not in configs)
    hits = []
    for path in configs:
        try:
            q = json.loads(path.read_text()).get("quantization_config", {})
        except (OSError, ValueError):
            continue
        if q.get("quant_method") == "compressed-tensors" and "mtp_routed_experts" in q.get("config_groups", {}):
            hits.append(path.parent)
    if len(hits) != 1:
        raise ValueError(f"expected one INT4 g32 MTP checkpoint under {root}, found {len(hits)}")
    return hits[0]


# ----------------------------------------------------------------------------------------------- training budget


def steps_that_fit(remaining_s: float, overhead_s: float, eval_s: float, step_s: float, slack: float = 1.1) -> int:
    """Optimizer steps a fresh training run can do in REMAINING_S: model load and the first evaluation (OVERHEAD_S),
    the steps at STEP_S each (times SLACK), the final evaluation (EVAL_S) and a minute for saving."""
    return max(0, math.floor((remaining_s - overhead_s - eval_s - 60.0) / max(step_s * slack, 1e-9)))


class TrainWatch:
    """Watches scripts/mtp_train.py train's train-log.jsonl. Once ``probe_steps`` steps are logged it projects the
    run's end (the remaining planned steps at the median step time so far, times ``slack``, plus a final evaluation as
    long as the first); when that passes the budget it stops the run and sets ``restart_steps``, the step count a
    fresh run fits in what is left (at least ``min_steps``), so the cosine schedule and the final evaluation and
    report still happen. It decides once."""

    def __init__(self, out: Path, total_steps: int | None, budget_s: float, *, probe_steps: int = 6,
                 slack: float = 1.1, min_steps: int = 10):
        self.out, self.total, self.budget = Path(out), total_steps, budget_s
        self.probe_steps, self.slack, self.min_steps = probe_steps, slack, min_steps
        self.first_seen: tuple[float, float] | None = None
        self.decided, self.restart_steps, self.info = False, None, {}

    def _lines(self) -> list[dict]:
        path = self.out / "train-log.jsonl"
        if not path.is_file():
            return []
        lines = []
        for text in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                line = json.loads(text)
            except ValueError:
                continue
            if isinstance(line, dict) and "step" in line and "seconds" in line:
                lines.append(line)
        return lines

    def __call__(self, elapsed: float) -> str | None:
        if self.decided or not self.total:
            return None
        lines = self._lines()
        if lines and self.first_seen is None:
            self.first_seen = (elapsed, float(lines[0]["seconds"]))
        if len(lines) < self.probe_steps:
            return None
        self.decided = True
        step_s = statistics.median(float(x["seconds"]) for x in lines[1:])
        try:
            eval_s = float(json.loads((self.out / "eval-original.json").read_text()).get("seconds") or 0.0)
        except (OSError, ValueError):
            eval_s = 0.0
        overhead = max(0.0, self.first_seen[0] - self.first_seen[1]) if self.first_seen else 0.0
        projected = elapsed + (self.total - len(lines)) * step_s * self.slack + eval_s + 60.0
        self.info = {"steps_logged": len(lines), "step_s": round(step_s, 2), "eval_s": round(eval_s, 1),
                     "overhead_s": round(overhead, 1), "projected_s": round(projected), "budget_s": round(self.budget),
                     "planned_steps": self.total}
        if projected <= self.budget:
            return None
        n = steps_that_fit(self.budget - elapsed, overhead, eval_s, step_s, self.slack)
        self.restart_steps = max(self.min_steps, min(n, self.total))
        self.info["restart_steps"] = self.restart_steps
        return (f"{self.total} planned steps would take ~{projected / 60:.0f} min (budget {self.budget / 60:.0f}); "
                f"restarting with --max-steps {self.restart_steps}")


# --------------------------------------------------------------------------------------------------- the session


def _http_json(url: str, timeout: float = 30.0):
    with _OPENER.open(url, timeout=timeout) as response:
        return response.status, response.read()


def default_census(paths) -> str:
    mem = meminfo()
    parts = []
    if mem.get("MemTotal"):
        parts.append(f"RAM {(mem['MemTotal'] - mem.get('MemAvailable', 0)) / GB:.1f}/{mem['MemTotal'] / GB:.1f} GB")
    if mem.get("Shmem") is not None:
        parts.append(f"shmem {mem['Shmem'] / GB:.1f} GB")
    for path in paths:
        try:
            st = os.statvfs(path)
        except OSError:
            continue
        total = st.f_blocks * st.f_frsize
        parts.append(f"{path} {(total - st.f_bavail * st.f_frsize) / GB:.1f}/{total / GB:.1f} GB")
    gpu = gpu_memory_mib()
    if gpu is not None:
        parts.append(f"GPU {gpu / 1024:.1f} GiB used")
    return " | ".join(parts)


def gpu_memory_mib() -> float | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15, check=False)
        values = [float(x) for x in r.stdout.split()]
        return sum(values) if values else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def gpu_pids() -> list[int]:
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15, check=False)
        return [int(x) for x in r.stdout.split() if x.strip().isdigit()]
    except (OSError, subprocess.SubprocessError):
        return []


def _terminate(proc, grace_s: float = 30.0) -> None:
    """SIGTERM to the process group, then SIGKILL (every command runs in its own session)."""
    for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 15.0)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(2)
        return s.connect_ex((host, int(port))) == 0


class Session:
    """The state of one session-A notebook (see the module docstring); ``config`` comes from the builder."""

    def __init__(self, working, config: dict, *, started: float | None = None, echo=None):
        self.working = Path(working)
        self.cfg = config
        self.started = started if started is not None else time.time()
        self.deadline = self.started + float(config.get("session_hours", 8)) * 3600
        self.scripts = Path(config["scripts_dir"])
        self.logs = self.working / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.state_path = self.working / "session-a.json"
        self.python = sys.executable
        self.echo = echo or (lambda message: print(message, flush=True))
        self.census_paths = tuple(config.get("census_paths") or ("/", str(self.working), "/tmp", "/dev/shm"))
        self.census = lambda: default_census(self.census_paths)
        self.poll_s = 2.0
        self.server = None
        self.server_kind = None
        self.launcher: dict = {}
        self.inputs: dict = {}
        self.storage_plan: dict = {}
        self._current = None
        self._last_output = ""
        self.state = {"format": "arc3-session-a", "version": 1, "started_utc": _utc(self.started),
                      "deadline_utc": _utc(self.deadline), "build": config.get("build", {}), "verdict": "running",
                      "stopped_at": None, "reason": None, "exit_code": None, "warnings": [], "steps": []}
        self.save()

    # ------------------------------------------------------------------------------------------ bookkeeping

    def save(self) -> None:
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, default=str) + "\n")
        tmp.replace(self.state_path)

    def warn(self, message: str) -> None:
        self.state["warnings"].append(message)
        self.save()
        self.echo(f"[session A] WARNING: {message}")

    def script(self, name: str) -> str:
        return str(self.scripts / name)

    def elapsed_min(self) -> float:
        return (time.time() - self.started) / 60

    def budget_s(self, minutes: float) -> float:
        """A step's budget, cut to what is left before the session deadline."""
        left = self.deadline - time.time()
        if left <= 60:
            raise self.fail(self._current["step"] if self._current else "?",
                            f"the session deadline ({self.cfg.get('session_hours', 8)} h) has passed")
        return min(minutes * 60.0, left)

    def fail(self, step: str, message: str, code: int = 1) -> Stop:
        """Record why the session stops here (stopping a running server first); returns the exception to raise."""
        if self.server is not None:
            try:
                self.stop_server(step, raising=False)
            except Exception as exc:  # the stop below matters more
                message += f" (stopping the server failed too: {exc!r})"
        if self.state["verdict"] in ("no-go", "failed"):  # already stopped: keep the first stop as the reason
            self.state["warnings"].append(f"{step} after the stop at {self.state['stopped_at']}: {message}")
        else:
            self.state.update(verdict="no-go" if code == 2 else "failed", stopped_at=step, reason=message,
                              exit_code=code, finished_utc=_utc())
        self._record_working()
        self.save()
        self.echo(f"[session A] STOP at {step}: {message}")
        return Stop(f"session A stops at {step}: {message} (reports: {self.state_path})", code)

    def _record_working(self) -> int:
        """What /kaggle/working holds now (bytes per entry, total), into the state."""
        sizes = {}
        for p in sorted(self.working.iterdir()):
            with contextlib.suppress(OSError):
                sizes[p.name] = dir_bytes(p) if p.is_dir() and not p.is_symlink() else os.lstat(p).st_size
        total = sum(sizes.values())
        self.state.update(working_bytes=sizes, working_total_gb=round(total / GB, 2))
        return total

    def go(self, step: str) -> bool:
        """Whether the cell of STEP runs: after a NO-GO at A5 every later step is skipped (the notebook ends
        normally, with session-a.json saying why); after a failure the notebook has already stopped."""
        if self.state["verdict"] == "no-go":
            self.echo(f"[session A {step}] skipped: the session stopped at {self.state['stopped_at']} "
                      f"({self.state['verdict']}, exit {self.state['exit_code']})")
            return False
        if self.state["verdict"] == "failed":
            raise Stop(f"session A stopped at {self.state['stopped_at']}: {self.state['reason']}", 1)
        return True

    def replica_gate(self) -> bool:
        """A5 as the notebook runs it: the replica check, which on NO-GO (exit 2) ends the session cleanly (reports
        written, no server running, nothing later runs) instead of failing the notebook. True on GO."""
        try:
            self.replica_check("A5")
        except Stop as exc:
            if exc.code != 2:
                raise
            self.finish()
            self.echo("[session A] NO-GO: the replica does not reproduce SGLang's accept counts, so nothing is "
                      "trained in this session (plan 6.6 stop rule). Read replica-check.json (section 10.3, "
                      "'Reading a failure').")
            return False
        return True

    @contextlib.contextmanager
    def step(self, step: str, title: str):
        entry = {"step": step, "title": title, "started_utc": _utc(), "minutes_into_session": round(self.elapsed_min(), 1),
                 "commands": [], "census_start": self._census()}
        self.state["steps"].append(entry)
        self._current = entry
        self.save()
        self.echo(f"\n[session A {step}] {title} ({entry['started_utc']} UTC, {entry['minutes_into_session']:.0f} min "
                  f"into the session)\n[session A {step}] {entry['census_start']}")
        t0 = time.time()
        try:
            yield entry
            entry["status"] = "ok"
        except Stop:
            entry["status"] = "stopped"
            raise
        except BaseException as exc:
            entry["status"] = f"error: {exc!r}"[:500]
            if self.state["verdict"] == "running":
                self.state.update(verdict="failed", stopped_at=step, reason=f"{exc!r}"[:1000], exit_code=1)
            if self.server is not None:
                with contextlib.suppress(Exception):
                    self.stop_server(step, raising=False)
            raise
        finally:
            entry["seconds"] = round(time.time() - t0, 1)
            entry["census_end"] = self._census()
            self._current = None
            self.save()
            self.echo(f"[session A {step}] {entry.get('status')} after {entry['seconds'] / 60:.1f} min; {entry['census_end']}")

    def _census(self) -> str:
        try:
            return str(self.census())
        except Exception as exc:
            return f"census failed: {exc!r}"

    def _pump(self, stream, logf, keep: list) -> None:
        for line in iter(stream.readline, b""):
            logf.write(line)
            text = line.decode("utf-8", "replace").rstrip("\n")
            keep.append(text)
            self.echo(text)
        stream.close()

    def run(self, step: str, argv, *, env: dict | None = None, minutes: float, ok=(0,), watch=None,
            name: str | None = None, check: bool = True) -> dict:
        """Run one command; its output goes to the notebook log and to logs/<name>.log. Past its budget (or the
        session deadline) it is stopped (SIGTERM, then SIGKILL to its process group), and so it is when ``watch``
        (called with the elapsed seconds) returns a reason. With ``check`` an exit code outside ``ok``, or a stop,
        raises :class:`Stop`."""
        name = name or step
        budget = self.budget_s(minutes)
        argv = [str(a) for a in argv]
        rec = {"name": name, "argv": argv, "log": f"logs/{name}.log", "budget_s": round(budget), "started_utc": _utc()}
        if self._current is not None:
            self._current["commands"].append(rec)
        self.echo(f"[session A {step}] $ {shlex.join(argv)}")
        keep: list[str] = []
        t0 = time.time()
        with open(self.logs / f"{name}.log", "ab", buffering=0) as logf:
            logf.write(f"$ {shlex.join(argv)}\n".encode())
            proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            pump = threading.Thread(target=self._pump, args=(proc.stdout, logf, keep), daemon=True)
            pump.start()
            stopped = None
            while proc.poll() is None:
                elapsed = time.time() - t0
                if elapsed > budget:
                    stopped = f"its budget of {budget / 60:.1f} min ran out"
                elif watch is not None:
                    stopped = watch(elapsed)
                if stopped:
                    self.echo(f"[session A {step}] stopping {name}: {stopped}")
                    _terminate(proc)
                    break
                time.sleep(self.poll_s)
            proc.wait()
            pump.join(timeout=30)
        self._last_output = "\n".join(keep)
        rec.update(exit=proc.returncode, seconds=round(time.time() - t0, 1), stopped=stopped, finished_utc=_utc())
        self.save()
        self.echo(f"[session A {step}] {name}: exit {proc.returncode} after {rec['seconds'] / 60:.1f} min"
                  + (f" (stopped: {stopped})" if stopped else ""))
        if check and (stopped or proc.returncode not in ok):
            raise self.fail(step, f"{name} " + (f"was stopped ({stopped})" if stopped else
                                                f"exited with {proc.returncode}") + f"; see {rec['log']}")
        return rec

    # ---------------------------------------------------------------------------------------------- inputs

    def find_inputs(self, *, wait_s: float | None = None, root: str = INPUT_ROOT, poll_s: float = 5.0) -> dict:
        """Resolve every input of this session (``config["inputs"]``) in either mount layout, checked against its
        pinned sha256 or sizes; waits up to ``wait_s`` for mounts, then stops (listing /kaggle/input)."""
        specs = self.cfg["inputs"]
        wait_s = float(self.cfg.get("wait_inputs_s", 300) if wait_s is None else wait_s)
        with self.step("inputs", "find this session's inputs (both mount layouts) and check them"):
            t0 = time.time()
            while True:
                found, missing = {}, []
                for key, spec in specs.items():
                    try:
                        hit = locate(spec, root)
                    except InputError as exc:
                        raise self.fail("inputs", str(exc)) from None
                    if hit is None:
                        missing.append(f"{key} ({spec['kind']} {spec['id']})")
                    else:
                        found[key] = hit
                if not missing:
                    break
                if time.time() - t0 >= wait_s:
                    raise self.fail("inputs", f"not mounted after {wait_s:g} s: {missing}; under {root}: "
                                              + "; ".join(input_listing(root)))
                time.sleep(poll_s)
            for key, hit in found.items():
                labels = specs[key].get("labels") or {}
                versions = sorted({labels[v] for k, v in hit.items() if k.endswith(".sha256") and v in labels})
                hit["versions"] = versions
                self.echo(f"[session A inputs] {key}: {hit['dir']} ({specs[key]['kind']} {specs[key]['id']}), "
                          f"{len(specs[key].get('files') or {})} file(s) checked"
                          + (f"; {', '.join(versions)}" if versions else ""))
            self.inputs = found
            self.state["inputs"] = found
        return found

    def input_file(self, key: str, name: str) -> str:
        return self.inputs[key][name]

    # --------------------------------------------------------------------------------------------- A0, A1

    def storage(self) -> dict:
        cfg = self.cfg["storage"]
        with self.step("A0", "storage test: write to the candidate dump directories; place and cap the dumps"):
            tests = [write_test(d, gb=cfg["test_gb"], seconds=cfg["test_seconds"], echo=self.echo)
                     for d in cfg["candidates"]]
            mem = meminfo()
            plan = plan_storage(tests, mem_total_gb=mem.get("MemTotal", 0) / GB, probe_gb=cfg["probe_gb"],
                                train_gb=cfg["train_gb"], holdout_gb=cfg["holdout_gb"], min_train_gb=cfg["min_train_gb"],
                                server_ram_gb=cfg["server_ram_gb"], margin_gb=cfg["margin_gb"])
            self.state["storage"] = {"tests": tests, "plan": plan, "mem_total_gb": round(mem.get("MemTotal", 0) / GB, 1),
                                     "mem_available_gb": round(mem.get("MemAvailable", 0) / GB, 1)}
            if plan["error"]:
                raise self.fail("A0", plan["error"])
            plan["probe_dump"] = str(Path(plan["probe_dir"]) / "arc3-hc-probe")
            plan["train_dump"] = str(Path(plan["train_dir"]) / "arc3-hc-train")
            self.storage_plan = plan
            self.save()
            self.echo(f"[session A A0] probe dump -> {plan['probe_dump']} ({plan['probe_gb']} GB), training dump -> "
                      f"{plan['train_dump']} ({plan['train_gb']} GB, holdout part {plan['holdout_gb']} GB)")
        return plan

    def set_launcher(self, *, args, env, log, port, model_name, python_torch, venv, model_dir, draft_dir, generic_map,
                     precache_thread=None) -> None:
        """What cell 12 built: its server arguments and environment (the server it would have started), its log,
        the venv's python and the model paths; albucino's directory is found as its prepare_draft_view finds it."""
        self.launcher = {"args": [str(a) for a in args], "env": {k: str(v) for k, v in env.items()}, "log": str(log),
                         "port": int(port), "model": str(model_name), "python_torch": str(python_torch),
                         "venv": str(venv), "model_dir": str(model_dir), "draft_dir": str(draft_dir),
                         "generic_map": str(generic_map), "precache_thread": precache_thread}
        try:
            self.launcher["albucino"] = str(draft_source(draft_dir))
        except ValueError as exc:
            raise self.fail("A1", str(exc)) from None
        want = (self.cfg.get("probe") or {}).get("generic_map_sha256")
        if want:  # the map the reference run served with (cell 12's own assert checks the same file)
            got = sha256_file(generic_map) if Path(generic_map).is_file() else None
            if got != want:
                raise self.fail("A1", f"{generic_map}: sha256 {got} is not the generic FR-Spec map the reference run "
                                      f"served with ({want[:12]}...)")
        self.state["launcher"] = {k: v for k, v in self.launcher.items() if k not in ("env", "precache_thread")}
        self.state["launcher"]["env_keys"] = sorted(self.launcher["env"])
        self.save()

    def torch_env(self) -> dict:
        """The launcher's environment (CUDA, caches, libraries) for the venv's torch scripts, without dump or REAP
        variables."""
        return {k: v for k, v in self.launcher["env"].items() if not k.startswith(DUMP_ENV) and k != REAP_ENV}

    def apply_dump_patch(self) -> None:
        reap = bool(self.cfg["server"]["reap"])
        with self.step("A1", "install the hidden-state dump patch (after REAP when REAP is served)"):
            models = sorted(Path(self.launcher["venv"]).glob("lib/python*/site-packages/sglang/srt/models/qwen4_exp.py"))
            if len(models) != 1:
                raise self.fail("A1", f"expected one installed sglang qwen4_exp.py, found {models}")
            self.run("A1", [self.python, "-I", "-u", self.script("sglang_hc_dump_patch.py"), "apply",
                            "--site-packages", models[0].parents[3]], minutes=self.cfg["budgets_min"]["A1"])
            want = "(base: the wheel's file + scripts/sglang_reap_patch.py)" if reap else "(base: the wheel's file)"
            if want not in self._last_output:
                raise self.fail("A1", f"the dump patch did not report {want!r}: REAP must be applied first exactly when "
                                      "the training dump serves it")

    # ------------------------------------------------------------------------------------------- servers

    def start_server(self, step: str, kind: str) -> None:
        """A2 (kind "probe") or A6 (kind "train"): start a dump server from the launcher's args and env (see
        :func:`dump_server_args`), wait for /health, check /server_info."""
        if kind not in ("probe", "train"):
            raise ValueError(kind)
        reap = bool(self.cfg["server"]["reap"]) and kind == "train"
        plan = self.storage_plan
        dump_dir = Path(plan[f"{kind}_dump"])
        with self.step(step, f"start the {kind} dump server"):
            if kind == "train":
                self.require_go(step)
            if self.server is not None:
                raise self.fail(step, "a server is already running")
            if dump_dir.exists() and any(dump_dir.iterdir()):
                raise self.fail(step, f"{dump_dir} is not empty")
            dump_dir.mkdir(parents=True, exist_ok=True)
            if kind == "probe":
                dump_env = {DUMP_ENV: dump_dir, DUMP_ENV + "_KEEP": "all", DUMP_ENV + "_DTYPE": "bf16",
                            DUMP_ENV + "_MAX_GB": plan["probe_gb"]}
            else:
                dump_env = {DUMP_ENV: dump_dir, DUMP_ENV + "_MAX_GB": plan["train_gb"]}
                if int(self.cfg["dump"]["scale_groups"]) != 1:
                    dump_env[DUMP_ENV + "_SCALE_GROUPS"] = self.cfg["dump"]["scale_groups"]
            try:
                args = dump_server_args(self.launcher["args"], reap=reap)
                env = dump_server_env(self.launcher["env"], dump_env, reap=reap)
            except ValueError as exc:
                raise self.fail(step, str(exc)) from None
            record = {"kind": kind, "reap": reap, "dump_dir": str(dump_dir), "args": args,
                      "diff": args_diff(self.launcher["args"], args),
                      "env": {k: v for k, v in env.items() if k.startswith((DUMP_ENV, REAP_ENV))}}
            self.state.setdefault("servers", {})[step] = record
            self.save()
            self.launch(step, args, env)
            record["healthy_after_s"] = self.wait_healthy(step)
            info = self.server_info()
            expect = self.cfg["server"]["num_experts"] if reap else None
            record["server_info"] = info
            record["problems"] = server_problems(info, num_experts=expect)
            record["notes"] = server_notes(info)
            self.save()
            for note in record["notes"]:
                self.warn(f"{step}: {note}")
            if record["problems"]:
                raise self.fail(step, "the dump server is not configured as the dump needs: " + "; ".join(record["problems"]))
            self.server_kind = kind

    def launch(self, step: str, args: list, env: dict) -> None:
        """Start the server as cell 12's launch section does (same log, same Popen arguments); its model precache
        thread is started with the first server."""
        port, log = self.launcher["port"], Path(self.launcher["log"])
        if port_open(port):
            raise self.fail(step, f"port {port} is occupied; stop the existing server first")
        thread = self.launcher.get("precache_thread")
        if thread is not None and getattr(thread, "ident", 1) is None:
            thread.start()
        self.echo("\n" + shlex.join(args) + "\n")
        log_offset = log.stat().st_size if log.exists() else 0
        with open(log, "ab", buffering=0) as logf:
            proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        self.echo(f"pid {proc.pid} -> {log}")
        self.server, self.server_log_offset = proc, log_offset
        self.state["servers"][step].update(pid=proc.pid, log_offset=log_offset, started_utc=_utc())

    def log_tail(self, n_bytes: int = 20000) -> str:
        log = Path(self.launcher["log"])
        try:
            with open(log, "rb") as f:
                f.seek(0, 2)
                f.seek(max(getattr(self, "server_log_offset", 0), f.tell() - n_bytes))
                return f.read().decode(errors="replace")
        except OSError as exc:
            return f"(no server log: {exc})"

    def log_size(self) -> int:
        try:
            return Path(self.launcher["log"]).stat().st_size
        except OSError:
            return -1

    def load_progress(self) -> str:
        """The weight loader's last shard count in the server log ("weights 29/38 shards"), or ""."""
        done = SHARDS_RE.findall(self.log_tail(4000))
        return f"weights {done[-1][0]}/{done[-1][1]} shards" if done else ""

    def wait_healthy(self, step: str) -> float:
        """Wait for /health. budgets_min.boot stops a server whose log has not grown for boot_stall minutes; one
        still writing to its log (shard progress, graph capture) may take up to boot_max. Weights load from
        /kaggle/input, which can be several times slower than usual: session A v1 (2026-10-10) read 26-53 s per
        shard instead of 5-9 and was stopped at 29 of 38 shards by a fixed 20-minute limit."""
        b = self.cfg["budgets_min"]
        url = f"http://127.0.0.1:{self.launcher['port']}/health"
        t0 = last = grew = time.time()
        quiet_stops = t0 + self.budget_s(b["boot"])
        cap = t0 + self.budget_s(max(b["boot"], b["boot_max"]))
        size = self.log_size()
        while True:
            if self.server.poll() is not None:
                self.echo(self.log_tail())
                raise self.fail(step, f"the server exited with code {self.server.returncode} before it was healthy "
                                      "(serve.log has the reason)")
            try:
                status, _ = _http_json(url, timeout=5)
                if status == 200:
                    self.echo(f"[session A {step}] server healthy after {time.time() - t0:.0f} s")
                    return round(time.time() - t0, 1)
            except (OSError, urllib.error.URLError, ValueError):
                pass
            now, new = time.time(), self.log_size()
            if new != size:
                size, grew = new, now
            quiet = now - grew > b["boot_stall"] * 60
            if now > cap or (now > quiet_stops and quiet):
                self.echo(self.log_tail())
                why = [f"its log had not grown for {(now - grew) / 60:.0f} min" if quiet else
                       f"still loading at the {b['boot_max']}-minute limit", self.load_progress()]
                raise self.fail(step, f"the server was not healthy after {(now - t0) / 60:.0f} min "
                                      f"({'; '.join(x for x in why if x)})")
            if now - last >= 30:
                progress = self.load_progress()
                self.echo(f"  {now - t0:.0f}s ..." + (f" ({progress})" if progress else ""))
                last = now
            time.sleep(min(3.0, self.poll_s))

    def server_info(self) -> dict:
        try:
            status, raw = _http_json(f"http://127.0.0.1:{self.launcher['port']}/server_info", timeout=60)
        except (OSError, urllib.error.URLError) as exc:
            return {"error": repr(exc)}
        if status != 200:
            return {"error": f"HTTP {status}"}
        try:
            info = json.loads(raw)
        except ValueError:
            return {"error": "not JSON"}
        return info if isinstance(info, dict) else {"error": "not an object"}

    def stop(self, step: str) -> None:
        """A4: stop the dump server as a step of its own."""
        with self.step(step, "stop the dump server"):
            self.stop_server(step)

    def require_go(self, step: str) -> None:
        """Every step after A5 runs only after a GO, also when its cell is run by hand after a NO-GO."""
        check = self.state.get("replica_check") or {}
        if check.get("exit") != 0:
            raise self.fail(step, f"the replica check (A5) has not passed (exit {check.get('exit')})",
                            code=2 if check.get("exit") == 2 else 1)

    def stop_server(self, step: str, *, raising: bool = True) -> None:
        """Stop the server's process group; wait until its GPU memory and port are free."""
        proc = self.server
        if proc is None:
            return
        self.server = None
        self.echo(f"[session A {step}] stopping the {self.server_kind} dump server (pid {proc.pid})")
        _terminate(proc, grace_s=60.0)
        t0 = time.time()
        while _group_alive(proc.pid) and time.time() - t0 < 60:
            time.sleep(2)
        if _group_alive(proc.pid):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
        limit = float(self.cfg["server"].get("gpu_free_mib", 3000))
        t0, killed = time.time(), []
        while True:
            used = gpu_memory_mib()
            if used is None or used < limit:
                break
            if time.time() - t0 > 120 and not killed:
                killed = [pid for pid in gpu_pids() if pid != os.getpid()]
                for pid in killed:
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        os.kill(pid, signal.SIGKILL)
                self.warn(f"{step}: GPU memory still in use 2 min after the server stopped; killed {killed}")
            if time.time() - t0 > 240:
                message = f"the GPU still holds {used:.0f} MiB 4 min after the server stopped"
                if raising:
                    raise self.fail(step, message)
                self.echo(f"[session A {step}] {message}")
                break
            time.sleep(3)
        t0 = time.time()
        while port_open(self.launcher["port"]) and time.time() - t0 < 60:
            time.sleep(2)
        server = self.state.get("servers", {})
        for record in server.values():
            if record.get("pid") == proc.pid:
                record["stopped_utc"] = _utc()
                record["exit"] = proc.poll()
        self.server_kind = None
        self.save()

    # ------------------------------------------------------------------------------------------- A3, A5, A10

    def probe_dump(self) -> dict:
        cfg = self.cfg["probe"]
        out = self.working / "probe"
        with self.step("A3", "dump the held-out probe requests plus the reference's greedy outputs"):
            max_rows = int(min(cfg["max_rows"], self.storage_plan["probe_gb"] * GB / BF16_ROW_BYTES))
            self.run("A3", [self.python, "-I", "-u", self.script("mtp_probe_dump.py"),
                            "--data", self.input_file("prompts", "requests.jsonl"),
                            "--reference", self.input_file("reference", "fidelity.json"), "--pass", cfg["pass"],
                            "--count", cfg["count"], "--max-rows", max_rows, "--out", out,
                            "--dump-dir", self.storage_plan["probe_dump"], "--base-url", self.base_url(),
                            "--model", self.launcher["model"], "--health-wait", 300],
                     minutes=self.cfg["budgets_min"]["A3"], ok=(0, 1))  # 1: some request was not dumped completely
            path = out / "probe-dump-summary.json"
            if not path.is_file():
                raise self.fail("A3", f"{path.name} was not written; see logs/A3.log")
            summary = json.loads(path.read_text())
            result = {k: summary.get(k) for k in ("sent", "ok", "dump_ok", "stopped")}
            self.state["probe_dump"] = result
            self.save()
            if (summary.get("dump_ok") or 0) < cfg["min_requests"]:
                raise self.fail("A3", f"only {summary.get('dump_ok')} of {summary.get('sent')} probe requests dumped "
                                      f"completely (the replica check needs {cfg['min_requests']}; "
                                      f"{summary.get('stopped') or 'see probe/probe-dump.jsonl'})")
            if summary.get("dump_ok") != summary.get("sent"):
                self.warn(f"A3: {summary.get('sent') - summary.get('dump_ok')} probe request(s) not dumped "
                          "completely; the replica check leaves them out")
        return result

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.launcher['port']}"

    def replica_check(self, step: str, *, trained: bool = False) -> dict:
        """A5 (the gate: exit 2 = NO-GO stops the session) or A10 (``trained``: the forecast; exit 0 and 2 are
        both results)."""
        out = self.working / ("replica-check-trained.json" if trained else "replica-check.json")
        title = ("replica check of the trained dense weights (forecast of the probe gate)" if trained else
                 "replica check: the replica's greedy chain against SGLang's accept counts (go/no-go)")
        with self.step(step, title):
            if self.server is not None:
                raise self.fail(step, "a server is still running (the check needs the GPU)")
            argv = [self.launcher["python_torch"], "-I", "-u", self.script("mtp_replica.py"), "check",
                    "--draft", self.launcher["albucino"], "--token-map", self.launcher["generic_map"],
                    "--dump", self.storage_plan["probe_dump"], "--probe", self.working / "probe" / "probe-dump.jsonl",
                    "--reference", self.input_file("reference", "fidelity.json"), "--pass", self.cfg["probe"]["pass"],
                    "--out", out]
            if trained:
                self.require_go(step)
                dense = self.working / "mtp-train" / "trained-dense.safetensors"
                if not dense.is_file():
                    raise self.fail(step, f"{dense} does not exist")
                argv += ["--trained", dense]
            rec = self.run(step, argv, env=self.torch_env(), minutes=self.cfg["budgets_min"][step], ok=(0, 2))
            if not out.is_file():
                raise self.fail(step, f"{out.name} was not written")
            result = json.loads(out.read_text())
            summary = {"exit": rec["exit"], "verdict": result.get("verdict"), "gate_variant": result.get("gate_variant"),
                       "summary": result.get("summary"), "embed_mode": result.get("embed_mode")}
            self.state["replica_check_trained" if trained else "replica_check"] = summary
            self.save()
            if not trained and rec["exit"] == 2:
                reasons = (result.get("verdict") or {}).get("reasons")
                raise self.fail(step, f"replica check NO-GO (exit 2): {reasons}; no training in this session (plan 6.6 "
                                      "stop rule)", code=2)
        return summary

    # ------------------------------------------------------------------------------------------------- A7

    def dump_training(self) -> dict:
        cfg, plan = self.cfg["dump"], self.storage_plan
        logs = self.inputs["logs"]["dir"]
        common = ["--dump-dir", plan["train_dump"], "--base-url", self.base_url(), "--model", self.launcher["model"],
                  "--health-wait", 300]
        driver = self.script("hc_dump_driver.py")
        result = {"holdout": {}, "train": None}
        with self.step("A7", "dump the training data: one snapshot per held-out game, then the train split"):
            self.require_go("A7")
            self._server_alive("A7")
            # The held-out games share one time budget and a size cap. Both caps are cumulative over the dump
            # directory: the driver's --max-dump-gb compares the server's running byte count, and before each game
            # the dump's index says how much the earlier games wrote.
            t0 = time.time()
            for game in cfg["holdout_games"]:
                left_min = cfg["holdout_minutes"] - (time.time() - t0) / 60
                written_gb = index_stats(plan["train_dump"])["bytes"] / GB
                if left_min < 1 or written_gb >= plan["holdout_gb"]:
                    self.warn(f"A7: held-out games from {game} on skipped ({written_gb:.2f} GB written of "
                              f"{plan['holdout_gb']} GB, {max(left_min, 0):.0f} of {cfg['holdout_minutes']} min left)")
                    break
                out = self.working / "hc-holdout" / game
                files = sorted(Path(logs).glob(f"{game}-*_p*_requests.jsonl")) or [Path(logs)]
                self.run("A7", [self.python, "-I", "-u", driver, *[x for f in files for x in ("--logs", f)],
                                "--split", "holdout", "--games", game,
                                "--max-snapshots", cfg["holdout_snapshots_per_game"], "--out", out,
                                "--max-dump-gb", plan["holdout_gb"], "--max-minutes", round(left_min, 1), *common],
                         name=f"A7-holdout-{game}", minutes=left_min + 5, ok=(0, 1))
                result["holdout"][game] = self._replay_summary(out)
                self._server_alive("A7")
            if not any((s or {}).get("dump_ok") for s in result["holdout"].values()):
                raise self.fail("A7", "no held-out snapshot was dumped completely (is the server dumping?)")
            out = self.working / "hc-train"
            self.run("A7", [self.python, "-I", "-u", driver, "--logs", logs, "--split", "train", "--out", out,
                            "--max-dump-gb", round(plan["train_gb"] - 0.5, 2), "--max-minutes", cfg["minutes"], *common],
                     name="A7-train", minutes=cfg["minutes"] + 20, ok=(0, 1))
            result["train"] = self._replay_summary(out)
            if not (result["train"] or {}).get("dump_ok"):
                raise self.fail("A7", "no train snapshot was dumped completely")
            if (result["train"] or {}).get("failed"):
                self.warn("A7: the train dump stopped on failed requests (" + str(result["train"].get("stopped"))
                          + "); training uses what was dumped")
            result["index"] = index_stats(plan["train_dump"])
            self.state["dump"] = result
            self.save()
            q = result["index"]
            self.echo(f"[session A A7] dump: {q['requests']} requests, {q['kept_rows']:,} rows ({q['span_rows']:,} span "
                      f"rows), {q['bytes'] / GB:.2f} GB; qerr_stream_max max {q['qerr_stream_max']['max']} p99 "
                      f"{q['qerr_stream_max']['p99']}; nonfinite rows {q['nonfinite_rows']}; errors {q['errors']}")
            if q["nonfinite_rows"]:
                self.warn(f"A7: {q['nonfinite_rows']} dumped rows hold NaN/Inf")
            p99 = q["qerr_stream_max"]["p99"]
            if p99 is not None and p99 > cfg.get("qerr_warn", 0.05) and int(cfg["scale_groups"]) == 1:
                self.warn(f"A7: qerr_stream_max p99 {p99} > {cfg.get('qerr_warn', 0.05)}: one FP8 scale per row loses "
                          "a weak stream; the next dump should use scale_groups 4 (ARC3_HC_DUMP_SCALE_GROUPS=4)")
        return result

    def _replay_summary(self, out: Path) -> dict | None:
        path = Path(out) / "replay-summary.json"
        if not path.is_file():
            return None
        s = json.loads(path.read_text())
        return {k: s.get(k) for k in ("planned", "sent", "ok", "dump_ok", "prefill_tokens", "dump_bytes", "stopped",
                                      "failed", "elapsed_s", "prefill_tok_s")}

    def _server_alive(self, step: str) -> None:
        if self.server is None or self.server.poll() is not None:
            self.echo(self.log_tail())
            raise self.fail(step, "the dump server is gone")

    # ------------------------------------------------------------------------------------------- A8, A9

    def snapshot_args(self) -> list[str]:
        files = [*sorted((self.working / "hc-holdout").glob("*/snapshots.jsonl")),
                 self.working / "hc-train" / "snapshots.jsonl"]
        out = []
        for path in files:
            if path.is_file():
                out += ["--snapshots", str(path)]
        return out

    def train_plan(self) -> dict:
        with self.step("A8", "stop the server; plan the training windows"):
            self.require_go("A8")
            self.stop_server("A8")
            self.run("A8", [self.launcher["python_torch"], "-I", "-u", self.script("mtp_train.py"), "plan",
                            "--dump", self.storage_plan["train_dump"], *self.snapshot_args(),
                            *self.cfg["train"].get("plan_args", [])],
                     env=self.torch_env(), minutes=self.cfg["budgets_min"]["A8"])
            info = last_json_object(self._last_output)
            if info is None:
                self.warn("A8: could not read the plan's JSON; training runs without the step-time projection")
                info = {}
            self.state["train_plan"] = info
            self.save()
            if info and not (info.get("train") or {}).get("windows"):
                raise self.fail("A8", f"no training windows in the dump: {info}")
        return info

    def train(self) -> dict:
        cfg = self.cfg["train"]
        out = self.working / "mtp-train"
        result: dict = {}
        with self.step("A9", "train the MTP draft's dense weights"):
            self.require_go("A9")
            if self.server is not None:
                raise self.fail("A9", "a server is still running (training needs the GPU)")
            if cfg.get("map"):  # the map the notebook writes (sha256 known at build time)
                amap = Path(cfg["map"])
                if not amap.is_file() or sha256_file(amap) != cfg["map_sha256"]:
                    raise self.fail("A9", f"{amap} is missing or not the map this notebook was built with")
            else:  # the wheelhouse's generic map, checked by cell 12's sha256 assert in A1
                amap = Path(self.launcher["generic_map"])
            left = self.deadline - time.time() - cfg["reserve_minutes"] * 60
            budget_min = min(cfg["minutes"], left / 60)
            if budget_min < cfg["min_minutes"]:
                raise self.fail("A9", f"{budget_min:.0f} min left for training (< {cfg['min_minutes']})")
            argv = [self.launcher["python_torch"], "-I", "-u", self.script("mtp_train.py"), "train",
                    "--draft", self.launcher["albucino"], "--target-dir", self.launcher["model_dir"],
                    "--token-map", amap, "--dump", self.storage_plan["train_dump"], *self.snapshot_args(),
                    "--checkpoint-every", cfg["checkpoint_every"], *cfg.get("args", [])]
            total = (self.state.get("train_plan") or {}).get("total_steps")
            watch = TrainWatch(out, total, budget_min * 60, probe_steps=cfg["probe_steps"], min_steps=cfg["min_steps"])
            t0 = time.time()
            rec = self.run("A9", [*argv, "--out", out], env=self.torch_env(), minutes=budget_min + cfg["grace_minutes"],
                           watch=watch, check=False, name="A9-train")
            result["projection"] = watch.info
            if watch.restart_steps is not None and rec["stopped"]:  # stopped by the watch, not finished
                first = self.working / "mtp-train-attempt1"
                if first.exists():
                    shutil.rmtree(first)
                out.rename(first)
                remaining = budget_min - (time.time() - t0) / 60
                result["restarted_with_max_steps"] = watch.restart_steps
                rec = self.run("A9", [*argv, "--out", out, "--max-steps", watch.restart_steps], env=self.torch_env(),
                               minutes=remaining + cfg["grace_minutes"], check=False, name="A9-train-restart")
            dense, report = out / "trained-dense.safetensors", out / "train-report.json"
            result.update(exit=rec["exit"], stopped=rec["stopped"], completed=rec["exit"] == 0 and report.is_file(),
                          trained_dense=dense.is_file())
            if report.is_file():
                rep = json.loads(report.read_text())
                result.update({k: rep.get(k) for k in ("windows", "total_steps", "steps_done", "train_seconds",
                                                       "comparison")})
            self.state["train"] = result
            self.save()
            if not dense.is_file():
                raise self.fail("A9", f"training left no {dense.name} (exit {rec['exit']}, {rec['stopped'] or 'not stopped'})")
            if not result["completed"]:
                self.warn(f"A9: training did not complete (exit {rec['exit']}, {rec['stopped']}); A10 and A11 use its "
                          "last checkpoint")
        return result

    # ------------------------------------------------------------------------------------------- A11, end

    def write_draft(self) -> dict:
        out = self.working / "mtp-draft"
        with self.step("A11", "write the fine-tuned draft directory (albucino's files, dense tensors replaced)"):
            self.require_go("A11")
            need = dir_bytes(self.launcher["albucino"])  # the writer copies albucino's files (~4.1 GB)
            limit = float(self.cfg.get("working_limit_gb", 19.5)) * GB
            have = self._record_working()
            if have + need > limit - 0.5 * GB:
                raise self.fail("A11", f"/kaggle/working holds {have / GB:.2f} GB; the draft directory needs "
                                       f"{need / GB:.2f} GB more, past Kaggle's {limit / GB:g} GB output limit "
                                       "(trained-dense.safetensors is kept; write the draft in the next session)")
            self.run("A11", [self.python, "-I", "-u", self.script("mtp_write_draft.py"), "write",
                             "--draft", self.launcher["albucino"],
                             "--trained", self.working / "mtp-train" / "trained-dense.safetensors", "--out", out],
                     minutes=self.cfg["budgets_min"]["A11"])
            manifest = json.loads((out / "arc3-draft-manifest.json").read_text())
            result = {"dir": str(out), "replaced": len(manifest.get("replaced") or []),
                      "files": len(manifest.get("files") or {}), "bytes": dir_bytes(out)}
            self.state["draft"] = result
            self.save()
        return result

    def finish(self) -> dict:
        """The last cell (also run by :meth:`replica_gate` on NO-GO): stops a server left running, records what
        /kaggle/working holds and the verdict, and prints the summary."""
        if self.server is not None:
            self.stop_server("finish", raising=False)
        total = self._record_working()
        sizes = self.state["working_bytes"]
        self.state.update(finished_utc=_utc(), minutes=round(self.elapsed_min(), 1))
        if total > float(self.cfg.get("working_limit_gb", 19.5)) * GB * 0.95:
            self.warn(f"/kaggle/working holds {total / GB:.1f} GB, near Kaggle's limit")
        if self.state["verdict"] == "running":
            self.state["verdict"] = "complete"
        self.save()
        self.echo(f"[session A] {self.state['verdict']} after {self.state['minutes']:.0f} min; outputs "
                  f"{total / GB:.2f} GB: " + ", ".join(f"{k} {v / 1e6:.0f} MB" for k, v in sizes.items() if v > 1e6))
        for key in ("replica_check", "replica_check_trained"):
            s = (self.state.get(key) or {}).get("summary") or {}
            for variant, v in s.items():
                self.echo(f"  {key} {variant}: replica {v.get('replica_mean')} sglang {v.get('sglang_mean')} "
                          f"diff {v.get('mean_diff')} r {v.get('pearson')} (n={v.get('requests')})")
        if self.state.get("train", {}).get("comparison"):
            self.echo(f"  training: {json.dumps(self.state['train']['comparison'])}")
        return self.state


def dir_bytes(path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            with contextlib.suppress(OSError):
                total += os.lstat(os.path.join(root, name)).st_size
    return total


def last_json_object(text: str):
    """The last JSON object printed with json.dumps(..., indent=N) in TEXT (a line "{" to a line "}"), or None."""
    lines = text.splitlines()
    ends = [i for i, line in enumerate(lines) if line == "}"]
    for end in reversed(ends):
        for start in (i for i in range(end, -1, -1) if lines[i] == "{"):
            try:
                value = json.loads("\n".join(lines[start:end + 1]))
            except ValueError:
                continue
            if isinstance(value, dict):
                return value
    return None


def _percentile(values: list, q: float):
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))], 5)


def index_stats(dump_dir) -> dict:
    """Counts of a dump's index.jsonl: requests, rows, bytes, the FP8 error statistics and problem events."""
    path = Path(dump_dir) / "index.jsonl"
    lines = []
    if path.is_file():
        for text in path.read_text().splitlines():
            with contextlib.suppress(ValueError):
                lines.append(json.loads(text))
    chunks = [x for x in lines if "file" in x]
    qs = [float(x.get("qerr_stream_max") or 0.0) for x in chunks if x.get("kept")]
    qr = [float(x.get("qerr_max") or 0.0) for x in chunks if x.get("kept")]
    return {"requests": len({x["rid"] for x in chunks}), "chunks": len(chunks),
            "kept_rows": sum(int(x.get("kept") or 0) for x in chunks),
            "span_rows": sum(int(x.get("span_rows") or 0) for x in chunks),
            "bytes": sum(int(x.get("bytes") or 0) for x in chunks),
            "nonfinite_rows": sum(int(x.get("nonfinite_rows") or 0) for x in chunks),
            "qerr_stream_max": {"max": max(qs) if qs else None, "p99": _percentile(qs, 0.99),
                                "median": _percentile(qs, 0.5)},
            "qerr_max": {"max": max(qr) if qr else None, "median": _percentile(qr, 0.5)},
            "scale_groups": sorted({x.get("scale_groups") for x in chunks if "scale_groups" in x}),
            "errors": sum(1 for x in lines if x.get("event") == "error"),
            "stopped": next((x for x in lines if x.get("event") == "stopped"), None)}
