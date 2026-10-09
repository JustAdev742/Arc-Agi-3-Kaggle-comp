#!/usr/bin/env python
"""Build the session-A notebook of the MTP draft fine-tune (docs/research/beat-tufa/mtp-drafter-finetune.md, sections
10.4 and 11): a private Kaggle notebook that runs steps A0-A11 unattended through scripts/mtp_session_a.py.

    .venv/bin/python scripts/build_mtp_session.py --out DIR [--slug arc3-mtp-session-a]
    .venv/bin/python scripts/build_mtp_session.py --out DIR --logs-dataset scottmahony/arc3-exp073-request-logs
    .venv/bin/python scripts/build_mtp_session.py --out DIR --set storage.train_gb=30 --set train.minutes=60

The notebook is the public D' notebook (kaggle/dprime/, Franzen's Milestone 2 solution with a new slot priority,
Apache-2.0) up to its SGLang launcher, built with the helpers of scripts/build_franzen_nb.py:

- cells 0-11 as D' has them, with ``--input-fallback`` (every /kaggle/input path resolved in either mount layout,
  lesson 0030) and ``--wait-inputs`` (cell 4 waits for its inputs before copying the bundle);
- right after cell 4: our scripts (scripts/mtp_session_a.py and the eight it runs) written to /kaggle/arc3-mtp/ by
  ``--compact`` cells (zlib + base64, sha256 checked before writing), then a cell that loads the module, builds the
  session from the configuration below, finds and checks the inputs (``inputs``) and runs the storage test (A0);
- before the launcher: REAP-448's three files (``--reap-kept``, compact) and the ARC FR-Spec map (the
  ``--hot-tokens`` writer's encoding, written to /kaggle/arc3-hot-tokens.pt with a sha256 known here; A9 trains with
  it);
- cell 12 (the launcher) with REAP's two edits, cut right before ``# ---- launch detached and wait for health ----``:
  it installs Pennyroyal, patches sglang with REAP, validates the target and the draft view and builds ``args`` and
  ``env``, but starts no server. The FR-Spec lookup is unchanged, so ``tok`` is the wheelhouse's generic
  hot_tokens_64k.pt, checked by his own sha256 assert (the map the reference probe run served with: A5 and A10 use it);
- the cells of A1-A11, each one call of the module, and a last cell that writes the summary. Cells 13 and later of
  D' (the benchmark, the monitor, the diagnostics) are dropped.

Inputs (kernel metadata): D''s sources, the probe prompts dataset (kaggle/fidelity/: scottmahony/arc3-fidelity-prompts),
and two kernel outputs: scottmahony/arc3-fidelity-base (its fidelity.json, the replica check's reference; version 1 =
runs/fidelity-base or version 2 = runs/fidelity-base2, both sha256-pinned, whichever Kaggle mounts) and exp-073's
request logs (scottmahony/arc3-dprime-reap448-r14-full; 25 ``*_p0_requests.jsonl`` at the output's top level, byte
sizes pinned in kaggle/mtp/exp073-request-logs.json). ``--logs-dataset OWNER/SLUG`` takes the logs from a dataset
instead, and ``--reference-dataset OWNER/SLUG`` the reference's fidelity.json (the fallback if a kernel output cannot
be mounted: the owner downloads the files with ``kaggle kernels output ... --file-pattern`` and uploads them as a
private dataset with ``-t``; the same sizes and sha256 are checked; one dataset may hold both).

``--set KEY=VALUE`` (repeatable) changes one entry of the session configuration (dotted path, JSON value), e.g.
``--set storage.train_gb=30``. Writes ``<out>/<slug>.ipynb`` and ``<out>/kernel-metadata.json`` (private, internet
off, RTX PRO 6000, the pinned image).
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import pprint
import re
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_franzen_nb as bf  # noqa: E402

SCRIPTS_DIR = "/kaggle/arc3-mtp"
SCRIPTS = ("mtp_session_a.py", "sglang_hc_dump_patch.py", "hc_dump_driver.py", "fidelity_probe.py",
           "fidelity_sample.py", "mtp_probe_dump.py", "mtp_replica.py", "mtp_train.py", "mtp_write_draft.py")
MODULE = "mtp_session_a.py"
LAUNCH_CUT = "# ---- launch detached and wait for health ----\n"
SA_BEGIN = "# >>> ours (session A)"
SA_END = "# <<< ours (session A)"
MAP_BEGIN = "# >>> ours (session A, ARC FR-Spec map)"
MAP_END = "# <<< ours (session A, ARC FR-Spec map)"
ARC_MAP_FILE = bf.HOT_MAP_FILE  # /kaggle/arc3-hot-tokens.pt
DEFAULT_REAP = ROOT / "kaggle" / "franzen" / "reap448_kept_experts.json"
DEFAULT_MAP = ROOT / "kaggle" / "franzen" / "hot_tokens_64k_arc.pt"
PROBE_DIR = ROOT / "kaggle" / "fidelity"
LOGS_MANIFEST = ROOT / "kaggle" / "mtp" / "exp073-request-logs.json"
GENERIC_MAP_SHA256 = "becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25"  # D' cell 12 TOKEN_MAP_SHA
# The replica check's reference: the fidelity probe's base arm (D', Intel W4A16, no REAP, albucino draft, generic
# map, lossless, greedy). Its kernel has two versions; a kernel source mounts one of them (the latest, as far as we
# know), and both are runs of the same notebook, so either serves.
REFERENCE = {"kind": "kernel", "id": "scottmahony/arc3-fidelity-base",
             "files": {"fidelity.json": ["aae535ab4983512cf03149253fa106af45098b59f1a8d91e530072eeae46068c",
                                         "32e1aeae58b0c503cd8d385c96750ed48328eb040d3489a1ee624294061f0cd9"]},
             "labels": {"aae535ab4983512cf03149253fa106af45098b59f1a8d91e530072eeae46068c": "runs/fidelity-base",
                        "32e1aeae58b0c503cd8d385c96750ed48328eb040d3489a1ee624294061f0cd9": "runs/fidelity-base2"}}
HOLDOUT = ["ar25", "ft09", "lp85", "r11l", "re86", "sb26", "sc25", "tn36", "tr87", "tu93", "vc33"]  # hc_dump_driver

# Session configuration (scripts/mtp_session_a.py reads it; --set changes an entry). Sizes in GB, times in minutes.
CONFIG = {
    "session_hours": 7.0,           # the module's own deadline; Kaggle stops a GPU session at 12 h
    "scripts_dir": SCRIPTS_DIR,
    "wait_inputs_s": 300,
    "working_limit_gb": 19.5,       # Kaggle's /kaggle/working output limit
    "census_paths": ["/", "/kaggle/working", "/tmp", "/dev/shm"],
    "storage": {
        "candidates": ["/tmp", "/dev/shm"],  # disk first: a tmpfs is RAM, and the dump server holds ~106-119 GB
        "test_gb": 12.0, "test_seconds": 150,
        "probe_gb": 8.5,            # 16 probe requests, ~388k BF16 rows (plan 10.3: ~8.0 GB)
        "train_gb": 34.0,           # the training dump's cap: held-out snapshots + train split (FP8 rows)
        "holdout_gb": 6.0,          # of which the held-out games (the evaluation set) take at most this
        "min_train_gb": 6.0,
        "server_ram_gb": 125.0,     # host RAM a dump server uses (exp-073's census: 106-119 of 176.9 GB)
        "margin_gb": 3.0,
    },
    "server": {"reap": True, "num_experts": 448, "gpu_free_mib": 3000},
    "probe": {"pass": "seq", "count": 16, "max_rows": 400_000, "min_requests": 10,
              "generic_map_sha256": GENERIC_MAP_SHA256},
    "dump": {"holdout_games": HOLDOUT, "holdout_snapshots_per_game": 1, "holdout_minutes": 20, "minutes": 30,
             "scale_groups": 1, "qerr_warn": 0.05},
    "train": {"minutes": 75, "min_minutes": 15, "reserve_minutes": 50, "grace_minutes": 10, "probe_steps": 6,
              "min_steps": 10, "checkpoint_every": 50, "args": [], "plan_args": []},
    "budgets_min": {"A1": 5, "boot": 20, "A3": 20, "A5": 30, "A8": 10, "A10": 30, "A11": 15},
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _set_config(config: dict, items: list[str]) -> list[str]:
    changes = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set {item!r}: expected KEY=VALUE")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        node, parts = config, key.split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise SystemExit(f"--set {key}: no such section {part!r}")
            node = node[part]
        if parts[-1] not in node:
            raise SystemExit(f"--set {key}: no such entry (known: {', '.join(sorted(node))})")
        if type(node[parts[-1]]) is not type(value) and not (isinstance(node[parts[-1]], float)
                                                              and isinstance(value, int)):
            raise SystemExit(f"--set {key}: expected {type(node[parts[-1]]).__name__} like {node[parts[-1]]!r}")
        node[parts[-1]] = float(value) if isinstance(node[parts[-1]], float) else value
        changes.append(f"{key}={json.dumps(value)}")
    return changes


def worst_case_minutes(config: dict) -> float:
    """The longest the module's steps can take before their budgets stop them (the install in cell 12 and the
    inputs wait not counted): what config['session_hours'] must cover."""
    b, d, t = config["budgets_min"], config["dump"], config["train"]
    return (config["wait_inputs_s"] / 60 + 5  # inputs, A0
            + b["A1"] + 2 * b["boot"] + b["A3"] + 5 + b["A5"]  # A1-A5 (A4: the stop)
            + d["holdout_minutes"] + 5 + d["minutes"] + 20  # A7
            + b["A8"] + t["minutes"] + t["grace_minutes"] + b["A10"] + b["A11"])


def _arc_map_cell(path: Path) -> tuple[str, str, int]:
    """(cell, sha256 of the file it writes, ids): the FR-Spec map as build_franzen_nb.py --hot-tokens ships it
    (zlib-compressed delta varints written as torch's zip layout with fixed metadata), for A9."""
    ids = bf._read_hot_ids(path)
    raw, prev = bytearray(), -1
    for i in ids:
        delta, prev = i - prev - 1, i
        while True:
            byte, delta = delta & 0x7F, delta >> 7
            raw.append(byte | (0x80 if delta else 0))
            if not delta:
                break
    packed = base64.b64encode(zlib.compress(bytes(raw), 9)).decode()
    varint_sha = _sha(bytes(raw))
    with tempfile.TemporaryDirectory() as tmp:
        ns: dict = {}
        exec(bf.HOT_WRITER, ns)
        out = Path(tmp) / "hot.pt"
        if ns["_ours_write_hot_tokens"](packed, varint_sha, str(out)) != len(ids) or bf._read_hot_ids(out) != ids:
            raise SystemExit(f"--hot-tokens {path}: the notebook writer does not round-trip the ids")
        sha = _sha(out.read_bytes())
    lines = "\n".join(packed[i:i + 120] for i in range(0, len(packed), 120))
    cell = (f"{MAP_BEGIN}: the FR-Spec map A9 trains with ({path.name}, {len(ids)} ids;\n"
            "# scripts/frspec_map.py), the map the fine-tuned draft will serve with. The replica checks (A5, A10) use\n"
            "# the wheelhouse's generic hot_tokens_64k.pt, as the reference probe run served it (cell 12 finds and\n"
            "# checks it). A9 checks this file's sha256."
            + bf.HOT_WRITER
            + f"_ours_n = _ours_write_hot_tokens(\"\"\"\n{lines}\n\"\"\", {varint_sha!r}, {ARC_MAP_FILE!r})\n"
            f"print(f'ours: FR-Spec map {path.name} written to {ARC_MAP_FILE}: {{_ours_n}} ids')\n"
            f"{MAP_END}\n")
    return cell, sha, len(ids)


def _cut_launcher(cell: str) -> str:
    """Cell 12 up to (not including) its launch section, plus a note; refuses a launcher it does not know."""
    if cell.count(LAUNCH_CUT) != 1:
        raise SystemExit(f"cell 12: {LAUNCH_CUT.strip()!r} found {cell.count(LAUNCH_CUT)} times (expected once)")
    head, tail = cell.split(LAUNCH_CUT)
    for needle in ("precache_model_thread.start()", "subprocess.Popen(args, env=env", "def show_log_tail"):
        if needle not in tail:
            raise SystemExit(f"cell 12: {needle!r} is not in the launch section; the cut would keep or lose more")
    for needle in ("args = [SGLANG, \"serve\"", "env = dict(os.environ)", "tok = find_unique(WHEELHOUSE_DIR",
                   "DRAFT_VIEW = prepare_draft_view("):
        if needle not in head:
            raise SystemExit(f"cell 12: {needle!r} is not before the launch section")
    return head + (
        f"{SA_BEGIN}: cell 12 ends here, before its launch section would start the server. The session-A cells\n"
        "# below start the dump servers themselves, from `args` and `env` as built above (scripts/mtp_session_a.py:\n"
        "# dump_server_args, dump_server_env), with the same log, port and Popen arguments.\n"
        "print(\"\\n\" + shlex.join(args) + \"\\n\")\n"
        "print('ours (session A): the launcher arguments and environment are ready; no server started here')\n"
        f"{SA_END}\n")


def _setup_cell(config: dict) -> str:
    literal = pprint.pformat(config, width=116, sort_dicts=False)
    return (
        f"{SA_BEGIN}: the MTP draft fine-tune, session A (docs/research/beat-tufa/mtp-drafter-finetune.md 10.4, 11).\n"
        f"# Load the session module written above, check every input (both mount layouts, pinned sha256 or sizes)\n"
        "# and run the storage test A0, before the install and the server. A failure stops the notebook here.\n"
        "import importlib.util as _sa_util\n"
        f"_sa_spec = _sa_util.spec_from_file_location('arc3_session_a', {SCRIPTS_DIR + '/' + MODULE!r})\n"
        "arc3_sa = _sa_util.module_from_spec(_sa_spec)\n"
        "sys.modules['arc3_session_a'] = arc3_sa\n"
        "_sa_spec.loader.exec_module(arc3_sa)\n"
        "if TRUE_SUBMISSION:\n"
        "    raise RuntimeError('the session-A notebook is not a submission')\n"
        f"SESSION_A_CONFIG = {literal}\n"
        "SESSION = arc3_sa.Session(WORKING_DIR, SESSION_A_CONFIG, started=NOTEBOOK_START_TIME)\n"
        "SESSION.find_inputs()\n"
        "SESSION.storage()  # A0\n"
        f"{SA_END}\n")


STEP_CELLS = [
    ("A1", "hand cell 12's launcher (args, env, paths, the generic map `tok`) to the session; install the dump patch "
           "on the installed sglang (after REAP, which cell 12 applied)",
     "if SESSION.go('A1'):\n"
     "    SESSION.set_launcher(args=args, env=env, log=LOG, port=SERVED_MODEL_PORT, model_name=SERVED_MODEL_NAME,\n"
     "                         python_torch=PYTHON, venv=VENV, model_dir=MODEL_DIR, draft_dir=DRAFT_MODEL_DIR,\n"
     "                         generic_map=tok, precache_thread=precache_model_thread)\n"
     "    SESSION.apply_dump_patch()\n"),
    ("A2", "probe dump server: the launcher's args without --speculative-* and REAP's override, plus "
           "--disable-cuda-graph --disable-radix-cache --max-running-requests 1 --chunked-prefill-size 8192; every "
           "row, BF16",
     "if SESSION.go('A2'):\n    SESSION.start_server('A2', 'probe')\n"),
    ("A3", "dump the held-out probe requests plus the reference run's greedy outputs (scripts/mtp_probe_dump.py)",
     "if SESSION.go('A3'):\n    SESSION.probe_dump()\n"),
    ("A4", "stop the probe dump server", "if SESSION.go('A4'):\n    SESSION.stop('A4')\n"),
    ("A5", "the replica check (scripts/mtp_replica.py check): GO, or NO-GO (exit 2), which ends the session here "
           "with its reports written; every later cell is then skipped",
     "SESSION_A_GO = SESSION.go('A5') and SESSION.replica_gate()\n"),
    ("A6", "training dump server: as A2 with the served target (REAP-448's override kept), assistant spans with 256 "
           "rows of context, FP8",
     "if SESSION.go('A6'):\n    SESSION.start_server('A6', 'train')\n"),
    ("A7", "dump one snapshot per held-out game (the evaluation set), then the train split of exp-073's logs "
           "(scripts/hc_dump_driver.py)",
     "if SESSION.go('A7'):\n    SESSION.dump_training()\n"),
    ("A8", "stop the server; plan the training windows (scripts/mtp_train.py plan)",
     "if SESSION.go('A8'):\n    SESSION.train_plan()\n"),
    ("A9", "train the draft's dense weights (scripts/mtp_train.py train; restarted once with fewer steps if the "
           "planned ones cannot fit the budget)",
     "if SESSION.go('A9'):\n    SESSION.train()\n"),
    ("A10", "the replica check with the trained weights: the forecast of the probe gate",
     "if SESSION.go('A10'):\n    SESSION.replica_check('A10', trained=True)\n"),
    ("A11", "write the fine-tuned draft directory /kaggle/working/mtp-draft (scripts/mtp_write_draft.py write)",
     "if SESSION.go('A11'):\n    SESSION.write_draft()\n"),
    ("end", "the summary: verdict, outputs, replica checks, training comparison (session-a.json)",
     "SESSION.finish()\n"),
]


def _step_cell(step: str, what: str, code: str) -> str:
    return f"{SA_BEGIN} {step}: {what}.\n{code}{SA_END}\n"


def _steps_markdown(config: dict) -> str:
    return (
        "## Session A (ours): steps A1-A11\n\n"
        "Each cell below is one step of the MTP draft fine-tune's session A "
        "(docs/research/beat-tufa/mtp-drafter-finetune.md 10.4 and 11), run by `scripts/mtp_session_a.py`. "
        "Every command's output is also in `/kaggle/working/logs/`; `/kaggle/working/session-a.json` is rewritten "
        "after every step (inputs, storage plan, servers, commands, verdicts). A failed step stops its server and "
        "the notebook. A NO-GO of the replica check (A5) ends the session cleanly: no training, the later cells "
        f"print 'skipped'. The module's own deadline is {config['session_hours']:g} h after the notebook started.")


def build(out: Path, slug: str = "arc3-mtp-session-a", *, reap_kept: Path | None = DEFAULT_REAP,
          hot_tokens: Path = DEFAULT_MAP, logs_dataset: str | None = None,
          reference_dataset: str | None = None, wait_inputs: float = 120.0,
          settings: list[str] = (), note: str = "") -> dict:
    raw = bf.DPRIME.read_bytes()
    if _sha(raw) != bf.DPRIME_SHA256:
        raise SystemExit(f"{bf.DPRIME} is not the vendored copy (sha256 differs); it must stay unmodified")
    nb = json.loads(raw)
    nb.setdefault("metadata", {}).setdefault("kaggle", {})["accelerator"] = "nvidiaRtxPro6000"  # D' lacks it
    cells = nb["cells"]
    sources = ["".join(c["source"]) for c in cells]
    code = [c["cell_type"] == "code" for c in cells]
    changes: list[str] = []
    config = copy.deepcopy(CONFIG)
    changes += [f"session config {c}" for c in _set_config(config, list(settings))]
    if not config["server"]["reap"] and reap_kept:
        raise SystemExit("--set server.reap=false needs --no-reap (the launcher would still prune)")

    # cell 4 and every input path; cell 12 with REAP, cut before its launch section
    setup = bf._one_cell(sources, code, bf.SETUP_ANCHOR, "cell 4")
    launch = bf._one_cell(sources, code, bf.LAUNCH_ANCHOR, "cell 12")
    if launch <= setup:
        raise SystemExit("cell 12 is not after cell 4")
    sources, n_inputs = bf._input_fallback(sources, code, setup)
    sources[setup] = bf._replace_once(sources[setup], bf.WAIT_ANCHOR, bf._wait_code(wait_inputs) + bf.WAIT_ANCHOR,
                                      "--wait-inputs")
    changes.append(f"resolves its {n_inputs} /kaggle/input paths in either Kaggle mount layout and waits up to "
                   f"{wait_inputs:g} s for its mounted inputs before copying the bundle")
    if re.search(r"^\s*(SPEC|FRSPEC)=False", sources[launch], re.M):
        raise SystemExit("cell 12: CFG SPEC/FRSPEC is off, so `tok` (the generic FR-Spec map) would not be defined")
    reap_files: list[tuple[str, str]] = []
    if reap_kept:
        import sglang_reap_patch
        sources[launch], reap_files, done = bf._reap(sources[launch], Path(reap_kept))
        changes.append(done)
        kept = len(sglang_reap_patch.load_kept(Path(reap_kept))[0])
        if config["server"]["reap"] and config["server"]["num_experts"] != kept:
            raise SystemExit(f"--reap-kept keeps {kept} experts but the session expects {config['server']['num_experts']}")
    else:
        config["server"].update(reap=False, num_experts=None)
    sources[launch] = _cut_launcher(sources[launch])
    changes.append("cell 12 installs Pennyroyal and builds the launch arguments and environment but starts no server "
                   "(cut before its launch section); the cells after it (benchmark, monitor, diagnostics) are dropped")

    # inputs
    probe = bf._probe_spec(PROBE_DIR)
    logs_manifest = json.loads(LOGS_MANIFEST.read_text())
    logs = {"kind": "kernel", "id": logs_manifest["kernel"], "files": dict(logs_manifest["files"])}
    reference = copy.deepcopy(REFERENCE)
    for flag, slug_, spec in (("--logs-dataset", logs_dataset, logs), ("--reference-dataset", reference_dataset,
                                                                        reference)):
        if slug_:
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9-]*", slug_):
                raise SystemExit(f"{flag} {slug_!r}: expected owner/slug")
            spec.update(kind="dataset", id=slug_)
    config["inputs"] = {"prompts": {"kind": "dataset", "id": probe["dataset"], "files": dict(probe["files"])},
                        "reference": reference, "logs": logs}

    # files our cells write
    map_cell, map_sha, map_ids = _arc_map_cell(Path(hot_tokens))
    config["train"].update(map=ARC_MAP_FILE, map_sha256=map_sha)
    shipped = {name: (ROOT / "scripts" / name).read_text() for name in SCRIPTS}
    config["build"] = {"builder": "scripts/build_mtp_session.py", "base": "dprime", "slug": slug,
                       "scripts": {n: _sha(t.encode())[:12] for n, t in shipped.items()},
                       "reap_kept": (f"{Path(reap_kept).name} sha256 {_sha(Path(reap_kept).read_bytes())[:12]}"
                                     if reap_kept else None),
                       "train_map": f"{Path(hot_tokens).name} ({map_ids} ids), file sha256 {map_sha[:12]}",
                       "logs_source": f"{logs['kind']} {logs['id']}",
                       "reference_source": f"{reference['kind']} {reference['id']}", "note": note}
    hours = float(config["session_hours"])
    worst = worst_case_minutes(config)
    if not 0 < hours <= 10 or worst > hours * 60:
        raise SystemExit(f"session_hours {hours:g}: must be at most 10 (Kaggle stops at 12 h) and cover the steps' "
                         f"budgets ({worst:.0f} min)")

    def new_cell(template: int, text: str) -> dict:
        cell = copy.deepcopy(cells[template])
        cell["outputs"], cell["execution_count"] = [], None
        cell["source"] = text.splitlines(keepends=True)
        return cell

    def markdown(text: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

    for cell, s in zip(cells, sources):
        cell["source"] = s.splitlines(keepends=True)
    after_setup = [markdown("## Session A (ours): scripts, inputs and the storage test (A0)\n\nOur scripts are written "
                            f"to `{SCRIPTS_DIR}/` (compressed, sha256 checked), then the session checks its inputs "
                            "and the dump directories before anything is installed."),
                   new_cell(setup, f"{SA_BEGIN}: the folder our scripts are written to\n"
                                   f"os.makedirs({SCRIPTS_DIR!r}, exist_ok=True)\n{SA_END}\n")]
    after_setup += [new_cell(setup, bf._file_cell(f"{SCRIPTS_DIR}/{name}", text, True)) for name, text in shipped.items()]
    after_setup.append(new_cell(setup, _setup_cell(config)))
    before_launch = [new_cell(launch, bf._file_cell(path, text, True)) for path, text in reap_files]
    before_launch.append(new_cell(launch, map_cell))
    after_launch = [markdown(_steps_markdown(config))] + [new_cell(launch, _step_cell(*s)) for s in STEP_CELLS]
    nb["cells"] = (cells[:setup + 1] + after_setup + cells[setup + 1:launch] + before_launch + [cells[launch]]
                   + after_launch)
    changes.append(f"session A: {len(shipped)} scripts written to {SCRIPTS_DIR} after cell 4 (compressed, sha256 "
                   "checked), the inputs check and the storage test (A0) right after them, the ARC FR-Spec map "
                   f"{Path(hot_tokens).name} written before the launcher (A9 trains with it; file sha256 "
                   f"{map_sha[:12]}), and one cell per step A1-A11 after it (scripts/mtp_session_a.py)")
    changes.append(f"inputs: {probe['dataset']} (probe prompts), {reference['kind']} {reference['id']} (fidelity.json, "
                   f"the replica check's reference), {logs['kind']} {logs['id']} ({len(logs['files'])} request logs)")
    if reap_files:
        changes.append(f"the {len(reap_files)} REAP files are shipped compressed (sha256 checked)")
    cells[0]["source"] = ["".join(cells[0]["source"]) + "\n\n**Our session-A notebook (scottmahony, built by "
                          "scripts/build_mtp_session.py from the unmodified dprime notebook):** " + "; ".join(changes)
                          + (f". {note}" if note else "") + "\n"]

    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    sources_meta = copy.deepcopy(bf.SOURCES)
    sources_meta["dataset_sources"].append(probe["dataset"])
    sources_meta["kernel_sources"] = []
    for spec in (reference, logs):
        listed = sources_meta["dataset_sources" if spec["kind"] == "dataset" else "kernel_sources"]
        if spec["id"] not in listed:
            listed.append(spec["id"])
    meta = {"id": f"scottmahony/{slug}", "title": slug.replace("-", " "), "code_file": f"{slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": False, "keywords": [], **sources_meta,
            "docker_image": bf.IMAGE, "docker_image_pinning_type": "original", "machine_shape": bf.MACHINE_SHAPE}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return {"changes": changes, "config": config, "notebook": out / f"{slug}.ipynb",
            "bytes": (out / f"{slug}.ipynb").stat().st_size, "worst_case_minutes": round(worst)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--slug", default="arc3-mtp-session-a")
    ap.add_argument("--reap-kept", type=Path, default=DEFAULT_REAP, metavar="FILE",
                    help="the served target's REAP list (default: REAP-448, as exp-073 served it)")
    ap.add_argument("--no-reap", action="store_true", help="the training dump serves the unpruned target")
    ap.add_argument("--hot-tokens", type=Path, default=DEFAULT_MAP, metavar="FILE",
                    help="the FR-Spec map A9 trains with (default: the ARC map)")
    ap.add_argument("--logs-dataset", default=None, metavar="OWNER/SLUG",
                    help="exp-073's request logs from this dataset instead of the kernel's output")
    ap.add_argument("--reference-dataset", default=None, metavar="OWNER/SLUG",
                    help="the reference fidelity.json from this dataset instead of the kernel's output")
    ap.add_argument("--wait-inputs", type=float, default=120.0, metavar="SECONDS")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="settings",
                    help="change one entry of the session configuration (dotted key, JSON value)")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    settings = list(args.settings) + (["server.reap=false"] if args.no_reap else [])
    result = build(args.out, args.slug, reap_kept=None if args.no_reap else args.reap_kept,
                   hot_tokens=args.hot_tokens, logs_dataset=args.logs_dataset,
                   reference_dataset=args.reference_dataset, wait_inputs=args.wait_inputs,
                   settings=settings, note=args.note)
    print(f"built {result['notebook']} ({result['bytes']:,} bytes; steps' worst case {result['worst_case_minutes']} "
          f"min of {result['config']['session_hours']:g} h): " + "; ".join(result["changes"]))


if __name__ == "__main__":
    main()
