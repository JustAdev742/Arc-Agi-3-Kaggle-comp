#!/usr/bin/env python
"""Build our private arms of Daniel Franzen's Milestone 2 notebook (public LB 27.89; Apache-2.0, kaggle/franzen/).

    .venv/bin/python scripts/build_franzen_nb.py --out DIR --slug arc3-franzen-m2              # unchanged copy
    .venv/bin/python scripts/build_franzen_nb.py --out DIR --slug arc3-franzen-m2-full25 --full25 121
    .venv/bin/python scripts/build_franzen_nb.py ... --env MULTIMODAL_UPSCALE=8 --env ARC3_MAX_ACTIVE_STREAMS=12
    .venv/bin/python scripts/build_franzen_nb.py ... --patch ours.patch --env-add EXPOSE_RESET=on
    .venv/bin/python scripts/build_franzen_nb.py ... --cfg MAXREQ=12 --cfg MEMFRAC=0.975 --server-env SGLANG_SM120_ONLINE_MXFP8=true
    .venv/bin/python scripts/build_franzen_nb.py ... --reap-kept kaggle/franzen/reap448_kept_experts.json --cfg MAXREQ=16

Without options the notebook is byte-for-byte his (only our kernel metadata differs): a Kaggle "Save & Run" of it plays
his 10-game demo subset for 25 minutes per game, and a competition rerun plays the hidden set exactly as his did.
Options change a non-submission run only, or a named setting everywhere:

- ``--full25 MIN``: the Save & Run plays all 25 public games (his demo list emptied) with MIN minutes per game. 121
  matches the hidden set's compute per game (110 games share his 10 admission slots for 532 minutes: 48.4
  slot-minutes per game; 25 games x 48.4 / 10 slots = 121 minutes); 25 gives 25 games 25 minutes each, all started
  at once and contending for his 10 admission slots. The competition rerun is untouched.
- ``--env KEY=VALUE`` (repeatable): override one key of his ``setup_env`` / priority-scheduling dictionaries in cell 4
  (they become environment variables for the harness); the key must already exist there exactly once.
- ``--env-add KEY=VALUE`` (repeatable): add a key his cell 4 does not set at all to ``setup_env`` (e.g.
  ``EXPOSE_RESET=on``); refused if the key already occurs in cell 4 (then ``--env`` is the flag).
- ``--cfg KEY=VALUE`` (repeatable): change one entry of ``CFG = dict(...)`` in the SGLang launcher cell (cell 12),
  e.g. ``MAXREQ=12``, ``CUDAGRAPH_MAXBS=12``, ``MAMBA_CACHE=72``, ``MEMFRAC=0.975``. The new value keeps the
  entry's Python type as written there (int, float, bool or string; an int may be given as arithmetic such as
  ``(116+12+8)*1024``); entries that are not literals (SERVED_NAME) are refused. MAXREQ is the server's request
  limit: his ``ARC3_MAX_ACTIVE_STREAMS`` (cell 4) admits that many games, so change both together.
- ``--server-env KEY=VALUE`` (repeatable): change a string the launcher cell puts in the SGLang server's
  environment through its ``env.update({...})`` (e.g. ``SGLANG_SM120_ONLINE_MXFP8=true``); the key must occur in
  cell 12 exactly once, with a plain string value. This is the effective value: cell 12 starts ``sglang serve``
  with ``subprocess.Popen(args, env=env)`` from that dictionary and never sources the wheelhouse's runtime_env.sh
  (which also exports SGLANG_SM120_ONLINE_MXFP8=0, but only for a shell that sources it), and nothing in the cell
  sets the key after ``env.update``.
- ``--base dprime``: start from the public D' notebook (kaggle/dprime/: his notebook with only the scheduler's slot
  priority replaced; its Save & Run plays one game for 5 minutes) instead of his; all options work the same.
- ``--patch FILE`` (repeatable): our harness changes on top of his, as a unified diff against the tree his notebook
  builds (paths ``a/ARC3-Inference/...`` or ``a/tufa-arc-agi-framework/...``; make one with
  ``scripts/franzen_tree.py build DIR`` + edit + ``scripts/franzen_tree.py diff DIR``). Each patch becomes a
  ``%%writefile /kaggle/ours-NN-NAME.patch`` cell after his patch cell, and cell 4 applies them in order with
  ``git apply -v`` right after his patch, in the same directory; the notebook stops there if a patch fails or
  applies to fewer files than it names. The builder first applies his patch and ours to the exact tree the
  notebook builds (scripts/franzen_tree.py, from his repo or a download of the bundle) and refuses to build when
  that fails; ``--no-apply-check`` skips that (only when neither source is available).

- ``--reap-kept FILE``: serve the target with only the routed experts FILE lists per layer (REAP pruning at load
  time, kaggle/franzen/reap448_kept_experts.json; docs/research/beat-tufa/reap-at-load.md). Adds ``%%writefile``
  cells (scripts/sglang_reap_patch.py, FILE and its ``.meta.json``) before the launcher cell; in cell 12, right after
  ``env.update({...})``, runs the patch on the installed sglang (the cell raises if the file is not the analysed
  one or an anchor is missing) and sets ARC3_REAP_KEPT_EXPERTS for the server; before the prefetch flag adds
  ``--json-model-override-args '{"text_config": {"num_experts": K}}' --speculative-draft-model-override-args '{}'``
  (the MTP draft keeps its 512 experts). Pair it with ``--cfg MAXREQ=.. --cfg CUDAGRAPH_MAXBS=.. --cfg MAMBA_CACHE=..``.
- ``--probe DIR``: a fidelity-probe notebook (docs/research/beat-tufa/fidelity-probe.md). DIR is the probe dataset's
  folder or the repo's copy of its metadata (kaggle/fidelity/: ``dataset-metadata.json`` + ``manifest.json`` from
  scripts/fidelity_sample.py). Everything up to and including the server launch stays; right after cell 4 a
  ``%%writefile /kaggle/arc3-fidelity-probe.py`` cell (scripts/fidelity_probe.py) and a check that the dataset is
  mounted with the manifest's sha256 (before the server starts); the benchmark cell (``await bm.run(...)``) becomes
  the probe: wait for the server's /health (raise if it dies or is not healthy 55 min after the notebook started),
  replay every sampled request greedy (temperature 0, max_tokens 192, logprobs with top 5) one at a time and then
  8 in flight, and write /kaggle/working/fidelity.json; the cells after it (diagnostics) are dropped and the dataset
  joins the kernel's sources. Needs ``--input-fallback``; refuses ``--full25``, ``--patch`` and speculative
  acceptance thresholds other than 1.0. A REAP arm is the same command plus ``--reap-kept``.

- ``--model NAME``: serve another checkpoint of the same architecture and format instead of Intel's W4A16; ``swift``
  is UkisAI's Swift-1.5 (a reasoning-efficient RL/OPD derivative of Flash-Next, the same tokenizer and chat
  template), whose Kaggle copy is one HF repo split over two model instances. Cell 4's MODEL_DIR becomes a directory
  of symlinks to both instances (built once they are mounted; the draft is still his albucino MTP checkpoint) and the
  kernel's model source is swapped. With ``--reap-kept`` it needs ``--reap-no-verify``.
- ``--reap-no-verify``: ``--reap-kept`` without the list's ``.meta.json``, so the server drops the same expert ids
  without checking the routers' sha256 (which belong to the checkpoint the list was made for).
- ``--fail-fast`` (test arms, with ``--full25``): a watchdog thread started right before ``await bm.run(`` exits the
  kernel when the SGLang server process has exited, or when it was never healthy 55 min after the notebook
  started. His notebook releases the benchmark without a server on purpose (a rerun must not idle out), so a test
  arm whose server died used to play its whole budget without a model. Off in a competition rerun.

- ``--hot-tokens FILE``: the MTP draft proposes only tokens in its FR-Spec map; his launcher uses Pennyroyal's
  generic 64k map, which lacks 1.2-1.5% of our model's output tokens ("Hmm", " BFS", ".ascii", grid runs).
  kaggle/franzen/hot_tokens_64k_arc.pt (scripts/frspec_map.py; same size, special tokens kept) covers 99.9%. A cell
  before the launcher writes FILE (base64, sha256 checked) to /kaggle/arc3-hot-tokens.pt; cell 12 reads it there,
  and its TOKEN_MAP_SHA becomes FILE's sha256, so his own assert checks it.
- ``--draft OWNER/KERNEL[/SUBDIR] --draft-manifest FILE``: serve a fine-tuned MTP draft written by MTP session A
  (scripts/mtp_write_draft.py: albucino's files with the dense tensors replaced) instead of his albucino checkpoint.
  The kernel's output joins the kernel sources and the albucino model source is dropped; cell 4's DRAFT_MODEL_DIR
  becomes SUBDIR (default ``mtp-draft``) of the mounted output, in either mount layout
  (``/kaggle/input/notebooks/OWNER/KERNEL`` or ``/kaggle/input/KERNEL``), once mounted. FILE is that output's
  ``arc3-draft-manifest.json`` as pulled from the run; the notebook refuses a mounted draft whose manifest has another
  sha256 (another version of the kernel) or whose dense shard does not match it.
- ``--compact``: each file our cells write (``--patch``, ``--reap-kept``, ``--probe``) is shipped zlib-compressed in
  base64 instead of as a ``%%writefile`` cell: a code cell checks the sha256 of exactly the bytes the %%writefile cell
  would write and writes them (Kaggle refuses a notebook near 1 MB with a bare HTTP 400; lesson 0033).
  scripts/franzen_tree.py reads both kinds of cell back (``written_file``), so scripts/franzen_bed.py takes either.

Every change is anchored on text that must occur exactly once, and listed in the first markdown cell. Writes
``<out>/<slug>.ipynb`` and ``<out>/kernel-metadata.json`` (private, internet off, RTX PRO 6000).
"""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import json
import re
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import franzen_tree  # noqa: E402

BASE = ROOT / "kaggle" / "franzen" / "arc-agi-3-milestone-2-solution.ipynb"
BASE_SHA256 = "7b76c194b478faa0309b01e5fba05840f1314da7f288e05e5f4b09972e4f9b4c"
SOURCES = {  # his kernel-metadata.json, 2026-10-02
    "dataset_sources": ["dfranzen/pennyroyal-v253", "dfranzen/taaf-kaggle-source-bundle-copy"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "model_sources": ["dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1",
                      "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1"],
    "kernel_sources": [],
}
# The Kaggle image his notebook ran in (his kernel-metadata.json; his v3 played 8.2 h on the RTX PRO 6000 in it on
# 2026-10-03), pinned on every kernel we build, D' included: by 2026-10-07 Kaggle's latest image was Python 3.13 while
# Pennyroyal's wheels are cp312 only (exp-070/070d v1 failed in the install cell), and the image D''s metadata names
# (gcr.io/kaggle-images/python, Kaggle's CPU image) gave a session without CUDA (exp-070d v2).
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c"
MACHINE_SHAPE = "NvidiaRtxPro6000"
DEMO_ANCHOR = ("demo_excluded_games = [] if TRUE_SUBMISSION else ['bp35', 'cd82', 'cn04', 'dc22', 'g50t', 'ka59', "
               "'lf52', 'ls20', 'm0r0', 's5i5', 'sk48', 'sp80', 'su15', 'tn36', 'wa30']")
BUDGET_ANCHOR = "        bm.solver.max_runtime_s_per_game = 25*60 #532*60 * bm.solver.concurrency // 110"
# --base dprime: the public "D'" notebook (kaggle/dprime/, Franzen's notebook + a new slot priority; cells 4 and 12
# byte-identical to his, so every option below applies unchanged; only its Save & Run demo lines differ)
DPRIME = ROOT / "kaggle" / "dprime" / "affectify-arc-31-54-in-a-single-sub.ipynb"
DPRIME_SHA256 = "f649d005040ee367bec580c180b3712c81583f622b5690c8e0e1a745dc0858c7"
DPRIME_DEMO_ANCHOR = ("demo_excluded_games = [] if TRUE_SUBMISSION else ['ar25', 'bp35', 'cd82', 'cn04', 'dc22', "
                      "'g50t', 'ka59', 'lf52', 'lp85', 'ls20', 'm0r0', 'r11l', 're86', 's5i5', 'sb26', 'sc25', 'sk48', "
                      "'sp80', 'su15', 'tn36', 'tr87', 'tu93', 'vc33', 'wa30']")
DPRIME_BUDGET_ANCHOR = "        bm.solver.max_runtime_s_per_game = 5*60  # rehearsal only proves the run completes"
SETUP_ANCHOR = "setup_env = {\n"                       # cell 4
APPLY_ANCHOR = "print('harness patch applied successfully')\n"  # cell 4, right after his git apply
WAIT_ANCHOR = "!rm -Rf $BUNDLE_DIR\n"                   # cell 4, right before his copy of the mounted bundle
LAUNCH_ANCHOR = "CFG = dict(\n"                        # cell 12
SERVER_ENV_ANCHOR = "env.update({\n"                   # cell 12
HIS_PATCH_ANCHOR = "%%writefile /kaggle/harness-changes.patch"  # cell 2
PATCH_ROOTS = ("ARC3-Inference/", "tufa-arc-agi-framework/")
OURS_BEGIN = "# >>> ours (--patch)"
OURS_END = "# <<< ours (--patch)"
# --reap-kept: cell 12 installs the wheel, builds `env`, then the launch args; we patch the installed sglang right
# after `env.update({...})` (before the nvcc probe) and add the model-override flags before the prefetch flag.
REAP_APPLY_ANCHOR = 'run([str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], env=env)\n'
REAP_ARGS_ANCHOR = 'if CFG["PREFETCH_CHECKPOINTS"]: args += ["--weight-loader-prefetch-checkpoints"]\n'
REAP_SCRIPT = ROOT / "scripts" / "sglang_reap_patch.py"
REAP_FILES = {"script": "/kaggle/arc3-reap-patch.py", "kept": "/kaggle/arc3-reap-kept.json",
              "meta": "/kaggle/arc3-reap-kept.meta.json"}  # meta: sglang_reap_patch.meta_path(kept)
REAP_BEGIN = "# >>> ours (--reap-kept)"
REAP_END = "# <<< ours (--reap-kept)"
# --probe: the fidelity probe (scripts/fidelity_probe.py) replaces the benchmark cell (his cell 20, D' cell 22)
PROBE_RUN_ANCHOR = "await bm.run("
PROBE_MD_ANCHOR = "## 9. Run the benchmark"
PROBE_SCRIPT = ROOT / "scripts" / "fidelity_probe.py"
PROBE_FILE = "/kaggle/arc3-fidelity-probe.py"
PROBE_BEGIN = "# >>> ours (--probe)"
PROBE_END = "# <<< ours (--probe)"
PROBE_PARAMS = {"max_tokens": 192, "top_logprobs": 5, "concurrency": 8, "health_minutes": 55, "data_wait_s": 300,
                "max_minutes": 150}
# --model: serve another checkpoint of the same architecture and format in place of Intel's W4A16 (cell 4's MODEL_DIR
# and the kernel's model source); the MTP draft stays his albucino checkpoint
MODEL_LINE = ("MODEL_DIR         = '/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/"
              "default/1'\n")
INTEL_SOURCE = "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1"
MODELS = {
    "swift": {
        "what": ("Swift-1.5 Qwen3.8-Flash-Next W4A16 AutoRound (UkisAI's reasoning-efficient RL/OPD derivative of "
                 "Flash-Next; HF ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AutoRound, Kaggle copy "
                 "phuongncn/arc3-qwen38-swift-w4a16-autoround in two instances)"),
        "sources": ["phuongncn/arc3-qwen38-swift-w4a16-autoround/PyTorch/w4a16-a/1",
                    "phuongncn/arc3-qwen38-swift-w4a16-autoround/PyTorch/w4a16-b/1"],
        # its own BF16 MTP weights (model_mtp.safetensors) stay in: the index names them (exp-076 v1 stopped at
        # "Missing/empty shard" without them), and the target skips mtp.* tensors as it does Intel's
        "skip": [],
    },
}
MODEL_VIEW = "/tmp/ours-model-view"
# --draft: a fine-tuned MTP draft from MTP session A's kernel output in place of his albucino checkpoint
DRAFT_LINE = ("DRAFT_MODEL_DIR   = '/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/"
              "default/1'\n")
ALBUCINO_SOURCE = "dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1"
DRAFT_MANIFEST = "arc3-draft-manifest.json"  # scripts/mtp_write_draft.py
DRAFT_DENSE = "mtp-dense.safetensors"
DRAFT_SUBDIR = "mtp-draft"  # scripts/mtp_session_a.py A11
DRAFT_BEGIN = "# >>> ours (--draft)"
DRAFT_END = "# <<< ours (--draft)"
MODEL_BEGIN = "# >>> ours (--model)"
MODEL_END = "# <<< ours (--model)"
# --fail-fast: a test arm ends when its server is gone (his notebook plays on without one, by design for the rerun)
FAIL_FAST_BEGIN = "# >>> ours (--fail-fast)"
FAIL_FAST_END = "# <<< ours (--fail-fast)"
FAIL_FAST_HEALTH_MINUTES = 55  # from the notebook start; MTP session A v1 (2026-10-10) needed ~27 min to load weights
# --hot-tokens: the MTP draft's FR-Spec vocabulary (his launcher takes Pennyroyal's generic 64k map and checks its sha)
HOT_MAP_FILE = "/kaggle/arc3-hot-tokens.pt"
HOT_SHA_RE = re.compile(r'^TOKEN_MAP_SHA = "([0-9a-f]{64})"$', re.M)
HOT_FIND_ANCHOR = ('        tok = find_unique(WHEELHOUSE_DIR, "hot_tokens_64k.pt", required=False)\n'
                   '        if tok is None: tok = find_unique(WHEELHOUSE_DIR, "flash-next-64k.pt")\n')
HOT_BEGIN = "# >>> ours (--hot-tokens)"
HOT_END = "# <<< ours (--hot-tokens)"
COMPACT_BEGIN = "# >>> ours (--compact)"
COMPACT_END = "# <<< ours (--compact)"
COMPACT_WRITER = f"""
def {franzen_tree.PACKED_WRITER}(path, packed_b64, sha):
    # the bytes a %%writefile cell would write, zlib-compressed in base64 and checked before they are written
    import base64, hashlib, zlib
    data = zlib.decompress(base64.b64decode(packed_b64))
    if hashlib.sha256(data).hexdigest() != sha:
        raise RuntimeError(f"{{path}}: corrupted in the notebook")
    with open(path, "wb") as f:
        f.write(data)
    print(f"Writing {{path}}")
"""


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"{what}: anchor found {text.count(old)} times in the base notebook (expected once)")
    return text.replace(old, new)


def _one_cell(sources: list[str], code: list[bool], needle: str, what: str) -> int:
    hits = [i for i, s in enumerate(sources) if code[i] and needle in s]
    if len(hits) != 1 or sources[hits[0]].count(needle) != 1:
        raise SystemExit(f"{what}: anchor {needle.strip()!r} found in cells {hits} (expected once in one cell)")
    return hits[0]


def _block(text: str, start: str, end: str, what: str) -> tuple[int, int]:
    """(begin, end) offsets of the text between START (once) and the first END after it."""
    if text.count(start) != 1:
        raise SystemExit(f"{what}: anchor {start.strip()!r} found {text.count(start)} times (expected once)")
    begin = text.index(start) + len(start)
    stop = text.find(end, begin)
    if stop < 0:
        raise SystemExit(f"{what}: no {end.strip()!r} after {start.strip()!r}")
    return begin, stop


def _env_literal(value: str) -> str:
    return value if re.fullmatch(r"-?\d+(\.\d+)?", value) else repr(value)


_ARITH = (ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
          ast.USub, ast.UAdd, ast.Mod)


def _safe_value(text: str):
    """A Python literal, or integer/float arithmetic on literals; None when TEXT is anything else."""
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError:
        return None
    if not all(isinstance(node, _ARITH) for node in ast.walk(tree)):
        return None
    return eval(compile(tree, "<cfg>", "eval"), {"__builtins__": {}}, {})  # only literals and arithmetic


def _typed_cfg_value(key: str, old_text: str, new: str) -> str:
    old = _safe_value(old_text)
    if old is None:
        raise SystemExit(f"--cfg {key}: its value {old_text!r} is not a literal; refusing to guess its type")
    if isinstance(old, bool):
        low = new.strip().lower()
        if low in ("true", "1", "yes", "on"):
            return "True"
        if low in ("false", "0", "no", "off"):
            return "False"
        raise SystemExit(f"--cfg {key}: expected a bool like {old_text}, got {new!r}")
    if isinstance(old, (int, float)):
        value = _safe_value(new)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SystemExit(f"--cfg {key}: expected a number like {old_text}, got {new!r}")
        if isinstance(old, int) and not isinstance(value, int):
            raise SystemExit(f"--cfg {key}: expected an int like {old_text}, got {new!r}")
        if isinstance(old, float) and isinstance(value, int):
            return repr(float(value))
        return new.strip()
    if isinstance(old, str):
        value = new
        if len(new) >= 2 and new[0] == new[-1] and new[0] in "'\"":
            value = ast.literal_eval(new)
        return json.dumps(value)
    raise SystemExit(f"--cfg {key}: unsupported value type {type(old).__name__}")


def _set_env(cell: str, env: dict[str, str]) -> tuple[str, list[str]]:
    """--env: change a knob cell 4 already sets, either an entry of his setup_env dict or one of the plain
    ``os.environ['KEY'] = '...'`` lines above it (e.g. ARC3_HTTP_RETRY_INITIAL_SECONDS)."""
    changes = []
    for key, value in env.items():
        pattern = re.compile(rf"^(\s*)'{re.escape(key)}': ([^,\n]+),", re.M)
        assign = re.compile(rf"^os\.environ\['{re.escape(key)}'\] = '[^'\n]*'$", re.M)
        hits, assigns = pattern.findall(cell), assign.findall(cell)
        if len(hits) + len(assigns) != 1:
            raise SystemExit(f"--env {key}: anchor found {len(hits) + len(assigns)} times in cell 4 (expected once; "
                             "--env-add adds a key)")
        if hits:
            new_value = _env_literal(value)
            cell = pattern.sub(lambda m, nv=new_value, k=key: f"{m.group(1)}'{k}': {nv},  # ours (--env)", cell)
        else:
            cell = assign.sub(lambda m, k=key, v=str(value): f"os.environ[{k!r}] = {v!r}  # ours (--env)", cell)
        changes.append(f"env {key}={value}")
    return cell, changes


def _add_env(cell: str, env_add: dict[str, str]) -> tuple[str, list[str]]:
    if not env_add:
        return cell, []
    for key in env_add:
        if f"'{key}'" in cell or f'"{key}"' in cell:
            raise SystemExit(f"--env-add {key}: cell 4 already sets it; use --env {key}=...")
    _, stop = _block(cell, SETUP_ANCHOR, "\n}\n", "--env-add")
    lines = "".join(f"    '{k}': {_env_literal(v)},  # ours (--env-add)\n" for k, v in env_add.items())
    cell = cell[:stop + 1] + "\n    ########## ours (--env-add) ##########\n" + lines + cell[stop + 1:]
    return cell, [f"env added {k}={v}" for k, v in env_add.items()]


def _set_cfg(cell: str, cfg: dict[str, str]) -> tuple[str, list[str]]:
    changes = []
    for key, value in cfg.items():
        begin, stop = _block(cell, LAUNCH_ANCHOR, "\n)\n", f"--cfg {key}")
        block = cell[begin:stop + 1]
        pattern = re.compile(rf"^(\s*){re.escape(key)}=(.*?),[ \t]*(#.*)?$", re.M)
        hits = pattern.findall(block)
        if len(hits) != 1:
            raise SystemExit(f"--cfg {key}: found {len(hits)} times in CFG = dict(...) (expected once)")
        new_value = _typed_cfg_value(key, hits[0][1], value)
        block = pattern.sub(lambda m, nv=new_value, k=key: f"{m.group(1)}{k}={nv},  # ours (--cfg)", block)
        cell = cell[:begin] + block + cell[stop + 1:]
        changes.append(f"SGLang CFG {key}={value}")
    return cell, changes


def _set_server_env(cell: str, server_env: dict[str, str]) -> tuple[str, list[str]]:
    changes = []
    for key, value in server_env.items():
        named = cell.count(f'"{key}"') + cell.count(f"'{key}'")
        if named != 1:
            raise SystemExit(f"--server-env {key}: named {named} times in the launcher cell (expected once, in its "
                             f"env.update)")
        begin, stop = _block(cell, SERVER_ENV_ANCHOR, "\n})\n", f"--server-env {key}")
        block = cell[begin:stop + 1]
        pattern = re.compile(rf'"{re.escape(key)}":\s*("(?:[^"\\\n]|\\.)*"|[^,\n]+)')
        hits = pattern.findall(block)
        if len(hits) != 1:
            raise SystemExit(f"--server-env {key}: not in the launcher's env.update({{...}})")
        if not hits[0].startswith('"'):
            raise SystemExit(f"--server-env {key}: its value {hits[0]!r} is not a plain string; refusing")
        block = pattern.sub(lambda m, k=key, v=value: f'"{k}": {json.dumps(v)}', block)
        line_start = block.rfind("\n", 0, block.index(f'"{key}"')) + 1
        line_end = block.index("\n", line_start)
        if "# ours (--server-env)" not in block[line_start:line_end]:
            block = block[:line_end] + "  # ours (--server-env)" + block[line_end:]
        cell = cell[:begin] + block + cell[stop + 1:]
        changes.append(f"SGLang server env {key}={value}")
    return cell, changes


def _patch_name(index: int, path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", path.stem).strip("-.") or "patch"
    return f"ours-{index:02d}-{stem}.patch"


def _check_patch_text(path: Path, text: str) -> list[str]:
    """The files PATH changes; refuses what the notebook cell or its git apply would mangle."""
    if not text.startswith("diff --git "):
        raise SystemExit(f"--patch {path}: must be `git diff` output starting with 'diff --git ' (scripts/franzen_tree.py diff)")
    if "GIT binary patch" in text:
        raise SystemExit(f"--patch {path}: binary patches are not supported")
    files = []
    for line in text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            names = [p[2:] for p in parts[2:4]]
            if len(parts) != 4 or not all(n.startswith(PATCH_ROOTS) for n in names):
                raise SystemExit(f"--patch {path}: {line!r} is outside {PATCH_ROOTS}")
            files.append(names[1])
    return files


def _apply_code(names: list[str]) -> str:
    paths = ", ".join(repr(f"/kaggle/{n}") for n in names)
    return (
        f"{OURS_BEGIN}: our harness patches on top of his (scripts/build_franzen_nb.py), applied the same way.\n"
        "# A patch that fails, or applies to fewer files than it names, stops the notebook here.\n"
        f"for _ours_patch in [{paths}]:\n"
        "    _ours_run = subprocess.run([\"git\", \"apply\", \"-v\", _ours_patch], cwd=f\"{BUNDLE_DIR}/src\",\n"
        "                               capture_output=True, text=True,\n"
        "                               env=dict(os.environ, GIT_CEILING_DIRECTORIES=str(BUNDLE_DIR), LC_ALL=\"C\"))\n"
        "    _ours_log = _ours_run.stdout + _ours_run.stderr\n"
        "    print(_ours_log, end=\"\")\n"
        "    _ours_text = Path(_ours_patch).read_text()\n"
        "    _ours_named = _ours_text.startswith(\"diff --git \") + _ours_text.count(\"\\ndiff --git \")\n"
        "    if _ours_run.returncode != 0 or _ours_log.count(\"Applied patch \") != _ours_named:\n"
        "        raise RuntimeError(f\"{_ours_patch} did not apply: exit {_ours_run.returncode}, \"\n"
        "                           f\"{_ours_log.count('Applied patch ')} of {_ours_named} files\")\n"
        f"print('our harness patches applied successfully: {len(names)}')\n"
        f"{OURS_END}\n"
    )


def _reap(cell: str, kept_path: Path, verify: bool = True) -> tuple[str, list[tuple[str, str]], str]:
    """Cell 12 with the REAP steps, the (path, text) files the notebook must write first, and the change line.
    verify=False leaves the list's .meta.json (the router sha256 of the checkpoint it was made for) out, for a
    derivative checkpoint whose routers differ: the same expert ids are dropped without the check."""
    import sglang_reap_patch

    if REAP_BEGIN in cell or "--json-model-override-args" in cell:
        raise SystemExit("--reap-kept: the launcher cell already overrides the model config")
    try:
        kept = sglang_reap_patch.load_kept(kept_path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--reap-kept: {exc}") from None
    num_kept = len(kept[0])
    files = [(REAP_FILES["script"], REAP_SCRIPT.read_text()), (REAP_FILES["kept"], kept_path.read_text())]
    meta = sglang_reap_patch.meta_path(kept_path)
    total = "?"
    if meta.is_file() and verify:
        layers = json.loads(meta.read_text())["layers"]
        if sorted(int(k) for k in layers) != sorted(kept):
            raise SystemExit(f"--reap-kept: {meta} does not cover the layers of {kept_path}")
        total = layers["0"]["num_experts"]
        files.append((REAP_FILES["meta"], meta.read_text()))
    apply = (
        f"{REAP_BEGIN}: REAP expert pruning at load time (scripts/sglang_reap_patch.py). Patch the installed sglang\n"
        "# (stops here if the file is not the analysed one or an anchor is missing), then tell the server the list.\n"
        '_reap_model = sorted(Path(VENV).glob("lib/python*/site-packages/sglang/srt/models/qwen4_exp.py"))\n'
        "if len(_reap_model) != 1:\n"
        '    raise RuntimeError(f"--reap-kept: expected one installed sglang qwen4_exp.py, found {_reap_model}")\n'
        f'run([sys.executable, "-I", "{REAP_FILES["script"]}", "apply", "--site-packages", str(_reap_model[0].parents[3]),\n'
        f'     "--kept", "{REAP_FILES["kept"]}"])\n'
        f'env["{sglang_reap_patch.ENV}"] = "{REAP_FILES["kept"]}"\n'
        f"{REAP_END}\n")
    override = json.dumps({"text_config": {"num_experts": num_kept}})
    args = (
        f"{REAP_BEGIN}: the target is built with {num_kept} routed experts per layer; the MTP draft keeps its own\n"
        "# config (unset, the draft would inherit --json-model-override-args and be built with the pruned count)\n"
        f"args += [\"--json-model-override-args\", {override!r},\n"
        "         \"--speculative-draft-model-override-args\", \"{}\"]\n"
        f"{REAP_END}\n")
    cell = _replace_once(cell, REAP_APPLY_ANCHOR, apply + REAP_APPLY_ANCHOR, "--reap-kept apply")
    cell = _replace_once(cell, REAP_ARGS_ANCHOR, args + REAP_ARGS_ANCHOR, "--reap-kept args")
    sha = hashlib.sha256(kept_path.read_bytes()).hexdigest()[:12]
    change = (f"REAP expert pruning at load: {kept_path.name} (sha256 {sha}) keeps {num_kept} of {total} routed "
              f"experts in each of {len(kept)} layers; cell 12 patches the installed sglang with "
              f"scripts/sglang_reap_patch.py (sha256 {hashlib.sha256(REAP_SCRIPT.read_bytes()).hexdigest()[:12]}) "
              f"after the install, sets {sglang_reap_patch.ENV} for the server and adds --json-model-override-args "
              f"{override} --speculative-draft-model-override-args {{}} (the MTP draft keeps all experts)"
              + ("" if verify else "; the router sha256 check is off (--reap-no-verify: the list was made for "
                                   "another checkpoint of this architecture)"))
    return cell, files, change


def _model_code(name: str) -> str:
    """Cell-4 lines that replace his MODEL_DIR constant: link the checkpoint's Kaggle model instances (one checkpoint
    split over several instances by Kaggle's 50-file limit) into one directory once they are mounted."""
    spec = MODELS[name]
    patterns = []
    for source in spec["sources"]:
        owner, slug, _framework, instance, version = source.split("/")
        patterns.append(f"/kaggle/input/models/{owner}/{slug}/*/{instance}/{version}")  # framework dir: lower case
    return (
        f"{MODEL_BEGIN} {name}: {spec['what']}.\n"
        "# Its instances are linked (symlinks, no copies) into one directory for the server once they are mounted;\n"
        "# the files in the skip set stay out.\n"
        "def _ours_model_view(patterns, view, skip, timeout_s=600.0):\n"
        "    import glob as _ours_glob\n"
        "    t0 = time.time()\n"
        "    while True:\n"
        "        parts = [sorted(p for p in _ours_glob.glob(pat) if os.path.isdir(p)) for pat in patterns]\n"
        "        if all(len(p) == 1 for p in parts):\n"
        "            break\n"
        "        if time.time() - t0 > timeout_s:\n"
        "            raise RuntimeError(f'model instances not mounted (or ambiguous) after {timeout_s:g} s: '\n"
        "                               + str(dict(zip(patterns, parts))))\n"
        "        time.sleep(5)\n"
        "    os.makedirs(view, exist_ok=True)\n"
        "    for (part,) in parts:\n"
        "        for name in sorted(os.listdir(part)):\n"
        "            if name in skip:\n"
        "                continue\n"
        "            src, dst = os.path.join(part, name), os.path.join(view, name)\n"
        "            if os.path.lexists(dst):\n"
        "                if os.path.realpath(dst) != os.path.realpath(src):\n"
        "                    raise RuntimeError(f'model view: {name} is in more than one instance')\n"
        "                continue\n"
        "            os.symlink(src, dst)\n"
        "    if not os.path.isfile(os.path.join(view, 'model.safetensors.index.json')):\n"
        "        raise RuntimeError(f'model view {view}: no model.safetensors.index.json')\n"
        "    print(f'ours: model view {view}: {len(os.listdir(view))} files from ' + ', '.join(p[0] for p in parts))\n"
        "    return view\n"
        f"MODEL_DIR         = _ours_model_view({patterns!r}, {MODEL_VIEW!r}, {sorted(spec['skip'])!r})\n"
        f"{MODEL_END}\n")


def _draft_spec(draft: str, manifest: Path) -> dict:
    """--draft OWNER/KERNEL[/SUBDIR] and its manifest: the mount candidates and the sha256 values the notebook checks."""
    parts = draft.strip("/").split("/")
    if len(parts) not in (2, 3) or not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", x) for x in parts):
        raise SystemExit(f"--draft {draft!r}: expected OWNER/KERNEL or OWNER/KERNEL/SUBDIR")
    owner, kernel, subdir = [*parts, DRAFT_SUBDIR][:3]
    try:
        raw = Path(manifest).read_bytes()
        files = json.loads(raw)["files"]
        dense = files[DRAFT_DENSE]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"--draft-manifest {manifest}: not a {DRAFT_MANIFEST} with the sha256 of {DRAFT_DENSE} "
                         f"({exc!r})") from None
    if not re.fullmatch(r"[0-9a-f]{64}", str(dense)):
        raise SystemExit(f"--draft-manifest {manifest}: {DRAFT_DENSE} has no sha256 ({dense!r})")
    return {"kernel": f"{owner}/{kernel}", "subdir": subdir, "manifest_sha": hashlib.sha256(raw).hexdigest(),
            "dense_sha": dense, "candidates": [f"/kaggle/input/notebooks/{owner}/{kernel}/{subdir}",
                                               f"/kaggle/input/{kernel}/{subdir}"]}


def _draft_code(spec: dict) -> str:
    """Cell-4 lines that replace his DRAFT_MODEL_DIR constant: wait for the kernel output in either layout, then
    check its manifest and dense shard against the build's sha256 before anything reads the draft."""
    return (
        f"{DRAFT_BEGIN}: the MTP draft is {spec['kernel']}'s output {spec['subdir']}/ (MTP session A: albucino's files\n"
        "# with the dense tensors fine-tuned on ARC traffic, scripts/mtp_write_draft.py), mounted as a kernel source.\n"
        "def _ours_draft_dir(candidates, manifest_sha, dense_sha, timeout_s=600.0):\n"
        "    import hashlib\n"
        "    def sha256(path):\n"
        "        h = hashlib.sha256()\n"
        "        with open(path, 'rb') as f:\n"
        "            for block in iter(lambda: f.read(1 << 24), b''):\n"
        "                h.update(block)\n"
        "        return h.hexdigest()\n"
        "    t0 = time.time()\n"
        f"    while not (found := [p for p in candidates if os.path.isfile(os.path.join(p, {DRAFT_MANIFEST!r}))]):\n"
        "        if time.time() - t0 > timeout_s:\n"
        "            raise RuntimeError(f'MTP draft not mounted after {timeout_s:g} s: {candidates}')\n"
        "        time.sleep(5)\n"
        "    path = found[0]\n"
        f"    got = sha256(os.path.join(path, {DRAFT_MANIFEST!r}))\n"
        "    if got != manifest_sha:\n"
        "        raise RuntimeError(f'MTP draft {path}: manifest sha256 {got}, the build expects {manifest_sha} '\n"
        "                           '(another version of the kernel output?)')\n"
        f"    got = sha256(os.path.join(path, {DRAFT_DENSE!r}))\n"
        "    if got != dense_sha:\n"
        f"        raise RuntimeError(f'MTP draft {{path}}: {DRAFT_DENSE} sha256 {{got}}, its manifest says {{dense_sha}}')\n"
        "    print(f'ours: MTP draft {path} after {time.time() - t0:.0f} s (manifest {manifest_sha[:12]}, '\n"
        "          f'dense {dense_sha[:12]})')\n"
        "    return path\n"
        f"DRAFT_MODEL_DIR   = _ours_draft_dir({spec['candidates']!r}, {spec['manifest_sha']!r}, {spec['dense_sha']!r})\n"
        f"{DRAFT_END}\n")


HOT_WRITER = """
def _ours_write_hot_tokens(packed_b64, varint_sha, path):
    # FR-Spec ids: zlib-compressed delta varints -> the list torch.save would store, written as torch's zip layout
    # (fixed record names, times and serialization id, so the file and its sha256 are the same on every run)
    import base64, hashlib, pickle, zipfile, zlib
    raw = zlib.decompress(base64.b64decode(packed_b64))
    if hashlib.sha256(raw).hexdigest() != varint_sha:
        raise RuntimeError("FR-Spec map corrupted in the notebook")
    ids, prev, i = [], -1, 0
    while i < len(raw):
        delta, shift = 0, 0
        while True:
            byte = raw[i]
            i += 1
            delta |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                break
        prev += delta + 1
        ids.append(prev)
    records = [("data.pkl", pickle.dumps(ids, protocol=2)), (".format_version", b"1"), (".storage_alignment", b"64"),
               ("byteorder", b"little"), ("version", b"3\\n"), (".data/serialization_id", b"0" * 40)]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        for name, data in records:
            info = zipfile.ZipInfo("arc3_hot_tokens/" + name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            z.writestr(info, data)
    return len(ids)
"""


def _read_hot_ids(path: Path) -> list[int]:
    """The id list in a torch.save'd FR-Spec map (a zip with a pickled list), read without torch or any global."""
    import pickle
    import zipfile

    class _NoGlobals(pickle.Unpickler):
        def find_class(self, module, name):
            raise SystemExit(f"--hot-tokens {path}: refusing pickled global {module}.{name}")

    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith("/data.pkl")]
        if len(names) != 1:
            raise SystemExit(f"--hot-tokens {path}: not a torch.save archive with one data.pkl")
        ids = _NoGlobals(z.open(names[0])).load()
    if not (isinstance(ids, list) and ids == sorted(set(ids)) and all(isinstance(i, int) and i >= 0 for i in ids)):
        raise SystemExit(f"--hot-tokens {path}: expected a sorted list of distinct token ids")
    return ids


def _hot_tokens(cell: str, path: Path) -> tuple[str, str, str]:
    """Cell 12 reading our FR-Spec map, the code cell that writes it before the launcher, and the change line.

    The map travels as zlib-compressed delta varints (about 3 KB: a Kaggle notebook over ~1 MB is refused with HTTP
    400, as the first build of exp-077 was) and is written as torch's zip layout with fixed metadata, so its sha256
    is known here and replaces his TOKEN_MAP_SHA: his own assert checks it."""
    import base64
    import zlib

    ids = _read_hot_ids(path)
    raw, prev = bytearray(), -1
    for i in ids:
        delta, prev = i - prev - 1, i
        while True:
            byte, delta = delta & 0x7F, delta >> 7
            raw.append(byte | (0x80 if delta else 0))
            if not delta:
                break
    packed = base64.b64encode(zlib.compress(bytes(raw), 9)).decode()
    varint_sha = hashlib.sha256(bytes(raw)).hexdigest()
    with tempfile.TemporaryDirectory() as tmp:  # the file the notebook will write, to know its sha256 here
        ns: dict = {}
        exec(HOT_WRITER, ns)
        out = Path(tmp) / "hot.pt"
        if ns["_ours_write_hot_tokens"](packed, varint_sha, str(out)) != len(ids):
            raise SystemExit(f"--hot-tokens {path}: the notebook writer does not round-trip the ids")
        sha = hashlib.sha256(out.read_bytes()).hexdigest()
    if len(HOT_SHA_RE.findall(cell)) != 1:
        raise SystemExit("--hot-tokens: TOKEN_MAP_SHA not found once in the launcher cell")
    cell = HOT_SHA_RE.sub(f'TOKEN_MAP_SHA = "{sha}"  # ours (--hot-tokens): {path.name}, as the cell above writes it',
                          cell)
    cell = _replace_once(cell, HOT_FIND_ANCHOR, f"        tok = Path({HOT_MAP_FILE!r})  # ours (--hot-tokens)\n",
                         "--hot-tokens")
    if re.search(r"^\s*FRSPEC=False", cell, re.M):
        raise SystemExit("--hot-tokens: CFG FRSPEC is off, so the map would not be used")
    lines = "\n".join(packed[i:i + 120] for i in range(0, len(packed), 120))
    writer = (
        f"{HOT_BEGIN}: the MTP draft's FR-Spec vocabulary is {path.name} (scripts/frspec_map.py, {len(ids)} ids)\n"
        "# instead of Pennyroyal's generic hot_tokens_64k.pt; the launcher's sha256 assert checks the file written here."
        + HOT_WRITER
        + f"_ours_n = _ours_write_hot_tokens(\"\"\"\n{lines}\n\"\"\", {varint_sha!r}, {HOT_MAP_FILE!r})\n"
        f"print(f'ours: FR-Spec map {path.name} written to {HOT_MAP_FILE}: {{_ours_n}} ids')\n"
        f"{HOT_END}\n")
    change = (f"MTP draft FR-Spec map {path.name} ({len(ids)} ids; written by a cell before the launcher as a "
              f"{len(packed)}-character payload, file sha256 {sha[:12]}) instead of Pennyroyal's generic "
              "hot_tokens_64k.pt")
    return cell, writer, change


def _fail_fast_code() -> str:
    """Lines before `await bm.run(` that end a test arm when its SGLang server is gone. His notebook releases the
    benchmark even when the server is still loading or has died (so a rerun never idles out), which in a test arm
    means playing the whole budget without a model (exp-072j: ~0.7 GPU-h). Never active in a competition rerun."""
    return (
        f"{FAIL_FAST_BEGIN}: a test arm ends (the kernel exits) when its SGLang server process has exited, or when\n"
        f"# the server was never healthy {FAIL_FAST_HEALTH_MINUTES} min after the notebook started. Off in a "
        "competition rerun.\n"
        f"def _ours_server_watch(health_deadline_s={FAIL_FAST_HEALTH_MINUTES * 60}, period_s=30):\n"
        "    healthy = False\n"
        "    while True:\n"
        "        time.sleep(period_s)\n"
        "        if proc.poll() is not None:\n"
        "            print(f'ours (--fail-fast): the SGLang server exited with code {proc.returncode}; ending this '\n"
        "                  'test arm', flush=True)\n"
        "            try:\n"
        "                show_log_tail()\n"
        "            finally:\n"
        "                os._exit(3)\n"
        "        if not healthy:\n"
        "            try:\n"
        "                with urllib.request.urlopen(f'http://127.0.0.1:{SERVED_MODEL_PORT}/health', timeout=5) as r:\n"
        "                    healthy = r.status == 200\n"
        "            except Exception:\n"
        "                pass\n"
        "            if not healthy and time.time() - NOTEBOOK_START_TIME > health_deadline_s:\n"
        "                print(f'ours (--fail-fast): the server is not healthy {health_deadline_s / 60:g} min after the '\n"
        "                      'start; ending this test arm', flush=True)\n"
        "                os._exit(3)\n"
        "if not TRUE_SUBMISSION:\n"
        "    import threading as _ours_threading\n"
        "    _ours_threading.Thread(target=_ours_server_watch, daemon=True, name='ours-fail-fast').start()\n"
        "    print('ours (--fail-fast): server watchdog on')\n"
        f"{FAIL_FAST_END}\n")


INPUT_LITERAL =re.compile(r"""(['"])(/kaggle/input/(?:datasets|competitions|models)/[^'"]+)\1""")
INPUT_HELPER = (
    "# >>> ours (--input-fallback): Kaggle mounts inputs as /kaggle/input/{datasets/<owner>,competitions}/<slug> or\n"
    "# as /kaggle/input/<slug>; GPU sessions of these notebooks got either on 2026-10-07. Use what this one has.\n"
    "import os as _ours_os\n"
    "def _ours_input(path):\n"
    "    if _ours_os.path.exists(path):\n"
    "        return path\n"
    "    parts = path.split('/')  # '', 'kaggle', 'input', kind, ...\n"
    "    if len(parts) > 5 and parts[3] == 'datasets':\n"
    "        alt = '/'.join(parts[:3] + parts[5:])\n"
    "    elif len(parts) > 4 and parts[3] == 'competitions':\n"
    "        alt = '/'.join(parts[:3] + parts[4:])\n"
    "    else:\n"
    "        return path\n"
    "    if _ours_os.path.exists(alt):\n"
    "        print(f'ours: {path} -> {alt}')\n"
    "        return alt\n"
    "    return path\n"
    "# <<< ours (--input-fallback)\n"
)


def _input_fallback(sources: list[str], code: list[bool], setup: int) -> tuple[list[str], int]:
    """Wrap every /kaggle/input/{datasets,competitions,models}/... string literal in a code cell in _ours_input(),
    defined at the top of cell 4 (which runs before any cell that reads an input)."""
    out, n = list(sources), 0
    for i, (src, is_code) in enumerate(zip(sources, code)):
        if not is_code:
            continue
        found = INPUT_LITERAL.findall(src)
        if found and i < setup:
            raise SystemExit(f"--input-fallback: cell {i} reads an input before cell 4 defines the helper")
        out[i], k = INPUT_LITERAL.subn(lambda m: f"_ours_input({m.group(0)})", src)
        n += k
    out[setup] = INPUT_HELPER + out[setup]
    return out, n


def _wait_code(seconds: float) -> str:
    """Cell-4 lines (before his bundle copy) that wait for the four inputs his cell reads.

    On 2026-10-07 four of eight GPU sessions of these notebooks failed 6 s in at that copy ("cp: cannot stat
    .../taaf-kaggle-source-bundle-copy") while a CPU session with the same sources saw every input mounted; waiting
    costs nothing when they are there and turns a silent 6-second failure into a wait or a listing.
    """
    return (
        "# >>> ours (--wait-inputs): wait for the mounted inputs instead of failing at the copy below\n"
        "_ours_need = [ORIG_BUNDLE_DIR + '/src', WHEELHOUSE_DIR + '/wheels', MODEL_DIR, DRAFT_MODEL_DIR]\n"
        "_ours_t0 = time.time()\n"
        "while [p for p in _ours_need if not os.path.isdir(p)]:\n"
        f"    if time.time() - _ours_t0 > {float(seconds)!r}:\n"
        "        for _ours_root, _ours_dirs, _ours_files in os.walk('/kaggle/input'):\n"
        "            if _ours_root.count(os.sep) <= 5:\n"
        "                print(_ours_root, sorted(_ours_dirs)[:8], len(_ours_files))\n"
        "        raise RuntimeError('inputs not mounted after "
        f"{float(seconds):g} s: ' + str([p for p in _ours_need if not os.path.isdir(p)]))\n"
        "    time.sleep(5)\n"
        "print(f'ours: inputs mounted after {time.time() - _ours_t0:.0f} s')\n"
        "# <<< ours (--wait-inputs)\n"
    )


def _probe_spec(folder: Path) -> dict:
    """The probe dataset: its Kaggle id (dataset-metadata.json) and data files with sha256 (manifest.json)."""
    try:
        meta = json.loads((folder / "dataset-metadata.json").read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--probe {folder}: needs dataset-metadata.json and manifest.json ({exc})") from None
    dataset = str(meta.get("id", ""))
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9-]*", dataset):
        raise SystemExit(f"--probe {folder}: dataset-metadata.json id {dataset!r} is not owner/slug")
    files = {str(f.get("name")): str(f.get("sha256")) for f in manifest.get("files") or []}
    if not files or not all(re.fullmatch(r"[\w.-]+", n) and re.fullmatch(r"[0-9a-f]{64}", s) for n, s in files.items()):
        raise SystemExit(f"--probe {folder}: manifest.json lists no data files with sha256")
    for name, expected in files.items():  # the dataset folder itself: the manifest must describe what is uploaded
        local = folder / name
        if local.is_file() and hashlib.sha256(local.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"--probe {folder}: {name} does not match manifest.json (rerun scripts/fidelity_sample.py)")
    requests = int(manifest.get("requests") or sum(int(f.get("requests") or 0) for f in manifest["files"]))
    return {"dataset": dataset, "files": files, "requests": requests}


def _probe_code(spec: dict, arm: str, num_experts: int | None, build_info: dict) -> tuple[str, str, str]:
    """(preflight cell after cell 4, probe cell replacing the benchmark cell, its markdown header)."""
    path = f"/kaggle/input/datasets/{spec['dataset']}"
    p = PROBE_PARAMS
    preflight = (
        f"{PROBE_BEGIN}: load the fidelity probe (scripts/fidelity_probe.py, written by the cell above) and check its\n"
        f"# prompts ({spec['dataset']}) before the server starts; stops here if they are missing or differ.\n"
        "import importlib.util as _probe_util\n"
        f"_probe_spec = _probe_util.spec_from_file_location('arc3_fidelity_probe', {PROBE_FILE!r})\n"
        "arc3_probe = _probe_util.module_from_spec(_probe_spec)\n"
        "_probe_spec.loader.exec_module(arc3_probe)\n"
        f"PROBE_FILES = arc3_probe.find_dataset([_ours_input({path!r})], {spec['files']!r},\n"
        f"                                      wait_s={p['data_wait_s']})\n"
        f"{PROBE_END}\n")
    probe = (
        f"{PROBE_BEGIN}: the fidelity probe replaces the benchmark (scripts/fidelity_probe.py;\n"
        "# docs/research/beat-tufa/fidelity-probe.md). Every sampled request, greedy, with logprobs, one at a time and\n"
        "# then several in flight (prefix cache flushed before each pass); writes /kaggle/working/fidelity.json.\n"
        f"# Raises if the server dies or is not healthy {p['health_minutes']} min after the notebook started.\n"
        "if TRUE_SUBMISSION:\n"
        "    raise RuntimeError('a fidelity-probe notebook is not a submission')\n"
        "arc3_probe.run(\n"
        "    PROBE_FILES, WORKING_DIR / 'fidelity.json',\n"
        "    base_url=f'http://{SERVED_MODEL_HOST}:{SERVED_MODEL_PORT}', model=SERVED_MODEL_NAME,\n"
        f"    arm={arm!r}, expect_num_experts={num_experts!r},\n"
        f"    build={build_info!r},\n"
        f"    max_tokens={p['max_tokens']}, top_logprobs={p['top_logprobs']}, concurrency={p['concurrency']}, "
        f"max_minutes={p['max_minutes']},\n"
        "    alive=lambda: proc.poll() is None, on_failure=show_log_tail,\n"
        f"    health_deadline=NOTEBOOK_START_TIME + {p['health_minutes']} * 60)\n"
        f"{PROBE_END}\n")
    markdown = (
        "## 9. Fidelity probe (ours, replaces the benchmark)\n\n"
        f"Replays the {spec['requests']} sampled ARC agent requests of `{spec['dataset']}` through the server: "
        f"greedy (temperature 0), {p['max_tokens']} tokens, logprobs with the top {p['top_logprobs']}, first one at a "
        f"time and then {p['concurrency']} in flight, and writes `fidelity.json` "
        "(scripts/fidelity_probe.py, read by scripts/fidelity_compare.py).")
    return preflight, probe, markdown


def build(out: Path, slug: str, full25: float | None = None, env: dict[str, str] | None = None,
          note: str = "", *, env_add: dict[str, str] | None = None, cfg: dict[str, str] | None = None,
          server_env: dict[str, str] | None = None, patches: list[Path] | tuple = (), apply_check: bool = True,
          his_repo: Path | None = None, bundle: Path | None = None, base: str = "franzen",
          reap_kept: Path | None = None, wait_inputs: float | None = None,
          input_fallback: bool = False, probe: Path | None = None, model: str | None = None,
          reap_verify: bool = True, fail_fast: bool = False, hot_tokens: Path | None = None,
          compact: bool = False, draft: str | None = None, draft_manifest: Path | None = None) -> list[str]:
    if (draft is None) != (draft_manifest is None):
        raise SystemExit("--draft and --draft-manifest go together")
    draft_spec = _draft_spec(draft, Path(draft_manifest)) if draft is not None else None
    if model is not None and model not in MODELS:
        raise SystemExit(f"--model {model!r}: unknown (known: {', '.join(sorted(MODELS))})")
    if not reap_verify and not reap_kept:
        raise SystemExit("--reap-no-verify only applies to --reap-kept")
    if model is not None and reap_kept and reap_verify:
        import sglang_reap_patch
        if sglang_reap_patch.meta_path(Path(reap_kept)).is_file():
            raise SystemExit("--model with --reap-kept: the list's .meta.json holds the router sha256 of Intel's "
                             "checkpoint, so the server would refuse another one; add --reap-no-verify to drop the "
                             "same expert ids without that check")
    if fail_fast and (full25 is None or probe is not None):
        raise SystemExit("--fail-fast is for test arms: it needs --full25 and does not combine with --probe (which "
                         "has its own health checks)")
    probe_spec = None
    if probe is not None:
        if not input_fallback:
            raise SystemExit("--probe needs --input-fallback (Kaggle mounts inputs in two layouts; lesson 0030)")
        if full25 is not None or patches:
            raise SystemExit("--probe replaces the benchmark run; --full25 and --patch would change nothing it runs")
        for key in ("SPEC_ACCEPT_SINGLE", "SPEC_ACCEPT_ACC"):
            if key in (cfg or {}) and _safe_value(cfg[key]) != 1.0:
                raise SystemExit(f"--probe: --cfg {key} must stay 1.0 (greedy outputs only compare under lossless "
                                 "speculative decoding)")
        probe_spec = _probe_spec(Path(probe))
    if base == "franzen":
        path, sha, demo_anchor, budget_anchor = BASE, BASE_SHA256, DEMO_ANCHOR, BUDGET_ANCHOR
    elif base == "dprime":
        path, sha, demo_anchor, budget_anchor = DPRIME, DPRIME_SHA256, DPRIME_DEMO_ANCHOR, DPRIME_BUDGET_ANCHOR
    else:
        raise SystemExit(f"unknown --base {base!r}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise SystemExit(f"{path} is not the vendored copy (sha256 differs); it must stay unmodified")
    nb = json.loads(raw)
    nb.setdefault("metadata", {}).setdefault("kaggle", {})["accelerator"] = "nvidiaRtxPro6000"  # D' lacks it
    cells = nb["cells"]
    sources = ["".join(c["source"]) for c in cells]
    code = [c["cell_type"] == "code" for c in cells]
    changes: list[str] = []

    if full25 is not None:
        i = _one_cell(sources, code, demo_anchor, "--full25")
        s = _replace_once(sources[i], demo_anchor, "demo_excluded_games = []  # ours (--full25): all 25 public games",
                          "demo")
        sources[i] = _replace_once(s, budget_anchor,
                                   f"        bm.solver.max_runtime_s_per_game = {float(full25)!r}*60  # ours (--full25)",
                                   "budget")
        changes.append(f"Save & Run plays all 25 public games, {full25:g} min per game (competition rerun unchanged)")

    setup = _one_cell(sources, code, SETUP_ANCHOR, "cell 4")
    if model is not None:  # before --input-fallback, which wraps the literal this replaces
        sources[setup] = _replace_once(sources[setup], MODEL_LINE, _model_code(model), "--model")
        changes.append(f"serves {MODELS[model]['what']} instead of Intel's W4A16 (the MTP draft stays his albucino "
                       f"checkpoint); cell 4 links its {len(MODELS[model]['sources'])} Kaggle model instances into "
                       f"{MODEL_VIEW}")
    if draft_spec is not None:  # before --input-fallback too
        sources[setup] = _replace_once(sources[setup], DRAFT_LINE, _draft_code(draft_spec), "--draft")
        changes.append(f"MTP draft: {draft_spec['kernel']}'s output {draft_spec['subdir']}/ (a kernel source, either "
                       f"mount layout; manifest sha256 {draft_spec['manifest_sha'][:12]}, {DRAFT_DENSE} "
                       f"{draft_spec['dense_sha'][:12]}, both checked in cell 4) instead of his albucino checkpoint")
    if input_fallback:
        sources, n = _input_fallback(sources, code, setup)
        changes.append(f"resolves its {n} /kaggle/input paths in either Kaggle mount layout")
    if wait_inputs is not None:
        sources[setup] = _replace_once(sources[setup], WAIT_ANCHOR, _wait_code(wait_inputs) + WAIT_ANCHOR,
                                       "--wait-inputs")
        changes.append(f"waits up to {wait_inputs:g} s for its mounted inputs before copying the bundle")
    sources[setup], done = _set_env(sources[setup], env or {})
    changes += done
    sources[setup], done = _add_env(sources[setup], env_add or {})
    changes += done
    reap_files: list[tuple[str, str]] = []
    hot_writer = None
    if cfg or server_env or reap_kept or hot_tokens:
        launch = _one_cell(sources, code, LAUNCH_ANCHOR, "cell 12")
        sources[launch], done = _set_cfg(sources[launch], cfg or {})
        changes += done
        sources[launch], done = _set_server_env(sources[launch], server_env or {})
        changes += done
        if reap_kept:
            sources[launch], reap_files, done = _reap(sources[launch], Path(reap_kept), verify=reap_verify)
            changes.append(done)
        if hot_tokens:
            sources[launch], hot_writer, done = _hot_tokens(sources[launch], Path(hot_tokens))
            changes.append(done)

    preflight = None
    if probe_spec is not None:
        arm, num_experts = "base", None
        if reap_kept:
            import sglang_reap_patch
            num_experts = len(sglang_reap_patch.load_kept(Path(reap_kept))[0])
            arm = f"reap{num_experts}"
        if hot_tokens:
            arm += "-" + Path(hot_tokens).stem.removeprefix("hot_tokens_64k_").replace("_", "-")
        if draft_spec is not None:
            arm += "-draft"
        run_cell = _one_cell(sources, code, PROBE_RUN_ANCHOR, "--probe")
        header = [i for i, s in enumerate(sources) if not code[i] and PROBE_MD_ANCHOR in s]
        if len(header) != 1:
            raise SystemExit(f"--probe: markdown {PROBE_MD_ANCHOR!r} found in cells {header} (expected one)")
        probe_sha = hashlib.sha256(PROBE_SCRIPT.read_bytes()).hexdigest()[:12]
        build_info = {"builder": "scripts/build_franzen_nb.py", "base": base, "slug": slug,
                      "dataset": probe_spec["dataset"], "data_files": probe_spec["files"], "probe_sha256": probe_sha,
                      "reap_kept": (f"{Path(reap_kept).name} sha256 "
                                    f"{hashlib.sha256(Path(reap_kept).read_bytes()).hexdigest()[:12]}"
                                    if reap_kept else None),
                      "cfg": dict(cfg or {}), "server_env": dict(server_env or {}),
                      "hot_tokens": (f"{Path(hot_tokens).name} sha256 "
                                     f"{hashlib.sha256(Path(hot_tokens).read_bytes()).hexdigest()[:12]}"
                                     if hot_tokens else None),
                      "draft": ({k: draft_spec[k] for k in ("kernel", "subdir", "manifest_sha", "dense_sha")}
                                if draft_spec is not None else None)}
        preflight, sources[run_cell], sources[header[0]] = _probe_code(probe_spec, arm, num_experts, build_info)
        p = PROBE_PARAMS
        changes.append(
            f"fidelity probe instead of the benchmark ({arm} arm): replays the {probe_spec['requests']} requests of "
            f"{probe_spec['dataset']} ("
            + ", ".join(f"{n} sha256 {s[:12]}" for n, s in probe_spec["files"].items())
            + f"; checked right after cell 4) greedy (temperature 0, max_tokens {p['max_tokens']}, logprobs with top "
            f"{p['top_logprobs']}), one at a time and then {p['concurrency']} in flight with the prefix cache flushed "
            f"before each pass, writes fidelity.json and raises if the server dies or is not healthy "
            f"{p['health_minutes']} min after the start (scripts/fidelity_probe.py sha256 {probe_sha}); the cells "
            "after the benchmark cell are dropped")

    if fail_fast:
        run_cell = _one_cell(sources, code, PROBE_RUN_ANCHOR, "--fail-fast")
        at = sources[run_cell].rfind("\n", 0, sources[run_cell].index(PROBE_RUN_ANCHOR)) + 1
        sources[run_cell] = sources[run_cell][:at] + _fail_fast_code() + sources[run_cell][at:]
        changes.append(f"test arm fails fast: a watchdog started before the benchmark exits the kernel when the SGLang "
                       f"server process has exited or is not healthy {FAIL_FAST_HEALTH_MINUTES} min after the start "
                       "(off in a competition rerun)")

    new_cells = []
    if patches:
        texts = []
        for n, path in enumerate(patches, start=1):
            text = Path(path).read_text(encoding="utf-8")
            if not text.endswith("\n"):
                text += "\n"
            files = _check_patch_text(Path(path), text)
            texts.append((_patch_name(n, Path(path)), text, files))
        if apply_check:
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    franzen_tree.notebook_bundle(Path(tmp) / "b", [(name, text) for name, text, _ in texts],
                                                 his_repo=his_repo, bundle=bundle)
                except franzen_tree.TreeError as exc:
                    raise SystemExit(f"--patch: the notebook's tree could not be patched (pass --no-apply-check only "
                                     f"when neither his repo nor the bundle is available):\n{exc}") from None
        his = _one_cell(sources, code, HIS_PATCH_ANCHOR, "his patch cell")
        for name, text, files in texts:
            cell = copy.deepcopy(cells[his])
            cell["outputs"], cell["execution_count"] = [], None
            cell["source"] = _file_cell(f"/kaggle/{name}", text, compact)
            new_cells.append(cell)
            changes.append(f"harness patch {name} ({len(files)} file(s): {', '.join(sorted(set(files)))}; "
                           f"sha256 {hashlib.sha256(text.encode()).hexdigest()[:12]})")
        sources[setup] = _replace_once(sources[setup], APPLY_ANCHOR,
                                       APPLY_ANCHOR + _apply_code([name for name, _, _ in texts]), "--patch apply")

    for cell, s in zip(cells, sources):
        cell["source"] = s.splitlines(keepends=True)
    for cell in new_cells:
        cell["source"] = cell["source"].splitlines(keepends=True)
    if new_cells:
        his = next(i for i, c in enumerate(cells) if "".join(c["source"]).startswith(HIS_PATCH_ANCHOR))
        cells[his + 1:his + 1] = new_cells
    if reap_files or hot_writer:  # cells right before the launcher cell, which reads their files
        launch = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and LAUNCH_ANCHOR in "".join(c["source"]))
        written = []
        for text in [_file_cell(path, text, compact) for path, text in reap_files] + ([hot_writer] if hot_writer else []):
            cell = copy.deepcopy(cells[launch])
            cell["outputs"], cell["execution_count"] = [], None
            cell["source"] = text.splitlines(keepends=True)
            written.append(cell)
        cells[launch:launch] = written
    if preflight is not None:  # drop what follows the probe cell; the probe module and the data check after cell 4
        probe_cell = next(i for i, c in enumerate(cells) if c["cell_type"] == "code"
                          and "arc3_probe.run(" in "".join(c["source"]))
        del cells[probe_cell + 1:]
        setup_cell = next(i for i, c in enumerate(cells) if c["cell_type"] == "code"
                          and SETUP_ANCHOR in "".join(c["source"]))
        added = []
        for text in (_file_cell(PROBE_FILE, PROBE_SCRIPT.read_text(), compact), preflight):
            cell = copy.deepcopy(cells[setup_cell])
            cell["outputs"], cell["execution_count"] = [], None
            cell["source"] = text.splitlines(keepends=True)
            added.append(cell)
        cells[setup_cell + 1:setup_cell + 1] = added
    if compact and (texts_written := len(patches) + len(reap_files) + (preflight is not None)):
        changes.append(f"the {texts_written} file(s) our cells write are shipped zlib-compressed (--compact; the same "
                       "bytes, sha256 checked)")
    if changes or note:
        cells[0]["source"] = ["".join(cells[0]["source"]) + "\n\n**Our arm (scottmahony, built by "
                              f"scripts/build_franzen_nb.py from the unmodified {base} notebook):** "
                              + ("; ".join(changes) or "unchanged") + (f". {note}" if note else "") + "\n"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    sources_meta = copy.deepcopy(SOURCES)
    if model is not None:
        i = sources_meta["model_sources"].index(INTEL_SOURCE)
        sources_meta["model_sources"][i:i + 1] = MODELS[model]["sources"]
    if probe_spec is not None:
        sources_meta["dataset_sources"].append(probe_spec["dataset"])
    if draft_spec is not None:
        sources_meta["model_sources"].remove(ALBUCINO_SOURCE)
        sources_meta["kernel_sources"].append(draft_spec["kernel"])
    meta = {"id": f"scottmahony/{slug}", "title": slug.replace("-", " "), "code_file": f"{slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": False, "keywords": [], **sources_meta,
            "docker_image": IMAGE, "docker_image_pinning_type": "original", "machine_shape": MACHINE_SHAPE}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return changes


def _file_cell(path: str, text: str, compact: bool) -> str:
    """The source of a cell that writes TEXT to PATH: ``%%writefile PATH``, or with compact=True a code cell writing
    the same bytes from a zlib-compressed base64 payload (franzen_tree.written_file reads both back)."""
    source = f"%%writefile {path}\n{text}"
    if not compact:
        return source
    data = franzen_tree.writefile_body(source)[1].encode("utf-8")
    packed = base64.b64encode(zlib.compress(data, 9)).decode()
    lines = "\n".join(packed[i:i + 120] for i in range(0, len(packed), 120))
    sha = hashlib.sha256(data).hexdigest()
    cell = (f"{COMPACT_BEGIN}: {path}, {len(data)} bytes (sha256 {sha[:12]}), shipped compressed: Kaggle refuses a\n"
            "# notebook near 1 MB."
            + COMPACT_WRITER
            + f"{franzen_tree.PACKED_WRITER}({path!r}, \"\"\"\n{lines}\n\"\"\", {sha!r})\n"
            f"{COMPACT_END}\n")
    if franzen_tree.written_file(cell) != (path, data.decode("utf-8")):
        raise SystemExit(f"--compact: the cell for {path} does not round-trip")
    return cell


def _pairs(values: list[str], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"{flag} {item!r}: expected KEY=VALUE")
        key, value = item.split("=", 1)
        if key in out:
            raise SystemExit(f"{flag} {key}: given twice")
        out[key] = value
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--full25", type=float, default=None, metavar="MIN")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--env-add", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--cfg", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--server-env", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--patch", action="append", default=[], type=Path, metavar="FILE")
    ap.add_argument("--no-apply-check", action="store_true")
    ap.add_argument("--his-repo", type=Path, default=None, help="for the apply check (default: scripts/franzen_tree.py)")
    ap.add_argument("--bundle", type=Path, default=None, help="for the apply check: a download of the bundle dataset")
    ap.add_argument("--note", default="")
    ap.add_argument("--base", choices=["franzen", "dprime"], default="franzen",
                    help="franzen (kaggle/franzen/, default) or dprime (kaggle/dprime/: his notebook + the D' slot priority)")
    ap.add_argument("--reap-kept", type=Path, default=None, metavar="FILE",
                    help="serve with only these routed experts (kaggle/franzen/reap448_kept_experts.json)")
    ap.add_argument("--input-fallback", action="store_true",
                    help="resolve /kaggle/input paths in either mount layout (datasets/<owner>/<slug> or <slug>)")
    ap.add_argument("--wait-inputs", type=float, default=None, metavar="SECONDS",
                    help="cell 4 waits up to SECONDS for its mounted inputs before copying the bundle")
    ap.add_argument("--probe", type=Path, default=None, metavar="DIR",
                    help="fidelity probe instead of the benchmark; DIR has the probe dataset's dataset-metadata.json "
                         "and manifest.json (kaggle/fidelity/)")
    ap.add_argument("--model", choices=sorted(MODELS), default=None,
                    help="serve this checkpoint instead of Intel's W4A16 (the MTP draft stays his)")
    ap.add_argument("--reap-no-verify", action="store_true",
                    help="--reap-kept without the router sha256 check (for a derivative checkpoint, e.g. --model)")
    ap.add_argument("--fail-fast", action="store_true",
                    help="test arms (--full25): exit the kernel when the SGLang server is gone")
    ap.add_argument("--hot-tokens", type=Path, default=None, metavar="FILE",
                    help="the MTP draft's FR-Spec map (kaggle/franzen/hot_tokens_64k_arc.pt)")
    ap.add_argument("--draft", default=None, metavar="OWNER/KERNEL[/SUBDIR]",
                    help="serve MTP session A's fine-tuned draft (its output's mtp-draft/) instead of albucino")
    ap.add_argument("--draft-manifest", type=Path, default=None, metavar="FILE",
                    help="that output's arc3-draft-manifest.json, as pulled (pins the version the notebook accepts)")
    ap.add_argument("--compact", action="store_true",
                    help="ship the files our cells write zlib-compressed (Kaggle refuses notebooks near 1 MB)")
    args = ap.parse_args()
    changes = build(args.out, args.slug, args.full25, _pairs(args.env, "--env"), args.note,
                    env_add=_pairs(args.env_add, "--env-add"), cfg=_pairs(args.cfg, "--cfg"),
                    server_env=_pairs(args.server_env, "--server-env"), patches=args.patch,
                    apply_check=not args.no_apply_check, his_repo=args.his_repo, bundle=args.bundle, base=args.base,
                    reap_kept=args.reap_kept, wait_inputs=args.wait_inputs, input_fallback=args.input_fallback,
                    probe=args.probe, model=args.model, reap_verify=not args.reap_no_verify, fail_fast=args.fail_fast,
                    hot_tokens=args.hot_tokens, compact=args.compact, draft=args.draft,
                    draft_manifest=args.draft_manifest)
    print(f"built {args.out / (args.slug + '.ipynb')}: {changes or 'unchanged'}")


if __name__ == "__main__":
    main()
