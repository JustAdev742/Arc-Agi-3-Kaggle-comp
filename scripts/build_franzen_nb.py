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

Every change is anchored on text that must occur exactly once, and listed in the first markdown cell. Writes
``<out>/<slug>.ipynb`` and ``<out>/kernel-metadata.json`` (private, internet off, RTX PRO 6000).
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
import sys
import tempfile
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
    changes = []
    for key, value in env.items():
        pattern = re.compile(rf"^(\s*)'{re.escape(key)}': ([^,\n]+),", re.M)
        hits = pattern.findall(cell)
        if len(hits) != 1:
            raise SystemExit(f"--env {key}: anchor found {len(hits)} times in cell 4 (expected once; --env-add adds "
                             f"a key)")
        new_value = _env_literal(value)
        cell = pattern.sub(lambda m, nv=new_value, k=key: f"{m.group(1)}'{k}': {nv},  # ours (--env)", cell)
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


def _reap(cell: str, kept_path: Path) -> tuple[str, list[tuple[str, str]], str]:
    """Cell 12 with the REAP steps, the (path, text) files the notebook must write first, and the change line."""
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
    if meta.is_file():
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
              f"{override} --speculative-draft-model-override-args {{}} (the MTP draft keeps all experts)")
    return cell, files, change


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


def build(out: Path, slug: str, full25: float | None = None, env: dict[str, str] | None = None,
          note: str = "", *, env_add: dict[str, str] | None = None, cfg: dict[str, str] | None = None,
          server_env: dict[str, str] | None = None, patches: list[Path] | tuple = (), apply_check: bool = True,
          his_repo: Path | None = None, bundle: Path | None = None, base: str = "franzen",
          reap_kept: Path | None = None, wait_inputs: float | None = None) -> list[str]:
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
    if wait_inputs is not None:
        sources[setup] = _replace_once(sources[setup], WAIT_ANCHOR, _wait_code(wait_inputs) + WAIT_ANCHOR,
                                       "--wait-inputs")
        changes.append(f"waits up to {wait_inputs:g} s for its mounted inputs before copying the bundle")
    sources[setup], done = _set_env(sources[setup], env or {})
    changes += done
    sources[setup], done = _add_env(sources[setup], env_add or {})
    changes += done
    reap_files: list[tuple[str, str]] = []
    if cfg or server_env or reap_kept:
        launch = _one_cell(sources, code, LAUNCH_ANCHOR, "cell 12")
        sources[launch], done = _set_cfg(sources[launch], cfg or {})
        changes += done
        sources[launch], done = _set_server_env(sources[launch], server_env or {})
        changes += done
        if reap_kept:
            sources[launch], reap_files, done = _reap(sources[launch], Path(reap_kept))
            changes.append(done)

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
            cell["source"] = f"%%writefile /kaggle/{name}\n{text}"
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
    if reap_files:  # %%writefile cells right before the launcher cell, which reads them
        launch = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and LAUNCH_ANCHOR in "".join(c["source"]))
        written = []
        for path, text in reap_files:
            cell = copy.deepcopy(cells[launch])
            cell["outputs"], cell["execution_count"] = [], None
            cell["source"] = f"%%writefile {path}\n{text}".splitlines(keepends=True)
            written.append(cell)
        cells[launch:launch] = written
    if changes or note:
        cells[0]["source"] = ["".join(cells[0]["source"]) + "\n\n**Our arm (scottmahony, built by "
                              f"scripts/build_franzen_nb.py from the unmodified {base} notebook):** "
                              + ("; ".join(changes) or "unchanged") + (f". {note}" if note else "") + "\n"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta = {"id": f"scottmahony/{slug}", "title": slug.replace("-", " "), "code_file": f"{slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": False, "keywords": [], **SOURCES,
            "docker_image": IMAGE, "docker_image_pinning_type": "original", "machine_shape": MACHINE_SHAPE}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return changes


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
    ap.add_argument("--wait-inputs", type=float, default=None, metavar="SECONDS",
                    help="cell 4 waits up to SECONDS for its mounted inputs before copying the bundle")
    args = ap.parse_args()
    changes = build(args.out, args.slug, args.full25, _pairs(args.env, "--env"), args.note,
                    env_add=_pairs(args.env_add, "--env-add"), cfg=_pairs(args.cfg, "--cfg"),
                    server_env=_pairs(args.server_env, "--server-env"), patches=args.patch,
                    apply_check=not args.no_apply_check, his_repo=args.his_repo, bundle=args.bundle, base=args.base,
                    reap_kept=args.reap_kept, wait_inputs=args.wait_inputs)
    print(f"built {args.out / (args.slug + '.ipynb')}: {changes or 'unchanged'}")


if __name__ == "__main__":
    main()
