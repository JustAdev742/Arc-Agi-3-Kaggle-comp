#!/usr/bin/env python
"""Daniel Franzen's harness tree exactly as his Milestone 2 notebook builds it, locally (no Kaggle, no GPU).

His notebook (kaggle/franzen/, Apache-2.0) copies Tufa Labs' source bundle (Kaggle dataset
dfranzen/taaf-kaggle-source-bundle-copy, MIT code) to /kaggle/taaf-kaggle-source-share and, in its src/, runs
``git apply --include=ARC3-Inference/* -v /kaggle/harness-changes.patch`` (the patch is cell 2). His GitHub repo
(da-fr/arc-agi-3-solution @ 10882e3) holds his patched ARC3-Inference, but it is NOT the tree the notebook runs:
the bundle is Tufa's June Kaggle snapshot, which differs from Tufa's GitHub release (the base of his repo) in
inference/framework/run.py (48 more lines, so his patch applies there "with offset 48 lines" in his Kaggle log),
inference/tools/{eval,significance,traces}.py, two extra inference/utils/rearc_*.py, seven tufa-arc-agi-framework
files (game_api.py, competition_arcade.py, deploy*.py, standard_benchmarks.py) and the pyproject/uv.lock/README/
Makefile/configs/viewer files. Everything else under inference/agent and inference/utils is identical.

So this script rebuilds the bundle byte-for-byte from his repo: reverse his patch (giving Tufa's GitHub files),
drop CONFIGURATION.md, apply the vendored delta kaggle/franzen/bundle/his-repo-to-bundle.diff, add the bundle's
root files (pickled benchmark and deployment target, kaggle/franzen/bundle/root/), and check every file against
the sha256 manifest of the real dataset (kaggle/franzen/bundle/MANIFEST.json, downloaded 2026-10-02). Then it runs
his notebook's own command for his patch, checks the result against the manifest again, and applies our patches
the way scripts/build_franzen_nb.py --patch makes the notebook apply them. A downloaded copy of the dataset can
replace his repo as the source (--bundle DIR); both are verified against the same manifest.

    .venv/bin/python scripts/franzen_tree.py build DIR [--patch P ...]   # git repo of the notebook's src/ to edit
    .venv/bin/python scripts/franzen_tree.py diff DIR > ours.patch          # our edits as a patch for --patch
    .venv/bin/python scripts/franzen_tree.py check --patch P ...            # does P apply on top of his patch?
    .venv/bin/python scripts/franzen_tree.py bundle DIR [--patch P ...]     # /kaggle/taaf-kaggle-source-share after cell 4
    .venv/bin/python scripts/franzen_tree.py vendor --bundle DL             # maintainer: regenerate kaggle/franzen/bundle/

``build`` makes DIR a git repository of the notebook's src/ (ARC3-Inference/, tufa-arc-agi-framework/) with
one commit per stage, tagged ``bundle`` (Tufa's bundle), ``franzen`` (after his patch) and ``ours`` (after the
--patch files, if any). Edit files there, then ``diff`` prints ``git diff HEAD`` (new files included) with the
a/ b/ prefixes the notebook's ``git apply`` expects.

His repo is found at --his-repo, else $FRANZEN_REPO, else /home/user/da-fr/arc-agi-3-solution. To get it:
``git clone https://github.com/da-fr/arc-agi-3-solution`` and check out 10882e3. To use the dataset instead:
``kaggle datasets download dfranzen/taaf-kaggle-source-bundle-copy --unzip -p DL``.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "kaggle" / "franzen" / "arc-agi-3-milestone-2-solution.ipynb"
DATA = ROOT / "kaggle" / "franzen" / "bundle"
MANIFEST = DATA / "MANIFEST.json"
DELTA = DATA / "his-repo-to-bundle.diff"
DEFAULT_HIS_REPO = Path("/home/user/da-fr/arc-agi-3-solution")
REPOS = ("ARC3-Inference", "tufa-arc-agi-framework")
HIS_INCLUDE = "ARC3-Inference/*"  # cell 4: git apply --include=ARC3-Inference/* -v /kaggle/harness-changes.patch
IGNORED = {".git", "__pycache__", ".venv", ".ruff_cache", ".pytest_cache", ".cache", "node_modules", ".DS_Store"}


class TreeError(RuntimeError):
    """The tree could not be built exactly (missing source, hash mismatch, a patch that does not apply)."""


# --- the notebook's cells ----------------------------------------------------------------------------------------


def writefile_body(source: str) -> tuple[str, str]:
    """(path, text) that IPython's ``%%writefile path`` writes for this cell source.

    IPython appends a newline to a cell that lacks one before splitting off the magic line, so the written file is
    the rest of the cell plus a final newline (his cell 2 is stored without it).
    """
    first, _, body = source.partition("\n")
    if not first.startswith("%%writefile "):
        raise TreeError(f"not a %%writefile cell: {first[:60]!r}")
    if not body.endswith("\n"):
        body += "\n"
    return first.split(None, 1)[1].strip(), body


def notebook_cells(notebook: Path = NOTEBOOK) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(Path(notebook).read_text())["cells"]]


def his_patch_text(notebook: Path = NOTEBOOK) -> str:
    """His harness patch, as cell 2 writes it to /kaggle/harness-changes.patch."""
    for source in notebook_cells(notebook):
        if source.startswith("%%writefile /kaggle/harness-changes.patch"):
            return writefile_body(source)[1]
    raise TreeError(f"{notebook}: no %%writefile /kaggle/harness-changes.patch cell")


PACKED_WRITER = "_ours_write_packed"  # scripts/build_franzen_nb.py --compact


def written_file(source: str) -> tuple[str, str] | None:
    """(path, text) a cell writes: a ``%%writefile`` cell, or a scripts/build_franzen_nb.py --compact cell (the bytes
    a %%writefile cell would write, zlib-compressed in base64, checked by sha256); None for any other cell."""
    if source.startswith("%%writefile "):
        return writefile_body(source)
    if PACKED_WRITER + "(" not in source:
        return None
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == PACKED_WRITER]
    if len(calls) != 1 or len(calls[0].args) != 3:
        raise TreeError(f"--compact cell with {len(calls)} {PACKED_WRITER} calls (expected one with 3 arguments)")
    path, packed, sha = (ast.literal_eval(arg) for arg in calls[0].args)
    data = zlib.decompress(base64.b64decode(packed))
    if hashlib.sha256(data).hexdigest() != sha:
        raise TreeError(f"--compact cell for {path}: sha256 mismatch")
    return path, data.decode("utf-8")


def our_patch_cells(notebook: Path) -> list[tuple[str, str]]:
    """(path, text) of the /kaggle/ours-*.patch files scripts/build_franzen_nb.py --patch adds (as %%writefile cells,
    or --compact cells)."""
    files = [written_file(s) for s in notebook_cells(notebook)]
    return [f for f in files if f is not None and f[0].startswith("/kaggle/ours-")]


# --- files and hashes --------------------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    root = Path(root)
    out = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.is_file() and not (set(rel.parts) & IGNORED):
            out[rel.as_posix()] = sha256(path)
    return out


def _verify(root: Path, expected: dict[str, str], what: str) -> None:
    got = tree_hashes(root)
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    wrong = sorted(k for k in set(expected) & set(got) if expected[k] != got[k])
    if missing or extra or wrong:
        raise TreeError(f"{what} is not byte-identical to the manifest: missing {missing[:5]}, extra {extra[:5]}, "
                        f"different {wrong[:5]} ({len(missing)}/{len(extra)}/{len(wrong)})")


def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _copy_files(src_root: Path, dst_root: Path, rels) -> None:
    for rel in rels:
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_root / rel, dst)


def git_apply(cwd: Path, patch: Path, *extra: str) -> str:
    """Run ``git apply -v`` like the notebook does; raise TreeError unless every file named in the patch applied.

    GIT_CEILING_DIRECTORIES keeps git from finding an enclosing repository: inside one, git apply resolves paths
    from that repository's top and silently skips files outside the current directory.
    """
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(Path(cwd).resolve().parent), LC_ALL="C")
    r = subprocess.run(["git", "apply", *extra, "-v", str(Path(patch).resolve())], cwd=cwd, capture_output=True,
                       text=True, env=env)
    log = r.stdout + r.stderr
    text = Path(patch).read_text(encoding="utf-8")
    named = text.startswith("diff --git ") + text.count("\ndiff --git ")
    includes = [e.split("=", 1)[1] for e in extra if e.startswith("--include=")]
    if includes:
        import fnmatch
        names = [line.split()[2][2:] for line in text.splitlines() if line.startswith("diff --git ")]
        named = sum(any(fnmatch.fnmatch(n, pat) for pat in includes) for n in names)
    applied = log.count("Applied patch ")
    if r.returncode != 0 or applied != named:
        raise TreeError(f"git apply {' '.join(extra)} {Path(patch).name} in {cwd}: exit {r.returncode}, "
                        f"{applied} of {named} files applied\n{log}")
    return log


# --- the bundle ----------------------------------------------------------------------------------------------------


def his_repo_path(his_repo: Path | str | None = None) -> Path:
    return Path(his_repo or os.environ.get("FRANZEN_REPO") or DEFAULT_HIS_REPO)


def build_bundle(dest: Path, *, his_repo: Path | str | None = None, bundle: Path | str | None = None) -> Path:
    """Tufa's bundle as the notebook finds it in the dataset: DEST/{root files, src/...}, verified by sha256."""
    m = manifest()
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise TreeError(f"{dest} is not empty")
    dest.mkdir(parents=True, exist_ok=True)
    if bundle is not None:
        bundle = Path(bundle)
        _copy_files(bundle, dest, [k for k in m["bundle"] if (bundle / k).is_file()])
    else:
        repo = his_repo_path(his_repo)
        if not (repo / "ARC3-Inference").is_dir():
            raise TreeError(f"his repo not found at {repo}: clone https://github.com/da-fr/arc-agi-3-solution "
                            f"(commit {m['his_repo_commit']}) and pass --his-repo, or pass --bundle with a download "
                            f"of {m['bundle_dataset']}")
        differ = [k for k, h in m["his_repo"].items() if not (repo / k).is_file() or sha256(repo / k) != h]
        if differ:
            raise TreeError(f"{repo} is not his repo at {m['his_repo_commit']} ({len(differ)} files differ, e.g. "
                            f"{differ[:3]}); check that commit out, or pass --bundle")
        src = dest / "src"
        _copy_files(repo, src, m["his_repo"])
        with tempfile.TemporaryDirectory() as tmp:
            his = Path(tmp) / "his.patch"
            his.write_text(his_patch_text(), encoding="utf-8")
            git_apply(src, his, "-R", f"--include={HIS_INCLUDE}")
        for rel in m["remove"]:
            (src / rel).unlink()
        git_apply(src, DELTA)
        _copy_files(DATA / "root", dest, [k for k in m["bundle"] if not k.startswith("src/")])
    _verify(dest, m["bundle"], f"the bundle built in {dest}")
    return dest


def apply_his_patch(src: Path, his_patch: str | None = None) -> str:
    """Cell 4: ``git apply --include=ARC3-Inference/* -v /kaggle/harness-changes.patch`` in the bundle's src/."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "harness-changes.patch"
        path.write_text(his_patch if his_patch is not None else his_patch_text(), encoding="utf-8")
        log = git_apply(src, path, f"--include={HIS_INCLUDE}")
    if his_patch is None or his_patch == his_patch_text():
        _verify(src, manifest()["notebook_src"], f"{src} after his patch")
    return log


def apply_our_patch(src: Path, patch: Path | str, text: str | None = None) -> str:
    """Our notebook cell: ``git apply -v ours.patch`` in the same src/; every file it names must apply."""
    if text is None:
        return git_apply(src, Path(patch))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(patch).name
        path.write_text(text, encoding="utf-8")
        return git_apply(src, path)


def notebook_bundle(dest: Path, patches=(), *, his_repo=None, bundle=None, his_patch: str | None = None) -> dict:
    """/kaggle/taaf-kaggle-source-share as the notebook leaves it after cell 4 (his patch, then ours, in order).

    ``patches``: paths, or (name, text) pairs as extracted from a built notebook's %%writefile cells.
    """
    build_bundle(dest, his_repo=his_repo, bundle=bundle)
    logs = {"his": apply_his_patch(Path(dest) / "src", his_patch)}
    for p in patches:
        name, text = (p if isinstance(p, tuple) else (str(p), None))
        logs[Path(name).name] = apply_our_patch(Path(dest) / "src", name, text)
    return logs


# --- the editable repository -------------------------------------------------------------------------------------


def _git(repo: Path, *args: str, capture: bool = False) -> str:
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(Path(repo).resolve().parent), LC_ALL="C")
    cmd = ["git", "-c", "user.name=franzen_tree", "-c", "user.email=franzen_tree@localhost", "-c", "core.autocrlf=false",
           "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args]
    r = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise TreeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout if capture else ""


def _sync(src: Path, repo: Path) -> None:
    for child in repo.iterdir():
        if child.name != ".git":
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    for child in src.iterdir():
        (shutil.copytree if child.is_dir() else shutil.copy2)(child, repo / child.name)


def materialise(dest: Path, patches=(), *, his_repo=None, bundle=None) -> Path:
    """A git repository of the notebook's src/ tree with commits/tags bundle, franzen and (with patches) ours."""
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise TreeError(f"{dest} is not empty")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "bundle"
        build_bundle(work, his_repo=his_repo, bundle=bundle)
        src = work / "src"
        dest.mkdir(parents=True, exist_ok=True)
        _git(dest, "init", "-q")
        (dest / ".git" / "info").mkdir(parents=True, exist_ok=True)
        (dest / ".git" / "info" / "exclude").write_text("__pycache__/\n*.pyc\n.venv/\n")
        stages = [("bundle", "Tufa's source bundle (dataset dfranzen/taaf-kaggle-source-bundle-copy)", None)]
        stages.append(("franzen", "Franzen's harness patch (notebook cell 2, applied as cell 4 does)", "his"))
        stages += [(None, f"ours: {Path(p).name}", p) for p in patches]
        for tag, message, step in stages:
            if step == "his":
                apply_his_patch(src)
            elif step is not None:
                apply_our_patch(src, step)
            _sync(src, dest)
            _git(dest, "add", "-A")
            _git(dest, "commit", "-q", "--allow-empty", "-m", message)
            if tag:
                _git(dest, "tag", tag)
        if patches:
            _git(dest, "tag", "ours")
    return dest


def diff(repo: Path, base: str = "HEAD") -> str:
    """Our edits in REPO since BASE (new files included), in the form the notebook's git apply expects."""
    repo = Path(repo)
    _git(repo, "add", "-A", "-N")
    return _git(repo, "diff", "--no-color", "--no-ext-diff", "--no-renames", "--src-prefix=a/", "--dst-prefix=b/",
                base, "--", *REPOS, capture=True)


# --- maintainer: regenerate kaggle/franzen/bundle/ ----------------------------------------------------------------


def vendor(download: Path, his_repo: Path | str | None = None, commit: str = "10882e3") -> dict:
    """Rebuild MANIFEST.json, the delta and root/ from a download of the dataset and his repo at COMMIT."""
    download, repo = Path(download), his_repo_path(his_repo)
    bundle_hashes = tree_hashes(download)
    his_files = {k: v for k, v in tree_hashes(repo).items() if k.split("/", 1)[0] in REPOS}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        his = tmp / "his.patch"
        his.write_text(his_patch_text(), encoding="utf-8")
        base = tmp / "repo"
        _copy_files(repo, base, his_files)
        git_apply(base, his, "-R", f"--include={HIS_INCLUDE}")
        bundle_src = {k[4:] for k in bundle_hashes if k.startswith("src/")}
        remove = sorted(k for k in tree_hashes(base) if k not in bundle_src)
        for rel in remove:
            (base / rel).unlink()
        _git(base, "init", "-q")
        _git(base, "add", "-A")
        _git(base, "commit", "-q", "-m", "base")
        _sync(download / "src", base)
        _git(base, "add", "-A")
        delta = _git(base, "diff", "--cached", "--binary", "--full-index", "--no-color", "--no-ext-diff",
                     "--no-renames", "--src-prefix=a/", "--dst-prefix=b/", capture=True)
        nb_src = tmp / "nb"
        shutil.copytree(download / "src", nb_src)
        git_apply(nb_src, his, f"--include={HIS_INCLUDE}")
        notebook_src = tree_hashes(nb_src)
    DATA.mkdir(parents=True, exist_ok=True)
    DELTA.write_text(delta, encoding="utf-8")
    root = DATA / "root"
    shutil.rmtree(root, ignore_errors=True)
    _copy_files(download, root, [k for k in bundle_hashes if not k.startswith("src/")])
    data = {"bundle_dataset": "dfranzen/taaf-kaggle-source-bundle-copy", "downloaded": "2026-10-02",
            "his_repo": his_files, "his_repo_commit": commit, "his_patch_sha256": hashlib.sha256(
                his_patch_text().encode()).hexdigest(), "remove": remove, "bundle": bundle_hashes,
            "notebook_src": notebook_src}
    MANIFEST.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "bundle", "check"):
        p = sub.add_parser(name)
        if name != "check":
            p.add_argument("dir", type=Path)
        p.add_argument("--patch", action="append", default=[], type=Path, metavar="FILE")
        p.add_argument("--his-repo", type=Path, default=None)
        p.add_argument("--bundle", type=Path, default=None, help="a download of the dataset instead of his repo")
    p = sub.add_parser("diff")
    p.add_argument("dir", type=Path)
    p.add_argument("--base", default="HEAD", help="commit or tag to diff against (franzen: everything of ours)")
    p = sub.add_parser("vendor")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--his-repo", type=Path, default=None)
    args = ap.parse_args()
    try:
        if args.cmd == "build":
            materialise(args.dir, args.patch, his_repo=args.his_repo, bundle=args.bundle)
            print(f"built {args.dir}: tags {'bundle, franzen' + (', ours' if args.patch else '')}; edit, then "
                  f"scripts/franzen_tree.py diff {args.dir} > ours.patch")
        elif args.cmd == "bundle":
            logs = notebook_bundle(args.dir, args.patch, his_repo=args.his_repo, bundle=args.bundle)
            print(f"built {args.dir} (as /kaggle/taaf-kaggle-source-share after cell 4): {', '.join(logs)} applied")
        elif args.cmd == "check":
            with tempfile.TemporaryDirectory() as tmp:
                logs = notebook_bundle(Path(tmp) / "b", args.patch, his_repo=args.his_repo, bundle=args.bundle)
            for name, log in logs.items():
                if name != "his":
                    print(f"== {name}\n{log.strip()}")
            print(f"ok: {len(args.patch)} patch(es) apply on top of his")
        elif args.cmd == "diff":
            sys.stdout.write(diff(args.dir, args.base))
        elif args.cmd == "vendor":
            data = vendor(args.bundle, args.his_repo)
            print(f"wrote {DATA}: {len(data['bundle'])} bundle files, delta {DELTA.stat().st_size} bytes")
    except TreeError as exc:
        raise SystemExit(f"franzen_tree: {exc}") from None


if __name__ == "__main__":
    main()
