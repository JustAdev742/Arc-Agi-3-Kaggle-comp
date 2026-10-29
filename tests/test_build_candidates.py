"""scripts/build_candidates.sh: the leaderboard candidates rebuild byte for byte from the repo (the sha256 values below
are the notebooks pushed on 2026-10-10 as exp-083 v1, exp-084 v1 and exp-085 v1/v2, the versions the owner submits),
a new draft changes only the draft, its version pins, the slugs and the notes, and the notice-v2 arm exp-087 (not
pushed, so not pinned) is exp-085 with only the untried patch, its flag value, the slug and the note changed."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import franzen_tree  # noqa: E402

SCRIPT = ROOT / "scripts" / "build_candidates.sh"
MANIFEST = ROOT / "kaggle" / "franzen" / "drafts" / "arc3-mtp-session-a.manifest.json"
PUSHED = {  # sha256 of the files pushed as exp-083 / exp-084 version 1
    "exp083/arc3-dprime-r14a05-arcmap-draft-full.ipynb":
        "64bc9bda8d50292fec6a328e4f05b3c26b2cfc56355276f2665d67e0822b3af3",
    "exp083/kernel-metadata.json": "ffc81d48a10561bb7ed02428fe0fa85f7d2d05ca4470c3020904018c1e61146d",
    "exp084/arc3-dprime-r14a05-harness4-percept-draft-full.ipynb":
        "846a5677676aed36892a7499f5330386c163b56a24710df3b05176863c17e625",
    "exp084/kernel-metadata.json": "3456d2b602a66761150bfbac83e4d5d6e57c642008ea8d158030ed175f3c46a0",
    "exp085/arc3-dprime-r14a05-h4-percept-untried-full.ipynb":
        "4919e6028c47fbc998280253a2f0b29bdfa6c19688ed4586644002041dd033db",
    "exp085/kernel-metadata.json": "20a366147dc6662fc93b790e3b135b01fcdb899cd205ccbc3aa1dcf29790caf1",
}
SLUG085, SLUG087 = "arc3-dprime-r14a05-h4-percept-untried-full", "arc3-dprime-r14a05-h4-percept-untried2-full"
EXP087 = (f"exp087/{SLUG087}.ipynb", "exp087/kernel-metadata.json")  # built, not pinned
PATCHES = ROOT / "kaggle" / "franzen" / "patches"

# the --patch apply check needs his repo (scripts/franzen_tree.py)
pytestmark = pytest.mark.skipif(not (franzen_tree.his_repo_path() / "ARC3-Inference").is_dir(),
                                reason="needs Franzen's repo (scripts/franzen_tree.py)")


def _build(out: Path, *args: str) -> None:
    env = {**os.environ, "PYTHON": sys.executable}
    run = subprocess.run(["bash", str(SCRIPT), str(out), *args], capture_output=True, text=True, env=env,
                         timeout=600, check=False)
    assert run.returncode == 0, run.stderr[-3000:]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_manifest_copy_is_the_adopted_drafts():
    assert _sha(MANIFEST).startswith("be8c2d3ae23d")  # what exp-083/084 and probe B1 check in cell 4
    assert json.loads(MANIFEST.read_text())["files"]["mtp-dense.safetensors"].startswith("642797acaa0b")


def test_the_candidates_rebuild_byte_for_byte(tmp_path):
    _build(tmp_path)
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file()) == sorted([*PUSHED,
                                                                                                         *EXP087])
    assert {name: _sha(tmp_path / name) for name in PUSHED} == PUSHED


def test_exp087_is_exp085_with_the_v2_patch_its_flag_value_its_slug_and_its_note(tmp_path):
    _build(tmp_path)
    assert len(SLUG087.replace("-", " ")) <= 50  # Kaggle's title limit (lesson 0033)
    old_meta = json.loads((tmp_path / "exp085" / "kernel-metadata.json").read_text())
    new_meta = json.loads((tmp_path / "exp087" / "kernel-metadata.json").read_text())
    assert (new_meta["id"], new_meta["title"], new_meta["code_file"]) == (
        "scottmahony/" + SLUG087, SLUG087.replace("-", " "), SLUG087 + ".ipynb")
    names = ("id", "title", "code_file")
    assert {k: v for k, v in old_meta.items() if k not in names} == {k: v for k, v in new_meta.items() if k not in names}
    old = json.loads((tmp_path / "exp085" / f"{SLUG085}.ipynb").read_text())
    new = json.loads((tmp_path / "exp087" / f"{SLUG087}.ipynb").read_text())
    assert {k: v for k, v in old.items() if k != "cells"} == {k: v for k, v in new.items() if k != "cells"}
    assert [{k: v for k, v in c.items() if k != "source"} for c in old["cells"]] == [
        {k: v for k, v in c.items() if k != "source"} for c in new["cells"]]
    old_cells, new_cells = (["".join(c["source"]) for c in nb["cells"]] for nb in (old, new))
    # the patch cell ships the repo's patch files, byte for byte
    v1, v2 = (PATCHES / "ours-10-untried.patch").read_text(), (PATCHES / "ours-10v2-untried.patch").read_text()
    written = [(franzen_tree.written_file(a), franzen_tree.written_file(b)) for a, b in zip(old_cells, new_cells)]
    assert [pair for pair in written if pair[0] != pair[1]] == [(("/kaggle/ours-07-ours-10-untried.patch", v1),
                                                                 ("/kaggle/ours-07-ours-10v2-untried.patch", v2))]
    patch_cell = next(i for i, (a, b) in enumerate(written) if a != b)
    # everything else is equal once the patch's name, size and sha256, the flag value and the note are swapped back
    sha1, sha2 = (hashlib.sha256(t.encode()).hexdigest() for t in (v1, v2))
    swaps = [("ours-07-ours-10v2-untried.patch", "ours-07-ours-10-untried.patch"), (sha2, sha1), (sha2[:12], sha1[:12]),
             (f"{len(v2.encode())} bytes", f"{len(v1.encode())} bytes"), ("'OURS_UNTRIED': 'v2',", "'OURS_UNTRIED': 1,"),
             ("env added OURS_UNTRIED=v2", "env added OURS_UNTRIED=1")]
    changed = []
    for i, (a, b) in enumerate(zip(old_cells, new_cells, strict=True)):
        if a == b:
            continue
        changed.append(i)
        if i == patch_cell:  # the payload itself was compared above
            a, b = (re.sub(r'"""\n.*?\n"""', '"""(packed)"""', s, flags=re.S) for s in (a, b))
        a, b = (re.sub(r"sha256 checked\)\. exp-08[57]: .*", "sha256 checked). (note)", s) for s in (a, b))
        for new_text, old_text in swaps:
            b = b.replace(new_text, old_text)
        assert a == b, f"cell {i}"
    assert len(changed) == 3  # the arm description, the patch cell and cell 4 (patch list, flag)
    assert "exp-087: exp-085" in new_cells[0] and "'OURS_UNTRIED': 'v2'," in "".join(new_cells)


def test_another_draft_changes_only_the_draft_its_pins_the_slugs_and_the_notes(tmp_path):
    manifest = json.loads(MANIFEST.read_text())
    manifest["files"]["mtp-dense.safetensors"] = "ab" * 32
    other = tmp_path / "other-manifest.json"
    other.write_text(json.dumps(manifest, indent=1))
    _build(tmp_path / "old")
    _build(tmp_path / "new", "scottmahony/arc3-mtp-session-a2", str(other), "draft2", "MTP session A2's draft")
    for exp, old_slug, new_slug in (
            ("exp083", "arc3-dprime-r14a05-arcmap-draft-full", "arc3-dprime-r14a05-arcmap-draft2-full"),
            ("exp084", "arc3-dprime-r14a05-harness4-percept-draft-full", "arc3-dprime-r14a05-harness4-percept-draft2-full"),
            ("exp085", "arc3-dprime-r14a05-h4-percept-untried-full", "arc3-dprime-r14a05-h4-percept-untried-draft2-full")):
        assert len(new_slug.replace("-", " ")) <= 50  # Kaggle's title limit (lesson 0033)
        old_meta = json.loads((tmp_path / "old" / exp / "kernel-metadata.json").read_text())
        new_meta = json.loads((tmp_path / "new" / exp / "kernel-metadata.json").read_text())
        assert new_meta["id"] == "scottmahony/" + new_slug
        assert new_meta["kernel_sources"] == ["scottmahony/arc3-mtp-session-a2"]
        same = {k: v for k, v in old_meta.items() if k not in ("id", "title", "code_file", "kernel_sources")}
        assert same == {k: v for k, v in new_meta.items() if k not in ("id", "title", "code_file", "kernel_sources")}
        old_cells = ["".join(c["source"]) for c in json.loads(
            (tmp_path / "old" / exp / f"{old_slug}.ipynb").read_text())["cells"]]
        new_cells = ["".join(c["source"]) for c in json.loads(
            (tmp_path / "new" / exp / f"{new_slug}.ipynb").read_text())["cells"]]
        assert len(old_cells) == len(new_cells)
        changed = [(a, b) for a, b in zip(old_cells, new_cells, strict=True) if a != b]
        assert len(changed) == 2  # the arm description and cell 4's draft lines
        dense = json.loads(MANIFEST.read_text())["files"]["mtp-dense.safetensors"]
        swaps = [("arc3-mtp-session-a2", "arc3-mtp-session-a"),
                 ("MTP session A2's draft", "MTP session A v2's fine-tuned draft" if exp != "exp085"
                  else "MTP session A v2's draft"),
                 (_sha(other), _sha(MANIFEST)), (_sha(other)[:12], _sha(MANIFEST)[:12]),
                 ("ab" * 32, dense), ("ab" * 6, dense[:12])]
        for a, b in changed:
            for new, old in swaps:
                b = b.replace(new, old)
            assert a == b
