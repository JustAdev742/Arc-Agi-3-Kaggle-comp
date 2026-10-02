"""scripts/franzen_tree.py: the tree Daniel Franzen's notebook builds (Tufa's bundle + his patch), rebuilt locally.

These pin three facts the notebook path depends on: his GitHub repo plus the vendored delta reproduces the Kaggle
bundle byte for byte (sha256 manifest of the downloaded dataset); his patch applies to it exactly as in his Kaggle
log; and a patch made with ``build`` + ``diff`` applies on top. Needs his repo (FRANZEN_REPO or
/home/user/da-fr/arc-agi-3-solution at 10882e3); the tests that need it skip without it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import franzen_tree as ft  # noqa: E402

SAMPLE = ROOT / "tests" / "fixtures" / "franzen" / "sample-ours.patch"
needs_repo = pytest.mark.skipif(not (ft.his_repo_path() / "ARC3-Inference").is_dir(),
                                reason="his repo (da-fr/arc-agi-3-solution) not available")

# Files of the notebook's tree (bundle + his patch) that his GitHub repo has differently or lacks, and the one it
# has extra. Everything else under ARC3-Inference/ and tufa-arc-agi-framework/ is byte-identical.
DIFFERS_FROM_HIS_REPO = {
    "ARC3-Inference/Makefile", "ARC3-Inference/README.md", "ARC3-Inference/configs/inference.json",
    "ARC3-Inference/configs/inference.openrouter.json", "ARC3-Inference/inference/framework/run.py",
    "ARC3-Inference/inference/tools/eval.py", "ARC3-Inference/inference/tools/significance.py",
    "ARC3-Inference/inference/tools/traces.py", "ARC3-Inference/pyproject.toml", "ARC3-Inference/uv.lock",
    "ARC3-Inference/viewer/data.py", "ARC3-Inference/viewer/index.html", "tufa-arc-agi-framework/README.md",
    "tufa-arc-agi-framework/pyproject.toml", "tufa-arc-agi-framework/src/taaf/competition_arcade.py",
    "tufa-arc-agi-framework/src/taaf/deploy.py", "tufa-arc-agi-framework/src/taaf/deploy_kaggle.py",
    "tufa-arc-agi-framework/src/taaf/deploy_slurm.py", "tufa-arc-agi-framework/src/taaf/game_api.py",
    "tufa-arc-agi-framework/src/taaf/standard_benchmarks.py", "tufa-arc-agi-framework/uv.lock",
}
ONLY_IN_NOTEBOOK_TREE = {"ARC3-Inference/inference/utils/rearc_baselines.py",
                         "ARC3-Inference/inference/utils/rearc_version.py"}
ONLY_IN_HIS_REPO = {"ARC3-Inference/CONFIGURATION.md"}


def test_writefile_body_follows_ipython():
    assert ft.writefile_body("%%writefile /kaggle/x.patch\nabc") == ("/kaggle/x.patch", "abc\n")
    assert ft.writefile_body("%%writefile /kaggle/x.patch\nabc\n") == ("/kaggle/x.patch", "abc\n")
    patch = ft.his_patch_text()
    assert patch.startswith("diff --git a/ARC3-Inference/inference/agent/action_names.py") and patch.endswith("\n")
    assert ft.manifest()["his_patch_sha256"] == __import__("hashlib").sha256(patch.encode()).hexdigest()


def test_git_apply_is_not_fooled_by_an_enclosing_repository(tmp_path):
    """Inside another repository git apply skips files outside the current directory and exits 0."""
    outer = tmp_path / "outer"
    (outer / "sub" / "d").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(outer)], check=True)
    (outer / "sub" / "d" / "a.txt").write_text("one\ntwo\n")
    patch = tmp_path / "p.patch"
    patch.write_text("diff --git a/d/a.txt b/d/a.txt\n--- a/d/a.txt\n+++ b/d/a.txt\n@@ -1,2 +1,2 @@\n one\n-two\n+TWO\n")
    ft.git_apply(outer / "sub", patch)
    assert (outer / "sub" / "d" / "a.txt").read_text() == "one\nTWO\n"
    with pytest.raises(ft.TreeError, match="0 of 1 files applied"):
        ft.git_apply(outer / "sub", patch)  # already applied: the context no longer matches


@needs_repo
def test_his_repo_plus_delta_is_the_kaggle_bundle_byte_for_byte(tmp_path):
    dest = ft.build_bundle(tmp_path / "b")
    assert ft.tree_hashes(dest) == ft.manifest()["bundle"]  # build_bundle verifies this too
    assert (dest / "benchmark_initial.pkl").is_file() and (dest / "src" / "ARC3-Inference").is_dir()


@needs_repo
def test_his_patch_applies_as_in_his_kaggle_log(tmp_path):
    dest = ft.build_bundle(tmp_path / "b")
    log = ft.apply_his_patch(dest / "src")
    # his Kaggle log (2026-09-30): the bundle's run.py has 48 more lines than Tufa's GitHub release
    assert "Hunk #1 succeeded at 467 (offset 48 lines)." in log and "Hunk #2 succeeded at 1271 (offset 48 lines)." in log
    assert log.count("Applied patch ") == 16 and log.count("offset") == 2
    assert ft.tree_hashes(dest / "src") == ft.manifest()["notebook_src"]


@needs_repo
def test_the_notebook_tree_is_not_his_repo(tmp_path):
    dest = tmp_path / "b"
    ft.notebook_bundle(dest)
    tree = ft.tree_hashes(dest / "src")
    his = ft.manifest()["his_repo"]
    assert {k for k in tree.keys() & his.keys() if tree[k] != his[k]} == DIFFERS_FROM_HIS_REPO
    assert tree.keys() - his.keys() == ONLY_IN_NOTEBOOK_TREE and his.keys() - tree.keys() == ONLY_IN_HIS_REPO


@needs_repo
def test_a_drifted_repo_is_refused(tmp_path):
    import shutil
    repo = tmp_path / "repo"
    for name in ft.REPOS:
        shutil.copytree(ft.his_repo_path() / name, repo / name)
    (repo / "ARC3-Inference" / "inference" / "agent" / "prompts.py").write_text("# changed\n")
    with pytest.raises(ft.TreeError, match="is not his repo"):
        ft.build_bundle(tmp_path / "b", his_repo=repo)


@needs_repo
def test_build_edit_diff_round_trip(tmp_path):
    repo = ft.materialise(tmp_path / "r", [SAMPLE])
    tags = subprocess.run(["git", "tag"], cwd=repo, capture_output=True, text=True, check=True).stdout.split()
    assert sorted(tags) == ["bundle", "franzen", "ours"]
    agent = repo / "ARC3-Inference" / "inference" / "agent"
    assert "OURS_SYSTEM_PROMPT_SUFFIX" in (agent / "tool_agent.py").read_text()
    with open(agent / "prompts.py", "a") as f:
        f.write("\nOURS_EXTRA = 1\n")
    (agent / "ours_new.py").write_text("X = 1\n")
    patch = tmp_path / "second.patch"
    patch.write_text(ft.diff(repo))
    text = patch.read_text()
    assert "a/ARC3-Inference/inference/agent/prompts.py" in text and "b/ARC3-Inference/inference/agent/ours_new.py" in text
    assert "tool_agent.py" not in text  # HEAD already holds the first patch
    ft.notebook_bundle(tmp_path / "b", [SAMPLE, patch])
    assert (tmp_path / "b" / "src" / "ARC3-Inference" / "inference" / "agent" / "ours_new.py").read_text() == "X = 1\n"
    cumulative = ft.diff(repo, "franzen")
    assert "tool_agent.py" in cumulative and "ours_new.py" in cumulative


@needs_repo
def test_check_cli_reports_a_patch_that_does_not_apply(tmp_path):
    bad = tmp_path / "bad.patch"
    bad.write_text(SAMPLE.read_text().replace("    return prompt\n", "    return prompt + ''\n", 1))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "franzen_tree.py"), "check", "--patch", str(bad)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "0 of 1 files applied" in r.stderr
    ok = subprocess.run([sys.executable, str(ROOT / "scripts" / "franzen_tree.py"), "check", "--patch", str(SAMPLE)],
                        capture_output=True, text=True)
    assert ok.returncode == 0 and "Applied patch ARC3-Inference/inference/agent/tool_agent.py cleanly." in ok.stdout
