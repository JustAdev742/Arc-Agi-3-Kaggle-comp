"""kaggle/franzen/patches/ours-03-win-ledger.patch (method M4): exact win records at level-ups and a ledger re-pinned
right after the system prompt at every context trim, behind OURS_WIN_LEDGER=1.

The checks that need the notebook tree run in the franzen_bed venv through tests/franzen_ledger_checks.py: unit checks
of the record builder, the trim re-insertion and the sandbox's `level_wins`; and a deterministic drive of the real
ToolAgent.analyze() over a real game (scripted model, engine-backed step_env), compared byte for byte between the
tree without this patch and the tree with it (flag unset and "0"), and read for the prefix-cache property with it on.
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import franzen_bed  # noqa: E402
import franzen_tree  # noqa: E402

PATCHES = ROOT / "kaggle" / "franzen" / "patches"
SANDBOX_PATCH = PATCHES / "ours-sandbox-timeout-keeps-work.patch"
PATCH = PATCHES / "ours-03-win-ledger.patch"
CHECKS = ROOT / "tests" / "franzen_ledger_checks.py"
BED_PY = Path.home() / ".cache" / "arc3-franzen-bed" / "venv" / "bin" / "python"
GAMES = ROOT / "environment_files"
LEDGER_HEAD = "Ledger of exact facts from the recorded boards"

needs_tree = pytest.mark.skipif(not (franzen_tree.his_repo_path() / "ARC3-Inference").is_dir(),
                                reason="needs Franzen's repo (scripts/franzen_tree.py)")
needs_bed = pytest.mark.skipif(not BED_PY.exists(), reason="needs the franzen_bed venv (scripts/franzen_bed.py)")
needs_games = pytest.mark.skipif(not (GAMES / "ls20").is_dir(), reason="needs the game files (environment_files/)")


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    """The notebook's src/ after cell 4: without this patch (his + the sandbox patch) and with it."""
    out = {}
    for name, patches in (("base", [SANDBOX_PATCH]), ("ledger", [SANDBOX_PATCH, PATCH])):
        dest = tmp_path_factory.mktemp(name)
        logs = franzen_tree.notebook_bundle(dest, patches)
        assert [p.name for p in patches] == [k for k in logs if k != "his"]
        out[name] = dest / "src" / "ARC3-Inference"
    return out


def harness_env(**extra: str) -> dict:
    """The environment his notebook's cell 4 sets, with the bed's small context so history is trimmed early."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "OURS_WIN_LEDGER")}
    nb_env, _ = franzen_bed.notebook_env(franzen_bed._cell(franzen_tree.notebook_cells(), "setup_env = {"))
    env.update(nb_env)
    env.update(franzen_bed.BED_CONTEXT)
    env.update({"ARC3_MAX_ACTIVE_STREAMS": "0", "PYTHONHASHSEED": "0", **extra})
    return env


def test_patch_is_a_notebook_patch_confined_to_the_harness():
    text = PATCH.read_text()
    names = [line.split()[2][2:] for line in text.splitlines() if line.startswith("diff --git ")]
    assert names == ["ARC3-Inference/inference/agent/python_tool_sandbox.py",
                     "ARC3-Inference/inference/agent/tool_agent.py"]
    assert "GIT binary patch" not in text and "OURS_WIN_LEDGER" in text


@needs_tree
def test_patch_applies_after_his_patch_and_the_sandbox_patch(trees):
    agent = (trees["ledger"] / "inference/agent/tool_agent.py").read_text()
    sandbox = (trees["ledger"] / "inference/agent/python_tool_sandbox.py").read_text()
    assert '_get_env_bool("OURS_WIN_LEDGER", False)' in agent
    assert 'startswith("Tool timed out")' in agent  # the sandbox patch is still there underneath
    assert 'if "level_wins" in state_payload:' in sandbox


@needs_tree
@needs_bed
def test_units_record_builder_trim_reinsertion_and_sandbox(trees):
    r = subprocess.run([str(BED_PY), str(CHECKS), "units", str(trees["ledger"])], capture_output=True, text=True,
                       timeout=300, env=harness_env(OURS_WIN_LEDGER="1"))
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout[-2000:] + r.stderr[-4000:]


def _drive(trees, tmp_path, game: str, turns: int, runs: dict) -> dict:
    """Run the drives in parallel; {name: output} for runs {name: (tree, OURS_WIN_LEDGER or None)}."""
    procs = {}
    for name, (tree, flag) in runs.items():
        out = tmp_path / f"{game}-{name}.json"
        extra = {} if flag is None else {"OURS_WIN_LEDGER": flag}
        procs[name] = (out, subprocess.Popen(
            [str(BED_PY), str(CHECKS), "drive", str(trees[tree]), str(GAMES), str(out), game, str(turns)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=harness_env(**extra)))
    results = {}
    for name, (out, proc) in procs.items():
        stdout, stderr = proc.communicate(timeout=600)
        assert proc.returncode == 0, stdout[-2000:] + stderr[-4000:]
        results[name] = json.loads(out.read_text())
    return results


def _first_divergence(before: list, after: list) -> int | None:
    """Index of the first message that differs, or None when `after` extends `before`."""
    for i, key in enumerate(before):
        if i >= len(after) or after[i] != key:
            return i
    return None


@needs_tree
@needs_bed
@needs_games
@pytest.mark.parametrize("game", ["ls20", "vc33"])
def test_flag_off_is_byte_identical_and_flag_on_breaks_the_prefix_only_after_the_system_prompt(trees, tmp_path, game):
    res = _drive(trees, tmp_path, game, 22, {"base": ("base", None), "off": ("ledger", None),
                                             "zero": ("ledger", "0"), "on": ("ledger", "1")})
    base = res["base"]
    assert base["level_ups"] >= 1 and len(base["requests"]) >= 40, "the drive must reach a level-up and trims"
    breaks_off = [i for i, (p, q) in enumerate(itertools.pairwise(base["requests"]), 1)
                  if _first_divergence(p["keys"], q["keys"]) is not None]
    assert len(breaks_off) >= 3, breaks_off
    # flag unset or "0": the same requests on the wire, the same stored history, the same transcript
    for name in ("off", "zero"):
        other = res[name]
        assert [q["sha"] for q in other["requests"]] == [q["sha"] for q in base["requests"]], name
        assert (other["history_sha"], other["transcript_sha"]) == (base["history_sha"], base["transcript_sha"]), name
        assert not any(q["ledger_at"] or q["records"] for q in other["requests"])
    # flag on: the same game and turns; each break of the prefix is right after the system prompt, where the ledger
    # then sits (one copy), and between breaks the ledger is unchanged
    on = res["on"]
    assert (on["actions"], on["level_ups"], on["game_overs"], len(on["requests"])) == (
        base["actions"], base["level_ups"], base["game_overs"], len(base["requests"]))
    breaks_on = []
    for i, (p, q) in enumerate(itertools.pairwise(on["requests"]), 1):
        at = _first_divergence(p["keys"], q["keys"])
        if at is None:
            assert q["ledger"] == p["ledger"]
            continue
        breaks_on.append(i)
        assert at == 1 and q["ledger_at"] == [1], (i, at, q["ledger_at"])
    assert len(breaks_on) >= 3 and abs(len(breaks_on) - len(breaks_off)) <= 2, (breaks_on, breaks_off)
    assert all(q["ledger_at"] in ([], [1]) for q in on["requests"])
    assert on["requests"][breaks_on[0]]["ledger_at"] == [1] and not on["requests"][0]["ledger_at"]
    last = next(q["ledger"] for q in reversed(on["requests"]) if q["ledger"])
    assert last.startswith(LEDGER_HEAD) and "- Level 1: won in " in last and "(current, since step " in last
    assert "- Retained functions: drive_probe." in last
    # the level-up opener carries the record, and `level_wins` reaches the sandbox afterwards
    assert len(on["transcript_records"]) == on["level_ups"] and on["transcript_records"][0].startswith(
        "Level 1 record (exact facts from the recorded boards): won in ")
    assert max(q["level_wins_seen"] for q in on["requests"]) == on["level_ups"]


@needs_tree
@needs_bed
@needs_games
def test_level_one_records_on_real_boards(trees, tmp_path):
    """The opener records for the bed's level-1 solutions, as the engine plays them (checked once against the boards
    with tools that share no code with the patch: see docs/research/beat-tufa/patch-win-ledger.md)."""
    res = _drive(trees, tmp_path, "ls20", 3, {"on": ("ledger", "1")})["on"]
    assert res["transcript_records"] == [
        "Level 1 record (exact facts from the recorded boards): won in 13 actions (steps 1-13), no game over."]
    res = _drive(trees, tmp_path, "vc33", 3, {"on": ("ledger", "1")})["on"]
    assert res["transcript_records"] == [
        "Level 1 record (exact facts from the recorded boards): won in 3 actions (steps 1-3), no game over."]


@pytest.mark.slow
@needs_tree
@needs_bed
@needs_games
def test_bed_runs_with_the_ledger_and_shows_it_after_trims(tmp_path):
    result = franzen_bed.run_bed(tmp_path / "bed", games=["ls20", "vc33"], seconds=90,
                                 patches=[SANDBOX_PATCH, PATCH], sets={"OURS_WIN_LEDGER": "1"},
                                 expect=LEDGER_HEAD, expect_in="any")
    assert all(result["checks"].values()), result["checks"]
    assert result["facts"]["expect_at"].get("1", 0) > 0, result["facts"]
