"""kaggle/franzen/patches/ours-07-fresh-start.patch: a fresh start of the conversation once a level has cost a set
number of generated tokens and minutes without a level-up, behind OURS_FRESH_START=1.

The checks that need the notebook tree run in the franzen_bed venv: the stagnation rule on synthetic histories and the
reset itself (tests/franzen_fresh_start_checks.py units); flag-off identity by driving the real ToolAgent.analyze()
over real games on the tree without the patch and with it (tests/franzen_ledger_checks.py drive, compared byte for
byte); and, with the flag on and a low threshold, the same drive read for where the history is cleared, what the next
request holds, how often it happens per level and what the priority gate is asked to do
(tests/franzen_fresh_start_checks.py drive). The slow test runs the bed with the bundle's six patches and this one,
every flag on.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import franzen_bed  # noqa: E402
import franzen_tree  # noqa: E402

PATCHES = ROOT / "kaggle" / "franzen" / "patches"
BUNDLE = [PATCHES / name for name in ("ours-sandbox-timeout-keeps-work.patch", "ours-02-budget-meter.patch",
                                      "ours-04-search-helper.patch", "ours-03b-win-ledger-on-02-04.patch",
                                      "ours-05-level-mem.patch", "ours-06b-effect-table-on-02-04-03b-05.patch")]
PATCH = PATCHES / "ours-07-fresh-start.patch"
LEDGER_CHECKS = ROOT / "tests" / "franzen_ledger_checks.py"
CHECKS = ROOT / "tests" / "franzen_fresh_start_checks.py"
BED_PY = Path.home() / ".cache" / "arc3-franzen-bed" / "venv" / "bin" / "python"
GAMES = ROOT / "environment_files"
LEDGER_HEAD = "Ledger of exact facts from the recorded boards"
FRESH = "Conversation history was cleared at step"
# the other flags of the bundle arm (exp-077 and after)
OTHERS = {"OURS_BUDGET_METER": "1", "OURS_SEARCH_HELPER": "1", "OURS_WIN_LEDGER": "1", "OURS_LEVEL_MEM": "1",
          "OURS_EFFECT_TABLE": "1", "EXPOSE_RESET": "on"}
LOW = {"OURS_FRESH_START_TOKENS": "1000", "OURS_FRESH_START_MINUTES": "0"}

needs_tree = pytest.mark.skipif(not (franzen_tree.his_repo_path() / "ARC3-Inference").is_dir(),
                                reason="needs Franzen's repo (scripts/franzen_tree.py)")
needs_bed = pytest.mark.skipif(not BED_PY.exists(), reason="needs the franzen_bed venv (scripts/franzen_bed.py)")
needs_games = pytest.mark.skipif(not (GAMES / "ls20").is_dir(), reason="needs the game files (environment_files/)")


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    """The notebook's src/ after cell 4: the bundle order 01, 02, 04, 03b, 05, 06b without this patch and with it."""
    out = {}
    for name, patches in (("base", BUNDLE), ("fresh", [*BUNDLE, PATCH])):
        dest = tmp_path_factory.mktemp(name)
        logs = franzen_tree.notebook_bundle(dest, patches)
        assert [p.name for p in patches] == [k for k in logs if k != "his"]
        out[name] = dest / "src" / "ARC3-Inference"
    return out


def harness_env(**extra: str) -> dict:
    """The environment his notebook's cell 4 sets, with the bed's small context so history is trimmed early."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith("OURS_")
           and k != "EXPOSE_RESET"}
    nb_env, _ = franzen_bed.notebook_env(franzen_bed._cell(franzen_tree.notebook_cells(), "setup_env = {"))
    env.update(nb_env)
    env.update(franzen_bed.BED_CONTEXT)
    env.update({"ARC3_MAX_ACTIVE_STREAMS": "0", "PYTHONHASHSEED": "0", **extra})
    return env


def test_patch_is_a_notebook_patch_confined_to_the_harness():
    text = PATCH.read_text()
    names = [line.split()[2][2:] for line in text.splitlines() if line.startswith("diff --git ")]
    assert names == ["ARC3-Inference/inference/agent/tool_agent.py"]
    assert "GIT binary patch" not in text and '_get_env_bool("OURS_FRESH_START", False)' in text
    assert not [line for line in text.splitlines() if line.startswith("-") and not line.startswith("---")], \
        "the patch only adds lines"


@needs_tree
def test_patch_applies_after_the_bundle_order(trees):
    agent = (trees["fresh"] / "inference/agent/tool_agent.py").read_text()
    assert "def _ours_fresh_start(" in agent and "def _ours_ledger_pin(" in agent  # on top of 03b
    harness = "".join(path.read_text() for path in sorted((trees["fresh"] / "inference").rglob("*.py")))
    for flag in ("OURS_BUDGET_METER", "OURS_SEARCH_HELPER", "OURS_WIN_LEDGER", "OURS_LEVEL_MEM", "OURS_EFFECT_TABLE"):
        assert flag in harness, flag  # the five patches underneath are all there
    assert "_ours_fresh" not in (trees["base"] / "inference/agent/tool_agent.py").read_text()


@needs_tree
@needs_bed
def test_units_threshold_rule_and_reset(trees):
    r = subprocess.run([str(BED_PY), str(CHECKS), "units", str(trees["fresh"])], capture_output=True, text=True,
                       timeout=300, env=harness_env(OURS_FRESH_START="1"))
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout[-2000:] + r.stderr[-4000:]


def _run(script: Path, trees, tmp_path, game: str, turns: int, runs: dict) -> dict:
    """Run drives in parallel; {name: output} for runs {name: (tree, extra environment)}."""
    procs = {}
    for name, (tree, extra) in runs.items():
        out = tmp_path / f"{game}-{name}.json"
        procs[name] = (out, subprocess.Popen(
            [str(BED_PY), str(script), "drive", str(trees[tree]), str(GAMES), str(out), game, str(turns)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=harness_env(**extra)))
    results = {}
    for name, (out, proc) in procs.items():
        stdout, stderr = proc.communicate(timeout=600)
        assert proc.returncode == 0, stdout[-2000:] + stderr[-4000:]
        results[name] = json.loads(out.read_text())
    return results


@needs_tree
@needs_bed
@needs_games
@pytest.mark.parametrize("game,others", [("ls20", "on"), ("ls20", "off"), ("vc33", "on")])
def test_flag_off_is_byte_identical(trees, tmp_path, game, others):
    """Flag unset, "0", or on with the default threshold (80k tokens and 20 min, never reached here): the same
    requests on the wire, the same stored history and transcript as the tree without the patch."""
    flags = dict(OTHERS) if others == "on" else {}
    res = _run(LEDGER_CHECKS, trees, tmp_path, game, 22, {
        "base": ("base", flags), "off": ("fresh", flags), "zero": ("fresh", {**flags, "OURS_FRESH_START": "0"}),
        "default": ("fresh", {**flags, "OURS_FRESH_START": "1"})})
    base = res["base"]
    assert base["level_ups"] >= 1 and len(base["requests"]) >= 40, "the drive must reach a level-up and trims"
    for name in ("off", "zero", "default"):
        other = res[name]
        assert [q["sha"] for q in other["requests"]] == [q["sha"] for q in base["requests"]], name
        assert (other["history_sha"], other["transcript_sha"]) == (base["history_sha"], base["transcript_sha"]), name


def _diverges_at(before: list, after: list) -> int | None:
    for i, key in enumerate(before):
        if i >= len(after) or after[i] != key:
            return i
    return None


@needs_tree
@needs_bed
@needs_games
def test_flag_on_clears_history_keeps_the_ledger_and_hands_the_slot_over_once(trees, tmp_path):
    """ls20 with every bundle flag on, a priority gate, and a low threshold (1000 generated tokens, no minimum time):
    the scripted model replies by request count, so the game is the same with and without fresh starts."""
    gate = {**OTHERS, "ARC3_MAX_ACTIVE_STREAMS": "1"}
    res = _run(CHECKS, trees, tmp_path, "ls20", 30, {
        "off": ("fresh", gate), "on": ("fresh", {**gate, **LOW, "OURS_FRESH_START": "1"})})
    off, on = res["off"], res["on"]
    # the same game, and the same counts for the gate to price it with
    assert (on["actions"], on["level_ups"], on["game_overs"], len(on["requests"])) == (
        off["actions"], off["level_ups"], off["game_overs"], len(off["requests"]))
    assert on["level_ups"] == 1 and on["gate_inputs"] == off["gate_inputs"], (on["gate_inputs"], off["gate_inputs"])
    assert not off["statuses"] and not any(q["fresh_at"] for q in off["requests"])
    # two fresh starts, both on level 2 (level 1 is won by the second request), then none: at most 2 per level
    fields = on["status_fields"]
    assert [(int(f[1]), int(f[2]), int(f[3])) for f in fields] == [(1, 2, 2), (2, 2, 2)], fields
    assert int(fields[0][4]) >= 1000 and int(fields[1][4]) - int(fields[0][4]) >= 1500, fields
    assert len(on["warnings"]) == 2 and on["repeated"] == 0, on["warnings"]
    opened = [i for i, q in enumerate(on["requests"]) if q["fresh_at"] == [q["n"] - 1]]
    assert len(opened) == 2, opened
    for i in opened:
        q, before = on["requests"][i], on["requests"][i - 1]
        # the turn's first request; only the system prompt, the ledger and the opener with the line
        assert q["first_of_turn"] and q["n"] == 3 and q["roles"] == ["system", "user", "user"], q
        assert q["ledger_at"] == [1] and q["ledger"].startswith(LEDGER_HEAD), q
        assert "- Level 1: won in 13 actions (steps 1-13), no game over." in q["ledger"], q["ledger"]
        assert "(current, since step 13)" in q["ledger"] and "- Retained functions: drive_probe." in q["ledger"]
        assert before["n"] > 3 and q["keys"][0] == before["keys"][0], "same system prompt, the rest dropped"
        assert _diverges_at(before["keys"], q["keys"]) == 1
        assert q["fresh_at"] == [2] and i in on["handovers"], (i, on["handovers"])  # one handover, right before it
    # never a tool call without its result; the gate is asked only where the prefix breaks anyway
    assert not any(q["dangling"] for q in on["requests"]) and not any(q["dangling"] for q in off["requests"])
    for run in (on, off):
        requests = run["requests"]
        assert all(_diverges_at(requests[i - 1]["keys"], requests[i]["keys"]) is not None
                   for i in run["handovers"] if i > 0), run["handovers"]
    assert len(set(on["handovers"])) == len(on["handovers"])
    # every request that carries the line has the ledger right after the system prompt
    assert all(q["ledger_at"] == [1] for q in on["requests"] if q["fresh_at"])
    # the line itself: exact numbers, no advice
    line = next(m for m in on["statuses"])
    assert re.match(r"^ours_fresh_start: conversation history cleared at step \d+ \(fresh start 1 of at most 2 on "
                    r"level 2\) after \d+ generated tokens", line), line


@pytest.mark.slow
@needs_tree
@needs_bed
@needs_games
def test_bed_with_all_seven_patches_and_flags_shows_fresh_starts_with_the_ledger(tmp_path):
    out = tmp_path / "bed"
    result = franzen_bed.run_bed(out, games=["ls20", "vc33", "sb26"], seconds=120, patches=[*BUNDLE, PATCH],
                                 env_add={**OTHERS, "OURS_FRESH_START": "1"},
                                 sets={"OURS_FRESH_START_TOKENS": "4000", "OURS_FRESH_START_MINUTES": "0.25"},
                                 expect=FRESH, expect_in="any", expect_pinned=LEDGER_HEAD,
                                 programs=["search", "mem", "effects"])
    failed = [name for name, ok in result["checks"].items() if not ok]
    assert not failed, (failed, result["facts"])
    assert result["facts"]["expect_opened"] >= 1, result["facts"]
    per_level = Counter()
    for transcript in (out / "transcripts").glob("*.txt"):
        for match in re.finditer(r"^ours_fresh_start: .* on level (\d+)\)", transcript.read_text(), re.M):
            per_level[(transcript.name, match.group(1))] += 1
    assert per_level and max(per_level.values()) <= 2, per_level


def test_flag_names_do_not_collide_with_cell_4():
    """--env-add refuses a key cell 4 already sets; an arm adds OURS_FRESH_START (and any threshold knob) that way."""
    cell4 = franzen_bed._cell(franzen_tree.notebook_cells(), "setup_env = {")
    assert "OURS_FRESH_START" not in cell4
