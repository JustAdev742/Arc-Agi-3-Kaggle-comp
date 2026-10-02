"""scripts/franzen_bed.py: Daniel Franzen's real harness on real public games against a scripted mock model.

The fast tests check the pieces (the scripted model, the environment read from the notebook's cell 4). The slow one
(``pytest -m slow``, about two minutes) runs the whole bed with our sample patch: it needs his repo, the game files
and a venv with the harness dependencies (created once by the bed under ~/.cache/arc3-franzen-bed, with uv).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_franzen_nb as bf  # noqa: E402
import franzen_bed as fb  # noqa: E402
import franzen_tree as ft  # noqa: E402

SAMPLE = ROOT / "tests" / "fixtures" / "franzen" / "sample-ours.patch"


def _call(code: str, n: int = 1) -> dict:
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": f"call_{n}", "type": "function",
                            "function": {"name": "python", "arguments": json.dumps({"code": code})}}]}


def test_the_script_walks_its_programs_from_the_conversation_alone():
    system, opener = {"role": "system", "content": "s"}, {"role": "user", "content": "state"}
    assert fb.next_step([system, opener]) == ("solve", 0)
    solve = [system, opener, _call(fb.SNIPPETS["solve"][0]), {"role": "tool", "tool_call_id": "call_1", "content": "x"}]
    assert fb.next_step(solve) == ("fallback", 0)  # an acting snippet that acted nothing: act now
    turn2 = [*solve, {"role": "user", "content": "next turn"}]
    assert fb.next_step(turn2) == ("define", 0)
    defined = [*turn2, _call(fb.SNIPPETS["define"][0], 2), {"role": "tool", "tool_call_id": "call_2", "content": "y"}]
    assert fb.next_step(defined) == ("define", 1) and "bed_pick(valid_actions" in fb.SNIPPETS["define"][1]
    chat = [*defined, {"role": "assistant", "content": "bed:chat:0 later"}, {"role": "user", "content": fb.NUDGE}]
    assert fb.next_step(chat) == ("think", 0)
    think = [*chat, {"role": "assistant", "content": "bed:think:0"}, {"role": "user", "content": "resumed"}]
    assert fb.next_step(think) == (fb.CYCLE[0], 0)
    for program, snippets in fb.SNIPPETS.items():
        for i, code in enumerate(snippets):
            compile(code, f"{program}:{i}", "exec")
            assert code.startswith(f"# bed:{program}:{i}\n")


def test_the_mock_reports_usage_prefix_cache_overflow_and_each_tool_result_once(tmp_path):
    model = fb.MockModel(tmp_path, latency=0, overflow_every=3, expect="MARK")
    system = {"role": "system", "content": "x" * 3300 + " MARK"}
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    opener = {"role": "user", "content": [{"type": "text", "text": "state"}, image]}
    status, first = model.handle({"messages": [system, opener]})
    assert status == 200 and first["choices"][0]["message"]["tool_calls"]
    usage = first["usage"]
    assert usage["prompt_tokens"] >= 1000 + 402 and usage["prompt_tokens_details"]["image_tokens"] == 402
    assert usage["prompt_tokens_details"]["cached_tokens"] == 0
    reply = first["choices"][0]["message"]
    tool = {"role": "tool", "tool_call_id": reply["tool_calls"][0]["id"], "content": "BED retained True no_op_action"}
    status, second = model.handle({"messages": [system, opener, reply, tool]})
    cached = second["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert status == 200 and 1000 <= cached < second["usage"]["prompt_tokens"] and cached % 64 == 0
    long = [system, opener, reply, tool] + [{"role": "user", "content": "u"}] * 10
    status, third = model.handle({"messages": long})
    assert status == 400 and "is longer than the model's context length" in third["message"]
    model.close()
    records = [json.loads(line) for line in (tmp_path / "mock.jsonl").read_text().splitlines()]
    assert [r["tags"] for r in records] == [[], ["retained", "batch_noop"], []]  # the tool result counted once
    assert all(r["expect_in_system"] for r in records)
    serve = (tmp_path / "serve.log").read_text()
    assert serve.count("ReqTimeStats(") == 2 and '400 Bad Request' in serve


def test_the_harness_environment_is_read_from_cell_4():
    env, names = fb.notebook_env(fb._cell(ft.notebook_cells(), "setup_env = {"))
    assert names["USE_PRIORITY_SCHEDULING"] is True and len(env) == 60
    assert env["LOCAL_ANALYZER_CONTEXT_WINDOW"] == str(128 * 1024) and env["ARC3_CONTEXT_DRAIN_TOKENS"] == str(58 * 1024)
    assert env["EXPOSE_UNDO"] == "on" and env["ARC3_MAX_ACTIVE_STREAMS"] == "10" and env["ARC3_GUARDS_FROM_LEVEL"] == "2"
    assert env["TAAF_RUN_AS_SUBMISSION"] == "0" and env["ARC3_WARMUP_ACTION_GAMES"] == "1"


def test_an_arm_built_with_env_flags_changes_the_bed_environment(tmp_path):
    bf.build(tmp_path, "arm", env={"MULTIMODAL_UPSCALE": "8"}, env_add={"EXPOSE_RESET": "on"})
    env, _ = fb.notebook_env(fb._cell(ft.notebook_cells(tmp_path / "arm.ipynb"), "setup_env = {"))
    assert env["MULTIMODAL_UPSCALE"] == "8" and env["EXPOSE_RESET"] == "on" and len(env) == 61


@pytest.mark.slow
def test_bed_runs_his_harness_with_our_patch_and_exercises_gate_guards_retention_and_trimming(tmp_path):
    if not (ft.his_repo_path() / "ARC3-Inference").is_dir():
        pytest.skip("his repo (da-fr/arc-agi-3-solution) not available")
    if not (ROOT / "environment_files" / "ls20").exists():
        pytest.skip("game files not downloaded")
    try:
        python = fb.ensure_venv(fb.DEFAULT_VENV)
    except (subprocess.CalledProcessError, OSError) as exc:
        pytest.skip(f"no venv for the harness: {exc}")
    result = fb.run_bed(tmp_path / "bed", games=["ls20", "vc33", "sb26"], seconds=90, patches=[SAMPLE],
                        sets={"OURS_SYSTEM_PROMPT_SUFFIX": "[ours-bed-marker]"}, expect="[ours-bed-marker]",
                        python=python)
    failed = [name for name, ok in result["checks"].items() if not ok]
    assert not failed, (failed, result["facts"])
    facts = result["facts"]
    assert facts["levels_completed"]["ls20-9607627b"] >= 1 and facts["levels_completed"]["vc33-5430563c"] >= 1
    assert facts["expect_seen"] == facts["requests"]  # our patch runs on every request
    assert (tmp_path / "bed" / "benchmark.json").exists() and (tmp_path / "bed" / "serve.log").exists()
