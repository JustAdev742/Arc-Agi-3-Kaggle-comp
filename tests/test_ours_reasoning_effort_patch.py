"""kaggle/franzen/patches/ours-09-reasoning-effort.patch (OURS_REASONING_EFFORT): Franzen's harness sends the chat
template's reasoning_effort with every request when the env is set (e.g. "medium"); unset, it sends exactly what it
sent before. The truncation ladder (ARC3_REASONING_EFFORT_LADDER), when on, still steps below the static value.

The patch applies alone, on exp-074t's stack (the sandbox fix) and on exp-081's (01, 02, 04, 03b, 05, 08b). In the
bed venv the real ToolAgent builds the kwargs; the Flash-Next chat template (tests/fixtures) shows what each value
changes in the system prompt.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import franzen_tree  # noqa: E402

PATCHES = ROOT / "kaggle" / "franzen" / "patches"
PATCH = PATCHES / "ours-09-reasoning-effort.patch"
SANDBOX = PATCHES / "ours-sandbox-timeout-keeps-work.patch"
EXP081 = [PATCHES / name for name in (
    "ours-sandbox-timeout-keeps-work.patch", "ours-02-budget-meter.patch", "ours-04-search-helper.patch",
    "ours-03b-win-ledger-on-02-04.patch", "ours-05-level-mem.patch", "ours-08b-perception-on-01-02-04-03b-05.patch")]
BED_PY = Path.home() / ".cache" / "arc3-franzen-bed" / "venv" / "bin" / "python"
TEMPLATE = ROOT / "tests" / "fixtures" / "flashnext_chat_template" / "chat_template.jinja"

needs_tree = pytest.mark.skipif(not (franzen_tree.his_repo_path() / "ARC3-Inference").is_dir(),
                                reason="needs Franzen's repo (scripts/franzen_tree.py)")
needs_bed = pytest.mark.skipif(not BED_PY.exists(), reason="needs the franzen_bed venv (scripts/franzen_bed.py)")

PROBE = r'''
import json, os, sys, types
sys.path.insert(0, sys.argv[1])
from inference.agent import tool_agent
out = {}
for name, env in (("unset", {}), ("medium", {"OURS_REASONING_EFFORT": "medium"}),
                  ("medium+ladder", {"OURS_REASONING_EFFORT": "medium", "ARC3_REASONING_EFFORT_LADDER": "low"})):
    for key in ("OURS_REASONING_EFFORT", "ARC3_REASONING_EFFORT_LADDER"):
        os.environ.pop(key, None)
    os.environ.update(env)
    for rung in (-1, 0):
        me = types.SimpleNamespace(_reasoning_effort_rung=rung)
        out[f"{name} rung {rung}"] = tool_agent.ToolAgent._harness_template_kwargs(me)
print(json.dumps(out))
'''


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    out = {}
    for name, patches in (("base", []), ("ours", [PATCH]), ("074t", [SANDBOX, PATCH]), ("081", [*EXP081, PATCH])):
        dest = tmp_path_factory.mktemp(name)
        logs = franzen_tree.notebook_bundle(dest, patches)
        assert [p.name for p in patches] == [k for k in logs if k != "his"]
        out[name] = dest / "src" / "ARC3-Inference"
    return out


@needs_tree
def test_the_patch_applies_alone_and_on_both_candidate_stacks_and_touches_only_the_kwargs(trees):
    base = (trees["base"] / "inference/agent/tool_agent.py").read_text()
    ours = (trees["ours"] / "inference/agent/tool_agent.py").read_text()
    assert 'os.environ.get("OURS_REASONING_EFFORT", "").strip()' in ours
    assert [line for line in base.splitlines() if line not in set(ours.splitlines())] == []  # it only adds lines
    plus = {line[1:] for line in PATCH.read_text().splitlines() if line.startswith("+") and not line.startswith("+++")}
    assert {line for line in ours.splitlines() if line not in set(base.splitlines())} <= plus
    for stack in ("074t", "081"):
        assert "_static_reasoning_effort()" in (trees[stack] / "inference/agent/tool_agent.py").read_text()


@needs_tree
@needs_bed
@pytest.mark.parametrize("stack", ["ours", "081"])
def test_the_kwargs_carry_the_effort_only_when_set_and_the_ladder_still_wins(trees, stack, tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OURS_", "ARC3_"))}
    run = subprocess.run([str(BED_PY), "-c", PROBE, str(trees[stack])], capture_output=True, text=True, env=env,
                         cwd=tmp_path, timeout=300, check=False)
    assert run.returncode == 0, run.stderr[-3000:]
    got = json.loads(run.stdout.strip().splitlines()[-1])
    assert got["unset rung -1"] == got["unset rung 0"] == {"preserve_thinking": True}  # as before the patch
    assert got["medium rung -1"] == {"preserve_thinking": True, "reasoning_effort": "medium"}
    assert got["medium+ladder rung -1"]["reasoning_effort"] == "medium"
    assert got["medium+ladder rung 0"]["reasoning_effort"] == "low"  # a truncation still steps below it


def test_medium_removes_the_xhigh_instruction_from_the_system_prompt():
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATE.parent)))
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(ValueError(message))
    template = env.get_template(TEMPLATE.name)
    tools = [{"type": "function", "function": {"name": "python", "parameters": {"type": "object"}}}]
    messages = [{"role": "system", "content": "You play a game."}, {"role": "user", "content": "Frame 1"}]

    def render(**kwargs) -> str:
        return template.render(messages=messages, tools=tools, add_generation_prompt=True, **kwargs)

    default, xhigh, medium = render(), render(reasoning_effort="xhigh"), render(reasoning_effort="medium")
    assert default == xhigh and "Reasoning effort is set to xhigh. Please think carefully" in xhigh
    assert "Reasoning effort" not in medium
    assert medium == xhigh.replace(xhigh[xhigh.index("Reasoning effort"):xhigh.index("# Tools")], "")
    with pytest.raises(ValueError, match="Unexpected reasoning effort"):
        render(reasoning_effort="high")
