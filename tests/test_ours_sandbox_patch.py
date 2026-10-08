"""kaggle/franzen/patches/ours-sandbox-timeout-keeps-work.patch: a timed-out python tool call keeps the retained
functions (it used to clear them all: r11l lost 13 helpers to one 30 s timeout), and `import time` is allowed."""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import franzen_tree  # noqa: E402

PATCH = ROOT / "kaggle" / "franzen" / "patches" / "ours-sandbox-timeout-keeps-work.patch"
BED_PY = Path.home() / ".cache" / "arc3-franzen-bed" / "venv" / "bin" / "python"


def _has_tree_source() -> bool:
    return (franzen_tree.his_repo_path() / "ARC3-Inference").is_dir()


needs_tree = pytest.mark.skipif(not _has_tree_source(), reason="needs Franzen's repo (scripts/franzen_tree.py)")


@pytest.fixture(scope="module")
def patched(tmp_path_factory):
    dest = tmp_path_factory.mktemp("share")
    logs = franzen_tree.notebook_bundle(dest, [PATCH])
    assert PATCH.name in logs
    return dest / "src" / "ARC3-Inference"


@needs_tree
def test_patch_applies_on_the_notebook_tree(patched):
    agent = (patched / "inference/agent/tool_agent.py").read_text()
    sandbox = (patched / "inference/agent/python_tool_sandbox.py").read_text()
    assert 'startswith("Tool timed out")' in agent
    assert '        "time",\n' in sandbox


CHECK = textwrap.dedent('''
    import os, sys
    sys.path.insert(0, sys.argv[1])
    os.environ["ARC3_PERSISTENT_FUNCTIONS"] = "1"
    os.environ["ARC3_PERSISTENT_FUNCTIONS_SCOPE"] = "game"
    from inference.agent import tool_agent as ta
    from inference.agent import python_tool_sandbox as sb

    class Fake:
        pass

    f = Fake()
    f._kept_functions = {"plan": "def plan(x):\\n    return x\\n", "foot_ok": "def foot_ok(a, b):\\n    return a == b\\n"}
    payload = {}
    ta.ToolAgent._record_retained_functions(f, {"error": "Tool timed out after 30s", "stdout": ""}, payload)
    assert set(f._kept_functions) == {"plan", "foot_ok"}, f._kept_functions
    assert "still available" in payload["function_retention"]
    payload = {}
    ta.ToolAgent._record_retained_functions(f, {"keepable_functions": [{"name": "plan", "source": f._kept_functions["plan"]}]}, payload)
    assert set(f._kept_functions) == {"plan"}  # the normal path is unchanged
    r = sb.run_sandboxed_python(code="import time\\nprint(time.time() > 0)\\n", timeout_seconds=10, initial_state={},
                                action_handler=lambda *a, **k: {})
    assert (r.get("stdout") or "").strip() == "True", r
    r = sb.run_sandboxed_python(code="import time\\ntime.sleep(5)\\n", timeout_seconds=2, initial_state={},
                                action_handler=lambda *a, **k: {})
    assert str(r.get("error")).startswith("Tool timed out") and not r.get("keepable_functions"), r
    print("ok")
''')


@needs_tree
@pytest.mark.skipif(not BED_PY.exists(), reason="needs the franzen_bed venv (scripts/franzen_bed.py creates it)")
def test_timeout_keeps_retained_functions_and_time_imports(patched, tmp_path):
    script = tmp_path / "check.py"
    script.write_text(CHECK)
    r = subprocess.run([str(BED_PY), str(script), str(patched)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout + r.stderr
