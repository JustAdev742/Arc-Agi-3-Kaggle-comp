"""Submission path: framework adapter, crash recovery, notebook build."""
import base64
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT / "agent"))

framework = pytest.importorskip("agents.agent", reason="vendor/ARC-AGI-3-Agents missing (run make setup)")


def make_framework_agent(monkeypatch, game_id="ls20", agent="explorer", config=None, budget_s=60):
    monkeypatch.setenv("ARC3_AGENT", agent)
    monkeypatch.setenv("ARC3_AGENT_CONFIG", json.dumps(config or {}))
    monkeypatch.setenv("ARC3_TIME_BUDGET_S", str(budget_s))
    monkeypatch.setenv("ARC3_RESERVE_S", "0")
    import arc3.kaggle as K

    K._deadline = None
    K.DEFAULT_BUDGET_S = float(budget_s)
    K.DEFAULT_RESERVE_S = 0.0
    import importlib

    import my_agent

    importlib.reload(my_agent)
    from arc3.env import make_arcade

    arc = make_arcade("environment_files")
    card = arc.open_scorecard(tags=["test"])
    env = arc.make(game_id, scorecard_id=card)
    a = my_agent.MyAgent(card_id=card, game_id=game_id, agent_name="test", ROOT_URL="http://localhost",
                         record=False, arc_env=env, tags=["test"])
    return a, arc, card


def test_framework_loop_runs_explorer(monkeypatch):
    a, arc, card = make_framework_agent(monkeypatch)
    a.MAX_ACTIONS = 30
    a.main()
    sc = arc.close_scorecard(card)
    run = sc.find_environment("ls20").runs[0]
    assert run.actions == 31  # framework loop runs MAX_ACTIONS+1 times
    assert a.driver.crashes == 0


def test_crash_recovery_switches_to_fallback(monkeypatch):
    from arc3.agents import REGISTRY, register
    from arc3.agents.base import Agent

    @register("boom")
    class Boom(Agent):
        def act(self, frame):
            raise RuntimeError("kaboom")

    try:
        a, arc, card = make_framework_agent(monkeypatch, agent="boom")
        a.MAX_ACTIONS = 10
        a.main()
        assert a.driver.crashes == 1 and a.driver.agent_name == "explorer"
        sc = arc.close_scorecard(card)
        assert sc.find_environment("ls20").runs[0].actions == 11
    finally:
        REGISTRY.pop("boom", None)


def test_deadline_stops_the_game(monkeypatch):
    a, arc, card = make_framework_agent(monkeypatch, budget_s=1)
    a.MAX_ACTIONS = 10**6
    t0 = time.time()
    a.main()
    assert time.time() - t0 < 30
    assert a.action_counter < 10**6


def test_build_notebook_roundtrip(tmp_path):
    out = tmp_path / "submission.ipynb"
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "build_notebook.py"), "--out", str(out), "--agent", "explorer"])
    nb = json.loads(out.read_text())
    assert nb["nbformat"] == 4 and len(nb["cells"]) == 6
    src = "".join(nb["cells"][2]["source"])
    b64 = src.split("'''")[1]
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(b64)), mode="r:gz") as tar:
        names = tar.getnames()
    assert "arc3/__init__.py" in names and "arc3/agents/repl_agent.py" in names
    assert "%%writefile /tmp/my_agent.py" in "".join(nb["cells"][3]["source"])
    assert "KAGGLE_IS_COMPETITION_RERUN" in "".join(nb["cells"][1]["source"])
    assert nb["metadata"]["kaggle"]["isInternetEnabled"] is False
