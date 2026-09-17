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
        assert a.driver.crashes == 1 and a.driver.agent_name == "rules"
        sc = arc.close_scorecard(card)
        assert sc.find_environment("ls20").runs[0].actions == 11
    finally:
        REGISTRY.pop("boom", None)


def test_deadline_stops_the_game(monkeypatch):
    a, _arc, _card = make_framework_agent(monkeypatch, budget_s=1)
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
    assert "arc3/memory.py" in names and "arc3/data/skills.json" in names  # the offline skill library ships with the code
    assert "LICENSE" in names and "Apache License 2.0" in "".join(nb["cells"][0]["source"])  # the public copy carries its license
    assert "ARC3_MEMORY_PATH" in "".join(nb["cells"][1]["source"])
    assert "%%writefile /tmp/my_agent.py" in "".join(nb["cells"][3]["source"])
    assert "KAGGLE_IS_COMPETITION_RERUN" in "".join(nb["cells"][1]["source"])
    assert nb["metadata"]["kaggle"]["isInternetEnabled"] is False


def test_click_coordinates_do_not_go_through_the_shared_enum(monkeypatch):
    """Two games choosing clicks concurrently must each send their own (x, y)."""
    from arcengine import GameAction

    a, arc, card = make_framework_agent(monkeypatch, game_id="vc33")
    a.driver.last_data = {"x": 7, "y": 9}
    GameAction.ACTION6.set_data({"x": 60, "y": 61})  # another thread clobbered the shared member
    sent = {}

    def fake_step(action, data=None, reasoning=None):
        sent.update(data or {})
        return a.arc_env.observation_space

    monkeypatch.setattr(a.arc_env, "step", fake_step)
    a.do_action_request(GameAction.ACTION6)
    assert sent == {"x": 7, "y": 9}
    arc.close_scorecard(card)


def test_notebook_specialist_ladder_source():
    """The council notebook tries every attached specialist checkpoint through the ladder before sharing the coordinator."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_notebook import specialist_refs, vllm_setup_source
    src = vllm_setup_source("o/model", "o/wheels", "o/spec-fp8,o/spec-nvfp4")
    assert specialist_refs("o/spec-fp8, o/spec-nvfp4,") == ["o/spec-fp8", "o/spec-nvfp4"]
    assert "/kaggle/input/spec-fp8" in src and "/kaggle/input/spec-nvfp4" in src
    assert "serve.specialist_attempts(SPECIALIST_DIRS, gpu_mem=0.30)" in src and "attempts=ladder" in src
    assert "gpu_mem=0.60 if two else 0.90" in src and "SPECIALIST SERVER FAILED" in src
    src_single = vllm_setup_source("o/model", "o/wheels", "")
    assert "SPECIALIST_DIRS = [d for d in [] if d]" in src_single
    compile(src, "<cell>", "exec")
    compile(src_single, "<cell>", "exec")


def test_eval_notebook_cells_compile_with_apostrophe_in_note():
    """exp-018 (2026-09-16) died at cell 4 with SyntaxError: the run note contained an apostrophe inside a single-quoted
    string. Every generated code cell must compile whatever the note says."""
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_eval_notebook import build
    a = argparse.Namespace(agent="repl", split="dev", seed=0, time_per_game=1200, max_actions=2000, workers=8, budget_min=150,
                           config='{"reasoning_effort": "low"}', run_name="t", note="the submission's operating point (3 h) \"quoted\"",
                           model_dataset="o/m", wheels_dataset="o/w", specialist_dataset="", username="u", slug="s")
    nb = build(a)
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            compile(cell["source"], f"<cell {i}>", "exec")


def test_driver_resets_after_game_over_and_converts_frames(monkeypatch):
    """The framework hands the Driver only the latest frame: a GAME_OVER frame must yield RESET (the only legal
    action), and the conversion keeps every layer with the decision frame last."""
    from arcengine import FrameData, GameAction, GameState

    from arc3.kaggle import Driver, to_frame

    monkeypatch.setenv("ARC3_RESERVE_S", "0")
    g0 = [[0] * 64 for _ in range(64)]
    g1 = [[1] * 64 for _ in range(64)]
    fd = FrameData(game_id="ls20", frame=[g0, g1], state=GameState.GAME_OVER, levels_completed=1, win_levels=3,
                   available_actions=[0], full_reset=False)
    f = to_frame(fd, step=5, level_step=2)
    assert len(f.layers) == 2 and f.grid[0, 0] == 1 and f.layers[0][0, 0] == 0
    assert f.game_over and f.level == 2 and f.step == 5 and f.level_step == 2
    d = Driver("ls20", "rules", deadline=time.time() + 60)
    try:
        assert d.choose(fd) is GameAction.RESET and d.step == 1
        assert d.done(fd) is False  # game over is not the end of the game: the level restarts
        assert d.done(FrameData(game_id="ls20", frame=[g1], state=GameState.WIN, levels_completed=3, win_levels=3,
                                available_actions=[], full_reset=False)) is True
    finally:
        d.close()


def test_notebook_model_ref_mount_and_custom_ladder(tmp_path):
    """A candidate model attached as a Kaggle model (owner/slug/framework/variation/version) mounts under
    /kaggle/input/models/...; the setup cell resolves it to the folder holding config.json and can start the server
    with a custom attempt ladder (parsers, MoE backend, no image limit) instead of the 27B defaults."""
    import json as _json
    import subprocess
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_notebook import source_lists, vllm_setup_source
    assert source_lists("o/wheels", "o/m/transformers/default/1", "", "o/spec") == (["o/wheels", "o/spec"], ["o/m/transformers/default/1"])
    ladder = '[{"label": "x", "mtp_tokens": 0, "images_per_prompt": 0, "extra": "--reasoning-parser-plugin {MODEL_DIR}/p.py"}]'
    src = vllm_setup_source("o/m/transformers/default/1", "o/wheels", "", attempts_json=ladder)
    assert "/kaggle/input/models/o/m/transformers/default/1" in src and "config.json" in src and "import glob" in src
    assert "attempts=attempts" in src and "replace('{MODEL_DIR}', MODEL_DIR)" in src
    compile(src, "<cell>", "exec")
    # the default (no ladder) path is unchanged for the 27B datasets
    src27 = vllm_setup_source("o/m", "o/wheels", "")
    assert "ATTEMPTS_JSON = ''" in src27 and "/kaggle/input/m'" in src27
    compile(src27, "<cell>", "exec")
    # the eval builder files the model ref under model_sources and keeps the wheelhouse under dataset_sources
    out = tmp_path / "nb"
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "build_eval_notebook.py"), "--agent", "repl", "--split", "dev",
                          "--model-dataset", "o/m/transformers/default/1", "--wheels-dataset", "o/wheels", "--slug", "t",
                          "--config", '{"image": false, "effort_in_request": true}', "--attempts-json", ladder, "--out", str(out)])
    meta = _json.loads((out / "kernel-metadata.json").read_text())
    assert meta["model_sources"] == ["o/m/transformers/default/1"] and meta["dataset_sources"] == ["o/wheels"]
    nb = _json.loads((out / "eval.ipynb").read_text())
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"<cell {i}>", "exec")
    assert any("ATTEMPTS_JSON = '[{" in "".join(c["source"]) for c in nb["cells"])
