"""kaggle/franzen/patches/ours-05-level-mem.patch (M1c): `mem`, a dict the python sandbox keeps across calls on one
level, behind OURS_LEVEL_MEM=1 (read at call time, default off).

The scenario below runs real `ToolAgent._run_python_tool` calls (each one a real sandbox subprocess) against a small
fake game, in the bed venv, on the tree the notebook builds: his patch, ours-01, then this patch. With the flag off
the system prompt and every tool result must be byte-identical to the tree without this patch.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import franzen_bed  # noqa: E402
import franzen_tree  # noqa: E402

PATCHES = ROOT / "kaggle" / "franzen" / "patches"
OURS01 = PATCHES / "ours-sandbox-timeout-keeps-work.patch"
PATCH = PATCHES / "ours-05-level-mem.patch"
BED_PY = Path.home() / ".cache" / "arc3-franzen-bed" / "venv" / "bin" / "python"
PROMPT_LINES = (
    "- `mem` is a dict kept across `python` calls on the current level; it is emptied when the level changes. "
    "Other variables still reset.\n"
    "- It holds only JSON data (str keys; lists, not tuples or sets) up to 200 KB. A call that breaks this has its "
    "changes to `mem` refused.\n"
    "- Tool results list mem's keys with the `current_frame.step` at which each was last written.\n"
)

needs_tree = pytest.mark.skipif(not (franzen_tree.his_repo_path() / "ARC3-Inference").is_dir(),
                                reason="needs Franzen's repo (scripts/franzen_tree.py)")
needs_bed = pytest.mark.skipif(not BED_PY.exists(), reason="needs the franzen_bed venv (scripts/franzen_bed.py)")

DRIVER = textwrap.dedent('''
    import json, sys, tempfile
    from pathlib import Path
    sys.path.insert(0, sys.argv[1])
    from inference.agent import tool_agent as ta
    from inference.agent.runtime_state import RUNTIME_STATE_FILENAME, Frame, HistoryEntry, write_runtime_state

    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

    class Game:
        """UP changes a cell, DOWN wins the level, LEFT is a game over (the level stays)."""
        def __init__(self, path):
            self.path, self.step, self.level, self.history = path, 0, 1, []
            self.grid = [[0] * 8 for _ in range(8)]
            self.write()
        def frame(self):
            return Frame(grid=tuple(tuple(row) for row in self.grid), step=self.step, level=self.level)
        def write(self):
            write_runtime_state(self.path, current_frame=self.frame(), history=self.history)
        def __call__(self, arguments):
            actions, last = arguments.get("actions") or [], None
            for index, item in enumerate(actions):
                name = item["action"]
                self.step += 1
                won, over = name == "DOWN", name == "LEFT"
                if name == "UP":
                    self.grid[self.step % 8][0] = 1 + self.step % 9
                if won:
                    self.level += 1
                    self.grid = [[self.level] * 8 for _ in range(8)]
                self.history.append(HistoryEntry(action=name, frame=self.frame(), result={}))
                self.write()
                last = {"executed": True, "action_num": self.step, "level": self.level, "score": self.level - 1,
                        "state": "GAME_OVER" if over else "NOT_FINISHED", "valid_actions": ACTIONS,
                        "board_changed": name in ("UP", "DOWN"), "gameplay_changed": name in ("UP", "DOWN"),
                        "done": False, "level_completed": won, "game_over": over, "run_complete": False,
                        "action_name": name, "action_display": name, "requested_count": len(actions),
                        "executed_count": index + 1}
                if won or over:
                    break
            return last

    SCENARIO = [
        ("write", "mem['walls'] = [[1, 2], [3, 4]]\\nmem['door'] = {'r': 5, 'c': 6}\\nprint(sorted(mem))"),
        ("read", "print(mem['walls'], mem['door']['r'])"),
        ("write-act-write", "mem['before'] = current_frame.step\\naction('UP')\\nmem['after'] = current_frame.step"),
        ("in-place", "mem['walls'].append([7, 8])\\naction('UP')"),
        ("retained", "def walls_n():\\n    return len(mem['walls'])\\nprint(walls_n())"),
        ("refuse-set", "mem['ok'] = 1\\nmem['seen'] = {1, 2}"),
        ("refuse-tuple", "mem['walls'][0] = (1, 2)"),
        ("refuse-key", "mem['dist'] = {(1, 2): 3}"),
        ("refuse-cap", "mem['big'] = 'x' * 300000"),
        ("refuse-replaced", "mem = [1, 2]"),
        ("after-refusals", "print(sorted(mem), 'ok' in mem, mem['walls'][0])"),
        ("timeout", "mem['t'] = 1\\nwhile True:\\n    pass"),
        ("after-timeout", "print(sorted(mem), walls_n())"),
        ("error", "mem['e'] = 1\\nraise ValueError('boom')"),
        ("game-over", "mem['g'] = 1\\naction('LEFT')\\nprint(current_frame.level)"),
        ("level-up", "mem['x'] = 1\\naction('DOWN')\\nprint(current_frame.level)"),
        ("new-level", "print(sorted(mem))"),
        ("write-l2", "mem['l2'] = 'two'"),
        ("<level 3 outside a call>", None),
        ("between", "print(sorted(mem))"),
        ("write-l3", "mem['c'] = [1]"),
        ("clear", "mem.clear()"),
        ("deleted", "mem['d'] = 1\\ndel mem"),
        ("plain", "print(1 + 1)"),
    ]

    agent = ta.ToolAgent(model="mock", base_url="http://127.0.0.1:9/v1")
    game = Game(Path(tempfile.mkdtemp()) / f"g_{RUNTIME_STATE_FILENAME}")
    agent._step_env_callback = game
    agent._current_valid_actions = list(ACTIONS)
    out = {"system_prompt": agent._system_prompt, "results": {}}
    for name, code in SCENARIO:
        if code is None:
            game.level += 1
            game.write()
            continue
        agent._python_timeout = 2 if name == "timeout" else 30
        out["results"][name] = agent._run_python_tool(game.path, {"code": code}).content
    print(json.dumps(out))
''')


def _clean_env() -> dict[str, str]:
    """This process's environment without anything that steers the harness."""
    prefixes = ("ARC3_", "LOCAL_ANALYZER_", "OURS_", "OPENAI_", "INFERENCE_", "MULTIMODAL_", "EXPOSE_")
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith(prefixes)}


def _drive(tree: Path, script: Path, env: dict[str, str]) -> str:
    r = subprocess.run([str(BED_PY), str(script), str(tree / "src" / "ARC3-Inference")], capture_output=True,
                       text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    base, patched = tmp_path_factory.mktemp("base"), tmp_path_factory.mktemp("patched")
    franzen_tree.notebook_bundle(base, [OURS01])
    logs = franzen_tree.notebook_bundle(patched, [OURS01, PATCH])
    assert list(logs) == ["his", OURS01.name, PATCH.name]
    return base, patched


@pytest.fixture(scope="module")
def runs(trees, tmp_path_factory):
    """The scenario on both trees: the notebook's harness environment, and the harness defaults."""
    base, patched = trees
    script = tmp_path_factory.mktemp("driver") / "drive.py"
    script.write_text(DRIVER)
    notebook, _ = franzen_bed.notebook_env(franzen_bed._cell(franzen_tree.notebook_cells(), "setup_env = {"))
    out = {}
    for env_name, extra in (("notebook", notebook), ("default", {})):
        for tree_name, tree, flag in (("base", base, None), ("patched", patched, None), ("patched", patched, "0"),
                                      ("patched", patched, "1")):
            if env_name == "default" and flag is not None:
                continue
            env = {**_clean_env(), **extra, **({} if flag is None else {"OURS_LEVEL_MEM": flag})}
            out[env_name, tree_name, flag] = _drive(tree, script, env)
    return out


def _results(raw: str) -> dict[str, dict]:
    return {name: json.loads(content) for name, content in json.loads(raw)["results"].items()}


@needs_tree
def test_patch_applies_on_ours01_and_adds_its_files(trees):
    src = trees[1] / "src" / "ARC3-Inference" / "inference"
    assert "_ours_mem_result()" in (src / "agent" / "python_tool_sandbox.py").read_text()
    assert 'payload["mem"] = ours_mem_text' in (src / "agent" / "tool_agent.py").read_text()
    assert (src / "utils" / "level_mem.py").is_file()


@needs_tree
@needs_bed
def test_flag_off_is_byte_identical_to_the_tree_without_the_patch(runs):
    for env_name in ("notebook", "default"):
        assert runs[env_name, "patched", None] == runs[env_name, "base", None], env_name
    assert runs["notebook", "patched", "0"] == runs["notebook", "base", None]
    on, off = (json.loads(runs["notebook", tree, flag])["system_prompt"] for tree, flag in (("patched", "1"),
                                                                                          ("base", None)))
    assert on == off + PROMPT_LINES  # the three lines close the tool session rules
    assert "NameError: name 'mem' is not defined" in _results(runs["notebook", "base", None])["read"]["error"]


@needs_tree
@needs_bed
def test_mem_survives_calls_and_lists_keys_with_their_write_steps(runs):
    r = _results(runs["notebook", "patched", "1"])
    assert r["write"]["mem"] == "walls (step 0), door (step 0); now step 0, 44 bytes of 200 KB"
    assert r["read"]["stdout"] == "[[1, 2], [3, 4]] 5\n"
    # a write records the step it happened at; a change inside a stored value, the step the call ended at
    assert r["write-act-write"]["mem"].startswith("walls (step 0), door (step 0), before (step 0), after (step 1);")
    assert r["in-place"]["mem"].startswith("walls (step 2), door (step 0)")
    assert r["retained"]["stdout"] == "3\n" and "walls_n()" in r["retained"]["function_retention"]
    assert r["error"]["mem"].split(";")[0].endswith("e (step 2)")  # writes before an exception are kept
    assert r["game-over"]["stdout"] == "1\n" and "g (step 2)" in r["game-over"]["mem"]  # a death keeps the level
    assert "mem" not in r["plain"]  # an empty mem costs no tokens


@needs_tree
@needs_bed
def test_refused_writes_name_the_value_and_keep_the_previous_mem(runs):
    r = _results(runs["notebook", "patched", "1"])
    kept = ". mem keeps its contents from before this call. Keys: walls (step 2), door (step 0), "
    json_only = "; JSON gives back only dict, list, str, int, float, bool and None"
    assert r["refuse-set"]["mem"].startswith("NOT SAVED: mem['seen'] is a set" + json_only + kept)
    assert r["refuse-tuple"]["mem"].startswith("NOT SAVED: mem['walls'][0] is a tuple" + json_only + kept)
    assert r["refuse-key"]["mem"].startswith("NOT SAVED: mem['dist'] has the key (1, 2) (a tuple); JSON keeps "
                                             "only str keys" + kept)
    assert r["refuse-cap"]["mem"].startswith("NOT SAVED: mem would be 293.0 KB, over its 200 KB limit" + kept)
    assert r["refuse-replaced"]["mem"].startswith("NOT SAVED: `mem` was replaced by a list" + kept)
    # never half-written: the valid write made in the refused call is gone too
    assert r["after-refusals"]["stdout"] == "['after', 'before', 'door', 'walls'] False [1, 2]\n"
    assert r["deleted"]["mem"] == "NOT SAVED: `mem` was deleted. mem keeps its contents from before this call."


@needs_tree
@needs_bed
def test_a_timeout_keeps_the_previous_mem_and_functions(runs):
    r = _results(runs["notebook", "patched", "1"])
    assert r["timeout"]["error"] == "Tool timed out after 2s"
    assert r["timeout"]["mem"].startswith("This call did not finish, so mem keeps its contents from before it. "
                                          "Keys: walls (step 2), door (step 0)")
    assert "t (step" not in r["timeout"]["mem"] and "still available" in r["timeout"]["function_retention"]
    assert r["after-timeout"]["stdout"] == "['after', 'before', 'door', 'walls'] 3\n"


@needs_tree
@needs_bed
def test_mem_is_emptied_at_a_level_change(runs):
    r = _results(runs["notebook", "patched", "1"])
    assert r["level-up"]["mem"] == ("mem was emptied because the level changed during this call (level 1 -> 2); "
                                    "nothing this call wrote to it was kept.")
    assert r["new-level"]["stdout"] == "[]\n" and "mem" not in r["new-level"]
    assert r["write-l2"]["mem"] == "l2 (step 4); now step 4, 12 bytes of 200 KB"
    assert r["between"] == {"tool": "python", "mem": "mem was emptied because the level changed (level 2 -> 3).",
                            "returncode": 0, "stdout": "[]\n"}
    assert r["clear"]["mem"] == "mem is empty now."


@needs_tree
def test_the_host_store_lists_keys_compactly(trees):
    spec = importlib.util.spec_from_file_location(
        "level_mem", trees[1] / "src" / "ARC3-Inference" / "inference" / "utils" / "level_mem.py")
    lm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lm)
    store = lm.open_store(None, "/game", 2)
    assert lm.sandbox_payload(store) == {"values": {}, "steps": {}, "cap": 200 * 1024}
    assert lm.close_store(store, None, 2, 7) == (store, "")  # nothing to say about an empty mem
    values = {"x" * 50: 1} | {f"k{i}": i for i in range(45)}
    kept, text = lm.close_store(store, {"values": values, "steps": {"k0": 3}, "bytes": 3000}, 2, 7)
    assert text.startswith("x" * 37 + "... (step 7), k0 (step 3), k1 (step 7)")
    assert "k38 (step 7), and 6 more keys" in text
    assert text.endswith("; now step 7, 2.9 KB of 200 KB")
    assert lm.open_store(kept, "/game", 2) is kept
    assert lm.open_store(kept, "/other-game", 1)["note"] == ""  # a new game starts empty, without a note
    assert lm.close_store(lm.open_store(None, "/game", 2), {"values": {}}, 3, 9)[1] == ""


def test_the_bed_has_a_mem_program_that_writes_then_reads():
    assert "mem" in franzen_bed.OPTIONAL_PROGRAMS
    cycle = franzen_bed.CYCLE + ["mem"]
    system, opener = {"role": "system", "content": "s"}, {"role": "user", "content": "x"}
    think = {"role": "assistant", "content": "bed:think:0"}
    assert franzen_bed.next_step([system, opener, think, opener], cycle) == ("mem", 0)
    call = {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
        "name": "python", "arguments": json.dumps({"code": franzen_bed.SNIPPETS["mem"][0]})}}]}
    tool = {"role": "tool", "tool_call_id": "c1", "content": "BED mem write"}
    assert franzen_bed.next_step([system, opener, call, tool], cycle) == ("mem", 1)
    assert "action(" not in franzen_bed.SNIPPETS["mem"][0] and "action(" in franzen_bed.SNIPPETS["mem"][1]
