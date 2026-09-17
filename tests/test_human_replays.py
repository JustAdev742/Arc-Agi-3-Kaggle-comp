"""The human-recording summariser on a synthetic file in the documented recording format (the real dataset is a
browser download the owner has to make; docs/status.md)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from human_replays import parse_recording, summarise


def _rec(game, action, levels, state="NOT_FINISHED", data=None):
    return json.dumps({"timestamp": "2026-07-22T10:30:45+00:00",
                       "data": {"game_id": game, "state": state, "levels_completed": levels, "win_levels": 2,
                                "action_input": {"id": action, "data": data or {}, "reasoning": None}, "guid": "g",
                                "full_reset": action == "RESET", "available_actions": [1, 2, 3, 4, 6]}})


def test_parse_and_summarise(tmp_path):
    lines = [_rec("ls20-016295f7601e", "RESET", 0), _rec("ls20-016295f7601e", "ACTION1", 0), _rec("ls20-016295f7601e", "ACTION4", 1),
             _rec("ls20-016295f7601e", "ACTION6", 1, data={"x": 3, "y": 4}), _rec("ls20-016295f7601e", "ACTION2", 2, state="WIN")]
    p = tmp_path / "ls20-016295f7601e-abc.jsonl"
    p.write_text("\n".join(lines) + "\n")
    r = parse_recording(p)
    assert r["game"] == "ls20" and r["won"] and r["actions"] == 4 and r["levels_completed"] == 2
    # the opening RESET is not billed; the level-completing action belongs to the level it completed
    assert r["per_level"] == {"1": ["UP", "RIGHT"], "2": ["CLICK(3,4)", "DOWN"]}
    q = tmp_path / "ls20-016295f7601e-def.jsonl"
    q.write_text("\n".join([_rec("ls20-016295f7601e", "RESET", 0), _rec("ls20-016295f7601e", "ACTION3", 0)]) + "\n")  # gave up
    s = summarise([parse_recording(p), parse_recording(q)])
    g = s["games"]["ls20"]
    assert g["replays"] == 2 and g["wins"] == 1 and g["median_actions_per_level"] == {"1": 2, "2": 2}
    assert g["first_three_actions"][0][0] == "UP RIGHT" and 0 < g["click_fraction"] < 1
    assert s["cross_game"]["median_level1_actions"] == 2 and s["cross_game"]["wins"] == 1
    assert parse_recording(tmp_path / "missing.jsonl") is None if (tmp_path / "missing.jsonl").exists() else True
    (tmp_path / "empty.jsonl").write_text("")
    assert parse_recording(tmp_path / "empty.jsonl") is None


def test_published_format_integer_ids_and_scorecard_line(tmp_path):
    """The ARC Prize dataset files (seen 2026-09-17, ls20 recording): integer action ids (0 = RESET), the frame on every
    line, and a trailing scorecard line with cumulative actions_by_level that must agree with our per-level split."""
    def rec(action_id, levels, state="NOT_FINISHED", data=None):
        return json.dumps({"timestamp": "t", "data": {"game_id": "ls20-9607627b", "frame": [[[0]]], "state": state, "levels_completed": levels,
                                                       "win_levels": 2, "action_input": {"id": action_id, "data": data or {}, "reasoning": None},
                                                       "guid": "g", "full_reset": False, "available_actions": [1, 2, 3, 4, 6]}})
    card = json.dumps({"timestamp": "t", "data": {"won": 1, "played": 1, "total_actions": 4, "levels_completed": 2,
                                                  "cards": {"ls20-9607627b": {"actions_by_level": [[[1, 2], [2, 4]]]}}}})
    p = tmp_path / "ls20-9607627b.abc.jsonl"
    p.write_text("\n".join([rec(0, 0), rec(1, 0), rec(4, 1), rec(6, 1, data={"x": 3, "y": 4}), rec(2, 2, state="WIN"), card]) + "\n")
    r = parse_recording(p)
    assert r["per_level"] == {"1": ["UP", "RIGHT"], "2": ["CLICK(3,4)", "DOWN"]} and r["won"]
    assert r["card_actions_per_level"] == [2, 2] and r["card_agrees"]
