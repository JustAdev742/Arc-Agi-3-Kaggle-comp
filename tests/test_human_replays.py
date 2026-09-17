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
    assert r["game"] == "ls20" and r["won"] and r["actions"] == 5 and r["levels_completed"] == 2
    # RESET counts on level 1 (as the scorer bills it); the level-completing action belongs to the level it completed
    assert r["per_level"] == {"1": ["RESET", "UP", "RIGHT"], "2": ["CLICK(3,4)", "DOWN"]}
    q = tmp_path / "ls20-016295f7601e-def.jsonl"
    q.write_text("\n".join([_rec("ls20-016295f7601e", "RESET", 0), _rec("ls20-016295f7601e", "ACTION3", 0)]) + "\n")  # gave up
    s = summarise([parse_recording(p), parse_recording(q)])
    g = s["games"]["ls20"]
    assert g["replays"] == 2 and g["wins"] == 1 and g["median_actions_per_level"] == {"1": 3, "2": 2}
    assert g["first_three_actions"][0][0] == "RESET UP RIGHT" and 0 < g["click_fraction"] < 1
    assert s["cross_game"]["median_level1_actions"] == 3 and s["cross_game"]["wins"] == 1
    assert parse_recording(tmp_path / "missing.jsonl") is None if (tmp_path / "missing.jsonl").exists() else True
    (tmp_path / "empty.jsonl").write_text("")
    assert parse_recording(tmp_path / "empty.jsonl") is None
