"""Lessons store: dedupe, caps, rendering, the run-wide shared file, and skill retrieval."""
import json

from arc3.memory import Lessons, level_signature, load_skills, match_skills, render_skills


def test_lessons_dedupe_cap_and_render(tmp_path):
    m = Lessons("ab12", out_dir=str(tmp_path), cap=5, share=False)
    assert m.add("mistake", "  UP does nothing  next to colour 3 ", level=1)
    assert not m.add("mistake", "up does nothing next to colour 3", level=2)  # same text: reinforced, not duplicated
    assert m.items[0]["n"] == 2 and m.items[0]["level"] == 2
    for i in range(6):
        m.add("mechanic", f"mechanic {i}", level=1)
    assert len(m.items) == 5
    # The cap drops the least valuable kind first (mechanics), never the mistake.
    assert any(it["kind"] == "mistake" for it in m.items)
    m.add("recipe", "Level 1 completed in 9 actions; the last actions were UP, UP, ACT", level=1)
    txt = m.render()
    assert txt.startswith("Lessons (this game")
    lines = txt.splitlines()[1:]
    assert lines[0].startswith("- [recipe L1]"), lines  # recipes first, then mistakes, then mechanics
    assert "(x2)" in txt
    path = m.save()
    data = json.loads(open(path).read())
    assert data["game"] == "ab12" and len(data["lessons"]) == len(m.items) and "key" not in data["lessons"][0]


def test_shared_file_is_read_by_other_games_only(tmp_path):
    a = Lessons("aaaa", out_dir=str(tmp_path), others_refresh_s=0.0)
    b = Lessons("bbbb", out_dir=str(tmp_path), others_refresh_s=0.0)
    assert a.shared_path == b.shared_path
    a.add("hazard", "touching colour 9 ends the game", level=2)
    a.add("mistake", "private mistake", level=2)  # mistakes are not shared
    assert b.render_others() == "" or "private mistake" not in b.render_others()
    others = b.others(force=True)
    assert [o["text"] for o in others] == ["touching colour 9 ends the game"]
    assert a.others(force=True) == []  # a game never reads its own lessons back
    txt = b.render_others()
    assert "From other games" in txt and "[aaaa hazard]" in txt
    # Appending more keeps only the newest unique texts.
    for i in range(10):
        a.add("mechanic", f"mechanic {i}")
    assert len(b.others(limit=4, force=True)) == 4


def test_disabled_store_is_inert(tmp_path):
    m = Lessons("cccc", out_dir=str(tmp_path), enabled=False)
    assert not m.add("mistake", "x")
    assert m.render() == "" and m.save() is None and m.others() == []


def test_skill_matching_prefers_signature_agreement(tmp_path):
    skills = [
        {"game": "g1", "signature": {"avatar": True, "click_only": False, "many_entities": False, "hud": True, "keys": ["UP", "DOWN"]},
         "strategy": "walk the avatar onto the unique-colour tile", "wins": 3, "actions": 12},
        {"game": "g2", "signature": {"avatar": False, "click_only": True, "many_entities": True, "hud": False, "keys": []},
         "strategy": "click every entity of the odd colour", "wins": 2, "actions": 8},
        {"game": "g3", "signature": {"avatar": True, "click_only": False, "many_entities": False, "hud": True, "keys": ["UP"]},
         "strategy": "same shape, fewer wins", "wins": 1, "actions": 30},
    ]
    sig = level_signature(has_avatar=True, click_only=False, keymap={"UP": (0, -1)}, n_entities=5, tile=4, hud=True)
    got = match_skills(skills, sig, limit=2)
    assert [s["game"] for s in got] == ["g1", "g3"]
    assert match_skills(skills, sig, limit=2, exclude_game="g1")[0]["game"] == "g3"
    txt = render_skills(got)
    assert "walk the avatar" in txt and "(won 3x" in txt
    p = tmp_path / "skills.json"
    p.write_text(json.dumps({"skills": skills}))
    assert len(load_skills(str(p))) == 3
    assert load_skills(str(tmp_path / "missing.json")) == []
