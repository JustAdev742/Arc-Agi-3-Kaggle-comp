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


def test_mine_skills_extracts_a_card_per_completed_level(tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    from mine_skills import build_library, mine_transcript

    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    obs1 = ("Level 1/3 | step 0 | this level: 0 actions | time left 19m | legal: UP, DOWN, LEFT, RIGHT | state NOT_FINISHED\n\n"
            "Entities (persistent ids; tile 4; roles from evidence): #1 c3 @(0,0) 64x1 [hud]; #2 c5 @(8,8) 4x4; #3 c9 @(40,40) 4x4\n\n"
            "Avatar: #2 moves with keys {'UP': (0, -4), 'RIGHT': (4, 0)}")
    obs2 = ("Level 2/3 | step 9 | this level: 0 actions | time left 15m | legal: UP, DOWN, LEFT, RIGHT | state NOT_FINISHED\n\n"
            "LEVEL 1 COMPLETED after 9 actions on it (the last actions were: RIGHT, RIGHT, UP). Win conditions consistent: touch #2 #3.")
    recs = [
        {"kind": "meta", "game": "zz99", "stats": {"levels_completed": 1}, "notes": []},
        {"kind": "observation", "turn": 1, "text": obs1},
        {"kind": "assistant", "turn": 1, "code": ["# goal: unknown | learned: nothing yet | now: probe keys\nact('UP','DOWN','LEFT','RIGHT')"]},
        {"kind": "tool", "turn": 1, "level": 1, "actions": 4, "output": "..."},
        {"kind": "assistant", "turn": 1, "code": ["# goal: walk #2 onto #3 | learned: RIGHT moves 4px | now: plan\nact(plan_to_entity(3))"]},
        {"kind": "tool", "turn": 1, "level": 2, "actions": 5, "output": "..."},
        {"kind": "observation", "turn": 2, "text": obs2},
    ]
    (run / "zz99.transcript.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    acts = [{"step": i + 1, "action": a, "x": None, "y": None, "levels": 0} for i, a in
            enumerate(["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION4", "ACTION4", "ACTION4", "ACTION4", "ACTION1"])]
    (run / "zz99.jsonl").write_text("\n".join(json.dumps(a) for a in acts) + "\n")
    cards = mine_transcript(run / "zz99.transcript.jsonl")
    assert len(cards) == 1
    c = cards[0]
    assert c["game"] == "zz99" and c["level"] == 1 and c["actions"] == 9
    assert c["goal"] == "walk #2 onto #3" and c["learned"] == "RIGHT moves 4px"
    assert c["sequence"] == "UP, DOWN, LEFT, RIGHTx5, UP"
    assert c["signature"] == {"avatar": True, "click_only": False, "keys": ["RIGHT", "UP"], "many_entities": False, "tile": 4, "hud": True}
    lib = build_library(tmp_path / "runs")
    assert len(lib["skills"]) == 1 and lib["skills"][0]["wins"] == 1
    assert lib["skills"][0]["strategy"].startswith("avatar with keys RIGHT/UP: goal 'walk #2 onto #3'; solved in 9 actions (UP, DOWN, LEFT, RIGHTx5, UP)")


def test_concurrent_adds_and_shared_file_are_safe(tmp_path):
    """Lessons are added from the worker thread while the observation renders on the harness thread, and every
    game of a run appends to the same shared file; no lesson may be lost or duplicated under contention."""
    import threading

    stores = [Lessons(f"g{k}", out_dir=str(tmp_path), others_refresh_s=0.0) for k in range(4)]

    def writer(store: Lessons, k: int) -> None:
        for i in range(50):
            store.add("mechanic", f"game {k} mechanic {i}")
            store.add("mistake", f"game {k} mistake {i}")  # private: never shared

    threads = [threading.Thread(target=writer, args=(st, k)) for k, st in enumerate(stores)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for st in stores:
        assert len(st.items) == 40 and st.cap == 40  # capped, and every add went through the lock
    lines = (tmp_path / "shared_lessons.jsonl").read_text().splitlines()
    assert len(lines) == 4 * 50  # every shared mechanic landed exactly once, none interleaved
    assert all(json.loads(line)["kind"] == "mechanic" for line in lines)
    others = stores[0].others(limit=1000, force=True)
    assert len(others) == 150 and not any(o["game"] == "g0" for o in others)
