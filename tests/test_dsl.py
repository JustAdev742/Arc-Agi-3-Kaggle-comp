import numpy as np

from arc3 import dsl
from arc3.dsl import (
    Cls,
    Ent,
    Move,
    OnOverlap,
    Push,
    Recolor,
    Vanish,
    auto_rules,
    explain,
    fit,
    goal_predicates,
    make_log,
    plan,
    simulate,
)
from arc3.entities import Tracker


def E(i, color, x, y, w=1, h=1, shape=None):
    return Ent(i, color, x, y, w, h, w * h, shape or f"s{w}x{h}")


def walls():
    # a box of colour-5 walls around a 10x10 arena at (0..11)
    out = [E(100 + k, 5, k, 0, 1, 1) for k in range(12)] + [E(120 + k, 5, k, 11, 1, 1) for k in range(12)]
    out += [E(140 + k, 5, 0, k, 1, 1) for k in range(1, 11)] + [E(160 + k, 5, 11, k, 1, 1) for k in range(1, 11)]
    return out


def test_fit_move_learns_keymap_and_wall_blocking():
    W = walls()
    av = E(1, 1, 2, 4, 2, 2)
    frames = [tuple(W + [av])]
    actions = []
    # RIGHT x3 (moves 2 each), UP (moves to y=2), UP (would enter the wall at y=0 -> stays), LEFT
    for a, d in [("RIGHT", (2, 0)), ("RIGHT", (2, 0)), ("RIGHT", (2, 0)), ("UP", (0, -2)), ("UP", (0, 0)), ("LEFT", (-2, 0))]:
        av = av.moved(*d)
        frames.append(tuple(W + [av]))
        actions.append(a)
    log = make_log(frames, actions)
    rules = fit("move", log)["move"]
    assert rules, "no move rule fitted"
    mv, score = rules[0]
    assert isinstance(mv, Move) and score.contradictions == 0
    assert mv.keymap["RIGHT"] == (2, 0) and mv.keymap["UP"] == (0, -2) and mv.keymap["LEFT"] == (-2, 0)
    assert mv.keymap["DOWN"] == (0, 2)  # mirrored guess for the unpressed key
    assert mv.blocked_by == "any" or 5 in mv.blocked_by
    rep = explain([mv], log)
    assert rep["contradictions"] == 0 and rep["fully_explained"] == len(log)


def test_fit_push_and_blocked_push():
    W = walls()
    av = E(1, 1, 2, 5)
    box = E(2, 3, 4, 5)
    frames = [tuple(W + [av, box])]
    actions = []
    # RIGHT: avatar to 3; RIGHT: avatar into box -> both move; RIGHT x5 until the box hits the wall at x=11
    xs = [(3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (9, 10)]
    for ax, bx in xs:
        av = E(1, 1, ax, 5)
        box = E(2, 3, bx, 5)
        frames.append(tuple(W + [av, box]))
        actions.append("RIGHT")
    log = make_log(frames, actions)
    res = fit(None, log)
    assert res["push"], "no push rule fitted"
    push, score = res["push"][0]
    assert isinstance(push, Push) and push.pushable.color == 3 and score.contradictions == 0
    rules, rep = auto_rules(log)
    assert rep["contradictions"] == 0 and rep["fully_explained"] == len(log), rep
    assert any(isinstance(r, Push) for r in rules)
    # simulate: pushing against the wall moves nothing
    last = frames[-1]
    assert dsl.frame_key(simulate(last, "RIGHT", rules)) == dsl.frame_key(last)


def test_overlap_vanish_and_plan_to_collect():
    W = walls()
    av = E(1, 1, 2, 2)
    key = E(2, 4, 6, 2)
    frames = [tuple(W + [av, key])]
    actions = []
    for x in (3, 4, 5):
        av = E(1, 1, x, 2)
        frames.append(tuple(W + [av, key]))
        actions.append("RIGHT")
    av = E(1, 1, 6, 2)
    frames.append(tuple(W + [av]))  # the key vanishes when covered
    actions.append("RIGHT")
    log = make_log(frames, actions)
    rules, rep = auto_rules(log)
    kinds = {r.kind for r in rules}
    assert "move" in kinds and "overlap" in kinds, rules
    assert rep["fully_explained"] == len(log), rep
    # plan from the start frame to collect the key: 4 RIGHTs
    def goal(f):
        return not any(e.color == 4 for e in f)

    acts = dsl.planning_actions(rules, frames[0])
    path = plan(rules, frames[0], goal, acts)
    assert path == ["RIGHT"] * 4, path


def test_recolor_toggle_on_click_and_counter():
    hud = E(9, 7, 0, 63, 20, 1)
    a = E(1, 2, 10, 10, 3, 3)
    b = E(2, 2, 20, 10, 3, 3)
    frames = [(hud, a, b)]
    actions = []
    # click a: a -> 8; click b: b -> 8; click a: a -> 2; HUD shrinks by 2 cells per action
    seq = [(("CLICK", 11, 11), 8, 2), (("CLICK", 21, 11), 8, 8), (("CLICK", 11, 11), 2, 8)]
    size = 20
    for act, ca, cb in seq:
        size -= 2
        frames.append((Ent(9, 7, 0, 63, size, 1, size, f"s{size}x1"), a.recolored(ca), b.recolored(cb)))
        actions.append(act)
    log = make_log(frames, actions)
    res = fit(None, log)
    rec = [r for r, _ in res["recolor"]]
    assert any(isinstance(r, Recolor) and r.trigger == "CLICK@self" and r.toggle for r in rec), rec
    assert res["counter"] and res["counter"][0][0].per == 2
    _rules, rep = auto_rules(log)
    assert rep["contradictions"] == 0 and rep["coverage"] == 1.0, rep


def test_explain_reports_unexplained_events():
    W = walls()
    av = E(1, 1, 2, 2)
    door = E(2, 6, 8, 8)
    frames = [tuple(W + [av, door]), tuple(W + [E(1, 1, 3, 2), door]), tuple(W + [E(1, 1, 4, 2)])]
    actions = ["RIGHT", "RIGHT"]
    log = make_log(frames, actions)
    mv = Move(Cls(color=1), {"RIGHT": (1, 0)}, "any")
    rep = explain([mv], log)
    assert rep["contradictions"] == 0
    assert rep["explained"] == 2 and rep["events"] == 3
    assert rep["unexplained"] and rep["unexplained"][0]["event"] == "gone" and rep["unexplained"][0]["id"] == 2
    # a wrong rule is contradicted with an example
    bad = Move(Cls(color=1), {"RIGHT": (2, 0)}, "any")
    rep2 = explain([bad], log)
    assert rep2["contradictions"] == 2 and rep2["contradicted"][0]["claim"].moved == (2, 0)


def test_vanish_on_act_trigger():
    a = E(1, 3, 5, 5)
    b = E(2, 3, 9, 9)
    c = E(3, 4, 1, 1)
    log = make_log([(a, b, c), (a, b, c), (c,)], ["UP", "ACT"])
    res = fit("vanish", log)["vanish"]
    assert res and isinstance(res[0][0], Vanish) and res[0][0].trigger == "ACT" and res[0][0].cls.color == 3


def test_predictor_from_tracker_grid():
    g = np.zeros((16, 16), dtype=np.int16)
    g[0, :] = 5
    g[15, :] = 5
    g[:, 0] = 5
    g[:, 15] = 5
    g[3:5, 3:5] = 1  # avatar 2x2
    g[3:5, 10:12] = 4  # target
    tr = Tracker()
    tr.reset(g)
    frames = [tuple(dsl.ent_from_tracker(e) for e in tr.current)]
    actions = []
    grids = [g.copy()]
    for _ in range(2):  # two steps: the avatar stops short of the target
        g2 = g.copy()
        ys, xs = np.nonzero(g == 1)
        g2[ys, xs] = 0
        g2[ys, xs + 2] = 1
        tr.update(g2, "RIGHT")
        frames.append(tuple(dsl.ent_from_tracker(e) for e in tr.current))
        actions.append("RIGHT")
        grids.append(g2.copy())
        g = g2
    log = make_log(frames, actions)
    rules, rep = auto_rules(log)
    assert rep["fully_explained"] == 2
    shapes = {dsl.ent_from_tracker(e).shape: e.mask for e in tr.current}
    pred = dsl.predictor(rules, shapes, 0)
    out = pred(grids[0], "RIGHT")
    assert np.array_equal(out, grids[1])
    out2 = pred(grids[1], "LEFT")
    assert np.array_equal(out2, grids[0])


def test_goal_predicates_from_won_level():
    W = walls()
    av = E(1, 1, 2, 2)
    key = E(2, 4, 4, 2)
    fr = [tuple(W + [av, key]), tuple(W + [E(1, 1, 3, 2), key]), tuple(W + [E(1, 1, 4, 2)])]
    goals = goal_predicates([(fr, True)])
    names = {gl["goal"] for gl in goals}
    assert "none_left(colour 4)" in names
    assert "none_left(colour 1)" not in names


def test_simulate_order_push_then_overlap():
    av = E(1, 1, 2, 2)
    box = E(2, 3, 3, 2)
    key = E(3, 4, 4, 2)
    mv = Move(Cls(color=1), {"RIGHT": (1, 0)}, frozenset({5}))
    push = Push(mv, Cls(color=3), None)
    ov = OnOverlap(Cls(color=3), Cls(color=4), "target_vanishes")
    after = simulate((av, box, key), "RIGHT", [push, mv, ov])
    ids = {e.id: e for e in after}
    assert ids[1].x0 == 3 and ids[2].x0 == 4 and 3 not in ids


def test_goal_kinds_distances_progress_and_falsification():
    """road-to-100 item 4: hypotheses are ranked by a cheap distance and a hypothesis that comes true on an unfinished
    level is falsified (satisfied without a win = not the goal)."""
    from arc3.dsl import goal_distance, goal_kind, goal_progress, render_goal_progress
    assert goal_kind("same_box(colour 4, colour 11)") == ("same_box", (4, 11))
    assert goal_kind("count(colour 3) == 2") == ("count", (3, 2))
    assert goal_kind("all_same_colour_as(7)") == ("all_same_colour_as", (7,))
    assert goal_kind({"none_left": 5}) == ("none_left", (5,)) and goal_kind({"reach_entity": 7}) == ("reach_entity", (7,))
    assert goal_kind({"touch": (1, 4)}) == ("touch", (1, 4)) and goal_kind("nonsense") is None and goal_kind({"bogus": 1}) is None
    av, tgt = E(1, 1, 2, 2), E(2, 4, 8, 2)
    f0, f1, f2 = (av, tgt), (E(1, 1, 5, 2), tgt), (E(1, 1, 8, 2), tgt)
    assert goal_distance("reach_entity", (2,), f0, avatar_id=1) == 6 and goal_distance("reach_entity", (2,), f2, avatar_id=1) == 0
    assert goal_distance("reach_entity", (2,), f0) is None  # no avatar known
    assert goal_distance("same_box", (1, 4), f0) == 12 and goal_distance("same_box", (1, 4), f2) == 0
    assert goal_distance("touch", (1, 4), f1) == 2 and goal_distance("overlap", (1, 4), f1) == 3
    assert goal_distance("none_left", (4,), f0) == 1 and goal_distance("count", (1, 3), f0) == 2
    assert goal_distance("same_columns", (1, 4), f2) == 1  # same columns but overlapping vertically: not the relation
    assert goal_distance("inside", (1, 4), f0) is None  # no smaller entity of colour 1 inside a colour-4 box
    assert goal_distance("aligned", (1,), f0) is None and goal_distance("aligned", (1,), (av, E(3, 1, 9, 2))) == 0
    rows = goal_progress(["same_box(colour 1, colour 4)", {"none_left": 4}, "touch(colour 1, colour 4)"], [f0, f1, f2], avatar_id=1)
    by = {r["goal"]: r for r in rows}
    assert by["same_box(colour 1, colour 4)"]["falsified"] and by["same_box(colour 1, colour 4)"]["falsified_at"] == 2
    assert by["touch(colour 1, colour 4)"]["falsified"] and not by["none_left(4)"]["falsified"]
    assert rows[0]["goal"] == "none_left(4)" and rows[0]["dist"] == 1  # live first
    assert by["same_box(colour 1, colour 4)"]["delta"] == -12
    text = render_goal_progress(rows)
    assert "1 live, 2 falsified" in text and "none_left(4) dist 1" in text and "Falsified this level" in text
    # incremental use: a caller keeps the falsified dict and passes only new frames as history
    known: dict = {}
    r1 = goal_progress(["same_box(colour 1, colour 4)"], [f0, f1], avatar_id=1, history=[f0, f1], falsified=known)
    assert not r1[0]["falsified"] and known == {}
    r2 = goal_progress(["same_box(colour 1, colour 4)"], [f0, f1, f2], avatar_id=1, history=[f2], falsified=known, history_offset=2)
    assert r2[0]["falsified"] and r2[0]["falsified_at"] == 2 and known == {"same_box(colour 1, colour 4)": 2}
    assert goal_progress([], [f0]) == [] and goal_progress(["none_left(colour 4)"], []) == []
