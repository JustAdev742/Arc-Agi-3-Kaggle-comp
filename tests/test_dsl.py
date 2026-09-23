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


def test_shape_matches_goal_kind():
    from arc3.dsl import goal_distance, goal_kind, goal_predicates, same_mask
    a, b = E(1, 1, 2, 2, w=2, h=3), E(2, 4, 10, 10, w=2, h=3)  # equal boxes and sizes, unknown masks: match
    c = E(3, 4, 20, 20, w=3, h=3)
    assert same_mask(a, b) and not same_mask(a, c)
    assert goal_kind("shape_matches(colour 1, colour 4)") == ("shape_matches", (1, 4))
    assert goal_distance("shape_matches", (1, 4), (a, c)) == 1 + 0 + 3 and goal_distance("shape_matches", (1, 4), (a, b)) == 0
    fr = [(a, c), (E(1, 1, 3, 2, w=2, h=3), c), (E(1, 1, 3, 2, w=3, h=3), c)]  # the colour-1 shape grows into the reference shape
    names = {g["goal"] for g in goal_predicates([(fr, True)])}
    assert "shape_matches(colour 1, colour 4)" in names and "shape_matches(colour 4, colour 1)" in names


def test_vanish_shape_goal_kind_survives_other_entities_of_the_colour():
    from arc3.dsl import goal_distance, goal_kind, goal_predicates
    ring = E(5, 3, 20, 20, w=9, h=9, shape="ring9")
    wall = E(6, 3, 0, 0, w=64, h=2, shape="wall")
    av = E(1, 5, 2, 2, w=7, h=7, shape="av")
    fr = [(ring, wall, av), (ring, wall, E(1, 5, 10, 10, w=7, h=7, shape="av")), (wall, E(1, 5, 20, 20, w=7, h=7, shape="av"))]
    goals = goal_predicates([(fr, True)])
    by = {g["goal"]: g for g in goals}
    name = "vanish(colour 3, shape ring9)"
    assert name in by and by[name]["kind"] == "vanish_shape" and by[name]["args"] == (3, "ring9")
    assert "none_left(colour 3)" not in by  # the wall stays, so colour-only vanishing is wrong here
    assert goal_kind({"vanish_shape": (3, "ring9")}) == ("vanish_shape", (3, "ring9"))
    assert goal_distance("vanish_shape", (3, "ring9"), fr[0]) == 1 and goal_distance("vanish_shape", (3, "ring9"), fr[-1]) == 0


def test_avatar_relative_goal_kinds_need_the_level_avatar():
    from arc3.dsl import goal_distance, goal_predicate, goal_predicates
    sock = E(9, 5, 20, 20, w=7, h=7, shape="sock")
    hud_block = E(7, 5, 0, 50, w=10, h=10, shape="hud")
    hud_piece = E(8, 9, 2, 52, w=3, h=3, shape="piece")  # a legend: colour 9 inside colour 5 from the start
    key = lambda x, y: E(1, 9, x, y, w=5, h=5, shape="key")  # noqa: E731
    fr = [(sock, hud_block, hud_piece, key(2, 2)), (sock, hud_block, hud_piece, key(10, 10)), (sock, hud_block, hud_piece, key(21, 21))]
    plain = {g["goal"] for g in goal_predicates([(fr, True)])}
    assert "inside(colour 9, colour 5)" not in plain  # true from the start because of the legend
    goals = goal_predicates([(fr, True)], avatar_ids=[1])
    by = {g["goal"]: g for g in goals}
    assert "avatar_inside(colour 5)" in by and by["avatar_inside(colour 5)"]["predicate"] is None
    assert "avatar_touch(colour 5)" in by
    p = goal_predicate("avatar_inside", (5,), avatar_id=1)
    assert p(fr[-1]) and not p(fr[0]) and goal_predicate("avatar_inside", (5,)) is None
    assert goal_distance("avatar_inside", (5,), fr[0], avatar_id=1) == 18 + 18 and goal_distance("avatar_inside", (5,), fr[-1], avatar_id=1) == 0
    assert goal_distance("avatar_touch", (5,), fr[0], avatar_id=1) == 13 and goal_distance("avatar_touch", (5,), fr[-1], avatar_id=1) == 0  # gap 14 - 1
    assert goal_predicates([(fr, True)], avatar_ids=[None]) == goal_predicates([(fr, True)])  # unknown avatar: no avatar kinds


def test_goal_candidates_dual_adds_raw_only_predicates():
    from arc3.dsl import goal_candidates_dual, goal_progress_dual
    # compound view: the socket got absorbed at the terminal; raw view: key (avatar 1) ends inside the colour-5 socket
    sock = E(9, 5, 20, 20, w=7, h=7, shape="sock")
    key = lambda x, y: E(1, 9, x, y, w=5, h=5, shape="key")  # noqa: E731
    comp = [(sock, key(2, 2)), (sock, key(10, 10)), (E(1, 9, 20, 20, w=7, h=7, shape="merged"),)]
    raw = [(sock, key(2, 2)), (sock, key(10, 10)), (sock, key(21, 21))]
    goals = goal_candidates_dual([(comp, True)], [(raw, True)], avatar_ids=[1])
    by = {g["goal"]: g for g in goals}
    assert "avatar_inside(colour 5)" in by and by["avatar_inside(colour 5)"]["rep"] == "raw"
    assert all(g["rep"] in ("compound", "raw") for g in goals)
    # a missing raw level (simulated terminal) disables the raw candidates
    assert all(g["rep"] == "compound" for g in goal_candidates_dual([(comp, True)], [None], avatar_ids=[1]))
    rows = goal_progress_dual(goals, comp[:2], raw[:2], avatar_id=1)
    r = next(r for r in rows if r["goal"] == "avatar_inside(colour 5)")
    assert r["dist"] == (20 - 10) * 2 and not r["falsified"]


def test_forall_goals_need_every_target_and_have_a_distance():
    # Two colour-4 slot outlines, two colour-9 pieces (census 2026-09-23: multi-target games need every target).
    slots = [Ent(1, 4, 10, 10, 6, 6, 20, None), Ent(2, 4, 30, 10, 6, 6, 20, None)]

    def pieces(p, q):
        return [Ent(3, 9, p[0], p[1], 2, 2, 4, None), Ent(4, 9, q[0], q[1], 2, 2, 4, None)]

    start = tuple(slots + pieces((12, 30), (32, 30)))
    partial = tuple(slots + pieces((12, 12), (32, 30)))  # one piece in its slot: "some" holds, "every" does not
    won = tuple(slots + pieces((12, 12), (32, 12)))
    names = {g["goal"] for g in goal_predicates([([start, partial, won], True)], forall=True)}
    assert "every_inside(colour 9, colour 4)" in names
    assert "inside(colour 9, colour 4)" not in names  # the partial state falsified the existential form
    assert not any(n.startswith("every_") for n in (g["goal"] for g in goal_predicates([([start, partial, won], True)])))
    d = [dsl.goal_distance("every_inside", (9, 4), f) for f in (start, partial, won)]
    assert d[0] > d[1] > d[2] == 0
    assert dsl.goal_kind("every_inside(colour 9, colour 4)") == ("every_inside", (9, 4))
    assert dsl.forall_distance("in", 9, 4, won) == 0 and dsl.forall_distance("in", 9, 4, start) > 0
    assert dsl.forall_distance("inside", 9, 7, won) is None  # no colour-7 targets: not vacuously true


def test_lifted_goals_transfer_a_kind_across_colours():
    # vc33 (exp-028): the goal is "aligned" on every level, on colour 11 at level 1 and colour 14 at level 2.
    assert dsl.goal_signature("aligned", (11,)) == ("aligned", False)
    assert dsl.goal_signature("same_columns", (11, 11)) == ("same_columns", True)
    assert dsl.goal_signature("every_inside", (9, 4)) == ("every_inside", False)
    assert dsl.goal_signature("count", (3, 2)) is None
    level2 = (Ent(1, 14, 5, 5, 2, 2, 4, None), Ent(2, 14, 9, 20, 2, 2, 4, None), Ent(3, 7, 30, 30, 4, 4, 16, None))
    goals = dsl.instantiate_goals([("aligned", False), ("same_columns", True)], level2)
    names = {g["goal"] for g in goals}
    assert "aligned(colour 14)" in names  # the colour-14 pair is not aligned at the start: a live candidate
    assert "aligned(colour 7)" not in names  # a single entity is never aligned: not computable as a goal here
    won = (Ent(1, 14, 5, 5, 2, 2, 4, None), Ent(2, 14, 5, 20, 2, 2, 4, None), Ent(3, 7, 30, 30, 4, 4, 16, None))
    assert any(g["predicate"](won) for g in goals if g["goal"] == "aligned(colour 14)")
    assert all(not g["predicate"](level2) for g in goals)  # nothing already satisfied at the level start
