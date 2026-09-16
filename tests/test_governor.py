from arc3.governor import Governor


def test_even_split_and_reserve():
    g = Governor(total_budget_s=3600, n_games=6, reserve_s=600, min_per_game_s=10, start_time=1000.0)
    assert g.remaining_s(now=1000.0) == 3000
    assert g.slice_for_next_game(now=1000.0) == 500


def test_unused_time_flows_to_later_games():
    g = Governor(total_budget_s=1000, n_games=2, reserve_s=0, min_per_game_s=1, start_time=0.0)
    d1 = g.start_game("a", now=0.0)
    assert d1 == 500
    g.finish_game("a")  # finished instantly at t=10
    d2 = g.start_game("b", now=10.0)
    assert d2 == 1000  # all remaining time goes to the last game


def test_never_exceeds_hard_deadline():
    g = Governor(total_budget_s=100, n_games=1, reserve_s=0, min_per_game_s=1000, start_time=0.0)
    assert g.slice_for_next_game(now=0.0) == 100
    assert g.slice_for_next_game(now=150.0) == 0
    assert g.out_of_time(now=150.0)


def test_parallel_waves():
    g = Governor(total_budget_s=1000, n_games=8, reserve_s=0, min_per_game_s=1, parallel=4, start_time=0.0)
    assert g.slice_for_next_game(now=0.0) == 500  # two waves of four


def test_max_per_game_cap():
    g = Governor(total_budget_s=10000, n_games=1, reserve_s=0, max_per_game_s=900, start_time=0.0)
    assert g.slice_for_next_game(now=0.0) == 900
