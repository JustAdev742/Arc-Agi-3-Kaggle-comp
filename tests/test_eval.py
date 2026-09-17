"""Run records: summary.json is written after every game (partial), atomically, and a dead worker does not sink the run."""
import json
from pathlib import Path

import pytest

from arc3 import eval as ev

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "environment_files" / "ls20").exists(), reason="game files not downloaded")


def _meta(games):
    return {"run_name": "t", "agent": "random", "split": ",".join(games), "games": games, "seed": 0, "time_budget_s": 5,
            "max_actions": 5, "config": {}, "note": "", "harness_commit": "x", "arc3_version": "0", "python": "3",
            "platform": "p", "started": "now"}


def _result(game, score=0.0, levels=0, win_levels=3, state="NOT_FINISHED", failure="timeout"):
    return {"game_id": game, "score": score, "levels_completed": levels, "win_levels": win_levels, "actions": 3,
            "state": state, "failure": failure, "level_actions": [], "baselines": [], "wall_s": 1.0}


def test_summarize_partial_scores_only_finished_games(tmp_path):
    meta = _meta(["ls20", "vc33"])
    s = ev.write_summary(tmp_path, meta, [_result("vc33", score=50.0, levels=1)], wall_s=2.0, partial=True)
    on_disk = json.loads((tmp_path / "summary.json").read_text())
    assert on_disk == s and s["partial"] is True and s["games_finished"] == 1
    assert s["score"] == 50.0 and s["score_dev"] == 50.0 and s["score_val"] is None
    assert not (tmp_path / "summary.json.tmp").exists()  # atomic replace leaves no temp file behind
    # Results come back in split order whatever order the games finished in.
    s2 = ev.write_summary(tmp_path, meta, [_result("vc33", score=50.0), _result("ls20")], wall_s=3.0, partial=False)
    assert [r["game_id"] for r in s2["results"]] == ["ls20", "vc33"] and s2["partial"] is False
    assert s2["score"] == 25.0 and s2["failures"] == {"timeout": 2}
    assert "PARTIAL" in ev.format_summary(s) and "PARTIAL" not in ev.format_summary(s2)


def test_run_eval_writes_partial_then_final_summary(tmp_path, monkeypatch):
    seen: list[bool] = []
    real = ev.write_summary

    def spy(out_dir, meta, results, *, wall_s, partial):
        seen.append(partial)
        return real(out_dir, meta, results, wall_s=wall_s, partial=partial)

    monkeypatch.setattr(ev, "write_summary", spy)
    s = ev.run_eval("random", "smoke", time_budget_s=20, max_actions=5, workers=1, run_name="t", runs_dir=tmp_path,
                    environments_dir=str(ROOT / "environment_files"))
    assert seen == [True, True, False]
    on_disk = json.loads((tmp_path / "t" / "summary.json").read_text())
    assert on_disk["partial"] is False and on_disk["games_finished"] == 2
    assert {r["game_id"] for r in on_disk["results"]} == {"ls20", "vc33"} and s["actions"] == 10
    assert (tmp_path / "t" / "ls20.jsonl").exists()


def _flaky_worker(args):
    """Module-level so the process pool can pickle it by reference; ls20's worker dies outside play_game."""
    if args["game_id"] == "ls20":
        raise RuntimeError("worker process lost")
    return _REAL_WORKER(args)


_REAL_WORKER = ev._worker


def test_dead_worker_is_recorded_as_a_crash_and_the_run_continues(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "_worker", _flaky_worker)
    s = ev.run_eval("random", "smoke", time_budget_s=20, max_actions=5, workers=2, run_name="t", runs_dir=tmp_path,
                    environments_dir=str(ROOT / "environment_files"))
    by = {r["game_id"]: r for r in s["results"]}
    assert by["ls20"]["failure"] == "crash" and "worker process lost" in by["ls20"]["error"]
    assert by["vc33"]["actions"] == 5 and s["failures"]["crash"] == 1 and s["partial"] is False
