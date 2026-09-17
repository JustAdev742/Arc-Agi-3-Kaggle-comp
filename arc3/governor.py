"""Wall-clock governor: divides the remaining budget across the games still to play.

Status (lesson 0011, 2026-09-16): NOT on the submission path. The Kaggle framework plays every game
in its own thread at the same time, so the submission uses one shared deadline for all games
(``arc3.kaggle.global_deadline``) and the local harness a fixed per-game budget (``arc3.eval``).
This module is kept, tested, for a sequential runner that plays games one after another.

The whole hidden set must finish inside the competition runtime limit and unfinished
levels score zero, so seconds per action is a first-class metric. The governor is
deliberately simple and testable:

  * ``total_budget_s`` is the full budget minus a fixed safety reserve;
  * each game gets ``remaining_budget / remaining_games`` at the moment it starts,
    clipped to ``[min_per_game, max_per_game]``;
  * games that finish early return their unused time to the pool;
  * with parallel games (the Kaggle framework runs one thread per game), pass
    ``parallel=n`` so the per-game slice reflects that ``n`` games share wall-clock.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Governor:
    total_budget_s: float
    n_games: int
    reserve_s: float = 300.0
    min_per_game_s: float = 60.0
    max_per_game_s: Optional[float] = None
    parallel: int = 1
    start_time: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _started: int = 0
    _finished: int = 0
    _deadlines: dict[str, float] = field(default_factory=dict)

    @property
    def hard_deadline(self) -> float:
        return self.start_time + max(0.0, self.total_budget_s - self.reserve_s)

    def remaining_s(self, now: Optional[float] = None) -> float:
        now = time.time() if now is None else now
        return max(0.0, self.hard_deadline - now)

    def games_left(self) -> int:
        return max(1, self.n_games - self._finished)

    def slice_for_next_game(self, now: Optional[float] = None) -> float:
        """Seconds the next game may use. Thread-safe."""
        now = time.time() if now is None else now
        with self._lock:
            left = self.games_left()
            # With `parallel` games in flight, each wave of games shares the wall-clock.
            waves = max(1.0, left / max(1, self.parallel))
            s = self.remaining_s(now) / waves
            s = max(self.min_per_game_s, s)
            if self.max_per_game_s is not None:
                s = min(self.max_per_game_s, s)
            # Never promise more than what is left before the hard deadline.
            return max(0.0, min(s, self.remaining_s(now)))

    def start_game(self, game_id: str, now: Optional[float] = None) -> float:
        """Register a game start and return its absolute deadline."""
        now = time.time() if now is None else now
        s = self.slice_for_next_game(now)
        with self._lock:
            self._started += 1
            self._deadlines[game_id] = now + s
        return now + s

    def finish_game(self, game_id: str) -> None:
        with self._lock:
            self._finished += 1
            self._deadlines.pop(game_id, None)

    def deadline_for(self, game_id: str) -> Optional[float]:
        return self._deadlines.get(game_id)

    def out_of_time(self, now: Optional[float] = None) -> bool:
        return self.remaining_s(now) <= 0.0

    def status(self) -> dict[str, float | int]:
        return {
            "elapsed_s": round(time.time() - self.start_time, 1),
            "remaining_s": round(self.remaining_s(), 1),
            "started": self._started,
            "finished": self._finished,
            "n_games": self.n_games,
        }
