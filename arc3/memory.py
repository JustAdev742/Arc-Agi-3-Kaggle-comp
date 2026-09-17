"""Learning memory for the REPL agent: "what did we learn?" lessons.

Three layers, cheapest first:

* **Lessons (this game).** Short natural-language facts the harness writes on its own when something
  decisive happens (a level is completed: the recipe; a game over: the hazard; a world model is retired
  or a batch is stopped: the mistake) plus what the model records itself with ``learn(text)``. They are
  shown every turn and survive level changes, so a mistake made on level 2 is still in front of the
  model on level 5. This is the Reflexion idea (verbal reflection kept in an episodic buffer) with the
  reflection triggered by evidence the harness already has, not by an extra model call.
* **From other games (this run).** Every game appends its shareable lessons (mechanics, hazards,
  recipes) to one run-wide JSONL file under a file lock; each game reads the others' most recent ones.
  On Kaggle all hidden games play concurrently, so this is live cross-game transfer (ExpeL-style
  insights), framed as "may not apply" because games differ.
* **Skill library (offline).** ``scripts/mine_skills.py`` distils solved levels from past run transcripts
  into ``arc3/data/skills.json``; :func:`load_skills` / :func:`match_skills` retrieve the strategies whose
  signature matches the current level (an avatar with a key map, click-only, ...) for the prompt.

Everything here is plain text for the prompt; nothing is executed. The store deduplicates by
normalised text, caps its size and never raises into the agent (a memory failure must not cost a game).
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

try:  # POSIX only; Kaggle and the dev box are Linux, tests run there too
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

log = logging.getLogger("arc3.memory")

KINDS = ("recipe", "hazard", "mistake", "mechanic", "goal", "strategy")
SHARED_KINDS = ("mechanic", "hazard", "recipe", "strategy")  # what other games may benefit from
_RANK = {k: i for i, k in enumerate(KINDS)}  # render order: recipes and hazards first
DEFAULT_SKILLS = Path(__file__).with_name("data") / "skills.json"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


class Lessons:
    """Per-game lesson store with an optional run-wide shared file.

    ``add`` returns True when the lesson is new for this game. ``render`` gives the prompt block for
    this game, ``render_others`` the block of lessons other games in the run shared. ``save`` writes
    ``<out_dir>/<game>.lessons.json`` for the post-mortems and the skill miner.
    """

    def __init__(self, game_id: str, *, out_dir: Optional[str] = None, shared_path: Optional[str] = None,
                 cap: int = 40, enabled: bool = True, share: bool = True, others_refresh_s: float = 20.0):
        self.game_id = str(game_id)
        self.out_dir = Path(out_dir) if out_dir else None
        env_path = os.environ.get("ARC3_MEMORY_PATH")
        sp = shared_path or env_path or (str(self.out_dir / "shared_lessons.jsonl") if self.out_dir else None)
        self.shared_path = Path(sp) if (sp and share) else None
        self.cap = int(cap)
        self.enabled = bool(enabled)
        self.items: list[dict[str, Any]] = []
        self.model_lessons = 0
        self.auto_lessons = 0
        self._lock = threading.Lock()
        self._others: list[dict[str, Any]] = []
        self._others_read_at = 0.0
        self._others_refresh_s = float(others_refresh_s)
        self._shared_offset = 0
        self._shared_warned = False  # a broken shared file is reported once, then ignored (it must not cost a game)

    # ----------------------------------------------------------------------------------- this game
    def add(self, kind: str, text: str, *, level: Optional[int] = None, source: str = "auto",
            evidence: Optional[str] = None, share: Optional[bool] = None) -> bool:
        if not self.enabled:
            return False
        kind = kind if kind in KINDS else "mistake"
        text = re.sub(r"\s+", " ", str(text)).strip()
        if not text:
            return False
        text = text[:400]
        key = _norm(text)
        with self._lock:
            for it in self.items:
                if it["key"] == key:
                    it["n"] += 1
                    it["t"] = time.time()
                    if level is not None:
                        it["level"] = int(level)
                    return False
            item = {"kind": kind, "text": text, "level": (int(level) if level is not None else None), "source": source,
                    "evidence": (str(evidence)[:300] if evidence else None), "t": time.time(), "n": 1, "key": key}
            self.items.append(item)
            if source == "model":
                self.model_lessons += 1
            else:
                self.auto_lessons += 1
            if len(self.items) > self.cap:
                # Drop the oldest of the least valuable kind first (mechanics and goals are re-derived cheaply).
                victim = max(range(len(self.items)), key=lambda i: (_RANK[self.items[i]["kind"]], -self.items[i]["t"]))
                del self.items[victim]
        if share if share is not None else kind in SHARED_KINDS:
            self._share(item)
        return True

    def add_many(self, lessons: Iterable[Any], *, level: Optional[int] = None, source: str = "model") -> int:
        """Lessons as the sandbox reports them: dicts {kind, text} or plain strings."""
        n = 0
        for entry in lessons or []:
            if isinstance(entry, dict):
                kind, text = str(entry.get("kind", "mistake")), str(entry.get("text", ""))
            else:
                kind, text = "mistake", str(entry)
            try:
                n += int(self.add(kind, text, level=level, source=source))
            except Exception:
                log.debug("skipping malformed lesson %r", entry, exc_info=True)
        return n

    def render(self, limit: int = 14) -> str:
        if not self.items:
            return ""
        with self._lock:
            items = sorted(self.items, key=lambda it: (_RANK[it["kind"]], -it["t"]))[:limit]
        lines = []
        for it in items:
            tag = it["kind"] + (f" L{it['level']}" if it.get("level") is not None else "")
            rep = f" (x{it['n']})" if it.get("n", 1) > 1 else ""
            lines.append(f"- [{tag}] {it['text']}{rep}")
        return "Lessons (this game; they carry over between levels):\n" + "\n".join(lines)

    def to_list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{k: v for k, v in it.items() if k != "key"} for it in self.items]

    def save(self) -> Optional[str]:
        if not self.out_dir or not self.enabled:
            return None
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            p = self.out_dir / f"{self.game_id}.lessons.json"
            p.write_text(json.dumps({"game": self.game_id, "lessons": self.to_list(),
                                     "model_lessons": self.model_lessons, "auto_lessons": self.auto_lessons}, indent=1))
            return str(p)
        except OSError as e:
            log.warning("%s: could not save lessons: %s", self.game_id, e)
            return None

    # --------------------------------------------------------------------------------- shared file
    def _shared_failed(self, what: str, e: Exception) -> None:
        if not self._shared_warned:
            self._shared_warned = True
            log.warning("%s: shared lessons file %s unusable (%s: %s); continuing without cross-game memory",
                        self.game_id, self.shared_path, what, e)

    def _share(self, item: dict[str, Any]) -> None:
        if not self.shared_path:
            return
        rec = {"game": self.game_id, "kind": item["kind"], "text": item["text"], "level": item.get("level"),
               "source": item.get("source"), "t": item["t"]}
        try:
            self.shared_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.shared_path, "a") as f:
                if fcntl is not None:
                    fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                finally:
                    if fcntl is not None:
                        fcntl.flock(f, fcntl.LOCK_UN)
        except OSError as e:
            self._shared_failed("write", e)

    def others(self, limit: int = 8, *, force: bool = False) -> list[dict[str, Any]]:
        """Most recent lessons other games shared (one per text; newest first), read at most every few seconds."""
        if not self.shared_path or not self.enabled:
            return []
        now = time.time()
        if force or now - self._others_read_at >= self._others_refresh_s:
            self._others_read_at = now
            try:
                if self.shared_path.exists():
                    with open(self.shared_path) as f:
                        if fcntl is not None:
                            fcntl.flock(f, fcntl.LOCK_SH)
                        try:
                            f.seek(self._shared_offset)
                            chunk = f.read()
                            self._shared_offset = f.tell()
                        finally:
                            if fcntl is not None:
                                fcntl.flock(f, fcntl.LOCK_UN)
                    for line in chunk.splitlines():
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue  # a line another game is still writing; it is complete on the next read
                        if not isinstance(rec, dict) or rec.get("game") == self.game_id or not rec.get("text"):
                            continue
                        self._others.append(rec)
                    del self._others[:-200]
            except OSError as e:
                self._shared_failed("read", e)
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for rec in reversed(self._others):
            k = _norm(rec["text"])
            if k in seen:
                continue
            seen.add(k)
            out.append(rec)
            if len(out) >= limit:
                break
        return out

    def render_others(self, limit: int = 6) -> str:
        recs = self.others(limit)
        if not recs:
            return ""
        lines = [f"- [{r.get('game', '?')} {r.get('kind', '')}] {r['text']}" for r in recs]
        return ("From other games in this run (different games, so treat as hints to test, not facts):\n"
                + "\n".join(lines))


# ------------------------------------------------------------------------------------ skill library
def load_skills(path: Optional[str] = None) -> list[dict[str, Any]]:
    p = Path(path) if path else DEFAULT_SKILLS
    try:
        if not p.exists():
            return []
        data = json.loads(p.read_text())
        return list(data.get("skills", data) if isinstance(data, dict) else data)
    except Exception:  # noqa: BLE001
        return []


def level_signature(*, has_avatar: bool, click_only: bool, keymap: Optional[dict] = None, n_entities: int = 0,
                    tile: int = 1, hud: bool = False) -> dict[str, Any]:
    """The coarse description a skill is matched on. Every field is computed by code from the tracker."""
    return {"avatar": bool(has_avatar), "click_only": bool(click_only), "keys": sorted((keymap or {}).keys()),
            "many_entities": int(n_entities) >= 12, "tile": int(tile), "hud": bool(hud)}


def match_skills(skills: list[dict[str, Any]], sig: dict[str, Any], *, limit: int = 3,
                 exclude_game: Optional[str] = None) -> list[dict[str, Any]]:
    """Rank skills by signature agreement; ties by how many times the strategy won and how few actions it took."""
    rank = {"validated": 2, "candidate": 1}
    scored = []
    for s in skills:
        if exclude_game and s.get("game") == exclude_game:
            continue
        if s.get("status") == "deprecated":
            continue  # a one-off that later runs did not reproduce is not a hint
        ss = s.get("signature") or {}
        score = 0
        for k in ("avatar", "click_only", "many_entities", "hud"):
            if k in ss and k in sig:
                score += 2 if ss[k] == sig[k] else -2
        if ss.get("keys") and sig.get("keys"):
            score += 1 if set(ss["keys"]) & set(sig["keys"]) else 0
        if score <= 0:
            continue
        scored.append((score, rank.get(s.get("status", "candidate"), 1), int(s.get("wins", 1)), -float(s.get("actions", 1e9)), s))
    scored.sort(key=lambda t: (t[0], t[1], t[2], t[3]), reverse=True)
    return [s for _, _, _, _, s in scored[:limit]]


def render_skills(skills: list[dict[str, Any]]) -> str:
    if not skills:
        return ""
    lines = []
    for s in skills:
        txt = str(s.get("strategy", "")).strip()
        if not txt:
            continue
        tag = ""
        if s.get("wins"):
            tag = " (" + (f"{s['status']}: " if s.get("status") else "") + f"won {s['wins']}x"
            if s.get("failures"):
                tag += f", failed {s['failures']}x"
            tag += f", ~{s['actions']} actions)"
        lines.append(f"- {txt}{tag}")
    if not lines:
        return ""
    return "Strategies that solved levels of similar shape in earlier games (hints, not rules):\n" + "\n".join(lines)
