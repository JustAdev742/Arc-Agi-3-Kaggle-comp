#!/usr/bin/env python
"""Our changes to the Duck harness (TAAF + ARC3-Inference, MIT) as exact-text patches on the published anim bundle.

    python scripts/taaf_ours_patch.py <anim bundle dir> <out dir>     # copy the bundle and patch the copy

The base is jakobbrggen/taaf-kaggle-source-anim-20260807-anim (CC0 bundle of Tufa Labs' MIT code). Every patch is an
(old, new) pair that must match exactly once, so a changed upstream fails loudly instead of half-applying. The
submission notebook inlines this file, copies the mounted bundle to /tmp and applies ``apply()`` before importing the
solver, so no dataset of ours is needed. Each patch is listed in docs/research_log.md with its experiment.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

TOOL_AGENT = "src/ARC3-Inference/inference/agent/tool_agent.py"

# P1 (exp-034): the carried world-model note is also read from the hidden reasoning when the visible reply has none.
# Qwen models put the `World model:` / `Plan:` lines in the thinking block and send the tool call with empty content
# (67% of tool-call turns in one public Qwen run, discussion 734843), so the note the harness carries between turns
# rarely advanced.
P1_NOTE_FN = '''_NOTE_LABELS = (
    "World model",
    "Goal model",
    "Action model",
    "Recent findings",
    "Open questions",
    "Plan",
    "Cross-level notes",
    "Hypothesis",
    "History check",
    "Next test",
)
_REASONING_NOTE_MAX_LINES = 8
_REASONING_NOTE_MAX_CHARS = 600
_REASONING_LABEL_RE = re.compile(
    r"^(world model|goal model|action model|recent findings|open questions|plan|cross-level notes|hypothesis"
    r"|history check|next test)(?:\\s+[^:\\n]{0,40})?:\\s*(.*)$",
    re.IGNORECASE,
)


def _extract_scientist_note_from_reasoning(reasoning: str) -> dict[str, str]:
    """The carried note, read from hidden reasoning when the visible reply had none.

    Qwen models often write their `World model:` / `Plan:` lines only inside the
    thinking block and send the tool call with empty visible content, so the
    note never advanced. Reasoning is long and exploratory, so only the LAST
    block per label counts, a block ends at a blank line or the next label,
    and each block is capped.
    """
    if not reasoning or not reasoning.strip():
        return {}
    by_lower = {label.lower(): label for label in _NOTE_LABELS}
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in reasoning.splitlines():
        stripped = raw_line.strip()
        candidate = stripped
        while candidate.startswith(("-", "*", "#")):
            candidate = candidate[1:].lstrip()
        candidate = candidate.replace("**", "")
        found = _REASONING_LABEL_RE.match(candidate)
        matched = by_lower.get(found.group(1).lower()) if found is not None else None
        if matched is not None:
            current = matched
            inline = found.group(2).strip()
            blocks[current] = [inline] if inline else []
            continue
        if not stripped:
            current = None
            continue
        if current is not None and len(blocks[current]) < _REASONING_NOTE_MAX_LINES:
            blocks[current].append(stripped)
    extracted = {
        label: _normalize_summary_text("\\n".join(lines).strip(), max_chars=_REASONING_NOTE_MAX_CHARS)
        for label, lines in blocks.items()
        if "\\n".join(lines).strip()
    }
    if not extracted:
        return {}
    return {
        "world_model": extracted.get("World model", "") or extracted.get("Hypothesis", ""),
        "goal_model": extracted.get("Goal model", ""),
        "action_model": extracted.get("Action model", ""),
        "recent_findings": extracted.get("Recent findings", "") or extracted.get("History check", ""),
        "open_questions": extracted.get("Open questions", ""),
        "current_plan": extracted.get("Plan", "") or extracted.get("Next test", ""),
        "cross_level_notes": extracted.get("Cross-level notes", ""),
    }


'''

PATCHES: dict[str, list[tuple[str, str, str]]] = {
    "P1": [
        (TOOL_AGENT,
         "def _empty_world_model() -> dict[str, str]:",
         P1_NOTE_FN + "def _empty_world_model() -> dict[str, str]:"),
        (TOOL_AGENT,
         "        self.animation_counters: dict[str, int] = {}\n        self._reset_animation_hint_state()",
         "        self.animation_counters: dict[str, int] = {}\n        self.note_counters: dict[str, int] = {}\n"
         "        self._reset_animation_hint_state()"),
        (TOOL_AGENT,
         "    def _update_summarized_knowledge_from_assistant(self, content: str) -> None:\n"
         "        note = _extract_scientist_note(content)\n"
         "        if not note:\n",
         "    def _update_summarized_knowledge_from_assistant(self, content: str, reasoning: str = \"\") -> None:\n"
         "        note = _extract_scientist_note(content)\n"
         "        if not any(note.values()) and reasoning:\n"
         "            note = _extract_scientist_note_from_reasoning(reasoning)\n"
         "            if any(note.values()):\n"
         "                self.note_counters[\"note_from_reasoning\"] = self.note_counters.get(\"note_from_reasoning\", 0) + 1\n"
         "        elif any(note.values()):\n"
         "            self.note_counters[\"note_from_content\"] = self.note_counters.get(\"note_from_content\", 0) + 1\n"
         "        if not note:\n"),
        (TOOL_AGENT,
         "                if not tool_calls:\n"
         "                    if content:\n"
         "                        self._update_summarized_knowledge_from_assistant(content)\n"
         "                        append_transcript(\"ASSISTANT\", content)",
         "                if not tool_calls:\n"
         "                    if content or reasoning:\n"
         "                        self._update_summarized_knowledge_from_assistant(content or \"\", reasoning or \"\")\n"
         "                    if content:\n"
         "                        append_transcript(\"ASSISTANT\", content)"),
        (TOOL_AGENT,
         "                if content:\n"
         "                    self._update_summarized_knowledge_from_assistant(content)\n"
         "                    append_transcript(\"ASSISTANT\", content)\n"
         "                    assistant_message[\"content\"] = content\n"
         "                assistant_message[\"tool_calls\"] = tool_calls",
         "                if content or reasoning:\n"
         "                    self._update_summarized_knowledge_from_assistant(content or \"\", reasoning or \"\")\n"
         "                if content:\n"
         "                    append_transcript(\"ASSISTANT\", content)\n"
         "                    assistant_message[\"content\"] = content\n"
         "                assistant_message[\"tool_calls\"] = tool_calls"),
    ],
}


ACTION_NAMES = "src/ARC3-Inference/inference/agent/action_names.py"
SANDBOX = "src/ARC3-Inference/inference/agent/python_tool_sandbox.py"

# P1b: note labels with a qualifier ("World model update:", "World model (level 2):", "Plan for next probe:") were
# rejected by the exact `startswith("world model:")` match; 272 of 1,329 responses in the thui run carried such headers.
P1B_MATCH_OLD = """        for target in targets:
            if lowered.startswith(target):
                matched_label = normalized_labels[target[:-1]]
                inline_value = candidate[len(target):].strip()
                break
"""
P1B_MATCH_NEW = """        for target in targets:
            if lowered.startswith(target):
                matched_label = normalized_labels[target[:-1]]
                inline_value = candidate[len(target):].strip()
                break
        if matched_label is None:
            lenient = _LENIENT_LABEL_RE.match(candidate.replace("**", ""))
            if lenient is not None:
                matched_label = normalized_labels.get(lenient.group(1).lower())
                inline_value = lenient.group(2).strip() if matched_label is not None else ""
"""
P1B_RE = '''_LENIENT_LABEL_RE = re.compile(
    r"^(world model|goal model|action model|recent findings|open questions|plan|cross-level notes|hypothesis"
    r"|history check|next test)(?:\\s+[^:\\n]{0,40})?:\\s*(.*)$",
    re.IGNORECASE,
)


def _extract_labeled_blocks('''

# P2: the engine's ACTION7 (UNDO in the ARC-AGI-3 docs) had no model-facing name, so the model saw "ACTION7" among the
# valid actions and every attempt was rejected ("Unknown action"); in the thui run 15 attempts in 6 games, and in 26
# calls the model concluded ACTION7 does nothing.
P2_OLD = '    "ACTION6": "MOUSE",\n    "RESET": "RESET",\n'
P2_NEW = '    "ACTION6": "MOUSE",\n    "ACTION7": "UNDO",\n    "RESET": "RESET",\n'

# P3: a new level cleared the goal and action models (and every other note except cross-level notes, which the model
# wrote once in a whole run), so each level restarted from an empty note. A game's kind of win condition is constant
# across levels (lesson 0016) and action semantics rarely change, so the goal and action models are kept, marked for
# verification; a GAME_OVER restart of the same level keeps the whole note.
P3_OLD = """        if summary.get("level_transition") or summary.get("run_complete") or summary.get("game_over"):
            for key in (
                "world_model",
                "goal_model",
                "action_model",
                "recent_findings",
                "open_questions",
                "current_plan",
            ):
                self._summarized_knowledge[key] = ""
"""
P3_NEW = """        if summary.get("level_transition") or summary.get("run_complete"):
            for key in ("goal_model", "action_model"):
                value = self._summarized_knowledge.get(key, "")
                if value and not value.startswith("[from an earlier level"):
                    self._summarized_knowledge[key] = (
                        "[from an earlier level; verify on this level before relying on it] " + value
                    )
            for key in (
                "world_model",
                "recent_findings",
                "open_questions",
                "current_plan",
            ):
                self._summarized_knowledge[key] = ""
"""

# P7: pre-import the allowed modules and the usual collections names in every python call; 20 NameErrors in the thui
# run came from missing imports (json, Counter, ...) or earlier calls' names.
P7_OLD = '        runtime_globals["__builtins__"]["__import__"] = _safe_import\n'
P7_NEW = ('        runtime_globals["__builtins__"]["__import__"] = _safe_import\n'
          '        for _pre in ("json", "math", "collections", "itertools", "functools", "heapq", "re", "copy"):\n'
          '            try:\n'
          '                runtime_globals[_pre] = _safe_import(_pre)\n'
          '            except Exception:\n'
          '                pass\n'
          '        try:\n'
          '            from collections import Counter, defaultdict, deque\n'
          '            runtime_globals.update(Counter=Counter, defaultdict=defaultdict, deque=deque)\n'
          '        except Exception:\n'
          '            pass\n')

# P4: once a user message is older than the newest one, its standing instructions (about 2,400 characters repeated
# every turn), its copy of the carried note (superseded by the newest) and its board image carry nothing new; they are
# cut from the stored history. The newest user message is still sent in full with the current image.
P4_FN = '''_HISTORY_USER_CUT_MARKER = "\\nOnly tool: `python`."
_STRIP_PAST_REASONING = os.environ.get("OURS_STRIP_PAST_REASONING", "1") == "1"


def _compress_history_message(message: dict[str, Any]) -> dict[str, Any]:
    """An older turn as it is kept in history.

    User turns lose their standing instructions, stale note and image (the newest turn keeps all three). Assistant
    turns lose their reasoning: the served template does not render reasoning from before the latest user turn,
    so it only counted against the history budget.
    """
    role = str(message.get("role", "")).strip()
    if role == "assistant" and _STRIP_PAST_REASONING and message.get("reasoning"):
        stripped = dict(message)
        stripped.pop("reasoning", None)
        return stripped
    if role != "user":
        return message
    content = message.get("content")
    if isinstance(content, list):
        text = "\\n".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    elif isinstance(content, str):
        text = content
    else:
        return message
    cut = text.find(_HISTORY_USER_CUT_MARKER)
    if cut > 0:
        text = text[:cut] + "\\n[Earlier turn; its standing instructions, world-model copy and image are omitted.]"
    elif isinstance(content, str):
        return message
    compressed = dict(message)
    compressed["content"] = text
    return compressed


'''
P4_OLD = "        return self._drop_until_first_user_message(history)\n"
P4_NEW = ("        return [\n"
          "            _compress_history_message(message)\n"
          "            for message in self._drop_until_first_user_message(history)\n"
          "        ]\n")

# P8: the harness carries `Cross-level notes:` to every later level, but the model wrote that label once in a whole
# run, so each level was re-learned from scratch (a median of 3 calls, 8 minutes, before the first action on a new
# level). A game's kind of win condition and its action semantics usually persist (lesson 0016), so a completed level
# now asks for that note before acting.
P8_OLD = '                lines.append("You have progressed to a new level!")\n'
P8_NEW = ('                lines.append("You have progressed to a new level!")\n'
          '                lines.append(\n'
          '                    "Before acting on the new level, write one line starting with `Cross-level notes:` that "\n'
          '                    "states what completed the previous level (the goal as you now understand it) and what each "\n'
          '                    "action does; it is kept for every later level. Then check the new board against it."\n'
          '                )\n')

PATCHES.update({
    "P8": [(TOOL_AGENT, P8_OLD, P8_NEW)],
    "P4": [
        (TOOL_AGENT, "def _empty_world_model(", P4_FN + "def _empty_world_model("),
        (TOOL_AGENT, P4_OLD, P4_NEW),
    ],
    "P1B": [
        (TOOL_AGENT, "def _extract_labeled_blocks(", P1B_RE),
        (TOOL_AGENT, P1B_MATCH_OLD, P1B_MATCH_NEW),
    ],
    "P2": [(ACTION_NAMES, P2_OLD, P2_NEW)],
    "P3": [(TOOL_AGENT, P3_OLD, P3_NEW)],
    "P7": [(SANDBOX, P7_OLD, P7_NEW)],
})


def apply(bundle_dir: Path | str, names: list[str] | None = None) -> list[str]:
    """Apply the named patches (default: all) in place under ``bundle_dir``; return what was applied."""
    root = Path(bundle_dir)
    applied = []
    for name in names or list(PATCHES):
        for rel, old, new in PATCHES[name]:
            path = root / rel
            text = path.read_text(encoding="utf-8")
            count = text.count(old)
            if count != 1:
                raise RuntimeError(f"patch {name}: expected exactly one match in {rel}, found {count}")
            path.write_text(text.replace(old, new, 1), encoding="utf-8")
            applied.append(f"{name}:{rel}")
    return applied


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    names = sys.argv[3:] or None
    for item in apply(out, names):
        print("applied", item)


if __name__ == "__main__":
    main()
