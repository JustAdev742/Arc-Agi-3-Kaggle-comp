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
    lowered_labels = {label.lower() + ":": label for label in _NOTE_LABELS}
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in reasoning.splitlines():
        stripped = raw_line.strip()
        candidate = stripped
        while candidate.startswith(("-", "*", "#")):
            candidate = candidate[1:].lstrip()
        candidate = candidate.replace("**", "")
        lowered = candidate.lower()
        matched = next((label for key, label in lowered_labels.items() if lowered.startswith(key)), None)
        if matched is not None:
            current = matched
            inline = candidate[len(matched) + 1:].strip()
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
