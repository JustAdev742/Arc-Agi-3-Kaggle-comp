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
# run came from missing imports (json, Counter, ...) or earlier calls' names. Also allow `class` statements: the
# restricted builtins lacked __build_class__, so any class definition failed with "NameError: __build_class__ not found".
# The whitelist also lacked object, super, setattr and the common exception classes, so `except KeyError:` raised
# NameError exactly when the exception fired; `__name__` is "__main__" so a `if __name__ == "__main__":` block runs.
P7_OLD = '        runtime_globals["__builtins__"]["__import__"] = _safe_import\n'
P7_NEW = ('        runtime_globals["__builtins__"]["__import__"] = _safe_import\n'
          '        runtime_globals["__builtins__"]["__build_class__"] = builtins.__build_class__\n'
          '        for _name in ("object", "super", "property", "staticmethod", "classmethod", "setattr", "delattr",\n'
          '                      "id", "vars", "BaseException", "KeyError", "IndexError", "AttributeError",\n'
          '                      "StopIteration", "ZeroDivisionError", "AssertionError", "NotImplementedError",\n'
          '                      "LookupError", "ArithmeticError", "OverflowError", "RecursionError", "NameError"):\n'
          '            runtime_globals["__builtins__"].setdefault(_name, getattr(builtins, _name))\n'
          '        runtime_globals["__name__"] = "__main__"\n'
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

# P7 (cont.): the final `result` was converted to JSON after stdout was restored, so an object whose __str__ prints
# wrote a stray line into the host protocol and the call came back as "invalid response"; convert inside the redirect.
P7_FINAL_OLD = """            with contextlib.redirect_stdout(stdout):
                exec(compiled, runtime_globals, runtime_globals)
            _send(
                {
                    "type": "final",
                    "stdout": stdout.getvalue(),
                    "result": _json_safe(runtime_globals.get("result")),
                    "action_results": _json_safe(action_results),
                }
            )
"""
P7_FINAL_NEW = """            with contextlib.redirect_stdout(stdout):
                exec(compiled, runtime_globals, runtime_globals)
                _final_result = _json_safe(runtime_globals.get("result"))
                _final_actions = _json_safe(action_results)
            _send(
                {
                    "type": "final",
                    "stdout": stdout.getvalue(),
                    "result": _final_result,
                    "action_results": _final_actions,
                }
            )
"""

# P4: once a user message is older than the newest one, its standing instructions (about 600 tokens repeated every
# turn), its copy of the carried note (superseded by the newest) and its board image (about 200 tokens) carry nothing
# new; they are cut from the stored history, and so is the reasoning of finished turns (35% of all prompt tokens). The
# newest user message is still sent in full with the current image. Pair with a lower LOCAL_ANALYZER_CONTEXT_WINDOW,
# or the harness refills the freed budget with more history.
P4_FN = '''_HISTORY_USER_CUT_MARKER = "\\nOnly tool: `python`."
_STRIP_PAST_REASONING = os.environ.get("OURS_STRIP_PAST_REASONING", "1") == "1"


def _compress_history_message(message: dict[str, Any]) -> dict[str, Any]:
    """An older turn as it is kept in history.

    User turns lose their standing instructions, stale note and image (the newest turn keeps all three). Assistant
    turns lose their reasoning: the served template renders it (35% of prompt tokens in a public run), while Qwen's
    guidance is to keep only the final output of earlier turns; the current turn keeps its reasoning.
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
        text = text[:cut] + "\\n[older turn: instructions, note and image omitted]"
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

# P9: after an action sequence the model usually spends its next call computing what changed (62% of calls re-split
# `.ascii` in the thui run). The harness now states it: changed cells grouped into up to three regions (cells within
# two of each other join), each with its bounding box and main colour transitions, in the prompt's colour letters.
P9_FN = '''def _board_diff_line(before: Any, after: Any) -> str:
    """One line describing how the board changed between two grids (empty when shapes differ)."""
    from collections import Counter

    from inference.utils.grid_utils import ARC_COLOR_CHARS

    if not before or not after or len(before) != len(after):
        return ""
    cells: dict[tuple[int, int], tuple[int, int]] = {}
    for r, (row_b, row_a) in enumerate(zip(before, after)):
        if row_b == row_a or len(row_b) != len(row_a):
            continue
        for c, (vb, va) in enumerate(zip(row_b, row_a)):
            if vb != va:
                cells[(r, c)] = (int(vb), int(va))
    if not cells:
        return "Board diff over that sequence: no cell changed on the final board."

    def sym(v: int) -> str:
        return ARC_COLOR_CHARS[max(0, min(15, v))]

    seen: set[tuple[int, int]] = set()
    regions: list[list[tuple[int, int]]] = []
    for start in cells:
        if start in seen:
            continue
        seen.add(start)
        stack, comp = [start], []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for dr in (-2, -1, 0, 1, 2):
                for dc in (-2, -1, 0, 1, 2):
                    nb = (r + dr, c + dc)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        regions.append(comp)
    regions.sort(key=len, reverse=True)
    parts = []
    for comp in regions[:3]:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        trans = Counter(f"{sym(cells[p][0])}>{sym(cells[p][1])}" for p in comp).most_common(2)
        parts.append(
            f"{len(comp)} cells in rows {min(rows)}-{max(rows)}, cols {min(cols)}-{max(cols)} ("
            + ", ".join(f"{t} x{n}" for t, n in trans) + ")"
        )
    more = f"; {len(regions) - 3} smaller region(s) not listed" if len(regions) > 3 else ""
    return (f"Board diff over that sequence: {len(cells)} cells changed in {len(regions)} region(s): "
            + "; ".join(parts) + more + ".")


'''
P9_OLD = '''            animation_line = describe_animation(previous_step_summary.get("animation"))
            if animation_line:
                lines.append(animation_line)
'''
P9_NEW = '''            animation_line = describe_animation(previous_step_summary.get("animation"))
            if animation_line:
                lines.append(animation_line)
            try:
                executed = int(previous_step_summary.get("executed_count") or 0)
            except (TypeError, ValueError):
                executed = 0
            if (
                executed > 0
                and not previous_step_summary.get("level_transition")
                and not previous_step_summary.get("game_over")
                and current_frame is not None
                and len(history_entries) > executed
            ):
                diff_line = _board_diff_line(history_entries[-executed - 1].frame.grid, current_frame.grid)
                if diff_line:
                    lines.append(diff_line)
'''

# P10: the score weights level k by k, so level 1 is the cheapest level to spend actions on (1/28 of a 7-level game),
# yet the prompt asks for the fewest actions everywhere and the model deliberates for minutes before its first probes
# (a median of 3 calls, 8 minutes, before the first action on a new level). On level 1 only, the prompt now says that
# quick, batched probe actions are cheap there.
P10_OLD = '        lines.append("end of world model. ")\n'
P10_NEW = ('        lines.append("end of world model. ")\n'
           '        if current_level == 1:\n'
           '            lines.append(\n'
           '                "Scoring note: level 1 counts least toward the score, so actions spent here to learn what each "\n'
           '                "action does and what completes a level are cheap; prefer a few quick probe actions (batch "\n'
           '                "several in one call) over long deliberation until you know. From level 2 on, plan before acting."\n'
           '            )\n')

# P6: every python call started from a blank namespace, so helpers were rewritten again and again (54% of 1,116
# function definitions in the thui run redefined an existing name; 32% of code lines repeated earlier lines) and names
# from earlier calls raised NameError. Top-level functions and classes (undecorated), imports and UPPER_CASE literal
# constants from a successful call are kept per game (at most 12,000 characters, oldest dropped) and replayed silently
# before the next call's code; each replayed piece runs in its own try, so a stale one cannot break the call.
PROMPTS = "src/ARC3-Inference/inference/agent/prompts.py"
P6_FN = '''_PERSIST_RESERVED = {
    "action", "animation", "current_frame", "latest_frame", "previous_frame", "history", "transitions",
    "last_transition", "last_action", "last_action_frame", "last_action_result", "valid_actions", "result", "print",
}
_PERSIST_MAX_CHARS = 12000


def _persist_definitions(store: dict[str, str], code: str) -> None:
    """Keep top-level defs, classes, imports and UPPER_CASE literal constants of a successful call."""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return
    for node in tree.body:
        key = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.decorator_list:
            key = node.name if node.name not in _PERSIST_RESERVED else None
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            key = "import:" + (ast.get_source_segment(code, node) or ",".join(alias.name for alias in node.names))
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.isupper()
            and node.targets[0].id not in _PERSIST_RESERVED
        ):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                continue
            key = "const:" + node.targets[0].id
        if key is None:
            continue
        source = ast.get_source_segment(code, node)
        if source:
            store.pop(key, None)
            store[key] = source
    while store and sum(len(v) for v in store.values()) > _PERSIST_MAX_CHARS:
        store.pop(next(iter(store)))


def _persisted_helper_names(store: dict[str, str]) -> list[str]:
    import ast

    names: list[str] = []
    for key, source in store.items():
        if key.startswith(("import:", "const:")):
            if key.startswith("const:"):
                names.append(key[len("const:"):])
            continue
        try:
            node = ast.parse(source).body[0]
        except (SyntaxError, IndexError):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(f"{node.name}({', '.join(a.arg for a in node.args.args)})")
        else:
            names.append(key)
    return names[-16:]


'''
P6_SANDBOX_CHILD_OLD = '''        _refresh_state(initial.get("state") or {})

        try:
            compiled = compile(str(initial.get("code", "")), "<python_tool>", "exec")'''
P6_SANDBOX_CHILD_NEW = '''        _refresh_state(initial.get("state") or {})
        _live = {_k: runtime_globals.pop(_k) for _k in ("action", "animation") if _k in runtime_globals}
        with contextlib.redirect_stdout(io.StringIO()):
            for _snippet in initial.get("prelude") or []:
                try:
                    exec(compile(str(_snippet), "<persisted>", "exec"), runtime_globals, runtime_globals)
                except Exception:
                    pass
        runtime_globals.update(_live)

        try:
            compiled = compile(str(initial.get("code", "")), "<python_tool>", "exec")'''
P6_SANDBOX_SIG_OLD = ("    animation_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,\n"
                      ") -> dict[str, Any]:\n")
P6_SANDBOX_SIG_NEW = ("    animation_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,\n"
                      "    prelude: list[str] | None = None,\n"
                      ") -> dict[str, Any]:\n")
P6_SANDBOX_MSG_OLD = '                "animation_enabled": animation_handler is not None,\n'
P6_SANDBOX_MSG_NEW = ('                "animation_enabled": animation_handler is not None,\n'
                      '                "prelude": list(prelude or []),\n')
# P6: a traceback inside a kept helper (compiled as "<persisted>") showed only the caller's line; keep those frames.
P6_TB_OLD = """        user_frames = [frame for frame in extracted if frame.filename == "<python_tool>"]
        lines = ["Traceback (most recent call last):"]
        for frame in user_frames or extracted[-1:]:
            lines.append(f'  File "<python_tool>", line {frame.lineno}, in {frame.name}')
"""
P6_TB_NEW = """        user_frames = [frame for frame in extracted if frame.filename in ("<python_tool>", "<persisted>")]
        lines = ["Traceback (most recent call last):"]
        for frame in user_frames or extracted[-1:]:
            _shown = frame.filename if frame.filename == "<persisted>" else "<python_tool>"
            lines.append(f'  File "{_shown}", line {frame.lineno}, in {frame.name}')
"""
P6_CALL_OLD = '''        sandbox_result = run_sandboxed_python(
            code=code,
'''
P6_CALL_NEW = '''        if not hasattr(self, "_persisted_defs"):
            self._persisted_defs = {}
        sandbox_result = run_sandboxed_python(
            code=code,
            prelude=list(self._persisted_defs.values()),
'''
P6_STORE_OLD = '        step_executed = any(bool(item.get("executed")) for item in action_results)\n'
P6_STORE_NEW = ('        if not rendered_error:\n'
                '            _persist_definitions(self._persisted_defs, code)\n'
                '        step_executed = any(bool(item.get("executed")) for item in action_results)\n')
P6_PROMPT_OLD = '        lines.extend(self._summarized_knowledge_lines())\n'
P6_PROMPT_NEW = ('        _helpers = _persisted_helper_names(getattr(self, "_persisted_defs", {}) or {})\n'
                 '        if _helpers:\n'
                 '            lines.append(\n'
                 '                "Kept from your earlier successful python calls and re-created in every call: "\n'
                 '                + ", ".join(_helpers)\n'
                 '                + ". Call them directly instead of rewriting them; define one again to replace it."\n'
                 '            )\n'
                 '        lines.extend(self._summarized_knowledge_lines())\n')
P6_SYS1_OLD = '    "- Every `python` tool call starts fresh. Re-import modules or re-define any custom utility logic you need.\\n"\n'
P6_SYS1_NEW = ('    "- Each `python` tool call starts with fresh variables, but the top-level functions, classes, imports and '
               'UPPER_CASE constants of your earlier successful calls are re-created automatically (the user message lists '
               'them): reuse them instead of rewriting them. Other variables are not kept.\\n"\n')
P6_SYS2_OLD = '    "- The `python` tool code is not saved between calls, so rewrite any custom utility logic you still need.\\n"\n'
P6_SYS2_NEW = ('    "- Variables are not saved between calls; your top-level functions, classes, imports and UPPER_CASE '
               'constants are (see above).\\n"\n')
P6_TOOLDESC_OLD = '"Python code to run. The snippet is ephemeral and is not saved across tool calls."'
P6_TOOLDESC_NEW = ('"Python code to run. Variables do not persist across calls; top-level functions, classes, imports and '
                   'UPPER_CASE constants of successful calls do."')

# P11: the served Flash-Next chat template defaults to reasoning effort "xhigh" (it prepends "Reasoning effort is set to
# xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives ..." to the
# system prompt) and also accepts "medium" (no instruction) and "low"; it renders the reasoning of every past assistant
# turn unless `preserve_thinking` is false. About 81% of the output is reasoning and the score keeps rising until the
# time limit in every harvested run, so both are knobs: OURS_REASONING_EFFORT (xhigh | medium | low; unset keeps the
# template default) and OURS_PRESERVE_THINKING (0 | 1; unset keeps the template default, which renders them).
UTILS_COMPAT = "src/ARC3-Inference/inference/utils/openai_compat.py"
P11_IMPORT_OLD = "from typing import Any\n"
P11_IMPORT_NEW = "import os\nfrom typing import Any\n"
P11_OLD = '        payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}\n'
P11_NEW = (P11_OLD
           + '        _effort = os.environ.get("OURS_REASONING_EFFORT", "").strip().lower()\n'
           + '        if thinking and _effort in ("xhigh", "medium", "low"):\n'
           + '            payload["chat_template_kwargs"]["reasoning_effort"] = _effort\n'
           + '        _preserve = os.environ.get("OURS_PRESERVE_THINKING", "").strip()\n'
           + '        if _preserve in ("0", "1"):\n'
           + '            payload["chat_template_kwargs"]["preserve_thinking"] = _preserve == "1"\n')

PATCHES.update({
    "P11": [(UTILS_COMPAT, P11_IMPORT_OLD, P11_IMPORT_NEW), (UTILS_COMPAT, P11_OLD, P11_NEW)],
    "P6": [
        (TOOL_AGENT, "def _empty_world_model(", P6_FN + "def _empty_world_model("),
        (SANDBOX, P6_SANDBOX_CHILD_OLD, P6_SANDBOX_CHILD_NEW),
        (SANDBOX, P6_SANDBOX_SIG_OLD, P6_SANDBOX_SIG_NEW),
        (SANDBOX, P6_SANDBOX_MSG_OLD, P6_SANDBOX_MSG_NEW),
        (SANDBOX, P6_TB_OLD, P6_TB_NEW),
        (TOOL_AGENT, P6_CALL_OLD, P6_CALL_NEW),
        (TOOL_AGENT, P6_STORE_OLD, P6_STORE_NEW),
        (TOOL_AGENT, P6_PROMPT_OLD, P6_PROMPT_NEW),
        (PROMPTS, P6_SYS1_OLD, P6_SYS1_NEW),
        (PROMPTS, P6_SYS2_OLD, P6_SYS2_NEW),
        (TOOL_AGENT, P6_TOOLDESC_OLD, P6_TOOLDESC_NEW),
    ],
    "P10": [(TOOL_AGENT, P10_OLD, P10_NEW)],
    "P9": [
        (TOOL_AGENT, "def _empty_world_model(", P9_FN + "def _empty_world_model("),
        (TOOL_AGENT, P9_OLD, P9_NEW),
    ],
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
    "P7": [(SANDBOX, P7_OLD, P7_NEW), (SANDBOX, P7_FINAL_OLD, P7_FINAL_NEW)],
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
