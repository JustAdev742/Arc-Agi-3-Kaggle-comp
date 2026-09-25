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
                try:  # prompt building runs outside the analyzer's try: a bad entry must not end the game
                    diff_line = _board_diff_line(history_entries[-executed - 1].frame.grid, current_frame.grid)
                except Exception:
                    diff_line = ""
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

# P12: the python tool's compile pre-check caught only SyntaxError, so code that compile() rejects otherwise (a null
# byte: ValueError; a lone surrogate: UnicodeEncodeError; a very deep expression: RecursionError) raised out of the
# tool runner and ended the game (code review 2026-09-23). Now it is an ordinary tool error the model can fix.
P12_OLD = """        try:
            compile(code, "<python_tool>", "exec")
        except SyntaxError as exc:
            return _ToolDispatchResult(json.dumps({"error": f"Python syntax error: {exc}"}, indent=2))
"""
P12_NEW = P12_OLD + """        except (ValueError, UnicodeError, RecursionError, MemoryError, OverflowError) as exc:
            return _ToolDispatchResult(json.dumps(
                {"error": f"Python code could not be compiled: {type(exc).__name__}: {exc}"}, indent=2))
"""

# P13-P16 (exp-039): from the level-2+ stall analysis of the thui run (17 games that solved some levels, 1,252 stuck
# minutes; research log 2026-09-23): 68% of stuck time went to calls that took no action, new mechanics on a level
# went unprobed while level-1 assumptions were kept (lf52's arrow keys first pressed 42 min into level 2), the
# previous level left the context a median 22 min after level-up, and three games read the cumulative `score` as a
# per-action reward.
#
# P13: say what `score` and `reward` in an action result mean.
PROMPTS_SCORE_OLD = "and `last_action_result['valid_actions']`.\\n\"\n"
PROMPTS_SCORE_NEW = (PROMPTS_SCORE_OLD
                     + "    \"- `last_action_result['score']` counts the levels completed so far in this game: it stays the same "
                       "while you play a level and rises by 1 only when a level is completed, so it is not a reward for the last "
                       "action. `reward` is non-zero only on the action that completes a level.\\n\"\n")

# P14: at a level change the harness compares the new board with the start of the previous level (colours and small
# object shapes that are new, exact from the frames) and lists the actions never tried in this game, and asks for one
# cheap probe of each before the previous level's plan is reused. P15: after a stretch with no action (8+ minutes on a
# level that is 10+ minutes old) the next prompt asks for a 1-3 action probe with a stated prediction. P16: a record of
# how each completed level ended (its action count and last actions, from the game log) stays in every prompt, since
# the previous level's turns leave the context about 22 minutes after level-up.
P14_FN = '''def _ours_object_kinds(grid: Any) -> dict[tuple[str, str], tuple[int, int, tuple[int, int, int, int]]]:
    """(colour, shape hash) -> (count, cells, bounding box of one instance) for one board."""
    from inference.utils.grid_utils import ARC_COLOR_CHARS
    from inference.utils.segmentation import segment_layer

    kinds: dict[tuple[str, str], tuple[int, int, tuple[int, int, int, int]]] = {}
    for node in segment_layer(grid, ARC_COLOR_CHARS).get("nodes", []):
        rows = [point[0] for point in node["boundary"]]
        cols = [point[1] for point in node["boundary"]]
        key = (str(node["color"]), str(node["hash"]))
        count, cells, box = kinds.get(key, (0, int(node["pixels"]), (min(rows), min(cols), max(rows), max(cols))))
        kinds[key] = (count + 1, cells, box)
    return kinds


def _ours_untried_actions(history_entries: list[Any], valid_actions: list[str] | None) -> list[str]:
    used = set()
    for entry in history_entries:
        name = str(getattr(entry, "action", "") or "").split("(", 1)[0].strip().upper()
        if name:
            used.add(name)
    return [a for a in _normalize_valid_actions(valid_actions) if a.upper() not in used and a.upper() != "RESET"]


def _ours_level_start_lines(history_entries: list[Any], current_frame: Any, level: int,
                            valid_actions: list[str] | None) -> list[str]:
    """What is new on a level's first board compared with the previous level's first board, and untried actions."""
    lines = [
        "If this board has new kinds of objects (listed below) or actions you have never tried, test each cheaply "
        "(one action or one short probe) and note what it does before reusing the previous level's plan."
    ]
    prev_start = next((e.frame for e in history_entries
                       if getattr(e, "frame", None) is not None and e.frame.level == level - 1), None)
    if prev_start is not None and current_frame is not None:
        before, after = _ours_object_kinds(prev_start.grid), _ours_object_kinds(current_frame.grid)
        new_colours = sorted({c for c, _ in after} - {c for c, _ in before})
        new_kinds = sorted(((k, v) for k, v in after.items() if k not in before and v[1] <= 25),
                           key=lambda kv: (kv[1][1], kv[1][0]))
        gone = sorted(((k, v) for k, v in before.items() if k not in after and v[1] <= 25),
                      key=lambda kv: (kv[1][1], kv[1][0]))

        def _kind(kv: Any) -> str:
            (colour, _), (count, cells, (r0, c0, r1, c1)) = kv
            return f"{colour} {r1 - r0 + 1}x{c1 - c0 + 1} x{count} (e.g. rows {r0}-{r1}, cols {c0}-{c1})"

        parts = [f"colours new on this level: {', '.join(new_colours) if new_colours else 'none'}"]
        if new_kinds:
            parts.append("new small object kinds: " + "; ".join(_kind(kv) for kv in new_kinds[:5]))
        if gone:
            parts.append("kinds no longer present: " + "; ".join(_kind(kv) for kv in gone[:3]))
        lines.append(f"Harness comparison with the first board of level {level - 1} (exact): " + ". ".join(parts) + ".")
    untried = _ours_untried_actions(history_entries, valid_actions)
    if untried:
        lines.append("Actions you have never tried in this game: " + ", ".join(untried) + ".")
    return lines


def _ours_level_record(history_entries: list[Any], level: int, limit: int = 16) -> str:
    """How level `level` ended, from the game log: its action count and last actions, run-length encoded."""
    names = [str(history_entries[i].action or "") for i in range(1, len(history_entries))
             if getattr(history_entries[i - 1], "frame", None) is not None
             and history_entries[i - 1].frame.level == level]
    if not names:
        return ""
    runs: list[list[Any]] = []
    for name in names[-limit:]:
        if runs and runs[-1][0] == name:
            runs[-1][1] += 1
        else:
            runs.append([name, 1])
    tail = ", ".join(f"{n} x{k}" if k > 1 else n for n, k in runs)
    return f"level {level} took {len(names)} actions; the last {min(limit, len(names))}: {tail}"


'''
P14_OLD = ('            if previous_step_summary.get("game_over"):\n'
           '                lines.append("The game is over.")\n')
P14_NEW = (P14_OLD
           + '            if (previous_step_summary.get("level_transition") and not previous_step_summary.get("run_complete")\n'
           + '                    and getattr(self, "_ours_p14_level", None) != current_level):\n'
           + '                self._ours_p14_level = current_level\n'
           + '                try:  # prompt building runs outside the analyzer\'s try\n'
           + '                    lines.extend(_ours_level_start_lines(history_entries, current_frame, current_level, valid_actions))\n'
           + '                    _record = _ours_level_record(history_entries, current_level - 1)\n'
           + '                    if _record:\n'
           + '                        _records = getattr(self, "_ours_level_records", None) or {}\n'
           + '                        _records[current_level - 1] = _record\n'
           + '                        self._ours_level_records = _records\n'
           + '                except Exception:\n'
           + '                    pass\n')
P16_OLD = "        lines.extend(self._summarized_knowledge_lines())\n"
P16_NEW = (P16_OLD
           + '        _records = getattr(self, "_ours_level_records", None) or {}\n'
           + '        if _records:\n'
           + '            lines.append("Harness record of completed levels (exact, from the game log): "\n'
           + '                         + "; ".join(_records[k] for k in sorted(_records)[-3:]) + ".")\n')
P15_OLD = '        lines.append("end of world model. ")\n'
P15_NEW = (P15_OLD
           + '        try:\n'
           + '            _now = time.monotonic()\n'
           + '            if getattr(self, "_ours_t_level", None) is None:\n'
           + '                self._ours_t_level = self._ours_t_act = _now\n'
           + '                self._ours_level, self._ours_n_hist = current_level, len(history_entries)\n'
           + '            if current_level != self._ours_level:\n'
           + '                self._ours_t_level = self._ours_t_act = _now\n'
           + '                self._ours_level = current_level\n'
           + '            if len(history_entries) != self._ours_n_hist:\n'
           + '                self._ours_t_act = _now\n'
           + '                self._ours_n_hist = len(history_entries)\n'
           + '            _idle = (_now - self._ours_t_act) / 60.0\n'
           + '            if (_idle >= float(os.environ.get("OURS_GOVERNOR_IDLE_MIN", "8"))\n'
           + '                    and (_now - self._ours_t_level) / 60.0 >= float(os.environ.get("OURS_GOVERNOR_LEVEL_MIN", "10"))):\n'
           + '                lines.append(\n'
           + '                    f"No action for {_idle:.0f} minutes on this level: in your next python call, run a probe of 1-3 "\n'
           + '                    "actions that tests your best current hypothesis. Print the outcome you predict before acting "\n'
           + '                    "and compare it after; a wrong prediction is progress."\n'
           + '                )\n'
           + '            if len(history_entries) > 5 and not any(l.startswith("Actions you have never tried") for l in lines):\n'
           + '                _untried = _ours_untried_actions(history_entries, valid_actions)\n'
           + '                if _untried:\n'
           + '                    lines.append("Actions you have never tried in this game: " + ", ".join(_untried) + ".")\n'
           + '        except Exception:\n'
           + '            pass\n')

# P17: 12 of 34 failed tool calls in the stall analysis involved animation(): its error replies carry only "error",
# so model code reading `steps` or `frames` raised KeyError; and the stage-3 hint re-fired on a stuck level with no new
# animation to look at (s5i5 re-read one stale animation 7 times after 4 hints). Error replies now keep the usual keys
# (empty), and the hint fires again on a level only after a new transient animation.
P17_VIEW_OLD = """            if not view.get("error"):
                self._bump_animation_counter("stage2_animation_requests_served")
            return view
"""
P17_VIEW_NEW = """            if not view.get("error"):
                self._bump_animation_counter("stage2_animation_requests_served")
            else:  # ours P17: keep the usual keys so code that reads them does not raise KeyError
                view = {"action": None, "action_num": request.get("action_num"), "frames": 0, "unique_frames": 0,
                        "board_unchanged": None, "steps": [], "frame": None, "region": "", "ascii": "", **view}
            return view
"""
P17_HINT_OLD = """        self._animation_turns_since_hint = 0
        self._animation_hint_follow_window = ANIMATION_HINT_FOLLOW_WINDOW_TURNS
"""
P17_HINT_NEW = """        _last_hint = getattr(self, "_ours_hint_anims", None)
        if _last_hint is not None and _last_hint[0] == current_level and self._animation_transient_animations <= _last_hint[1]:
            return ""  # ours P17: nothing new to look at since the last hint on this level
        self._ours_hint_anims = (current_level, self._animation_transient_animations)
""" + P17_HINT_OLD

# P18 (from VISTA and arc3cb, the frontier harnesses at 100 on the public games): before each action the model states
# the change it expects, and the next prompt's board diff (P9) shows what happened, so a wrong model is caught by
# one action instead of by a stalled level.
P18_OLD = ('                "When ready, call `action(actions)` from inside the `python` tool with the best valid action or ordered '
           'batch selected by your code. If your code has found a reliable short sequence, prefer batching it in one call.",\n')
P18_NEW = (P18_OLD
           + '                "Before each `action(...)` call, print one line starting with `expect:` that states the change you '
             'predict (for example `expect: the red block moves one cell left`); afterwards compare it with what happened and '
             'fix your world model where they differ. Where you can, test a hypothesis on `history`/`transitions` in Python '
             'before spending an action on it.",\n')

# P19 (from NVIDIA AVO's supervisor): when a level has made no progress for OURS_SUPERVISOR_MIN minutes (default 30,
# above the 24-31 min median time of a solved level), the harness makes one extra call to the same model, as a
# reviewer that sees the notes, this level's actions, the untried actions and the board, and asks what is most likely
# wrong and for two alternative hypotheses with a cheap probe each. Its reply stays in the prompt until the next
# review or level. At most one review per OURS_SUPERVISOR_MIN minutes on a level; 0 turns it off.
P19_FN = '''def _ours_supervisor_review(agent: Any, history_entries: list[Any], current_frame: Any, level: int,
                            valid_actions: list[str] | None, minutes: float) -> str:
    """One reviewer call on a stalled level; returns its reply as one line (empty on any failure)."""
    from inference.utils.grid_utils import format_grid_ascii

    names = [str(history_entries[i].action or "") for i in range(1, len(history_entries))
             if getattr(history_entries[i - 1], "frame", None) is not None
             and history_entries[i - 1].frame.level == level]
    runs: list[list[Any]] = []
    for name in names[-60:]:
        if runs and runs[-1][0] == name:
            runs[-1][1] += 1
        else:
            runs.append([name, 1])
    used = {str(getattr(e, "action", "") or "").split("(", 1)[0].strip().upper() for e in history_entries}
    valid = _normalize_valid_actions(valid_actions)
    untried = [a for a in valid if a.upper() not in used and a.upper() != "RESET"]
    records = getattr(agent, "_ours_level_records", None) or {}
    board = format_grid_ascii(current_frame.grid) if current_frame is not None else "(no board)"
    prompt = (
        f"You are reviewing an agent that plays a grid puzzle game (a 64x64 board of colour letters; actions: "
        f"{', '.join(valid)}). It has been on level {level} for {minutes:.0f} minutes without completing it.\\n\\n"
        "Its current notes:\\n" + ("\\n".join(agent._summarized_knowledge_lines()) or "(empty)") + "\\n\\n"
        + ("How earlier levels ended: " + "; ".join(records[k] for k in sorted(records)) + "\\n\\n" if records else "")
        + f"Actions on this level so far ({len(names)}, last 60 run-length encoded): "
        + (", ".join(f"{n} x{k}" if k > 1 else n for n, k in runs) or "none") + "\\n"
        + f"Actions never tried in this game: {', '.join(untried) or 'none'}\\n\\n"
        + "Current board:\\n" + board + "\\n\\n"
        "Say what is most likely wrong: which assumption in the notes was never tested, which hypothesis keeps failing, "
        "what the agent keeps repeating. Then give two different hypotheses about the goal or the mechanics that fit "
        "the evidence, each with one cheap probe (at most 3 actions) that would confirm or refute it. Reply in at most "
        "8 short lines of plain text."
    )
    result = agent._chat_completion([{"role": "user", "content": prompt}], tools=None, request_timeout_seconds=900)
    text = str((result.message or {}).get("content") or "").strip()
    return " | ".join(line.strip() for line in text.splitlines() if line.strip())[:1500]


'''
P19_OLD = '        lines.append("end of world model. ")\n'
P19_NEW = (P19_OLD
           + '        try:  # ours P19: a reviewer call on a stalled level (prompt building runs outside the analyzer\'s try)\n'
           + '            _now_s = time.monotonic()\n'
           + '            if getattr(self, "_ours_sup_level", None) != current_level:\n'
           + '                self._ours_sup_level, self._ours_sup_t, self._ours_sup_note = current_level, _now_s, ""\n'
           + '            _sup_min = float(os.environ.get("OURS_SUPERVISOR_MIN", "30"))\n'
           + '            if _sup_min > 0 and (_now_s - self._ours_sup_t) / 60.0 >= _sup_min:\n'
           + '                _minutes = (_now_s - getattr(self, "_ours_sup_level_t0", {}).get(current_level, self._ours_sup_t)) / 60.0\n'
           + '                self._ours_sup_t = _now_s\n'
           + '                self._ours_sup_note = _ours_supervisor_review(self, history_entries, current_frame, current_level,\n'
           + '                                                              valid_actions, max(_minutes, _sup_min))\n'
           + '                self._ours_sup_calls = getattr(self, "_ours_sup_calls", 0) + 1\n'
           + '            _t0s = getattr(self, "_ours_sup_level_t0", {})\n'
           + '            _t0s.setdefault(current_level, _now_s)\n'
           + '            self._ours_sup_level_t0 = _t0s\n'
           + '            if self._ours_sup_note:\n'
           + '                lines.append("Supervisor review of this stalled level (a second opinion; test it, do not trust it "\n'
           + '                             "blindly): " + self._ours_sup_note)\n'
           + '        except Exception:\n'
           + '            pass\n')

# P20: lever L1, no-impact detection, ported from the public notebook sahasawatt/thui-l1-v0 (implementation Sahasawat
# Wittayaprasit; idea and measurement Son Pham, sonpham-org/arc-3); yocybercode/thui-l1-v0-full25-r1 ran it on the
# public 25 (10.93, one run; its base notebook 9.32, one run). A per-game learner finds the rows that change on at
# least 90% of actions (a step counter or energy strip; at most 4 rows, after 20 actions); an action that changes only
# those rows is reported as board_changed=False and no_impact, and the outcome line says it had no gameplay effect.
# Targets the stall-analysis failure of reading HUD ticks as effects. Appended to the end of solver.py ("<EOF>").
SOLVER = "src/ARC3-Inference/inference/framework/solver.py"
P20_NEW = '''

# ---- ours P20: lever L1 no-impact detection (idea Son Pham, sonpham-org/arc-3; implementation Sahasawat
# Wittayaprasit, kaggle sahasawatt/thui-l1-v0), ported to wrap this module's session and the tool agent ----
_OURS_HUD_FRAC, _OURS_HUD_MIN, _OURS_HUD_MAX_ROWS = 0.9, 20, 4


def _ours_hud_new() -> dict:
    return {"n": 0, "rows": {}, "band": None, "noimp": 0}


def _ours_hud_update(st: dict, changed_rows: set, learn: bool = True) -> bool:
    """True when changed_rows is non-empty and lies inside the learned counter band."""
    if not changed_rows:
        return False
    if learn:
        st["n"] += 1
        for r in changed_rows:
            st["rows"][r] = st["rows"].get(r, 0) + 1
        if st["n"] >= _OURS_HUD_MIN:
            band = {r for r, k in st["rows"].items() if k / st["n"] >= _OURS_HUD_FRAC}
            st["band"] = band if 0 < len(band) <= _OURS_HUD_MAX_ROWS else None
    band = st.get("band")
    return bool(band) and changed_rows <= band


def _ours_install_l1() -> None:
    orig_exec = _HarnessGameSession._execute_action
    orig_compact = ToolAgent._compact_action_result
    orig_summ = ToolAgent._summarize_step_sequence
    orig_desc = ToolAgent._describe_last_outcome

    def _exec(self, action, **kw):
        prev = _grid_from_state(self.game.current_state)
        payload = orig_exec(self, action, **kw)
        try:
            if not payload.get("executed", True):
                return payload
            new = _grid_from_state(self.game.current_state)
            changed = {r for r in range(min(len(prev), len(new))) if prev[r] != new[r]} if prev and new else set()
            st = self.__dict__.setdefault("_ours_hud", _ours_hud_new())
            learn = getattr(getattr(action, "id", None), "name", "") != "RESET" and not payload.get("level_completed")
            if learn and _ours_hud_update(st, changed):
                st["noimp"] += 1
                payload["board_changed"] = False
                payload["no_impact"] = True
                payload["hud_rows"] = sorted(st["band"])
        except Exception:  # never let the detector break a step
            pass
        return payload

    def _compact(self, payload):
        compact = orig_compact(self, payload)
        if payload.get("no_impact"):
            compact["no_impact"] = True
            compact["hud_rows"] = payload.get("hud_rows")
        return compact

    def _summ(self, action_results):
        summary = orig_summ(self, action_results)
        if summary is not None:
            hits = [item for item in action_results if item.get("executed") and item.get("no_impact")]
            summary["no_impact_count"] = len(hits)
            if hits:
                summary["hud_rows"] = hits[-1].get("hud_rows")
        return summary

    def _desc(self, summary):
        text = orig_desc(self, summary)
        if summary and summary.get("no_impact_count"):
            text += (f" {summary['no_impact_count']} of these actions changed only the game's counter strip (rows "
                     f"{summary.get('hud_rows')}) and had NO impact on gameplay objects; do not read them as effects.")
        return text

    _HarnessGameSession._execute_action = _exec
    ToolAgent._compact_action_result = _compact
    ToolAgent._summarize_step_sequence = _summ
    ToolAgent._describe_last_outcome = _desc


_ours_install_l1()
'''

# P20 (cont.): this bundle never calls _describe_last_outcome, so the no-impact note goes into the prompt's summary of
# the last action sequence, next to the animation line.
P20_PROMPT_OLD = P9_OLD
P20_PROMPT_NEW = P9_OLD + """            if previous_step_summary.get("no_impact_count"):
                lines.append(
                    f"{previous_step_summary['no_impact_count']} of these actions changed only the game's counter strip "
                    f"(rows {previous_step_summary.get('hud_rows')}) and had NO impact on gameplay objects; do not read "
                    "them as effects."
                )
"""

# P21: the server is the bottleneck (4-6 requests fit the KV cache, about 20 more wait about 126 s each in vLLM's
# first-come queue) and the score weights level k by k, so the calls of a game on level 4 are worth about four times a
# level-1 call, and that game's model has already worked the game out. A gate in front of the server keeps at most
# OURS_GATE_SLOTS calls in flight (default 8, the server's max_num_seqs; 0 turns it off) and lets the waiting call with
# the largest weight x wait go next, so no game starves. Weight 1 + OURS_GATE_LEVEL_WEIGHT x (level - 1) (default
# step 1), fading as a level stalls past OURS_GATE_STALL_MIN minutes (default 45). Every OURS_GATE_PRINT_S seconds
# (default 600) the gate prints its counts to the notebook log. The base's 180 s turn yield counted wall time, which
# with about 145 s per call meant 1-2 calls per turn (95% of turns in exp-032); behind the gate a waiting game would
# yield after every call and a fast one after four, so while the gate is on a turn yields after OURS_GATE_TURN_CALLS
# calls (default 2; 0 turns the rule off) and gate waits do not count toward the time rule. Review fixes (subagent,
# 2026-09-23): the periodic print cannot leak a slot, a retry after a gate timeout keeps its accumulated wait, and no
# slot is taken with under OURS_GATE_MIN_LEFT_S (30) seconds of the call's budget left.
P21_FN = '''import threading as _ours_threading

_OURS_GATE_LOCK = _ours_threading.Lock()
_OURS_GATE_STATE: dict = {}


class _OursCallGate:
    """At most ``slots`` model calls in flight; the waiting call with the largest weight x wait goes next."""

    def __init__(self, slots: int) -> None:
        self.slots = max(1, int(slots))
        self._cv = _ours_threading.Condition()
        self._busy = 0
        self._waiting: dict[int, tuple[float, float]] = {}
        self._next_ticket = 0
        self.admitted = 0
        self.timeouts = 0
        self.waited_s = 0.0
        self._last_print = time.monotonic()

    def _best(self, now: float) -> int | None:
        best, best_score = None, -1.0
        for ticket, (weight, t0) in self._waiting.items():
            score = weight * (now - t0)
            if score > best_score:
                best, best_score = ticket, score
        return best

    def acquire(self, weight: float, timeout: float | None, should_stop: Callable[[], bool] | None = None,
                since: float | None = None) -> float:
        """Wait for a slot; return the seconds waited. Raises requests.Timeout past ``timeout``.

        ``since`` is when this call first started waiting: a retry after a timeout keeps its accumulated priority.
        With OURS_GATE_MIN_LEFT_S (default 30) or less of the budget left no slot is taken (the call could not finish);
        the call times out instead and its retry keeps its place. A budget under twice that takes any slot it gets.
        """
        start = time.monotonic()
        t0 = start if since is None else min(start, float(since))
        deadline = None if timeout is None else start + max(0.0, float(timeout))
        min_left = _get_env_float("OURS_GATE_MIN_LEFT_S", 30.0)
        if timeout is not None and float(timeout) < 2 * min_left:
            min_left = 0.0
        message, waited = "", 0.0
        with self._cv:
            ticket = self._next_ticket
            self._next_ticket += 1
            self._waiting[ticket] = (max(0.05, float(weight)), t0)
            try:
                while True:
                    now = time.monotonic()
                    if deadline is not None and now >= deadline - min_left:
                        self.timeouts += 1
                        raise requests.Timeout(f"waited {now - start:.0f}s for a model slot")
                    if self._busy < self.slots and self._best(now) == ticket:
                        waited = now - start
                        self.admitted += 1
                        self.waited_s += waited
                        if now - self._last_print >= _get_env_float("OURS_GATE_PRINT_S", 600.0):
                            self._last_print = now
                            weights = sorted(round(w, 1) for w, _ in self._waiting.values())
                            message = (f"ours-gate: admitted {self.admitted}, mean wait "
                                       f"{self.waited_s / self.admitted:.1f}s, timeouts {self.timeouts}, "
                                       f"busy {self._busy + 1}/{self.slots}, waiting weights {weights}")
                        self._busy += 1  # last: nothing between here and the caller's try/finally can raise
                        break
                    if should_stop is not None:
                        try:
                            stop = bool(should_stop())
                        except Exception:
                            stop = False
                        if stop:
                            raise requests.RequestException("stopped while waiting for a model slot")
                    left = 1.0 if deadline is None else deadline - min_left - now
                    self._cv.wait(max(0.01, min(1.0, left)))
            finally:
                self._waiting.pop(ticket, None)
                self._cv.notify_all()
        if message:  # printed outside the lock, and a broken stdout cannot cost the slot
            try:
                print(message, flush=True)
            except Exception:
                pass
        return waited

    def release(self) -> None:
        with self._cv:
            self._busy = max(0, self._busy - 1)
            self._cv.notify_all()


def _ours_call_gate() -> "_OursCallGate | None":
    with _OURS_GATE_LOCK:
        if "gate" not in _OURS_GATE_STATE:
            slots = _get_env_int("OURS_GATE_SLOTS", 8)
            _OURS_GATE_STATE["gate"] = _OursCallGate(slots) if slots > 0 else None
        return _OURS_GATE_STATE["gate"]


def _ours_gate_weight(agent: Any) -> float:
    """1 on level 1; 1 + step x (level - 1) on later levels, fading once a level has stalled past OURS_GATE_STALL_MIN."""
    level = max(1, int(getattr(agent, "_ours_gate_level", 1) or 1))
    step = _get_env_float("OURS_GATE_LEVEL_WEIGHT", 1.0)
    stall_min = _get_env_float("OURS_GATE_STALL_MIN", 45.0)
    now = time.monotonic()
    minutes = (now - float(getattr(agent, "_ours_gate_level_t0", now))) / 60.0
    fade = 1.0 if stall_min <= 0 or minutes <= stall_min else stall_min / minutes
    return 1.0 + max(0.0, step) * (level - 1) * fade


'''
P21_LEVEL_OLD = ("        current_frame, history_entries = load_runtime_state(state_path)\n"
                 "        user_prompt = self._build_user_prompt(\n")
P21_LEVEL_NEW = ("        current_frame, history_entries = load_runtime_state(state_path)\n"
                 "        _ours_lv = int(getattr(current_frame, 'level', 1) or 1) if current_frame is not None else 1\n"
                 "        if _ours_lv != getattr(self, '_ours_gate_level', None):  # ours P21: level and its start time\n"
                 "            self._ours_gate_level, self._ours_gate_level_t0 = _ours_lv, time.monotonic()\n"
                 "        self._ours_gate_stop = should_stop\n"
                 "        user_prompt = self._build_user_prompt(\n")
P21_CALL_OLD = "        response = post_chat(payload)\n"
P21_CALL_NEW = """        _ours_gate = _ours_call_gate()  # ours P21: weighted fair share of the model server across games
        if _ours_gate is None:
            response = post_chat(payload)
        else:
            _ours_budget = request_timeout_seconds if request_timeout_seconds is not None else self._timeout
            _ours_since, _ours_try = getattr(self, "_ours_gate_since", None), time.monotonic()
            try:
                _ours_waited = _ours_gate.acquire(_ours_gate_weight(self), _ours_budget,
                                                  getattr(self, "_ours_gate_stop", None), since=_ours_since)
            except requests.Timeout:
                self._ours_gate_since = _ours_try if _ours_since is None else _ours_since  # the retry keeps its place
                raise
            self._ours_gate_since = None
            try:
                if _ours_budget is not None:
                    request_timeout_seconds = max(1.0, float(_ours_budget) - _ours_waited)
                response = post_chat(payload)
            finally:
                _ours_gate.release()
            self._ours_gate_wait_turn = getattr(self, "_ours_gate_wait_turn", 0.0) + _ours_waited
"""
P21_TURN_OLD = "        turn_started_at = time.monotonic()\n"
P21_TURN_NEW = ("        turn_started_at = time.monotonic()\n"
                "        self._ours_gate_wait_turn = 0.0  # ours P21: gate waits do not count toward the turn yield\n")
P21_YIELD_OLD = ("            if self._yield_seconds is not None and (time.monotonic() - turn_started_at) >= self._yield_seconds:\n"
                 "                return \"turn_time_budget\"\n")
P21_YIELD_NEW = ("            _ours_gated = _ours_call_gate() is not None  # ours P21: the base's turn shape behind the gate\n"
                 "            _ours_busy_s = time.monotonic() - turn_started_at - getattr(self, '_ours_gate_wait_turn', 0.0)\n"
                 "            if self._yield_seconds is not None and _ours_busy_s >= self._yield_seconds:\n"
                 "                return \"turn_time_budget\"\n"
                 "            _ours_turn_calls = _get_env_int(\"OURS_GATE_TURN_CALLS\", 2)  # 0 or less: off\n"
                 "            if self._yield_seconds is not None and _ours_gated and 0 < _ours_turn_calls <= turn_count:\n"
                 "                return \"turn_time_budget\"\n")

# P22 (level-transition study, 2026-09-23, 38 level starts in exp-032): at a new level's first calls `previous_frame`
# and `last_transition` still held the level-completing step, so a diff compared the old level's last board with the
# new level's first; 5 level starts spent reasoning on it ("previous_frame is the L1 end screen ... not useful"). A
# transition that crosses a level boundary is no longer offered as the latest change: on a new level `previous_frame`,
# `last_transition` and `last_action_frame` are None until the first action there, as at a game's start. `transitions`
# keeps the full list.
P22_OLD = """            runtime_globals["previous_frame"] = (
                last_transition.before_frame if last_transition is not None else None
            )
"""
P22_NEW = """            if last_transition is not None and getattr(last_transition.before_frame, "level", None) != getattr(
                    last_transition.after_frame, "level", None):  # ours P22: a level-up is not the latest change
                last_transition = None
                runtime_globals["last_transition"] = None
            runtime_globals["previous_frame"] = (
                last_transition.before_frame if last_transition is not None else None
            )
"""

# P23 and P24 (hard-games level-1 study, 2026-09-25: 40 level-1 attempts on the 8 hardest public games, 18 unsolved).
# The most common primary failure (9 of 31 problem runs, involved in 16) was a misread action effect: g50t's SPACE
# makes a gray clone appear on the next move and replay the path, and tn36's piece moves 8 rows during the run
# animation and comes back, both invisible in a cell count or in the final board. P23 states, after every executed
# sequence, what each action did at the object level: objects (4-connected same-colour areas of up to 400 cells that
# contain a changed cell) that moved (with the offset), turned, appeared, vanished, changed colour or changed shape,
# and what happened only during the action's animation (moved and came back, showed up and was gone). Six or more
# objects moving by the same offset are one phrase (a scroll). Up to 3 actions are reported one by one, a longer
# sequence as its net change plus its last action. P24: every glyph in tr87 is drawn at a random rotation, and all 5
# runs stalled on pairing them (a rotation-invariant key pairs all 5 source glyphs; the translation-only hash pairs 2).
# At the first prompt of each level the harness lists the objects of 5-49 cells (solid rectangles left out) whose shapes
# are equal up to rotation, reflection or colour, with the rotation or mirror of each. Both cost no actions and about
# 100-300 prompt tokens; each report is given once (P23 once per sequence, P24 once per level).
OURS_OBJECTS_FN = '''from inference.utils.grid_utils import ARC_COLOR_CHARS as _OURS_COLOUR_CHARS  # ours: shared by P23 and P24

_OURS_OBJ_MAX_PX = 400  # a same-colour area larger than this is floor, wall or background, not an object
_OURS_D4 = ((1, 0, 0, 1), (0, 1, -1, 0), (-1, 0, 0, -1), (0, -1, 1, 0),   # identity, rotations 90/180/270 clockwise
            (1, 0, 0, -1), (-1, 0, 0, 1), (0, 1, 1, 0), (0, -1, -1, 0))  # mirror left-right, up-down, two diagonals
_OURS_D4_NAMES = ("same", "rotated 90", "rotated 180", "rotated 270", "mirrored left-right", "mirrored up-down",
                  "mirrored on a diagonal", "mirrored on a diagonal")


def _ours_norm(cells: Any, t: int = 0) -> tuple:
    a, b, c, d = _OURS_D4[t]
    pts = [(a * r + b * col, c * r + d * col) for r, col in cells]
    r0 = min(r for r, _ in pts)
    c0 = min(col for _, col in pts)
    return tuple(sorted((r - r0, col - c0) for r, col in pts))


def _ours_d4_key(cells: Any) -> tuple:
    """Shape key that ignores position, rotation and reflection."""
    return min(_ours_norm(cells, t) for t in range(8))


def _ours_d4_relation(cells_a: Any, cells_b: Any) -> str:
    """How shape b relates to shape a ('same', 'rotated 90', ...), or '' when they differ."""
    base = _ours_norm(cells_b)
    for t in range(8):
        if _ours_norm(cells_a, t) == base:
            return _OURS_D4_NAMES[t]
    return ""


def _ours_component(grid: Any, start: tuple, seen: set, limit: int = _OURS_OBJ_MAX_PX) -> tuple:
    """(colour, cells) of the 4-connected same-colour area at `start`; cells is None when it has over `limit` cells.
    The whole area is added to `seen` either way, so a large area is flooded once, not once per cell."""
    colour = grid[start[0]][start[1]]
    height, width = len(grid), len(grid[0])
    cells = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells and grid[nr][nc] == colour:
                cells.add((nr, nc))
                stack.append((nr, nc))
    seen.update(cells)
    return int(colour), (frozenset(cells) if len(cells) <= limit else None)


def _ours_box(cells: Any) -> tuple:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _ours_obj_text(colour: int, cells: Any) -> str:
    r0, c0, r1, c1 = _ours_box(cells)
    size = f"{r1 - r0 + 1}x{c1 - c0 + 1}"
    if len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1):
        size += f" ({len(cells)} px)"
    return f"{_OURS_COLOUR_CHARS[max(0, min(15, colour))]} {size}"


'''
P23_FN = '''def _ours_changed_objects(grid: Any, changed: Any) -> list:
    """The objects (colour, cells) of one board that contain a changed cell. The board's most common colour (the
    floor) and areas of over _OURS_OBJ_MAX_PX cells are not objects."""
    counts: dict = {}
    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    floor = max(counts, key=counts.get)
    seen: set = set()
    out = []
    for cell in sorted(changed):
        if cell in seen or grid[cell[0]][cell[1]] == floor:
            continue
        colour, cells = _ours_component(grid, cell, seen)
        if cells is not None:
            out.append((colour, cells))
    return out


def _ours_move_text(dr: int, dc: int) -> str:
    parts = []
    if dr:
        parts.append(f"{'down' if dr > 0 else 'up'} {abs(dr)}")
    if dc:
        parts.append(f"{'right' if dc > 0 else 'left'} {abs(dc)}")
    return " ".join(parts) or "in place"


def _ours_object_diff(before: Any, after: Any) -> dict:
    """Object-level difference between two boards of the same size: moved, appeared, vanished, recoloured, reshaped.

    Objects are 4-connected same-colour areas of up to _OURS_OBJ_MAX_PX cells that contain a changed cell. `other`
    counts changed cells that belong to no listed object (large areas such as the floor under a moved object are not
    counted, as they are covered by the object's old or new position)."""
    changed = [(r, c) for r, (row_b, row_a) in enumerate(zip(before, after)) if row_b != row_a
               for c, (vb, va) in enumerate(zip(row_b, row_a)) if vb != va]
    result = {"changed": len(changed), "moved": [], "appeared": [], "vanished": [], "recoloured": [],
              "turned": [], "reshaped": [], "other": 0}
    if not changed:
        return result
    old = _ours_changed_objects(before, changed)
    new = _ours_changed_objects(after, changed)
    if len(old) + len(new) > 120:  # a scroll of a textured board or a full redraw: too many objects to pair or list
        result["other"] = len(changed)
        result["crowded"] = True
        return result
    covered = set()
    for _, cells in old + new:
        covered |= cells
    result["other"] = sum(1 for cell in changed if cell not in covered)
    # recoloured: the same cells, another colour
    by_cells = {cells: i for i, (_, cells) in enumerate(new)}
    used_old, used_new = set(), set()
    for i, (colour, cells) in enumerate(old):
        j = by_cells.get(cells)
        if j is not None and j not in used_new:
            result["recoloured"].append((colour, new[j][0], cells))
            used_old.add(i)
            used_new.add(j)
    # moved (the same colour and shape elsewhere), then turned (the same colour and shape up to rotation or
    # reflection: a sprite that turned as it moved); nearest first
    tops_old = [_ours_box(cells)[:2] for _, cells in old]
    tops_new = [_ours_box(cells)[:2] for _, cells in new]
    for kind, keyf in (("moved", _ours_norm), ("turned", _ours_d4_key)):
        buckets: dict = {}
        for j, (colour, cells) in enumerate(new):
            if j not in used_new:
                buckets.setdefault((colour, len(cells), keyf(cells)), []).append(j)
        pairs = []
        for i, (colour, cells) in enumerate(old):
            if i in used_old:
                continue
            (r0, c0) = tops_old[i]
            for j in buckets.get((colour, len(cells), keyf(cells)), ()):
                r1, c1 = tops_new[j]
                if len(cells) <= 2 and abs(r1 - r0) + abs(c1 - c0) > 8:
                    continue  # a dot has no identity: far apart, it is one vanishing and another appearing
                pairs.append((abs(r1 - r0) + abs(c1 - c0), i, j, r1 - r0, c1 - c0))
        for _, i, j, dr, dc in sorted(pairs):
            if i in used_old or j in used_new:
                continue
            used_old.add(i)
            used_new.add(j)
            result[kind].append((old[i][0], old[i][1], new[j][1], dr, dc))
    # reshaped: the same colour, overlapping boxes (grew, shrank, rotated in place)
    for i, (colour, cells) in enumerate(old):
        if i in used_old:
            continue
        a0, b0, a1, b1 = _ours_box(cells)
        for j, (colour2, cells2) in enumerate(new):
            if j in used_new or colour2 != colour:
                continue
            p0, q0, p1, q1 = _ours_box(cells2)
            if p0 <= a1 + 1 and a0 <= p1 + 1 and q0 <= b1 + 1 and b0 <= q1 + 1:
                used_old.add(i)
                used_new.add(j)
                result["reshaped"].append((colour, cells, cells2))
                break
    # grew or shrank without its remaining cells changing: the object still has unchanged cells on the other board
    for objects, index_used, other, other_grid, forward in ((old, used_old, new, after, True),
                                                         (new, used_new, old, before, False)):
        for i, (colour, cells) in enumerate(objects):
            if i in index_used:
                continue
            anchor = next((cell for cell in cells if other_grid[cell[0]][cell[1]] == colour), None)
            if anchor is None:
                continue
            colour2, cells2 = _ours_component(other_grid, anchor, set())
            if cells2 is None or any(cells2 == c for _, c in other):
                continue
            index_used.add(i)
            result["reshaped"].append((colour, cells, cells2) if forward else (colour, cells2, cells))
    result["vanished"] = [old[i] for i in range(len(old)) if i not in used_old]
    result["appeared"] = [new[j] for j in range(len(new)) if j not in used_new]
    return result


def _ours_diff_phrases(diff: dict, limit: int = 4) -> list:
    """Short phrases for one object diff, most informative first, at most `limit` (+ a count of the rest)."""
    phrases = []
    groups: dict = {}
    for colour, cells, cells2, dr, dc in diff["moved"]:
        groups.setdefault((dr, dc), []).append((colour, cells, cells2))
    for (dr, dc), members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(members) >= 6:  # many objects with one displacement: the view scrolled or a group moved together
            box = _ours_box(set().union(*(m[2] for m in members)))
            phrases.append(f"{len(members)} objects moved {_ours_move_text(dr, dc)} together "
                           f"(now rows {box[0]}-{box[2]}, cols {box[1]}-{box[3]})")
            continue
        for colour, cells, cells2 in members:
            r0, c0 = _ours_box(cells)[:2]
            r1, c1 = _ours_box(cells2)[:2]
            phrases.append(f"{_ours_obj_text(colour, cells)} moved {_ours_move_text(dr, dc)} ({r0},{c0})->({r1},{c1})")
    for colour, cells, cells2, dr, dc in diff["turned"]:
        r0, c0 = _ours_box(cells)[:2]
        r1, c1 = _ours_box(cells2)[:2]
        where = f"moved {_ours_move_text(dr, dc)} and " if dr or dc else ""
        phrases.append(f"{_ours_obj_text(colour, cells)} {where}{_ours_d4_relation(cells, cells2)} ({r0},{c0})->({r1},{c1})")
    for verb, objects in (("appeared at", diff["appeared"]), ("vanished from", diff["vanished"])):
        alike: dict = {}  # identical objects are one phrase
        for colour, cells in objects:
            alike.setdefault((colour, _ours_norm(cells)), []).append(cells)
        for (colour, _), members in alike.items():
            where = ", ".join(f"({_ours_box(c)[0]},{_ours_box(c)[1]})" for c in members[:4])
            where += ", ..." if len(members) > 4 else ""
            count = f"{len(members)} x " if len(members) > 1 else ""
            phrases.append(f"{count}{_ours_obj_text(colour, members[0])} {verb} {where}")
    recolours: dict = {}
    for colour, colour2, cells in diff["recoloured"]:
        recolours.setdefault((colour, colour2), []).append(cells)
    for (colour, colour2), members in recolours.items():
        r0, c0 = _ours_box(members[0])[:2]
        to = _OURS_COLOUR_CHARS[max(0, min(15, colour2))]
        if len(members) == 1:
            phrases.append(f"{_ours_obj_text(colour, members[0])} at ({r0},{c0}) turned {to}")
        else:
            phrases.append(f"{len(members)} {_OURS_COLOUR_CHARS[max(0, min(15, colour))]} objects turned {to} "
                           f"(first at ({r0},{c0}))")
    for colour, cells, cells2 in diff["reshaped"]:
        r0, c0 = _ours_box(cells)[:2]
        how = _ours_d4_relation(cells, cells2)
        how = how if how and how != "same" else f"became {_ours_obj_text(colour, cells2).split(' ', 1)[1]}"
        phrases.append(f"{_ours_obj_text(colour, cells)} at ({r0},{c0}) {how}")
    if len(phrases) > limit:
        phrases = phrases[:limit] + [f"{len(phrases) - limit} more object changes"]
    if diff.get("crowded"):
        phrases.append(f"{diff['other']} cells changed, too many objects to list")
    elif diff["other"]:
        phrases.append(f"{diff['other']} cells changed in large areas")
    return phrases


def _ours_dedupe(frames: Any, limit: int = 16) -> list:
    out = []
    for frame in frames:
        if not out or frame != out[-1]:
            out.append(frame)
    if len(out) > limit:  # keep the first and last, sample the rest evenly
        step = (len(out) - 1) / (limit - 1)
        out = [out[round(i * step)] for i in range(limit)]
    return out


def _ours_transient_phrases(before: Any, frames: Any, limit: int = 3) -> list:
    """What happened during an animation that the final board does not show: objects that moved and were back in
    place at the end, and objects that showed up and were gone at the end."""
    dedup = _ours_dedupe(frames)
    inner, final = dedup[:-1], dedup[-1] if dedup else None
    if not inner or not before or final is None or len(final) != len(before):
        return []

    def present(grid: Any, colour: int, cells: Any) -> bool:
        cell = min(cells)
        if grid[cell[0]][cell[1]] != colour:
            return False
        return _ours_component(grid, cell, set())[1] == cells

    farthest: dict = {}
    flashes: dict = {}
    for index, frame in enumerate(inner, 1):
        if len(frame) != len(before):
            continue
        diff = _ours_object_diff(before, frame)
        for colour, cells, _, dr, dc in diff["moved"]:
            if abs(dr) + abs(dc) > farthest.get(cells, (0,))[0]:
                farthest[cells] = (abs(dr) + abs(dc), dr, dc, colour, index)
        shown = [(colour, cells) for colour, cells in diff["appeared"]]
        shown += [(colour2, cells) for _, colour2, cells in diff["recoloured"]]
        for colour, cells in shown:
            key = (colour, _ours_box(cells)[:2])
            if key not in flashes:
                flashes[key] = (colour, cells, index)
    phrases = []
    for cells, (_, dr, dc, colour, index) in sorted(farthest.items(), key=lambda kv: -kv[1][0]):
        if present(final, colour, cells):  # back where it started
            r0, c0 = _ours_box(cells)[:2]
            phrases.append(f"{_ours_obj_text(colour, cells)} at ({r0},{c0}) moved as far as "
                           f"{_ours_move_text(dr, dc)} (frame {index}) and was back in place at the end")
    same: dict = {}
    for colour, cells, index in flashes.values():
        if not present(final, colour, cells):
            same.setdefault((colour, _ours_d4_key(cells)), []).append((cells, index))
    for (colour, _), members in same.items():
        cells, index = members[0]
        r0, c0 = _ours_box(cells)[:2]
        if len(members) == 1:
            phrases.append(f"{_ours_obj_text(colour, cells)} showed at ({r0},{c0}) from frame {index}, gone at the end")
        else:
            where = ", ".join(f"({_ours_box(c)[0]},{_ours_box(c)[1]})" for c, _ in members[:4])
            where += ", ..." if len(members) > 4 else ""
            phrases.append(f"{len(members)} x {_ours_obj_text(colour, cells)} showed at {where} from frame {index}, "
                           "gone at the end")
    if len(phrases) > limit:
        phrases = phrases[:limit] + [f"{len(phrases) - limit} more"]
    return phrases


def _ours_effect_lines(grids: list, names: list, animations: dict) -> list:
    """The object-change report for one executed sequence: `grids` holds the board before the sequence and after each
    action, `names` the action names, `animations` maps an action's index to its animation frames."""
    count = len(names)
    lines = []

    def one(label: str, before: Any, after: Any, frames: Any = None) -> None:
        diff = _ours_object_diff(before, after)
        phrases = _ours_diff_phrases(diff) or ["no change"]
        text = f"- {label}: " + "; ".join(phrases)
        if frames:
            during = _ours_transient_phrases(before, frames)
            if during:
                text += f". During its animation ({len(frames)} frames): " + "; ".join(during)
        lines.append(text + ".")

    if count <= 3:
        for i in range(count):
            one(names[i], grids[i], grids[i + 1], animations.get(i))
    else:
        one(f"net over the {count} actions", grids[0], grids[-1])
        for i in range(count):
            if animations.get(i) and len(lines) < 2:
                during = _ours_transient_phrases(grids[i], animations[i])
                if during:
                    lines.append(f"- during {names[i]} (action {i + 1} of {count}): " + "; ".join(during) + ".")
        one(f"last action ({names[-1]})", grids[-2], grids[-1])
    return lines


def _ours_effect_report(step_env: Any, summary: dict, history_entries: list, current_frame: Any) -> list:
    """P23's prompt lines for the last executed sequence (none after a level change or a game over)."""
    try:
        executed = int(summary.get("executed_count") or 0)
    except (TypeError, ValueError):
        return []
    if (executed <= 0 or summary.get("level_transition") or summary.get("game_over") or current_frame is None
            or len(history_entries) <= executed):
        return []
    entries = history_entries[-executed - 1:]
    if any(getattr(e, "frame", None) is None or e.frame.level != current_frame.level for e in entries):
        return []
    grids = [e.frame.grid for e in entries]
    if not grids[0] or any(len(g) != len(grids[0]) or len(g[0]) != len(grids[0][0]) for g in grids):
        return []
    names = [str(e.action or "?") for e in entries[1:]]
    animations: dict = {}
    start = summary.get("start_action_num")
    if step_env is not None and start is not None:
        for i in range(executed):  # the stored frames of each animated action; this executes nothing
            try:
                raw = step_env({"query": "animation", "action_num": int(start) + i})
            except Exception:
                continue
            record = raw.get("record") if isinstance(raw, dict) else None
            frames = record.get("frames") if isinstance(record, dict) else None
            if frames and len(frames) > 1 and all(len(f) == len(grids[0]) for f in frames):
                animations[i] = list(frames)
    return ["Object changes from your last actions (exact, computed by the harness from the frames; (row,col) is "
            "the top-left of an object's box):"] + _ours_effect_lines(grids, names, animations)


'''
P24_FN = '''def _ours_shape_match_line(grid: Any, min_px: int = 5, max_px: int = 49, max_groups: int = 10) -> str:
    """Objects on one board with the same shape up to rotation, reflection or colour (solid rectangles left out)."""
    seen: set = set()
    objects = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if (r, c) in seen:
                continue
            colour, cells = _ours_component(grid, (r, c), seen, limit=max_px + 1)
            if cells is None or not min_px <= len(cells) <= max_px:
                continue
            r0, c0, r1, c1 = _ours_box(cells)
            if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1):
                continue
            objects.append((colour, cells, r0, c0))
    groups: dict = {}
    for obj in objects:
        groups.setdefault(_ours_d4_key(obj[1]), []).append(obj)
    kept = []
    for members in groups.values():
        variants = {(colour, _ours_norm(cells)) for colour, cells, _, _ in members}
        if len(members) >= 2 and len(variants) >= 2:
            kept.append(members)
    if not kept:
        return ""
    kept.sort(key=lambda m: (-len({_ours_norm(c) for _, c, _, _ in m}), -len(m), m[0][2], m[0][3]))
    parts = []
    for n, members in enumerate(kept[:max_groups], 1):
        first = members[0]
        items = [f"{_OURS_COLOUR_CHARS[first[0]]} ({first[2]},{first[3]})"]
        for colour, cells, r0, c0 in members[1:8]:
            rel = _ours_d4_relation(first[1], cells)
            items.append(f"{_OURS_COLOUR_CHARS[colour]} ({r0},{c0})" + ("" if rel == "same" else f" {rel}"))
        if len(members) > 8:
            items.append(f"{len(members) - 8} more")
        parts.append(f"[{n}] {len(first[1])} px: " + ", ".join(items))
    more = f" ({len(kept) - max_groups} more groups not listed)" if len(kept) > max_groups else ""
    return ("Objects with the same shape up to rotation, reflection or colour (exact; non-rectangular objects of "
            f"{min_px}-{max_px} px; (row,col) is the top-left of each object's box; the rotation or mirror is "
            "relative to the first object in its group): " + "; ".join(parts) + more + ".")


'''
P23_NEW = P9_OLD + """            _p23_key = (self._session_runtime_dir, previous_step_summary.get("start_action_num"),
                        previous_step_summary.get("end_action_num"))
            if getattr(self, "_ours_p23_key", None) != _p23_key:  # ours P23: once per executed sequence
                self._ours_p23_key = _p23_key
                try:  # prompt building runs outside the analyzer's try: a bad entry must not end the game
                    lines.extend(_ours_effect_report(self._step_env_callback, previous_step_summary, history_entries,
                                                     current_frame))
                except Exception:
                    pass
"""
P24_OLD = "        hint_line = self._animation_hint_line(previous_step_summary, current_level)\n"
P24_NEW = """        _p24_key = (self._session_runtime_dir, current_level)
        if (current_frame is not None and getattr(self, "_ours_p24_key", None) != _p24_key
                and (not previous_step_summary or previous_step_summary.get("level_transition"))):
            self._ours_p24_key = _p24_key  # ours P24: once per level, at its first prompt
            try:
                _p24 = _ours_shape_match_line(current_frame.grid)
            except Exception:
                _p24 = ""
            if _p24:
                lines.append(_p24)
""" + P24_OLD


PATCHES.update({
    "P22": [(SANDBOX, P22_OLD, P22_NEW)],
    "P23": [(TOOL_AGENT, "<ONCE>def _empty_world_model(", OURS_OBJECTS_FN),
            (TOOL_AGENT, "def _empty_world_model(", P23_FN + "def _empty_world_model("),
            (TOOL_AGENT, P9_OLD, P23_NEW)],
    "P24": [(TOOL_AGENT, "<ONCE>def _empty_world_model(", OURS_OBJECTS_FN),
            (TOOL_AGENT, "def _empty_world_model(", P24_FN + "def _empty_world_model("),
            (TOOL_AGENT, P24_OLD, P24_NEW)],
    "P21": [(TOOL_AGENT, "def _empty_world_model(", P21_FN + "def _empty_world_model("),
            (TOOL_AGENT, P21_LEVEL_OLD, P21_LEVEL_NEW),
            (TOOL_AGENT, P21_CALL_OLD, P21_CALL_NEW),
            (TOOL_AGENT, P21_TURN_OLD, P21_TURN_NEW),
            (TOOL_AGENT, P21_YIELD_OLD, P21_YIELD_NEW)],
    "P11": [(UTILS_COMPAT, P11_IMPORT_OLD, P11_IMPORT_NEW), (UTILS_COMPAT, P11_OLD, P11_NEW)],
    "P12": [(TOOL_AGENT, P12_OLD, P12_NEW)],
    "P13": [(PROMPTS, PROMPTS_SCORE_OLD, PROMPTS_SCORE_NEW)],
    "P14": [(TOOL_AGENT, "def _empty_world_model(", P14_FN + "def _empty_world_model("),
            (TOOL_AGENT, P14_OLD, P14_NEW)],
    "P15": [(TOOL_AGENT, P15_OLD, P15_NEW)],
    "P16": [(TOOL_AGENT, P16_OLD, P16_NEW)],
    "P17": [(TOOL_AGENT, P17_VIEW_OLD, P17_VIEW_NEW), (TOOL_AGENT, P17_HINT_OLD, P17_HINT_NEW)],
    "P18": [(TOOL_AGENT, P18_OLD, P18_NEW)],
    "P20": [(SOLVER, "<EOF>", P20_NEW), (TOOL_AGENT, P20_PROMPT_OLD, P20_PROMPT_NEW)],
    "P19": [(TOOL_AGENT, "def _empty_world_model(", P19_FN + "def _empty_world_model("),
            (TOOL_AGENT, P19_OLD, P19_NEW)],
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
            if old.startswith("<ONCE>"):  # a block shared by several patches: insert it before the anchor unless present
                if new not in text:
                    anchor = old[len("<ONCE>"):]
                    if text.count(anchor) != 1:
                        raise RuntimeError(f"patch {name}: expected exactly one match in {rel}, found {text.count(anchor)}")
                    path.write_text(text.replace(anchor, new + anchor, 1), encoding="utf-8")
                applied.append(f"{name}:{rel}")
                continue
            if old == "<EOF>":  # append to the end of the file, once
                if new in text:
                    raise RuntimeError(f"patch {name}: already appended to {rel}")
                path.write_text(text + new, encoding="utf-8")
                applied.append(f"{name}:{rel}")
                continue
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
