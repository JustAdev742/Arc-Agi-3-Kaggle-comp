"""Mechanism counts of our harness patches in a run's artifacts (last-call logs + benchmark.json).

    .venv/bin/python -I scripts/mech_lastcall.py runs/<run>/kernel-output [...]

The last-call logs (prompts/*.log) hold each game's final request: the turns still in context, not the whole game.
So the per-turn counts are a sample of the late game; benchmark.json gives exact RESET/action counts for the whole game.
"""
import collections
import glob
import json
import re
import sys

SEG = re.compile(r"^\[ASSISTANT\]\n(.*?)(?=^\[(?:TOOL RESULT|USER|SYSTEM|ASSISTANT\]|MODEL RESPONSE META|TURN TRANSCRIPT|ANALYZER STATUS))", re.S | re.M)
CALL = re.compile(r"^\[ASSISTANT TOOL CALL[^\n]*\n(.*?)(?=^\[(?:TOOL RESULT|USER|SYSTEM|ASSISTANT|MODEL RESPONSE META|TURN TRANSCRIPT|ANALYZER STATUS))", re.S | re.M)
RESULT = re.compile(r"^\[TOOL RESULT[^\n]*\n(.*?)(?=^\[(?:USER|SYSTEM|ASSISTANT|MODEL RESPONSE META|TURN TRANSCRIPT|ANALYZER STATUS|TOOL RESULT))", re.S | re.M)

TEXT = {  # text each patch puts in front of the model (docs/research/beat-tufa/patch-*.md)
    "budget bar": "Budget bar (",
    "ledger": "Ledger of exact facts from the recorded boards",
    "mem doc": "`mem` is a dict kept across `python` calls on the current level",
    "search doc": "searches a model of the game that you write; it never acts",
    "effects doc": "returns the action-effect table of the current level",
    "fresh start": "Conversation history was cleared at step",
    "retained fns": "previously retained functions are still available",
}
CALLS = {"search(": r"\bsearch\(", "run_plan(": r"\brun_plan\(", "mem[": r"\bmem\[", "mem.": r"\bmem\.(?:get|setdefault|update|keys|items|pop)\(",
         "effects(": r"\beffects\(", "frame_diff(": r"\bframe_diff\(", "RESET": r"\bRESET\b", "UNDO": r"\bUNDO\b"}
ERRORS = {"NameError": "NameError", "Traceback": "Traceback", "timed out": "timed out", "SyntaxError": "SyntaxError"}


def run(d):
    logs = sorted(glob.glob(d + "/prompts/*.log"))
    seg_n = seg_rep = 0
    text_games = collections.Counter()
    text_hits = collections.Counter()
    calls = collections.Counter()
    call_games = collections.Counter()
    errs = collections.Counter()
    ncalls = 0
    for f in logs:
        s = open(f, errors="replace").read()
        segs = [re.sub(r"^id: \S+\n", "", m.group(1), flags=re.M) for m in SEG.finditer(s)]
        c = collections.Counter(segs)
        seg_n += len(segs)
        seg_rep += sum(v - 1 for v in c.values() if v > 1)
        for k, t in TEXT.items():
            n = s.count(t)
            text_hits[k] += n
            text_games[k] += n > 0
        code = [m.group(1) for m in CALL.finditer(s)]
        ncalls += len(code)
        for k, pat in CALLS.items():
            n = sum(len(re.findall(pat, x)) for x in code)
            calls[k] += n
            call_games[k] += n > 0
        for r in RESULT.finditer(s):
            for k, t in ERRORS.items():
                errs[k] += t in r.group(1)
    print(f"== {d}: {len(logs)} last-call logs, {seg_n} assistant turns in context, exact repeats {seg_rep} "
          f"({seg_rep / max(seg_n, 1):.0%}), {ncalls} tool calls")
    print("  patch text (occurrences / games): " + ", ".join(f"{k} {text_hits[k]}/{text_games[k]}" for k in TEXT))
    print("  calls (occurrences / games): " + ", ".join(f"{k} {calls[k]}/{call_games[k]}" for k in CALLS))
    print("  tool results with: " + ", ".join(f"{k} {errs[k]}" for k in ERRORS))
    try:
        b = json.load(open(d + "/benchmark.json"))
    except OSError:
        return
    resets = undos = acts = 0
    per = []
    for g in b["game_runs"]:
        ids = [h["action"]["id"] for h in g["history"]]
        r = ids.count("RESET")
        resets += r
        undos += ids.count("UNDO") + ids.count("ACTION7")
        acts += len(ids)
        if r:
            per.append(f"{g['game_id'][:4]}:{r}")
    print(f"  benchmark.json: {acts} actions, RESET {resets} ({', '.join(per)}), UNDO/ACTION7 {undos}")


for d in sys.argv[1:]:
    run(d.rstrip("/"))
