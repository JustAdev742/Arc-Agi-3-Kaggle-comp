"""Whole-game mechanism counts from Franzen's per-game solver_analysis/*.html (every turn of every game).

    .venv/bin/python -I scripts/mech_wholegame.py DIR_WITH_solver_analysis [...]

Pull the files first (about 2 MB per game):
    kaggle kernels output OWNER/KERNEL -p DIR --file-pattern 'solver_analysis/.*\\.html$' -q -o
"""
import collections
import glob
import html
import re
import sys

SEC = re.compile(r"^\[(THINKING|TOOL CALL|TOOL RESULT|ANALYZER STATUS|SYSTEM PROMPT|USER PROMPT|MODEL RESPONSE META)[^\n]*\]\n"
                 r"(.*?)(?=^\[(?:THINKING|TOOL CALL|TOOL RESULT|ANALYZER STATUS|SYSTEM PROMPT|USER PROMPT|MODEL RESPONSE META)"
                 r"[^\n]*\]\n|\Z)", re.S | re.M)
CALLS = {"search(": r"\bsearch\(", "run_plan(": r"\brun_plan\(", "mem[": r"\bmem\[", "effects(": r"\beffects\(",
         "frame_diff(": r"\bframe_diff\(", "RESET": r"\bRESET\b", "UNDO": r"\bUNDO\b"}
TEXT = {"budget bar": "Budget bar (", "ledger": "Ledger of exact facts from the recorded boards",
        "fresh-start line": "Conversation history was cleared at step"}
ERRS = {"NameError": "NameError", "Traceback": "Traceback", "timed out": "timed out"}
FRESH = re.compile(r"ours_fresh_start: conversation history cleared at step (\d+) \(fresh start (\d+) of at most (\d+) "
                   r"on level (\d+)\) after (\d+) generated tokens and ([\d.]+) min")


def text_of(path):
    s = open(path, errors="replace").read()
    return html.unescape(re.sub(r"<[^>]+>", "\n", s))


def run(d):
    files = sorted(glob.glob(d.rstrip("/") + "/solver_analysis/*.html"))
    tot = collections.Counter()
    status = collections.Counter()
    games_with = collections.Counter()
    fresh = []
    rep_turns = turns = 0
    for f in files:
        game = f.split("/")[-1][:4]
        t = text_of(f)
        secs = [(m.group(1), m.group(2)) for m in SEC.finditer(t)]
        calls = [b for k, b in secs if k == "TOOL CALL"]
        results = [b for k, b in secs if k == "TOOL RESULT"]
        thinks = [b for k, b in secs if k == "THINKING"]
        users = [b for k, b in secs if k in ("USER PROMPT", "SYSTEM PROMPT")]
        stats = [b.split("\n", 1)[0] for k, b in secs if k == "ANALYZER STATUS"]
        tot["tool calls"] += len(calls)
        tot["turns"] += len(thinks)
        for k, pat in CALLS.items():
            n = sum(len(re.findall(pat, c)) for c in calls)
            tot[k] += n
            games_with[k] += n > 0
        for k, txt in ERRS.items():
            n = sum(txt in r for r in results)
            tot[k] += n
        for k, txt in TEXT.items():
            n = sum(txt in u for u in users)
            tot[k] += n
            games_with[k] += n > 0
        for s in stats:
            status[s.split(":")[0].split(" ")[0]] += 1
            m = FRESH.search(s)
            if m:
                fresh.append((game, int(m.group(4)), int(m.group(2)), int(m.group(1)), int(m.group(5)), float(m.group(6))))
        seg = collections.Counter(a + "\x00" + b for a, b in zip(thinks, calls))
        turns += len(thinks)
        rep_turns += sum(v - 1 for v in seg.values() if v > 1)
    print(f"== {d}: {len(files)} games, {tot['turns']} turns, {tot['tool calls']} tool calls, exact repeated turns "
          f"{rep_turns} ({rep_turns / max(turns, 1):.1%})")
    print("  calls (total / games): " + ", ".join(f"{k} {tot[k]}/{games_with[k]}" for k in CALLS))
    print("  system/user prompt sections with (total / games): " + ", ".join(f"{k} {tot[k]}/{games_with[k]}" for k in TEXT))
    print("  tool results with: " + ", ".join(f"{k} {tot[k]}" for k in ERRS))
    print("  analyzer statuses: " + ", ".join(f"{k} {v}" for k, v in status.most_common(12)))
    if fresh:
        print(f"  fresh starts: {len(fresh)} in {len({g for g, *_ in fresh})} games")
        for g, lv, n, step, toks, mins in fresh:
            print(f"    {g} level {lv} #{n} at step {step} after {toks} tokens / {mins:.1f} min")


for d in sys.argv[1:]:
    run(d)
