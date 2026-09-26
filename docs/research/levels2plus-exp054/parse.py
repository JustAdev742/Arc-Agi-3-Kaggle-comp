"""Parse exp-054 pair transcripts into per-turn records (read-only)."""
import json, re, sys
from pathlib import Path

ROOT = Path('/home/user/Arc-Agi-3-Kaggle-comp/runs')
RUNS = {'A': 'exp054-fix-kv775-obj', 'B': 'exp054r-fix-kv775-obj-r2'}
TURN_RE = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d):(\d\d):(\d\d) \| tool-agent ---$", re.M)
LEVEL_RE = re.compile(r"^Current state: step (\d+), level (\d+)", re.M)
NOTE_RE = re.compile(r"Working world model carried from earlier turns:\n(.*?)end of world model\.", re.S)
SEC_RE = re.compile(r"^\[(SYSTEM PROMPT|USER PROMPT|THINKING|ASSISTANT|MODEL RESPONSE META|ANALYZER STATUS|TOOL CALL: python|TOOL RESULT: python)\]\s*$", re.M)


def sections(body):
    ms = list(SEC_RE.finditer(body))
    out = []
    for i, m in enumerate(ms):
        out.append((m.group(1), body[m.end(): ms[i + 1].start() if i + 1 < len(ms) else len(body)]))
    return out


def turns_of(text):
    heads = list(TURN_RE.finditer(text))
    out = []; day = 0; last = None
    for i, m in enumerate(heads):
        body = text[m.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        t = int(m.group(3)) * 3600 + int(m.group(4)) * 60 + int(m.group(5))
        if last is not None and t + day < last - 3600:
            day += 86400
        last = t + day
        secs = sections(body)
        user = next((s for k, s in secs if k == 'USER PROMPT'), '')
        lv = LEVEL_RE.search(user)
        note = NOTE_RE.search(user)
        thinking = [s.strip() for k, s in secs if k == 'THINKING']
        assistant = [s.strip() for k, s in secs if k == 'ASSISTANT']
        calls = [s for k, s in secs if k == 'TOOL CALL: python']
        results = [s for k, s in secs if k == 'TOOL RESULT: python']
        p23 = ''
        mm = re.search(r"^Object changes from your last .*?\n((?:- .*\n?)+)", user, re.M)
        if mm:
            p23 = mm.group(0)
        p24 = re.search(r"^Harness shape matches.*$", user, re.M)
        ex = re.search(r"^Executed actions: (.*)$", user, re.M)
        out.append(dict(step=int(m.group(1)), action=int(m.group(2)), t=t + day,
                        level=int(lv.group(2)) if lv else None, note=note.group(1).strip() if note else '',
                        user=user, thinking=thinking, assistant=assistant, calls=calls, results=results,
                        responses=body.count('[MODEL RESPONSE META]'), p23=p23, p24=p24.group(0) if p24 else '',
                        executed=ex.group(1) if ex else '', newlevel='You have progressed to a new level!' in user,
                        gameover='The game is over.' in user or 'GAME_OVER' in user,
                        yielded='turn_time_budget' in body, lengthcut=body.count('finish_reason: length')))
    # acted: the next turn's action number is higher
    for i, x in enumerate(out):
        x['acted'] = i + 1 < len(out) and out[i + 1]['action'] > x['action']
    return out


def load(run_key, game):
    d = ROOT / RUNS[run_key] / 'kernel-output' / 'transcripts'
    f = next(d.glob(f'{game}-*_p0.txt'))
    return turns_of(f.read_text(errors='replace'))


def summary(run_key):
    s = json.load(open(ROOT / RUNS[run_key] / 'summary.json'))
    return {g['game_id']: g for g in s['results']}


def bench(run_key):
    b = json.load(open(ROOT / RUNS[run_key] / 'kernel-output' / 'benchmark.json'))
    return {g['game_id'].split('-')[0]: g for g in b['game_runs']}
