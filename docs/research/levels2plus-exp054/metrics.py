"""Per (run, game, level): minutes, calls, actions vs baseline, idle stretches, P23 usage."""
import json, re, statistics, sys
from collections import defaultdict
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from parse import load, summary, bench, RUNS

GAMES = sorted(summary('A').keys())
OUT = __import__('pathlib').Path(__file__).parent


def lvl_score(b, a):
    return min((b / a) ** 2, 1.15) if a else 0.0


P23_REF = re.compile(r"harness|object[- ]change|object report|the report|report says|report shows|reported|per the report|changes report", re.I)


def p23_mentions(turn):
    """strict: the first reasoning block names the report; loose: also quotes a coordinate pair from it."""
    if not turn['p23'] or not turn['thinking']:
        return (False, False)
    th = turn['thinking'][0]
    strict = bool(P23_REF.search(th))
    coords = set(re.findall(r"\((\d+),(\d+)\)", turn['p23']))
    loose = strict
    for r, c in coords:
        if re.search(rf"\(\s*{r}\s*,\s*{c}\s*\)", th) or re.search(rf"row\s*{r}\b.*?col\s*{c}\b", th):
            loose = True
            break
    # phrases characteristic of the report
    if re.search(r"moved (up|down|left|right) \d+|appeared at|vanished from|became \d+x\d+|changed colour|showed at|back in place|moved together", th):
        loose = True
    return (strict, loose)


rows = []
for rk in RUNS:
    S = summary(rk)
    B = bench(rk)
    for g in GAMES:
        s = S[g]
        T = load(rk, g)
        t0 = T[0]['t']
        hist = B[g]['history']
        # offset: run-seconds at transcript t0 (first turn start). next turn after an acting turn starts ~ at last action time
        offs = []
        for i, x in enumerate(T[:-1]):
            if x['acted']:
                a_last = T[i + 1]['action'] - 1  # 1-based index of last executed action
                if 0 < a_last <= len(hist):
                    offs.append(hist[a_last - 1]['wallclock_seconds'] - (T[i + 1]['t'] - t0))
        off = statistics.median(offs) if offs else 0.0
        end_s = s['wall_s']
        done = s['level_done_s']
        nlev = s['levels_total']
        lc = s['levels_completed']
        for L in range(1, min(lc + 1, nlev) + 1):
            start = 0.0 if L == 1 else done[L - 2]
            stop = done[L - 1] if L <= lc else end_s
            tl = [x for x in T if x['level'] == L]
            calls = sum(x['responses'] for x in tl)
            acting = sum(1 for x in tl if x['acted'])
            # idle stretches: times of acting turns' ends ~ next turn start
            act_times = [start]
            for i, x in enumerate(T[:-1]):
                if x['level'] == L and x['acted']:
                    act_times.append(T[i + 1]['t'] - t0 + off)
            act_times.append(stop)
            gaps = [(b - a) / 60 for a, b in zip(act_times, act_times[1:])]
            longest = max(gaps) if gaps else 0
            idle_tail = (stop - act_times[-2]) / 60 if L > lc else None  # minutes since last action at run end
            p23 = [x for x in tl if x['p23']]
            ment = [p23_mentions(x) for x in p23]
            acts = s['level_actions'][L - 1] if L - 1 < len(s['level_actions']) else 0
            base = s['baselines'][L - 1]
            rows.append(dict(run=rk, game=g, level=L, solved=L <= lc, start_min=round(start / 60, 1),
                             minutes=round((stop - start) / 60, 1), calls=calls, turns=len(tl), acting_turns=acting,
                             actions=acts, baseline=base, score=round(lvl_score(base, acts), 3) if L <= lc else 0.0,
                             longest_idle=round(longest, 1), idle_tail=round(idle_tail, 1) if idle_tail is not None else None,
                             p23_reports=len(p23), p23_strict=sum(m[0] for m in ment), p23_loose=sum(m[1] for m in ment),
                             yields=sum(x['yielded'] for x in tl), lengthcuts=sum(x['lengthcut'] for x in tl),
                             gameovers=sum(1 for x in tl if x['gameover']), offset_s=round(off, 1)))
json.dump(rows, open(OUT / 'levels.json', 'w'), indent=0)
for r in rows:
    print(r['run'], r['game'], 'L%d' % r['level'], 'solved' if r['solved'] else 'STUCK', 'start', r['start_min'], 'min', r['minutes'],
          'calls', r['calls'], 'acting', r['acting_turns'], 'acts', r['actions'], '/', r['baseline'], 'sc', r['score'],
          'idle', r['longest_idle'], 'tail', r['idle_tail'], 'p23', r['p23_reports'], r['p23_strict'], r['p23_loose'],
          'go', r['gameovers'], 'len', r['lengthcuts'], 'off', r['offset_s'])
