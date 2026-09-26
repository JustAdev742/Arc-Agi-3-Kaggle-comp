"""Dump belief snapshots for each stuck level (and optionally a given level) of each game in both runs."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from parse import load, summary, RUNS

OUT = Path(__file__).parent / 'dumps'
OUT.mkdir(exist_ok=True)
games = sys.argv[1:] or sorted(summary('A').keys())


def clip(s, n):
    s = s.strip()
    return s if len(s) <= n else s[:n] + ' …[+%d]' % (len(s) - n)


for g in games:
    lines = []
    for rk in RUNS:
        S = summary(rk)[g]
        T = load(rk, g)
        t0 = T[0]['t']
        lc = S['levels_completed']
        stuck = lc + 1
        done = S['level_done_s']
        lines.append(f"\n################ {g} run {rk}: levels {lc}/{S['levels_total']} actions {S['level_actions']} base {S['baselines']} done_min {[round(x/60,1) for x in done]}")
        # previous level's end: last 2 turns' executed actions + first recap on stuck level
        tl = [(i, x) for i, x in enumerate(T) if x['level'] == stuck]
        if not tl:
            lines.append('no turns on stuck level')
            continue
        lines.append(f"stuck L{stuck}: turns {len(tl)} calls {sum(x['responses'] for _, x in tl)}")
        # previous level: note at its last turn
        prev = [x for x in T if x['level'] == stuck - 1]
        if prev:
            lines.append('PREV-LEVEL LAST NOTE: ' + clip(prev[-1]['note'], 1500))
        # timeline of actions on the stuck level
        tlparts = []
        for i, x in tl:
            m = (x['t'] - t0) / 60
            if x['executed'] and x['action'] != T[i - 1]['action']:
                tlparts.append(f"{m:.0f}m[{clip(x['executed'], 160)}]{' GAMEOVER' if x['gameover'] else ''}")
        lines.append('ACTIONS (executed, reported in next prompt): ' + ' | '.join(tlparts[:80]))
        if tl[-1][1]['acted'] is False:
            pass
        # P24 line at level start
        if tl[0][1]['p24']:
            lines.append('P24: ' + clip(tl[0][1]['p24'], 600))
        n = len(tl)
        picks = sorted(set([0, 1, n // 4, n // 2, (3 * n) // 4, n - 2, n - 1]))
        for k in picks:
            if k < 0 or k >= n:
                continue
            i, x = tl[k]
            m = (x['t'] - t0) / 60
            lines.append(f"\n--- turn {k+1}/{n} @ {m:.1f} min (level +{m - (done[stuck-2]/60 if stuck >= 2 else 0):.1f}) action#{x['action']} calls {x['responses']} acted {x['acted']}")
            if x['p23']:
                lines.append('P23: ' + clip(x['p23'].replace('\n', ' '), 500))
            if x['note']:
                lines.append('NOTE: ' + clip(x['note'], 1800))
            if x['thinking']:
                lines.append('THINK[0]: ' + clip(x['thinking'][0], 900))
                if len(x['thinking']) > 1:
                    lines.append('THINK[-1]: ' + clip(x['thinking'][-1], 1200))
    (OUT / f'{g}.txt').write_text('\n'.join(lines))
    print(g, len('\n'.join(lines)))
