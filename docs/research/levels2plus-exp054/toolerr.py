import re, sys, collections, json
sys.path.insert(0, '/home/user/Arc-Agi-3-Kaggle-comp/docs/research/levels2plus-exp054')
from parse import turns_of, ROOT, RUNS
cats = collections.Counter(); ex = {}
nturns = nerr = 0
kinds = collections.Counter()
for rk, name in RUNS.items():
    for f in sorted((ROOT / name / 'kernel-output' / 'transcripts').glob('*_p0.txt')):
        T = turns_of(f.read_text(errors='replace'))
        for x in T:
            nturns += 1
            errs = []
            for r in x['results']:
                m = re.search(r'"error":\s*"((?:[^"\\]|\\.)*)"', r)
                if m: errs.append(m.group(1))
                elif 'Traceback' in r: errs.append(r[r.find('Traceback'):][:300])
            if errs:
                nerr += 1
                e = errs[-1]
                m2 = re.search(r'(\w+(?:Error|Exception))\b', e)
                k = m2.group(1) if m2 else e[:50]
                kinds[k] += 1
                ex.setdefault(k, []).append((f.name[:4], rk, x['step'], e[:220].replace('\\n', ' | ')))
print('turns', nturns, 'turns with an error in the last-or-any result', nerr)
for k, n in kinds.most_common(25):
    print(n, k)
    for e in ex[k][:2]: print('     ', e)
