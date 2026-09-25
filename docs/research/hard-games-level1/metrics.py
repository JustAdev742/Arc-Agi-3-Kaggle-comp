import re, sys, json
from pathlib import Path
sys.path.insert(0, '/tmp/claude-0/-home-user-Arc-Agi-3-Kaggle-comp/d342458e-03bd-545b-8a6d-06bca061963e/scratchpad/research/hg')
from extract import turns_of, RUNS, HARD, ROOT
from collections import Counter
rows=[]
summ = {r: {g['game_id'][:4]: g for g in json.load(open(ROOT/r/'summary.json'))['results']} for r in RUNS}
for g in (sys.argv[1:] or HARD):
    for r in RUNS:
        f = next((ROOT/r/'kernel-output'/'transcripts').glob(f'{g}-*.txt'))
        T = turns_of(f.read_text(errors='replace'))
        t0 = T[0]['t']
        acts=[]; gameover=0; yields=0; execd=0; resp=0; l1=[]
        prev_level = 1
        first_try = {}
        for i,x in enumerate(T):
            u = x['user']
            m = re.search(r'^Executed actions: (.*)$', u, re.M)
            if m and prev_level == 1 and (i == 0 or x["action"] != T[i-1]["action"]):
                a = [s.strip().rstrip('.') for s in re.split(r', (?=[A-Z])', m.group(1))]
                for s in a:
                    k = s.split('(')[0]
                    first_try.setdefault(k, ((x['t']-t0)/60, len(acts)+1))
                    acts.append(k)
            if prev_level == 1 and 'The game is over.' in u and (i == 0 or x['action'] != T[i-1]['action']): gameover += 1
            if x['level'] == 1:
                l1.append(x)
                resp += x['responses']
                if 'turn_time_budget' in x['body']: yields += 1
                if 'message: Step executed.' in x['body']: execd += 1
            prev_level = x['level'] or prev_level
        # longest no-action stretch on L1
        longest=0; last=t0; lastact=None
        for i,x in enumerate(l1):
            if lastact is None or x['action']!=lastact:
                if lastact is not None: longest=max(longest,(x['t']-last)/60)
                last=x['t']; lastact=x['action']
        end_t = l1[-1]['t'] if l1 else t0
        nxt = [x for x in T if x['level'] and x['level']>=2]
        if nxt: end_t = nxt[0]['t']
        else: end_t = T[-1]['t'] + 180
        longest=max(longest,(end_t-last)/60)
        s = summ[r][g]
        solved = s['levels_completed']>=1
        rows.append(dict(game=g, run=r, solved=solved, l1_actions=s['level_actions'][0], base=s['baselines'][0],
                         l1_min=round((end_t-t0)/60,1), l1_turns=len(l1), l1_resp=resp, yields=yields, execd=execd,
                         gameovers=gameover, longest_idle=round(longest,1), acts=dict(Counter(acts)),
                         first=dict((k,(round(v[0]),v[1])) for k,v in first_try.items())))
for r in rows:
    print(json.dumps(r))
