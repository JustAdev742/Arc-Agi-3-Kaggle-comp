import re, sys, json
from pathlib import Path
sys.path.insert(0, '/home/user/Arc-Agi-3-Kaggle-comp/scripts')
from taaf_mechanisms import TURN_RE, LEVEL_RE, NOTE_RE
ROOT = Path('/home/user/Arc-Agi-3-Kaggle-comp/runs')
RUNS = ['exp032-anim-flashnext','exp042-ours-g','exp042r-ours-g-r2','exp048-fix-gate-kv65','exp045-gate']
HARD = ['sk48','bp35','g50t','dc22','tn36','ls20','tr87','su15','m0r0']
OUT = Path(sys.argv[1]) if __name__ == "__main__" else None

def turns_of(text):
    heads = list(TURN_RE.finditer(text))
    out=[]; day=0; last=None
    for i,m in enumerate(heads):
        body = text[m.end(): heads[i+1].start() if i+1 < len(heads) else len(text)]
        t = int(m.group(3))*3600+int(m.group(4))*60+int(m.group(5))
        if last is not None and t+day < last-3600: day += 86400
        last = t+day
        lv = LEVEL_RE.search(body)
        note = NOTE_RE.search(body)
        up = body.split('[MODEL RESPONSE META]')[0]
        up = up.split('[USER PROMPT]')[-1]
        out.append(dict(step=int(m.group(1)), action=int(m.group(2)), t=t+day, level=int(lv.group(1)) if lv else None,
                        note=note.group(1).strip() if note else '', body=body, user=up,
                        responses=body.count('[MODEL RESPONSE META]')))
    return out

def assistant_blocks(body):
    return re.findall(r'^\[ASSISTANT\]\n(.*?)(?=^\[TOOL CALL|^\[MODEL RESPONSE META|\Z)', body, re.M|re.S)

def executed(body):
    # executed actions reported in tool results
    return re.findall(r"'executed_actions': \[(.*?)\]", body)

for g in (HARD if __name__ == "__main__" else []):
    lines=[]
    for r in RUNS:
        f = next((ROOT/r/'kernel-output'/'transcripts').glob(f'{g}-*.txt'))
        T = turns_of(f.read_text(errors='replace'))
        t0 = T[0]['t']
        l1_end = next((x for x in T if x['level'] and x['level']>=2), None)
        l1_turns = [x for x in T if x['level']==1]
        resp_l1 = sum(x['responses'] for x in l1_turns)
        lines.append(f"\n######## {g} {r}: turns={len(T)} responses={sum(x['responses'] for x in T)} L1 turns={len(l1_turns)} L1 responses={resp_l1} "
                     f"L1 end={'%.1f min at action %d'%((l1_end['t']-t0)/60, l1_end['action']) if l1_end else 'never'} last action={T[-1]['action']}")
        # timeline compact
        tl = ' '.join(f"{(x['t']-t0)/60:.0f}m:a{x['action']}:L{x['level']}" for x in T)
        lines.append('TIMELINE ' + tl)
        for mark in (15, 60, 120):
            cands = [x for x in T if (x['t']-t0)/60 <= mark]
            x = cands[-1]
            lines.append(f"--- @{mark}min -> turn step={x['step']} action={x['action']} level={x['level']} t={(x['t']-t0)/60:.1f}")
            if x['note']:
                lines.append('NOTE: ' + x['note'][:1800])
            ab = []
            for y in reversed(cands):
                ab = assistant_blocks(y['body'])
                if ab: break
            if ab:
                lines.append('LAST ASSISTANT: ' + ab[-1].strip()[:1500])
    (OUT/f'{g}.txt').write_text('\n'.join(lines))
    print(g, 'done')
