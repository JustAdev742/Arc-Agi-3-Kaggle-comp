import sys, re
sys.path.insert(0, '/tmp/claude-0/-home-user-Arc-Agi-3-Kaggle-comp/d342458e-03bd-545b-8a6d-06bca061963e/scratchpad/research/hg')
from extract import turns_of, ROOT, assistant_blocks
run, game, lo, hi = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
width = int(sys.argv[5]) if len(sys.argv) > 5 else 900
f = next((ROOT/run/'kernel-output'/'transcripts').glob(f'{game}-*.txt'))
T = turns_of(f.read_text(errors='replace')); t0 = T[0]['t']
for x in T:
    m = (x['t']-t0)/60
    if lo <= m <= hi:
        print(f"=== {m:.0f}min action={x['action']} level={x['level']} resp={x['responses']}")
        for a in assistant_blocks(x['body']):
            print('  A:', a.strip()[:width].replace('\n', ' | '))
        th = re.findall(r'^\[THINKING\]\n(.*?)(?=^\[TOOL CALL|^\[ASSISTANT\]|^\[MODEL RESPONSE META|\Z)', x['body'], re.M|re.S)
        if th: print('  T(last):', th[-1].strip()[-width:].replace('\n', ' | '))
