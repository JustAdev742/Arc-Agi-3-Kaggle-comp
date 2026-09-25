import sys, re, json
sys.path.insert(0, '/tmp/claude-0/-home-user-Arc-Agi-3-Kaggle-comp/d342458e-03bd-545b-8a6d-06bca061963e/scratchpad/research/hg')
from extract import turns_of, ROOT, RUNS, HARD
ALL = ['ar25','bp35','cd82','cn04','dc22','ft09','g50t','ka59','lf52','lp85','ls20','m0r0','r11l','re86','s5i5','sb26','sc25','sk48','sp80','su15','tn36','tr87','tu93','vc33','wa30']
CODE_RE = re.compile(r'^\[TOOL CALL: python\]\n<tool_call>\n<function=python>\n<parameter=code>\n(.*?)</parameter>', re.M|re.S)
tot = {}
for g in ALL:
    for r in RUNS:
        f = next((ROOT/r/'kernel-output'/'transcripts').glob(f'{g}-*.txt'))
        T = turns_of(f.read_text(errors='replace'))
        n=act=diff=anim=err=0
        for x in T:
            if x['level'] != 1: continue
            for c in CODE_RE.findall(x['body']):
                n += 1
                if 'action(' in c and not re.search(r'^\s*#.*action\(', c, re.M): act += 1
                if re.search(r'previous_frame|before_frame|history\[', c): diff += 1
                if 'animation(' in c: anim += 1
        tot[(g,r)] = (n, act, diff, anim)
hard = [k for k in tot if k[0] in HARD and k[0] != 'su15']
easy = [k for k in tot if k[0] not in HARD]
for name, ks in (('hard8', hard), ('easy16', easy)):
    s = [sum(tot[k][i] for k in ks) for i in range(4)]
    print(name, 'calls', s[0], 'with action()', s[1], f'{s[1]/s[0]:.2f}', 'diff-code', s[2], f'{s[2]/s[0]:.2f}', 'animation()', s[3], f'{s[3]/s[0]:.2f}')
for g in HARD:
    ks=[k for k in tot if k[0]==g]
    s = [sum(tot[k][i] for k in ks) for i in range(4)]
    print(g, s, f'act {s[1]/max(1,s[0]):.2f} diff {s[2]/max(1,s[0]):.2f}')
