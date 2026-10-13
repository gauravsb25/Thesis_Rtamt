import sys; sys.path.insert(0, r"C:\Users\gaura\thesis_rtamt")
import numpy as np, project_code as pc
from collections import Counter, defaultdict
f = dict(np.load(r"C:\Users\gaura\thesis_rtamt\measurements_v2.npz", allow_pickle=True))
f['label']=f['label'].astype(str); f['session']=f['session'].astype(str)
f['dev']=np.array([s.startswith('sub_01') and int(s[-2:])<=30 for s in f['session']])
m = dict(np.load("meta.npz", allow_pickle=True))
pred = pc.predict(f); lab = f['label']
key = np.array([f"{s}|{t}|{r}" for s,t,r in zip(m['session'], m['task'], m['run'])])
for name, mask in pc.splits(f):
    print("==", name)
    for c in ("LEFT","RIGHT","DOWN"):
        sel = np.flatnonzero(mask & (lab==c))
        groups = defaultdict(list)
        for i in sel: groups[key[i]].append(pred[i])
        fr = np.array([np.mean(np.array(v)==c) for v in groups.values()])
        hist = np.histogram(fr, bins=[0,0.001,0.25,0.5,0.75,0.999,1.0001])[0]
        dom = Counter()
        for v in groups.values():
            cnt = Counter(v); dom[cnt.most_common(1)[0][0]] += 1
        print(f"  {c:6s} movements {len(groups):3d}  window-acc {np.mean(pred[sel]==c)*100:5.1f}  movement-majority-correct {np.mean(fr>0.5)*100:5.1f}  frac-correct hist [0|0-.25|.25-.5|.5-.75|.75-1|1] {hist.tolist()}  majority-pred {dict(dom)}")
