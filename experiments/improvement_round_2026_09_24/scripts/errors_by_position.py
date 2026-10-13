import sys; sys.path.insert(0, r"C:\Users\gaura\thesis_rtamt")
import numpy as np, project_code as pc
f = dict(np.load(r"C:\Users\gaura\thesis_rtamt\measurements_v2.npz", allow_pickle=True))
f['label']=f['label'].astype(str); f['session']=f['session'].astype(str)
f['dev']=np.array([s.startswith('sub_01') and int(s[-2:])<=30 for s in f['session']])
m = dict(np.load("meta.npz", allow_pickle=True))
assert (m['session']==f['session']).all() and (m['label']==f['label']).all()
pred = pc.predict(f)
lab = f['label']
off = m['movement_offset'].astype(float); dur = m['movement_duration'].astype(float)
end = off + dur
splits = pc.splits(f)
for name, mask in splits:
    print("==", name)
    for c in ("LEFT","RIGHT","DOWN"):
        sel = mask & (lab==c)
        # movement center position in window
        cen = (off + dur/2)[sel]
        ok = (pred[sel]==c)
        bins = [0,1.0,1.25,1.5,1.75,2.0,3.0]
        idx = np.digitize(cen, bins)-1
        s = []
        for b in range(len(bins)-1):
            k = idx==b
            if k.sum(): s.append(f"[{bins[b]:.2f},{bins[b+1]:.2f}) {ok[k].mean()*100:5.1f}% n={k.sum()}")
        conf = {q: int((pred[sel]==q).sum()) for q in pc.CLASSES}
        print(f"  {c:6s} acc {ok.mean()*100:5.1f}  confusions {conf}")
        print("     by movement centre:", " | ".join(s))
