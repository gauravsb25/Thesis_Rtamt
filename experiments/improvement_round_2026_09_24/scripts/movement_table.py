"""Per-movement summary: majority prediction, median features, for dev sessions."""
import sys; sys.path.insert(0, r"C:\Users\gaura\thesis_rtamt")
import numpy as np, project_code as pc
from collections import Counter
f = dict(np.load(r"C:\Users\gaura\thesis_rtamt\measurements_v2.npz", allow_pickle=True))
f['label']=f['label'].astype(str); f['session']=f['session'].astype(str)
F = dict(np.load("feat_base.npz")); m = dict(np.load("meta.npz", allow_pickle=True))
pred = pc.predict(f); lab = f['label']
key = np.array([f"{s}|{t}|{r}" for s,t,r in zip(m['session'], m['task'], m['run'])])
rows = []
for k in np.unique(key[lab != "STRAIGHT"]):
    idx = np.flatnonzero((key == k) & (lab != "STRAIGHT"))
    c = Counter(pred[idx]).most_common(1)[0][0]
    rows.append((k, lab[idx[0]], c, len(idx), np.median(F["nod"][idx]), np.median(F["balance"][idx]), np.median(F["symmetry"][idx]),
                 np.median(F["sideways"][idx]), np.median(F["turn"][idx]), np.median(F["contrast"][idx]), np.median(F["deflect"][idx]), np.median(F["delta"][idx])))
import pickle; pickle.dump(rows, open("movements.pkl", "wb"))
print(f"{'movement':32s} {'true':6s} {'pred':8s} n   nod    bal    sym   side   turn  contr defl  delta")
for r in rows:
    s = r[0].split("|")[0]
    if s.startswith("sub_01") and int(s[-2:]) <= 30 and r[1] in ("LEFT","RIGHT") and r[2] == "DOWN":
        print(f"{r[0]:32s} {r[1]:6s} {r[2]:8s} {r[3]:2d} " + " ".join(f"{x:6.2f}" for x in r[4:]))
print("--- correct turns, dev, medians of per-movement medians")
A = np.array([r[4:] for r in rows if r[0].startswith("sub_01") and int(r[0].split('|')[0][-2:])<=30 and r[1] in ("LEFT","RIGHT") and r[2]==r[1]])
print("   ", " ".join(f"{x:6.2f}" for x in np.median(A, 0)))
print("--- correct DOWN, dev")
A = np.array([r[4:] for r in rows if r[0].startswith("sub_01") and int(r[0].split('|')[0][-2:])<=30 and r[1]=="DOWN" and r[2]=="DOWN"])
print("   ", " ".join(f"{x:6.2f}" for x in np.median(A, 0)))
