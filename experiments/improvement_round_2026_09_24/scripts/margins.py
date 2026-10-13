import numpy as np, harness as H
f = H.load(); base = H.predict(f); lab = f["label"]; p = H.SHIPPED
m = dict(np.load("meta.npz", allow_pickle=True))
key = np.array([f"{s}|{t}|{r}" for s,t,r in zip(m['session'], m['task'], m['run'])])
# per movement medians of the relevant margins, dev only
g = f["dev"]
cats = {}
for k in np.unique(key[g & (lab != "STRAIGHT")]):
    idx = np.flatnonzero((key == k) & (lab != "STRAIGHT"))
    from collections import Counter
    pr = Counter(base[idx]).most_common(1)[0][0]; tr = lab[idx[0]]
    if pr == tr: continue
    c = f"{tr}->{pr}"
    nod_m = np.median(f["nod"][idx] - p["n"]); side_m = np.median(f["sideways"][idx] - p["v"])
    con = np.median(np.log(f["contrast"][idx] / p["g"])); tu = np.median(f["turn"][idx])
    dl = np.median(f["delta"][idx] - p["d"]); de = np.median(f["deflect"][idx] - p["q"])
    cats.setdefault(c, []).append((nod_m, side_m, con, dl, de, tu))
for c, v in sorted(cats.items()):
    v = np.array(v)
    print(f"{c:16s} n={len(v):2d}  nod-n {np.round(np.sort(v[:,0]),2)}")
    print(f"{'':16s}       log(contrast/g) {np.round(np.sort(v[:,2]),2)}   turn {np.round(np.sort(v[:,5]),2)}")
