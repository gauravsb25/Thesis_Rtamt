"""Per movement: sign of the initial deflection of several pairs after the ANNOTATED onset (oracle onset),
to see if any pair's first deflection is subject-consistent. Causal 0.3-6 Hz filter over the whole run."""
import pickle, numpy as np
from scipy.signal import butter, lfilter
runs = pickle.load(open("runs.pkl","rb"))
CH = ["Fp1","Fp2","F3","Fz","F4","T3","Cz","T4"]; I = {c:i for i,c in enumerate(CH)}
b, a = butter(2, [0.3/128, 6/128], btype="band")
pairs = {"F4-T3": ("F4","T3"), "T3-T4": ("T3","T4"), "F3-F4": ("F3","F4"), "Fp1-Fp2": ("Fp1","Fp2"),
         "Fp2-T3": ("Fp2","T3"), "F3-T4": ("F3","T4"), "Fp1-T4": ("Fp1","T4"), "Fp2-F4": ("Fp2","F4")}
# expected sign for LEFT (from sub_01 dev medians): F4-T3 negative first, T3-T4 positive, F3-F4 positive?, Fp1-Fp2 positive
rows = []
for k, v in runs.items():
    if v["label"] not in ("LEFT","RIGHT"): continue
    x = lfilter(b, a, v["sig"].astype(float), axis=-1)
    on = int(round(v["onset"]))
    base = slice(max(0, on-64), on)
    rec = {"key": k, "label": v["label"], "sub": v["session"][:6], "ses": int(v["session"][-2:])}
    for nm, (p, q) in pairs.items():
        d = x[I[p]] - x[I[q]]
        b0 = d[base].mean() if on > 0 else 0.0
        for w0, w1 in ((0.0, 0.15), (0.0, 0.25), (0.05, 0.3), (0.0, 0.4), (0.1, 0.4)):
            seg = d[on+int(w0*256):on+int(w1*256)] - b0
            rec[f"{nm}|{w0}-{w1}"] = seg.mean()
    rows.append(rec)
keys = [k for k in rows[0] if "|" in k]
lab = np.array([r["label"] for r in rows]); sub = np.array([r["sub"] for r in rows]); ses = np.array([r["ses"] for r in rows])
grp = np.where(sub=="sub_11", "s11", np.where(ses<=30, "dev", "t01"))
print(f"{'feature':22s}  acc(sign, LEFT=+ oriented on dev)   dev    t01    s11")
res = []
for kk in keys:
    x = np.array([r[kk] for r in rows])
    # orient on dev
    d = grp=="dev"
    s = np.sign(np.median(x[d & (lab=="LEFT")]) - np.median(x[d & (lab=="RIGHT")]))
    acc = {g: np.mean((s*x[grp==g] > 0) == (lab[grp==g]=="LEFT"))*100 for g in ("dev","t01","s11")}
    res.append((kk, acc["dev"], acc["t01"], acc["s11"]))
for r in sorted(res, key=lambda r: -min(r[1], r[3])):
    print(f"{r[0]:22s} {'':34s} {r[1]:5.1f}  {r[2]:5.1f}  {r[3]:5.1f}")
