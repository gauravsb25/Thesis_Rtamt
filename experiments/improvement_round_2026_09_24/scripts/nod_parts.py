import numpy as np
F = dict(np.load("feat_base.npz")); m = dict(np.load("meta.npz", allow_pickle=True))
lab = m["label"].astype(str); ses = m["session"].astype(str)
print(f"{'session':14s} | balance turn / down  gap | symmetry turn / down  gap | nod turn/down")
for s in sorted(set(ses)):
    k = ses==s; t = k & np.isin(lab,["LEFT","RIGHT"]); d = k & (lab=="DOWN"); r = k & (lab=="STRAIGHT")
    def md(x, mask): return np.median(x[mask]) if mask.any() else np.nan
    bt, bd = md(F["balance"], t), md(F["balance"], d); st, sd = md(F["symmetry"], t), md(F["symmetry"], d)
    br, sr = md(F["balance"], r), md(F["symmetry"], r)
    print(f"{s:14s} | {bt:6.2f} {bd:6.2f} {bd-bt:5.2f} rest {br:5.2f} | {st:6.2f} {sd:6.2f} {sd-st:5.2f} rest {sr:5.2f} | {md(F['nod'],t):6.2f} {md(F['nod'],d):6.2f}")
