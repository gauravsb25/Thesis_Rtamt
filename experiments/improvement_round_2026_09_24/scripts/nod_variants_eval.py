import numpy as np, harness as H, itertools
f = H.load(); F = dict(np.load("feat_base.npz")); V = dict(np.load("feat_nodvar.npz")); lab = f["label"]
base = H.predict(f); mv = base != "STRAIGHT"
def auc(x, pos, neg):
    a, b = x[pos], x[neg]; r = np.argsort(np.argsort(np.concatenate([a, b])))
    return (r[:len(a)].sum() - len(a)*(len(a)-1)/2) / (len(a)*len(b))
cands = {"nod (shipped)": F["nod"], "balance": F["balance"], "symmetry": F["symmetry"],
         "bal_loud1": V["bal_loud1"], "bal_loud15": V["bal_loud15"], "bal_loud1+sym": V["bal_loud1"] + F["symmetry"],
         "bal_evt+sym_evt": V["bal_evt"] + V["sym_evt"], "bal_evt": V["bal_evt"], "sym_evt": V["sym_evt"],
         "cm_all_evt": V["cm_all_evt"], "fb_evt": V["fb_evt"], "bal+cm_all_evt": F["balance"] + V["cm_all_evt"]}
print(f"{'feature':18s}  AUC(DOWN vs turn among detected)  dev    t01    s11   | per-session min AUC dev")
for nm, x in cands.items():
    a = []
    for g in ("dev","t01","s11"):
        t = f[g] & mv & np.isin(lab, ["LEFT","RIGHT"]); d = f[g] & mv & (lab == "DOWN")
        a.append(auc(x, d, t))
    ps = []
    for s in np.unique(f["session"][f["dev"]]):
        k = (f["session"] == s) & mv
        t = k & np.isin(lab, ["LEFT","RIGHT"]); d = k & (lab == "DOWN")
        if t.sum() > 5 and d.sum() > 5: ps.append(auc(x, d, t))
    print(f"{nm:18s}  {'':32s}{a[0]:.3f}  {a[1]:.3f}  {a[2]:.3f}  | {np.min(ps):.3f} median {np.median(ps):.3f}")
