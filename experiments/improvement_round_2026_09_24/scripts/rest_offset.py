import numpy as np, harness as H
f = H.load(); F = dict(np.load("feat_base.npz")); lab = f["label"]; ses = f["session"]
base = H.predict(f)
rows = []
for s in np.unique(ses):
    k = ses == s
    r = k & (lab == "STRAIGHT") & (base == "STRAIGHT")        # rest windows the monitor itself calls STRAIGHT (label-free in deployment)
    t = k & np.isin(lab, ["LEFT","RIGHT"]); d = k & (lab == "DOWN")
    rows.append((s, np.median(F["nod"][r]), np.median(F["balance"][r]), np.median(F["symmetry"][r]),
                 np.median(F["nod"][t]) if t.sum() > 3 else np.nan, np.median(F["nod"][d]) if d.sum() > 3 else np.nan))
R = np.array([r[1:] for r in rows], float); names = [r[0] for r in rows]
grp = np.array(["dev" if n.startswith("sub_01") and int(n[-2:]) <= 30 else ("t01" if n.startswith("sub_01") else "s11") for n in names])
for g in ("dev", "t01", "s11", "all"):
    k = (grp == g) if g != "all" else np.ones(len(grp), bool)
    for j, nm in ((3, "turn"), (4, "down")):
        ok = k & np.isfinite(R[:, j])
        c = np.corrcoef(R[ok, 0], R[ok, j])[0, 1]
        cb = np.corrcoef(R[ok, 1], R[ok, j])[0, 1]; cs = np.corrcoef(R[ok, 2], R[ok, j])[0, 1]
        print(f"{g:4s} corr(rest nod, {nm} nod) = {c:5.2f}   (rest balance {cb:5.2f}, rest symmetry {cs:5.2f})  n={ok.sum()}")
# windowed rest nod over the per-window median of a moving pool is the deployable form; check pooled separation after offsetting by the session rest median
rel = F["nod"].copy()
for s, r0 in zip(names, R[:, 0]): rel[ses == s] -= r0
def auc(x, pos, neg):
    a, b = x[pos], x[neg]; r = np.argsort(np.argsort(np.concatenate([a, b])))
    return (r[:len(a)].sum() - len(a)*(len(a)-1)/2) / (len(a)*len(b))
mv = base != "STRAIGHT"
for g, m in (("dev", f["dev"]), ("t01", f["t01"]), ("s11", f["s11"])):
    t = m & mv & np.isin(lab, ["LEFT","RIGHT"]); d = m & mv & (lab == "DOWN")
    print(f"{g}: AUC DOWN vs turn  nod {auc(F['nod'], d, t):.3f}   nod - session rest median {auc(rel, d, t):.3f}")
