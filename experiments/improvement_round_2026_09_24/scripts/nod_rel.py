import numpy as np
F = dict(np.load("feat_base.npz")); H = dict(np.load("feat_hist.npz")); m = dict(np.load("meta.npz", allow_pickle=True))
lab = m["label"].astype(str); ses = m["session"].astype(str)
v = H["valid"] > 0.5
for kind in ("mean", "med"):
    rest = np.log(H[f"R_mid_{kind}"]/H[f"R_tmp_{kind}"]) + np.log(H[f"R_cm_{kind}"]/H[f"R_df_{kind}"])
    rel = F["nod"] - rest
    print(f"=== kind {kind}")
    def auc(x, pos, neg):
        a, b = x[pos], x[neg]
        if len(a)==0 or len(b)==0: return np.nan
        r = np.argsort(np.argsort(np.concatenate([a,b])))
        return (r[:len(a)].sum() - len(a)*(len(a)-1)/2) / (len(a)*len(b))
    rows = []
    for s in sorted(set(ses)):
        k = (ses==s) & v; t = k & np.isin(lab,["LEFT","RIGHT"]); d = k & (lab=="DOWN")
        if t.sum()<5 or d.sum()<5: continue
        rows.append((s, np.median(F["nod"][t]), np.median(F["nod"][d]), np.median(rel[t]), np.median(rel[d]),
                     auc(F["nod"], d, t), auc(rel, d, t), np.median(rest[k])))
    rows = np.array(rows, dtype=object)
    for r in rows: print(f"{r[0]:14s} nod T {r[1]:6.2f} D {r[2]:6.2f} | rel T {r[3]:6.2f} D {r[4]:6.2f} | AUC nod {r[5]:.3f} rel {r[6]:.3f} | rest {r[7]:5.2f}")
    for grp, sel in (("dev", [int(r[0][-2:])<=30 and r[0].startswith("sub_01") for r in rows]),
                     ("s01test", [int(r[0][-2:])>30 and r[0].startswith("sub_01") for r in rows]),
                     ("s11", [r[0].startswith("sub_11") for r in rows])):
        sel = np.array(sel)
        R = rows[sel]
        print(f"  {grp}: sd across sessions of median turn nod {np.std(R[:,1].astype(float)):.2f} -> rel {np.std(R[:,3].astype(float)):.2f};"
              f" down {np.std(R[:,2].astype(float)):.2f} -> {np.std(R[:,4].astype(float)):.2f}")
    # pooled AUC by split
    dev = np.array([s.startswith("sub_01") and int(s[-2:])<=30 for s in ses]); t1 = np.array([s.startswith("sub_01") and int(s[-2:])>30 for s in ses]); s11 = np.char.startswith(ses, "sub_11")
    for grp, g in (("dev", dev), ("s01test", t1), ("s11", s11)):
        t = g & v & np.isin(lab,["LEFT","RIGHT"]); d = g & v & (lab=="DOWN")
        print(f"  pooled {grp}: AUC nod {auc(F['nod'], d, t):.3f}  rel {auc(rel, d, t):.3f}   (n turn {t.sum()}, down {d.sum()})")
