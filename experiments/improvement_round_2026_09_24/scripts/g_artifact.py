import numpy as np, harness as H
f = H.load(); T = dict(np.load("true_hist.npz")); lab = f["label"]
ft = dict(f); ft["background"] = T["background"]; ft["background_valid"] = T["background_valid"]
for proto, ff in (("rebuilt history", f), ("TRUE history", ft)):
    v = ff["background_valid"] >= 0.5
    print(f"== {proto}: share of windows WITHOUT history  movement {np.mean(~v[lab!='STRAIGHT'])*100:.1f}%  rest {np.mean(~v[lab=='STRAIGHT'])*100:.1f}%")
    for g in (7.81, 6.0, 5.0, 4.5, 4.0):
        pr = H.predict(ff, dict(H.SHIPPED, g=g))
        cells = []
        for nm in ("dev", "t01", "s11"):
            m = ff[nm]; rest = m & (lab == "STRAIGHT")
            cells.append(f"{nm} {H.balanced(ff, pr, m):6.2f} (FA {int((pr[rest]!='STRAIGHT').sum()):4d})")
        print(f"   g={g:5.2f}  " + " | ".join(cells))
    base = H.predict(ff)
    grid = [dict(H.SHIPPED, g=round(float(c), 2)) for c in np.arange(2.5, 10.01, 0.25)]
    cvp, ch = H.cv(ff, lambda p: H.predict(ff, p), grid)
    print(f"   CV re-selecting g: choices {[c['g'] for c in ch]}  (gain, better, worse) {H.compare(ff, base, cvp, ff['dev'])}")
