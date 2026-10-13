import numpy as np, harness as H
f = H.load(); lab = f["label"]; Z = np.load("bg_variants.npz"); V = Z["variants"]
base = H.predict(f)
# sanity: variant 0 on rebuilt must equal the shipped cache
assert np.allclose(Z["rebuilt"][:, 0, 0], f["background"]) and np.array_equal(Z["rebuilt"][:, 0, 1] >= .5, f["background_valid"] >= .5)
# fair CV baseline for the shipped variant (b re-selected under CV)
grid = [dict(H.SHIPPED, b=round(float(b), 2)) for b in np.arange(4.0, 10.01, 0.25)]
for proto in ("rebuilt", "true"):
    print(f"===== history: {proto}")
    cv_base = None
    for j, v in enumerate(V):
        g = dict(f); g["background"] = Z[proto][:, j, 0]; g["background_valid"] = Z[proto][:, j, 1]
        pr = H.predict(g)
        cvp, ch = H.cv(g, lambda p: H.predict(g, p), grid)
        if j == 0: cv_base = cvp
        gain, bet, wor = H.compare(f, base, pr, f["dev"])
        print(f"{v:22s} valid mov {np.mean(g['background_valid'][lab!='STRAIGHT']>=.5)*100:5.1f}% rest {np.mean(g['background_valid'][lab=='STRAIGHT']>=.5)*100:5.1f}%"
              f" | b=6.61: " + " ".join(f"{k} {H.balanced(g, pr, g[k]):6.2f}" for k in ("dev","t01","s11"))
              + f" | dev sessions +{bet}/-{wor} | CV-b {[c['b'] for c in ch]} dev-CV {H.balanced(g, cvp, g['dev']):6.2f} (vs same-protocol shipped-variant CV {H.balanced(g, cv_base, g['dev']):6.2f})")
