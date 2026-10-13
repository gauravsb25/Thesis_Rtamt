import numpy as np, harness as H, itertools
f = H.load(); S = dict(np.load("feat_selfbg.npz")); lab = f["label"]; base = H.predict(f)
valid = f["background_valid"] >= 0.5
for q in (10, 25, 50):
    x = S[f"selfbg_q{q}"]
    def make(g):
        loud = np.where(valid, f["background"] >= H.SHIPPED["b"], x >= g["c"])
        mv = loud & (f["delta"] >= H.SHIPPED["d"]) & (f["deflect"] >= H.SHIPPED["q"])
        return H.predict(f, move=mv)
    grid = [dict(c=c) for c in np.exp(np.linspace(np.log(1.5), np.log(80), 90))]
    cvp, ch = H.cv(f, make, grid)
    print(f"q{q}: CV choices {[round(float(c['c']),2) for c in ch]}  (gain, better, worse) vs shipped {H.compare(f, base, cvp, f['dev'])}")
# fair baseline: re-select g for the contrast gate under the same CV
def make0(g):
    loud = np.where(valid, f["background"] >= H.SHIPPED["b"], f["contrast"] >= g["c"])
    return H.predict(f, move=loud & (f["delta"] >= H.SHIPPED["d"]) & (f["deflect"] >= H.SHIPPED["q"]))
grid = [dict(c=c) for c in np.exp(np.linspace(np.log(2), np.log(40), 90))]
cv0, ch0 = H.cv(f, make0, grid)
print("contrast gate, g re-selected under CV:", [round(float(c['c']),2) for c in ch0], H.compare(f, base, cv0, f["dev"]))
