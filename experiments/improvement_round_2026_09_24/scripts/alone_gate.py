import numpy as np, harness as H
f = H.load(); S = dict(np.load("feat_selfbg.npz")); lab = f["label"]; p = H.SHIPPED
def pr_gate(x, c):
    mv = (x >= c) & (f["delta"] >= p["d"]) & (f["deflect"] >= p["q"])
    return H.predict(f, move=mv)
alone = H.predict(f, history=False)
print("alone baseline:", {k: round(H.balanced(f, alone, f[k]), 2) for k in ("dev","t01","s11")})
grid_c = [dict(c=c) for c in np.exp(np.linspace(np.log(2), np.log(40), 120))]
cv0, ch0 = H.cv(f, lambda g: pr_gate(f["contrast"], g["c"]), grid_c)
print("contrast, g under CV", [round(float(c['c']),2) for c in ch0], H.compare(f, alone, cv0, f["dev"]))
for q in (10, 25, 50):
    x = S[f"selfbg_q{q}"]
    grid = [dict(c=c) for c in np.exp(np.linspace(np.log(1.2), np.log(80), 150))]
    cvp, ch = H.cv(f, lambda g: pr_gate(x, g["c"]), grid)
    print(f"selfbg q{q} under CV", [round(float(c['c']),2) for c in ch], "vs alone-baseline", H.compare(f, alone, cvp, f["dev"]), " vs contrast-CV", H.compare(f, cv0, cvp, f["dev"]))
