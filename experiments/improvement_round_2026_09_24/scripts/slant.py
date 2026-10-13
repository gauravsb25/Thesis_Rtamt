import numpy as np, harness as H, itertools
f = H.load(); base = H.predict(f); p = H.SHIPPED
nod, side = f["nod"], f["sideways"]
grid = [dict(a=a, n=n, v=v) for a, n, v in itertools.product(np.arange(0.0, 1.01, 0.1), np.arange(-2.5, 2.01, 0.1), (7.54, 99.0))]
def make(g): return H.predict(f, nod=(nod - g["a"] * side >= g["n"]) & (side <= g["v"]))
cvp, chosen = H.cv(f, make, grid)
print("chosen", [{k: round(float(v), 2) for k, v in c.items()} for c in chosen])
print("CV (gain, better, worse):", H.compare(f, base, cvp, f["dev"]))
# the baseline itself under the same CV (re-select n with a=0, v fixed)
grid0 = [dict(a=0.0, n=n, v=7.54) for n in np.arange(-2.5, 2.01, 0.1)]
cv0, ch0 = H.cv(f, make, grid0)
print("baseline re-selected n under CV:", [round(float(c["n"]), 2) for c in ch0], H.compare(f, base, cv0, f["dev"]))
