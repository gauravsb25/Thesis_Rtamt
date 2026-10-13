import numpy as np, harness as H
f = H.load(); base = H.predict(f); lab = f["label"]
mv = H.move_mask(f)
at = np.abs(f["turn"])
for nm, g in (("dev", f["dev"]),):
    for c in ("LEFT","RIGHT","DOWN"):
        k = g & mv & (lab == c)
        print(nm, c, "abs turn pct 10/25/50/75/90:", np.percentile(at[k], [10,25,50,75,90]).round(2))
grid = [dict(t=t) for t in np.arange(2.0, 8.01, 0.25)]
def make(p): return H.predict(f, notnod_extra=at > p["t"])
cvp, chosen = H.cv(f, make, grid)
print("chosen", chosen)
print("CV gain, better/worse sessions:", H.compare(f, base, cvp, f["dev"]))
for p in (dict(t=3.0), dict(t=4.0), dict(t=5.0)):
    pr = make(p)
    print(p, {nm: round(H.balanced(f, pr, m) - H.balanced(f, base, m), 2) for nm, m in (("dev", f["dev"]),)}, H.compare(f, base, pr, f["dev"]))
