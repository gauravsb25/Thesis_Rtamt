import numpy as np, harness as H, itertools
f = H.load(); T = dict(np.load("true_hist.npz")); lab = f["label"]
g = dict(f); g["background"] = T["background"]; g["background_valid"] = T["background_valid"]
base = H.predict(g)
def show(nm, pr):
    print(f"{nm:40s}", " | ".join(f"{k} {H.balanced(g, pr, g[k]):6.2f}" for k in ("dev","t01","s11")))
show("true history, shipped", base)
# b only
grid = [dict(H.SHIPPED, b=round(float(b), 2)) for b in np.arange(3.0, 12.01, 0.25)]
cvp, ch = H.cv(g, lambda p: H.predict(g, p), grid)
print("b-only CV choices", [c["b"] for c in ch], "CV gain vs shipped (gain, better, worse)", H.compare(g, base, cvp, g["dev"]))
# b and g jointly
grid = [dict(H.SHIPPED, b=round(float(b), 2), g=round(float(gg), 2)) for b, gg in itertools.product(np.arange(3.0, 12.01, 0.5), np.arange(5.0, 11.01, 0.5))]
cvp2, ch2 = H.cv(g, lambda p: H.predict(g, p), grid)
print("b,g CV choices", [(c["b"], c["g"]) for c in ch2], "CV gain", H.compare(g, base, cvp2, g["dev"]))
# in-sample curve over b on dev
for b in (4.0, 4.5, 5.0, 5.5, 6.0, 6.61, 7.0, 8.0):
    pr = H.predict(g, dict(H.SHIPPED, b=b))
    print(f"   b={b:5.2f} dev {H.balanced(g, pr, g['dev']):6.2f}  [" + " ".join(f"{np.mean(pr[g['dev'] & (lab==c)]==c)*100:5.1f}" for c in H.CLASSES) + "]")
