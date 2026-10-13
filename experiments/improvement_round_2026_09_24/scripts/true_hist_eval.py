import numpy as np, harness as H
f = H.load(); T = dict(np.load("true_hist.npz")); lab = f["label"]
g = dict(f); g["background"] = T["background"]; g["background_valid"] = T["background_valid"]
base = H.predict(f); tru = H.predict(g)
def row(name, ff, pr):
    s = []
    for nm, m in (("dev", ff["dev"]), ("t01", ff["t01"]), ("s11", ff["s11"])):
        s.append(f"{nm} {H.balanced(ff, pr, m):6.2f} [" + " ".join(f"{np.mean(pr[m & (lab==c)]==c)*100:5.1f}" for c in H.CLASSES) + "]")
    print(f"{name:34s} " + " | ".join(s))
row("rebuilt history (shipped headline)", f, base)
row("TRUE run history, shipped b", g, tru)
for nm, m in (("dev", f["dev"]), ("t01", f["t01"]), ("s11", f["s11"])):
    mv = m & (lab != "STRAIGHT"); rs = m & (lab == "STRAIGHT")
    print(f"  {nm}: history valid  movement windows rebuilt {np.mean(f['background_valid'][mv]>=.5)*100:5.1f}% true {np.mean(T['background_valid'][mv]>=.5)*100:5.1f}%"
          f" | rest rebuilt {np.mean(f['background_valid'][rs]>=.5)*100:5.1f}% true {np.mean(T['background_valid'][rs]>=.5)*100:5.1f}%")
# where both valid, do the background values agree?
both = (f["background_valid"] >= .5) & (T["background_valid"] >= .5)
print("windows valid in both:", both.sum(), " median |rel diff| of background:", np.median(np.abs(f["background"][both]-T["background"][both])/T["background"][both]))
