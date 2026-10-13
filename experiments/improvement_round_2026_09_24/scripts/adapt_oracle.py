import numpy as np, harness as H
f = H.load(); lab = f["label"]; base = H.predict(f); p = H.SHIPPED
mvm = H.move_mask(f)
nod, side = f["nod"], f["sideways"]
def pred_with_thr(thr):   # thr: per-window nod threshold
    isnod = (nod >= thr) & (side <= p["v"])
    return np.where(~mvm, "STRAIGHT", np.where(isnod, "DOWN", np.where(f["turn"] > 0, "LEFT", "RIGHT")))
# 1) oracle per-session threshold (labels) on dev
thr = np.full(len(lab), p["n"])
for s in np.unique(f["session"][f["dev"]]):
    k = f["session"] == s
    best, bt = -1, p["n"]
    for t in np.arange(-5, 5, 0.05):
        tt = thr.copy(); tt[k] = t
        a = H.balanced(f, pred_with_thr(tt), k)
        if a > best + 1e-9: best, bt = a, t
    thr[k] = bt
pr = pred_with_thr(thr)
print("ORACLE per-session nod threshold, dev:", round(H.balanced(f, pr, f["dev"]), 2), "vs", round(H.balanced(f, base, f["dev"]), 2))
# 2) label-free, non-causal: largest gap in the session's MOVE-window nod values (Otsu-like 2-means on 1-D)
def otsu_thr(x, minfrac=0.15, min_sep=1.5):
    x = np.sort(x); n = len(x)
    if n < 10: return None
    best, bt = -1, None
    for i in range(int(minfrac*n), int((1-minfrac)*n)):
        a, b = x[:i], x[i:]
        w0, w1 = len(a)/n, len(b)/n
        between = w0*w1*(a.mean()-b.mean())**2
        if between > best: best, bt = between, (x[i-1]+x[i])/2
    if bt is None: return None
    a, b = x[x < bt], x[x >= bt]
    pooled = np.sqrt((a.var()*len(a) + b.var()*len(b))/n) + 1e-9
    return bt if (b.mean() - a.mean())/pooled >= min_sep else None
for minfrac in (0.1, 0.2):
    for sep in (1.5, 2.5, 3.5):
        thr2 = np.full(len(lab), p["n"]); used = 0
        for s in np.unique(f["session"][f["dev"]]):
            k = f["session"] == s
            t = otsu_thr(nod[k & mvm], minfrac, sep)
            if t is not None: thr2[k] = t; used += 1
        pr2 = pred_with_thr(thr2)
        print(f"label-free session split minfrac {minfrac} sep {sep}: dev {H.balanced(f, pr2, f['dev']):6.2f}  sessions adapted {used}/30  (gain, better, worse) {H.compare(f, base, pr2, f['dev'])}")
