import numpy as np, harness as H
f = H.load(); S = dict(np.load("feat_selfbg.npz")); lab = f["label"]; p = H.SHIPPED
other = (f["delta"] >= p["d"]) & (f["deflect"] >= p["q"])
mov = lab != "STRAIGHT"
def recall_at_fa(x, m, fa_targets):
    # for each target FA rate on rest windows of mask m, threshold and movement recall
    rest = x[m & ~mov & other]; nrest = (m & ~mov).sum()
    out = []
    for fa in fa_targets:
        k = int(round(fa * nrest))
        thr = np.sort(rest)[::-1][k] if k < len(rest) else -np.inf
        det = (x > thr) & other
        out.append(det[m & mov].mean() * 100)
    return out
fas = [0.02, 0.05, 0.08, 0.10, 0.15]
print("movement recall at matched rest false-alarm rate (alone protocol, delta & deflect fixed)")
for split in ("dev", "t01", "s11"):
    m = f[split]
    print(f"== {split}  FA rates {fas}")
    for nm, x in (("contrast", f["contrast"]), ("selfbg_q10", S["selfbg_q10"]), ("selfbg_q25", S["selfbg_q25"]), ("selfbg_q50", S["selfbg_q50"])):
        print(f"   {nm:11s}", " ".join(f"{r:6.2f}" for r in recall_at_fa(x, m, fas)))
