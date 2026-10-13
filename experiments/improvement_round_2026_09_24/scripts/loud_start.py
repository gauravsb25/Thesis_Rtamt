import numpy as np, harness as H
f = H.load(); F = dict(np.load("feat_base.npz")); m = dict(np.load("meta.npz", allow_pickle=True))
pred = H.predict(f); lab = f["label"]
mv = pred != "STRAIGHT"
s15 = F["loud15_start"]; s1 = F["loud1_start"]
off = m["movement_offset"].astype(float)*256; dur = m["movement_duration"].astype(float)*256
for nm, g in (("dev", f["dev"]), ("t01", f["t01"]), ("s11", f["s11"])):
    t = g & np.isin(lab, ["LEFT","RIGHT"]) & np.isin(pred, ["LEFT","RIGHT"])
    right = np.where(lab == "LEFT", f["turn"] > 0, f["turn"] < 0)
    early = s15 < 32
    cov = (s15 <= off) & (s15 + 384 >= off + np.minimum(dur, 384))   # loud block contains movement start
    print(f"{nm}: turn windows called turn {t.sum()}; loud1.5 at start(<0.125s) {early[t].mean()*100:.1f}% acc there {right[t & early].mean()*100 if (t&early).any() else np.nan:.1f} vs {right[t & ~early].mean()*100:.1f}")
    # position of loud block start relative to movement onset
    rel = (s15 - off)/256
    for lo, hi in ((-3,-0.5),(-0.5,-0.1),(-0.1,0.1),(0.1,0.3),(0.3,0.6),(0.6,3)):
        k = t & (rel >= lo) & (rel < hi)
        if k.sum(): print(f"    loud1.5 start - onset in [{lo},{hi}) s: n {k.sum():4d}  direction acc {right[k].mean()*100:5.1f}")
