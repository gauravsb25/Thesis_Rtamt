import numpy as np, harness as H
from collections import Counter
f = H.load(); lab = f["label"]; base = H.predict(f); p = H.SHIPPED
m = dict(np.load("meta.npz", allow_pickle=True))
key = np.array([f"{s}|{t}|{r}" for s,t,r in zip(m['session'], m['task'], m['run'])])
for split in ("s11", "t01", "dev"):
    sel = f[split] & (lab != "STRAIGHT") & (base == "STRAIGHT")
    print(f"== {split}: missed movement windows {sel.sum()}")
    for k in np.unique(key[sel]):
        idx = np.flatnonzero((key == k) & (lab != "STRAIGHT"))
        miss = idx[base[idx] == "STRAIGHT"]
        print(f"   {k:26s} {lab[idx[0]]:5s} missed {len(miss):2d}/{len(idx):2d}  contrast med {np.median(f['contrast'][miss]):6.2f}  delta {np.median(f['delta'][miss]):.3f}  deflect {np.median(f['deflect'][miss]):.2f}  hist {np.mean(f['background_valid'][miss]>=.5):.1f}")
    if split == "s11": pass
