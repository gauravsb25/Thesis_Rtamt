"""Direction from the FIRST F4-T3 lobe that reaches h x the largest excursion inside the loudest 1.5 s
(the peak rule is h = 1). Scored on windows the shipped monitor calls a turn; per-movement view for the swaps."""
import numpy as np, harness as H
from pathlib import Path
from collections import Counter
from scipy.signal import lfilter
from feat import B_DIR, B_DET, I, block_sums
DATA = Path(r"C:\Users\gaura\thesis_rtamt\extracted_npz_dataset_v2")
HS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
reads = {h: [] for h in HS}
for p in sorted(DATA.glob("*/ses-*/clean_epochs.npz")):
    X = np.load(p)["epochs"][:, :8] * 1e6
    det = lfilter(*B_DET, X, axis=-1)
    sp = 1.4826*np.median(np.abs(det - np.median(det, -1, keepdims=True)), -1, keepdims=True) + 1e-12
    energy = np.mean((det/sp)**2, 1); s15 = block_sums(energy, 384).argmax(1)
    dr = lfilter(*B_DIR, X, axis=-1); pair = dr[:, I["F4"]] - dr[:, I["T3"]]
    seg = np.take_along_axis(pair, s15[:, None] + np.arange(384)[None], 1); seg = seg - np.median(seg, -1, keepdims=True)
    mx = np.abs(seg).max(1, keepdims=True)
    for h in HS:
        hit = np.abs(seg) >= h * mx - 1e-12
        first = hit.argmax(1)
        reads[h].append(-seg[np.arange(len(seg)), first])          # >0 LEFT
R = {h: np.concatenate(v) for h, v in reads.items()}
f = H.load(); lab = f["label"]; base = H.predict(f)
m = dict(np.load("meta.npz", allow_pickle=True))
key = np.array([f"{s}|{t}|{r}" for s, t, r in zip(m["session"], m["task"], m["run"])])
assert np.allclose(np.sign(R[1.0]), np.sign(f["turn"]))
turnwin = np.isin(lab, ["LEFT","RIGHT"]) & np.isin(base, ["LEFT","RIGHT"])
right = lambda x: np.where(lab == "LEFT", x > 0, x < 0)
swapped = [k for k in np.unique(key[f["dev"] & np.isin(lab, ["LEFT","RIGHT"])])
           if {Counter(base[(key == k) & (lab != "STRAIGHT")]).most_common(1)[0][0], lab[(key == k) & (lab != "STRAIGHT")][0]} == {"LEFT","RIGHT"}]
print("direction accuracy among windows called a turn (dev / 31-36 / sub_11), and swapped dev movements fixed")
for h in HS:
    acc = [np.mean(right(R[h])[f[s] & turnwin]) * 100 for s in ("dev", "t01", "s11")]
    fixed = sum(np.mean(right(R[h])[(key == k) & np.isin(base, ["LEFT","RIGHT"])]) > 0.5 for k in swapped)
    print(f"  h={h:.1f}: {acc[0]:5.1f} / {acc[1]:5.1f} / {acc[2]:5.1f}   swapped dev movements now right: {fixed}/{len(swapped)}")
