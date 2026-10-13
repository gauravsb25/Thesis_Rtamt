"""Compute nod-feature variants per window (window only, like the monitor) and compare their
nod-vs-turn separation per split, and their CV value inside the full monitor."""
import numpy as np
from pathlib import Path
from scipy.signal import lfilter
from feat import B_CLS, B_DET, I, block_sums
DATA = Path(r"C:\Users\gaura\thesis_rtamt\extracted_npz_dataset_v2")
out = {k: [] for k in ("bal_loud1", "bal_loud15", "sym_win", "bal_evt", "sym_evt", "cm_all_evt", "lr_all_evt", "fb_evt")}
for p in sorted(DATA.glob("*/ses-*/clean_epochs.npz")):
    X = np.load(p)["epochs"][:, :8] * 1e6
    det = lfilter(*B_DET, X, axis=-1)
    sp = 1.4826*np.median(np.abs(det - np.median(det, -1, keepdims=True)), -1, keepdims=True) + 1e-12
    energy = np.mean((det/sp)**2, 1)
    cls = lfilter(*B_CLS, X, axis=-1); cls = cls - np.median(cls, -1, keepdims=True)
    n = X.shape[0]
    b1 = block_sums(energy, 256); s1 = b1.argmax(1)
    b15 = block_sums(energy, 384); s15 = b15.argmax(1)
    def seg(start, w):
        idx = start[:, None] + np.arange(w)[None]
        return np.take_along_axis(cls, np.repeat(idx[:, None, :], 8, 1), 2)
    S1, S15 = seg(s1, 256), seg(s15, 384)
    def bal(S): 
        e = (S**2).mean(-1); return np.log((e[:, I["Fz"]] + e[:, I["Cz"]] + 1e-18) / (e[:, I["T3"]] + e[:, I["T4"]] + 1e-18))
    def sym(S):
        return np.log((np.mean((S[:, I["F3"]] + S[:, I["F4"]])**2, -1) + 1e-18) / (np.mean((S[:, I["F3"]] - S[:, I["F4"]])**2, -1) + 1e-18))
    out["bal_loud1"].append(bal(S1)); out["bal_loud15"].append(bal(S15))
    out["sym_win"].append(sym(cls))
    # event = samples where pooled energy exceeds 25% of its max (whole-window, like the loudest block)
    thr = 0.25 * energy.max(1, keepdims=True); ev = energy >= thr
    w = ev[:, None, :].astype(float)
    e = ((cls**2) * w).sum(-1) / (w.sum(-1) + 1e-9)
    out["bal_evt"].append(np.log((e[:, I["Fz"]] + e[:, I["Cz"]] + 1e-18) / (e[:, I["T3"]] + e[:, I["T4"]] + 1e-18)))
    cm = ((cls[:, I["F3"]] + cls[:, I["F4"]])**2 * ev).sum(-1); dm = ((cls[:, I["F3"]] - cls[:, I["F4"]])**2 * ev).sum(-1)
    out["sym_evt"].append(np.log((cm + 1e-18) / (dm + 1e-18)))
    # all homologous pairs: common vs differential; front-back gradient
    pairs = [("Fp1","Fp2"), ("F3","F4"), ("T3","T4")]
    C = sum(((cls[:, I[a]] + cls[:, I[b]])**2 * ev).sum(-1) for a, b in pairs)
    D = sum(((cls[:, I[a]] - cls[:, I[b]])**2 * ev).sum(-1) for a, b in pairs)
    out["cm_all_evt"].append(np.log((C + 1e-18) / (D + 1e-18)))
    LR = D
    FB = (((cls[:, I["Fp1"]] + cls[:, I["Fp2"]]) / 2 - cls[:, I["Cz"]])**2 * ev).sum(-1)
    out["lr_all_evt"].append(np.log(LR + 1e-18)); out["fb_evt"].append(np.log((FB + 1e-18) / (LR + 1e-18)))
np.savez("feat_nodvar.npz", **{k: np.concatenate(v) for k, v in out.items()})
print("ok")
