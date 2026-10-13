import sys; sys.path.insert(0, r"C:\Users\gaura\thesis_rtamt")
import numpy as np, pickle, harness as H
from dataclasses import replace
from monitoring.rtamt_signals import RtamtConfig, stream_background
runs = pickle.load(open("runs.pkl", "rb")); m = dict(np.load("meta.npz", allow_pickle=True))
f = H.load(); lab = f["label"]
def compute(settle, minimum):
    cfg = replace(RtamtConfig(), background_settle_seconds=settle, background_min_seconds=minimum)
    Hs = int(round((settle + cfg.background_seconds) * 256))
    bg = np.zeros(len(lab)); va = np.zeros(len(lab))
    for i in range(len(lab)):
        sig = runs[f"{m['session'][i]}|{m['task'][i]}|{int(m['run'][i])}"]["sig"]; ro = int(m["run_offset"][i])
        win = sig[:, ro:ro+768].astype(float); mu = win.mean(1, keepdims=True); lo = max(0, ro - Hs)
        b, v = stream_background(win - mu, (sig[:, lo:ro] - mu) if ro > lo else None, cfg)
        bg[i], va[i] = b.max(), v.max()
    return bg, va
for settle, minimum in ((1.0, 1.0), (0.5, 0.5), (0.5, 1.0), (0.25, 0.5)):
    bg, va = compute(settle, minimum)
    g = dict(f); g["background"] = bg; g["background_valid"] = va
    pr = H.predict(g)
    grid = [dict(H.SHIPPED, b=round(float(b), 2)) for b in np.arange(4.0, 10.01, 0.25)]
    cvp, ch = H.cv(g, lambda p: H.predict(g, p), grid)
    print(f"settle {settle} min {minimum}: valid mov {np.mean(va[lab!='STRAIGHT']>=.5)*100:5.1f}% rest {np.mean(va[lab=='STRAIGHT']>=.5)*100:5.1f}% | shipped b: "
          + " ".join(f"{k} {H.balanced(g, pr, g[k]):6.2f}" for k in ("dev","t01","s11"))
          + f" | CV b {[c['b'] for c in ch]} dev-CV {H.balanced(g, cvp, g['dev']):6.2f}")
