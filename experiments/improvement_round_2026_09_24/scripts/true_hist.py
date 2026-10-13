"""Background gate with the TRUE preceding samples of each window's own run (from the CSV recordings)."""
import sys; sys.path.insert(0, r"C:\Users\gaura\thesis_rtamt")
import numpy as np, pickle
from monitoring.rtamt_signals import RtamtConfig, stream_background
runs = pickle.load(open("runs.pkl", "rb"))
m = dict(np.load("meta.npz", allow_pickle=True))
cfg = RtamtConfig()
H = int(round((cfg.background_settle_seconds + cfg.background_seconds) * 256))   # 1280, what StreamMonitor keeps
n = len(m["label"])
bg = np.zeros(n); valid = np.zeros(n); hist_len = np.zeros(n, int)
for i in range(n):
    key = f"{m['session'][i]}|{m['task'][i]}|{int(m['run'][i])}"
    sig = runs[key]["sig"].astype(float); ro = int(m["run_offset"][i])
    win = sig[:, ro:ro + 768]; mu = win.mean(axis=1, keepdims=True)
    lo = max(0, ro - H)
    hist_len[i] = ro - lo
    hist = sig[:, lo:ro] - mu if ro > lo else None
    b, v = stream_background(win - mu, hist, cfg)
    bg[i] = b.max(); valid[i] = v.max()
np.savez("true_hist.npz", background=bg, background_valid=valid, hist_len=hist_len)
print("valid fraction", valid.mean())
