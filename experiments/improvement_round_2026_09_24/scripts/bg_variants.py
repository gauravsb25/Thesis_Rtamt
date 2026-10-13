"""Background gate variants under a given history source. Variant = (settle s, min s, init) where init is
'zero' (the shipped cold start) or 'zi' (steady-state start at the first history sample, scipy lfilter_zi)."""
import sys; sys.path.insert(0, r"C:\Users\gaura\thesis_rtamt")
import numpy as np, pickle
from pathlib import Path
from multiprocessing import Pool
from scipy.signal import lfilter, lfilter_zi
from monitoring.rtamt_signals import _BAND_DETECT
from monitoring.stream_history import windows_with_history
DATA = Path(r"C:\Users\gaura\thesis_rtamt\extracted_npz_dataset_v2")
VARIANTS = [(1.0, 1.0, "zero"), (0.5, 0.5, "zero"), (0.25, 0.5, "zero"), (0.0, 0.5, "zero"),
            (1.0, 1.0, "zi"), (0.25, 0.5, "zi"), (0.0, 0.5, "zi"), (0.0, 0.25, "zi"), (0.0, 1.0, "zi")]
ZI = lfilter_zi(*_BAND_DETECT)
def bg(window, hist, settle, minimum, init):
    """exactly monitoring.rtamt_signals.stream_background, plus the init option; hist may be None"""
    if hist is None: return 0.0, 0.0
    start = hist.shape[1]
    lo = max(int(round(settle*256)), start - 1024)
    if start - lo < int(round(minimum*256)): return 0.0, 0.0
    x = np.concatenate([hist, window], axis=1)
    if init == "zi":
        band, _ = lfilter(*_BAND_DETECT, x, axis=-1, zi=ZI[None, :] * x[:, :1])
    else:
        band = lfilter(*_BAND_DETECT, x, axis=-1)
    c = np.concatenate([np.zeros((8, 1)), np.cumsum(band**2, axis=1)], axis=1)
    e = np.zeros_like(band); e[:, 127:] = (c[:, 128:] - c[:, :-128]) / 128
    ref = np.median(e[:, lo:start:8], axis=1) + 1e-12
    return float(np.mean(e[:, start:start+768] / ref[:, None], axis=0).max()), 1.0
def session_rebuilt(path):
    out = np.zeros((len(VARIANTS), 2, 0)).tolist()
    rows = []
    for k, w, hist in windows_with_history(path, seconds=5.0):
        rows.append([bg(w, hist, *v) for v in VARIANTS])
    return np.array(rows)       # (n, V, 2)
if __name__ == "__main__":
    files = [str(p) for p in sorted(DATA.glob("*/ses-*/clean_epochs.npz"))]
    with Pool(6) as pool:
        parts = pool.map(session_rebuilt, files)
    R = np.concatenate(parts)
    # true history from the CSV runs
    runs = pickle.load(open("runs.pkl", "rb")); m = dict(np.load("meta.npz", allow_pickle=True))
    T = np.zeros_like(R)
    for i in range(len(m["label"])):
        sig = runs[f"{m['session'][i]}|{m['task'][i]}|{int(m['run'][i])}"]["sig"]; ro = int(m["run_offset"][i])
        win = sig[:, ro:ro+768].astype(float); mu = win.mean(1, keepdims=True); lo = max(0, ro - 1280)
        hist = (sig[:, lo:ro].astype(float) - mu) if ro > lo else None
        T[i] = [bg(win - mu, hist, *v) for v in VARIANTS]
    np.savez("bg_variants.npz", rebuilt=R, true=T, variants=np.array([str(v) for v in VARIANTS]))
    print(R.shape, T.shape)
