"""Per-window features that need the stream history (rebuilt exactly as the monitor's evaluation does)."""
import sys; sys.path.insert(0, r"C:\Users\gaura\thesis_rtamt")
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from scipy.signal import lfilter
from monitoring.stream_history import windows_with_history
from monitoring.rtamt_signals import RtamtConfig, _BAND_DETECT
from feat import B_CLS, B_DIR, B_PROM, I, block_sums
DATA = Path(r"C:\Users\gaura\thesis_rtamt\extracted_npz_dataset_v2")
SETTLE, BG = 256, 1024

def one(window, hist):
    """window (8,768), hist (8,h) or None -> dict of scalars"""
    o = {}
    if hist is None or hist.shape[1] - max(SETTLE, hist.shape[1]-BG) < 256:
        o["valid"] = 0.0
        return o
    o["valid"] = 1.0
    h = hist.shape[1]; lo = max(SETTLE, h - BG)
    full = np.concatenate([hist, window], axis=1)
    # background exactly as the monitor
    det = lfilter(*_BAND_DETECT, full, axis=-1)
    c = np.concatenate([np.zeros((8,1)), np.cumsum(det**2, axis=1)], axis=1)
    e = np.zeros_like(det); e[:, 127:] = (c[:, 128:] - c[:, :-128]) / 128
    ref = np.median(e[:, lo:h:8], axis=1) + 1e-12
    o["background"] = float(np.mean(e[:, h:h+768] / ref[:, None], axis=0).max())
    # component rest references in the classification band, filtered continuously
    cl = lfilter(*B_CLS, full, axis=-1)
    comps = {"cm": cl[I["F3"]] + cl[I["F4"]], "df": cl[I["F3"]] - cl[I["F4"]],
             "mid": None, "tmp": None}
    rest = cl[:, lo:h]
    # mean energy over the rest stretch
    E = np.mean(rest**2, axis=1)
    o["R_mid_mean"] = E[I["Fz"]] + E[I["Cz"]]; o["R_tmp_mean"] = E[I["T3"]] + E[I["T4"]]
    o["R_cm_mean"] = np.mean((rest[I["F3"]] + rest[I["F4"]])**2); o["R_df_mean"] = np.mean((rest[I["F3"]] - rest[I["F4"]])**2)
    # median of 0.5 s block energies (robust to a blink in the history)
    def medblk(x):
        b = block_sums(x**2, 128) / 128
        return float(np.median(b[::8]))
    o["R_mid_med"] = medblk(rest[I["Fz"]]) + medblk(rest[I["Cz"]]); o["R_tmp_med"] = medblk(rest[I["T3"]]) + medblk(rest[I["T4"]])
    o["R_cm_med"] = medblk(rest[I["F3"]] + rest[I["F4"]]); o["R_df_med"] = medblk(rest[I["F3"]] - rest[I["F4"]])
    for ch in ("Fp1","Fp2","F3","Fz","F4","T3","Cz","T4"):
        o[f"R_{ch}_med"] = medblk(rest[I[ch]])
    return o

def session(path):
    rows = []
    for k, w, hist in windows_with_history(path, config=RtamtConfig()):
        rows.append(one(w, hist))
    keys = sorted({k for r in rows for k in r})
    return {k: np.array([r.get(k, np.nan) for r in rows]) for k in keys}

if __name__ == "__main__":
    files = [str(p) for p in sorted(DATA.glob("*/ses-*/clean_epochs.npz"))]
    with Pool(6) as pool:
        parts = pool.map(session, files)
    keys = sorted({k for p in parts for k in p})
    out = {}
    for k in keys:
        out[k] = np.concatenate([p.get(k, np.full(len(p["valid"]), np.nan)) for p in parts])
    np.savez("feat_hist.npz", **out)
    f = dict(np.load(r"C:\Users\gaura\thesis_rtamt\measurements_v2.npz", allow_pickle=True))
    v = out["valid"] > 0.5
    print("valid", v.mean(), "cache valid", (f["background_valid"]>0.5).mean(), "agree", np.mean(v == (f["background_valid"]>0.5)))
    print("background max abs diff", np.nanmax(np.abs(out["background"][v] - f["background"][v])))
