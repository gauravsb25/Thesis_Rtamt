"""Measure every window of extracted_npz_dataset_v2 on the RAW (pre-ICA) and the ICA-cleaned recordings.

Each window is cut from its own run's CSV at the stored run_offset (the same 768 samples in both copies), mean
removed, with the REAL preceding 5 s of that run as history (no rebuilt history). For each window: the monitor's
measurements (as project_code.measure) plus two blink features read from the loudest 0.5 s:
    fp_dom     log( mean energy of Fp1,Fp2 / mean energy of T3,T4 ) in the loudest 0.5-s block (0.5-3 Hz),
               each channel's energy divided by its own median 0.5-s energy in the window (unit-free)
    event_w    width in seconds of the pooled-energy event around its peak, at half its height
A blink is forehead-dominant and short; a head movement drives the temples.
"""
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.signal import lfilter
from monitoring.rtamt_signals import CHANNELS, FS, RtamtConfig, _BAND_DETECT, _pooled_energy, compute_signals
from monitoring.offline_monitor import dataset_recording_path, load_recording

DATA = ROOT / "extracted_npz_dataset_v2"
ICA = ROOT / "dataset" / "eeg_cleaned_data"
RAW = Path("D:/errp/eeg_cleaned_data")
W, HIST = 3 * FS, 5 * FS
I = {c: k for k, c in enumerate(CHANNELS)}
KEYS = ("contrast", "background", "background_valid", "delta", "deflect", "nod", "sideways", "turn")


def load_raw(session, task, run):
    subject, ses = session.split("/")
    p = RAW / subject / ses / "pp_eeg" / f"{subject.replace('_', '-')}_{ses}_task-{task}_run-{int(run):02d}_eeg.csv"
    with open(p, newline="") as fh:
        r = csv.reader(fh)
        head = next(r)
        cols = [head.index(c) for c in CHANNELS]
        return np.array([[float(row[c]) for c in cols] for row in r if row]).T


def blink_features(x):
    band = lfilter(*_BAND_DETECT, x, axis=-1)
    cum = np.concatenate([np.zeros((8, 1)), np.cumsum(band ** 2, axis=1)], axis=1)
    e = (cum[:, 128:] - cum[:, :-128]) / 128                      # (8, 641) energy of each 0.5-s block
    e = e / (np.median(e, axis=1, keepdims=True) + 1e-18)
    pooled = _pooled_energy(x)
    blocks = np.convolve(pooled, np.ones(128), "valid")
    k = int(np.argmax(blocks))
    fp = np.mean(e[[I["Fp1"], I["Fp2"]], k])
    tt = np.mean(e[[I["T3"], I["T4"]], k])
    fp_dom = float(np.log((fp + 1e-12) / (tt + 1e-12)))
    smooth = np.convolve(pooled, np.ones(16) / 16, "same")
    peak = int(np.argmax(smooth))
    half = smooth[peak] / 2
    lo = peak
    while lo > 0 and smooth[lo] > half:
        lo -= 1
    hi = peak
    while hi < len(smooth) - 1 and smooth[hi] > half:
        hi += 1
    return fp_dom, (hi - lo) / FS


def session(path):
    cfg = RtamtConfig()
    p = Path(path)
    ses = f"{p.parent.parent.name}/{p.parent.name}"
    d = np.load(path, allow_pickle=True)
    task, run, off = d["task"].astype(str), d["run"].astype(int), d["run_offset"].astype(int)
    n = len(task)
    out = {src: {k: np.zeros(n) for k in KEYS + ("fp_dom", "event_w")} for src in ("raw", "ica")}
    for t, r in sorted(set(zip(task, run))):
        recs = {"raw": load_raw(ses, t, r), "ica": load_recording(dataset_recording_path(ICA, ses, t, r))}
        for k in np.flatnonzero((task == t) & (run == r)):
            o = int(off[k])
            for src, rec in recs.items():
                raw = rec[:, o:o + W]
                mean = raw.mean(axis=1, keepdims=True)
                win = raw - mean
                hist = rec[:, max(0, o - HIST):o] - mean if o > 0 else None
                s = compute_signals(win, cfg, hist)
                m = out[src]
                for key in ("contrast", "background", "background_valid", "delta", "deflect", "sideways"):
                    m[key][k] = s[key].max()
                m["nod"][k], m["turn"][k] = s["nod"][-1], s["turn"][-1]
                m["fp_dom"][k], m["event_w"][k] = blink_features(win)
    return ses, d["labels"].astype(str), out


if __name__ == "__main__":
    files = sorted(DATA.glob("*/ses-*/clean_epochs.npz"))
    with Pool(6) as pool:
        parts = pool.map(session, [str(f) for f in files])
    save = {"label": np.concatenate([p[1] for p in parts]),
            "session": np.concatenate([[p[0]] * len(p[1]) for p in parts])}
    for src in ("raw", "ica"):
        for key in KEYS + ("fp_dom", "event_w"):
            save[f"{src}_{key}"] = np.concatenate([p[2][src][key] for p in parts])
    np.savez(HERE / "measurements_raw_ica.npz", **save)
    print("saved", len(save["label"]), "windows")
