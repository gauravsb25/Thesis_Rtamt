"""The onset-timing and unit-free features from the old classify.py, on the current dataset.

WHY THIS ROUND EXISTS

`_archive/pipeline/classify.py` carries two things the current monitor does not use:

  1. RULES with absolute thresholds (T3T4_peak_signed >= 63.677, Cz_ptp <= 40.058, ...). Most are
     in microvolts, and that file's own comment records that sub_11 runs 2-2.6x hotter than sub_01,
     so those cannot transfer. Only five of its features are unit-free and therefore worth testing:
     T3T4_skew, T3_normpeak, Cz/T3_ratio, T4_spec_centroid (a frequency, in Hz) and
     temporal~central_gonsetlag.

  2. ONSET TIMING -- when each electrode first "goes high" -- and the claim, quoted from that file:
        "Cz fires ~110ms AFTER the temporal pair on DOWN but BEFORE it on LEFT/RIGHT
         (consistent on both subjects)."
     Cross-subject consistency is the one property nothing else in this project has, so it is worth
     a proper test.

WHAT WAS ALREADY TESTED, SO THIS DOES NOT REPEAT IT

`onset-order-does-not-add` (2026-09-03) measured onset ORDER as a LATERALITY tie-breaker, on the
superseded `monitoring/live_monitor.py`. This round is a different decision (DOWN versus turn) on
the current RTAMT monitor, so it is not the same experiment.

Definitions are copied from `_archive/lib/features.py` and `_archive/pipeline/classify.py` so a
feature cannot mean something different here than it did there.
usage: python features.py
"""
import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\gaura\thesis_rtamt")
HERE = Path(__file__).resolve().parent
CHANNELS = ("Fp1", "Fp2", "F3", "Fz", "F4", "T3", "Cz", "T4")
IDX = {c: i for i, c in enumerate(CHANNELS)}
FS = 256
ONSET_Z = 2.0                                   # identical to classify.py
ONSET_GROUPS = {
    "frontopolar": ["Fp1", "Fp2"],
    "frontal": ["F3", "Fz", "F4"],
    "temporal": ["T3", "T4"],
    "central": ["Cz"],
    "midline": ["Fz", "Cz"],
}
ONSET_GROUP_PAIRS = [
    ("temporal", "central"), ("temporal", "midline"), ("frontal", "temporal"),
    ("frontopolar", "temporal"), ("frontopolar", "central"),
]


def channel_onsets(x):
    """First TIME (seconds) each channel's robust z-score reaches ONSET_Z. Copied from classify.py."""
    t = np.arange(x.shape[1]) / FS
    out = {}
    for c in CHANNELS:
        v = x[IDX[c]]
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        mad = mad if mad > 1e-9 else (v.std() + 1e-9)
        z = np.abs((v - med) / mad)
        hits = np.flatnonzero(z >= ONSET_Z)
        out[c] = float(t[hits[0]]) if hits.size else np.nan
    return out


def window_features(x):
    row = {}
    onsets = channel_onsets(x)
    for c in CHANNELS:
        row[f"onset_{c}"] = onsets[c]
    gonset = {}
    for name, members in ONSET_GROUPS.items():
        vals = [onsets[c] for c in members if not np.isnan(onsets[c])]
        gonset[name] = float(np.mean(vals)) if vals else np.nan
    for a, b in ONSET_GROUP_PAIRS:
        va, vb = gonset[a], gonset[b]
        row[f"lag_{a}~{b}"] = (va - vb) if not (np.isnan(va) or np.isnan(vb)) else np.nan
    # the single-pair lateralisation lag the old file also reported
    row["lag_T3_T4"] = (onsets["T3"] - onsets["T4"]
                        if not (np.isnan(onsets["T3"]) or np.isnan(onsets["T4"])) else np.nan)

    # ---- the five unit-free features from SELECTED_RULES
    t3, t4, cz = x[IDX["T3"]], x[IDX["T4"]], x[IDX["Cz"]]
    d = t3 - t4
    dm = d - d.mean()
    sd = dm.std()
    row["T3T4_skew"] = float(np.mean(dm ** 3) / (sd ** 3)) if sd > 1e-12 else 0.0
    overall_rms = float(np.sqrt(np.mean(x ** 2)))
    row["T3_normpeak"] = float(np.abs(t3).max() / overall_rms) if overall_rms > 0 else 0.0
    pz, p3 = float(np.abs(cz).max()), float(np.abs(t3).max())
    row["Cz_T3_ratio"] = (pz / p3) if p3 > 0 else np.nan
    spec = np.abs(np.fft.rfft(t4 - t4.mean())) ** 2
    freq = np.fft.rfftfreq(len(t4), 1.0 / FS)
    keep = (freq >= 0.5) & (freq <= 45)
    row["T4_spec_centroid"] = float((freq[keep] * spec[keep]).sum() / (spec[keep].sum() + 1e-18))
    return row


def session_job(path):
    data = np.load(path, allow_pickle=True)
    epochs = data["epochs"][:, :8] * 1e6
    rows = [window_features(np.asarray(e, float)) for e in epochs]
    out = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0]}
    p = Path(path)
    out["session"] = np.array([f"{p.parent.parent.name}/{p.parent.name}"] * len(rows))
    out["label"] = data["labels"].astype(str)
    return out


if __name__ == "__main__":
    files = sorted(glob.glob(str(ROOT / "extracted_npz_dataset_v2" / "*" / "ses-*" / "clean_epochs.npz")),
                   key=lambda p: (p.split(os.sep)[-3], p.split(os.sep)[-2]))
    with Pool(6) as pool:
        parts = pool.map(session_job, files)
    res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez(HERE / "onset.npz", **res)
    print(f"windows {len(res['session'])}  features "
          f"{len([k for k in res if k not in ('session', 'label')])}")
