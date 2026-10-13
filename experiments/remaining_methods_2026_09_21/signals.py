"""A bank of RTAMT-expressible features: many operators x many channels and pairs.

WHY THESE OPERATORS AND NOT THE WHOLE LIST

For one signal compared with a constant, most of RTAMT's operator list cannot change which
windows are selected:

  abs/sqrt/pow/exp/log   monotone: pow(s,2) >= t picks the same samples as s >= sqrt(t).
                         Only a renamed threshold, so it is covered by sweeping the threshold.
  implies / unless       sugar: (a implies b) == (not a or b); unless is sugar over until.
  always / eventually    inside a FIXED 3 s window, eventually[0:3] at the first sample selects
                         the same windows as once[0:3] at the last sample. The future operators
                         mirror the past ones the monitors already use.
  next / prev            one sample at 256 Hz; alone they shift a signal by 3.9 ms.

What does add expressive power, and is therefore what this bank covers:

  A. SUB-INTERVAL timing   once[a:b] over thirds / halves instead of the whole window
  B. ORDERING              since, until, rise, fall: WHICH excursion came first, how many there were
  C. CROSS-SIGNAL          xor, iff between two different signals (searched in run.py)
  D. CROSS-CHANNEL         arithmetic (a - b) over all 28 pairs, not just F4-T3

SIGNALS (each divided by its own robust spread inside the window, so every feature is unit-free)
  ch_<C>     the 8 channels, 0.5-6 Hz
  pr_<A><B>  all 28 channel differences, 0.3-6 Hz
  en_<C>     the 8 channels' 0.5 s running energy over the window's median energy

OPERATOR REDUCTIONS applied to every signal (the STL each one stands for is in OPS below)
usage: python signals.py
"""
import glob
import itertools
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

ROOT = Path(r"C:\Users\gaura\thesis_rtamt")
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
CHANNELS = ("Fp1", "Fp2", "F3", "Fz", "F4", "T3", "Cz", "T4")
PAIRS = list(itertools.combinations(CHANNELS, 2))          # all 28

# name -> the STL construct it is the threshold-free scalar of
OPS = {
    "max":    "once[0:3](s >= t)",
    "min":    "historically[0:3](s >= t)",
    "absmax": "once[0:3](abs(s) >= t)",
    "e1":     "once[0:1](s >= t)          (first third)",
    "e2":     "once[1:2](s >= t)          (middle third)",
    "e3":     "once[2:3](s >= t)          (last third)",
    "dwell2": "fraction of the window with abs(s) >= 2   (historically/duration)",
    "dwell3": "fraction of the window with abs(s) >= 3",
    "rise2":  "number of rise(s >= 2) events",
    "rise3":  "number of rise(s >= 3) events",
    "since2": "sign of the LAST excursion past +-2:  (not(s>=2)) since (s<=-2)",
    "since3": "sign of the LAST excursion past +-3",
    "first2": "when s first crosses +-2, as a fraction of the window (until-style ordering)",
    "last2":  "when s last crosses +-2",
    "order":  "+1 if the maximum comes before the minimum, -1 otherwise (biphasic order)",
    "sep":    "samples between the maximum and the minimum, as a fraction of the window",
    # ---- FUTURE and NESTED operators. Offline the whole window is available, so these are
    #      legitimate; they are NOT reducible to the past-time ones above, because they are
    #      evaluated at intermediate samples, not only at the last one.
    "run2":   "eventually(always[0:k](abs(s) >= 2)): longest UNBROKEN stretch above 2",
    "run3":   "eventually(always[0:k](abs(s) >= 3)): longest unbroken stretch above 3",
    "calm":   "eventually(always[0:k](abs(s) <= 1)): longest unbroken quiet stretch",
    "resp":   "always((s >= 2) implies eventually[0:k](s <= -2)): longest wait from a positive "
              "excursion to the next negative one (the biphasic response time)",
    "ford":   "which comes first, (s >= 2) or (s <= -2):  +1 / -1  (until-style ordering)",
    "slope":  "max abs(next s - s): the steepest one-sample change (next/prev)",
    "m1":     "always[2:3](s >= t) over the first third (min, the dual of once)",
    "m2":     "always[1:2](s >= t) over the middle third",
    "m3":     "always[0:1](s >= t) over the last third",
}


def _norm(x):
    """divide each row by its own robust spread -> unit-free, no per-subject constant"""
    x = x - np.median(x, axis=-1, keepdims=True)
    return x / (1.4826 * np.median(np.abs(x), axis=-1, keepdims=True) + 1e-12)


def _longest_run(mask):
    """longest unbroken stretch of True: the scalar behind eventually(always[0:k](phi))"""
    if not mask.any():
        return 0
    d = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def _reduce(s, out, name):
    """every operator reduction for one normalised signal s (1-D)"""
    n = len(s)
    third = n // 3
    out[f"{name}|max"] = s.max()
    out[f"{name}|min"] = s.min()
    a = np.abs(s)
    out[f"{name}|absmax"] = a.max()
    out[f"{name}|e1"] = s[:third].max()
    out[f"{name}|e2"] = s[third:2 * third].max()
    out[f"{name}|e3"] = s[2 * third:].max()
    for k in (2, 3):
        above = a >= k
        out[f"{name}|dwell{k}"] = above.mean()
        # rise(phi) == not(prev phi) and phi  -> count the 0->1 transitions
        out[f"{name}|rise{k}"] = float(np.count_nonzero(above[1:] & ~above[:-1]) + above[0])
        idx = np.flatnonzero(above)
        out[f"{name}|since{k}"] = float(np.sign(s[idx[-1]])) if idx.size else 0.0
    idx2 = np.flatnonzero(a >= 2)
    out[f"{name}|first2"] = idx2[0] / n if idx2.size else 1.0
    out[f"{name}|last2"] = idx2[-1] / n if idx2.size else 0.0
    hi, lo = int(np.argmax(s)), int(np.argmin(s))
    out[f"{name}|order"] = 1.0 if hi < lo else -1.0
    out[f"{name}|sep"] = abs(hi - lo) / n

    # ---- future / nested operators (offline: the whole window is available)
    for key, mask in (("run2", a >= 2), ("run3", a >= 3), ("calm", a <= 1)):
        out[f"{name}|{key}"] = _longest_run(mask) / n
    up, dn = np.flatnonzero(s >= 2), np.flatnonzero(s <= -2)
    # always((s>=2) implies eventually[0:k](s<=-2)): the longest wait before a reversal
    if up.size and dn.size:
        nxt = np.searchsorted(dn, up)
        wait = np.where(nxt < dn.size, dn[np.minimum(nxt, dn.size - 1)] - up, n)
        out[f"{name}|resp"] = float(wait.max()) / n
    else:
        out[f"{name}|resp"] = 1.0
    if up.size and dn.size:
        out[f"{name}|ford"] = 1.0 if up[0] < dn[0] else -1.0
    else:
        out[f"{name}|ford"] = 1.0 if up.size else (-1.0 if dn.size else 0.0)
    out[f"{name}|slope"] = float(np.abs(np.diff(s)).max())
    out[f"{name}|m1"] = s[:third].min()
    out[f"{name}|m2"] = s[third:2 * third].min()
    out[f"{name}|m3"] = s[2 * third:].min()


def session_job(path):
    from monitoring.rtamt_signals import (_BAND_CLASS, _BAND_DETECT, _BAND_DIRECTION, _IDX, FS,
                                          check_window)
    data = np.load(path, allow_pickle=True)
    epochs = data["epochs"][:, :8] * 1e6
    block = int(0.5 * FS)
    rows = []
    for raw in epochs:
        x = check_window(raw)
        out = {}
        cls = _norm(lfilter(_BAND_CLASS[0], _BAND_CLASS[1], x, axis=-1))
        for c in CHANNELS:
            _reduce(cls[_IDX[c]], out, f"ch_{c}")
        dr = lfilter(_BAND_DIRECTION[0], _BAND_DIRECTION[1], x, axis=-1)
        for a, b in PAIRS:
            _reduce(_norm(dr[_IDX[a]] - dr[_IDX[b]]), out, f"pr_{a}{b}")
        det = lfilter(_BAND_DETECT[0], _BAND_DETECT[1], x, axis=-1) ** 2
        cum = np.concatenate([np.zeros((8, 1)), np.cumsum(det, axis=1)], axis=1)
        en = np.zeros_like(det)
        en[:, block - 1:] = (cum[:, block:] - cum[:, :-block]) / block
        en = en / (np.median(en[:, block - 1:], axis=1, keepdims=True) + 1e-18)
        for c in CHANNELS:
            _reduce(en[_IDX[c]], out, f"en_{c}")
        rows.append(out)
    keys = sorted(rows[0])
    res = {k: np.array([r[k] for r in rows], dtype=np.float32) for k in keys}
    p = Path(path)
    res["session"] = np.array([f"{p.parent.parent.name}/{p.parent.name}"] * len(rows))
    res["label"] = data["labels"].astype(str)
    return res


if __name__ == "__main__":
    files = sorted(glob.glob(str(ROOT / "extracted_npz_dataset_v2" / "*" / "ses-*" / "clean_epochs.npz")),
                   key=lambda p: (p.split(os.sep)[-3], p.split(os.sep)[-2]))
    with Pool(6) as pool:
        parts = pool.map(session_job, files)
    res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez(HERE / "bank.npz", **res)
    feats = [k for k in res if k not in ("session", "label")]
    print(f"windows {len(res['session'])}  features {len(feats)}  "
          f"= {len(CHANNELS) * 2 + len(PAIRS)} signals x {len(OPS)} operators")
