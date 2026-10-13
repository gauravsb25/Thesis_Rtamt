"""Polarity-free direction signals and wave-shape signals for every window of extracted_npz_dataset_v2.

DIRECTION WITHOUT POLARITY (the voltage sign flips between sessions; timing does not)
  lag_f3f4    t_peak(F3) - t_peak(F4), seconds, inside the loudest 1.5 s (peak = largest |deflection|)
  lag_t3t4    the same for T3 - T4
  lag_env     lag that best aligns the LEFT side envelope with the RIGHT side envelope
              (envelope = |0.3-6 Hz signal| smoothed over 0.1 s, so the sign cannot matter),
              searched within +-0.5 s by cross-correlation
  lag_env_q   how good that alignment is (the correlation at the best lag), 0..1

WAVE SHAPE, NOT SPACE (to separate nods from turns without any left/right energy ratio)
  Computed on the pooled envelope of the loudest 1 s (mean over channels of |0.5-6 Hz| / own spread):
  rise        seconds from 10% to 90% of the peak, before the peak
  fall        seconds from the peak down to 10%, after it
  asym        (fall - rise) / (fall + rise): positive = slow decay, negative = slow build-up
  steep       largest slope of the envelope divided by its peak (1/s)
  width       seconds the envelope stays above half of its peak

usage: python features.py
"""
import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

ROOT = Path(r"C:\Users\gaura\thesis_rtamt")
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent


def session_job(path):
    from monitoring.rtamt_signals import (_BAND_CLASS, _BAND_DIRECTION, _IDX, FS, RtamtConfig,
                                          _block_sums, _pooled_energy, check_window)
    cfg = RtamtConfig()
    data = np.load(path, allow_pickle=True)
    epochs = data["epochs"][:, :8] * 1e6
    rows = []
    for raw in epochs:
        x = check_window(raw)
        energy = _pooled_energy(x)
        dr = lfilter(_BAND_DIRECTION[0], _BAND_DIRECTION[1], x, axis=-1)
        cls = lfilter(_BAND_CLASS[0], _BAND_CLASS[1], x, axis=-1)
        cls = cls - np.median(cls, axis=1, keepdims=True)
        w15 = int(cfg.direction_seconds * FS)
        s15 = int(np.argmax(_block_sums(energy, w15)))
        w10 = int(cfg.symmetry_seconds * FS)
        s10 = int(np.argmax(_block_sums(energy, w10)))
        seg = dr[:, s15:s15 + w15]
        seg = seg - np.median(seg, axis=1, keepdims=True)

        def peak_time(channel):
            return int(np.argmax(np.abs(seg[_IDX[channel]]))) / FS

        # --- envelopes of the two sides (polarity cannot matter: absolute value)
        smooth = np.ones(max(int(0.1 * FS), 1)) / max(int(0.1 * FS), 1)
        def envelope(channels):
            v = np.mean([np.abs(seg[_IDX[c]]) / (1.4826 * np.median(np.abs(seg[_IDX[c]])) + 1e-12)
                         for c in channels], axis=0)
            return np.convolve(v, smooth, "same")
        left, right = envelope(["F3", "T3"]), envelope(["F4", "T4"])
        left = left - left.mean()
        right = right - right.mean()
        span = int(0.5 * FS)
        norm = np.sqrt(float(np.sum(left ** 2) * np.sum(right ** 2))) + 1e-12
        lags = np.arange(-span, span + 1)
        scores = np.array([float(np.sum(left[max(k, 0):len(left) + min(k, 0)]
                                        * right[max(-k, 0):len(right) + min(-k, 0)])) / norm for k in lags])
        best = int(np.argmax(scores))

        # --- wave shape of the loudest second
        env = np.mean(np.abs(cls[:, s10:s10 + w10]) /
                      (1.4826 * np.median(np.abs(cls), axis=1)[:, None] + 1e-12), axis=0)
        env = np.convolve(env, smooth, "same")
        peak = int(np.argmax(env))
        top, low, half = env[peak], 0.1 * env[peak], 0.5 * env[peak]
        before = np.flatnonzero(env[:peak + 1] <= low)
        start = before[-1] if len(before) else 0
        ninety = np.flatnonzero(env[start:peak + 1] >= 0.9 * top)
        rise = (ninety[0] if len(ninety) else peak - start) / FS
        after = np.flatnonzero(env[peak:] <= low)
        fall = (after[0] if len(after) else len(env) - peak) / FS
        rows.append(dict(lag_f3f4=peak_time("F3") - peak_time("F4"),
                         lag_t3t4=peak_time("T3") - peak_time("T4"),
                         lag_env=float(lags[best]) / FS, lag_env_q=float(scores[best]),
                         rise=rise, fall=fall,
                         asym=(fall - rise) / (fall + rise + 1e-12),
                         steep=float(np.abs(np.diff(env)).max() * FS / (top + 1e-12)),
                         width=float(np.sum(env >= half)) / FS))
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    path = Path(path)
    out["session"] = np.array([f"{path.parent.parent.name}/{path.parent.name}"] * len(rows))
    out["label"] = data["labels"].astype(str)
    out["task"] = data["task"].astype(str)
    out["run"] = data["run"].astype(int)
    out["onset"] = np.where(np.isfinite(data["movement_offset"]),
                            data["run_offset"] + np.nan_to_num(data["movement_offset"]) * FS, -1).astype(int)
    return out


if __name__ == "__main__":
    files = sorted(glob.glob(str(ROOT / "extracted_npz_dataset_v2" / "*" / "ses-*" / "clean_epochs.npz")),
                   key=lambda p: (p.split(os.sep)[-3], p.split(os.sep)[-2]))
    with Pool(6) as pool:
        parts = pool.map(session_job, files)
    res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez(HERE / "features.npz", **res)
    print("windows", len(res["session"]))
