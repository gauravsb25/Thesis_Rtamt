"""The four proposed methods, as new per-window signals (extracted_npz_dataset_v2, with stream history).

All signals are unit-free ratios, so no per-subject constant enters.

1. DIRECTION (multi-pair bilateral consensus)
   lateral_imbalance = (excursion(F4-T4) - excursion(F3-T3)) / (spread(F4-T4) + spread(F3-T3))
       excursion = largest |deviation| in the loudest 1.5 s, spread = robust spread of that pair
   lat_signed        = the same but with the SIGNED peaks (a magnitude difference cannot give a side)
   dir_frontal       = signed peak of F4-F3 in the loudest 1.5 s / its spread
   dir_temporal      = signed peak of T4-T3 in the loudest 1.5 s / its spread
2. TURNS CALLED DOWN (yaw versus pitch)
   yaw_ratio      = [energy(T3-T4) + energy(F4-F3)] / [energy(Fz) + energy(Cz)]   (0.3-4 Hz)
   yaw_ratio_loud = the same, restricted to the loudest 1 s
3. BLINKS CALLED DOWN
   ocular_ratio      = energy(Fp1+Fp2) / energy(Fz+Cz)          (0.3-4 Hz, whole window)
   ocular_ratio_loud = the same, restricted to the loudest 1 s
   bg_sustained_K    = max over the window of the smallest background value in the last K samples
                       (the STL "once(historically[0:K-1](background >= b))"): a blink is too brief to hold
4. WEAK NODS
   midline_deflect = largest |excursion| of Fz and Cz over their own robust spread (0.5-6 Hz)
   midline_deflect4 = the same over Fz, Cz, Fp1, Fp2

usage: python features.py
"""
import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.signal import lfilter

ROOT = Path(r"C:\Users\gaura\thesis_rtamt")
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
SUSTAIN = (32, 76, 128, 192)          # samples: 0.125 s, 0.3 s, 0.5 s, 0.75 s


def session_job(path):
    from monitoring.rtamt_signals import (_BAND_CLASS, _BAND_DIRECTION, _BAND_PROMINENCE, _IDX, FS,
                                          RtamtConfig, _block_sums, _pooled_energy, compute_signals)
    from monitoring.stream_history import windows_with_history
    cfg = RtamtConfig()
    out = {}
    rows = []
    for k, window, history in windows_with_history(path, config=cfg):
        s = compute_signals(window, cfg, history)
        x = np.asarray(window, dtype=float)
        energy = _pooled_energy(x)
        pro = lfilter(_BAND_PROMINENCE[0], _BAND_PROMINENCE[1], x, axis=-1)
        dr = lfilter(_BAND_DIRECTION[0], _BAND_DIRECTION[1], x, axis=-1)
        cls = lfilter(_BAND_CLASS[0], _BAND_CLASS[1], x, axis=-1)
        cls = cls - np.median(cls, axis=1, keepdims=True)
        w15 = int(cfg.direction_seconds * FS)
        s15 = int(np.argmax(_block_sums(energy, w15)))
        w10 = int(cfg.symmetry_seconds * FS)
        s10 = int(np.argmax(_block_sums(energy, w10)))

        def pair_peak(sig, a, b, start, width):
            v = (sig[_IDX[a]] - sig[_IDX[b]])[start:start + width]
            v = v - np.median(v)
            spread = 1.4826 * float(np.median(np.abs(v))) + 1e-12
            peak = float(v[int(np.argmax(np.abs(v)))])
            return peak, spread

        # --- 1. direction
        p_f4t4, s_f4t4 = pair_peak(pro, "F4", "T4", s15, w15)
        p_f3t3, s_f3t3 = pair_peak(pro, "F3", "T3", s15, w15)
        imbalance = (abs(p_f4t4) - abs(p_f3t3)) / (s_f4t4 + s_f3t3)
        signed = (p_f4t4 - p_f3t3) / (s_f4t4 + s_f3t3)
        p_front, s_front = pair_peak(dr, "F4", "F3", s15, w15)
        p_temp, s_temp = pair_peak(dr, "T4", "T3", s15, w15)

        # --- 2 and 3. energy ratios (0.3-4 Hz)
        def energy_of(channel, start=0, width=768):
            return float(np.mean(pro[_IDX[channel], start:start + width] ** 2))

        def pair_energy(a, b, start=0, width=768):
            v = pro[_IDX[a], start:start + width] - pro[_IDX[b], start:start + width]
            return float(np.mean(v ** 2))

        def ratios(start, width):
            yaw = (pair_energy("T3", "T4", start, width) + pair_energy("F4", "F3", start, width)) / \
                  (energy_of("Fz", start, width) + energy_of("Cz", start, width) + 1e-18)
            ocular = (energy_of("Fp1", start, width) + energy_of("Fp2", start, width)) / \
                     (energy_of("Fz", start, width) + energy_of("Cz", start, width) + 1e-18)
            return yaw, ocular

        yaw, ocular = ratios(0, 768)
        yaw_loud, ocular_loud = ratios(s10, w10)

        # --- 3b. how long the background stays above its level (the STL persistence gate)
        row = dict(lateral_imbalance=imbalance, lat_signed=signed,
                   dir_frontal=p_front / s_front, dir_temporal=p_temp / s_temp,
                   yaw_ratio=yaw, yaw_ratio_loud=yaw_loud,
                   ocular_ratio=ocular, ocular_ratio_loud=ocular_loud)
        for K in SUSTAIN:
            row[f"bg_sustained_{K}"] = float(minimum_filter1d(s["background"], K, origin=(K - 1) // 2,
                                                              mode="nearest").max())
            row[f"contrast_sustained_{K}"] = float(minimum_filter1d(s["contrast"], K, origin=(K - 1) // 2,
                                                                    mode="nearest").max())
        # --- 4. midline prominence
        spread = 1.4826 * np.median(np.abs(cls), axis=1) + 1e-12
        prominence = np.abs(cls).max(axis=1) / spread
        row["midline_deflect"] = float(np.mean(prominence[[_IDX["Fz"], _IDX["Cz"]]]))
        row["midline_deflect4"] = float(np.mean(prominence[[_IDX["Fz"], _IDX["Cz"], _IDX["Fp1"], _IDX["Fp2"]]]))
        rows.append(row)
    out = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    path = Path(path)
    out["session"] = np.array([f"{path.parent.parent.name}/{path.parent.name}"] * len(rows))
    out["index"] = np.arange(len(rows))
    return out


if __name__ == "__main__":
    files = sorted(glob.glob(str(ROOT / "extracted_npz_dataset_v2" / "*" / "ses-*" / "clean_epochs.npz")),
                   key=lambda p: (p.split(os.sep)[-3], p.split(os.sep)[-2]))
    with Pool(6) as pool:
        parts = pool.map(session_job, files)
    res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez(HERE / "features.npz", **res)
    print("windows", len(res["session"]))
