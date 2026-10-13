"""The 15 proposed methods as per-window signals (extracted_npz_dataset_v2, with the stream before it).

Group 1 subject-adaptive normalisation
  hist_z        max over the window of mean_c |x_c - median_history| / IQR_history   (method 1; the history
                is the stream before the window, up to 5 s -- the runs are too short for 30 s)
  hist_valid    1 when at least 1 s of history was available
  rel_share_*   variance share of a channel group in the loudest 1 s (method 2): temporal T3+T4,
                midline Fz+Cz, frontal Fp1+Fp2+F3+F4, and the largest single-channel share
  delta_share   largest over slices of the 1-4 Hz share of 0.5-30 Hz power, averaged over channels (method 3)
Group 2 direction without polarity
  power_asym    (P(F4)+P(T4) - P(F3)-P(T3)) / (sum), band power in the loudest 1.5 s          (method 4)
  phase_f, phase_t   zero-lag correlation of F3 with F4, and of T3 with T4, in the loudest 1 s (method 5)
  mag_imbalance (|F4-Fz| - |F3-Fz|) / (|F4-Fz| + |F3-Fz|), largest excursions                  (method 6)
  centroid, centroid_shift   lateral centre of mass of the channel variances, and its shift from the
                quietest second to the loudest second (method 7; 10-20 x coordinates from MNE)
Group 3 nods against turns
  cm_diff       Var(mean of the 8 channels) / mean(Var(F4-F3), Var(T4-T3))                     (method 8)
  coh_ratio     coherence(Fz,Cz) / coherence(T3,T4), averaged over 1-4 Hz                      (method 9)
  vel_gradient  mean(|d/dt Fp1|, |d/dt Fp2|) / mean(|d/dt Cz|, |d/dt Fz|)                      (method 10)
Group 4 artifacts
  ocular_decay, ocular_decay_loud   (E(Fp1)+E(Fp2)) / (E(Cz)+E(Fz))                            (method 11)
  flatness      spectral flatness (geometric mean / arithmetic mean of the power spectrum)     (method 12)
  curvature     max|second derivative| / max|first derivative|, averaged over channels         (method 13)
Group 5 is STL structure, computed in run.py from these and the shipped measurements (methods 14, 15).

usage: python features.py
"""
import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.signal import coherence, lfilter

ROOT = Path(r"C:\Users\gaura\thesis_rtamt")
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
GROUPS = dict(temporal=("T3", "T4"), midline=("Fz", "Cz"), frontal=("Fp1", "Fp2", "F3", "F4"))


def lateral_positions():
    import mne
    pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    names = {"T3": "T7", "T4": "T8"}
    from monitoring.rtamt_signals import CHANNELS
    return np.array([pos[names.get(c, c)][0] / np.linalg.norm(pos[names.get(c, c)]) for c in CHANNELS])


def session_job(path):
    from monitoring.rtamt_signals import (_BAND_CLASS, _BAND_DETECT, _BAND_DIRECTION, _BAND_PROMINENCE, _IDX,
                                          CHANNELS, FS, RtamtConfig, _block_sums, _pooled_energy)
    from monitoring.stream_history import windows_with_history
    cfg = RtamtConfig()
    X = lateral_positions()
    rows = []
    for k, window, history in windows_with_history(path, config=cfg):
        x = np.asarray(window, dtype=float)
        energy = _pooled_energy(x)
        pro = lfilter(_BAND_PROMINENCE[0], _BAND_PROMINENCE[1], x, axis=-1)
        dr = lfilter(_BAND_DIRECTION[0], _BAND_DIRECTION[1], x, axis=-1)
        cls = lfilter(_BAND_CLASS[0], _BAND_CLASS[1], x, axis=-1)
        cls = cls - np.median(cls, axis=1, keepdims=True)
        w10, w15 = int(cfg.symmetry_seconds * FS), int(cfg.direction_seconds * FS)
        blocks10 = _block_sums(energy, w10)
        s10, q10 = int(np.argmax(blocks10)), int(np.argmin(blocks10))
        s15 = int(np.argmax(_block_sums(energy, w15)))
        loud = slice(s10, s10 + w10)
        quiet = slice(q10, q10 + w10)
        row = {}

        # ---- method 1: normalise by the history's own median and IQR
        row["hist_valid"] = 0.0
        row["hist_z"] = 0.0
        if history is not None and history.shape[1] >= FS:
            det_hist = lfilter(_BAND_DETECT[0], _BAND_DETECT[1], history, axis=-1)[:, FS // 2:]
            if det_hist.shape[1] >= FS // 2:
                det = lfilter(_BAND_DETECT[0], _BAND_DETECT[1], x, axis=-1)
                med = np.median(det_hist, axis=1, keepdims=True)
                iqr = (np.percentile(det_hist, 75, axis=1) - np.percentile(det_hist, 25, axis=1))[:, None] + 1e-12
                row["hist_z"] = float(np.mean(np.abs(det - med) / iqr, axis=0).max())
                row["hist_valid"] = 1.0

        # ---- method 2: variance shares inside the loudest second
        var = np.var(cls[:, loud], axis=1)
        share = var / (var.sum() + 1e-18)
        for name, chans in GROUPS.items():
            row[f"rel_share_{name}"] = float(share[[_IDX[c] for c in chans]].sum())
        row["rel_share_max"] = float(share.max())

        # ---- method 3: 1-4 Hz share of 0.5-30 Hz power, per slice
        width, step = 128, 64
        taper = np.hanning(width)
        freq = np.fft.rfftfreq(width, 1.0 / FS)
        band, total = (freq >= 1.0) & (freq < 4.0), (freq >= 0.5) & (freq <= 30.0)
        shares = []
        for start in range(0, x.shape[1] - width + 1, step):
            seg = x[:, start:start + width]
            seg = seg - seg.mean(axis=1, keepdims=True)
            power = np.abs(np.fft.rfft(seg * taper, axis=1)) ** 2
            shares.append(float(np.mean(power[:, band].sum(1) / (power[:, total].sum(1) + 1e-18))))
        row["delta_share"] = max(shares)
        row["delta_share_range"] = max(shares) - min(shares)

        # ---- method 4: hemisphere power asymmetry (positive quantities: no polarity)
        power = np.var(pro[:, s15:s15 + w15], axis=1)
        right = power[_IDX["F4"]] + power[_IDX["T4"]]
        left = power[_IDX["F3"]] + power[_IDX["T3"]]
        row["power_asym"] = float((right - left) / (right + left + 1e-18))

        # ---- method 5: zero-lag correlation of the symmetric pairs
        def corr(a, b):
            u, v = dr[_IDX[a], loud], dr[_IDX[b], loud]
            u, v = u - u.mean(), v - v.mean()
            return float(np.sum(u * v) / (np.sqrt(np.sum(u ** 2) * np.sum(v ** 2)) + 1e-18))
        row["phase_f"], row["phase_t"] = corr("F3", "F4"), corr("T3", "T4")

        # ---- method 6: excursion away from the midline
        def excursion(a):
            v = dr[_IDX[a], s15:s15 + w15] - dr[_IDX["Fz"], s15:s15 + w15]
            return float(np.abs(v - np.median(v)).max())
        e4, e3 = excursion("F4"), excursion("F3")
        row["mag_imbalance"] = (e4 - e3) / (e4 + e3 + 1e-18)

        # ---- method 7: lateral centre of mass of the variance
        def centroid(sl):
            v = np.var(pro[:, sl], axis=1)
            return float(np.sum(X * v) / (np.sum(v) + 1e-18))
        row["centroid"] = centroid(loud)
        row["centroid_shift"] = row["centroid"] - centroid(quiet)

        # ---- method 8: common mode against differential
        common = np.var(pro[:, loud].mean(axis=0))
        diff = 0.5 * (np.var(pro[_IDX["F4"], loud] - pro[_IDX["F3"], loud])
                      + np.var(pro[_IDX["T4"], loud] - pro[_IDX["T3"], loud]))
        row["cm_diff"] = float(common / (diff + 1e-18))

        # ---- method 9: coherence of the midline against the temples
        def band_coherence(a, b):
            f, c = coherence(x[_IDX[a]], x[_IDX[b]], fs=FS, nperseg=256)
            m = (f >= 1.0) & (f <= 4.0)
            return float(np.mean(c[m]))
        row["coh_ratio"] = band_coherence("Fz", "Cz") / (band_coherence("T3", "T4") + 1e-12)

        # ---- method 10: front-to-centre velocity gradient
        speed = np.abs(np.diff(cls[:, loud], axis=1)).mean(axis=1)
        row["vel_gradient"] = float((speed[_IDX["Fp1"]] + speed[_IDX["Fp2"]])
                                    / (speed[_IDX["Cz"]] + speed[_IDX["Fz"]] + 1e-18))

        # ---- method 11: anterior-to-central decay
        def decay(sl):
            e = np.var(pro[:, sl], axis=1)
            return float((e[_IDX["Fp1"]] + e[_IDX["Fp2"]]) / (e[_IDX["Cz"]] + e[_IDX["Fz"]] + 1e-18))
        row["ocular_decay"], row["ocular_decay_loud"] = decay(slice(0, x.shape[1])), decay(loud)

        # ---- method 12: spectral flatness of the loudest second
        seg = x[:, loud] - x[:, loud].mean(axis=1, keepdims=True)
        power = np.abs(np.fft.rfft(seg * np.hanning(seg.shape[1]), axis=1)) ** 2
        f = np.fft.rfftfreq(seg.shape[1], 1.0 / FS)
        m = (f >= 0.5) & (f <= 45.0)
        p = power[:, m].mean(axis=0) + 1e-18
        row["flatness"] = float(np.exp(np.mean(np.log(p))) / np.mean(p))

        # ---- method 13: curvature against velocity
        first = np.abs(np.diff(cls[:, loud], axis=1)).max(axis=1)
        second = np.abs(np.diff(cls[:, loud], n=2, axis=1)).max(axis=1)
        row["curvature"] = float(np.mean(second / (first + 1e-12)))
        rows.append(row)
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    out["session"] = np.array([f"{path.parent.parent.name}/{path.parent.name}"] * len(rows))
    out["label"] = data["labels"].astype(str)
    return out


if __name__ == "__main__":
    files = sorted(glob.glob(str(ROOT / "extracted_npz_dataset_v2" / "*" / "ses-*" / "clean_epochs.npz")),
                   key=lambda p: (p.split(os.sep)[-3], p.split(os.sep)[-2]))
    with Pool(6) as pool:
        parts = pool.map(session_job, files)
    res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez(HERE / "features.npz", **res)
    print("windows", len(res["session"]), "| history usable for", int(res["hist_valid"].sum()))
