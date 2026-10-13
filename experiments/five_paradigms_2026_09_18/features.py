"""Four new paradigms as per-window signals (extracted_npz_dataset_v2, with the stream before each window).

1. SURFACE LAPLACIAN. Each channel minus the average of its neighbours on the 10-20 grid, which removes
   what is common to the whole scalp (volume conduction, common-mode head motion) and keeps what is local:
       Fp1: Fp2,F3 | Fp2: Fp1,F4 | F3: Fp1,Fz,T3 | Fz: F3,F4,Cz | F4: Fp2,Fz,T4 | T3: F3,Cz | Cz: Fz,T3,T4 | T4: F4,Cz
   The monitor's own six measurements are then recomputed on those Laplacian channels: lap_contrast,
   lap_delta, lap_deflect, lap_nod, lap_sideways, lap_turn.
2. SYMBOLIC COMPLEXITY. Permutation entropy of order 3 (the order of three neighbouring samples), computed
   per channel over 0.5 s slices and averaged over channels: rest is disordered (high entropy), a movement is
   a smooth trajectory (low entropy). Scale-free and sign-free by construction.
       pe_min   the lowest entropy reached in the window
       pe_drop  the window's typical entropy minus that lowest value
3. SEQUENTIAL EVIDENCE (SPRT / CUSUM) ON THE RTAMT ROBUSTNESS TRACE. The gate's robustness per sample is
   r(t) = min(log(background/b) or log(contrast/g), (delta-d)/scale, log(deflect/q)). The accumulator
   S(t) = max(0, S(t-1) + r(t) - k) is stored as its largest value for several allowances k.
       sprt_<k>  the largest accumulated evidence in the window
       rob_max   max r(t), which is what the current specification effectively tests
4. HILBERT INSTANTANEOUS PHASE. The analytic signal of the 0.5-6 Hz band gives a phase for every sample.
       phase_diff   wrapped phase difference F3 - F4 at the loudest sample (a direction candidate)
       phase_jump   largest |dphi/dt| on Fp1/Fp2 over the window's median (blinks jump, movements glide)
       phase_jump_ratio  the same divided by the largest |dphi/dt| on Fz/Cz
       phase_rough  spread of |dphi/dt| in the loudest second over its median (roughness of the glide)

usage: python features.py
"""
import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.signal import hilbert, lfilter

ROOT = Path(r"C:\Users\gaura\thesis_rtamt")
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
NEIGHBOURS = {"Fp1": ("Fp2", "F3"), "Fp2": ("Fp1", "F4"), "F3": ("Fp1", "Fz", "T3"), "Fz": ("F3", "F4", "Cz"),
              "F4": ("Fp2", "Fz", "T4"), "T3": ("F3", "Cz"), "Cz": ("Fz", "T3", "T4"), "T4": ("F4", "Cz")}
K_VALUES = (0.0, 0.05, 0.1, 0.2)


def laplacian(x, idx):
    out = np.zeros_like(x)
    for name, neighbours in NEIGHBOURS.items():
        out[idx[name]] = x[idx[name]] - np.mean([x[idx[n]] for n in neighbours], axis=0)
    return out


def ordinal_codes(band):
    """Order-3 ordinal pattern of every triple, for all channels at once: (channels, samples-2)."""
    triples = np.stack([band[:, :-2], band[:, 1:-1], band[:, 2:]], axis=-1)      # (ch, n-2, 3)
    ranks = np.argsort(np.argsort(triples, axis=-1), axis=-1)
    return ranks[..., 0] * 9 + ranks[..., 1] * 3 + ranks[..., 2]                 # 0..26, six of them occur


def slice_entropy(codes):
    """Normalised order-3 permutation entropy of one (channels, samples) block of codes."""
    total = codes.shape[1]
    out = 0.0
    for channel in codes:
        counts = np.bincount(channel, minlength=27)
        p = counts[counts > 0] / total
        out += -float(np.sum(p * np.log(p))) / np.log(6)
    return out / codes.shape[0]


def session_job(path):
    from monitoring.rtamt_signals import _BAND_CLASS, _IDX, FS, RtamtConfig, _block_sums, _pooled_energy, compute_signals
    from monitoring.stream_history import windows_with_history
    cfg = RtamtConfig()
    scale_delta = 0.0685                      # robust spread of `delta` on sub_01 sessions 1-30
    rows = []
    for k, window, history in windows_with_history(path, config=cfg):
        x = np.asarray(window, dtype=float)
        s = compute_signals(x, cfg, history)
        row = {}

        # ---- 1. the monitor's measurements on Laplacian channels
        lap = compute_signals(laplacian(x, _IDX), cfg, None if history is None else laplacian(np.asarray(history), _IDX))
        row.update(lap_contrast=float(lap["contrast"].max()), lap_delta=float(lap["delta"].max()),
                   lap_deflect=float(lap["deflect"].max()), lap_nod=float(lap["nod"][-1]),
                   lap_sideways=float(lap["sideways"].max()), lap_turn=float(lap["turn"][-1]),
                   lap_background=float(lap["background"].max()))

        # ---- 2. permutation entropy over 0.5 s slices
        band = lfilter(_BAND_CLASS[0], _BAND_CLASS[1], x, axis=-1)
        codes = ordinal_codes(band)
        values = np.array([slice_entropy(codes[:, start:start + 126])
                           for start in range(0, codes.shape[1] - 126 + 1, 32)])
        row.update(pe_min=float(values.min()), pe_drop=float(np.median(values) - values.min()))

        # ---- 3. sequential evidence on the gate's robustness trace
        if s["background_valid"].max() >= 0.5:
            gate = np.log(np.maximum(s["background"], 1e-9) / cfg.background_ratio)
        else:
            gate = np.log(np.maximum(s["contrast"], 1e-9) / cfg.gate_ratio)
        r = np.minimum.reduce([gate, (s["delta"] - cfg.delta_range_min) / scale_delta,
                               np.log(np.maximum(s["deflect"], 1e-9) / cfg.quiet_min)])
        row["rob_max"] = float(r.max())
        for kk in K_VALUES:
            acc, best = 0.0, 0.0
            for value in r:
                acc = max(0.0, acc + float(value) - kk)
                best = max(best, acc)
            row[f"sprt_{kk}"] = best / FS                      # nat-seconds, like the earlier CUSUM signal

        # ---- 4. Hilbert instantaneous phase
        analytic = hilbert(band, axis=-1)
        phase = np.unwrap(np.angle(analytic), axis=-1)
        speed = np.abs(np.diff(phase, axis=-1)) * FS           # rad/s
        energy = _pooled_energy(x)
        w10 = int(cfg.symmetry_seconds * FS)
        s10 = int(np.argmax(_block_sums(energy, w10)))
        peak = s10 + int(np.argmax(np.abs(band[:, s10:s10 + w10]).mean(axis=0)))
        diff = np.angle(np.exp(1j * (np.angle(analytic[_IDX["F3"], peak]) - np.angle(analytic[_IDX["F4"], peak]))))
        front = np.max(speed[[_IDX["Fp1"], _IDX["Fp2"]]])
        centre = np.max(speed[[_IDX["Fz"], _IDX["Cz"]]])
        loud = speed[:, s10:min(s10 + w10, speed.shape[1])]
        row.update(phase_diff=float(diff),
                   phase_jump=float(front / (np.median(speed) + 1e-12)),
                   phase_jump_ratio=float(front / (centre + 1e-12)),
                   phase_rough=float(np.std(loud) / (np.median(loud) + 1e-12)))
        rows.append(row)
    out = {key: np.array([r[key] for r in rows]) for key in rows[0]}
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
    print("windows", len(res["session"]))
