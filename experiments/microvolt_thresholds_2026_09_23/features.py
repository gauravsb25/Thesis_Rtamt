"""The same measurements the monitor makes, but in MICROVOLTS instead of unit-free ratios.

The shipped monitor divides everything by something from the same recording (its own spread, its own
quietest moment, its own recent background), so every threshold is a plain ratio. This round builds
the ABSOLUTE version of each measurement and re-fits every threshold on it, to measure what that
design choice is actually worth.

  unit-free (shipped)                      microvolt version built here
  contrast  loudest 0.5 s / quietest       uv_loud     loudest 0.5 s RMS, in uV
  delta     1-4 Hz SHARE above the lowest  uv_delta    1-4 Hz power of the loudest slice, uV^2
  deflect   excursion / own spread         uv_deflect  largest excursion, uV, averaged over channels
  nod       log(midline / temporal)        uv_nod      midline RMS minus temporal RMS, in uV
  sideways  excursion / own spread         uv_sideways largest |F4-T3| excursion, in uV
  turn      signed peak / own spread       UNCHANGED -- a sign test; scaling cannot change a sign

Same filters, same time windows, same channels as monitoring/rtamt_signals.py. The only difference
is that nothing is divided by a quantity from the recording.
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


def session_job(path):
    import sys as _s
    if str(ROOT) not in _s.path:
        _s.path.insert(0, str(ROOT))
    from monitoring.rtamt_signals import (_BAND_CLASS, _BAND_DETECT, _BAND_PROMINENCE, _IDX, FS,
                                          RtamtConfig, _block_sums, check_window)
    from scipy.signal import lfilter
    cfg = RtamtConfig()
    data = np.load(path, allow_pickle=True)
    epochs = data["epochs"][:, :8] * 1e6
    block = max(int(cfg.gate_seconds * FS), 1)
    width = max(int(cfg.delta_slice_seconds * FS), 8)
    taper = np.hanning(width)
    freq = np.fft.rfftfreq(width, 1.0 / FS)
    in_delta = (freq >= 1.0) & (freq < 4.0)
    rows = []
    for raw in epochs:
        x = check_window(raw)
        row = {}

        # uv_loud: the loudest 0.5 s, as an RMS in microvolts (no division by the quietest)
        det = lfilter(_BAND_DETECT[0], _BAND_DETECT[1], x, axis=-1)
        power = np.mean(det ** 2, axis=0)                     # pooled over channels, uV^2
        row["uv_loud"] = float(np.sqrt(_block_sums(power, block).max() / block))

        # uv_delta: the 1-4 Hz power of the loudest slice, in uV^2 (not a share)
        best = 0.0
        for s in range(0, x.shape[1] - width + 1, max(width // 2, 1)):
            seg = x[:, s:s + width]
            seg = seg - seg.mean(axis=1, keepdims=True)
            p = np.abs(np.fft.rfft(seg * taper, axis=1)) ** 2
            best = max(best, float(np.mean(p[:, in_delta].sum(axis=1))) / width)
        row["uv_delta"] = best

        # uv_deflect: the largest excursion in microvolts, averaged over the eight channels
        cls = lfilter(_BAND_CLASS[0], _BAND_CLASS[1], x, axis=-1)
        cls = cls - np.median(cls, axis=1, keepdims=True)
        row["uv_deflect"] = float(np.mean(np.abs(cls).max(axis=1)))

        # uv_nod: midline RMS minus temporal RMS, in microvolts (a difference, not a log ratio)
        mid = np.sqrt(np.mean(cls[[_IDX["Fz"], _IDX["Cz"]]] ** 2))
        tem = np.sqrt(np.mean(cls[[_IDX["T3"], _IDX["T4"]]] ** 2))
        row["uv_nod"] = float(mid - tem)

        # uv_sideways: the largest F4-T3 excursion, in microvolts
        pro = lfilter(_BAND_PROMINENCE[0], _BAND_PROMINENCE[1], x, axis=-1)
        v = pro[_IDX["F4"]] - pro[_IDX["T3"]]
        row["uv_sideways"] = float(np.abs(v - np.median(v)).max())
        rows.append(row)

    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
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
    np.savez(HERE / "uv.npz", **res)
    print(f"windows {len(res['session'])}")
    for k in ("uv_loud", "uv_delta", "uv_deflect", "uv_nod", "uv_sideways"):
        print(f"   {k:12s} median {np.median(res[k]):10.3f}")
