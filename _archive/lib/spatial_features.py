"""
Spatial re-referencing, Hjorth descriptors, phase and complexity features.

Four standard EEG families that were entirely absent from the feature set.

1. SPATIAL RE-REFERENCING  -- the important one for left vs right.
   Every channel is recorded against a common reference electrode, so the
   reference's own activity appears in ALL of them. A head movement that
   shifts the reference therefore moves every channel together, burying the
   lateral difference that separates a left turn from a right one.

   Common Average Reference (CAR) subtracts the across-channel mean at each
   instant, removing whatever is shared. A Laplacian goes further and
   subtracts each channel's neighbours, leaving only locally generated
   activity. Both are computed here and the resulting signals are described
   with the same signed statistics that worked on the raw data.

2. HJORTH PARAMETERS -- activity, mobility, complexity. Classic EEG
   descriptors from 1970. Mobility is the dominant frequency expressed in the
   time domain; complexity is how far the waveform departs from a pure sine.
   Cheap, and they capture waveform character that neither amplitude nor band
   power describes.

3. HILBERT PHASE -- instantaneous phase difference between the lateral pairs,
   and phase-locking value. Signed by construction: if T3 leads T4 the phase
   difference is positive, and it reverses for the opposite movement. Phase is
   independent of amplitude, so it survives any gain difference.

4. COMPLEXITY -- Katz and Higuchi fractal dimension. A smooth movement
   deflection has a low fractal dimension; noisy or oscillatory activity has a
   high one. Amplitude-normalised, so subject gain cancels.
"""

import numpy as np
from scipy import signal as _sig
from lib.constants import FS, LATERAL_PAIRS as LATERAL

_EPS = 1e-12

# rough neighbour sets for a Laplacian on this 8-electrode montage
NEIGHBOURS = {
    'Fp1': ['Fp2', 'F3'], 'Fp2': ['Fp1', 'F4'],
    'F3': ['Fp1', 'Fz', 'T3'], 'F4': ['Fp2', 'Fz', 'T4'],
    'Fz': ['F3', 'F4', 'Cz'], 'Cz': ['Fz', 'T3', 'T4'],
    'T3': ['F3', 'Cz'], 'T4': ['F4', 'Cz'],
}


def _hjorth(x):
    d1 = np.diff(x)
    d2 = np.diff(d1)
    v0 = np.var(x) + _EPS
    v1 = np.var(d1) + _EPS
    v2 = np.var(d2) + _EPS
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / (mob + _EPS)
    return float(v0), float(mob), float(comp)


def _katz_fd(x):
    d = np.abs(np.diff(x))
    L = d.sum() + _EPS
    n = len(x) - 1
    dmax = np.max(np.abs(x - x[0])) + _EPS
    return float(np.log10(n) / (np.log10(n) + np.log10(dmax / (L / n) + _EPS) + _EPS))


def _higuchi_fd(x, kmax=8):
    N = len(x)
    lk = []
    for k in range(1, kmax + 1):
        Lm = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            s = np.abs(np.diff(x[idx])).sum()
            Lm.append(s * (N - 1) / (len(idx) - 1) / k)
        if Lm:
            lk.append((np.log(1.0 / k), np.log(np.mean(Lm) + _EPS)))
    if len(lk) < 2:
        return 0.0
    a = np.array(lk)
    return float(np.polyfit(a[:, 0], a[:, 1], 1)[0])


def _shape_stats(d, name, row):
    """The signed descriptors that worked on the raw signals, applied to a
    re-referenced or derived signal."""
    d = d - d.mean()
    sd = d.std()
    k = int(np.argmax(np.abs(d)))
    row[f'{name}_peak_signed'] = float(d[k])
    row[f'{name}_peak_polarity'] = float(np.sign(d[k]))
    if sd > _EPS:
        z = d / sd
        row[f'{name}_skew'] = float((z ** 3).mean())
        row[f'{name}_kurt'] = float((z ** 4).mean())
    else:
        row[f'{name}_skew'] = 0.0
        row[f'{name}_kurt'] = 0.0
    row[f'{name}_ptp'] = float(d.max() - d.min())


def spatial_features(df, available):
    row = {}
    v = {c: df[c].values.astype(float) for c in available}
    if len(v) < 2:
        return row
    M = np.vstack([v[c] for c in available])

    # ---- 1. common average reference ------------------------------------
    car = M - M.mean(axis=0, keepdims=True)
    car_by = {c: car[i] for i, c in enumerate(available)}
    for c in available:
        x = car_by[c]
        row[f'{c}_car_rms'] = float(np.sqrt(np.mean(x ** 2)))
        row[f'{c}_car_peak'] = float(np.max(np.abs(x)))
    for a, b in LATERAL:
        if a in car_by and b in car_by:
            _shape_stats(car_by[a] - car_by[b], f'{a}{b}_car', row)
            ea = np.mean(car_by[a] ** 2) + _EPS
            eb = np.mean(car_by[b] ** 2) + _EPS
            row[f'{a}|{b}_car_enratio'] = float(np.log(ea / eb))

    # ---- 2. laplacian ----------------------------------------------------
    lap_by = {}
    for c in available:
        nb = [n for n in NEIGHBOURS.get(c, []) if n in v]
        if not nb:
            continue
        lap_by[c] = v[c] - np.mean([v[n] for n in nb], axis=0)
        row[f'{c}_lap_rms'] = float(np.sqrt(np.mean(lap_by[c] ** 2)))
    for a, b in LATERAL:
        if a in lap_by and b in lap_by:
            _shape_stats(lap_by[a] - lap_by[b], f'{a}{b}_lap', row)
            ea = np.mean(lap_by[a] ** 2) + _EPS
            eb = np.mean(lap_by[b] ** 2) + _EPS
            row[f'{a}|{b}_lap_enratio'] = float(np.log(ea / eb))

    # ---- 3. hjorth + complexity -----------------------------------------
    for c in available:
        act, mob, comp = _hjorth(v[c])
        row[f'{c}_hjorth_mobility'] = mob
        row[f'{c}_hjorth_complexity'] = comp
        row[f'{c}_katz_fd'] = _katz_fd(v[c])
        row[f'{c}_higuchi_fd'] = _higuchi_fd(v[c])

    # ---- 4. hilbert phase ------------------------------------------------
    for a, b in LATERAL:
        if a not in v or b not in v:
            continue
        # movement band, so the phase is the movement's phase
        try:
            sos = _sig.butter(4, [0.5 / (FS / 2), 8.0 / (FS / 2)], btype='band', output='sos')
            xa = _sig.sosfiltfilt(sos, v[a])
            xb = _sig.sosfiltfilt(sos, v[b])
        except Exception:
            xa, xb = v[a], v[b]
        ha, hb = _sig.hilbert(xa), _sig.hilbert(xb)
        dphi = np.angle(ha) - np.angle(hb)
        row[f'{a}|{b}_plv'] = float(np.abs(np.mean(np.exp(1j * dphi))))
        row[f'{a}|{b}_phase_diff'] = float(np.angle(np.mean(np.exp(1j * dphi))))
        ea, eb = np.abs(ha), np.abs(hb)
        row[f'{a}|{b}_env_ratio'] = float(np.log((ea.mean() + _EPS) / (eb.mean() + _EPS)))
        row[f'{a}|{b}_env_lag'] = float(
            (np.argmax(ea) - np.argmax(eb)) / FS)

    # ---- 5. derivative domain -------------------------------------------
    for a, b in LATERAL:
        if a in v and b in v:
            d = v[a] - v[b]
            _shape_stats(np.gradient(d, 1.0 / FS), f'{a}{b}_vel', row)
    return row
