"""
Temporal-dynamics features: what the movement does OVER the window.

Every other family collapses the 3 seconds into a single statistic -- a peak, a
mean, a band power. That throws away the movement's time course, and a head
turn is a time course: a deflection that rises, peaks and returns, with a
characteristic asymmetry and a characteristic order between electrodes.

Five families, all class-agnostic (nothing here is fitted to a label, so there
is no leakage), and all built to be comparable across subjects.

1. SUB-WINDOW TRAJECTORY
   Split the window into 6 slices of 0.5 s and take each bank's energy share.
   Gives the shape of the event in time rather than its size: where the energy
   sits, how concentrated it is, and whether it is front- or back-loaded.

2. DEFLECTION GEOMETRY
   On the T3-T4, F3-F4 and Fp1-Fp2 difference signals: how long the excursion
   lasts, how fast it rises versus falls, and where its peak sits. A turn and a
   return have different rise/fall asymmetry.

3. CROSS-CHANNEL LAG
   Peak cross-correlation and the lag at which it occurs, for the three lateral
   pairs. Which side leads, and by how much -- signed, so it separates left from
   right by construction.

4. SIGN STRUCTURE
   Zero crossings and the longest single-sign run. A clean monophasic
   deflection differs from oscillatory noise even at the same amplitude.

5. ENERGY CONCENTRATION
   What fraction of the window's total energy falls in its most active 0.5 s.
   High for a discrete movement, low for diffuse noise.
"""

import numpy as np
from lib.constants import FS

_EPS = 1e-12

DIFF_PAIRS = [('T3', 'T4'), ('F3', 'F4'), ('Fp1', 'Fp2')]
BANKS = {
    'front': ['Fp1', 'Fp2', 'F3', 'Fz', 'F4'],
    'temp':  ['T3', 'T4'],
    'cent':  ['Cz', 'Fz'],
}
N_SLICES = int(__import__("os").environ.get("N_SLICES", 4))


def _slices(x, n=N_SLICES):
    """Split a 1-D signal into n contiguous slices of equal length."""
    edges = np.linspace(0, len(x), n + 1).astype(int)
    return [x[a:b] for a, b in zip(edges[:-1], edges[1:])]


def dynamics_features(df, available):
    row = {}
    v = {c: df[c].values.astype(float) - df[c].values.astype(float).mean()
         for c in available}
    if not v:
        return row
    n = len(next(iter(v.values())))

    # ---- 1. sub-window energy trajectory, per bank -----------------------
    for bank, chans in BANKS.items():
        present = [c for c in chans if c in v]
        if not present:
            continue
        e = np.array([np.mean([np.mean(s ** 2) for s in
                               [_slices(v[c])[k] for c in present]])
                      for k in range(N_SLICES)])
        tot = e.sum() + _EPS
        share = e / tot
        for k in range(N_SLICES):
            row[f'{bank}_slice{k}_share'] = float(share[k])
        # where in the window the bank's energy is centred, 0..1
        row[f'{bank}_energy_centroid'] = float((share * np.arange(N_SLICES)).sum()
                                               / (N_SLICES - 1))
        # how concentrated: 1 = all in one slice, 0 = perfectly flat
        row[f'{bank}_energy_focus'] = float(share.max() - share.min())
        # front-loaded (negative) vs back-loaded (positive)
        half = N_SLICES // 2
        row[f'{bank}_energy_tilt'] = float(share[half:].sum() - share[:half].sum())

    # ---- 2. deflection geometry on the difference signals ----------------
    for a, b in DIFF_PAIRS:
        if a not in v or b not in v:
            continue
        d = v[a] - v[b]
        name = f'{a}{b}'
        k = int(np.argmax(np.abs(d)))
        amp = abs(d[k]) + _EPS
        row[f'{name}_peak_pos'] = float(k / max(n - 1, 1))          # 0..1 in window
        # width at half maximum, in seconds
        above = np.abs(d) >= amp / 2
        row[f'{name}_halfwidth'] = float(above.sum() / FS)
        # rise vs fall: samples from window start to peak, and peak to end,
        # restricted to the contiguous above-half region around the peak
        lo = k
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = k
        while hi < n - 1 and above[hi + 1]:
            hi += 1
        rise, fall = (k - lo) / FS, (hi - k) / FS
        row[f'{name}_rise'] = float(rise)
        row[f'{name}_fall'] = float(fall)
        # >0 means it rose faster than it fell
        row[f'{name}_rise_fall_asym'] = float((fall - rise) / (fall + rise + _EPS))
        # steepest slope, signed: which way the fastest change went
        g = np.gradient(d, 1.0 / FS)
        row[f'{name}_max_slope_signed'] = float(g[int(np.argmax(np.abs(g)))] / amp)

        # ---- 4. sign structure ------------------------------------------
        s = np.sign(d)
        row[f'{name}_zero_crossings'] = float((np.diff(s) != 0).sum())
        runs, cur = [], 1
        for i in range(1, n):
            if s[i] == s[i - 1]:
                cur += 1
            else:
                runs.append(cur); cur = 1
        runs.append(cur)
        row[f'{name}_longest_run'] = float(max(runs) / n)

    # ---- 3. cross-channel lag -------------------------------------------
    max_lag = int(0.4 * FS)
    for a, b in DIFF_PAIRS:
        if a not in v or b not in v:
            continue
        x = v[a] / (v[a].std() + _EPS)
        y = v[b] / (v[b].std() + _EPS)
        cc = np.correlate(x, y, 'full') / n
        lags = np.arange(-n + 1, n)
        sel = (lags >= -max_lag) & (lags <= max_lag)
        cs, ls = cc[sel], lags[sel]
        j = int(np.argmax(np.abs(cs)))
        row[f'{a}|{b}_xcorr_peak'] = float(cs[j])
        row[f'{a}|{b}_xcorr_lag'] = float(ls[j] / FS)

    # ---- 5. energy concentration ----------------------------------------
    allsq = np.mean([x ** 2 for x in v.values()], axis=0)
    w = int(0.5 * FS)
    if n >= w:
        roll = np.convolve(allsq, np.ones(w) / w, mode='valid')
        row['energy_in_best_halfsec'] = float(roll.max() * w / (allsq.sum() + _EPS))
        row['energy_peak_pos'] = float(np.argmax(roll) / max(len(roll) - 1, 1))
    return row
