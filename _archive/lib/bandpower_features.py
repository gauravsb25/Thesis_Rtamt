"""
Spectral band-power features for the 3-second windows.

Everything the pipeline used before this was a TIME-domain amplitude statistic
(peak, RMS, mean |difference|, onset latency). Band power asks a different
question -- how the window's energy is distributed across frequency -- so it
can separate two trials that reach the same amplitude by different means.

Three families:

1. `<ch>_bp_<band>`     absolute power in that band
2. `<ch>_rel_<band>`    that band's share of the channel's total power.
                        Scale-invariant: dividing by the channel's own total
                        cancels the ~2.75x amplitude difference between
                        subjects that broke earlier microvolt thresholds.
3. `<a>|<b>_asym_<band>` log(power_a / power_b) for the three lateral pairs.

Family 3 is the reason this module exists. It is the standard EEG
lateralisation index, and unlike the pipeline's existing `_netdiff` /
`_absmeandiff` pair features it is SIGNED: positive means the left electrode
carried more power in that band, negative means the right one did. A head turn
is a directional movement, so a directional feature is the natural descriptor
-- the same reasoning that made the signed TeLEX conjunctions work.

Being a log ratio it is also immune to any per-subject gain: a constant
amplitude scaling multiplies both sides and cancels.
"""

import numpy as np
from scipy import signal
from lib.constants import BANDS, FS, LATERAL_PAIRS

# Left/right electrode pairs. Order matters: log(a/b) is positive when the
# LEFT-hemisphere electrode carries more power.

_EPS = 1e-12


def _psd(x):
    """Welch PSD with 1-second segments -> ~1 Hz resolution over a 3 s window."""
    nperseg = min(int(FS), len(x))
    f, p = signal.welch(x, fs=FS, nperseg=nperseg, noverlap=nperseg // 2)
    return f, p


def bandpower_features(df, available):
    row = {}
    powers = {}

    for ch in available:
        x = df[ch].values.astype(float)
        x = x - x.mean()
        f, p = _psd(x)
        total = float(np.trapezoid(p, f)) if hasattr(np, 'trapezoid') else float(np.trapz(p, f))
        powers[ch] = {}
        for name, (lo, hi) in BANDS.items():
            m = (f >= lo) & (f < hi)
            bp = float(np.trapezoid(p[m], f[m])) if hasattr(np, 'trapezoid') else float(np.trapz(p[m], f[m]))
            powers[ch][name] = bp
            row[f'{ch}_bp_{name}'] = bp
            row[f'{ch}_rel_{name}'] = bp / (total + _EPS)
        row[f'{ch}_bp_total'] = total
        # spectral centroid: where the mass sits, in Hz
        row[f'{ch}_spec_centroid'] = float((f * p).sum() / (p.sum() + _EPS))

    # signed lateralisation index per band
    for a, b in LATERAL_PAIRS:
        if a in powers and b in powers:
            for name in BANDS:
                pa, pb = powers[a][name], powers[b][name]
                row[f'{a}|{b}_asym_{name}'] = float(np.log((pa + _EPS) / (pb + _EPS)))
            ta = sum(powers[a].values())
            tb = sum(powers[b].values())
            row[f'{a}|{b}_asym_total'] = float(np.log((ta + _EPS) / (tb + _EPS)))

    return row
