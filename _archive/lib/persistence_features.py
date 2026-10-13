"""
Temporal-persistence features: min / max / sustain / recurrence.

WHY THIS FAMILY IS DIFFERENT

Every existing feature collapses the window to one number in a way that is
order-blind: a peak, a mean, a band power, a skew. Shuffle the samples and most
of them are unchanged. Even the dynamics family only asks WHERE energy sits, in
four coarse slices.

rtamt_layer/mine_monitors.py searches formulas of the form

    always[0,T] ( eventually[0,d] ( signal > theta ) )     recurrence
    eventually[0,T] ( always[0,d] ( signal > theta ) )     sustain

and its own notes record these as the stats `min`, `max`, `recur_d`, `sustain_d`.
Those ask something no scalar summary can: not how BIG the signal got, but for
how LONG it stayed there, and how REGULARLY it came back. A brief spike and a
sustained plateau of the same height are identical to `_peak` and opposite to
`sustain`.

The threshold-free scalar equivalents are computed here, so the classifier can
use them without an STL engine:

  min / max          of a rolling signal, i.e. the tightest always-bound and
                     the eventually-bound the monitor would find
  sustain            longest CONTINUOUS stretch above the signal's own median,
                     as a fraction of the window
  recur              longest GAP between excursions above the median. Small
                     means the signal keeps coming back; large means one burst
  duty               fraction of the window spent above the median

Each is taken against the signal's OWN median, so no absolute threshold is
involved and any constant gain cancels -- the same reasoning that makes the
differential and ratio features survive across subjects.

SIGNALS

Per-channel robust-|z| envelope (gain-free), and the signed z-difference of the
three lateral pairs (gain-free AND directional, which is what separates left
from right).
"""

import numpy as np
from lib.constants import FS, LATERAL_PAIRS as LATERAL

_EPS = 1e-12

WIN = int(0.5 * FS)      # rolling window the "always/eventually" bound sees
STEP = int(0.125 * FS)   # 8 Hz sampling of the rolling signal


def _robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    mad = mad if mad > 1e-9 else (x.std() + 1e-9)
    return (x - med) / mad


def _rolling(x, fn):
    """Sample fn over sliding sub-windows -- the signal a monitor watches."""
    n = len(x)
    if n < WIN:
        return np.array([fn(x)])
    return np.array([fn(x[s:s + WIN]) for s in range(0, n - WIN + 1, STEP)])


def _persistence(sig, name, row):
    """min / max / sustain / recur / duty of one rolling signal."""
    if len(sig) == 0:
        return
    row[f'{name}_tmin'] = float(np.min(sig))
    row[f'{name}_tmax'] = float(np.max(sig))
    med = float(np.median(sig))
    above = sig >= med
    n = len(above)

    # longest continuous stretch above -- the sustain bound
    best = cur = 0
    for a in above:
        cur = cur + 1 if a else 0
        best = max(best, cur)
    row[f'{name}_sustain'] = float(best / n)

    # longest gap between excursions -- the recurrence bound
    gap = best_gap = 0
    for a in above:
        gap = 0 if a else gap + 1
        best_gap = max(best_gap, gap)
    row[f'{name}_recur'] = float(best_gap / n)

    row[f'{name}_duty'] = float(above.mean())


def persistence_features(df, available):
    row = {}
    v = {c: df[c].values.astype(float) for c in available}
    if not v:
        return row

    z = {c: _robust_z(v[c]) for c in available}

    # per-channel activation envelope: how strongly and how persistently
    # this electrode departs from its own baseline
    for c in available:
        _persistence(_rolling(np.abs(z[c]), lambda w: float(np.mean(w))), f'{c}_az', row)

    # signed lateral difference: directional, so left and right differ in sign
    for a, b in LATERAL:
        if a in z and b in z:
            d = z[a] - z[b]
            _persistence(_rolling(d, lambda w: float(np.mean(w))), f'{a}{b}_zdiff', row)
            # how long the difference holds ONE sign -- a clean turn holds it,
            # noise flips
            s = np.sign(d)
            best = cur = 0
            for i in range(1, len(s)):
                cur = cur + 1 if s[i] == s[i - 1] else 0
                best = max(best, cur)
            row[f'{a}{b}_signhold'] = float(best / len(s))
    return row
