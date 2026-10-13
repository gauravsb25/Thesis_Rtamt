"""
stl_features.py -- temporal-logic statistics, as ordinary scalar features.

WHY THIS FILE EXISTS
--------------------
pipeline/classify.py's rule table holds entries of the form

    (feature_name, threshold, direction, weight)

which is a threshold test on one number per window. An STL formula looks like
it could never fit that shape, because it talks about time. It can, and the
reason is simple:

    eventually[0,3](s > k)   passes exactly when   max(s) > k
    always[0,3](s > k)       passes exactly when   min(s) > k

So a temporal formula IS a threshold test -- on a number that summarises the
signal over the window. Compute that number as a feature and the formula drops
straight into the existing rule table with no change to the voter.

The nested forms work the same way. `sustain0.5` is the largest value the
signal held CONTINUOUSLY for half a second, so

    eventually[0,o](always[0,0.5](s > k))   passes exactly when   sustain0.5 > k

WHAT THESE ADD OVER THE EXISTING FEATURES
-----------------------------------------
lib/features.py already summarises a window many ways, but every one of those
summaries collapses the whole 3 seconds into a single statistic -- a peak, a
mean, a band power. None of them can distinguish a brief spike from a steady
plateau at the same height.

The `sustain` and `recur` statistics can, because they ask about DURATION:

    sustain<d>   the highest level the signal held for a solid d seconds
    recur<d>     the lowest level it fell to, considering only drops that
                 lasted at least d seconds

Measured on this corpus: adding 18 such rules to classify.py's 18 raised
accuracy from 0.821 to 0.867 (balanced 0.862 -> 0.897), and RIGHT specifically
from 0.707 to 0.800.

NOT THE SAME AS lib/persistence_features.py
-------------------------------------------
That module also has `sustain` and `recur`, but they mean something different
and are measured over a different set of signals:

  persistence_features   sustain = the longest FRACTION of the window the
                         signal stayed above a fixed level. A duration.
                         Computed on the raw channels and lateral pairs.

  this module            sustain0.5 = the highest LEVEL the signal held for a
                         solid half second. A value, not a duration.
                         Computed over signal_library's 185 signals, which
                         includes the common-average, Hilbert-envelope,
                         phase and running-correlation families that
                         persistence_features does not have.

The two are complementary. Keeping both is fine; they are not duplicates.

NAMING
------
Features are named `stl_<signal>_<statistic>`, with the decimal point in a
duration removed so the name is a legal identifier:

    T4_car        max         ->  stl_T4_car_max
    T3T4_corr     sustain0.5  ->  stl_T3T4_corr_sustain05

COST
----
The signal library builds all 185 signals for a window, including eight
short-time Fourier transforms, and that is the expensive part. Only the
signals actually named in WANTED are read out, but they are all computed. On
this corpus the extra cost is roughly a quarter of a second per window.
"""

import numpy as np

from lib.paths import bootstrap; bootstrap()
from rtamt_layer.signal_library import build_signals, nested_stats

# The (signal, statistic) pairs pipeline/classify.py's rule table refers to.
# Extend this list when new STL rules are added to SELECTED_RULES; anything
# not listed here is simply not computed, so the cost stays proportional to
# what is actually used.
WANTED = [
    ('F3_car', 'min'),
    ('F3_rel_delta', 'sustain0.5'),
    ('F3_spec_centroid', 'min'),
    ('F4_car', 'min'),
    ('F4_spec_centroid', 'min'),
    ('Fp2_spec_centroid', 'recur0.25'),
    ('T3T4_corr', 'recur0.25'),
    ('T3T4_corr', 'sustain1'),
    ('T3T4_cosdphi', 'recur1'),
    ('T3T4_zcr', 'sustain0.5'),
    ('T3T4_zdiff', 'recur0.25'),
    ('T4', 'sustain1'),
    ('T4_car', 'max'),
    ('T4_car', 'min'),
    ('T4_car', 'sustain1'),
    ('T4_henv', 'recur0.25'),
    ('T4_henv', 'recur0.5'),
    ('T4_rel_delta', 'sustain1'),
]


def feature_name(signal, stat):
    """('T3T4_corr', 'sustain0.5') -> 'stl_T3T4_corr_sustain05'."""
    return 'stl_%s_%s' % (signal, stat.replace('.', ''))


def stl_features(df, wanted=None):
    """{feature_name: value} for one window.

    Every value is the exact number the corresponding STL formula compares
    against its threshold, so a rule written here means the same thing as the
    formula rtamt would evaluate.
    """
    wanted = WANTED if wanted is None else wanted
    signals = build_signals(df)

    # group by signal so each one's statistics are computed in a single pass
    by_signal = {}
    for sig, stat in wanted:
        by_signal.setdefault(sig, set()).add(stat)

    row = {}
    for sig, stats in by_signal.items():
        if sig not in signals:
            # a signal that could not be built (missing channel) yields NaN;
            # classify.py's rule_fires() already treats NaN as "did not fire"
            for stat in stats:
                row[feature_name(sig, stat)] = np.nan
            continue
        t, v = signals[sig]
        nested = None
        for stat in stats:
            if stat == 'max':
                row[feature_name(sig, stat)] = float(np.max(v))
            elif stat == 'min':
                row[feature_name(sig, stat)] = float(np.min(v))
            else:
                if nested is None:
                    nested = nested_stats(t, v)
                val = nested.get(stat)
                row[feature_name(sig, stat)] = (float(val) if val is not None
                                                else np.nan)
    return row
