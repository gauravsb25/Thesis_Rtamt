"""
Features derived from the ICA TeLEX synthesis in clean_final_icaeeg_results/.

Imported by BOTH analyze_full_dataset.py and classify.py so the two can never
drift apart -- a rule fitted on a feature the classifier computes differently
would silently never fire.

WHY THESE FEATURES

The grouped synthesis on the ica_eeg corpus mined conjunctions that are
SIGNED and MIRROR-IMAGED between left and right turns:

    LEFT   F[0,3]( T3 > 25.2  &  F3 > +13.4  &  F4 < -15.3 )   neg_sat 0.00
    RIGHT  F[0,3]( T4 > 18.9  &  Fp2 > +16.3 &  Fp1 <  -5.7 )  neg_sat 0.00
    STRAIGHT G[0,3]( T3 < 29.8 & T4 < 31.9 )                   neg_sat 0.00

The existing feature table cannot express these. Its pair features are
`_netdiff` (max+min of a difference) and `_absmeandiff` (mean |difference|) --
both symmetric in sign, so they measure how much two channels diverge but not
WHICH ONE went positive. That is precisely the information separating LEFT
from RIGHT, and it is why those two classes were the pipeline's weak point.

Two families are added:

1. `<a>+<b>-_bipolar` -- the co-occurrence strength of "a goes high AND b goes
   low", i.e. min(max(a), -min(b)). Large only when a's positive excursion and
   b's negative excursion are BOTH large. This is the direct feature form of
   the mined conjunction `a > p & b < -q`.

2. `<a>&<b>_bothbelow` -- max(max|a|, max|b|), small only when both channels
   stay bounded for the whole window. Feature form of `G[0,3](a < p & b < q)`.

Plus scale-invariant shape descriptors of the temporal difference signal.
`T3T4_skew` measures the asymmetry of the T3-T4 distribution's tail -- which
direction the dominant excursion went. Measured on its own under grouped
5-fold CV it separates LEFT from RIGHT at 0.768 balanced accuracy, against
0.72 for the best pre-existing feature. Being normalised by its own standard
deviation, it is immune to the ~2.75x amplitude difference between subjects
that broke earlier absolute-microvolt rules.
"""

import numpy as np

# Ordered pairs the ICA synthesis actually mined as signed conjunctions.
# (a, b) means the mined shape was  a > +p  AND  b < -q.
BIPOLAR_PAIRS = [
    ('F3', 'F4'), ('F4', 'F3'),
    ('Fp2', 'Fp1'), ('Fp1', 'Fp2'),
    ('T3', 'T4'), ('T4', 'T3'),
    ('Cz', 'T3'), ('Cz', 'T4'),
    ('T4', 'Fz'), ('T3', 'Fz'),
]

# Unordered pairs mined as "both stay bounded" (the STRAIGHT shape).
BOTHBELOW_PAIRS = [('T3', 'T4'), ('F3', 'F4'), ('Fp1', 'Fp2'), ('Fz', 'Cz')]


def telex_features(df, available):
    """Return the TeLEX-derived feature dict for one 3-second window.

    `df` is the window, `available` the raw channels present in it.
    """
    row = {}
    v = {ch: df[ch].values.astype(float) for ch in available}

    # --- family 1: signed co-occurrence  (a > +p) & (b < -q) ---
    for a, b in BIPOLAR_PAIRS:
        if a in v and b in v:
            row[f'{a}+{b}-_bipolar'] = float(min(v[a].max(), -v[b].min()))

    # --- family 2: both channels bounded for the whole window ---
    for a, b in BOTHBELOW_PAIRS:
        if a in v and b in v:
            row[f'{a}&{b}_bothbelow'] = float(max(np.abs(v[a]).max(),
                                                  np.abs(v[b]).max()))

    # --- family 3: scale-invariant shape of a difference signal ---
    for a, b in [('T3', 'T4'), ('F3', 'F4'), ('Fp1', 'Fp2')]:
        if a not in v or b not in v:
            continue
        d = v[a] - v[b]
        d = d - d.mean()
        sd = d.std()
        name = f'{a}{b}'
        if sd > 1e-12:
            z = d / sd
            row[f'{name}_skew'] = float((z ** 3).mean())
            row[f'{name}_kurt'] = float((z ** 4).mean())
        else:
            row[f'{name}_skew'] = 0.0
            row[f'{name}_kurt'] = 0.0
        # sign of the largest excursion: which way the pair swung
        k = int(np.argmax(np.abs(d)))
        row[f'{name}_peak_polarity'] = float(np.sign(d[k]))
        row[f'{name}_peak_signed'] = float(d[k])

    return row
