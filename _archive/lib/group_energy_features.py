"""
Electrode-GROUP energy features, mirroring slidingwindow.py's voting methods.

slidingwindow.py already decides WHERE the movement is by scoring electrode
groups: total energy across all electrodes (U1), how many electrodes are
simultaneously active (U2), largest peak-to-trough swing (U3), and per-class
tiers -- T3 for LEFT, T4 for RIGHT, Fz / Fz+Cz / Fz+F3+F4 for DOWN
(CLASS_ELECTRODE_TIERS). Those groupings are the extractor's own hypothesis
about which electrodes respond to which movement.

The classifier never saw them. Its features were per-channel or per-PAIR; no
feature asked "how much of this window's energy sits in the frontal bank
versus the temporal bank". This module supplies that.

Three families:

1. `<grp>_energy`      mean square across the group's channels
2. `<grp>_relenergy`   that group's share of total energy across all channels.
                       Scale-invariant -- the per-subject gain cancels.
3. `<a>|<b>_enratio`   log(energy_a / energy_b) between two groups.

Family 3 is the one aimed at LEFT vs RIGHT. `lefttier|righttier_enratio`
compares the exact electrode tiers slidingwindow.py uses to find a left turn
against those it uses for a right turn, as a signed log ratio: positive means
the left-turn tier carried more energy. Like the other signed features that
worked (T3T4_skew, the TeLEX conjunctions), it encodes DIRECTION rather than
magnitude, which is what the LEFT/RIGHT confusion needs.

Plus two scalar summaries of slidingwindow.py's unsupervised methods:
`n_active_channels` (U2) and `max_swing` (U3).
"""

import numpy as np

_EPS = 1e-12
ACTIVE_Z = 2.0

# Anatomical banks, matching features.py's ONSET_GROUPS.
ANATOMICAL = {
    'frontopolar': ['Fp1', 'Fp2'],
    'frontal':     ['F3', 'Fz', 'F4'],
    'temporal':    ['T3', 'T4'],
    'central':     ['Cz'],
    'midline':     ['Fz', 'Cz'],
}

# The per-class electrode tiers slidingwindow.py scores when hunting for the
# movement window (CLASS_ELECTRODE_TIERS).
CLASS_TIERS = {
    'lefttier':  ['T3', 'F3', 'Cz'],
    'righttier': ['T4', 'F4', 'Cz'],
    'downtier':  ['Fz', 'F3', 'F4'],
    'lefthemi':  ['T3', 'F3', 'Fp1'],
    'righthemi': ['T4', 'F4', 'Fp2'],
}

# Group pairs worth a signed log ratio.
RATIO_PAIRS = [
    ('lefttier', 'righttier'),
    ('lefthemi', 'righthemi'),
    ('temporal', 'frontal'),
    ('frontal', 'central'),
    ('temporal', 'central'),
    ('downtier', 'temporal'),
    ('frontopolar', 'temporal'),
]

ALL_GROUPS = {**ANATOMICAL, **CLASS_TIERS}


def group_energy_features(df, available):
    row = {}
    v = {ch: df[ch].values.astype(float) for ch in available}
    if not v:
        return row

    # per-channel mean square, mean-centred so a DC offset is not "energy"
    ch_energy = {}
    for ch, x in v.items():
        c = x - x.mean()
        ch_energy[ch] = float(np.mean(c ** 2))

    total = sum(ch_energy.values())
    row['all_energy'] = total

    energies = {}
    for name, chans in ALL_GROUPS.items():
        present = [c for c in chans if c in ch_energy]
        if not present:
            continue
        e = float(np.mean([ch_energy[c] for c in present]))
        energies[name] = e
        row[f'{name}_energy'] = e
        row[f'{name}_relenergy'] = e / (total + _EPS)

    for a, b in RATIO_PAIRS:
        if a in energies and b in energies:
            row[f'{a}|{b}_enratio'] = float(np.log((energies[a] + _EPS) /
                                                   (energies[b] + _EPS)))

    # U2: how many electrodes are simultaneously "active" at their own peak
    # moment, using a robust per-channel z-score so subject gain does not
    # decide the count.
    zs = []
    for ch, x in v.items():
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        mad = mad if mad > 1e-9 else (x.std() + 1e-9)
        zs.append(np.abs((x - med) / mad))
    if zs:
        Z = np.vstack(zs)
        counts = (Z >= ACTIVE_Z).sum(axis=0)
        row['n_active_channels'] = float(counts.max())
        row['mean_active_channels'] = float(counts.mean())

    # U3: largest peak-to-trough swing on any single channel
    row['max_swing'] = float(max(x.max() - x.min() for x in v.values()))

    return row
