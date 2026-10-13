"""
Per-subject calibration: remove between-subject amplitude variation.

WHY

sub-11's EEG runs roughly 2.75x hotter than sub-01's on every channel. Any
rule with an absolute microvolt threshold tuned on one subject misfires on the
other. The pipeline works around this today by preferring scale-invariant
features (ratios, `_frac`, `_normrms`, skew, log-asymmetry), which is why
normalising the CURRENT two-subject corpus barely moves the numbers.

That workaround does not scale. It restricts the rule search to the subset of
features that happen to be scale-free, and it will not survive a corpus with
ten subjects at ten different gains. Normalising per subject removes the
variation at source, so absolute-amplitude features become usable too.

HOW -- and why it is not cheating

Each feature is converted to a robust z-score against that SUBJECT's own
distribution:

    z = (x - median_subject) / MAD_subject

using median/MAD rather than mean/std so a single artifact trial cannot define
the baseline.

The statistics are fitted on TRAINING data only and then applied to held-out
data (`fit()` then `apply()`). Inside cross-validation that means a fold's
statistics come from the other folds' sessions, never its own -- so the CV
number stays honest.

This mirrors how it works in deployment: a new subject records a short
calibration block, that block gives their median/MAD, and every later session
is normalised against it. A subject with no calibration data falls back to the
pooled statistics across all known subjects, which is the best available guess.
"""

import os

import numpy as np
import pandas as pd

_EPS = 1e-9


def _robust_stats(frame):
    med = frame.median()
    mad = (frame - med).abs().median()
    # a constant feature has MAD 0; fall back to std, then to 1
    std = frame.std()
    mad = mad.where(mad > _EPS, std)
    mad = mad.where(mad > _EPS, 1.0)
    return med, mad


def fit(X, subjects):
    """Per-subject median/MAD, plus a pooled fallback for unseen subjects.

    X        DataFrame of numeric features (training rows only)
    subjects array-like of subject ids, same length as X
    """
    subjects = np.asarray(subjects)
    stats = {}
    for s in pd.unique(subjects):
        med, mad = _robust_stats(X[subjects == s])
        stats[s] = (med, mad)
    stats['__pooled__'] = _robust_stats(X)
    return stats


# "center" subtracts the subject median then divides by its MAD.
# "scale"  divides by the MAD only.
#
# Centering is class-composition dependent: a subject missing a class has its
# median sitting somewhere else in class space, so subtracting it shifts that
# subject's classes relative to everyone else's. sub-11 contributes no STRAIGHT
# trials, which is exactly this situation. Scaling alone removes the gain
# difference -- the actual between-subject problem -- without that side effect.
MODE = os.environ.get("SUBJECT_NORM_STAT", "scale")


def apply(X, subjects, stats):
    """Normalise each row against its own subject's statistics."""
    subjects = np.asarray(subjects)
    out = X.copy()
    for s in pd.unique(subjects):
        m = subjects == s
        med, mad = stats.get(s, stats['__pooled__'])
        if MODE == "center":
            out.loc[m] = ((X[m] - med) / mad).values
        else:
            out.loc[m] = (X[m] / mad).values
    return out


def fit_apply(X, subjects):
    return apply(X, subjects, fit(X, subjects))


def save(stats, path):
    """Flatten to a CSV classify.py can load at inference time."""
    rows = []
    for subj, (med, mad) in stats.items():
        for feat in med.index:
            rows.append({'subject': subj, 'feature': feat,
                         'median': med[feat], 'mad': mad[feat]})
    pd.DataFrame(rows).to_csv(path, index=False)


def load(path):
    df = pd.read_csv(path)
    stats = {}
    for subj, g in df.groupby('subject'):
        stats[subj] = (g.set_index('feature')['median'],
                       g.set_index('feature')['mad'])
    return stats


def apply_row(row, subject, stats):
    """Normalise a single feature dict (classify.py's per-file path)."""
    med, mad = stats.get(subject, stats.get('__pooled__'))
    out = {}
    for k, v in row.items():
        if v is None or k not in med.index:
            out[k] = v
            continue
        try:
            d = float(mad[k])
            out[k] = (float(v) - float(med[k])) / (d if abs(d) > _EPS else 1.0)
        except (TypeError, ValueError):
            out[k] = v
    return out
