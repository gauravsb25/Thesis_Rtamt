"""
constants.py -- the definitions every stage of the pipeline shares.

WHY THIS FILE EXISTS
--------------------
Before it, the same handful of facts were written out separately in many
places: CLASSES in 8 files, FS in 7, RAW_CHANNELS in 4, the lateral electrode
pairs in 4 (under two different names, LATERAL and LATERAL_PAIRS).

Every copy agreed when this file was written -- that was checked, not assumed.
But copies only stay in agreement by luck. Add a channel, change the sampling
rate, or reorder the classes, and whichever copies you miss keep the old value
and fail quietly rather than loudly.

Two real examples that had already drifted:

  * pipeline/classify.py's label parser lowercased the WHOLE PATH, while the
    rtamt_layer copies lowercased only the filename. A directory named
    "left_hand" would have labelled every file under it LEFT. Harmless with
    the current folder names, which is exactly what makes it dangerous.

  * tools/evaluate_rule_voters.py's balanced accuracy had no guard for a class
    with no windows in it, so a fold missing STRAIGHT would return nan while
    the other two copies returned a number.

Both are fixed here, once.

IMPORTING
---------
    from lib.constants import CLASSES, FS, RAW_CHANNELS, LATERAL_PAIRS
    from lib.constants import label_of, balanced_accuracy
"""

import os

import numpy as np

# ---------------------------------------------------------------------------
# The recording
# ---------------------------------------------------------------------------

FS = 256.0                 # sampling rate, Hz
WINDOW_SEC = 3.0           # length of one classified window
WINDOW_SAMPLES = 768       # samples in that window at FS

# Order matters: it fixes the column order of every confusion matrix and score
# array in the project. Do not reorder without checking every consumer.
CLASSES = ['STRAIGHT', 'LEFT', 'RIGHT', 'DOWN']

RAW_CHANNELS = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'T3', 'Cz', 'T4']

# Left/right electrode pairs. Order within each pair matters: the left-hemisphere
# electrode comes first, so log(a/b) and (a - b) are positive when the LEFT side
# carries more. Several features depend on that sign convention.
LATERAL_PAIRS = [('T3', 'T4'), ('F3', 'F4'), ('Fp1', 'Fp2')]

# EEG frequency bands, Hz. Half-open [lo, hi) so adjacent bands do not overlap.
BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 45.0),
}

EPS = 1e-12                # guards divisions and logarithms


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def label_of(path):
    """True movement from a filename: 'sub-01_..._task-left_run-01' -> 'LEFT'.

    Only the FILENAME is examined, never the directory part. The older copy of
    this in pipeline/classify.py searched the whole path, which would pick up
    a movement word appearing in a folder name.

    Returns None when no movement word is present, so callers can skip
    unlabelled files rather than guess.
    """
    name = os.path.basename(str(path)).lower()
    for kw in ('left', 'right', 'down', 'straight'):
        if kw in name:
            return kw.upper()
    return None


def balanced_accuracy(y_true, y_pred, classes=None):
    """Mean per-class recall.

    Plain accuracy is misleading here: STRAIGHT has 24 windows and RIGHT has
    150, so a classifier can ignore STRAIGHT completely and still look good.
    Averaging each movement's own hit rate gives every movement equal say.

    A class with no windows present is skipped rather than counted as zero,
    which matters inside cross-validation folds where a small class can be
    absent from a split.
    """
    classes = CLASSES if classes is None else classes
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = [(y_pred[y_true == c] == c).mean()
               for c in classes if (y_true == c).any()]
    return float(np.mean(recalls)) if recalls else 0.0


def confusion(y_true, y_pred, classes=None):
    """Counts as a plain nested list: rows are true, columns are predicted."""
    classes = CLASSES if classes is None else classes
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return [[int(((y_true == t) & (y_pred == p)).sum()) for p in classes]
            for t in classes]
