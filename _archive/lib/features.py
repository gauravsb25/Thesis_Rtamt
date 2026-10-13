"""
THE single feature extractor for the whole pipeline.

analyze_full_dataset.py (which selects rules) and classify.py (which applies
them) both import compute_features() from here, so the two can never disagree
about what a feature means.

They used to hold separate copies. They drifted: analyze_full_dataset.py
computed 24 features -- <ch>_ptp, <ch>_argmaxpos, <ch>_laterat for all eight
channels -- that classify.py did not. build_rules.py was free to select those
features, and any rule using one was then DEAD in the classifier: the feature
simply was not there, so the rule never fired. That silently cost ~9 points of
accuracy (build_rules.py reported 79.0% in-sample while classify.py, running
the same rule table, produced 70.1%).

Add new features here and both stages see them at once.
"""

import itertools

import os

import numpy as np

from lib.telex_features import telex_features
from lib.bandpower_features import bandpower_features

# ablation switch: BANDPOWER=0 drops the 114 spectral features
# Ablation switches, for measuring what each feature family is worth.
# ABLATE=bandpower,onset,... drops those families from the feature table.
ABLATE = {x.strip() for x in os.environ.get("ABLATE", "").split(",") if x.strip()}
USE_BANDPOWER = "bandpower" not in ABLATE and os.environ.get("BANDPOWER", "1") not in ("0", "", "false", "False")
from lib.group_energy_features import group_energy_features
from lib.dynamics_features import dynamics_features
from lib.spatial_features import spatial_features
from lib.persistence_features import persistence_features
from lib.constants import RAW_CHANNELS

ONSET_Z = 2.0

ONSET_GROUPS = {
    'frontopolar': ['Fp1', 'Fp2'],
    'frontal':     ['F3', 'Fz', 'F4'],
    'temporal':    ['T3', 'T4'],
    'central':     ['Cz'],
    'midline':     ['Fz', 'Cz'],
}
ONSET_GROUP_PAIRS = [
    ('temporal', 'central'),
    ('temporal', 'midline'),
    ('frontal', 'temporal'),
    ('frontopolar', 'temporal'),
    ('frontopolar', 'central'),
]

def channel_onsets(df, available, k=ONSET_Z):
    """First time each channel's robust z-score exceeds k -- i.e. when that
    electrode 'goes high'. Robust (median/MAD) rather than mean/std so a
    single large artifact doesn't define its own baseline.

    This is the feature family that directly implements the temporal
    semantics TeLEX's t_delay templates were mining, e.g.
      F[0,5]((T3 > th1) & F[0,t_delay?](T4 > th2))
    which asks 'once T3 goes high, how long until T4 does'. Encoding it as
    a numeric lag lets the rule voter use it without running synthesis.
    """
    onsets = {}
    t = df['time'].values if 'time' in df.columns else np.arange(len(df))
    for ch in available:
        v = df[ch].values.astype(float)
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        mad = mad if mad > 1e-9 else (v.std() + 1e-9)
        z = np.abs((v - med) / mad)
        hits = np.flatnonzero(z >= k)
        onsets[ch] = float(t[hits[0]]) if len(hits) else np.nan
    return onsets


def compute_features(df):
    available = [c for c in RAW_CHANNELS if c in df.columns]
    row = {}

    # ---- raw per-channel features ----
    raw_peak = {}
    for ch in available:
        vals = df[ch].values.astype(float)
        peak = np.max(np.abs(vals))
        rms = float(np.sqrt(np.mean(vals ** 2)))
        floor = float(np.min(vals))
        ceil_ = float(np.max(vals))
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        argmax_t = float(np.argmax(np.abs(vals))) / len(vals)  # 0..1 position in window
        half = len(vals) // 2
        e1 = float(np.mean(vals[:half] ** 2))
        e2 = float(np.mean(vals[half:] ** 2))
        late_ratio = e2 / (e1 + e2) if (e1 + e2) > 0 else 0.5

        raw_peak[ch] = peak
        row[f'{ch}_peak'] = peak
        row[f'{ch}_rms'] = rms
        row[f'{ch}_floor'] = floor
        row[f'{ch}_ceil'] = ceil_
        row[f'{ch}_ptp'] = ceil_ - floor
        row[f'{ch}_mean'] = mean
        row[f'{ch}_std'] = std
        row[f'{ch}_argmaxpos'] = argmax_t
        row[f'{ch}_laterat'] = late_ratio

    overall_peak = max(raw_peak.values()) if raw_peak else 0.0
    overall_rms = np.sqrt(np.mean([row[f'{ch}_rms'] ** 2 for ch in available])) if available else 0.0

    for ch in available:
        row[f'{ch}_frac'] = raw_peak[ch] / overall_peak if overall_peak > 0 else 0.0
        # per-file-normalized peak: this file's own cross-channel RMS as the "gain" reference
        row[f'{ch}_normpeak'] = raw_peak[ch] / overall_rms if overall_rms > 0 else 0.0
        row[f'{ch}_normrms'] = row[f'{ch}_rms'] / overall_rms if overall_rms > 0 else 0.0

    # ---- onset timing: WHEN each electrode first goes high, and the lag
    # between pairs (which electrode leads, and by how long) ----
    onsets = channel_onsets(df, available)
    for ch in available:
        row[f'{ch}_onset'] = onsets[ch]

    group_onset = {}
    for gname, chs in ONSET_GROUPS.items():
        vals = [onsets[c] for c in chs if c in onsets and not np.isnan(onsets[c])]
        group_onset[gname] = float(np.mean(vals)) if vals else np.nan
        row[f'{gname}_gonset'] = group_onset[gname]
    for ga, gb in ONSET_GROUP_PAIRS:
        va, vb = group_onset.get(ga, np.nan), group_onset.get(gb, np.nan)
        row[f'{ga}~{gb}_gonsetlag'] = (va - vb) if not (np.isnan(va) or np.isnan(vb)) else None

    # ---- pairwise raw features ----
    for a, b in itertools.combinations(available, 2):
        diff = df[a].values - df[b].values
        row[f'{a}-{b}_netdiff'] = float(diff.max() + diff.min())
        row[f'{a}-{b}_absmeandiff'] = float(np.mean(np.abs(diff)))
        pa, pb = raw_peak[a], raw_peak[b]
        row[f'{a}/{b}_ratio'] = (pa / pb) if pb > 0 else None
        row[f'{b}/{a}_ratio'] = (pb / pa) if pa > 0 else None
        # negative lag => a fired before b
        oa, ob = onsets[a], onsets[b]
        row[f'{a}-{b}_onsetlag'] = (oa - ob) if (not np.isnan(oa) and not np.isnan(ob)) else None

    # ---- TeLEX-derived signed/shape features (shared module, identical in
    # analyze_full_dataset.py and classify.py) ----
    if "telex" not in ABLATE:
        row.update(telex_features(df, available))

    # ---- spectral band power / lateralisation index ----
    if USE_BANDPOWER:
        row.update(bandpower_features(df, available))

    # ---- electrode-group energy (mirrors slidingwindow.py's vote methods) ----
    if "groupenergy" not in ABLATE:
        row.update(group_energy_features(df, available))

    # ---- temporal dynamics: the movement's shape over the window ----
    if "dynamics" not in ABLATE:
        row.update(dynamics_features(df, available))

    # ---- spatial re-reference, Hjorth, Hilbert phase, complexity ----
    # spatial family measured 82.4% vs 83.4% without it -- 119 mostly
    # duplicate features that give the greedy search more chances to
    # overfit. CAR is also a no-op for pairwise differences. Opt in
    # with SPATIAL=1.
    # temporal persistence: min/max/sustain/recurrence, the scalar equivalents
    # of the always/eventually monitors mined in rtamt_layer/
    # measured 81.5% vs 82.0% without it. The stats are real -- T3T4_zdiff_tmax
    # reaches 0.711 for LEFT -- but they are weaker than the existing features
    # and 58 more columns give the greedy search more room to overfit (spread
    # widened from +/-6.6 to +/-8.4). Opt in with PERSISTENCE=1.
    if os.environ.get("PERSISTENCE", "0") not in ("0", "", "false", "False") and "persistence" not in ABLATE:
        row.update(persistence_features(df, available))

    USE_SPATIAL = os.environ.get("SPATIAL", "0") not in ("0", "", "false", "False")
    if USE_SPATIAL and "spatial" not in ABLATE:
        row.update(spatial_features(df, available))

    # ---- STL statistics: the scalar form of the mined temporal monitors ----
    # ON by default, unlike the two families above, because SELECTED_RULES now
    # REFERS to these 18 features -- without them those rules would silently
    # never fire. It is 18 columns, not 58 or 119, and every one was already
    # validated as useful before being added, so it does not carry the
    # "more columns, more room to overfit" problem the others do.
    # Cost is ~0.12 s per window. ABLATE=stl turns it off.
    if "stl" not in ABLATE:
        from lib.stl_features import stl_features
        row.update(stl_features(df))

    if "onset" in ABLATE:
        row = {k: v for k, v in row.items()
               if not (k.endswith("_onset") or k.endswith("_gonset")
                       or "_onsetlag" in k or "_gonsetlag" in k)}
    if "pairwise" in ABLATE:
        row = {k: v for k, v in row.items()
               if not (k.endswith("_netdiff") or k.endswith("_absmeandiff")
                       or k.endswith("_ratio"))}
    if "perchannel" in ABLATE:
        keep_sfx = ("_peak", "_rms", "_floor", "_ceil", "_ptp", "_mean", "_std",
                    "_argmaxpos", "_laterat", "_frac", "_normpeak", "_normrms")
        row = {k: v for k, v in row.items()
               if not (k.split("_")[0] in RAW_CHANNELS and k.endswith(keep_sfx))}
    return row, overall_peak, overall_rms

