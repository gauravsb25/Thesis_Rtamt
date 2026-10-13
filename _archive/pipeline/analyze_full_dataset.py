"""
Full-dataset, all-8-channel statistical analysis of the 3-second EEG windows.

3 seconds because that's what the real-time monitor actually runs on
(continuous, non-overlapping 3-second windows) -- an earlier pass of this
analysis was done on 5-second windows, which don't match deployment and
have been retired (see git history if you need to compare).

This script:
  - Reads ALL 442 files in 3SEC_EXTRACTS (both sub-01 and sub-11), not a
    10- or 100-file subset.
  - Computes features on every one of the 8 raw channels (Fp1, Fp2, F3, Fz,
    F4, T3, Cz, T4), not just the T3/T4/Fz/Cz quartet the old rules leaned on.
  - Computes BOTH raw (microvolt) and per-file-normalized versions of every
    feature, because EEG amplitude scale is known to drift across sessions/
    subjects (different impedance, gain, electrode placement). We check this
    explicitly below.
  - Scores every feature's discriminating power PER SUBJECT as well as
    pooled, so we can see which features are genuinely characteristic of a
    movement vs. which just happen to separate classes because of a
    subject-specific amplitude quirk (the classic reason a rule looks great
    in-sample and then falls apart on new sessions).

Output: prints a report and writes ranked feature tables to
CLASSIFICATION_RESULT/full_feature_analysis_<CLASS>.csv
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.paths import bootstrap; bootstrap()
from lib import paths

import os
import itertools
import numpy as np

import lib.subject_norm as subject_norm
from lib.features import compute_features, channel_onsets, RAW_CHANNELS as _RC
import pandas as pd
from lib.corpus_layout import list_corpus_csvs
from lib.constants import CLASSES, RAW_CHANNELS

INPUT_FOLDER = paths.EXTRACTS
OUTPUT_DIR = paths.RESULTS

# subject calibration file shows the per-subject median and IQR for each feature, which is used to normalise
CALIB_CSV = paths.results("subject_calibration.csv")

# Per-subject calibration (subject_norm.py). OFF by default.
# Measured on the current 2-subject corpus it costs ~14 points of CV accuracy,
# but NOT because normalising is wrong: sub-11 contributes zero STRAIGHT
# trials, so the two subjects' medians sit at different points in class space
# and centring shifts them inconsistently. STRAIGHT (0.995 bacc, defined
# entirely by sub-01) is what collapses. Restricted to LEFT/RIGHT/DOWN --
# the classes both subjects actually have -- calibration is neutral
# (+0.6 / -2.0 / +1.0 points).
# TURN THIS ON (SUBJECT_NORM=1) once every subject contributes all four
# classes. It is the right way to absorb between-subject gain differences as
# the corpus grows, and it frees the rule search from having to stick to
# scale-invariant features.
SUBJECT_NORM = os.environ.get("SUBJECT_NORM", "0") not in ("0", "", "false", "False")

# SUBJECT_NORM=all  -> calibrate every feature
# SUBJECT_NORM=raw  -> calibrate ONLY raw-amplitude features, leaving the ones
#                      that are already gain-independent untouched.
# Normalising an already scale-free feature does not remove a gain difference
# (there is none left to remove); it removes real between-subject variation in
# a quantity that was already comparable. Sign features (+-1) and latencies in
# seconds are actively damaged by it.
SUBJECT_NORM_MODE = os.environ.get("SUBJECT_NORM_MODE", "all")
SCALE_FREE_SUFFIXES = ('_frac', '_normrms', '_normpeak', '_ratio', '_skew', '_kurt',
                       '_rel_', '_asym_', '_relenergy', '_enratio', '_peak_polarity',
                       '_onset', '_gonset', '_onsetlag', '_gonsetlag',
                       '_spec_centroid', 'n_active', 'mean_active')


def _is_scale_free(name):
    return any(k in name for k in SCALE_FREE_SUFFIXES)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_meta(filename):
    name = filename.lower()
    label = None
    for kw in ["left", "right", "down", "straight"]:
        if kw in name:
            label = kw.upper()
            break
    sub = filename.split("_")[0]  # sub-01, sub-11
    ses = None
    for part in filename.split("_"):
        if part.startswith("ses-"):
            ses = part
            break
    return label, sub, ses


ONSET_Z = 2.0

# Anatomical banks. Group-mean onsets are more stable than any single
# electrode's onset (averaging 2-3 channels suppresses the per-electrode
# jitter that makes individual lags near-chance), and they turned out to
# carry the one genuinely cross-subject-consistent onset finding in this
# dataset -- see analyze_onset_lags.py: during DOWN the central electrode
# Cz fires roughly 110 ms AFTER the temporal pair, whereas during LEFT and
# RIGHT it fires before them. The temporal-vs-central relationship is also
# TeLEX-endorsed at the channel level (the (Cz,T3) pair appears in the
# mined templates), so this is a generalisation of a confirmed relationship
# rather than a new unvalidated one.
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


def best_threshold_bacc(values, is_class):
    valid = ~pd.isna(values)
    values = values[valid].values
    is_class = is_class[valid].values
    n_pos, n_neg = is_class.sum(), (~is_class).sum()
    if len(values) == 0 or n_pos == 0 or n_neg == 0:
        return 0.5, None, 'above'
    best_bacc, best_thresh, best_dir = 0.5, None, 'above'
    uniq = np.unique(values)
    # subsample thresholds for speed if too many unique values
    if len(uniq) > 400:
        uniq = np.quantile(uniq, np.linspace(0, 1, 400))
    for t in uniq:
        pred_above = values >= t
        recall = (pred_above & is_class).sum() / n_pos
        spec = ((~pred_above) & (~is_class)).sum() / n_neg
        bacc = (recall + spec) / 2
        if bacc > best_bacc:
            best_bacc, best_thresh, best_dir = bacc, t, 'above'
        pred_below = ~pred_above
        recall2 = (pred_below & is_class).sum() / n_pos
        spec2 = ((~pred_below) & (~is_class)).sum() / n_neg
        bacc2 = (recall2 + spec2) / 2
        if bacc2 > best_bacc:
            best_bacc, best_thresh, best_dir = bacc2, t, 'below'
    return best_bacc, best_thresh, best_dir


def main():
    csv_files = list_corpus_csvs(INPUT_FOLDER)
    print(f"Found {len(csv_files)} files in {INPUT_FOLDER}")

    rows = []
    for f in csv_files:
        name = os.path.basename(f)
        label, sub, ses = parse_meta(name)
        if label is None:
            continue
        df = pd.read_csv(os.path.join(INPUT_FOLDER, f))
        feats, overall_peak, overall_rms = compute_features(df)
        rows.append({"file": name, "label": label, "subject": sub, "session": ses,
                      "overall_peak": overall_peak, "overall_rms": overall_rms, **feats})

    feat_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # PER-SUBJECT CALIBRATION -- must happen HERE, before any threshold is
    # computed. Every threshold downstream (full_feature_analysis_*.csv,
    # SELECTED_RULES) is then expressed in calibrated units, and classify.py
    # applies the same transform via CLASSIFICATION_RESULT/subject_calibration.csv.
    # Normalising after thresholds were fitted on raw values silently breaks
    # every rule -- the numbers stop meaning the same thing.
    # ------------------------------------------------------------------
    if SUBJECT_NORM:
        fcols = [c for c in feat_df.columns
                 if c not in ('file', 'label', 'subject', 'session')
                 and pd.api.types.is_numeric_dtype(feat_df[c])]
        if SUBJECT_NORM_MODE == "raw":
            fcols = [c for c in fcols if not _is_scale_free(c)]
        raw = feat_df[fcols].replace([np.inf, -np.inf], np.nan)
        stats = subject_norm.fit(raw, feat_df['subject'].values)
        subject_norm.save(stats, CALIB_CSV)
        feat_df[fcols] = subject_norm.apply(raw, feat_df['subject'].values, stats)
        print(f"per-subject calibration: ON ({feat_df['subject'].nunique()} subjects) -> {CALIB_CSV}")
    else:
        print("per-subject calibration: OFF")

    print(f"Loaded {len(feat_df)} labeled files. Subjects: {feat_df['subject'].value_counts().to_dict()}")
    print(f"Class counts: {feat_df['label'].value_counts().to_dict()}\n")

    # ------------------------------------------------------------------
    # 1) Check the amplitude-scale-mismatch hypothesis directly
    # ------------------------------------------------------------------
    print("=" * 90)
    print("SUBJECT AMPLITUDE SCALE CHECK (median peak per channel per subject)")
    print("=" * 90)
    for ch in RAW_CHANNELS:
        col = f'{ch}_peak'
        by_sub = feat_df.groupby('subject')[col].median()
        print(f"  {col:12}: " + "  ".join(f"{s}={v:6.2f}" for s, v in by_sub.items()))
    print(f"\n  overall_peak: " + "  ".join(
        f"{s}={v:6.2f}" for s, v in feat_df.groupby('subject')['overall_peak'].median().items()))
    print(f"  overall_rms : " + "  ".join(
        f"{s}={v:6.2f}" for s, v in feat_df.groupby('subject')['overall_rms'].median().items()))

    # ------------------------------------------------------------------
    # 2) Per-feature, per-class discriminating power: pooled vs per-subject
    # ------------------------------------------------------------------
    feature_cols = [c for c in feat_df.columns if c not in
                    ['file', 'label', 'subject', 'session', 'overall_peak', 'overall_rms']]

    subjects = feat_df['subject'].unique().tolist()

    for cls in CLASSES:
        print("\n" + "=" * 90)
        print(f"CLASS: {cls}")
        print("=" * 90)
        is_class_all = feat_df['label'] == cls
        results = []
        for col in feature_cols:
            bacc_all, thresh_all, dir_all = best_threshold_bacc(feat_df[col], is_class_all)
            per_sub_bacc = {}
            for sub in subjects:
                sub_mask = feat_df['subject'] == sub
                sub_df = feat_df[sub_mask]
                is_class_sub = sub_df['label'] == cls
                # score THIS SAME threshold (from pooled fit) on each subject's own data,
                # to see if it actually generalizes rather than re-fitting per subject
                if thresh_all is None:
                    per_sub_bacc[sub] = 0.5
                    continue
                vals = sub_df[col]
                valid = ~pd.isna(vals)
                v = vals[valid].values
                ic = is_class_sub[valid].values
                if len(v) == 0 or ic.sum() == 0 or (~ic).sum() == 0:
                    per_sub_bacc[sub] = float('nan')
                    continue
                pred = (v >= thresh_all) if dir_all == 'above' else (v <= thresh_all)
                recall = (pred & ic).sum() / ic.sum()
                spec = ((~pred) & (~ic)).sum() / (~ic).sum()
                per_sub_bacc[sub] = (recall + spec) / 2
            worst_sub = min(per_sub_bacc.values()) if per_sub_bacc else 0.5
            results.append({
                "feature": col, "pooled_bacc": bacc_all, "threshold": thresh_all,
                "direction": dir_all, "worst_subject_bacc": worst_sub,
                **{f"bacc_{s}": v for s, v in per_sub_bacc.items()}
            })

        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values("worst_subject_bacc", ascending=False)
        res_df.to_csv(os.path.join(OUTPUT_DIR, f"full_feature_analysis_{cls}.csv"), index=False)

        print(f"\nTop 12 features ranked by WORST-CASE (cross-subject generalizing) balanced accuracy:")
        top = res_df.head(12)
        for _, r in top.iterrows():
            subcols = [c for c in res_df.columns if c.startswith("bacc_")]
            subvals = "  ".join(f"{c.replace('bacc_','')}={r[c]:.2f}" for c in subcols)
            print(f"  {r['feature']:22} pooled={r['pooled_bacc']:.3f}  worst_sub={r['worst_subject_bacc']:.3f}  "
                  f"({subvals})  thr={r['threshold']:.2f} {r['direction']}")

        print(f"\n  [for comparison] Top 6 by POOLED bacc alone (may be subject-scale artifacts):")
        top_pooled = res_df.sort_values("pooled_bacc", ascending=False).head(6)
        for _, r in top_pooled.iterrows():
            subcols = [c for c in res_df.columns if c.startswith("bacc_")]
            subvals = "  ".join(f"{c.replace('bacc_','')}={r[c]:.2f}" for c in subcols)
            print(f"  {r['feature']:22} pooled={r['pooled_bacc']:.3f}  worst_sub={r['worst_subject_bacc']:.3f}  ({subvals})")

    feat_df.to_csv(os.path.join(OUTPUT_DIR, "full_feature_table.csv"), index=False)
    print(f"\nFull feature table saved to {OUTPUT_DIR}/full_feature_table.csv")


if __name__ == "__main__":
    main()
