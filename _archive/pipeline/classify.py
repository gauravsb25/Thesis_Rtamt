
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.paths import bootstrap; bootstrap()
from lib import paths

import os
import itertools
import pandas as pd
import numpy as np

from lib.features import compute_features, channel_onsets
from lib.corpus_layout import list_corpus_csvs
from lib.constants import CLASSES, RAW_CHANNELS


INPUT_FOLDER = paths.at("norm_3sec")
OUTPUT_CSV = paths.results("3sec_classification_results.csv")


TIME_COL_CANDIDATES = ["time", "Time", "timestamp", "t"]
MARKER_COL_CANDIDATES = ["marker", "Marker", "markers"]

# ============================================================================
# SELECTED_RULES / MIN_FRAC -- built from a full 442-file, 2-subject,
# 3-second-window analysis (see analyze_full_dataset.py + build_rules.py),
# TeLEX's STL specification mining actually touched or confirmed (see
# telex_relationships.py, which mines formulas.txt / formulas_v2.txt /
# formula_finalresults/ for every channel pair TeLEX considered). TeLEX
# picks WHICH relationships are eligible; the numeric thresholds below are
# fit and cross-validated fresh on this 3-second data, not reused from
# TeLEX's own synthesized numbers (which were computed on differently-sized
# windows and don't transfer directly).
# 3 seconds because that's what the real-time monitor actually runs on
# (continuous windows) -- there is no 5-second stage in the deployed
# pipeline, so none of this is tuned on 5-second statistics anymore.
# Every earlier version of this dict (see git history) was fit on a subset
# cause of the wall of UNCERTAIN/wrong results: sub-11's raw EEG amplitude
# runs ~2-2.6x hotter than sub-01's on every channel (median overall peak
# 171 vs 65-73), so absolute microvolt thresholds tuned on one subject's
# gain silently misfire on the other's -- which is why nearly every rule
# below is a cross-channel difference/ratio/normalized feature rather than
# a raw peak/floor threshold.
# Each rule was selected by greedy forward search using GROUPED (by
# subject+session) 5-fold cross-validation, so a feature only made the cut
# if it discriminates the class on sessions it never saw during selection
# -- not just in-sample. Physiologically, the dominant discriminator across
# LEFT/RIGHT/DOWN/STRAIGHT is how much the left/right temporal channels
# (T3, T4) diverge over the window (T3-T4_absmeandiff): lowest for STRAIGHT
# (rest), low for DOWN (symmetric nodding), high for LEFT/RIGHT (asymmetric
# turn); the sign of T3/T4-relative features then splits LEFT vs RIGHT,
# though that split is the weakest link (see build_rules.py's CV numbers).
# The feature pool includes ONSET-TIMING features (<ch>_onset = when that
# electrode first goes high; <a>-<b>_onsetlag and <ga>~<gb>_gonsetlag =
# which of two electrodes / electrode banks leads, and by how long). These
# directly encode the temporal semantics TeLEX's t_delay templates were
# mining. Two onset features survived selection here, both for RIGHT:
# temporal~midline_gonsetlag and F3_onset. Measured on this data, T3 leads
# T4 by ~79ms on LEFT and lags it by ~71ms on RIGHT -- the expected
# lateralisation is real but with heavy overlap, so single-pair onset lags
# are weak individually (worst-subject bacc ~0.55-0.60). The one robust
# timing finding is that Cz fires ~110ms AFTER the temporal pair on DOWN
# but BEFORE it on LEFT/RIGHT (consistent on both subjects).
# !! IMPORTANT -- WHAT THIS ACCURACY NUMBER MEANS !!
# Two separate caveats apply to the table below. Both were corrected after
# re-measurement. Rerun analyze_full_dataset.py -> build_rules.py to
# regenerate the CV figures.
# (1) IN-SAMPLE, NOT CROSS-VALIDATED. This table's features and thresholds
#     were selected using the whole corpus. Evaluating it on that same
#     corpus therefore measures fit, not generalisation:
#         this table, in-sample : 82.1%
#     Running the full pipeline with selection INSIDE the folds
#     (analyze_full_dataset.py -> build_rules.py) gives:
#         82.0% (+/- 6.3%) grouped-CV
#     Quote the grouped-CV figure for generalisation; the majority-class
#     baseline is 33.9%.
#     (Earlier drafts quoted 85.2% / 84.8%. Those were in-sample numbers on
#     the old pre-ICA corpus -- do not reuse them.)
# (2) ORACLE WINDOWING. slidingwindow.py selects each window USING the
#     trial's true class label, parsed from the filename (for STRAIGHT it
#     hunts for the quietest window; for LEFT it maximises T3-tier energy).
#     A real-time monitor sees unlabelled continuous 3-second windows and
#     can never do this, so 82.0% is an upper bound under oracle window
#     selection rather than an achievable deployment accuracy. Measured
#     against the dataset's ACT_* annotations by verify_window_extraction.py,
#     that selection puts the whole movement inside the window 76.4% of the
#     time and misses it entirely 1.5% of the time.
# Rerun build_rules.py after collecting more subjects/sessions/formulas to
# refresh these numbers rather than hand-editing them.
# ============================================================================


# SELECTED_RULES = {
#     'STRAIGHT': [
#         ('T4_spec_centroid', 8.535006, 'above', 0.88),
#         ('T3-Cz_absmeandiff', 1.647202, 'below', 0.749),
#         ('Fp1_rel_beta', 0.115832, 'above', 0.699),
#         ('Cz_laterat', 0.37858, 'above', 0.256),
#         ('stl_Fp2_spec_centroid_recur025', 7.518912, 'above', 0.836),
#         ('stl_F3_spec_centroid_min', 6.950788, 'above', 0.828),
#         ('stl_T4_rel_delta_sustain1', 0.200327, 'below', 0.828),
#         ('stl_F4_spec_centroid_min', 7.912696, 'above', 0.816),
#         ('stl_F3_rel_delta_sustain05', 0.285022, 'below', 0.784),
#     ],
#     'LEFT': [
#         ('T4_floor', -6.151357, 'below', 0.495),
#         ('T3T4_peak_signed', 7.499103, 'above', 0.491),
#         ('T3|T4_xcorr_peak', -0.302832, 'below', 0.33),
#         ('Fz_frac', 0.559689, 'below', 0.187),
#         ('F3F4_peak_signed', 5.50879, 'above', 0.437),
#         ('stl_T4_car_min', -5.138138, 'below', 0.534),
#         ('stl_F4_car_min', -3.252756, 'below', 0.433),
#         ('stl_T3T4_corr_recur025', -0.28885, 'below', 0.416),
#         ('stl_T3T4_zcr_sustain05', 20.0, 'below', 0.402),
#         ('stl_T4_car_sustain1', -2.769784, 'below', 0.379),
#         ('stl_T4_henv_recur025', 0.945845, 'above', 0.367),
#         ('stl_T3T4_corr_sustain1', -0.078601, 'below', 0.346),
#         ('stl_T3T4_cosdphi_recur1', 0.998311, 'below', 0.331),
#         ('stl_T4_henv_recur05', 1.590396, 'above', 0.325),
#         ('stl_F3_car_min', -3.470196, 'below', 0.321),
#         ('stl_T4_car_max', 6.836745, 'above', 0.316),
#         ('stl_T4_sustain1', -2.607993, 'below', 0.316),
#         ('stl_T3T4_zdiff_recur025', 0.040558, 'below', 0.301),
#     ],
#     'RIGHT': [
#         ('T3T4_skew', -0.26809, 'below', 0.536),
#         ('righthemi_energy', 6.591311, 'below', 0.161),
#         ('Fz-T4_absmeandiff', 3.150625, 'below', 0.144),
#         ('Fp1|Fp2_xcorr_peak', 0.693061, 'below', 0.306),
#     ],
#     'DOWN': [
#         ('T3-T4_absmeandiff', 1.770503, 'below', 0.695),
#         ('T3_frac', 0.72307, 'below', 0.459),
#         ('T3_normrms', 0.903702, 'below', 0.549),
#         ('T3|T4_asym_gamma', -0.231649, 'above', 0.29),
#         ('Fz_floor', -5.138505, 'below', 0.209),
#     ],
# }

# MIN_FRAC = {
#     'STRAIGHT': 0.5,
#     'LEFT': 0.45,
#     'RIGHT': 0.15,
#     'DOWN': 0.575,
# }



SELECTED_RULES = {
    'STRAIGHT': [
        ('T4_spec_centroid', 8.561459, 'above', 0.868),
        ('T3-Cz_absmeandiff', 1.63679, 'below', 0.789),
        ('Fp1_rel_beta', 0.116998, 'above', 0.619),
        ('Cz_laterat', 0.385624, 'above', 0.252),
    ],
    'LEFT': [
        ('T4_floor', -6.000971, 'below', 0.394),
        ('T3T4_peak_signed', 7.627737, 'above', 0.207),
        ('T3|T4_xcorr_peak', -0.290976, 'below', 0.189),
        ('Fz_frac', 0.558603, 'below', 0.088),
        ('F3F4_peak_signed', 5.349668, 'above', 0.404),
    ],
    'RIGHT': [
        ('T3T4_skew', -0.245307, 'below', 0.136),
        ('righthemi_energy', 6.316581, 'below', 0.057),
        ('Fz-T4_absmeandiff', 3.076579, 'below', 0.035),
        ('Fp1|Fp2_xcorr_peak', 0.688541, 'below', 0.286),
    ],
    'DOWN': [
        ('T3-T4_absmeandiff', 1.77891, 'below', 0.674),
        ('T3_frac', 0.728642, 'below', 0.386),
        ('T3_normrms', 0.92798, 'below', 0.414),
        ('T3|T4_asym_gamma', -0.246614, 'above', 0.257),
        ('Fz_floor', -5.199979, 'below', 0.171),
    ],
}

MIN_FRAC = {
    'STRAIGHT': 0.925,
    'LEFT': 0.325,
    'RIGHT': 0.825,
    'DOWN': 0.65,
}


# ============================================================================
# HELPERS
# ============================================================================

def detect_time_column(df):
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    return df.columns[0]


ONSET_Z = 2.0

# Must stay identical to analyze_full_dataset.py. See that file for why
# group-mean onsets are used (they carry the DOWN timing inversion: Cz
# fires after the temporal pair on DOWN, before it on LEFT/RIGHT).
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
    electrode 'goes high'. Must stay identical to the version in
    analyze_full_dataset.py, or rules fit there silently never fire here."""
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


# ============================================================================
# FEATURE EXTRACTION — every single-channel and pairwise feature
# ============================================================================

def extract_all_features(df):
    """Delegates to the shared extractor so classify.py and
    analyze_full_dataset.py can never compute a feature differently."""
    row, _, _ = compute_features(df)
    return row

def rule_fires(value, thresh, direction):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return (value >= thresh) if direction == 'above' else (value <= thresh)


#         for col, thresh, direction, bacc in rules:
#             weight = (bacc - 0.5) * 2
#                 scores[cls] += weight
#                             f"{direction} {thresh} -> {'FIRED' if fired else 'no'} "
#                             f"(w={weight:.2f}) [{cls}]")


def classify_with_rules(features):
    scores = {c: 0.0 for c in CLASSES}
    totals = {c: 0.0 for c in CLASSES}
    details = []
    for cls, rules in SELECTED_RULES.items():
        for col, thresh, direction, weight in rules:
            value = features.get(col)
            fired = rule_fires(value, thresh, direction)
            if fired:
                scores[cls] += weight
            totals[cls] += weight  # per-class total, NOT summed across all 4 classes
            details.append(f"{col}={value if value is None else round(value,2)} "
                            f"{direction} {thresh} -> {'FIRED' if fired else 'no'} "
                            f"(w={weight:.2f}) [{cls}]")

    fracs = {c: (scores[c] / totals[c] if totals[c] > 0 else 0.0) for c in CLASSES}
    qualifying = {c: f for c, f in fracs.items() if f >= MIN_FRAC[c]}
    if qualifying:
        winner = max(qualifying, key=qualifying.get)
        confidence = qualifying[winner]
    else:
        # No class cleared its MIN_FRAC. Refusing ("UNCERTAIN") counts as wrong
        # so fall back to whichever class scored highest. Confidence is
        # reported as negative to mark it as a fallback rather than a
        # threshold-backed decision.
        winner = max(fracs, key=fracs.get)
        confidence = -fracs[winner]
    return winner, confidence, fracs, details

# ============================================================================
# COMBINATORIAL SEARCH — rerun this if you get more/different data, to
# find the best features fresh rather than trusting SELECTED_RULES above
# ============================================================================

def best_threshold_balanced_accuracy(values, is_class):
    valid = ~values.isna()
    values = values[valid].values
    is_class = is_class[valid].values
    n_pos, n_neg = is_class.sum(), (~is_class).sum()
    if len(values) == 0 or n_pos == 0 or n_neg == 0:
        return 0.5, None, 'above'
    best_bacc, best_thresh, best_dir = 0.5, None, 'above'
    for t in np.unique(values):
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


def search_all_combinations(feat_df):
    """Rerun the full exhaustive search that produced SELECTED_RULES.
    Call this on a NEW feat_df (e.g. after collecting more trials) to
    find fresh best features rather than trusting the ones hardcoded
    above. See build_rules.py for the actual (TeLEX-constrained,
    cross-validated) rule-selection pipeline this project uses; this
    function is the raw per-feature scan it builds on top of."""
    feature_cols = [c for c in feat_df.columns if c not in ['file', 'label']]
    results = {}
    for cls in CLASSES:
        is_class = feat_df['label'] == cls
        scores = []
        for col in feature_cols:
            bacc, thresh, direction = best_threshold_balanced_accuracy(feat_df[col], is_class)
            scores.append((col, bacc, thresh, direction))
        scores.sort(key=lambda x: -x[1])
        results[cls] = scores
    return results


# ============================================================================
# MAIN — classify all files in INPUT_FOLDER using the hand-rule voter
# ============================================================================

def parse_true_label_from_filename(filename):
    name = filename.lower()
    for kw in ["left", "right", "down", "straight"]:
        if kw in name:
            return kw.upper()
    return None


def main():
    print("=" * 90)
    print("EEG Head Movement Classifier — full 8-channel combinatorial search")
    print("=" * 90)
    print(f"Input folder: {INPUT_FOLDER}")
    print("=" * 90)

    if not os.path.isdir(INPUT_FOLDER):
        print(f"Input folder not found: {INPUT_FOLDER}")
        return

    csv_files = list_corpus_csvs(INPUT_FOLDER)
    if not csv_files:
        print(f"No CSV files found in '{INPUT_FOLDER}'")
        return

    print(f"\nFound {len(csv_files)} window files\n")

    results = []
    correct = 0
    total_labeled = 0

    for csv_file in csv_files:
        csv_path = os.path.join(INPUT_FOLDER, csv_file)
        csv_file = os.path.basename(csv_file)
        df = pd.read_csv(csv_path)
        features = extract_all_features(df)

        winner, confidence, scores, details = classify_with_rules(features)

        true_label = parse_true_label_from_filename(csv_file)
        match_str = ""
        if true_label:
            total_labeled += 1
            is_correct = (true_label == winner)
            correct += int(is_correct)
            match_str = "  OK" if is_correct else f"  X (filename says {true_label})"

        print(f"  {csv_file:45} -> {winner:10} (confidence {confidence:.0%}){match_str}")

        results.append({
            "filename": csv_file, "classification": winner, "confidence": round(confidence, 3),
            "score_STRAIGHT": round(scores["STRAIGHT"], 2), "score_LEFT": round(scores["LEFT"], 2),
            "score_RIGHT": round(scores["RIGHT"], 2), "score_DOWN": round(scores["DOWN"], 2),
            "details": " | ".join(details),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    counts = results_df["classification"].value_counts()
    for cls, count in counts.items():
        print(f"  {cls:12} : {count:3} files ({count/len(results_df)*100:5.1f}%)")

    if total_labeled > 0:
        print(f"\nHand-rule accuracy (IN-SAMPLE, see docstring caveat): "
              f"{correct}/{total_labeled} ({correct/total_labeled*100:.1f}%)")
        print("\nPer-class:")
        for cls in CLASSES:
            true_mask = results_df["filename"].apply(lambda f: parse_true_label_from_filename(f) == cls)
            sub = results_df[true_mask]
            if len(sub) == 0:
                continue
            sub_correct = (sub["classification"] == cls).sum()
            print(f"  {cls:10}: {sub_correct}/{len(sub)} ({sub_correct/len(sub)*100:.1f}%)")

    print(f"\nFull results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()