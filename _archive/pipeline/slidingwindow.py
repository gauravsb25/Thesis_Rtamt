
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.paths import bootstrap; bootstrap()
from lib import paths


# Extract one best three-second window per trial:
# python slidingwindow.py --mode sliding --step 1 --best-only
# Omit --best-only to save all scored sliding windows.
# Optional OFFLINE timestamp assistance: add --annotations processed_timeline.csv.
# Trials without ACT timestamps keep the original signal-based selection.

# The search is label-guided: the class is parsed from the filename and used to
# pick which electrode tier to score, and to hunt for the quietest window on
# straight trials. Windows therefore cannot be extracted from unlabelled data.

# ============================================================================


import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from lib.corpus_layout import list_corpus_csvs, output_path

# ============================================================================
# CONFIG
# ============================================================================

INPUT_FOLDER = paths.PROCESSED_TRACES
OUTPUT_FOLDER = paths.EXTRACTS
LOG_CSV = paths.at("3SEC_EXTRACTS", "Logs", "extraction_log.csv")

# The dataset marks unusable stretches with BAD_noise annotations. Without this
# the vote could place a window inside one -- 43 windows did, 27 of them
# entirely. The dataset's own reference epochs drop any window touching a BAD
# segment, so avoiding them matches the authors' intent.
# Build with:  python tools/export_bad_segments.py
BAD_SEGMENTS = paths.results("bad_segments.csv")
AVOID_BAD = os.environ.get("AVOID_BAD", "1") not in ("0", "", "false", "False")

_BAD_BY_FILE = {}
if AVOID_BAD and os.path.exists(BAD_SEGMENTS):
    _bad_df = pd.read_csv(BAD_SEGMENTS)
    for _f, _g in _bad_df.groupby("file"):
        _BAD_BY_FILE[_f] = list(zip(_g.bad_start.values, _g.bad_end.values))


def bad_penalty_mask(fname, times, window_sec):
    """True where a window STARTING at that time would touch a BAD segment."""
    segs = _BAD_BY_FILE.get(fname)
    if not segs:
        return None
    t = np.asarray(times, dtype=float)
    hit = np.zeros(len(t), dtype=bool)
    for a, b in segs:
        hit |= (t < b) & (t + window_sec > a)
    return hit

WINDOW_SIZE_SEC = 3.0
SLIDE_STEP_SEC = None  # None -> defaults to WINDOW_SIZE_SEC (non-overlapping) at runtime

TIME_COL_CANDIDATES = ["time", "Time", "timestamp", "t"]
MARKER_COL_CANDIDATES = ["marker", "Marker", "markers"]
DIFF_COL_CANDIDATES = ["T3-T4", "T3T4", "T3_T4"]

ACTIVITY_Z_THRESHOLD = 1.0 # z score threshold is used to determine if a method's window is considered "qualifying" for voting.
STRAIGHT_ABSOLUTE_CEILING_UV = 25.0

# Which electrodes to score for each movement class. Selected using the label
# parsed from the filename.
CLASS_ELECTRODE_TIERS = {
    "left": [
        {"channels": ["T3"],             "weight": 3.0},
        {"channels": ["T3", "T4"],       "weight": 2.5},
        {"channels": ["T3", "F3", "Cz"], "weight": 2.0},
        {"channels": ["T3-T4"],          "weight": 1.0},
    ],
    "right": [
        {"channels": ["T4"],             "weight": 3.0},
        {"channels": ["T3-T4"],          "weight": 2.5, "direction": "negative"},
        {"channels": ["T4", "F4", "Cz"], "weight": 2.0},
        {"channels": ["T3", "T4"],       "weight": 1.5},
    ],
    "down": [
        {"channels": ["Fz"],             "weight": 3.0},
        {"channels": ["Fz", "Cz"],       "weight": 2.5},
        {"channels": ["Fz", "F3", "F4"], "weight": 2.0},
        {"channels": ["T3-T4"],          "weight": 1.5, "direction": "positive"},
    ],
}

WEIGHT_ALL_ELECTRODES_JOINT = 2.0  # weight for the "all electrodes together" method, which sums energy across all channels
WEIGHT_MOST_ELECTRODES_TOGETHER = 1.5 # weight for the "most electrodes together" method, which counts how many channels are simultaneously active
WEIGHT_MAX_SWING = 2.0
# Tuned against the 390 ACT_* annotations in sliding --step 1 --best-only mode.
# 3.0 / 1.5 beats the no-U4/U5 baseline on every axis at once:
#   containment 76.4% -> 88.2%,  misses 1.5% -> 0.8%,  CV 81.8% -> 83.3%
# Pushing U4 harder keeps improving containment (89% at weight 15) but COSTS
# accuracy -- the vote stops being class-sensitive and every class's window
# starts to look alike. Physical window accuracy and class separability are
# not the same objective; these weights are the balance point.
WEIGHT_DELTA_BAND = float(os.environ.get("W_DELTA", 3.0))
WEIGHT_STEP_SHIFT = float(os.environ.get("W_STEP", 1.5))   # secondary; halves the complete-miss rate # weight for the "max swing" method, which looks for the largest peak-to-trough swing across all channels

STRAIGHT_QUIET_WEIGHTS = {
    "global_quietest": 3.0, # weight for the "global quietest" method, which looks for the window with the lowest overall energy across all channels
    "temporal_quietest": 2.5, # weight for the "temporal quietest" method, which looks for the window with the lowest energy in a temporal context
    "lowest_variance": 2.0, # weight for the "lowest variance" method, which looks for the window with the lowest variance in energy
    "absolute_range_check": 2.0,
    "delta_quietest": 3.0, # weight for the movement-band method, inverted: a resting window has the least 0.5-3 Hz power # weight for the "absolute range check" method, which looks for the window with the smallest absolute range in energy
}

CLASS_KEYWORDS = ["left", "right", "down", "straight"]


# ============================================================================
# COLUMN / SAMPLE-RATE DETECTION
# ============================================================================

def detect_time_column(df):
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No time column found. Columns present: {list(df.columns)}")


def detect_marker_column(df):
    for c in MARKER_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def detect_sample_rate(df, time_col):
    dt = df[time_col].diff().dropna()
    dt = dt[dt > 0] # drop any non-positive differences (in case of duplicate timestamps)
    if len(dt) == 0:
        raise ValueError("Could not determine sample rate — time column has no valid gaps.")
    median_dt = dt.median() # use median to be robust against outliers, we are using median instead of mean to avoid the influence of any irregularities in the time column
    return 1.0 / median_dt, median_dt


def parse_label_from_filename(filename): # used for logging and the authorized class-guided extraction
    name = filename.lower()
    for kw in CLASS_KEYWORDS:
        if kw in name:
            return kw
    return None


def ensure_diff_column(df, channels): # ensure that the difference column T3-T4 exists in the dataframe, if not, create it from available candidates
    for cand in DIFF_COL_CANDIDATES:
        if cand in df.columns:
            if cand != "T3-T4":
                df["T3-T4"] = df[cand]
            return True
    if "T3" in df.columns and "T4" in df.columns:
        df["T3-T4"] = df["T3"] - df["T4"]
        return True
    return False


# ============================================================================
# CORE ENERGY / ACTIVITY HELPERS -- used by sliding-window scoring
# ============================================================================


def zscore_channels(df, channels): 

    """
    # compute z-scores for each channel in the dataframe, using robust median and MAD (median absolute deviation) instead of mean and std to be more robust to outliers
    # z score is computed as (value - median) / MAD, where MAD is the median of the absolute deviations from the median
    # helps in identifying outliers in the data, which is important for detecting activity windows,
    # z score in here implemented is just for the purpose of computing robust activity z scores for each channel, which will be used later to evaluate how standout a particular window is relative to the channel's own distribution
    # robust activity z score mean that we are using median and MAD instead of mean and std to be more robust to outliers, which is important in EEG data where there can be spikes or artifacts that would skew the mean and std

    #formula : z = (x - median) / MAD
    # mad is calculated for each channel and for each channel we are calculating the median and MAD and then computing the z score for each value in that channel
    #need of z scores: z scores are used to standardize the data, so that we can compare values across different channels and different files, even if they have different scales or units. By converting to z scores, we can identify which windows are standout relative to the channel's own distribution, which is important for detecting activity windows.


    #the Z-score is calculated per channel, but it is applied to every value (sample) within that channel

    """
    z = pd.DataFrame(index=df.index)
    for ch in channels:
        vals = df[ch].astype(float)
        med = vals.median()
        mad = (vals - med).abs().median()     # robust MAD (median absolute deviation)
        mad = mad if mad > 1e-9 else vals.std() + 1e-9
        z[ch] = (vals - med) / mad
    return z


# this unction computes a robust activity z-score for a given series at a specific index. The robust activity z-score is a measure of how much a particular value stands out compared to the rest of the values in the series, using median and MAD (median absolute deviation) instead of mean and standard deviation to be more robust to outliers.
def robust_activity_z(series, idx):  

    # helps in finding out How much does this window stand out compared to all other windows in this file? 
    # will be used in the voting logic to determine if a method's window is considered "qualifying" for voting.

    
    valid = series.dropna()  # drop NaN values from the series to avoid issues with median and MAD calculations
    med = valid.median() 
    mad = (valid - med).abs().median() # robust MAD (median absolute deviation)
    mad = mad if mad > 1e-9 else valid.std() + 1e-9 # fallback to standard deviation if MAD is too small
    if idx not in series.index or pd.isna(series.loc[idx]):
        return None
    return (series.loc[idx] - med) / mad


# ---- Rolling-SERIES builders (computed once per file, reused everywhere) ----
# means that these functions compute rolling statistics over the entire dataframe for the specified channels and window size, and return the resulting series. These series are then used to evaluate sliding windows.


"""
this function computes the rolling energy series for the specified channels in the dataframe.
 It first computes the z-scores for each channel, then calculates the energy as the sum of squared z-scores 
 across all channels. Finally, it computes a rolling sum of the energy over the specified window size (in samples)
   and returns both the rolling energy series and the z-score dataframe.
"""

def energy_rolling_series(df, channels, window_samples):
    z = zscore_channels(df, channels)
    energy = (z ** 2).sum(axis=1)
    # energy is computed as the sum of squared z-scores across all channels, which gives a measure of overall activity in the window. 
    rolling = energy.rolling(window=window_samples, min_periods=window_samples).sum()
    return rolling, z


"""
compute the robust Z-score of every EEG sample in each channel using that channel's median and MAD.
 If the absolute Z-score of a sample is greater than or equal
   to the threshold (1.0), that sample in that channel is considered active
"""

def count_rolling_series(df, channels, window_samples):
    z = zscore_channels(df, channels)
    active_count = (z.abs() >= ACTIVITY_Z_THRESHOLD).sum(axis=1)
    rolling = active_count.rolling(window=window_samples, min_periods=window_samples).sum()
    return rolling, z

#rolling  - How many active-channel events occurred within each window.,z is the z score dataframe


def swing_rolling_series(df, channels, window_samples):
    rolling_max = df[channels].rolling(window=window_samples, min_periods=window_samples).max()
    rolling_min = df[channels].rolling(window=window_samples, min_periods=window_samples).min()
    swing = (rolling_max - rolling_min).sum(axis=1)
    return swing, rolling_max, rolling_min #for each window and each channel


def variance_rolling_series(df, channels, window_samples):
    return df[channels].rolling(window=window_samples, min_periods=window_samples).var().sum(axis=1)


def signed_rolling_series(df, channel, window_samples): 

    """
    computes the rolling sum of the specified channel in the dataframe.
      This is used for directional methods where the sign of the signal matters (e.g., for STRAIGHT's quiet methods).
    """

    vals = df[channel].astype(float)
    return vals.rolling(window=window_samples, min_periods=window_samples).sum()


def range_ok_rolling_series(df, channels, window_samples, ceiling):  # for straight

    """
    computes the rolling count of samples that are within the specified range for each channel.
    the specified range is defined by the ceiling value, which is the maximum absolute value allowed for a sample to be considered "within range".
    
    """

    if ceiling is None:
        return None
    within = (df[channels].abs() <= ceiling).all(axis=1)
    rolling_ok_count = within.astype(int).rolling(window=window_samples, min_periods=window_samples).sum()
    return rolling_ok_count


# ============================================================================
# METHOD SPEC BUILDER — describes every vote, used identically by both modes 
# ============================================================================

    """
    #this is the main function that builds a dictionary of methods to be used for evaluating windows.
    #  Each method is associated with a rolling series, a weight, and a kind that describes how to interpret the series. 
    # The methods are either class-agnostic (U1, U2, U3) or class-specific (C1, C2, etc.) based on the label provided.
    #  If no label is provided, only the universal methods are used.
    
    """


    #  Each method is associated with a rolling series, a weight, and a kind that describes how to interpret the series. 
    # The methods are either class-agnostic (U1, U2, U3) or class-specific (C1, C2, etc.) based on the label provided.
    #  If no label is provided, only the universal methods are used.

    """
    if no label is provided, only the universal methods are used. If a label is provided,
      the class-specific methods are also included based on the label's electrode tiers defined in CLASS_ELECTRODE_TIERS.
    """


# ============================================================================
# DELTA-BAND / STEP METHODS
# A head turn is a slow deflection, not an oscillation: its artifact energy
# sits below ~3 Hz. Broadband energy (U1) therefore also rewards EMG bursts,
# blinks and electrode pops -- everything that is NOT the movement. Scoring
# the 0.5-3 Hz band alone strips those out.
# Measured against the 390 ACT_* ground-truth annotations, picking the window
# by delta power puts the whole movement inside it 95.6% of the time, versus
# 72.6% for broadband energy and 78.2% for the full label-hinted vote as it
# stood. The effect is not knife-edge -- 0.5-4 Hz and 1-4 Hz both give 95.4%,
# while 0.5-2 Hz collapses to 28.7% (too narrow) and 0.5-12 Hz falls to 89.7%
# (starts admitting real EEG rhythms again).
# U5 is the sustained level shift across the window: |mean(2nd half) -
# mean(1st half)|. Alone it reaches 91.0%. Paired with U4 it trims the
# complete-miss rate (1.5% vs 2.1% for U4 alone), which matters more to the
# classifier than the last point of containment.
# ============================================================================

DELTA_BAND = (0.5, 3.0)


def _bandpass(x, fs, lo, hi):
    """Zero-phase band-pass. filtfilt so the filter adds no time shift --
    a lagged envelope would move the window it is supposed to locate."""
    from scipy import signal as _sig
    nyq = fs / 2.0
    lo_n, hi_n = max(lo / nyq, 1e-4), min(hi / nyq, 0.99)
    if lo_n >= hi_n:
        return x
    b, a = _sig.butter(4, [lo_n, hi_n], btype="band")
    pad = 3 * max(len(a), len(b))
    if len(x) <= pad:
        return x
    return _sig.filtfilt(b, a, x, axis=0)


def delta_rolling_series(df, channels, window_samples, fs,
                         band=DELTA_BAND):
    """Rolling power in the movement band, summed over channels.

    Channels are z-scored first, exactly as energy_rolling_series does, so a
    single high-gain electrode cannot dominate the vote.
    """
    z = zscore_channels(df, channels)
    filt = _bandpass(z.to_numpy(dtype=float), fs, band[0], band[1])
    power = pd.Series((filt ** 2).sum(axis=1), index=z.index)
    return power.rolling(window=window_samples, min_periods=window_samples).sum(), z


def step_rolling_series(df, channels, window_samples):
    """Rolling |mean(second half) - mean(first half)|, summed over channels.

    A sustained level shift: high when the window straddles a movement that
    moves the signal from one level to another, low on symmetric oscillation.
    """
    z = zscore_channels(df, channels)
    half = max(window_samples // 2, 1)
    csum = z.cumsum()
    prev = csum.shift(half).fillna(0.0)
    prev2 = csum.shift(window_samples).fillna(0.0)
    second = (csum - prev) / half
    first = (prev - prev2) / half
    step = (second - first).abs().sum(axis=1)
    step[: window_samples - 1] = np.nan
    return step


def build_method_series(df, channels, window_samples, label=None, fs=256.0): 
    methods = {}


    if label == "straight":  
        energy_all, _ = energy_rolling_series(df, channels, window_samples)
        methods["Q1_global_quietest"] = {"series": energy_all, "kind": "invert",
                                          "weight": STRAIGHT_QUIET_WEIGHTS["global_quietest"]} 

        temporal_channels = [c for c in ["T3", "T4"] if c in channels]
        if len(temporal_channels) == 2:
            energy_temp, _ = energy_rolling_series(df, temporal_channels, window_samples)
            methods["Q2_temporal_quietest"] = {"series": energy_temp, "kind": "invert",
                                                "weight": STRAIGHT_QUIET_WEIGHTS["temporal_quietest"]}

        delta_q, _ = delta_rolling_series(df, channels, window_samples, fs)
        methods["Q5_delta_quietest"] = {"series": delta_q, "kind": "invert",
                                         "weight": STRAIGHT_QUIET_WEIGHTS["delta_quietest"]}

        var_series = variance_rolling_series(df, channels, window_samples)
        methods["Q3_lowest_variance"] = {"series": var_series, "kind": "invert",
                                          "weight": STRAIGHT_QUIET_WEIGHTS["lowest_variance"]}

        range_series = range_ok_rolling_series(df, channels, window_samples, STRAIGHT_ABSOLUTE_CEILING_UV)
        if range_series is not None:
            methods[f"Q4_absolute_range_under_{STRAIGHT_ABSOLUTE_CEILING_UV}uV"] = {
                "series": range_series, "kind": "range",
                "weight": STRAIGHT_QUIET_WEIGHTS["absolute_range_check"],
                "window_samples": window_samples}
        return methods

    energy_all, _ = energy_rolling_series(df, channels, window_samples)
    methods["U1_all_electrodes_joint_peak"] = {"series": energy_all, "kind": "energy",
                                                "weight": WEIGHT_ALL_ELECTRODES_JOINT}

    count_all, _ = count_rolling_series(df, channels, window_samples)
    methods["U2_most_electrodes_together"] = {"series": count_all, "kind": "energy",
                                               "weight": WEIGHT_MOST_ELECTRODES_TOGETHER}

    swing_all, _, _ = swing_rolling_series(df, channels, window_samples)
    methods["U3_max_peak_trough_swing"] = {"series": swing_all, "kind": "energy",
                                            "weight": WEIGHT_MAX_SWING}

    # movement-band power: the single best locator of the actual movement
    delta_all, _ = delta_rolling_series(df, channels, window_samples, fs)
    methods["U4_delta_band_power"] = {"series": delta_all, "kind": "energy",
                                       "weight": WEIGHT_DELTA_BAND}

    step_all = step_rolling_series(df, channels, window_samples)
    methods["U5_step_shift"] = {"series": step_all, "kind": "energy",
                                 "weight": WEIGHT_STEP_SHIFT}


    """
    for straight the used methods are Q1_global_quietest, Q2_temporal_quietest, Q3_lowest_variance, Q4_absolute_range_under_25.0uV
    for left the used methods are C1_T3, C2_T3+T4, C3_T3+F3+Cz, C4_T3-T4
    for right the used methods are C1_T4, C2_T3-T4_negative, C3_T4+F4+Cz, C4_T3+T4
    for down the used methods are C1_Fz, C2_Fz+Cz,  C3_Fz+F3+F4, C4_T3-T4_positive
    """

    if label in CLASS_ELECTRODE_TIERS: # if a label is provided and it is one of the known classes (left, right, down), then we add the class-specific methods based on the electrode tiers defined in CLASS_ELECTRODE_TIERS
        for tier_idx, tier in enumerate(CLASS_ELECTRODE_TIERS[label]): # for each tier in the electrode tiers for the given label, we check if any of the channels in that tier are present in the dataframe. If so, we compute the rolling series for that tier and add it to the methods dictionary with the appropriate weight and kind.
            tier_channels = [c for c in tier["channels"] if c in channels]
            if not tier_channels:
                continue
            weight = tier["weight"]
            direction = tier.get("direction")

            if direction and len(tier_channels) == 1:
                series = signed_rolling_series(df, tier_channels[0], window_samples)
                kind = "signed_neg" if direction == "negative" else "signed_pos"
                name = f"C{tier_idx+1}_{tier_channels[0]}_{direction}"
            else:
                series, _ = energy_rolling_series(df, tier_channels, window_samples)
                kind = "energy"
                name = f"C{tier_idx+1}_{'+'.join(tier_channels)}"

            methods[name] = {"series": series, "kind": kind, "weight": weight}

    return methods


def evaluate_methods_at_index(methods, idx):

    """
    this function evaluates every method's activity_z at ONE specific index. 
    It returns a list of tuples containing (method_name, weight, activity_z, qualifies).
      The qualifies boolean indicates whether the activity_z meets or exceeds the ACTIVITY_Z_THRESHOLD.
    """

    results = []
    for name, spec in methods.items():
        kind = spec["kind"]  #kind is the type of method, which can be "energy", "invert", "signed_neg", "signed_pos", or "range"
        series = spec["series"] # series is the rolling series for the method, for eg- all electrodes joint peak, most electrodes together, max peak-trough swing, etc.
        weight = spec["weight"]

        if kind == "range":
            if idx not in series.index or pd.isna(series.loc[idx]):
                continue
            fully_within = series.loc[idx] >= spec["window_samples"]
            az = 3.0 if fully_within else -3.0  # binary, but keep sign meaningful
            qualifies = fully_within
            results.append((name, weight, az, qualifies))
            continue

        raw_z = robust_activity_z(series, idx)
        if raw_z is None:
            continue

        if kind == "invert":
            az = -raw_z          # smaller raw value (quieter) = bigger activity_z
        elif kind == "signed_neg":
            az = -raw_z          # most-negative raw sum = bigger activity_z
        else:  # "energy" or "signed_pos"
            az = raw_z

        qualifies = az >= ACTIVITY_Z_THRESHOLD
        results.append((name, weight, az, qualifies))
    return results


# ============================================================================
# SLIDING MODE — score EVERY window at fixed stride, using the SAME series
# ============================================================================

def sliding_windows_for_file(df, channels, window_samples, step_samples, label=None, fs=256.0):

    """
    this function implements the sliding window mode, where every possible window in the file is scored at fixed stride positions.
      It evaluates each fixed-stride window position using the method series
      and records the scores.
    """


    methods = build_method_series(df, channels, window_samples, label, fs)

    n = len(df)
    results = []
    end_idx = window_samples - 1
    while end_idx < n:
        evals = evaluate_methods_at_index(methods, end_idx)
        qualifying_score = sum(w for _, w, az, q in evals if q)
        soft_score = sum(w * max(0.0, az) for _, w, az, q in evals)
        breakdown = "; ".join(f"{name}(w={w:.1f},z={az:.2f},qualifies={q})"
                               for name, w, az, q in evals)
        start_idx = end_idx - window_samples + 1
        results.append({
            "start": start_idx, "end": end_idx,
            "qualifying_score": round(qualifying_score, 2),
            "soft_score": round(soft_score, 2),
            "n_methods_qualifying": sum(1 for *_, q in evals if q),
            "vote_breakdown": breakdown,
        })
        end_idx += step_samples

    return results


# ============================================================================
# PER-FILE EXTRACTION — SLIDING MODE
# ============================================================================

def extract_sliding_mode(csv_path, output_folder, window_size_sec, step_sec,
                          best_only=False):

    """
    this function implements the sliding mode extraction for a single file.
    It reads the CSV file, detects the time and marker columns, computes the sample rate, and then either extracts the entire file
      if it's shorter than the window size or scores every possible window at fixed stride positions.
    The results are saved to new CSV files in the output folder, and a list of result dictionaries is returned with metadata about each window's extraction process.
    """
    fname = Path(csv_path).name
    base_result = {"file": fname, "status": "ok", "label": None,
                    "sample_rate_hz": None, "total_duration_sec": None, "note": ""}
    rows = []

    try:
        label = parse_label_from_filename(fname)
        base_result["label"] = label if label else "UNKNOWN"
        label_for_search = label

        df = pd.read_csv(csv_path)
        time_col = detect_time_column(df)
        marker_col = detect_marker_column(df)
        df = df.sort_values(time_col).reset_index(drop=True)

        channels = [c for c in df.columns if c not in [time_col, marker_col]]
        if not channels:
            raise ValueError("No channel columns found after excluding time/marker.")

        ensure_diff_column(df, channels)
        if "T3-T4" in df.columns and "T3-T4" not in channels:
            channels.append("T3-T4")

        sample_rate, dt = detect_sample_rate(df, time_col)
        total_duration = df[time_col].iloc[-1] - df[time_col].iloc[0]
        base_result["sample_rate_hz"] = round(sample_rate, 2)
        base_result["total_duration_sec"] = round(total_duration, 3)

        window_samples = max(1, round(window_size_sec / dt))
        step_samples = max(1, round(step_sec / dt))

        if total_duration <= window_size_sec or len(df) <= window_samples:
            window_df = df.copy()
            window_df[time_col] = window_df[time_col] - window_df[time_col].iloc[0]
            os.makedirs(output_folder, exist_ok=True)
            out_path = output_path(output_folder, f"{Path(csv_path).stem}_win0_{window_size_sec:g}sec.csv")
            window_df.to_csv(out_path, index=False)
            rows.append({**base_result, "window_index": 0,
                         "window_start_sec": round(df[time_col].iloc[0], 3),
                         "window_end_sec": round(df[time_col].iloc[-1], 3),
                         "start_sample_idx": 0, "end_sample_idx": len(df) - 1,
                         "qualifying_score": None, "soft_score": None,
                         "n_methods_qualifying": None, "vote_breakdown": "",
                         "note": "file shorter than requested window size"})
            return rows

        windows = sliding_windows_for_file(df, channels, window_samples, step_samples, label_for_search, sample_rate)

        if best_only and windows:
            # Prefer a window that avoids the dataset's BAD_noise segments.
            # Only fall back to the contaminated pick if EVERY candidate window
            # in this trial touches a BAD segment -- dropping the trial would
            # lose it entirely, and a noisy window still carries the label.
            clean = windows
            mask = bad_penalty_mask(fname,
                                    [df[time_col].iloc[w["start"]] for w in windows],
                                    window_size_sec)
            if mask is not None and (~mask).any():
                clean = [w for w, bad in zip(windows, mask) if not bad]
            best_window = max(clean, key=lambda w: w["soft_score"])
            windows = [best_window]

        os.makedirs(output_folder, exist_ok=True)
        for w_idx, w in enumerate(windows):
            window_df = df.iloc[w["start"]:w["end"] + 1].copy()
            window_df[time_col] = window_df[time_col] - window_df[time_col].iloc[0]
            if best_only:
                out_name = f"{Path(csv_path).stem}_{window_size_sec:g}sec.csv"
            else:
                out_name = f"{Path(csv_path).stem}_win{w_idx}_{window_size_sec:g}sec.csv"
            out_path = output_path(output_folder, out_name)
            window_df.to_csv(out_path, index=False)

            rows.append({
                **base_result, "window_index": w_idx,
                "window_start_sec": round(df[time_col].iloc[w["start"]], 3),
                "window_end_sec": round(df[time_col].iloc[w["end"]], 3),
                "start_sample_idx": int(w["start"]), "end_sample_idx": int(w["end"]),
                "qualifying_score": w["qualifying_score"], "soft_score": w["soft_score"],
                "n_methods_qualifying": w["n_methods_qualifying"],
                "vote_breakdown": w["vote_breakdown"], "note": "",
            })

    except Exception as e:
        rows.append({**base_result, "status": "error", "note": str(e),
                     "window_index": None, "window_start_sec": None, "window_end_sec": None,
                     "start_sample_idx": None, "end_sample_idx": None,
                     "qualifying_score": None, "soft_score": None,
                     "n_methods_qualifying": None, "vote_breakdown": ""})

    return rows


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extract scored sliding movement windows.")
    parser.add_argument("--mode", default="sliding", choices=["sliding"],
                        help="Sliding-window extraction (the only supported mode)")
    parser.add_argument("--window", type=float, default=WINDOW_SIZE_SEC,
                        help=f"Window size in seconds (default: {WINDOW_SIZE_SEC})")
    parser.add_argument("--step", type=float, default=1.0,
                        help="Sliding stride in seconds (default: 1)")
    parser.add_argument("--best-only", action="store_true",
                        help="Save only the highest soft_score window per input trial")
    parser.add_argument("--annotations", default=None,
                        help="OFFLINE assistance: CSV with ACT/BAD times relative to the processed trial; requires --best-only")
    parser.add_argument("--input", default=INPUT_FOLDER, help="Input folder")
    parser.add_argument("--output", default=OUTPUT_FOLDER, help="Output folder")
    args = parser.parse_args()
    if args.window <= 0 or args.step <= 0:
        parser.error("--window and --step must be positive")
    if args.annotations and not args.best_only:
        parser.error("--annotations requires --best-only")
    annotation_map = None
    if args.annotations:
        from pipeline.annotation_windows import load_annotations, extract_annotation_window
        annotation_map = load_annotations(args.annotations)

    if not os.path.isdir(args.input):
        print(f"Input folder not found: {args.input}")
        return
    csv_files = list_corpus_csvs(args.input)
    if not csv_files:
        print(f"No CSV files found in {args.input}")
        return

    log_folder = os.path.join(args.output, "Logs")
    os.makedirs(log_folder, exist_ok=True)
    log_csv_path = os.path.join(log_folder, "extraction_log.csv")
    print(f"SLIDING MODE — window: {args.window}s | step: {args.step}s | "
          f"files: {len(csv_files)} | best-only: {args.best_only} | label-guided search")
    if annotation_map is not None:
        print("OFFLINE annotation-assisted selection; missing ACT timestamps use signal-based fallback.")
    all_rows = []
    for fname in csv_files:
        path = os.path.join(args.input, fname)
        if annotation_map is None:
            rows = extract_sliding_mode(path, args.output, args.window, args.step, args.best_only)
        else:
            rows = extract_annotation_window(path,args.output,args.window,args.step,
                                             annotation_map.get(Path(path).name,[]))
        all_rows.extend(rows)
        status = rows[0].get("status", "ok") if rows else "error"
        flag = "OK" if status == "ok" else "FAIL"
        n_windows = len([r for r in rows if r.get("window_index") is not None])
        print(f"[{flag}] {fname:40} -> {n_windows} window(s) saved")
        if status != "ok" and rows:
            print(f"        note: {rows[0]['note']}")

    log_df = pd.DataFrame(all_rows)
    log_df.to_csv(log_csv_path, index=False)
    n_files_ok = log_df[log_df["status"] == "ok"]["file"].nunique()
    n_windows_total = len(log_df[log_df["window_index"].notna()])
    print(f"Done. {n_files_ok} files processed, {n_windows_total} total windows saved.")
    print(f"Full manifest with scores: {log_csv_path}")
    print("Window selection is label-guided and offline; this is not live segmentation.")


if __name__ == "__main__":
    main()
