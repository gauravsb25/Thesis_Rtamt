"""Explicit OFFLINE timestamp-assisted crops; never a live segmentation claim."""
from pathlib import Path

import numpy as np
import pandas as pd


def load_annotations(path):
    """Times must be seconds relative to the first processed-trial sample."""
    table = pd.read_csv(path)
    required = {"file", "start", "end", "description", "time_reference"}
    if not required.issubset(table.columns):
        raise ValueError(f"Annotation CSV requires columns {sorted(required)}")
    if not table.time_reference.eq("processed_trial").all():
        raise ValueError("Convert raw/session timestamps to processed_trial time first")
    values = table[["start", "end"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not (values[:, 1] > values[:, 0]).all():
        raise ValueError("Annotation intervals must be finite and have positive duration")
    if table.file.isna().any() or table.description.isna().any():
        raise ValueError("Annotation file and description cannot be missing")
    if not table.file.map(lambda name: Path(name).name == name).all():
        raise ValueError("Annotation file must be a recording basename")
    return {name:g[["start", "end", "description"]].to_dict("records")
            for name,g in table.groupby("file", sort=False)}


def overlap_seconds(start, end, intervals):
    """Union duration, so overlapping BAD intervals are not double counted."""
    pieces = sorted((max(start,a["start"]), min(end,a["end"])) for a in intervals
                    if start < a["end"] and end > a["start"])
    total, previous_end = 0., float("-inf")
    for a,b in pieces:
        total += max(0., b-max(a,previous_end))
        previous_end = max(previous_end,b)
    return total


def choose_annotated_window(candidates, times, act, bads, window_sec=3., prefix_sec=1.):
    """Fixed ranking: complete event + clean prefix, then explicit compromises."""
    a,b = float(act["start"]), float(act["end"])
    if not np.isfinite([a,b]).all() or b <= a or not candidates:
        raise ValueError("A finite ACT interval and scored candidate windows are required")
    ranked = []
    for candidate in candidates:
        start = float(times[candidate["start"]])
        end = start+window_sec
        captured = max(0., min(end,b)-max(start,a))
        fraction = captured/(b-a)
        full = start <= a+1e-10 and end >= b-1e-10
        before = max(0., min(prefix_sec,a-start))
        prefix_bad = overlap_seconds(start,start+prefix_sec,bads)
        whole_bad = overlap_seconds(start,end,bads)
        clean_before = before
        for interval in bads:
            if start < interval["end"] and start+clean_before > interval["start"]:
                clean_before = max(0., interval["start"]-start)
        complete_prefix = before >= prefix_sec-1e-10
        clean_prefix = complete_prefix and prefix_bad <= 1e-10
        tier = 0 if full and clean_prefix else 1 if full and complete_prefix else 2 if full else 3
        # Time constraints precede the existing signal score; no fitted weights.
        rank = (-tier, fraction if tier == 3 else 0., clean_before if tier >= 2 else 0.,
                -whole_bad, candidate["soft_score"], -candidate["start"])
        details = dict(selection_method="annotation_assisted", selection_tier=tier,
            annotated_start_sec=a, annotated_end_sec=b, ACT_capture_fraction=min(1.,fraction),
            full_ACT_containment=bool(full), annotation_clean_prefix=bool(clean_prefix),
            pre_ACT_prefix_sec=before, clean_pre_ACT_prefix_sec=clean_before,
            prefix_BAD_sec=prefix_bad, window_BAD_sec=whole_bad,
            baseline_limitation=not clean_prefix, incomplete_movement=not full)
        ranked.append((rank,candidate,details))
    _, chosen, details = max(ranked, key=lambda item:item[0])
    details["candidate_windows"] = len(candidates)
    details["full_ACT_candidate_exists"] = any(d["full_ACT_containment"] for _,_,d in ranked)
    details["full_ACT_clean_prefix_candidate_exists"] = any(d["full_ACT_containment"] and d["annotation_clean_prefix"] for _,_,d in ranked)
    return chosen,details


def extract_annotation_window(csv_path, output_folder, window_size_sec, step_sec,
                              annotations, *, sliding_api=None):
    """One best window; use the unchanged signal-based extractor without ACT."""
    if sliding_api is None:
        from pipeline import slidingwindow as sliding_api
    acts = [a for a in annotations if a["description"].startswith("ACT_")]
    if not acts:
        rows = sliding_api.extract_sliding_mode(csv_path, output_folder, window_size_sec, step_sec, best_only=True)
        for row in rows:
            row.update(selection_method="signal_fallback", fallback_reason="No ACT timestamp for trial")
        return rows
    if len(acts) != 1:
        raise ValueError("Expected one movement interval per trial; review multiple ACT annotations")
    bads = [a for a in annotations if a["description"].upper().startswith("BAD")]
    df = pd.read_csv(csv_path)
    time_col = sliding_api.detect_time_column(df)
    marker_col = sliding_api.detect_marker_column(df)
    df = df.sort_values(time_col).reset_index(drop=True)
    channels = [c for c in df.columns if c not in (time_col,marker_col)]
    sliding_api.ensure_diff_column(df,channels)
    if "T3-T4" in df and "T3-T4" not in channels:
        channels.append("T3-T4")
    fs,dt = sliding_api.detect_sample_rate(df,time_col)
    width,step = round(window_size_sec/dt),round(step_sec/dt)
    if width <= 0 or step <= 0 or len(df) < width:
        raise ValueError("Invalid window/stride or annotated trial shorter than requested window")
    if not np.allclose(np.diff(df[time_col]),dt,atol=1e-10,rtol=0):
        raise ValueError("Uniform sampling required for annotation-assisted windows")
    label = sliding_api.parse_label_from_filename(Path(csv_path).name)
    candidates = sliding_api.sliding_windows_for_file(df,channels,width,step,label,fs)
    chosen,details = choose_annotated_window(candidates,df[time_col].to_numpy(),acts[0],bads,window_size_sec)
    start,end = chosen["start"],chosen["end"]
    selected = df.iloc[start:end+1].copy()
    selected[time_col] -= selected[time_col].iloc[0]
    path = sliding_api.output_path(output_folder,f"{Path(csv_path).stem}_{window_size_sec:g}sec.csv")
    selected.to_csv(path,index=False)
    note = "OFFLINE annotation-assisted selection"
    if details["baseline_limitation"]:
        note += "; no complete annotation-clean 1-second calibration prefix in selected window"
    if details["incomplete_movement"]:
        note += "; full movement cannot fit on requested window grid"
    return [dict(file=Path(csv_path).name, status="ok", label=label, sample_rate_hz=fs,
        total_duration_sec=float(df[time_col].iloc[-1]-df[time_col].iloc[0]), window_index=0,
        start_sample_idx=start,end_sample_idx=end, window_start_sec=float(df[time_col].iloc[start]),
        window_end_sec=float(df[time_col].iloc[end]), qualifying_score=chosen["qualifying_score"],
        soft_score=chosen["soft_score"], vote_breakdown=chosen["vote_breakdown"], note=note, **details)]
