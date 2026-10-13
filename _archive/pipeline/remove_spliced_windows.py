"""Build extracted_npz_dataset_v2: extracted_npz_dataset without windows that cross a run boundary.

Why. The task recordings join separate runs back to back. extracted_npz_dataset cut some of its
3-second windows across those joins, so 2,135 of its 23,164 windows are the end of one run spliced
to the start of a different run, recorded at another time, with a step at the join that a live
stream never contains (audit: experiments/npz_audit_2026_09_17).

How. For every window of extracted_npz_dataset, find the per-run CSV recording (dataset/.../ica_eeg,
*_eeg.csv) that contains all 768 samples: the window must equal 768 consecutive CSV samples with the
window's own mean removed, within 1e-10 V on every channel. Windows found inside one run are kept,
unchanged, with their run number and position recorded; every other window is dropped. No .fif file
is read, no label is changed and no sample is altered.

    python -m pipeline.remove_spliced_windows            # writes extracted_npz_dataset_v2/

Each output file keeps every key of the input file, plus
    run         the run number of the CSV recording the window lies in
    run_offset  the window's first sample inside that run
    v1_index    the window's row in extracted_npz_dataset
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from datetime import date
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "extracted_npz_dataset")
DST = os.path.join(ROOT, "extracted_npz_dataset_v2")
RAW = os.path.join(ROOT, "dataset", "eeg_cleaned_data")
EEG = ["Fp1", "Fp2", "F3", "Fz", "F4", "T3", "Cz", "T4"]
N = 768
TOL = 1e-10                      # volts: numerical agreement, far below any EEG difference
CLASSES = ("LEFT", "RIGHT", "DOWN", "STRAIGHT")


def load_runs(subject, session):
    runs = {}
    pattern = os.path.join(RAW, subject, session, "ica_eeg", "*_eeg.csv")
    for path in glob.glob(pattern):
        m = re.search(r"task-(\w+?)_run-(\d+)_eeg", os.path.basename(path))
        runs[(m.group(1), int(m.group(2)))] = pd.read_csv(path)[EEG].to_numpy().T
    return runs


def inside(window, run, k):
    """True if window == run[:, k:k+768] with the slice's own mean removed."""
    if k < 0 or k + N > run.shape[1]:
        return False
    seg = run[:, k:k + N]
    return float(np.abs((seg - seg.mean(axis=1, keepdims=True)) - window).max()) < TOL


def find(window, runs_of_task):
    """(run key, offset) of the run containing the whole window, or (None, None)."""
    dw = np.diff(window[0])                  # first differences do not depend on the removed mean
    for key, run in runs_of_task.items():
        if run.shape[1] < N:
            continue
        dr = np.diff(run[0])
        for k in np.flatnonzero(np.abs(dr[:run.shape[1] - N + 1] - dw[0]) < TOL):
            if abs(dr[k + 1] - dw[1]) < TOL and abs(dr[k + 2] - dw[2]) < TOL and inside(window, run, int(k)):
                return key, int(k)
    return None, None


def build_session(args):
    subject, session = args
    src = np.load(os.path.join(SRC, subject, session, "clean_epochs.npz"), allow_pickle=True)
    epochs = src["epochs"]
    task = np.array([str(t) for t in src["task"]])
    start = src["start_sample"]
    labels = np.array([str(v) for v in src["labels"]])
    runs = load_runs(subject, session)
    run_of = np.full(len(labels), -1)
    offset_of = np.full(len(labels), -1)
    for t in np.unique(task):
        runs_of_task = {key: run for key, run in runs.items() if key[0] == t}
        order = np.flatnonzero(task == t)[np.argsort(start[task == t], kind="stable")]
        previous = None
        for i in order:
            window = epochs[i, :len(EEG)]
            key = k = None
            if previous is not None:                      # consecutive windows usually share a run
                pkey, pk, ps = previous
                guess = pk + int(start[i] - ps)
                if inside(window, runs_of_task[pkey], guess):
                    key, k = pkey, guess
            if key is None:
                key, k = find(window, runs_of_task)
            if key is None:
                previous = None
                continue
            run_of[i], offset_of[i] = key[1], k
            previous = (key, k, start[i])
    keep = np.flatnonzero(run_of >= 0)
    os.makedirs(os.path.join(DST, subject, session), exist_ok=True)
    out = {}
    for name in src.files:
        value = src[name]
        out[name] = value[keep] if (value.ndim >= 1 and len(value) == len(labels)) else value
    out["run"] = run_of[keep]
    out["run_offset"] = offset_of[keep]
    out["v1_index"] = keep
    np.savez_compressed(os.path.join(DST, subject, session, "clean_epochs.npz"), **out)
    return dict(subject=subject, session=session, windows_in=int(len(labels)), windows_out=int(len(keep)),
                per_class={c: {"before": int((labels == c).sum()), "after": int((labels[keep] == c).sum())}
                           for c in CLASSES})


def main():
    sessions = sorted((os.path.basename(os.path.dirname(os.path.dirname(p))), os.path.basename(os.path.dirname(p)))
                      for p in glob.glob(os.path.join(SRC, "*", "ses-*", "clean_epochs.npz")))
    if os.path.exists(DST) and os.listdir(DST):
        sys.exit(f"{DST} already exists and is not empty; remove it first to rebuild")
    with Pool(max(1, min(5, (os.cpu_count() or 2) - 1))) as pool:
        rows = pool.map(build_session, sessions)
    totals = {c: {"before": sum(r["per_class"][c]["before"] for r in rows),
                  "after": sum(r["per_class"][c]["after"] for r in rows)} for c in CLASSES}
    manifest = {
        "built": str(date.today()),
        "source": "extracted_npz_dataset",
        "rule": "keep a window only if all 768 samples lie inside ONE run of the per-run CSV recordings "
                "(exact match within 1e-10 V after removing the window mean); windows spliced across two "
                "runs are dropped; no sample or label is changed; no .fif file is read",
        "windows_before": sum(r["windows_in"] for r in rows),
        "windows_after": sum(r["windows_out"] for r in rows),
        "totals": totals,
        "sessions": rows,
    }
    with open(os.path.join(DST, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    with open(os.path.join(DST, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(
            "# extracted_npz_dataset_v2\n\n"
            "`extracted_npz_dataset` without the windows that were cut across the join between two separate runs.\n\n"
            f"- windows: {manifest['windows_before']} -> {manifest['windows_after']}\n"
            + "".join(f"- {c}: {totals[c]['before']} -> {totals[c]['after']}\n" for c in CLASSES)
            + "\nEvery kept window is byte-identical to its row in `extracted_npz_dataset` (`v1_index`) and lies entirely "
              "inside one recorded run (`run`, starting at sample `run_offset`). Built by "
              "`python -m pipeline.remove_spliced_windows`; audit in `experiments/npz_audit_2026_09_17`.\n")
    print(json.dumps({k: manifest[k] for k in ("windows_before", "windows_after", "totals")}, indent=1))


if __name__ == "__main__":
    main()
