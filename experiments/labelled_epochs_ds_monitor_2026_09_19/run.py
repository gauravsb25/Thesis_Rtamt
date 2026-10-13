"""One-off evaluation of the unchanged shipped monitor on per-session labelled_epochs_ds.npz.

This archive contains finished epochs and labels only: it has no pre-window stream samples or run
order. It therefore evaluates the monitor's documented standalone-window/contrast-gate mode.
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from monitoring.rtamt_monitors import HeadMovementMonitor, satisfied_class
from monitoring.rtamt_signals import RtamtConfig, compute_signals

CLASSES = ("LEFT", "RIGHT", "DOWN", "STRAIGHT")


def session(path: str):
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        epochs = z["epochs"][:, :8]
        labels = np.char.upper(z["labels"].astype(str))
    cfg = RtamtConfig()
    predictions = []
    # Same four RTAMT specifications as HeadMovementMonitor; no history is passed.
    monitor = HeadMovementMonitor(cfg)
    for epoch in epochs:
        signals = compute_signals(epoch, cfg, history=None)
        rho = {label: item.robustness(signals) for label, item in monitor.monitors.items()}
        predictions.append(satisfied_class(rho))
    subject, ses = path.parts[-3], path.parts[-2]
    return subject, ses, labels, np.asarray(predictions)


def counts(labels, predictions, mask):
    result = {}
    recalls = []
    for label in CLASSES:
        here = mask & (labels == label)
        correct, total = int(np.sum(predictions[here] == label)), int(np.sum(here))
        result[label] = {"correct": correct, "total": total,
                         "accuracy_percent": 100 * correct / total if total else None}
        if total:
            recalls.append(correct / total)
    correct, total = int(np.sum(predictions[mask] == labels[mask])), int(np.sum(mask))
    return {"per_class": result, "total_correct": correct, "total": total,
            "overall_accuracy_percent": 100 * correct / total,
            "balanced_accuracy_percent": 100 * float(np.mean(recalls))}


def main():
    paths = sorted((ROOT / "dataset" / "eeg_cleaned_data").glob("sub_*/ses-*/labelled_epochs_ds.npz"))
    if len(paths) != 46:
        raise RuntimeError(f"expected 46 session archives, found {len(paths)}")
    with Pool(6) as pool:
        parts = pool.map(session, [str(path) for path in paths])
    subjects = np.concatenate([np.repeat(subject, len(labels)) for subject, _, labels, _ in parts])
    sessions = np.concatenate([np.repeat(f"{subject}/{ses}", len(labels)) for subject, ses, labels, _ in parts])
    labels = np.concatenate([labels for _, _, labels, _ in parts])
    predictions = np.concatenate([predictions for _, _, _, predictions in parts])
    sub01 = subjects == "sub_01"
    ses_num = np.asarray([int(value[-2:]) for value in sessions])
    groups = {
        "sub_01_sessions_1_30": sub01 & (ses_num <= 30),
        "sub_01_sessions_31_36": sub01 & (ses_num >= 31),
        "sub_11_all_sessions": subjects == "sub_11",
        "all_46_sessions": np.ones(len(labels), dtype=bool),
    }
    report = {
        "dataset": "dataset/eeg_cleaned_data/<subject>/<session>/labelled_epochs_ds.npz",
        "files": len(paths), "windows": len(labels),
        "mode": "unchanged HeadMovementMonitor, RTAMT formulas, standalone-window mode; history=None",
        "reason_no_history": "archives contain epochs and labels only, with no pre-window stream or run-order metadata",
        "splits": {name: counts(labels, predictions, mask) for name, mask in groups.items()},
    }
    (Path(__file__).resolve().parent / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
