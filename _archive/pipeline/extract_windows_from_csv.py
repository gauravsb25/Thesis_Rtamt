"""Build extracted_npz_dataset_v3_csv: 3-second windows cut from the ORIGINAL per-run CSV files.

No .fif file and no earlier extraction is read. The only inputs are
``dataset/eeg_cleaned_data/<subject>/<session>/ica_eeg/*_task-<task>_run-<nn>_eeg.csv``:
eight channels, 256 Hz, plus a ``marker`` column.

WHERE THE LABEL COMES FROM
  the task in the file name (left / right / down / straight).

WHERE THE TIMING COMES FROM  (new method, independent of the .fif annotations)
  marker codes: 1000 = recording start, 1001 = recording stop, 7006 = the movement cue. Every left and
  right run has a cue; DOWN runs have none, so the cue can only narrow the search, not replace it.

  A movement is then found in the signal by a detector built to be DIFFERENT from the monitor's gate,
  so that the windows are not defined by the thing being measured:
      band          0.3-4 Hz              (where head movement sits; the monitor uses 0.5-3 Hz)
      statistic     sum over channels of |deviation| in microvolts, zero-phase filtered
                    (the monitor uses squared energy, each channel divided by its own spread)
      reference     the run's own median level
                    (the monitor uses the quietest 0.5 s inside the window)
      search        left/right: within CUE_WINDOW seconds of the cue; down: the whole run
  The movement is the largest peak of that statistic in the search region, provided it reaches
  HIGH x the run median; its span is the surrounding stretch above LOW x the run median, capped at
  MAX_SPAN seconds. Runs where nothing reaches HIGH are recorded as "not found" and contribute rest
  windows only.

WHICH WINDOWS ARE KEPT  (the rule of extracted_npz_dataset_v2, so the two are comparable)
  movement window   the whole detected span lies inside the 3 s with at least MARGIN seconds spare
                    at both ends; taken every 0.1 s
  rest window       no detected span within GUARD seconds of the 3 s; taken every REST_STRIDE seconds
  Straight runs contribute rest windows only, and any run whose detector found a span keeps that span
  away from its rest windows.

Each window is stored like v2: 9 channels (the 8 electrodes and T3-T4), volts, with the window's own
per-channel mean removed.

usage: python -m pipeline.extract_windows_from_csv [--source dataset/eeg_cleaned_data] [--out extracted_npz_dataset_v3_csv]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[1]
CH = ["Fp1", "Fp2", "F3", "Fz", "F4", "T3", "Cz", "T4"]
FS = 256
WINDOW = 3 * FS
CUE_CODE = 7006
CUE_WINDOW = 4.0        # seconds around the cue to search in a left/right run
HIGH, LOW = 3.0, 1.5    # multiples of the run's median level: peak must reach HIGH, span stays above LOW
MAX_SPAN = 2.5          # seconds
MARGIN = 0.25           # seconds of clearance inside a movement window
GUARD = 1.0             # seconds a rest window must stay away from a span
REST_STRIDE = 0.5       # seconds between rest windows (movement windows use 0.1 s)
BAND_HZ = (0.3, 4.0)          # --band changes it; 0.1-1.2 Hz was the first try (see the README)
BAND = butter(2, [BAND_HZ[0] / (FS / 2), BAND_HZ[1] / (FS / 2)], btype="band")


def movement_span(x, cue, whole_run, band=None):
    """(start, end, peak, strength) of the detected movement in samples, or None."""
    dev = np.abs(filtfilt(*(band or BAND), x, axis=-1)).sum(axis=0)
    level = float(np.median(dev)) + 1e-12
    search = np.zeros(len(dev), bool)
    if whole_run or cue is None:
        search[:] = True
    else:
        search[max(cue - int(CUE_WINDOW * FS), 0):cue + int(CUE_WINDOW * FS)] = True
    masked = np.where(search, dev, -np.inf)
    peak = int(np.argmax(masked))
    strength = dev[peak] / level
    if strength < HIGH:
        return None
    start = end = peak
    while start > 0 and dev[start - 1] > LOW * level:
        start -= 1
    while end < len(dev) - 1 and dev[end + 1] > LOW * level:
        end += 1
    cap = int(MAX_SPAN * FS)
    if end - start > cap:                          # keep the cap centred on the peak
        start, end = max(peak - cap // 2, 0), min(peak + cap // 2, len(dev) - 1)
    return start, end, peak, float(strength)


def windows_of_run(n, span):
    """(movement offsets, rest offsets) for one run of n samples."""
    move, rest = [], []
    margin = int(MARGIN * FS)
    guard = int(GUARD * FS)
    for o in range(0, n - WINDOW + 1):
        if span is not None:
            a, b = span[0], span[1]
            if o % int(0.1 * FS) == 0 and o + margin <= a and b <= o + WINDOW - margin:
                move.append(o)
            if o % int(REST_STRIDE * FS) == 0 and (o + WINDOW <= a - guard or o >= b + guard):
                rest.append(o)
        elif o % int(REST_STRIDE * FS) == 0:
            rest.append(o)
    return move, rest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(ROOT / "dataset" / "eeg_cleaned_data"))
    ap.add_argument("--out", default=str(ROOT / "extracted_npz_dataset_v3_csv"))
    ap.add_argument("--band", nargs=2, type=float, default=list(BAND_HZ))
    args = ap.parse_args(argv)
    source, out = Path(args.source), Path(args.out)
    band = butter(2, [args.band[0] / (FS / 2), args.band[1] / (FS / 2)], btype="band")
    report = {"parameters": dict(cue_window=CUE_WINDOW, high=HIGH, low=LOW, max_span=MAX_SPAN,
                                margin=MARGIN, guard=GUARD, rest_stride=REST_STRIDE,
                                band=list(args.band), window_samples=WINDOW), "sessions": {}}
    totals = {"LEFT": 0, "RIGHT": 0, "DOWN": 0, "STRAIGHT": 0}
    found = missed = 0
    for session_dir in sorted(source.glob("sub_*/ses-*")):
        files = sorted((session_dir / "ica_eeg").glob("*_eeg.csv"))
        if not files:
            continue
        epochs, labels, meta = [], [], []
        for path in files:
            task = path.name.split("task-")[1].split("_")[0]
            run = int(path.name.split("run-")[1][:2])
            frame = pd.read_csv(path)
            x = frame[CH].to_numpy().T
            if np.nanpercentile(np.abs(x), 99) < 1e-3:         # volts -> microvolts
                x = x * 1e6
            marker = frame["marker"].to_numpy()
            cues = [int(i) for i in np.flatnonzero(marker == CUE_CODE)]
            span = movement_span(x, cues[0] if cues else None, whole_run=task in ("down", "straight"), band=band)
            if task in ("left", "right", "down"):
                found += span is not None
                missed += span is None
            move, rest = windows_of_run(x.shape[1], span)
            for kind, offsets in (("move", move), ("rest", rest)):
                if kind == "move" and task == "straight":
                    continue                                    # a straight run never carries a movement label
                label = task.upper() if kind == "move" else "STRAIGHT"
                for o in offsets:
                    seg = x[:, o:o + WINDOW] * 1e-6
                    seg = seg - seg.mean(axis=1, keepdims=True)
                    epochs.append(np.vstack([seg, (seg[CH.index("T3")] - seg[CH.index("T4")])[None]]))
                    labels.append(label)
                    meta.append((task, run, o, -1 if span is None else span[0], -1 if span is None else span[1],
                                 -1 if span is None else span[2], 0.0 if span is None else span[3],
                                 cues[0] if cues else -1))
        if not epochs:
            continue
        subject, session = session_dir.parent.name, session_dir.name
        target = out / subject / session
        target.mkdir(parents=True, exist_ok=True)
        meta = list(zip(*meta))
        np.savez(target / "clean_epochs.npz", epochs=np.array(epochs), labels=np.array(labels),
                 task=np.array(meta[0]), run=np.array(meta[1], dtype=int), run_offset=np.array(meta[2], dtype=int),
                 span_start=np.array(meta[3], dtype=int), span_end=np.array(meta[4], dtype=int),
                 span_peak=np.array(meta[5], dtype=int), span_strength=np.array(meta[6], dtype=float),
                 cue=np.array(meta[7], dtype=int), source=np.array([str(source)]))
        counts = {k: int(sum(1 for v in labels if v == k)) for k in totals}
        for k in totals:
            totals[k] += counts[k]
        report["sessions"][f"{subject}/{session}"] = counts
        print(f"{subject}/{session}: " + " ".join(f"{k} {v}" for k, v in counts.items()), flush=True)
    report["totals"] = totals
    report["movements_found"] = found
    report["movements_not_found"] = missed
    (out / "manifest.json").write_text(json.dumps(report, indent=1))
    (out / "README.md").write_text(
        "# extracted_npz_dataset_v3_csv\n\n"
        "3-second windows cut from the original per-run CSV files only (no .fif file, no earlier extraction).\n"
        "Labels come from the task in the file name; the movement is located by the 0.1-1.2 Hz detector described\n"
        "in `pipeline/extract_windows_from_csv.py`, with the 7006 cue marker narrowing the search in left and\n"
        f"right runs. Windows: {totals}. Movements found {found}, not found {missed}.\n\n"
        "Because movement windows are placed by a detector, the monitor's movement gate is flattered on them; the\n"
        "false-alarm rate on rest windows and the LEFT/RIGHT/DOWN split among movement windows are not.\n")
    print("totals", totals, "movements found", found, "not found", missed)


if __name__ == "__main__":
    main()
