"""Build extracted_npz_dataset_trials: ONE 3-second window per trial, 442 in total.

Input: only the original per-trial recordings
``dataset/eeg_cleaned_data/<subject>/<session>/ica_eeg/*_task-<task>_run-<nn>_eeg.csv``
(8 channels, 256 Hz). No .fif file, no earlier extraction, and no marker code is used to find the
movement -- the marker column is read only so that its cue position can be REPORTED next to the
detection. The label is the task in the file name.

FINDING THE MOVEMENT FROM THE SIGNAL ALONE

A head movement moves every electrode at once, slowly. A blink moves the two frontopolar channels, a
bad contact moves one channel. The detector uses that difference:

  1. band-pass 0.3-4 Hz (zero-phase), where head movement sits;
  2. divide each channel by its own robust spread over the whole trial -> unit-free, so a firmly and a
     loosely sitting electrode count the same;
  3. score(t) = MEDIAN over the eight channels, smoothed over 0.25 s. A median needs at least half the
     channels to be active at the same moment, so a one- or two-channel artifact cannot produce a peak;
  4. the movement is the largest score in the trial (ignoring EDGE seconds at each end, where the filter
     is unreliable);
  5. its CORE is the stretch around that peak where the score stays above CORE_LEVEL of the peak. A turn
     is out-and-back, so the core is meant to hold both phases; it is capped at MAX_CORE seconds;
  6. the window is the 3 seconds centred on the core, moved inside the trial if it would stick out.

For a straight trial there is no movement, so the window is the QUIETEST 3 seconds (lowest mean score):
a clean rest example.

Each window is stored like the earlier datasets: 9 channels (the 8 electrodes and T3-T4), volts, with
the window's own per-channel mean removed. Every trial also stores the detection (core, peak, strength),
whether the whole core fits inside the window, how much stream precedes the window inside the trial, and
the cue position for reporting.

usage: python -m pipeline.extract_trial_windows
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
BAND_HZ = (0.3, 4.0)
SMOOTH = 0.25        # seconds
EDGE = 0.3           # seconds ignored at each end of the trial
CORE_LEVEL = 0.35    # the core is the stretch above this share of the peak score
MAX_CORE = 2.5       # seconds
CUE_CODE = 7006      # read for the report only


def movement_score(x):
    """Unit-free, artifact-resistant movement score per sample."""
    band = butter(2, [BAND_HZ[0] / (FS / 2), BAND_HZ[1] / (FS / 2)], btype="band")
    y = filtfilt(*band, x, axis=-1)
    spread = 1.4826 * np.median(np.abs(y - np.median(y, axis=1, keepdims=True)), axis=1, keepdims=True) + 1e-12
    z = np.abs(y) / spread
    score = np.median(z, axis=0)                       # at least half the channels must move together
    width = max(int(SMOOTH * FS), 1)
    return np.convolve(score, np.ones(width) / width, "same")


def detect(score):
    """(core_start, core_end, peak, strength) in samples."""
    guard = int(EDGE * FS)
    inner = np.full(len(score), -np.inf)
    inner[guard:len(score) - guard] = score[guard:len(score) - guard]
    peak = int(np.argmax(inner))
    level = CORE_LEVEL * score[peak]
    start = end = peak
    while start > 0 and score[start - 1] >= level:
        start -= 1
    while end < len(score) - 1 and score[end + 1] >= level:
        end += 1
    cap = int(MAX_CORE * FS)
    if end - start > cap:
        centre = (start + end) // 2
        start, end = max(centre - cap // 2, 0), min(centre + cap // 2, len(score) - 1)
    strength = float(score[peak] / (np.median(score) + 1e-12))
    return start, end, peak, strength


def place(n, centre):
    """Start of the 3 s window centred on `centre`, kept inside a trial of n samples."""
    return int(np.clip(centre - WINDOW // 2, 0, max(n - WINDOW, 0)))


def quietest(score, n):
    """Start of the quietest 3 s (for a straight trial)."""
    if n <= WINDOW:
        return 0
    cumulative = np.concatenate([[0.0], np.cumsum(score)])
    means = (cumulative[WINDOW:] - cumulative[:-WINDOW]) / WINDOW
    return int(np.argmin(means))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(ROOT / "dataset" / "eeg_cleaned_data"))
    ap.add_argument("--out", default=str(ROOT / "extracted_npz_dataset_trials"))
    args = ap.parse_args(argv)
    source, out = Path(args.source), Path(args.out)
    counts = {"LEFT": 0, "RIGHT": 0, "DOWN": 0, "STRAIGHT": 0}
    sessions = {}
    for session_dir in sorted(source.glob("sub_*/ses-*")):
        files = sorted((session_dir / "ica_eeg").glob("*_eeg.csv"))
        if not files:
            continue
        rows = []
        for path in files:
            task = path.name.split("task-")[1].split("_")[0]
            run = int(path.name.split("run-")[1][:2])
            frame = pd.read_csv(path)
            x = frame[CH].to_numpy().T
            if np.nanpercentile(np.abs(x), 99) < 1e-3:
                x = x * 1e6
            n = x.shape[1]
            score = movement_score(x)
            core_start, core_end, peak, strength = detect(score)
            if task == "straight":
                offset = quietest(score, n)
                core_start = core_end = peak = -1
                strength = 0.0
            else:
                offset = place(n, (core_start + core_end) // 2)
            segment = x[:, offset:offset + WINDOW] * 1e-6
            if segment.shape[1] < WINDOW:                       # a trial shorter than 3 s (none in this data)
                segment = np.pad(segment, ((0, 0), (0, WINDOW - segment.shape[1])))
            segment = segment - segment.mean(axis=1, keepdims=True)
            cues = [int(i) for i in np.flatnonzero(frame["marker"].to_numpy() == CUE_CODE)]
            rows.append(dict(
                epoch=np.vstack([segment, (segment[CH.index("T3")] - segment[CH.index("T4")])[None]]),
                label="STRAIGHT" if task == "straight" else task.upper(),
                task=task, run=run, trial_samples=n, offset=offset,
                core_start=core_start, core_end=core_end, peak=peak, strength=strength,
                core_inside=bool(core_start < 0 or (offset <= core_start and core_end < offset + WINDOW)),
                history=offset, cue=cues[0] if cues else -1))
        subject, session = session_dir.parent.name, session_dir.name
        target = out / subject / session
        target.mkdir(parents=True, exist_ok=True)
        np.savez(target / "clean_epochs.npz",
                 epochs=np.array([r["epoch"] for r in rows]),
                 labels=np.array([r["label"] for r in rows]),
                 **{k: np.array([r[k] for r in rows]) for k in
                    ("task", "run", "trial_samples", "offset", "core_start", "core_end", "peak",
                     "strength", "core_inside", "history", "cue")})
        got = {k: sum(1 for r in rows if r["label"] == k) for k in counts}
        for k in counts:
            counts[k] += got[k]
        sessions[f"{subject}/{session}"] = got
        print(f"{subject}/{session}: {len(rows)} trials " + " ".join(f"{k} {v}" for k, v in got.items()), flush=True)
    total = sum(counts.values())
    (out / "manifest.json").write_text(json.dumps(
        {"windows": total, "counts": counts, "sessions": sessions,
         "parameters": dict(band=list(BAND_HZ), smooth=SMOOTH, edge=EDGE, core_level=CORE_LEVEL,
                            max_core=MAX_CORE, window_samples=WINDOW)}, indent=1))
    (out / "README.md").write_text(
        "# extracted_npz_dataset_trials\n\n"
        f"One 3-second window per trial: {total} windows ({counts}). Cut from the original per-trial CSV\n"
        "recordings only. The movement is found from the signal alone by the median-over-channels detector in\n"
        "`pipeline/extract_trial_windows.py`; no .fif file and no marker code is used for it. A straight trial\n"
        "contributes its quietest 3 seconds. Each window has 9 channels (8 electrodes and T3-T4), in volts, with\n"
        "its own per-channel mean removed.\n")
    print("total", total, counts)


if __name__ == "__main__":
    main()
