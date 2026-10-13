"""Re-measure the early-commit table with the CURRENT monitor (the stream-background gate).

The published table (81.97% at a 1.0 s commitment, and so on) was measured on 17 Sep, BEFORE the background
gate was adopted, so it no longer describes the shipped monitor. This recomputes it.

For every window the row monitor is run with the same stream history the window monitor gets, a verdict is
taken every 32 rows (0.125 s), and the sequence of verdicts is stored. A commitment of k seconds means: the
same movement class held for the last ceil(k * 256 / 32) verdicts; STRAIGHT is never committed early, and a
window that never commits keeps its verdict at row 768 (which equals HeadMovementMonitor).

Saved: verdicts.npz. Reported: four-class balanced accuracy and, for correctly classified movement windows,
the median time from the movement's start in the window to the decision.

usage: python run.py            (computes and reports; the cache is reused if present)
"""
import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\gaura\thesis_rtamt")
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
STEP = 32                      # a verdict every 0.125 s
CODES = {"STRAIGHT": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3, "INVALID": 0}
NAMES = np.array(["STRAIGHT", "DOWN", "LEFT", "RIGHT"])
COMMITS = (0.5, 0.75, 1.0, 1.5)


def session_job(path):
    from monitoring import RowStreamMonitor
    from monitoring.rtamt_signals import InvalidWindow, RtamtConfig
    from monitoring.stream_history import windows_with_history
    cfg = RtamtConfig()
    data = np.load(path, allow_pickle=True)
    rows = []
    for k, window, history in windows_with_history(path, config=cfg):
        try:
            monitor = RowStreamMonitor(cfg, decide_every=STEP, history=history)
            verdicts = monitor.push_block(window)
            sequence = [CODES[v.label] for v in verdicts]
        except InvalidWindow:
            sequence = [0] * (768 // STEP)
        rows.append(sequence + [0] * (768 // STEP - len(sequence)))
    path = Path(path)
    return (np.array(rows, dtype=np.int8),
            np.array([f"{path.parent.parent.name}/{path.parent.name}"] * len(rows)),
            data["labels"].astype(str), data["movement_offset"].astype(float))


def build():
    files = sorted(glob.glob(str(ROOT / "extracted_npz_dataset_v2" / "*" / "ses-*" / "clean_epochs.npz")),
                   key=lambda p: (p.split(os.sep)[-3], p.split(os.sep)[-2]))
    with Pool(6) as pool:
        parts = pool.map(session_job, files)
    np.savez(HERE / "verdicts.npz",
             verdicts=np.concatenate([p[0] for p in parts]),
             session=np.concatenate([p[1] for p in parts]),
             label=np.concatenate([p[2] for p in parts]),
             offset=np.concatenate([p[3] for p in parts]))


def commit(sequence, hold):
    """(row, label) of the first commitment, or (768, last verdict) if there is none."""
    need = max(int(round(hold * 256 / STEP)), 1)
    for i in range(need - 1, sequence.shape[0]):
        window = sequence[i - need + 1:i + 1]
        if window[0] != 0 and np.all(window == window[0]):
            return (i + 1) * STEP, int(window[0])
    return 768, int(sequence[-1])


def main():
    if not (HERE / "verdicts.npz").exists():
        build()
    d = dict(np.load(HERE / "verdicts.npz", allow_pickle=True))
    verdicts, session, label, offset = d["verdicts"], d["session"].astype(str), d["label"].astype(str), d["offset"]
    dev = np.array([s.startswith("sub_01") and int(s[-2:]) <= 30 for s in session])
    splits = (("sub_01 1-30", dev), ("sub_01 31-36", np.char.startswith(session, "sub_01") & ~dev),
              ("sub_11", np.char.startswith(session, "sub_11")))
    print(f"{len(label)} windows, a verdict every {STEP / 256:.3f} s\n")
    print(f"{'decision':22s}" + "".join(f"{name:>26s}" for name, _ in splits))
    print(f"{'':22s}" + "".join(f"{'balanced   median delay':>26s}" for _ in splits))
    for hold in (None,) + COMMITS:
        cells = []
        for _, m in splits:
            if hold is None:
                rows = np.full(len(label), 768)
                codes = verdicts[:, -1]
            else:
                out = [commit(v, hold) for v in verdicts[m]]
                rows = np.full(len(label), 768)
                codes = verdicts[:, -1].copy()
                rows[m] = [r for r, _ in out]
                codes[m] = [c for _, c in out]
            pred = NAMES[codes]
            balanced = 100 * np.mean([np.mean(pred[m & (label == k)] == k) for k in NAMES])
            move = m & (label != "STRAIGHT") & (pred == label) & np.isfinite(offset)
            delay = np.median(rows[move] / 256 - offset[move]) if move.any() else float("nan")
            cells.append(f"{balanced:9.2f}%{delay:14.2f} s")
        name = "wait for row 768" if hold is None else f"commit after {hold} s"
        print(f"{name:22s}" + "".join(f"{c:>26s}" for c in cells))
    print("\n(delay = time from the movement's start inside the window to the decision, over correctly"
          " classified movement windows; STRAIGHT is never committed early)")


if __name__ == "__main__":
    main()
