"""Rebuild the stream before each stored 3-second window, from the windows themselves.

The monitor's stream-background gate needs the samples that came before a window. On a live stream
they are simply the previous samples. For the stored dataset (extracted_npz_dataset_v2) they are
rebuilt from the dataset's own windows -- no other recording is read:

  * every stored window is the slice ``run[run_offset : run_offset + 768]`` of one run, with the
    window's own per-channel mean removed;
  * windows of the same run are placed at their offsets in order; each window's removed mean is
    restored from its overlap with the windows already placed (the overlap must agree to 1e-3 uV,
    otherwise ``ValueError``);
  * a stretch covered without gaps is then the run up to one constant, which no high-pass filter
    sees. The history of a window is the covered stretch before it, at most ``seconds`` long,
    shifted so that it continues the stored (mean-removed) window exactly.

    from monitoring.stream_history import windows_with_history
    for index, window, history in windows_with_history("extracted_npz_dataset_v2/sub_01/ses-05/clean_epochs.npz"):
        label = monitor.classify(window, history)       # history is None when nothing precedes the window
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

from .rtamt_signals import CHANNELS, FS, WINDOW_SAMPLES, RtamtConfig

_TOLERANCE_UV = 1e-3


def default_history_seconds(config: RtamtConfig | None = None) -> float:
    cfg = config or RtamtConfig()
    return cfg.background_settle_seconds + cfg.background_seconds


def rebuild_runs(data) -> dict:
    """{(task, run): (signal (8, n) uV, covered (n,) bool, stretch_start (n,) int)} from one session."""
    x = np.asarray(data["epochs"])[:, :len(CHANNELS)] * 1e6
    task = np.asarray(data["task"]).astype(str)
    run = np.asarray(data["run"]).astype(int)
    offset = np.asarray(data["run_offset"]).astype(int)
    runs = {}
    for key in sorted(set(zip(task, run))):
        members = np.flatnonzero((task == key[0]) & (run == key[1]))
        end = int(offset[members].max()) + WINDOW_SAMPLES
        signal = np.zeros((len(CHANNELS), end))
        covered = np.zeros(end, bool)
        for k in members[np.argsort(offset[members], kind="stable")]:
            o = int(offset[k])
            seen = covered[o:o + WINDOW_SAMPLES]
            if seen.any():
                shift = (signal[:, o:o + WINDOW_SAMPLES][:, seen] - x[k][:, seen]).mean(axis=1)
                residual = np.abs(signal[:, o:o + WINDOW_SAMPLES][:, seen] - x[k][:, seen] - shift[:, None]).max()
                if residual > _TOLERANCE_UV:
                    raise ValueError(f"task {key[0]} run {key[1]} offset {o}: overlapping windows disagree by {residual:.3g} uV")
                signal[:, o:o + WINDOW_SAMPLES] = x[k] + shift[:, None]
            else:
                signal[:, o:o + WINDOW_SAMPLES] = x[k]
            covered[o:o + WINDOW_SAMPLES] = True
        stretch_start = np.zeros(end, int)
        first = 0
        for t in range(end):
            if not covered[t]:
                first = t + 1
            stretch_start[t] = first
        runs[key] = (signal, covered, stretch_start)
    return runs


def windows_with_history(path, seconds: float | None = None,
                         config: RtamtConfig | None = None) -> Iterator[Tuple[int, np.ndarray, Optional[np.ndarray]]]:
    """Yield (index, window uV (8, 768), history uV (8, h) or None) for every stored window."""
    limit = int(round((default_history_seconds(config) if seconds is None else seconds) * FS))
    data = np.load(Path(path), allow_pickle=True)
    runs = rebuild_runs(data)
    x = np.asarray(data["epochs"])[:, :len(CHANNELS)] * 1e6
    task = np.asarray(data["task"]).astype(str)
    run = np.asarray(data["run"]).astype(int)
    offset = np.asarray(data["run_offset"]).astype(int)
    for k in range(len(x)):
        signal, _, stretch_start = runs[(task[k], run[k])]
        o = int(offset[k])
        begin = max(int(stretch_start[o]), o - limit)
        if begin >= o:
            yield k, x[k], None
            continue
        constant = signal[:, o:o + WINDOW_SAMPLES].mean(axis=1, keepdims=True)
        yield k, x[k], signal[:, begin:o] - constant


__all__ = ["rebuild_runs", "windows_with_history", "default_history_seconds"]
