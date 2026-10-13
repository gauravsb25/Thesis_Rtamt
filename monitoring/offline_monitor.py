"""Offline RTAMT monitoring of RECORDED EEG: the four head-movement specifications, evaluated with the
whole recording available.

The live monitor (:mod:`monitoring.rtamt_monitors`) may only use what a stream has delivered so far: the
window and at most 5 s before it. On this dataset that is a real handicap, because the recordings are short
runs and the movement comes 2-3 s after a run starts: 74.5% of movement windows (but only 15% of rest
windows) have no usable history, so the live monitor judges most movements with its weaker fallback, the
window's own loudest-over-quietest contrast.

Offline, the whole recording is known. This monitor compares every window with the REST OF ITS OWN
RECORDING, before and after it, so every window -- movement or rest, early or late in its run -- is judged
by the same stream-background test.

THE OFFLINE BACKGROUND SIGNAL (``background``)

    1. the recording is band-passed 0.5-3 Hz (the live gate's band; causal, started in steady state at the
       first sample so there is no start-up transient);
    2. each channel's energy over the 0.5 s ending at each sample;
    3. each channel's REFERENCE = the median of that energy over the rest of the recording (every 8th
       sample), leaving out blocks that end in the 0.5 s before the window or inside it;
    4. background(t) = mean over the eight channels of energy(t) / reference, for every sample t of the window.

Every quantity is a ratio of a channel to itself, so a subject whose signals are ten times larger gives the
same values. No microvolt level is stored anywhere.

THE FOUR OFFLINE SPECIFICATIONS  (RTAMT offline evaluator, bounded FUTURE-time operators, read at t = 0,
i.e. over the whole completed window [0:767])

    MOVE     = eventually[0:767](background >= b) and eventually[0:767](delta >= d)
               and eventually[0:767](deflect >= q)
    STRAIGHT = always[0:767](background < b) or always[0:767](delta < d) or always[0:767](deflect < q)
    DOWN     = MOVE and (nod >= n) and always[0:767](sideways <= v)
    LEFT     = MOVE and ((nod < n) or eventually[0:767](sideways > v)) and (turn > 0)
    RIGHT    = MOVE and ((nod < n) or eventually[0:767](sideways > v)) and (turn < 0)

``delta``, ``deflect``, ``sideways`` are the live monitor's per-sample signals of the same window. ``nod`` and
``turn`` are the live monitor's end-of-window values (the whole window's nod evidence and direction), held
constant over the window so that they are read at t = 0. Only the loudness test differs from the live
monitor, whose ``once(background >= b_live)`` exists only when history does and otherwise falls back to
``once(contrast >= g)``. The four specifications partition the space, so exactly one is satisfied.

THRESHOLDS

    b = 5.75   offline background ratio: selected on sub_01 sessions 1-20 only (default since 2026-10-07;
               blocked 3-fold CV picked 4.90 / 4.95 / 5.85). The curve is flat: 4.95 gives 86.23 on
               sessions 1-20 against 86.24 at 5.75.
    d, q, n, v the live monitor's values (0.16, 2.8, -0.6, 7.5), unchanged

MEASURED  (extracted_npz_dataset_v2, four-class balanced accuracy)

  CURRENT DEFAULT (thresholds selected on sub_01 sessions 1-20)
                              sub_01 1-20 (design)   sub_01 21-36 (test)   sub_11 (test)
    live monitor, history           86.13                 83.31               78.15
    offline monitor                 86.24                 81.52               78.86
  Offline removes false alarms (162 -> 67 and 97 -> 0) but misses more nods on sub_01 21-36 (DOWN 75.7%
  -> 66.9%), so it is BELOW the live monitor there.

  PREVIOUS DEFAULT (selected on sessions 1-30, offline b 4.95; record: experiments/offline_monitor_2026_09_25)
                              sub_01 1-30 (design)   sub_01 31-36 (test)   sub_11 (test)
    live monitor, history           85.01                 69.62               78.67
    offline monitor                 86.73                 71.80               79.37
  That design used sessions 1-30 only (CV gain +1.58, 18 sessions better / 9 worse); sub_11 false alarms
  fell from 23 to 0. This is OFFLINE accuracy: it uses samples after the
window, which a live system does not have. Use it for analysing recordings, not for live control.

    from monitoring.offline_monitor import OfflineHeadMovementMonitor, load_recording
    monitor = OfflineHeadMovementMonitor()
    recording = load_recording(csv_path)          # (8, n) microvolts, one recorded run
    monitor.classify(recording, start)            # the 3-s window recording[:, start:start+768]
    monitor.robustness(recording, start)          # RTAMT robustness of each specification at t = 0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from scipy.signal import lfilter, lfilter_zi

from .rtamt_monitors import CLASSES, import_rtamt, satisfied_class
from .rtamt_signals import (
    CHANNELS,
    FS,
    WINDOW_SAMPLES,
    InvalidWindow,
    RtamtConfig,
    _BAND_DETECT,
    check_window,
    compute_signals,
)

LAST = WINDOW_SAMPLES - 1
OFFLINE_SIGNALS = ("background", "delta", "deflect", "nod", "sideways", "turn")


@dataclass(frozen=True)
class OfflineConfig:
    """Thresholds of the offline specifications. Only ``background_ratio`` is new; the rest are the live
    monitor's values (``RtamtConfig``) and are carried over unchanged."""

    background_ratio: float = 5.75          # b, selected on sub_01 sessions 1-20 (4.95 until 2026-10-07)
    block_seconds: float = 0.5              # energy block, as in the live gate
    reference_step: int = 8                 # every 8th block enters the median, as in the live gate
    min_reference_blocks: int = 16          # fewer -> the whole recording's median is used
    live: RtamtConfig = RtamtConfig()       # delta d, deflect q, nod n, sideways v

    def __post_init__(self):
        if self.background_ratio <= 0:
            raise ValueError("background_ratio must be positive")


def load_recording(csv_path) -> np.ndarray:
    """One recorded run from the dataset's CSV files, as (8, n) microvolts (the CSVs store volts)."""
    import csv
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        cols = [header.index(c) for c in CHANNELS]
        rows = [[float(r[c]) for c in cols] for r in reader if r]
    return np.asarray(rows, dtype=float).T * 1e6


def recording_energy(recording, config: OfflineConfig | None = None) -> np.ndarray:
    """(8, n) per-channel 0.5-3 Hz energy of the 0.5 s ending at each sample; NaN before the first full block."""
    cfg = config or OfflineConfig()
    x = np.asarray(recording, dtype=float)
    zi = lfilter_zi(*_BAND_DETECT)
    band, _ = lfilter(*_BAND_DETECT, x, axis=-1, zi=zi[None, :] * x[:, :1])
    block = int(round(cfg.block_seconds * FS))
    cumulative = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(band ** 2, axis=1)], axis=1)
    energy = np.full(x.shape, np.nan)
    energy[:, block - 1:] = (cumulative[:, block:] - cumulative[:, :-block]) / block
    return energy


def offline_background(recording, start: int, config: OfflineConfig | None = None,
                       energy: np.ndarray | None = None) -> np.ndarray:
    """``background`` over the window recording[:, start:start+768] (768 values, 0 before the first full block)."""
    cfg = config or OfflineConfig()
    rec = np.asarray(recording, dtype=float)
    if rec.ndim != 2 or rec.shape[0] != len(CHANNELS):
        raise InvalidWindow(f"recording must have shape ({len(CHANNELS)}, n_samples), got {rec.shape}")
    if start < 0 or start + WINDOW_SAMPLES > rec.shape[1]:
        raise InvalidWindow(f"window [{start}, {start + WINDOW_SAMPLES}) is outside the recording "
                            f"of {rec.shape[1]} samples")
    e = recording_energy(rec, cfg) if energy is None else energy
    block = int(round(cfg.block_seconds * FS))
    complete = np.isfinite(e[0])
    outside = complete.copy()
    outside[max(0, start - (block - 1)):start + WINDOW_SAMPLES] = False
    chosen = np.flatnonzero(outside)[::cfg.reference_step]
    if len(chosen) < cfg.min_reference_blocks:
        chosen = np.flatnonzero(complete)[::cfg.reference_step]
    reference = np.median(e[:, chosen], axis=1) + 1e-12
    window = e[:, start:start + WINDOW_SAMPLES] / reference[:, None]
    values = np.mean(window, axis=0)
    return np.where(np.isfinite(values), values, 0.0)


def offline_formulas(config: OfflineConfig | None = None) -> Dict[str, str]:
    """The four offline STL specifications (future-time operators, read at t = 0)."""
    cfg = config or OfflineConfig()
    b = cfg.background_ratio
    d, q = cfg.live.delta_range_min, cfg.live.quiet_min
    n, v = cfg.live.down_threshold, cfg.live.down_veto_prominence
    w = f"[0:{LAST}]"
    move = (f"(eventually{w}(background >= {b!r})) and (eventually{w}(delta >= {d!r})) "
            f"and (eventually{w}(deflect >= {q!r}))")
    turned = f"((nod < {n!r}) or (eventually{w}(sideways > {v!r})))"
    return {
        "STRAIGHT": (f"(always{w}(background < {b!r})) or (always{w}(delta < {d!r})) "
                     f"or (always{w}(deflect < {q!r}))"),
        "DOWN": f"({move}) and (nod >= {n!r}) and (always{w}(sideways <= {v!r}))",
        "LEFT": f"({move}) and {turned} and (turn > 0)",
        "RIGHT": f"({move}) and {turned} and (turn < 0)",
    }


class OfflineHeadMovementMonitor:
    """The four offline RTAMT specifications over windows of a recorded run."""

    def __init__(self, config: OfflineConfig | None = None) -> None:
        self.config = config or OfflineConfig()
        self.formulas = offline_formulas(self.config)
        self._specs = None

    def _parsed(self):
        if self._specs is None:
            rtamt = import_rtamt()
            specs = {}
            for c in CLASSES:
                spec = rtamt.StlDiscreteTimeOfflineSpecification()
                for name in OFFLINE_SIGNALS:
                    spec.declare_var(name, "float")
                spec.spec = self.formulas[c]
                spec.parse()
                specs[c] = spec
            self._specs = specs
        return self._specs

    def signals(self, recording, start: int, energy: np.ndarray | None = None) -> Dict[str, np.ndarray]:
        """The six offline signals of the window recording[:, start:start+768], 768 values each."""
        rec = np.asarray(recording, dtype=float)
        if rec.ndim != 2 or start < 0 or start + WINDOW_SAMPLES > rec.shape[1]:
            raise InvalidWindow(f"window [{start}, {start + WINDOW_SAMPLES}) is outside the recording")
        window = check_window(rec[:, start:start + WINDOW_SAMPLES])
        window = window - window.mean(axis=1, keepdims=True)
        live = compute_signals(window, self.config.live)
        return {
            "background": offline_background(rec, start, self.config, energy),
            "delta": live["delta"], "deflect": live["deflect"], "sideways": live["sideways"],
            "nod": np.full(WINDOW_SAMPLES, live["nod"][-1]),
            "turn": np.full(WINDOW_SAMPLES, live["turn"][-1]),
        }

    def robustness_from_signals(self, signals: Dict[str, np.ndarray]) -> Dict[str, float]:
        data = {"time": list(range(WINDOW_SAMPLES))}
        data.update({k: [float(x) for x in signals[k]] for k in OFFLINE_SIGNALS})
        return {c: float(spec.evaluate(data)[0][1]) for c, spec in self._parsed().items()}

    def robustness(self, recording, start: int) -> Dict[str, float]:
        """RTAMT robustness of each specification at t = 0 (over the whole window)."""
        return self.robustness_from_signals(self.signals(recording, start))

    def explain(self, recording, start: int, energy: np.ndarray | None = None) -> dict:
        try:
            s = self.signals(recording, start, energy)
        except InvalidWindow as exc:
            return {"label": "INVALID", "reason": str(exc)}
        rho = self.robustness_from_signals(s)
        return {"label": satisfied_class(rho), "robustness": rho, "formulas": self.formulas}

    def classify(self, recording, start: int) -> str:
        """LEFT, RIGHT, DOWN, STRAIGHT -- or INVALID -- for the window recording[:, start:start+768]."""
        return self.explain(recording, start)["label"]

    def classify_recording(self, recording, starts: Iterable[int]) -> List[str]:
        """Verdicts for many windows of one recording (the recording's energy is computed once)."""
        rec = np.asarray(recording, dtype=float)
        energy = recording_energy(rec, self.config)
        return [self.explain(rec, int(s), energy)["label"] for s in starts]


def dataset_recording_path(dataset_root, session: str, task: str, run: int) -> Path:
    """The CSV recording of one run, e.g. session 'sub_01/ses-05', task 'right', run 6."""
    subject, ses = session.split("/")
    name = f"{subject.replace('_', '-')}_{ses}_task-{task}_run-{int(run):02d}_eeg.csv"
    return Path(dataset_root) / subject / ses / "ica_eeg" / name


__all__ = ["OfflineConfig", "OfflineHeadMovementMonitor", "offline_formulas", "offline_background",
           "recording_energy", "load_recording", "dataset_recording_path", "OFFLINE_SIGNALS"]
