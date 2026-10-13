"""A timestamped live-stream session around the RTAMT head-movement monitor.

Feed acquisition rows (eight microvolt values) with their timestamps. The session checks the
timestamps, restarts cleanly after a gap, runs :class:`monitoring.rtamt_monitors.StreamMonitor`,
and can confirm a label only after it repeats -- itself an RTAMT specification,
``historically[0:k-1](evidence > 0)``, in :mod:`monitoring.rtamt_persistence`.

Confirmation is an optional output filter. It is not an accuracy claim and never turns an
unconfirmed decision into STRAIGHT. Defaults: one verdict per non-overlapping 3-second window,
confirmations=1 (every verdict stands).
"""
from time import perf_counter

import numpy as np

from .rtamt_monitors import StreamMonitor
from .rtamt_signals import FS, WINDOW_SAMPLES


class LiveStreamSession:
    def __init__(self, config=None, *, decide_every=WINDOW_SAMPLES, confirmations=1,
                 max_gap_seconds=2 / FS, monitor=None, temporal_backend="rtamt"):
        if not isinstance(confirmations, int) or confirmations < 1:
            raise ValueError("confirmations must be a positive integer")
        if not np.isfinite(max_gap_seconds) or max_gap_seconds < 1 / FS:
            raise ValueError("max_gap_seconds must be finite and at least 1/256")
        if temporal_backend not in ("python", "rtamt"):
            raise ValueError("temporal_backend must be python or rtamt")
        self.stream = StreamMonitor(config, decide_every=decide_every, monitor=monitor)
        self.confirmations = confirmations
        self.max_gap = max_gap_seconds
        self.temporal_backend = temporal_backend
        self.reset()

    def reset(self):
        self.stream.reset()
        self.last_timestamp = None
        self.pending = None
        self.count = 0
        self.confirmed_label = None
        self.temporal = None
        if self.temporal_backend == "rtamt":
            from .rtamt_persistence import RtamtDecisionPersistence
            self.temporal = RtamtDecisionPersistence(self.confirmations)

    def push(self, row, timestamp):
        row = np.asarray(row, dtype=float)
        if row.shape != (8,) or not np.isfinite(row).all() or not np.isfinite(timestamp):
            self.reset()
            raise ValueError("finite timestamp and eight finite EEG values required; state reset")
        gap_reset = False
        if self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
            if dt <= 0:
                self.reset()
                raise ValueError("timestamps must increase; state reset")
            if dt > self.max_gap:
                self.reset()
                gap_reset = True
        self.last_timestamp = float(timestamp)
        start = perf_counter()
        label = self.stream.push(row)
        compute_ms = 1000 * (perf_counter() - start)
        if label is None:
            status = "GAP_RESET" if gap_reset else ("WARMUP" if not self.stream.ready else "WAITING")
            return {"status": status, "window_label": None,
                    "confirmed_label": self.confirmed_label,
                    "timestamp": float(timestamp), "compute_ms": compute_ms}
        if label == "INVALID":
            self.pending, self.count, self.confirmed_label = None, 0, None
        else:
            self.count = self.count + 1 if label == self.pending else 1
            self.pending = label
            self.confirmed_label = label if self.count >= self.confirmations else None
        robustness = None
        if self.temporal is not None:
            self.confirmed_label, robustness = self.temporal.update(label)
        compute_ms = 1000 * (perf_counter() - start)
        budget = 1000 * self.stream.decide_every / FS
        return {"status": "INVALID" if label == "INVALID"
                else ("CONFIRMED" if self.confirmed_label else "PENDING"),
                "window_label": label, "confirmed_label": self.confirmed_label,
                "timestamp": float(timestamp), "samples_seen": self.stream.samples_seen,
                "compute_ms": compute_ms, "temporal_backend": self.temporal_backend,
                "persistence_robustness": robustness, "decision_budget_ms": budget,
                "over_decision_budget": compute_ms > budget}
