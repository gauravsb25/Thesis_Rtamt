"""Row-by-row head-movement monitor: a verdict at every incoming EEG row.

:class:`HeadMovementMonitor` judges a COMPLETED 3-second window: several of its signals are
normalised with statistics of the whole window (each channel's median and spread, the quietest
0.5 s, the lowest slow-wave slice), so it can only answer after row 768. This module reads the same
data one row at a time, as it arrives from the headset:

  * every filter is causal and keeps its state between rows (exactly ``lfilter`` over the window);
  * every signal is a PREFIX statistic -- the monitor's measurement computed on rows 0..t only
    (loudest 0.5 s so far over quietest 0.5 s so far, largest excursion so far over its spread so
    far, the direction read in the loudest stretch so far, ...);
  * RTAMT's online monitor evaluates the four class specifications on those signals at every row,
    so a label and four robustness values exist after every row (an ANYTIME verdict).

At row 768 the prefix is the whole window, so the verdict and all four robustness values equal
:class:`HeadMovementMonitor` on the same window exactly.

EARLY COMMITMENT (optional, off by default). With ``commit_seconds=k`` the monitor also evaluates,
for LEFT, RIGHT and DOWN, the past-time STL specification

    historically[0:K-1]( <that class's specification> )        K = k seconds of rows

and commits to the first class it holds for; STRAIGHT is only given at the end of the window.
Measured on extracted_npz_dataset_v2 WITH the stream-background gate (4-class balanced accuracy;
median time from the movement's start in the window to the decision, over correctly classified movement
windows; a verdict every 0.125 s). Re-measured on 2026-09-19: experiments/commit_latency_2026_09_19.
This table was measured with the PREVIOUS thresholds (selected on sessions 1-30); it has not been
re-measured with the 1-20 default of 2026-10-07.

    decision                  sub_01 1-30        sub_01 31-36           sub_11
    wait for row 768        85.01%  2.22 s     69.62%  2.20 s      78.67%  2.31 s
    commit_seconds = 1.5    84.95%  2.03 s     69.60%  2.06 s      78.67%  2.15 s
    commit_seconds = 1.0    83.92%  1.79 s     69.93%  1.85 s      79.57%  1.76 s
    commit_seconds = 0.75   80.89%  1.56 s     69.98%  1.67 s      78.97%  1.50 s
    commit_seconds = 0.5    74.85%  1.24 s     67.53%  1.31 s      77.53%  1.23 s

commit_seconds = 1.0 decides about 0.45 s sooner and is slightly BETTER on both held-out sets
(+0.31 and +0.90), at a cost of 1.1 points on the design sessions. Shorter commitments trade accuracy
for speed. The choice is left to the application.

Only the default design (contrast gate, largest-peak direction) is supported row by row.

STREAM BACKGROUND. The monitor keeps the last settle + background seconds (5 s) of the rows it received
before the current window. When they are enough (``background_valid``), the gate is the
stream-background gate adopted on 2026-09-17:
- at the window start each channel's background level is computed from those rows;
- every new row updates that channel's 0.5 s energy;
- ``background`` is the largest channel-mean ratio so far.
At row 768 the verdict and robustness equal ``HeadMovementMonitor().classify(window, history)`` with the
same rows as history. The first window of a stream, or a window after ``clear()``, uses the contrast
gate. Seed a history explicitly with ``RowStreamMonitor(history=samples)``.

CENSORED BACKGROUND (``RtamtConfig(background_censor=True)``; tested 2026-10-07, OFF by default). The loud
0.5-s blocks of a window whose final label was LEFT, RIGHT or DOWN (or that committed early to one) are
marked and kept out of the background; up to ``background_memory_seconds`` of past rows are kept to find
clean stream behind them. When nothing in the last 5 s is marked, the background is exactly the uncensored
one. Trade-off measured on real runs: see ``RtamtConfig.background_censor``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import lfilter, lfilter_zi  # noqa: F401  (lfilter_zi kept for users extending this)

from .rtamt_monitors import CLASSES, import_rtamt, satisfied_class
from .rtamt_signals import (
    _BAND_CLASS,
    _BAND_DETECT,
    _BAND_DIRECTION,
    _BAND_PROMINENCE,
    _IDX,
    CHANNELS,
    FS,
    WINDOW_SAMPLES,
    InvalidWindow,
    RtamtConfig,
    background_positions,
    compute_signals,
    loud_samples,
)

ROW_SIGNALS = ("contrast", "delta", "deflect", "nod", "sideways", "sideways_fp1fp2", "sideways_fp2t3", "turn",
               "background", "background_valid")


def row_formulas(config: RtamtConfig | None = None) -> Dict[str, str]:
    """The four class specifications over prefix signals (read at the current row)."""
    cfg = config or RtamtConfig()
    g, d, q = cfg.gate_ratio, cfg.delta_range_min, cfg.quiet_min
    n, v = cfg.down_threshold, cfg.down_veto_prominence
    vetoes = [("sideways", v)]
    if cfg.down_veto_fp1fp2 is not None:
        vetoes.append(("sideways_fp1fp2", cfg.down_veto_fp1fp2))
    if cfg.down_veto_fp2t3 is not None:
        vetoes.append(("sideways_fp2t3", cfg.down_veto_fp2t3))
    loud, quiet = f"(contrast >= {g!r})", f"(contrast < {g!r})"
    if cfg.background_ratio is not None:
        b = cfg.background_ratio
        loud = f"((background >= {b!r}) or ((background_valid < 0.5) and (contrast >= {g!r})))"
        quiet = f"((background < {b!r}) and ((background_valid >= 0.5) or (contrast < {g!r})))"
    move = f"{loud} and (delta >= {d!r}) and (deflect >= {q!r})"
    no_swing = " and ".join(f"({s} <= {t!r})" for s, t in vetoes)
    swing = " or ".join(f"({s} > {t!r})" for s, t in vetoes)
    return {
        "STRAIGHT": f"{quiet} or (delta < {d!r}) or (deflect < {q!r})",
        "DOWN": f"({move}) and (nod >= {n!r}) and {no_swing}",
        "LEFT": f"({move}) and ((nod < {n!r}) or {swing}) and (turn > 0)",
        "RIGHT": f"({move}) and ((nod < {n!r}) or {swing}) and (turn < 0)",
    }


@dataclass
class RowVerdict:
    row: int                                  # number of rows received in this window (1..768)
    label: str                                # anytime label from rows 0..row-1
    robustness: Dict[str, float]
    committed: Optional[str] = None           # the committed label, once there is one
    committed_at_row: Optional[int] = None
    window_complete: bool = False             # True on row 768: label equals HeadMovementMonitor
    signals: Dict[str, float] = field(default_factory=dict)


class _Filter:
    """Causal IIR filter over (channels,) rows, identical to lfilter over the whole window."""

    def __init__(self, ba):
        self.b, self.a = ba
        self.zi = np.zeros((len(CHANNELS), max(len(self.a), len(self.b)) - 1))

    def step(self, column):
        y, self.zi = lfilter(self.b, self.a, column[:, None], axis=-1, zi=self.zi)
        return y[:, 0]


class RowStreamMonitor:
    """Push one EEG row (8 microvolt values) at a time; get a verdict every ``decide_every`` rows.

    ``commit_seconds``: None (default) decides at row 768 only, exactly like HeadMovementMonitor;
    a number enables early commitment to LEFT/RIGHT/DOWN after that many seconds of the class
    specification holding (an RTAMT ``historically`` specification).
    After row 768 the window is complete; the next row starts a new window.
    """

    def __init__(self, config: RtamtConfig | None = None, *, commit_seconds: float | None = None,
                 decide_every: int = 1, history=None) -> None:
        self.config = config or RtamtConfig()
        cfg = self.config
        self._short = (0 if cfg.background_ratio is None else
                       int(round((cfg.background_settle_seconds + cfg.background_seconds) * FS)))
        self._censor = bool(cfg.background_censor) and self._short > 0
        self._keep = (max(self._short, int(round((cfg.background_settle_seconds
                                                   + cfg.background_memory_seconds) * FS)))
                      if self._censor else self._short)
        self._past = np.zeros((len(CHANNELS), 0))
        if history is not None and self._keep:
            past = np.asarray(history, dtype=float)
            if past.ndim != 2 or past.shape[0] != len(CHANNELS) or not np.isfinite(past).all():
                raise InvalidWindow("history must be finite with shape (8, n_samples)")
            self._past = past[:, -self._keep:].copy()
        self._past_usable = np.ones(self._past.shape[1], dtype=bool)   # False: judged a movement
        if self.config.gate_statistic != "contrast" or self.config.direction_rule != "peak":
            raise NotImplementedError("row-by-row monitoring supports the default design only")
        if int(decide_every) < 1:
            raise ValueError("decide_every must be at least 1 row")
        self.decide_every = int(decide_every)
        self.commit_seconds = commit_seconds
        self.formulas = row_formulas(self.config)
        self.reset()

    def clear(self) -> None:
        """Forget the stream history too (a new session or a break in the recording)."""
        self._past = np.zeros((len(CHANNELS), 0))
        self._past_usable = np.ones(0, dtype=bool)
        self.reset()

    def reset(self) -> None:
        """Start a new 3-second window (the stream history before it is kept)."""
        n = WINDOW_SAMPLES
        self.rows = 0
        self._start_background()
        self.raw = np.zeros((len(CHANNELS), n))
        self.det = np.zeros((len(CHANNELS), n))
        self.cls = np.zeros((len(CHANNELS), n))
        self.pro = np.zeros((len(CHANNELS), n))
        self.dir = np.zeros((len(CHANNELS), n))
        self._filters = [_Filter(_BAND_DETECT), _Filter(_BAND_CLASS), _Filter(_BAND_PROMINENCE),
                         _Filter(_BAND_DIRECTION)]
        self._shares: List[float] = []
        self._decisions = 0
        self.committed = None
        self.committed_at = None
        self.last = None
        rtamt = import_rtamt()
        self._specs = {}
        for c, formula in self.formulas.items():
            self._specs[c] = self._online_spec(rtamt, formula)
        self._commit_specs = {}
        if self.commit_seconds is not None:
            steps = max(int(round(self.commit_seconds * FS / self.decide_every)), 1)
            for c in ("LEFT", "RIGHT", "DOWN"):
                self._commit_specs[c] = self._online_spec(rtamt, f"historically[0:{steps - 1}]({self.formulas[c]})")

    @staticmethod
    def _online_spec(rtamt, formula):
        spec = rtamt.StlDiscreteTimeOnlineSpecification()
        for name in ROW_SIGNALS:
            if name in formula:
                spec.declare_var(name, "float")
        spec.spec = formula
        spec.parse()
        spec._used = tuple(s for s in ROW_SIGNALS if s in formula)
        return spec

    def _start_background(self) -> None:
        """Background level per channel from the rows kept before this window (stream_background rules)."""
        cfg = self.config
        self._bg_valid = False
        self._bg_max = 0.0
        past, usable = self._past, self._past_usable
        self._bg_inputs = (None, None)            # (history, mask) this window is judged with
        if cfg.background_ratio is None or past.shape[1] == 0:
            return
        if usable[-self._short:].all():
            past, usable = past[:, -self._short:], None
        self._bg_inputs = (past, usable)
        start = past.shape[1]
        positions = background_positions(start, cfg, usable)
        if positions is None:
            return
        block = max(int(cfg.gate_seconds * FS), 1)
        b, a = _BAND_DETECT
        zi = np.zeros((len(CHANNELS), max(len(a), len(b)) - 1))
        band, zi = lfilter(b, a, past, axis=-1, zi=zi)
        cumulative = np.concatenate([np.zeros((len(CHANNELS), 1)), np.cumsum(band ** 2, axis=1)], axis=1)
        energy = np.zeros_like(band)
        energy[:, block - 1:] = (cumulative[:, block:] - cumulative[:, :-block]) / block
        self._bg_reference = np.median(energy[:, positions], axis=1) + 1e-12
        self._bg_filter = _Filter(_BAND_DETECT)
        self._bg_filter.zi = zi
        self._bg_ring = (band[:, -block:] ** 2).copy() if start >= block else None
        self._bg_block = block
        self._bg_valid = True

    def _update_background(self, column) -> None:
        if not self._bg_valid:
            return
        y = self._bg_filter.step(column)
        self._bg_ring = np.concatenate([self._bg_ring[:, 1:], (y ** 2)[:, None]], axis=1)
        value = float(np.mean(self._bg_ring.mean(axis=1) / self._bg_reference))
        self._bg_max = max(self._bg_max, value)

    def prefix_signals(self) -> Dict[str, float]:
        """The monitor's measurements computed on the rows received so far."""
        L = self.rows
        det, cls, pro, dr = self.det[:, :L], self.cls[:, :L], self.pro[:, :L], self.dir[:, :L]
        spread = 1.4826 * np.median(np.abs(det - np.median(det, axis=1, keepdims=True)), axis=1, keepdims=True) + 1e-12
        energy = np.mean((det / spread) ** 2, axis=0)
        cum = np.concatenate([[0.0], np.cumsum(energy)])

        def blocks(w):
            return cum[w:] - cum[:-w]

        gate = int(self.config.gate_seconds * FS)
        if L >= gate:
            b = blocks(gate)
            contrast = float(b.max() / (b.min() + 1e-18))
        else:
            contrast = 1.0
        delta = float(max(self._shares) - min(self._shares)) if self._shares else 0.0
        c = cls - np.median(cls, axis=1, keepdims=True)
        sp = 1.4826 * np.median(np.abs(c), axis=1) + 1e-12
        deflect = float(np.mean(np.abs(c).max(axis=1) / sp))
        sq = c ** 2
        balance = float(np.log((sq[[_IDX["Fz"], _IDX["Cz"]]].sum() + 1e-18) / (sq[[_IDX["T3"], _IDX["T4"]]].sum() + 1e-18)))
        w1 = min(int(self.config.symmetry_seconds * FS), L)
        s1 = int(np.argmax(blocks(w1)))
        f3, f4 = c[_IDX["F3"], s1:s1 + w1], c[_IDX["F4"], s1:s1 + w1]
        symmetry = float(np.log((np.mean((f3 + f4) ** 2) + 1e-18) / (np.mean((f3 - f4) ** 2) + 1e-18)))

        def excursion(first, second):
            vv = pro[_IDX[first]] - pro[_IDX[second]]
            vv = vv - np.median(vv)
            return float(np.abs(vv).max() / (1.4826 * np.median(np.abs(vv)) + 1e-12))

        a, bpair = self.config.lateral_peak_pair
        w15 = min(int(self.config.direction_seconds * FS), L)
        s15 = int(np.argmax(blocks(w15)))
        u = (dr[_IDX[a]] - dr[_IDX[bpair]])[s15:s15 + w15]
        u = u - np.median(u)
        peak = float(u[int(np.argmax(np.abs(u)))])
        turn = -peak / (1.4826 * float(np.median(np.abs(u))) + 1e-12)
        turn = turn if turn != 0.0 else 1e-12
        return {"contrast": contrast, "delta": delta, "deflect": deflect, "nod": balance + symmetry,
                "sideways": excursion(a, bpair), "sideways_fp1fp2": excursion("Fp1", "Fp2"),
                "sideways_fp2t3": excursion("Fp2", "T3"), "turn": turn,
                "background": self._bg_max if self._bg_valid else 0.0,
                "background_valid": 1.0 if self._bg_valid else 0.0}

    def push(self, row) -> Optional[RowVerdict]:
        """Consume one (8,) row in microvolts. Returns a RowVerdict when a decision is due."""
        column = np.asarray(row, dtype=float).reshape(-1)
        if column.shape[0] != len(CHANNELS):
            raise ValueError(f"expected {len(CHANNELS)} channel values, got {column.shape[0]}")
        if not np.isfinite(column).all():
            raise InvalidWindow("EEG samples must be finite")
        if self.rows >= WINDOW_SAMPLES:
            if self._keep:
                moved = self._censor and self.last is not None and (
                    self.last.label != "STRAIGHT" or self.committed not in (None, "STRAIGHT"))
                flags = np.ones(WINDOW_SAMPLES, dtype=bool)
                if moved:                         # censored background: mark the loud blocks of this movement
                    history, mask = self._bg_inputs
                    flags = ~loud_samples(compute_signals(self.raw, self.config, history, mask), self.config)
                self._past = np.concatenate([self._past, self.raw], axis=1)[:, -self._keep:]
                self._past_usable = np.concatenate([self._past_usable, flags])[-self._keep:]
            self.reset()
        t = self.rows
        self.raw[:, t] = column
        self._update_background(column)
        for store, filt in zip((self.det, self.cls, self.pro, self.dir), self._filters):
            store[:, t] = filt.step(column)
        self.rows += 1
        width, step = int(self.config.delta_slice_seconds * FS), max(int(self.config.delta_slice_seconds * FS) // 2, 1)
        if self.rows >= width and (self.rows - width) % step == 0:
            s = self.rows - width
            seg = self.raw[:, s:s + width] - self.raw[:, s:s + width].mean(axis=1, keepdims=True)
            power = np.abs(np.fft.rfft(seg * np.hanning(width), axis=1)) ** 2
            freq = np.fft.rfftfreq(width, 1.0 / FS)
            share = power[:, (freq >= 1.0) & (freq < 4.0)].sum(axis=1) / (power[:, (freq >= 0.5) & (freq <= 45.0)].sum(axis=1) + 1e-18)
            self._shares.append(float(np.mean(share)))
        complete = self.rows == WINDOW_SAMPLES
        if not complete and self.rows % self.decide_every:
            return None
        if complete and np.any(np.ptp(self.raw, axis=1) == 0):
            raise InvalidWindow("flat EEG channel in this window")
        sig = self.prefix_signals()
        tick = self._decisions
        self._decisions += 1
        rho = {c: float(spec.update(tick, [(k, sig[k]) for k in spec._used])) for c, spec in self._specs.items()}
        label = satisfied_class(rho)
        if self._commit_specs and self.committed is None:
            held = {c: float(spec.update(tick, [(k, sig[k]) for k in spec._used])) for c, spec in self._commit_specs.items()}
            steps = max(int(round(self.commit_seconds * FS / self.decide_every)), 1)
            ok = [c for c in ("DOWN", "LEFT", "RIGHT") if held[c] >= 0] if tick + 1 >= steps else []
            if ok:
                self.committed, self.committed_at = ok[0], self.rows
        if complete and self.committed is None:
            self.committed, self.committed_at = label, self.rows
        self.last = RowVerdict(row=self.rows, label=label, robustness=rho, committed=self.committed,
                               committed_at_row=self.committed_at, window_complete=complete, signals=sig)
        return self.last

    def push_block(self, x) -> List[RowVerdict]:
        """Feed many rows ((8, n) microvolts); returns every verdict produced."""
        block = np.asarray(x, dtype=float)
        if block.ndim != 2 or block.shape[0] != len(CHANNELS):
            raise ValueError(f"expected shape ({len(CHANNELS)}, n_samples), got {block.shape}")
        out = []
        for k in range(block.shape[1]):
            verdict = self.push(block[:, k])
            if verdict is not None:
                out.append(verdict)
        return out


__all__ = ["RowStreamMonitor", "RowVerdict", "row_formulas", "ROW_SIGNALS"]
