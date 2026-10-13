"""Four RTAMT monitors, one per head movement.

Each head movement is a Signal Temporal Logic specification, evaluated by RTAMT
(https://github.com/nickovic/rtamt, v0.4.10, bundled under ``.runtime/rtamt``) over the six
signals of :mod:`monitoring.rtamt_signals`. RTAMT returns a quantitative ROBUSTNESS per class:

    robustness > 0   the specification is satisfied, and the value says by how much
    robustness < 0   it is violated, and the value says by how far

THE FOUR SPECIFICATIONS  (sample index t = 0..767, one 3-second window at 256 Hz)

    LOUD     = once[0:767](background >= b)
               or ( historically[0:767](background_valid < 0.5) and once[0:767](contrast >= g) )

    MOVE     = LOUD  and  once[0:767](delta >= d)  and  once[0:767](deflect >= q)

    STRAIGHT = ( historically[0:767](background < b)
                 and ( once[0:767](background_valid >= 0.5) or historically[0:767](contrast < g) ) )
               or  historically[0:767](delta < d)  or  historically[0:767](deflect < q)
                                                                        -- exactly  not MOVE

LOUD is the stream-background gate adopted on 2026-09-17: the window's loudest 0.5 s must be b = 8.0
(6.61 until 2026-10-07) times the stream's own background over the 4 s before the window. When less than 1 s of earlier
stream exists (the first window, or a window called without ``history``) ``background_valid`` is 0
and LOUD is the original contrast gate, so the monitor is then exactly the previous monitor.

    DOWN     = MOVE  and  (nod >= n)  and  historically[0:767](sideways <= v)

    LEFT     = MOVE  and  ((nod < n)  or  once[0:767](sideways > v))  and  (turn > 0)
    RIGHT    = MOVE  and  ((nod < n)  or  once[0:767](sideways > v))  and  (turn < 0)

In words:
    STRAIGHT  throughout the window, at least one activity measure stayed low
    DOWN      the window moved, the midline dominated with common-mode symmetry, and there was
              never a large sideways excursion
    LEFT      the window moved, it was not a nod, and the saccade went left
    RIGHT     the window moved, it was not a nod, and the saccade went right

Only PAST-time operators are used (``once``, ``historically``). RTAMT's online monitor does not
implement bounded future operators such as ``eventually[a:b]`` -- a monitor cannot look ahead on
a live stream -- so each specification is stated about the window so far and read at its last
sample. The four specifications partition the space, so exactly one is satisfied and the label
is that class.

The thresholds b, g, d, q, n, v are the six dimensionless values of :class:`RtamtConfig`, selected
on sub_01 sessions 1-20 (the default since 2026-10-07; before that sessions 1-30). Nothing is
compared with a microvolt level.

TWO OPTIONS TESTED ON 2026-09-16, NOT ADOPTED  (``RtamtConfig(gate_statistic="cusum",
direction_rule="since")``; designed on sub_01 sessions 1-30 only)

    MOVE  uses  once[0:767](cusum >= c)   -- Page's CUSUM of the log-energy above the quietest
                                             0.5 s, allowance 2 nats: energy must rise AND stay up
    LEFT  uses  (not (lateral >= h)) since (lateral <= -h)
    RIGHT uses  (not (lateral <= -h)) since (lateral >= h)
                                          -- the LAST F4-T3 excursion reaching h = 0.95 of the
                                             largest decides, so a near tie goes to the later lobe

    4-class balanced accuracy        sessions 1-30   sub_01 31-36   sub_11
    default                             82.30%          66.28%       76.40%
    CUSUM gate + SINCE direction        83.50%          65.05%       75.48%

The gain on the design sessions did not transfer to either held-out set, so the defaults stay.

MEASURED ACCURACY  (extracted_npz_dataset_v2 = extracted_npz_dataset without the windows spliced
across two runs; four-class balanced accuracy, chance 25%; ``python project_code.py evaluate``)

CURRENT DEFAULT (since 2026-10-07): thresholds selected on sub_01 sessions 1-20
(b 8.0, g 5.0, d 0.16, q 2.8, n -0.6, v 7.5); tests = sub_01 sessions 21-36 and sub_11

  with the stream history before each window (rebuilt from the dataset's own windows)
    class        sub_01 ses 1-20 (design)   sub_01 ses 21-36 (test)   sub_11 (second subject)
    LEFT          87.0%  (556/639)           78.0% (341/437)            59.4%  (111/187)
    RIGHT         78.6%  (488/621)           82.7% (434/525)            76.9%  (170/221)
    DOWN          82.4%  (399/484)           75.7% (293/387)            95.4%  (272/285)
    STRAIGHT      96.5% (11184/11592)        96.9% (4982/5144)          80.9%  (410/507)
    4-class       86.1%                      83.3%                      78.2%
    false alarms  408                        162                        97

  every window alone (no history: the contrast gate, g = 5.0)
    STRAIGHT      69.8%  (8088/11592)        73.1% (3762/5144)          66.3%  (336/507)
    4-class       79.5%                      78.7%                      74.5%
  g = 5.0 is the lowest value the search allows. With history it rarely decides; without history
  (the first seconds of a stream) about 30% of rest windows become false alarms.

PREVIOUS DEFAULT (until 2026-10-07): thresholds selected on sub_01 sessions 1-30
(b 6.61, g 7.81, d 0.208, q 2.72, n -0.661, v 7.54); on sub_01 21-36: 81.76% with history, 79.57%
alone, 66 false alarms; sub_11 78.67% / 77.00%, 23 false alarms. The tables below are that version.

  with the stream history before each window (rebuilt from the dataset's own windows)
    class        sub_01 ses 1-30 (design)   sub_01 ses 31-36 (test)   sub_11 (second subject)
    LEFT          83.9%  (805/959)           59.0%  (69/117)            56.1%  (105/187)
    RIGHT         80.0%  (770/963)           64.5% (118/183)            74.7%  (165/221)
    DOWN          78.2%  (585/748)           57.7%  (71/123)            88.4%  (252/285)
    STRAIGHT      97.9% (15559/15890)        97.3% (823/846)            95.5%  (484/507)
    4-class       85.0%                      69.6%                      78.7%

  every window alone (no history: the contrast gate)
    LEFT          83.6%  (802/959)           59.0%  (69/117)            56.1%  (105/187)
    RIGHT         79.4%  (765/963)           64.5% (118/183)            74.7%  (165/221)
    DOWN          77.5%  (580/748)           57.7%  (71/123)            88.4%  (252/285)
    STRAIGHT      90.3% (14356/15890)        87.7% (742/846)            88.8%  (450/507)
    4-class       82.7%                      67.2%                      77.0%
    old extracted_npz_dataset (with spliced windows): 82.3% / 66.3% / 76.4% on the same splits

  The background ratio b was chosen by blocked cross-validation on sub_01 sessions 1-30 only; the two
  test sets were then scored once (they had been used in earlier rounds). Record:
  experiments/five_ideas_2026_09_17.

An earlier version appended three non-RTAMT Python corrections (a 70-formula TeLEx vote, a
48-formula specification bank and a nine-test STRAIGHT guard). They scored 84.1% on the tuning
data but 69.4% and 75.3% on the two held-out sets -- they did not transfer -- and they are not
RTAMT, so they were removed. This module is the four specifications and nothing else.

PROVENANCE: no parameter was fitted on sub_11, but sub_11 was used as an acceptance check during
design and influenced the electrode pairs (F4-T3 and F3/F4, not T3-T4). A design built from
sub_01 alone scores 58.1% on sub_11. Read 76.4% as an optimistic cross-subject estimate; only a
third subject gives a blind one.

LIVE STREAMING

:class:`StreamMonitor` takes one 8-channel sample at a time and emits a verdict once 768 samples
are buffered, then every ``decide_every`` samples. The default is one verdict per non-overlapping
3-second window, matching the deployment protocol. A window that cannot be judged -- non-finite
samples or a dead channel -- returns ``INVALID`` instead of raising.

``online=True`` feeds each window to RTAMT one sample at a time through its online monitor; the
default uses RTAMT's offline evaluator, which returns the identical robustness (verified) about
five times faster: roughly 0.36 s per window for all four specifications against 2.0 s online,
both inside the 3-second budget.

SEE THE FOUR MONITORS

    python -m monitoring                     # formulas + a demo on real windows
    python -m monitoring --session sub_01/ses-30
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path
from typing import Dict

import numpy as np

from .rtamt_signals import (
    CHANNELS,
    FS,
    SIGNALS,
    WINDOW_SAMPLES,
    InvalidWindow,
    RtamtConfig,
    check_window,
    compute_signals,
    loud_samples,
)

CLASSES = ("LEFT", "RIGHT", "DOWN", "STRAIGHT")
LAST = WINDOW_SAMPLES - 1


def import_rtamt():
    """Import RTAMT, falling back to the copy bundled under ``.runtime/rtamt``."""
    try:
        import rtamt
    except ModuleNotFoundError as exc:
        if exc.name != "rtamt":
            raise
        bundled = Path(__file__).resolve().parents[1] / ".runtime" / "rtamt"
        if not (bundled / "rtamt").is_dir():
            raise ImportError("RTAMT is required: pip install rtamt, or bundle it under "
                              ".runtime/rtamt") from exc
        sys.path.insert(0, str(bundled))
        import rtamt
    return rtamt


def class_formulas(config: RtamtConfig | None = None) -> Dict[str, str]:
    """The four STL specifications, with the configured thresholds written in."""
    cfg = config or RtamtConfig()
    if cfg.gate_statistic == "cusum":
        gate, g = "cusum", cfg.cusum_min
    else:
        gate, g = "contrast", cfg.gate_ratio
    d, q = cfg.delta_range_min, cfg.quiet_min
    n, v = cfg.down_threshold, cfg.down_veto_prominence
    w = f"[0:{LAST}]"
    loud = f"(once{w}({gate} >= {g!r}))"
    quiet = f"(historically{w}({gate} < {g!r}))"
    if cfg.background_ratio is not None and gate == "contrast":
        b = cfg.background_ratio
        loud = (f"((once{w}(background >= {b!r})) or ((historically{w}(background_valid < 0.5)) "
                f"and (once{w}(contrast >= {g!r}))))")
        quiet = (f"((historically{w}(background < {b!r})) and ((once{w}(background_valid >= 0.5)) "
                 f"or (historically{w}(contrast < {g!r}))))")
    move = f"{loud} and (once{w}(delta >= {d!r})) and (once{w}(deflect >= {q!r}))"
    vetoes = [("sideways", v)]
    if cfg.down_veto_fp1fp2 is not None:
        vetoes.append(("sideways_fp1fp2", cfg.down_veto_fp1fp2))
    if cfg.down_veto_fp2t3 is not None:
        vetoes.append(("sideways_fp2t3", cfg.down_veto_fp2t3))
    no_swing = " and ".join(f"(historically{w}({s} <= {t!r}))" for s, t in vetoes)
    swing = " or ".join(f"(once{w}({s} > {t!r}))" for s, t in vetoes)
    not_nod = f"((nod < {n!r}) or {swing})"
    if cfg.direction_rule == "since":
        h = cfg.direction_level
        went_left = f"((not (lateral >= {h!r})) since (lateral <= {-h!r}))"
        went_right = f"((not (lateral <= {-h!r})) since (lateral >= {h!r}))"
    else:
        went_left, went_right = "(turn > 0)", "(turn < 0)"
    return {
        "STRAIGHT": f"{quiet} or (historically{w}(delta < {d!r})) or (historically{w}(deflect < {q!r}))",
        "DOWN": f"({move}) and (nod >= {n!r}) and {no_swing}",
        "LEFT": f"({move}) and {not_nod} and {went_left}",
        "RIGHT": f"({move}) and {not_nod} and {went_right}",
    }


class ClassMonitor:
    """One head movement, one RTAMT specification, one robustness value."""

    label = ""

    def __init__(self, config: RtamtConfig | None = None, *, online: bool = False) -> None:
        self.config = config or RtamtConfig()
        self.formula = class_formulas(self.config)[self.label]
        self.variables = tuple(s for s in SIGNALS if re.search(rf"\b{s}\b", self.formula))
        self.online = online
        self._offline = None

    def _spec(self, online: bool):
        rtamt = import_rtamt()
        spec = (rtamt.StlDiscreteTimeOnlineSpecification() if online
                else rtamt.StlDiscreteTimeOfflineSpecification())
        for name in self.variables:
            spec.declare_var(name, "float")
        spec.spec = self.formula
        spec.parse()
        return spec

    def robustness_trace(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """Robustness at every sample of the window (RTAMT offline evaluator)."""
        if self._offline is None:
            self._offline = self._spec(online=False)
        data = {"time": list(range(len(signals[self.variables[0]])))}
        data.update({name: [float(x) for x in signals[name]] for name in self.variables})
        return np.array([row[1] for row in self._offline.evaluate(data)], dtype=float)

    def robustness_online(self, signals: Dict[str, np.ndarray]) -> float:
        """Robustness at the last sample, fed to RTAMT's online monitor one sample at a time."""
        spec = self._spec(online=True)
        value = 0.0
        for t in range(len(signals[self.variables[0]])):
            value = spec.update(t, [(name, float(signals[name][t])) for name in self.variables])
        return float(value)

    def robustness(self, signals: Dict[str, np.ndarray]) -> float:
        """Robustness of this class at the end of the window."""
        if self.online:
            return self.robustness_online(signals)
        return float(self.robustness_trace(signals)[-1])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.formula!r})"


class StraightMonitor(ClassMonitor):
    """STRAIGHT: throughout the window, at least one activity measure stayed low."""
    label = "STRAIGHT"


class DownMonitor(ClassMonitor):
    """DOWN: the window moved, the midline dominated, and nothing went sideways."""
    label = "DOWN"


class LeftMonitor(ClassMonitor):
    """LEFT: the window moved, it was not a nod, and the F4-T3 saccade went left."""
    label = "LEFT"


class RightMonitor(ClassMonitor):
    """RIGHT: the window moved, it was not a nod, and the F4-T3 saccade went right."""
    label = "RIGHT"


MONITORS = {"STRAIGHT": StraightMonitor, "DOWN": DownMonitor,
            "LEFT": LeftMonitor, "RIGHT": RightMonitor}


def satisfied_class(robustness: Dict[str, float]) -> str:
    """The class whose specification is satisfied.

    The specifications partition the space, so exactly one is satisfied. At an exact boundary
    the thresholds' >= semantics decide: STRAIGHT needs strictly positive robustness (its
    predicates are strict <), the others accept zero.
    """
    if robustness["STRAIGHT"] > 0:
        return "STRAIGHT"
    if robustness["DOWN"] >= 0:
        return "DOWN"
    return "LEFT" if robustness["LEFT"] >= 0 else "RIGHT"


class HeadMovementMonitor:
    """The four RTAMT class monitors over one 3-second window."""

    def __init__(self, config: RtamtConfig | None = None, *, online: bool = False) -> None:
        self.config = config or RtamtConfig()
        self.monitors = {c: MONITORS[c](self.config, online=online) for c in CLASSES}

    @property
    def formulas(self) -> Dict[str, str]:
        return {c: m.formula for c, m in self.monitors.items()}

    def signals(self, window, history=None, history_usable=None) -> Dict[str, np.ndarray]:
        return compute_signals(window, self.config, history, history_usable)

    def robustness(self, window, history=None, history_usable=None) -> Dict[str, float]:
        """RTAMT robustness of all four specifications at the end of the window.

        ``history``: optional (8, H) microvolt samples immediately before the window, for the
        stream-background gate (without it the contrast gate decides). ``history_usable``: optional
        per-sample mask of history samples allowed into the background (censored background).
        """
        signals = self.signals(window, history, history_usable)
        return {c: self.monitors[c].robustness(signals) for c in CLASSES}

    def explain(self, window, history=None, history_usable=None) -> dict:
        """The label, every class's robustness and the formulas that produced them."""
        try:
            raw = check_window(window)
            signals = compute_signals(raw, self.config, history, history_usable)
        except InvalidWindow as exc:
            return {"label": "INVALID", "reason": str(exc)}
        rho = {c: self.monitors[c].robustness(signals) for c in CLASSES}
        return {"label": satisfied_class(rho), "robustness": rho, "formulas": self.formulas,
                "background_used": bool(signals["background_valid"][-1] > 0.5)}

    def classify(self, window, history=None, history_usable=None) -> str:
        """One of LEFT, RIGHT, DOWN, STRAIGHT -- or INVALID for an unusable window."""
        return self.explain(window, history, history_usable)["label"]

    classify_window = classify


class StreamMonitor:
    """Push one 8-channel sample at a time; get a verdict when one is due.

    The first verdict is emitted on the 768th sample, then one every ``decide_every`` samples.
    Every verdict describes the 3 seconds ENDING at that sample and never uses a later sample.
    The default ``decide_every=768`` gives non-overlapping 3-second windows; a smaller value
    overlaps them, and must stay above the compute time (about 0.36 s per window offline).
    The samples received before each window (up to settle + background seconds) are passed to the
    monitor as ``history`` for the stream-background gate.

    CENSORED BACKGROUND (``RtamtConfig(background_censor=True)``; tested 2026-10-07, OFF by default). The
    loud 0.5-s blocks of every window judged LEFT, RIGHT or DOWN are marked; when marked samples lie in
    the 5 s before a window, the monitor passes up to ``background_memory_seconds`` of history plus that
    mask, and the background is measured on the newest unmarked stream instead of on the earlier
    movement. Otherwise (nothing marked in the last 5 s) the call is exactly the uncensored one.
    Real runs: second movements right 186 -> 251 of 318, single-run false alarms 348 -> 1214 (sub_01).
    """

    def __init__(self, config: RtamtConfig | None = None, *, online: bool = False,
                 decide_every: int = WINDOW_SAMPLES, monitor=None) -> None:
        if int(decide_every) < 1:
            raise ValueError("decide_every must be at least 1 sample")
        self.monitor = monitor if monitor is not None else HeadMovementMonitor(config,
                                                                              online=online)
        cfg = getattr(self.monitor, "config", None) or RtamtConfig()
        try:
            parameters = inspect.signature(self.monitor.classify_window).parameters
            self._takes_history = len(parameters) >= 2
            self._takes_mask = "history_usable" in parameters
        except (TypeError, ValueError):
            self._takes_history = self._takes_mask = False
        self.window = WINDOW_SAMPLES
        self.history = (0 if cfg.background_ratio is None else
                        int(round((cfg.background_settle_seconds + cfg.background_seconds) * FS)))
        self.censor = (bool(getattr(cfg, "background_censor", False)) and self.history > 0 and self._takes_mask
                       and hasattr(self.monitor, "signals"))
        self.memory = (max(self.history, int(round((cfg.background_settle_seconds
                                                    + cfg.background_memory_seconds) * FS)))
                       if self.censor else self.history)
        self.decide_every = int(decide_every)
        self.reset()

    def reset(self) -> None:
        """Forget the buffer. Call between subjects or after a break in the recording."""
        self._buffer = np.zeros((len(CHANNELS), self.window + self.memory))
        self._usable = np.ones(self.window + self.memory, dtype=bool)   # False: judged a movement
        self.samples_seen = 0
        self.last_label = None

    @property
    def ready(self) -> bool:
        return self.samples_seen >= self.window

    def push(self, sample):
        """Consume one (8,) column in microvolts; return a label when a verdict is due."""
        column = np.asarray(sample, dtype=float).reshape(-1)
        if column.shape[0] != len(CHANNELS):
            raise ValueError(f"expected {len(CHANNELS)} channel values, got {column.shape[0]}")
        self._buffer = np.roll(self._buffer, -1, axis=1)
        self._buffer[:, -1] = column
        self._usable = np.roll(self._usable, -1)
        self._usable[-1] = True
        self.samples_seen += 1
        if not self.ready or (self.samples_seen - self.window) % self.decide_every:
            return None
        M = self.memory
        earlier = min(self.samples_seen - self.window, self.history)
        history = self._buffer[:, M - earlier:M] if earlier > 0 else None
        window = self._buffer[:, M:]
        mask = None
        if self.censor and earlier > 0 and not self._usable[M - earlier:M].all():
            earlier = min(self.samples_seen - self.window, M)
            history, mask = self._buffer[:, M - earlier:M], self._usable[M - earlier:M].copy()
        if mask is not None:
            self.last_label = self.monitor.classify_window(window, history, history_usable=mask)
        elif self._takes_history:
            self.last_label = self.monitor.classify_window(window, history)
        else:                                               # a custom monitor without history support
            self.last_label = self.monitor.classify_window(window)
        if self.censor and self.last_label in ("LEFT", "RIGHT", "DOWN"):
            signals = self.monitor.signals(window, history, history_usable=mask)
            self._usable[M:][loud_samples(signals, self.monitor.config)] = False
        return self.last_label

    def push_block(self, x) -> list:
        """Feed many samples; returns [(sample_index, label), ...] for every verdict."""
        block = np.asarray(x, dtype=float)
        if block.ndim != 2 or block.shape[0] != len(CHANNELS):
            raise ValueError(f"expected shape ({len(CHANNELS)}, n_samples), got {block.shape}")
        out = []
        for k in range(block.shape[1]):
            label = self.push(block[:, k])
            if label is not None:
                out.append((self.samples_seen, label))
        return out


RtamtClassMonitor = ClassMonitor
RtamtStraightMonitor = StraightMonitor
RtamtDownMonitor = DownMonitor
RtamtLeftMonitor = LeftMonitor
RtamtRightMonitor = RightMonitor
RtamtHeadMovementMonitor = HeadMovementMonitor


def main(argv=None) -> None:
    """Print the four specifications and run them on real windows, one per class."""
    parser = argparse.ArgumentParser(description="Show the four RTAMT head-movement monitors.")
    parser.add_argument("--session", default="sub_11/ses-01",
                        help="extracted_npz_dataset_v2 session to demo on, e.g. sub_01/ses-30")
    args = parser.parse_args(argv)

    monitor = HeadMovementMonitor()
    print("THE FOUR RTAMT MONITORS\n")
    for c in ("STRAIGHT", "DOWN", "LEFT", "RIGHT"):
        print(f"{c}:\n    {monitor.formulas[c]}\n")

    path = (Path(__file__).resolve().parents[1] / "extracted_npz_dataset_v2" / args.session
            / "clean_epochs.npz")
    if not path.exists():
        print(f"(no demo: {path} not found)")
        return
    data = np.load(path, allow_pickle=True)
    labels = np.array([str(x).upper() for x in data["labels"]])
    print(f"DEMO on {args.session}: one window per class\n")
    print(f"{'true':10s}{'predicted':11s}" + "".join(f"{c:>11s}" for c in CLASSES))
    for c in CLASSES:
        idx = np.flatnonzero(labels == c)
        if not len(idx):
            continue
        window = data["epochs"][idx[0], :len(CHANNELS)] * 1e6
        d = monitor.explain(window)
        rho = d.get("robustness", {})
        print(f"{c:10s}{d['label']:11s}" + "".join(f"{rho.get(k, float('nan')):11.3f}"
                                                    for k in CLASSES))
    print("\nrobustness > 0 means that class's specification is satisfied.")


if __name__ == "__main__":
    main()


__all__ = ["CLASSES", "class_formulas", "import_rtamt", "satisfied_class", "ClassMonitor",
           "StraightMonitor", "DownMonitor", "LeftMonitor", "RightMonitor",
           "HeadMovementMonitor", "StreamMonitor", "RtamtHeadMovementMonitor",
           "RtamtStraightMonitor", "RtamtDownMonitor", "RtamtLeftMonitor",
           "RtamtRightMonitor", "RtamtClassMonitor"]
