"""Per-sample signals for the RTAMT head-movement monitors.

The four class monitors in :mod:`monitoring.rtamt_monitors` are Signal Temporal Logic
specifications evaluated by RTAMT. STL needs SIGNALS -- one value per sample -- and this module
turns one 3-second window of 8-channel EEG into the signals those specifications read:

    contrast   how lopsided in time the window is up to here: energy of the 0.5 s ending at t,
               over the energy of the window's quietest 0.5 s
    delta      how far the slow-wave (1-4 Hz) share of the 0.5 s slice ending at t sits above
               the window's lowest slice
    deflect    how far the eight channels have deflected so far, each over its own spread,
               averaged across channels
    nod        midline-versus-temporal energy balance so far, plus the F3/F4 common-mode
               symmetry of the loudest second seen so far
    sideways   the F4-T3 excursion at t, over that pair's own spread
    sideways_fp1fp2, sideways_fp2t3
               the same excursion on the Fp1-Fp2 and Fp2-T3 pairs (optional nod vetoes, off by default)
    turn       signed direction of the largest F4-T3 deflection inside the loudest 1.5 s seen
               so far: positive means LEFT, negative means RIGHT
    background how loud the 0.5 s ending at t is compared with the STREAM BEFORE the window: each
               channel's 0.5-3 Hz energy over that channel's median energy in the 4 s before the
               window, averaged over channels (0 when no earlier stream was given)
    background_valid
               1 when at least 1 s of earlier stream (after a 1 s filter settle) was available, else 0

and two signals read only by the OPTIONAL designs tested on 2026-09-16 (not the default):

    cusum      Page's CUSUM change detector on the window's movement energy: how much log-energy
               above the quietest 0.5 s (beyond an allowance of kappa nats) has accumulated by t
    lateral    the F4-T3 deflection inside the window's loudest 1.5 s, as a fraction of its
               largest excursion there (zero outside that stretch); read by a SINCE formula

This module is self-contained: numpy and scipy only.

WHY EACH SIGNAL IS BUILT THIS WAY

Every signal is constructed so that the STL operator applied to it recovers exactly one
physics measurement:

    once[0:767](cusum >= c)          <=>  largest CUSUM of log(energy / quietest energy) - kappa >= c
    once[0:767](delta >= d)          <=>  spread of the 1-4 Hz share across slices >= d
    once[0:767](deflect >= q)        <=>  mean channel prominence >= q
    nod at the last sample           ==   midline ratio + F3/F4 symmetry
    historically[0:767](sideways<=v) <=>  largest F4-T3 excursion / spread <= v
    historically[0:767](sideways_fp1fp2<=v1), historically[0:767](sideways_fp2t3<=v2)
                                     <=>  the same on Fp1-Fp2 and Fp2-T3
    (not (lateral >= h)) since (lateral <= -h)
                                     <=>  the LAST large F4-T3 excursion was negative (LEFT)
    once[0:767](contrast >= g)       <=>  loudest 0.5 s / quietest 0.5 s >= g
    once[0:767](background >= b)     <=>  loudest 0.5 s is >= b times the stream's recent background
    turn at the last sample          has the sign of the largest excursion

THE STREAM BACKGROUND (adopted 2026-09-17)

The contrast gate compares a window only with ITSELF, which fails in both directions: in a quiet
window a small residual blink is "loud" (a false alarm), and in a busy window a real nod does not
stand out (a miss). Radar detectors solve the same problem by comparing each cell with the recent
background (constant-false-alarm-rate detection). ``background`` compares the window with the 4 s
of stream before it. Every channel is divided by its own median, so the value is unit-free. When
fewer than 1 s of earlier stream exists, ``background_valid`` is 0 and the contrast gate decides as
before. Pass the earlier samples as ``history``: ``compute_signals(window, history=previous)``.
Result when adopted, 4-class balanced accuracy on extracted_npz_dataset_v2 (history rebuilt from the
dataset's own windows, see ``monitoring.stream_history``); background ratio 6.61 chosen by blocked CV on
sub_01 sessions 1-30. (Since 2026-10-07 all thresholds are selected on sessions 1-20 and b is 8.0;
current numbers: monitoring/rtamt_monitors.py, MEASURED ACCURACY.)
                                           sessions 1-30   sub_01 31-36   sub_11
    contrast gate alone                       82.74%          67.22%       77.00%
    stream background gate                    85.01%          69.62%       78.67%
The gain comes from rest windows (STRAIGHT 90.3 -> 97.9, 87.7 -> 97.3, 88.8 -> 95.5); the three
movement classes are unchanged or slightly better. Record: experiments/five_ideas_2026_09_17.

TWO OPTIONAL TECHNIQUES, TESTED ON 2026-09-16 AND NOT ADOPTED

Both were designed on sub_01 sessions 1-30 only and scored once on sub_01 sessions 31-36 and
sub_11. Select them with ``RtamtConfig(gate_statistic="cusum", direction_rule="since")``.

CUSUM gate. The default gate asks whether the loudest 0.5 s was g times the quietest 0.5 s, which
a brief spike (a blink, a swallow, an electrode pop) passes as easily as a head movement. CUSUM
accumulates the log-energy excess over time, g_t = max(0, g_{t-1} + x_t - kappa), so evidence
must be both large (above e**kappa times the quietest level) and sustained. It is the classical
sequential change detector, causal inside the window, and scale-free.

SINCE direction. A head turn is biphasic in F4-T3. The default rule takes the sign of the LARGEST
excursion, which is close to a coin toss when the two lobes are nearly equal (68.9% correct when
the smaller lobe is within 15% of the larger, on sessions 1-30). The STL formula
``(not (lateral >= h)) since (lateral <= -h)`` lets the LAST excursion reaching h of the largest
decide: identical to the peak rule when one lobe dominates, the later lobe when they tie.

Result (4-class balanced accuracy):     sessions 1-30   sub_01 31-36   sub_11
    default (contrast gate, peak)          82.30%          66.28%       76.40%
    CUSUM gate only                        82.81%          66.08%       76.16%
    SINCE direction only                   82.99%          65.26%       75.72%
    CUSUM gate + SINCE direction           83.50%          65.05%       75.48%
Both gains on the design sessions failed to transfer: SINCE broke one RIGHT turn in each held-out
set and fixed none, and CUSUM fixed 11 + 8 held-out windows but broke 12 + 12.

WHAT IS AND IS NOT CAUSAL

Values at sample t use only samples up to t, EXCEPT for the per-window normalisers: each
channel's median and robust spread, the quietest 0.5 s block, the lowest slice share, and the
position and largest excursion of the loudest 1.5 s that ``lateral`` is read in. Those describe
the whole completed window. That matches the deployment protocol -- the stream delivers
complete 3-second windows -- and it is what makes every signal scale-free: a per-subject gain
multiplies a value and its own normaliser alike and cancels. No quantity is compared with a
stored microvolt level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.signal import butter, lfilter

FS = 256
WINDOW_SAMPLES = 3 * FS
CHANNELS = ("Fp1", "Fp2", "F3", "Fz", "F4", "T3", "Cz", "T4")
_IDX = {c: i for i, c in enumerate(CHANNELS)}
CORE_SIGNALS = ("cusum", "contrast", "delta", "deflect", "nod", "sideways", "sideways_fp1fp2",
                "sideways_fp2t3", "lateral", "turn", "background", "background_valid")
AUXILIARY_SIGNALS = ("turn_fp2_t3", "turn_t3_t4", "turn_fp1_f4",
                     "turn_vote5", "turn_mean5", "turn_consensus")
SIGNALS = CORE_SIGNALS + AUXILIARY_SIGNALS
_DIRECTION_PAIRS = {
    "Fp1-T4": ("Fp1", "T4", -1), "Fp2-F4": ("Fp2", "F4", -1),
    "F3-T3": ("F3", "T3", 1), "F4-T3": ("F4", "T3", 1),
    "T3-T4": ("T3", "T4", -1), "Fp2-T3": ("Fp2", "T3", 1),
    "F3-T4": ("F3", "T4", -1), "Fp1-F4": ("Fp1", "F4", -1),
}
_VOTE_PAIRS = ("Fp1-T4", "Fp2-F4", "F3-T3", "F4-T3", "T3-T4")
_MEAN_PAIRS = ("Fp2-F4", "Fp2-T3", "F3-T4", "F4-T3", "T3-T4")
_TINY = 1e-12


def _band(lo: float, hi: float):
    return butter(2, [lo / (FS / 2), hi / (FS / 2)], btype="band")


_BAND_DETECT = _band(0.5, 3.0)       # where head movement lives; used to find the active stretch
_BAND_CLASS = _band(0.5, 6.0)        # channel prominence and the nod evidence
_BAND_DIRECTION = _band(0.3, 6.0)    # the direction read
_BAND_PROMINENCE = _band(0.3, 4.0)   # the sideways veto


@dataclass(frozen=True)
class RtamtConfig:
    """The thresholds the specifications test, and the geometry the signals use.

    Every threshold is dimensionless -- a ratio, a share, a log ratio or nat-seconds of log
    ratio -- and none is in microvolts. b, g, d, q, n, v were selected on sub_01 sessions 1-20
    (the default since 2026-10-07; the previous values, selected on sessions 1-30, are noted
    beside each field). g = 5.0 is the floor of the search grid.

    The defaults are the validated monitor: the contrast gate and the largest-peak direction.
    ``gate_statistic="cusum"`` and ``direction_rule="since"`` are the two options tested on
    2026-09-16 (see the module docstring): chosen on sub_01 sessions 1-30 only, they raised
    accuracy there but not on the held-out sessions, so they are not the default. ``cusum_min``
    started at the value with the contrast gate's false-alarm rate on sessions 1-30 and was moved
    by the one-standard-error rule; ``direction_level`` was chosen on the same sessions.

    Provenance, stated plainly: no threshold was fitted on sub_11, but sub_11 was used as an
    acceptance check during design, and that decided the electrode pairs below (F4-T3 and F3/F4
    rather than T3-T4, because sub_11's T4 swings by hundreds of uV during its left turns). A design built from sub_01
    alone chooses T3-T4 and scores 58.1% on sub_11 rather than 76.4%, so the sub_11 result is an
    optimistic cross-subject estimate, not a blind one.
    """

    gate_statistic: str = "contrast"    # "contrast" (default) or "cusum" (tested, not adopted)
    cusum_min: float = 0.0684           # c: accumulated log-energy excess, in nat-seconds
    cusum_kappa: float = 2.0            # allowance, nats: only energy above e**2 = 7.4x quietest counts
    cusum_smooth_seconds: float = 0.125
    cusum_reference_seconds: float = 0.5
    gate_ratio: float = 5.0             # g: loudest over quietest 0.5 s (gate_statistic="contrast"); 7.81 until 2026-10-07
    gate_seconds: float = 0.5
    delta_range_min: float = 0.16       # d: spread of the 1-4 Hz share; 0.208 until 2026-10-07
    delta_slice_seconds: float = 0.5
    quiet_min: float = 2.8              # q: mean channel prominence; 2.72 until 2026-10-07
    down_threshold: float = -0.6        # n: nod evidence, in nats; -0.661 until 2026-10-07
    down_veto_prominence: float = 7.5   # v: largest sideways excursion allowed for a nod; 7.54 until 2026-10-07
    down_veto_fp1fp2: float | None = None  # v1: largest Fp1-Fp2 excursion allowed for a nod (None: off)
    down_veto_fp2t3: float | None = None   # v2: largest Fp2-T3 excursion allowed for a nod (None: off)
    direction_rule: str = "peak"        # "peak" (default) or "since" (tested, not adopted)
    direction_level: float = 0.95       # h: an excursion counts when it reaches h of the largest
    symmetry_seconds: float = 1.0
    direction_seconds: float = 1.5
    lateral_peak_pair: tuple = ("F4", "T3")
    background_ratio: float | None = 8.0    # 6.61 until 2026-10-07
    background_seconds: float = 4.0         # length of the background stretch before the window
    background_min_seconds: float = 1.0     # less than this -> background_valid = 0
    background_settle_seconds: float = 1.0  # start of the given history skipped while the filter settles
    background_censor: bool = False
    background_memory_seconds: float = 20.0  # how far back the stream monitors keep samples for the background

    def __post_init__(self):
        if self.gate_statistic not in ("cusum", "contrast"):
            raise ValueError("gate_statistic must be 'cusum' or 'contrast'")
        if self.direction_rule not in ("since", "peak"):
            raise ValueError("direction_rule must be 'since' or 'peak'")
        if not 0.0 < self.direction_level <= 1.0:
            raise ValueError("direction_level must be in (0, 1]")
        if self.background_ratio is not None and self.background_ratio <= 0:
            raise ValueError("background_ratio must be positive or None")
        if self.background_memory_seconds < self.background_seconds:
            raise ValueError("background_memory_seconds must be at least background_seconds")


class InvalidWindow(ValueError):
    """The window cannot be judged: wrong shape, non-finite samples or a dead channel."""


def check_window(window) -> np.ndarray:
    """Validate one window and return it as a contiguous (8, 768) float array."""
    raw = np.asarray(window, dtype=float)
    if raw.ndim != 2 or raw.shape[0] != len(CHANNELS):
        raise InvalidWindow(f"expected shape ({len(CHANNELS)}, n_samples), got {raw.shape}")
    if raw.shape[1] < WINDOW_SAMPLES:
        raise InvalidWindow(f"expected at least {WINDOW_SAMPLES} samples, got {raw.shape[1]}")
    raw = np.ascontiguousarray(raw[:, :WINDOW_SAMPLES])
    if not np.isfinite(raw).all():
        raise InvalidWindow("EEG samples must be finite")
    dead = [CHANNELS[i] for i in range(len(CHANNELS)) if np.ptp(raw[i]) == 0]
    if dead:
        raise InvalidWindow(f"flat EEG channel(s): {', '.join(dead)}")
    return raw


def _pooled_energy(raw: np.ndarray) -> np.ndarray:
    """0.5-3 Hz energy pooled over channels, each divided by its own robust spread."""
    band = lfilter(_BAND_DETECT[0], _BAND_DETECT[1], raw, axis=-1)
    spread = (1.4826 * np.median(np.abs(band - np.median(band, axis=1, keepdims=True)),
                                 axis=1, keepdims=True) + 1e-12)
    return np.mean((band / spread) ** 2, axis=0)


def _block_sums(energy: np.ndarray, width: int) -> np.ndarray:
    """Sum of ``energy`` over each ``width``-sample block; element k covers k..k+width-1."""
    cumulative = np.concatenate([[0.0], np.cumsum(energy)])
    return cumulative[width:] - cumulative[:-width]


def _running_loudest(energy: np.ndarray, width: int) -> np.ndarray:
    """Start of the loudest ``width`` block among blocks finished by each sample; -1 before any.

    A strictly-greater comparison keeps the FIRST maximum, exactly as ``np.argmax`` does, so the
    choice at the last sample is the whole-window loudest block.
    """
    blocks = _block_sums(energy, width)
    start = np.full(len(energy), -1, dtype=int)
    best, best_value = -1, -np.inf
    for k, value in enumerate(blocks):
        if value > best_value:
            best, best_value = k, value
        start[k + width - 1] = best
    for t in range(1, len(start)):                 # hold the latest choice between block ends
        if start[t] < 0 <= start[t - 1]:
            start[t] = start[t - 1]
    return start


def background_positions(start: int, config: RtamtConfig | None = None, usable=None):
    """Indices into history (length ``start``) whose 0.5-s block energy forms the background median."""
    cfg = config or RtamtConfig()
    settle = int(round(cfg.background_settle_seconds * FS))
    need = int(round(cfg.background_seconds * FS))
    minimum = int(round(cfg.background_min_seconds * FS))
    if usable is None:
        lo = max(settle, start - need)
        return np.arange(lo, start, 8) if start - lo >= minimum else None
    bad = np.concatenate([[0], np.cumsum(~np.asarray(usable, dtype=bool))])
    ends = np.arange(start - 8, settle - 1, -8)                        # newest first
    first = np.maximum(ends - max(int(cfg.gate_seconds * FS), 1) + 1 - settle, 0)
    clean = ends[(bad[ends + 1] - bad[first]) == 0][:need // 8]
    return np.sort(clean) if len(clean) >= minimum // 8 else None


def loud_samples(signals: Dict[str, np.ndarray], config: RtamtConfig | None = None) -> np.ndarray:
    """Bool per window sample: True = inside a 0.5-s block that passed the loudness gate."""
    cfg = config or RtamtConfig()
    valid = bool(signals["background_valid"][-1] > 0.5) and cfg.background_ratio is not None
    trace = signals["background"] if valid else signals["contrast"]
    level = cfg.background_ratio if valid else cfg.gate_ratio
    block = max(int(cfg.gate_seconds * FS), 1)
    ends = np.flatnonzero(trace >= level)
    marks = np.zeros(len(trace) + 1, dtype=int)
    np.add.at(marks, np.maximum(ends - block + 1, 0), 1)
    np.add.at(marks, ends + 1, -1)
    return np.cumsum(marks)[:-1] > 0


def stream_background(raw: np.ndarray, history, config: RtamtConfig | None = None, usable=None):
    """(background, background_valid) for one window, 768 values each.

    ``history``: (8, H) microvolt samples that came IMMEDIATELY before the window, continuous with
    it (same offset). The 0.5-3 Hz filter runs over history + window from the first history
    sample; the first ``background_settle_seconds`` are skipped; the background of each channel is
    the median of its 0.5 s block energy over the last ``background_seconds`` of history before the
    window (every 8th sample). Values at sample t use only samples up to t.

    ``usable`` (optional, one bool per history sample): False marks samples of windows already judged
    a movement; their blocks are left out of the median (see :func:`background_positions`). None or
    all-True gives exactly the original rule.
    """
    cfg = config or RtamtConfig()
    n = raw.shape[1]
    zeros = (np.zeros(n), np.zeros(n))
    if history is None or cfg.background_ratio is None:
        return zeros
    hist = np.asarray(history, dtype=float)
    if hist.ndim != 2 or hist.shape[0] != len(CHANNELS):
        raise InvalidWindow(f"history must have shape ({len(CHANNELS)}, n_samples), got {hist.shape}")
    if not np.isfinite(hist).all():
        raise InvalidWindow("history samples must be finite")
    if usable is not None:
        usable = np.asarray(usable, dtype=bool).reshape(-1)
        if usable.shape[0] != hist.shape[1]:
            raise InvalidWindow(f"history_usable needs one value per history sample ({hist.shape[1]}), "
                                f"got {usable.shape[0]}")
        if usable.all():
            usable = None
    start = hist.shape[1]
    positions = background_positions(start, cfg, usable)
    if positions is None:
        return zeros
    block = max(int(cfg.gate_seconds * FS), 1)
    band = lfilter(_BAND_DETECT[0], _BAND_DETECT[1], np.concatenate([hist, raw], axis=1), axis=-1)
    cumulative = np.concatenate([np.zeros((band.shape[0], 1)), np.cumsum(band ** 2, axis=1)], axis=1)
    energy = np.zeros_like(band)
    energy[:, block - 1:] = (cumulative[:, block:] - cumulative[:, :-block]) / block
    reference = np.median(energy[:, positions], axis=1) + 1e-12
    background = np.mean(energy[:, start:start + n] / reference[:, None], axis=0)
    return background, np.ones(n)


def compute_signals(window, config: RtamtConfig | None = None, history=None,
                    history_usable=None) -> Dict[str, np.ndarray]:
    """The monitor signals plus six unused direction diagnostics; 768 values each.

    ``history`` (optional): the (8, H) microvolt samples immediately before the window, for the
    stream-background gate. Without it ``background`` and ``background_valid`` are 0.
    ``history_usable`` (optional): one bool per history sample; False = censored (see stream_background).
    """
    cfg = config or RtamtConfig()
    raw = check_window(window)
    n = raw.shape[1]
    energy = _pooled_energy(raw)
    background, background_valid = stream_background(raw, history, cfg, history_usable)

    gate = max(int(cfg.gate_seconds * FS), 1)
    blocks = _block_sums(energy, gate)
    contrast = np.ones(n)
    contrast[gate - 1:] = blocks / (float(blocks.min()) + 1e-18)

    reference = max(int(round(cfg.cusum_reference_seconds * FS)), 1)
    quietest = float((_block_sums(energy, reference) / reference).min())
    smooth = max(int(round(cfg.cusum_smooth_seconds * FS)), 1)
    excess = (np.log(_block_sums(energy, smooth) / smooth + 1e-18) - np.log(quietest + 1e-18)
              - cfg.cusum_kappa)
    cusum = np.zeros(n)
    accumulated = 0.0
    for k, x in enumerate(excess):                  # element k is the block ending at k+smooth-1
        accumulated = max(0.0, accumulated + float(x))
        cusum[k + smooth - 1] = accumulated / FS

    width = max(int(cfg.delta_slice_seconds * FS), 8)
    step = max(width // 2, 1)
    taper = np.hanning(width)
    freq = np.fft.rfftfreq(width, 1.0 / FS)
    in_delta = (freq >= 1.0) & (freq < 4.0)
    in_total = (freq >= 0.5) & (freq <= 45.0)
    starts = list(range(0, n - width + 1, step))
    shares = []
    for s in starts:
        seg = raw[:, s:s + width]
        seg = seg - seg.mean(axis=1, keepdims=True)
        power = np.abs(np.fft.rfft(seg * taper, axis=1)) ** 2
        shares.append(float(np.mean(power[:, in_delta].sum(axis=1)
                                    / (power[:, in_total].sum(axis=1) + 1e-18))))
    lowest = min(shares)
    delta = np.zeros(n)
    for s, share in zip(starts, shares):
        delta[s + width - 1:] = share - lowest

    classify = lfilter(_BAND_CLASS[0], _BAND_CLASS[1], raw, axis=-1)
    classify = classify - np.median(classify, axis=1, keepdims=True)
    spread = 1.4826 * np.median(np.abs(classify), axis=1) + 1e-12
    deflect = np.mean(np.maximum.accumulate(np.abs(classify), axis=1) / spread[:, None], axis=0)

    squared = classify ** 2
    midline = np.cumsum(squared[[_IDX["Fz"], _IDX["Cz"]]].sum(axis=0))
    temporal = np.cumsum(squared[[_IDX["T3"], _IDX["T4"]]].sum(axis=0))
    count = 2 * np.arange(1, n + 1)
    balance = np.log((midline / count + 1e-18) / (temporal / count + 1e-18))
    sym_width = int(cfg.symmetry_seconds * FS)
    symmetry = np.zeros(n)
    cache: Dict[int, float] = {}
    for t, s in enumerate(_running_loudest(energy, sym_width)):
        if s < 0:
            continue
        if s not in cache:
            seg = classify[:, s:s + sym_width]
            common = float(np.mean((seg[_IDX["F3"]] + seg[_IDX["F4"]]) ** 2))
            differential = float(np.mean((seg[_IDX["F3"]] - seg[_IDX["F4"]]) ** 2))
            cache[s] = float(np.log((common + 1e-18) / (differential + 1e-18)))
        symmetry[t] = cache[s]
    nod = balance + symmetry

    a, b = cfg.lateral_peak_pair
    prominence = lfilter(_BAND_PROMINENCE[0], _BAND_PROMINENCE[1], raw, axis=-1)

    def excursion(first, second):
        v = prominence[_IDX[first]] - prominence[_IDX[second]]
        v = v - np.median(v)
        return np.abs(v) / (1.4826 * np.median(np.abs(v)) + 1e-12)

    sideways = excursion(a, b)
    sideways_fp1fp2 = excursion("Fp1", "Fp2")
    sideways_fp2t3 = excursion("Fp2", "T3")

    direction = lfilter(_BAND_DIRECTION[0], _BAND_DIRECTION[1], raw, axis=-1)
    pair = direction[_IDX[a]] - direction[_IDX[b]]
    turn_width = int(cfg.direction_seconds * FS)
    turn = np.zeros(n)
    cache = {}
    for t, s in enumerate(_running_loudest(energy, turn_width)):
        if s < 0:
            continue
        if s not in cache:
            seg = pair[s:s + turn_width]
            seg = seg - np.median(seg)
            peak = float(seg[int(np.argmax(np.abs(seg)))])
            value = -peak / (1.4826 * float(np.median(np.abs(seg))) + 1e-12)
            cache[s] = value if value != 0.0 else _TINY    # a dead-level tie reads as LEFT
        turn[t] = cache[s]

    lateral = np.zeros(n)
    start = int(np.argmax(_block_sums(energy, turn_width)))
    seg = pair[start:start + turn_width]
    seg = seg - np.median(seg)
    largest = float(np.abs(seg).max())
    if largest > 0.0:
        lateral[start:start + turn_width] = seg / largest
    else:
        lateral[start] = -1.0                           # a dead-level pair reads as LEFT

    result = {"contrast": contrast, "cusum": cusum, "delta": delta, "deflect": deflect,
              "nod": nod, "sideways": sideways, "sideways_fp1fp2": sideways_fp1fp2,
              "sideways_fp2t3": sideways_fp2t3, "turn": turn, "lateral": lateral,
              "background": background, "background_valid": background_valid}
    result.update(_direction_bank(direction, energy, turn_width))
    votes = np.sign(np.array([result[k] for k in
        ("turn", "turn_fp2_t3", "turn_t3_t4", "turn_fp1_f4", "turn_vote5", "turn_mean5")])).sum(axis=0)
    result['turn_consensus'] = np.where(votes == 0, np.sign(turn), votes)
    return result


def _direction_bank(filtered, energy, width):
    """Diagnostic directions: positive LEFT, negative RIGHT, zero until ready.

    Same filtered peak and robust scaling as turn. Vote and mean memberships
    were selected on development sessions; neither enters the monitor formulas.
    """
    starts = _running_loudest(energy, width)
    names = tuple(_DIRECTION_PAIRS)
    traces = np.zeros((len(names), len(energy)))
    pairs = np.array([filtered[_IDX[a]] - filtered[_IDX[b]]
                      for a, b, _ in _DIRECTION_PAIRS.values()])
    polarity = np.array([p for _, _, p in _DIRECTION_PAIRS.values()])
    for start in np.unique(starts[starts >= 0]):
        block = pairs[:, start:start + width]
        block = block - np.median(block, axis=1, keepdims=True)
        peak = block[np.arange(len(names)), np.argmax(np.abs(block), axis=1)]
        score = -peak * polarity / (1.4826 * np.median(np.abs(block), axis=1) + _TINY)
        score = np.where(score == 0, _TINY, score)
        traces[:, starts == start] = score[:, None]
    by_pair = dict(zip(names, traces))
    vote = np.sign(np.array([by_pair[k] for k in _VOTE_PAIRS])).sum(axis=0)
    mean = np.mean([by_pair[k] for k in _MEAN_PAIRS], axis=0)
    mean = np.where((starts >= 0) & (mean == 0), by_pair['F4-T3'], mean)
    return dict(turn_fp2_t3=by_pair['Fp2-T3'], turn_t3_t4=by_pair['T3-T4'],
                turn_fp1_f4=by_pair['Fp1-F4'], turn_vote5=vote, turn_mean5=mean)


__all__ = ["FS", "WINDOW_SAMPLES", "CHANNELS", "SIGNALS", "CORE_SIGNALS", "AUXILIARY_SIGNALS", "RtamtConfig", "InvalidWindow",
           "check_window", "compute_signals", "stream_background", "background_positions", "loud_samples"]
