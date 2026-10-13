"""Tests for the OFFLINE RTAMT monitor (monitoring/offline_monitor.py).

  * the four offline specifications parse in RTAMT, use only future-time operators, and partition the space;
  * RTAMT's robustness at t = 0 equals the min/max composition of (signal - threshold) terms;
  * the offline background equals an independent numpy reference and is unit-free;
  * loudness is judged against the rest of the SAME recording: one burst is a movement in a quiet recording
    and rest in a recording that is equally loud elsewhere;
  * windows at the very start of a recording work; windows outside it are INVALID, not a crash.
"""
import numpy as np
import pytest
from scipy.signal import butter, lfilter, lfilter_zi

from monitoring.offline_monitor import (
    OFFLINE_SIGNALS,
    OfflineConfig,
    OfflineHeadMovementMonitor,
    offline_background,
    offline_formulas,
)
from monitoring.rtamt_monitors import CLASSES, satisfied_class
from monitoring.rtamt_signals import CHANNELS, FS

N = 3 * FS
I = {c: k for k, c in enumerate(CHANNELS)}
MONITOR = OfflineHeadMovementMonitor()


def recording(seconds=10.0, bursts=(4.0,), amplitude=40.0, noise=5.0, seed=0):
    """Noise plus turn-shaped bursts (T3/T4 and F3/F4 in opposition) at the given times, in microvolts."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    x = rng.normal(0.0, noise, size=(len(CHANNELS), n)) + 30.0          # an arbitrary DC offset
    t = np.arange(n) / FS
    for at in bursts:
        shape = amplitude * (np.exp(-((t - at - 0.2) ** 2) / (2 * 0.08 ** 2))
                             - 0.6 * np.exp(-((t - at - 0.7) ** 2) / (2 * 0.25 ** 2)))
        for name, gain in (("T3", 1.0), ("T4", -1.0), ("F3", 0.5), ("F4", -0.5)):
            x[I[name]] += gain * shape
    return x


def reference_background(rec, start):
    b, a = butter(2, [0.5 / (FS / 2), 3.0 / (FS / 2)], btype="band")
    s, _ = lfilter(b, a, rec, axis=-1, zi=lfilter_zi(b, a)[None, :] * rec[:, :1])
    e = np.full(s.shape, np.nan)
    for t in range(127, s.shape[1]):
        e[:, t] = np.mean(s[:, t - 127:t + 1] ** 2, axis=1)
    keep = [t for t in range(127, s.shape[1]) if not (start - 127 <= t < start + N)][::8]
    ref = np.median(e[:, keep], axis=1) + 1e-12
    out = np.mean(e[:, start:start + N] / ref[:, None], axis=0)
    return np.where(np.isfinite(out), out, 0.0)


def test_offline_specifications_use_future_time_operators_only():
    for formula in offline_formulas().values():
        assert "eventually" in formula or "always" in formula
        assert "once" not in formula and "historically" not in formula and "since" not in formula


def test_offline_specifications_parse_in_rtamt():
    specs = MONITOR._parsed()
    assert set(specs) == set(CLASSES)


def test_threshold_values_are_written_into_the_formulas():
    f = offline_formulas()
    assert "eventually[0:767](background >= 5.75)" in f["LEFT"]
    assert "always[0:767](background < 5.75)" in f["STRAIGHT"]
    assert "(nod >= -0.6)" in f["DOWN"] and "always[0:767](sideways <= 7.5)" in f["DOWN"]


def _random_signals(rng):
    s = {k: rng.uniform(0.0, 12.0, N) for k in ("background", "sideways")}
    s["delta"] = rng.uniform(0.0, 0.5, N)
    s["deflect"] = rng.uniform(1.0, 5.0, N)
    s["nod"] = np.full(N, rng.uniform(-3.0, 2.0))
    s["turn"] = np.full(N, rng.uniform(-5.0, 5.0))
    scale = rng.uniform(0.05, 1.0)                     # sometimes everything stays low
    for k in ("background", "delta", "deflect", "sideways"):
        s[k] = s[k] * scale
    return s


def _composed(s, cfg=OfflineConfig()):
    b, d, q = cfg.background_ratio, cfg.live.delta_range_min, cfg.live.quiet_min
    n, v = cfg.live.down_threshold, cfg.live.down_veto_prominence
    move = min(np.max(s["background"] - b), np.max(s["delta"] - d), np.max(s["deflect"] - q))
    turned = max(n - s["nod"][0], np.max(s["sideways"] - v))
    return {"STRAIGHT": max(np.min(b - s["background"]), np.min(d - s["delta"]), np.min(q - s["deflect"])),
            "DOWN": min(move, s["nod"][0] - n, np.min(v - s["sideways"])),
            "LEFT": min(move, turned, s["turn"][0]), "RIGHT": min(move, turned, -s["turn"][0])}


@pytest.mark.parametrize("seed", range(12))
def test_rtamt_robustness_equals_the_composed_arithmetic(seed):
    s = _random_signals(np.random.default_rng(seed))
    got = MONITOR.robustness_from_signals(s)
    want = _composed(s)
    for c in CLASSES:
        assert got[c] == pytest.approx(want[c], abs=1e-9)


@pytest.mark.parametrize("seed", range(40))
def test_specifications_partition_the_space(seed):
    rho = MONITOR.robustness_from_signals(_random_signals(np.random.default_rng(100 + seed)))
    satisfied = [c for c in CLASSES if rho[c] > 0]
    assert len(satisfied) == 1
    assert satisfied_class(rho) == satisfied[0]


@pytest.mark.parametrize("start", [0, 60, 900, 1792])
def test_background_equals_an_independent_reference(start):
    rec = recording(seed=start)
    assert np.allclose(offline_background(rec, start), reference_background(rec, start), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("gain", [0.05, 7.0])
def test_background_is_unit_free(gain):
    rec = recording(seed=3)
    assert np.allclose(offline_background(gain * rec, 900), offline_background(rec, 900), rtol=1e-6)


def test_a_burst_is_a_movement_in_a_quiet_recording_and_rest_in_a_loud_one():
    quiet = recording(bursts=(4.0,), seed=5)                      # one turn, quiet elsewhere
    busy = recording(bursts=(1.0, 2.0, 3.0, 4.0, 5.2, 6.4, 7.6, 8.8), seed=5)   # equally loud everywhere
    start = int(3.2 * FS)                                         # window 3.2-6.2 s holds the turn at 4 s
    assert MONITOR.classify(quiet, start) in ("LEFT", "RIGHT")
    assert MONITOR.classify(busy, start) == "STRAIGHT"


def test_same_verdict_for_one_window_or_the_whole_recording():
    rec = recording(bursts=(2.0, 6.0), seed=8)
    starts = [0, 256, 800, 1300, len(rec[0]) - N]
    assert MONITOR.classify_recording(rec, starts) == [MONITOR.classify(rec, s) for s in starts]


def test_first_samples_of_a_window_at_the_recording_start_carry_no_background():
    bg = offline_background(recording(seed=9), 0)
    assert np.all(bg[:127] == 0.0) and np.all(bg[127:] > 0.0)


@pytest.mark.parametrize("start", [-1, 10 * FS - N + 1])
def test_window_outside_the_recording_is_invalid(start):
    assert MONITOR.classify(recording(seed=1), start) == "INVALID"


def test_dead_channel_is_invalid():
    rec = recording(seed=2)
    rec[I["Cz"]] = 0.0
    assert MONITOR.classify(rec, 500) == "INVALID"


def test_offline_signals_are_the_declared_set():
    s = MONITOR.signals(recording(seed=4), 700)
    assert set(s) == set(OFFLINE_SIGNALS)
    assert all(len(v) == N for v in s.values())
    assert np.ptp(s["nod"]) == 0 and np.ptp(s["turn"]) == 0
