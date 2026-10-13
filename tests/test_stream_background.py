"""Tests for the stream-background gate (adopted 2026-09-17).

  * without history the monitor is exactly the contrast-gate monitor;
  * the background signal equals an independent numpy reference, and is unit-free;
  * the STL gate follows its declared semantics at the thresholds, with and without valid history;
  * the four specifications still partition the space, online RTAMT equals offline;
  * the same burst is a movement after a quiet stretch and rest after an equally loud stretch;
  * StreamMonitor passes the preceding samples as history;
  * runs rebuilt from the dataset's own windows continue each stored window exactly.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, lfilter

from monitoring import HeadMovementMonitor, StreamMonitor
from monitoring.rtamt_monitors import CLASSES, class_formulas, satisfied_class
from monitoring.rtamt_signals import CHANNELS, FS, RtamtConfig, compute_signals, stream_background

N = 3 * FS
I = {c: k for k, c in enumerate(CHANNELS)}
OFF = RtamtConfig(background_ratio=None)


def turn(seed=0, amplitude=40.0, event=1.2, noise=5.0, n=N):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, noise, size=(len(CHANNELS), n))
    if amplitude:
        t = np.arange(n) / FS - event
        shape = amplitude * (np.exp(-((t - 0.2) ** 2) / (2 * 0.08 ** 2)) - 0.6 * np.exp(-((t - 0.7) ** 2) / (2 * 0.25 ** 2)))
        for name, gain in (("T3", 1.0), ("T4", -1.0), ("F3", 0.5), ("F4", -0.5)):
            x[I[name]] += gain * shape
    return x


def reference_background(window, history, cfg=RtamtConfig()):
    b, a = butter(2, [0.5 / (FS / 2), 3.0 / (FS / 2)], btype="band")
    s = lfilter(b, a, np.concatenate([history, window], axis=1), axis=-1)
    o = history.shape[1]
    e = np.full(s.shape, np.nan)
    for t in range(127, s.shape[1]):
        e[:, t] = np.mean(s[:, t - 127:t + 1] ** 2, axis=1)
    lo = max(FS, o - 4 * FS)
    ref = np.median(e[:, lo:o:8], axis=1) + 1e-12
    return np.mean(e[:, o:o + N] / ref[:, None], axis=0)


def test_formulas_read_the_background_and_fall_back_to_contrast():
    f = class_formulas()
    assert "once[0:767](background >= 8.0)" in f["LEFT"]
    assert "historically[0:767](background_valid < 0.5)" in f["DOWN"]
    assert "historically[0:767](background < 8.0)" in f["STRAIGHT"]
    assert all("background" not in s for s in class_formulas(OFF).values())


BASE = dict(cusum=0.2, contrast=10.0, delta=.4, deflect=4.0, nod=-2.0, sideways=5.0, turn=1.0, lateral=-1.0,
            sideways_fp1fp2=3.0, sideways_fp2t3=3.0, background=10.0, background_valid=1.0)


@pytest.mark.parametrize("change,expected", [
    ({"background": 8.0}, "LEFT"),
    ({"background": 8.0 - 1e-5}, "STRAIGHT"),
    ({"background": 8.0 - 1e-5, "contrast": 1000.0}, "STRAIGHT"),    # valid history: contrast is not asked
    ({"background": 50.0, "contrast": 1.0}, "LEFT"),
    ({"background_valid": 0.0, "background": 0.0, "contrast": 5.0}, "LEFT"),          # no history: contrast gate
    ({"background_valid": 0.0, "background": 0.0, "contrast": 5.0 - 1e-5}, "STRAIGHT"),
    ({"background": 20.0, "delta": 0.1}, "STRAIGHT"),
    ({"nod": 0.0}, "DOWN"),
    ({"turn": -1.0}, "RIGHT"),
])
def test_gate_boundaries_follow_the_declared_semantics(change, expected):
    mon = HeadMovementMonitor()
    values = dict(BASE)
    values.update(change)
    signals = {k: np.full(N, v) for k, v in values.items()}
    rho = {c: mon.monitors[c].robustness(signals) for c in CLASSES}
    assert satisfied_class(rho) == expected
    assert len([c for c in CLASSES if rho[c] > 0]) <= 1


def test_without_history_the_monitor_is_the_contrast_gate_monitor():
    on, off = HeadMovementMonitor(), HeadMovementMonitor(OFF)
    for seed in range(3):
        for amp in (0.0, 40.0):
            x = turn(seed, amp)
            s = compute_signals(x)
            assert not s["background"].any() and not s["background_valid"].any()
            assert on.classify(x) == off.classify(x)


def test_too_little_history_is_not_used():
    x = turn(1)
    for seconds in (0.5, 1.0, 1.99):
        b, valid = stream_background(x, turn(2, 0.0, n=int(seconds * FS)))
        assert not valid.any() and not b.any()
    b, valid = stream_background(x, turn(2, 0.0, n=2 * FS))
    assert valid.all()


@pytest.mark.parametrize("seconds", [2, 4, 5, 9])
def test_background_matches_an_independent_reference(seconds):
    x, h = turn(3), turn(4, 0.0, n=seconds * FS)
    b, _ = stream_background(x, h)
    ref = reference_background(x, h)
    ok = np.isfinite(ref)
    assert np.allclose(b[ok], ref[ok], rtol=1e-9)


@pytest.mark.parametrize("gain", [0.01, 7.0, 300.0])
def test_background_and_robustness_are_unit_free(gain):
    x, h = turn(5), turn(6, 0.0, n=5 * FS)
    mon = HeadMovementMonitor()
    a, b = mon.robustness(x, h), mon.robustness(x * gain, h * gain)
    for c in CLASSES:
        assert b[c] == pytest.approx(a[c], rel=1e-6, abs=1e-9)


def test_the_same_burst_depends_on_what_came_before():
    """A burst after quiet stream is a movement; the same burst after an equally loud stream is not."""
    mon = HeadMovementMonitor()
    x = turn(7, amplitude=40.0)
    quiet = turn(8, amplitude=0.0, n=5 * FS)
    loud = np.concatenate([turn(9, 40.0, event=0.6, n=int(2.5 * FS)), turn(10, 40.0, event=0.6, n=int(2.5 * FS))], axis=1)
    assert mon.classify(x, quiet) in ("LEFT", "RIGHT")
    assert mon.classify(x, loud) == "STRAIGHT"
    assert compute_signals(x, history=quiet)["background"].max() > 8.0 > compute_signals(x, history=loud)["background"].max()


def test_online_rtamt_equals_offline_with_history():
    x, h = turn(11), turn(12, 0.0, n=5 * FS)
    a, b = HeadMovementMonitor().robustness(x, h), HeadMovementMonitor(online=True).robustness(x, h)
    for c in CLASSES:
        assert a[c] == pytest.approx(b[c], abs=1e-9)


def test_invalid_history_gives_invalid_not_a_crash():
    x = turn(13)
    h = turn(14, 0.0, n=3 * FS)
    h[0, 5] = np.nan
    assert HeadMovementMonitor().classify(x, h) == "INVALID"


def test_stream_monitor_passes_the_preceding_samples_as_history():
    stream = StreamMonitor()
    assert stream.history == 5 * FS
    first, second, third = turn(15, 0.0, n=N), turn(16, 40.0), turn(17, 40.0)
    x = np.concatenate([first, second, third], axis=1)                 # non-overlapping windows at 3, 6, 9 s
    out = stream.push_block(x)
    assert [k for k, _ in out] == [N, 2 * N, 3 * N]
    mon = HeadMovementMonitor()
    assert out[0][1] == mon.classify(first)                             # nothing before the first window
    assert out[1][1] == mon.classify(second, first)                     # 3 s before the second
    assert out[2][1] == mon.classify(third, x[:, 2 * N - 5 * FS:2 * N])  # the last 5 s before the third


DATA = Path(__file__).resolve().parents[1] / "extracted_npz_dataset_v2" / "sub_01" / "ses-01" / "clean_epochs.npz"


@pytest.mark.skipif(not DATA.exists(), reason="dataset not present")
def test_rebuilt_history_continues_each_stored_window_exactly():
    from monitoring.stream_history import windows_with_history
    d = np.load(DATA, allow_pickle=True)
    epochs = d["epochs"][:, :8] * 1e6
    key = {(str(t), int(r), int(o)): k for k, (t, r, o) in enumerate(zip(d["task"], d["run"], d["run_offset"]))}
    checked = 0
    for k, window, history in windows_with_history(DATA):
        if history is None or history.shape[1] < 60:
            continue
        o = int(d["run_offset"][k])
        for j in range(1, 60):
            earlier = key.get((str(d["task"][k]), int(d["run"][k]), o - j))
            if earlier is not None:
                stream = np.concatenate([history, window], axis=1)
                seg = stream[:, stream.shape[1] - N - j:stream.shape[1] - j]
                assert np.allclose(seg - seg.mean(axis=1, keepdims=True), epochs[earlier], atol=1e-6)
                checked += 1
                break
        if checked >= 25:
            break
    assert checked >= 10
