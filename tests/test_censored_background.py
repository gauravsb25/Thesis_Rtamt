"""Tests for the CENSORED stream background (2026-10-07).

  * an all-True mask, or no mask, is exactly the original stream-background rule;
  * a marked (already judged) movement in the history is left out of the background, so a turn
    after it is a movement again, while the unmasked rule calls it STRAIGHT;
  * the censored background equals an independent numpy implementation of the rule;
  * censoring is OFF by default (it raised real single-run false alarms 348 -> 1214 on sub_01);
  * switched on, StreamMonitor and RowStreamMonitor mark the loud 0.5-s blocks of windows they judged
    LEFT/RIGHT/DOWN and pass the longer history plus the mask; their verdicts equal the window monitor
    given the same inputs;
  * with background_censor=False both stream monitors behave exactly as before.
"""
import numpy as np
import pytest
from scipy.signal import butter, lfilter

from monitoring import HeadMovementMonitor, RowStreamMonitor, StreamMonitor
from monitoring.rtamt_signals import (CHANNELS, FS, InvalidWindow, RtamtConfig, background_positions,
                                      compute_signals, loud_samples, stream_background)

N = 3 * FS
I = {c: k for k, c in enumerate(CHANNELS)}
OFF = RtamtConfig(background_censor=False)
ON = RtamtConfig(background_censor=True)


def turn(seed=0, amplitude=40.0, event=1.2, noise=5.0, n=N):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, noise, size=(len(CHANNELS), n))
    if amplitude:
        t = np.arange(n) / FS - event
        shape = amplitude * (np.exp(-((t - 0.2) ** 2) / (2 * 0.08 ** 2)) - 0.6 * np.exp(-((t - 0.7) ** 2) / (2 * 0.25 ** 2)))
        for name, gain in (("T3", 1.0), ("T4", -1.0), ("F3", 0.5), ("F4", -0.5)):
            x[I[name]] += gain * shape
    return x


def loud(seed):
    return np.concatenate([turn(seed, 40.0, event=0.6, n=N // 2), turn(seed + 1, 40.0, event=0.6, n=N // 2)], axis=1)


def independent_background(window, history, usable):
    """The censored rule written out directly: newest 4 s of block ends whose block + 1 s guard is clean."""
    b, a = butter(2, [0.5 / (FS / 2), 3.0 / (FS / 2)], btype="band")
    s = lfilter(b, a, np.concatenate([history, window], axis=1), axis=-1)
    o = history.shape[1]
    e = np.full(s.shape, np.nan)
    for t in range(127, s.shape[1]):
        e[:, t] = np.mean(s[:, t - 127:t + 1] ** 2, axis=1)
    ends = [t for t in range(o - 8, FS - 1, -8) if usable[max(0, t - 127 - FS):t + 1].all()][:4 * FS // 8]
    ref = np.median(e[:, sorted(ends)], axis=1) + 1e-12
    return np.mean(e[:, o:o + N] / ref[:, None], axis=0)


def test_no_mask_and_an_all_true_mask_are_the_original_rule():
    x, h = turn(1), turn(2, 0.0, n=5 * FS)
    a, va = stream_background(x, h)
    b, vb = stream_background(x, h, usable=np.ones(h.shape[1], bool))
    assert np.array_equal(a, b) and np.array_equal(va, vb)
    assert np.array_equal(background_positions(5 * FS), np.arange(FS, 5 * FS, 8))


def test_a_marked_movement_is_left_out_of_the_background():
    mon = HeadMovementMonitor()
    x = turn(3)
    history = np.concatenate([turn(4, 0.0, n=5 * FS), loud(5)], axis=1)          # quiet 5 s, then 3 s judged a movement
    usable = np.r_[np.ones(5 * FS, bool), np.zeros(N, bool)]
    assert mon.classify(x, history[:, -5 * FS:]) == "STRAIGHT"                    # the old rule: compared with the movement
    assert mon.classify(x, history, history_usable=usable) in ("LEFT", "RIGHT")   # censored: compared with rest
    got, valid = stream_background(x, history, usable=usable)
    assert valid.all()
    ref = independent_background(x, history, usable)
    ok = np.isfinite(ref)
    assert np.allclose(got[ok], ref[ok], rtol=1e-9)


def test_too_little_clean_history_falls_back_to_the_contrast_gate():
    x = turn(6)
    history = np.concatenate([turn(7, 0.0, n=FS), loud(8)], axis=1)               # 1 s quiet (all settle), 3 s marked
    usable = np.r_[np.ones(FS, bool), np.zeros(N, bool)]
    b, valid = stream_background(x, history, usable=usable)
    assert not valid.any() and not b.any()


def test_mask_of_the_wrong_length_is_invalid():
    x, h = turn(9), turn(10, 0.0, n=5 * FS)
    with pytest.raises(InvalidWindow):
        compute_signals(x, history=h, history_usable=np.ones(10, bool))
    assert HeadMovementMonitor().classify(x, h, history_usable=np.ones(10, bool)) == "INVALID"


def test_memory_shorter_than_the_background_is_rejected():
    with pytest.raises(ValueError):
        RtamtConfig(background_memory_seconds=2.0)


def stream_input():
    quiet, first, second = turn(11, 0.0, n=2 * N), turn(12), turn(13)
    return quiet, first, second, np.concatenate([quiet, first, second], axis=1)


def expected_mask(first, history):
    """Usable flags after the first turn: its loud 0.5-s blocks are censored."""
    loud = loud_samples(HeadMovementMonitor(ON).signals(first, history), ON)
    assert loud.any() and not loud.all()
    return np.r_[np.ones(history.shape[1], bool), ~loud]


def test_censoring_is_off_by_default():
    assert RtamtConfig().background_censor is False
    assert StreamMonitor().memory == StreamMonitor().history == 5 * FS


def test_loud_samples_cover_each_loud_block():
    s = HeadMovementMonitor().signals(turn(14), turn(15, 0.0, n=5 * FS))
    loud = loud_samples(s)
    ends = np.flatnonzero(s["background"] >= RtamtConfig().background_ratio)
    want = np.zeros(N, bool)
    for t in ends:
        want[max(0, t - 127):t + 1] = True
    assert np.array_equal(loud, want) and loud.any()


def test_stream_monitor_marks_a_movement_and_passes_the_mask():
    quiet, first, second, x = stream_input()
    mon = HeadMovementMonitor(ON)
    out = StreamMonitor(ON).push_block(x)
    assert out[2][1] in ("LEFT", "RIGHT")                                         # the first turn is judged a movement
    usable = expected_mask(first, x[:, 2 * N - 5 * FS:2 * N] if 2 * N >= 5 * FS else x[:, :2 * N])
    usable = np.r_[np.ones(3 * N - len(usable), bool), usable]
    assert out[3][1] == mon.classify(second, x[:, :3 * N], history_usable=usable)
    old = StreamMonitor(OFF).push_block(x)
    assert old[3][1] == HeadMovementMonitor(OFF).classify(second, x[:, 3 * N - 5 * FS:3 * N])
    assert StreamMonitor(ON).memory == 21 * FS


def test_row_monitor_matches_the_window_monitor_under_censoring():
    quiet, first, second, x = stream_input()
    ends = [v for v in RowStreamMonitor(ON, decide_every=128).push_block(x) if v.window_complete]
    assert ends[2].label in ("LEFT", "RIGHT")
    usable = expected_mask(first, x[:, 2 * N - 5 * FS:2 * N])
    usable = np.r_[np.ones(3 * N - len(usable), bool), usable]
    mon = HeadMovementMonitor(ON)
    ref = mon.robustness(second, x[:, :3 * N], history_usable=usable)
    assert ends[3].label == mon.classify(second, x[:, :3 * N], history_usable=usable)
    for c in ref:
        assert ends[3].robustness[c] == pytest.approx(ref[c], abs=1e-9)


def test_without_censoring_the_row_monitor_is_unchanged():
    quiet, first, second, x = stream_input()
    ends = [v for v in RowStreamMonitor(OFF, decide_every=128).push_block(x) if v.window_complete]
    ref = HeadMovementMonitor(OFF).robustness(second, x[:, 3 * N - 5 * FS:3 * N])
    for c in ref:
        assert ends[3].robustness[c] == pytest.approx(ref[c], abs=1e-9)
