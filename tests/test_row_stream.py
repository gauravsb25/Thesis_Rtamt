"""Tests for the row-by-row monitor (monitoring/row_stream.py).

  * at row 768 its verdict and robustness equal HeadMovementMonitor on the same window;
  * it is causal: a verdict at row t does not change when later rows change;
  * it gives a verdict at every row (decide_every=1) or every k rows;
  * early commitment is an RTAMT historically[] specification and never commits STRAIGHT early;
  * unusable input raises instead of producing a label.
"""
import numpy as np
import pytest

from monitoring import HeadMovementMonitor, RowStreamMonitor, row_formulas
from monitoring.rtamt_signals import CHANNELS, FS, InvalidWindow, RtamtConfig

N = 3 * FS
I = {c: k for k, c in enumerate(CHANNELS)}


def window(seed=0, event=1.2, polarity=1.0, nod=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 5.0, size=(len(CHANNELS), N))
    if event is not None:
        t = np.arange(N) / FS - event
        shape = 40.0 * (np.exp(-((t - 0.2) ** 2) / (2 * 0.08 ** 2)) - 0.6 * np.exp(-((t - 0.7) ** 2) / (2 * 0.25 ** 2)))
        chans = (("Fz", 1), ("Cz", 1), ("F3", 1), ("F4", 1), ("Fp1", 1), ("Fp2", 1)) if nod else \
                (("T3", 1.0), ("T4", -1.0), ("F3", 0.5), ("F4", -0.5))
        for name, gain in chans:
            x[I[name]] += (1 if nod else polarity) * gain * shape
    return x


@pytest.mark.parametrize("kind", [{"event": None}, {}, {"polarity": -1.0}, {"nod": True}])
def test_row_768_equals_the_window_monitor(kind):
    x = window(4, **kind)
    verdicts = RowStreamMonitor(decide_every=64).push_block(x)
    end = verdicts[-1]
    assert end.row == N and end.window_complete
    ref = HeadMovementMonitor().robustness(x)
    assert end.label == HeadMovementMonitor().classify(x)
    for c in ref:
        assert end.robustness[c] == pytest.approx(ref[c], abs=1e-9)


def test_verdicts_are_causal():
    x = window(5)
    y = x.copy()
    y[:, 500:] = np.random.default_rng(9).normal(0, 50, size=(len(CHANNELS), N - 500))
    a = {v.row: v for v in RowStreamMonitor(decide_every=16).push_block(x)}
    b = {v.row: v for v in RowStreamMonitor(decide_every=16).push_block(y)}
    for row in range(16, 497, 16):
        assert a[row].label == b[row].label
        assert a[row].robustness == pytest.approx(b[row].robustness)


def test_a_verdict_every_row_or_every_k_rows():
    x = window(6)
    assert len(RowStreamMonitor(decide_every=1).push_block(x[:, :200])) == 200
    assert [v.row for v in RowStreamMonitor(decide_every=100).push_block(x)] == [100, 200, 300, 400, 500, 600, 700, 768]


def test_early_commitment_uses_a_historically_specification_and_sticks():
    rs = RowStreamMonitor(commit_seconds=0.25, decide_every=8)
    assert all("historically[0:7]" in spec.spec for spec in rs._commit_specs.values())
    verdicts = rs.push_block(window(7))
    committed = [v for v in verdicts if v.committed is not None]
    assert committed, "the synthetic turn should be committed before the window ends"
    first = committed[0]
    assert first.committed in ("LEFT", "RIGHT", "DOWN")
    assert all(v.committed == first.committed for v in committed)
    assert first.committed_at_row <= N


def test_rest_is_never_committed_early():
    verdicts = RowStreamMonitor(commit_seconds=0.25, decide_every=8).push_block(window(8, event=None))
    assert all(v.committed is None for v in verdicts[:-1])
    assert verdicts[-1].committed == verdicts[-1].label


def test_row_formulas_have_no_temporal_operator_and_match_the_design():
    f = row_formulas()
    assert set(f) == {"LEFT", "RIGHT", "DOWN", "STRAIGHT"}
    assert all("once" not in s and "historically" not in s for s in f.values())
    assert "sideways_fp1fp2" in row_formulas(RtamtConfig(down_veto_fp1fp2=7.1))["DOWN"]


def test_new_window_starts_after_row_768():
    rs = RowStreamMonitor(decide_every=256)
    rs.push_block(np.concatenate([window(1), window(2)], axis=1))
    assert rs.rows == N


def test_unusable_input_raises():
    rs = RowStreamMonitor()
    with pytest.raises(ValueError):
        rs.push(np.zeros(3))
    with pytest.raises(InvalidWindow):
        rs.push(np.full(8, np.nan))
    with pytest.raises(NotImplementedError):
        RowStreamMonitor(RtamtConfig(gate_statistic="cusum"))


def test_second_window_uses_the_first_as_stream_background():
    first, second = window(21, event=None), window(22)
    verdicts = RowStreamMonitor(decide_every=64).push_block(np.concatenate([first, second], axis=1))
    end = [v for v in verdicts if v.window_complete][1]
    assert end.signals["background_valid"] == 1.0
    ref = HeadMovementMonitor().robustness(second, first)
    assert end.label == HeadMovementMonitor().classify(second, first)
    for c in ref:
        assert end.robustness[c] == pytest.approx(ref[c], abs=1e-9)


def test_seeded_history_equals_the_window_monitor_with_the_last_five_seconds():
    rng = np.random.default_rng(23)
    history = rng.normal(0.0, 5.0, size=(len(CHANNELS), 6 * FS))
    x = window(24)
    end = RowStreamMonitor(decide_every=128, history=history).push_block(x)[-1]
    ref = HeadMovementMonitor().robustness(x, history[:, -5 * FS:])
    assert end.label == HeadMovementMonitor().classify(x, history[:, -5 * FS:])
    for c in ref:
        assert end.robustness[c] == pytest.approx(ref[c], abs=1e-9)


def test_background_formula_and_clear():
    f = row_formulas()
    assert "(background >= 8.0)" in f["LEFT"] and "(background_valid >= 0.5)" in f["STRAIGHT"]
    assert all("background" not in s for s in row_formulas(RtamtConfig(background_ratio=None)).values())
    rs = RowStreamMonitor(decide_every=256)
    rs.push_block(window(25, event=None))
    rs.clear()
    assert rs.push_block(window(26))[-1].signals["background_valid"] == 0.0
