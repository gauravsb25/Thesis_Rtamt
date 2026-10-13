"""The threshold report must score the rules exactly as the RTAMT monitor does, and its
selection rule must behave as documented."""
import json

import numpy as np
import pytest

from monitoring.rtamt_monitors import HeadMovementMonitor
from monitoring.rtamt_signals import CHANNELS, FS, RtamtConfig
from monitoring.threshold_report import (
    decide, load_thresholds, measure_window, parse_spec, save_thresholds, select_thresholds,
    thresholds_of,
)

N = 3 * FS
I = {c: k for k, c in enumerate(CHANNELS)}


def window(seed, event=1.2, polarity=1.0, nod=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 5.0, size=(len(CHANNELS), N))
    if event is not None:
        t = np.arange(N) / FS - event
        shape = 40.0 * (np.exp(-((t - 0.2) ** 2) / (2 * 0.08 ** 2))
                        - 0.6 * np.exp(-((t - 0.7) ** 2) / (2 * 0.25 ** 2)))
        chans = (("Fz", 1), ("Cz", 1), ("F3", 1), ("F4", 1)) if nod else \
                (("T3", 1.0), ("T4", -1.0), ("F3", 0.5), ("F4", -0.5))
        for name, gain in chans:
            x[I[name]] += (1 if nod else polarity) * gain * shape
    return x


@pytest.mark.parametrize("cfg", [RtamtConfig(), RtamtConfig(gate_statistic="cusum", direction_rule="since"),
                                 RtamtConfig(down_veto_fp1fp2=7.1, down_veto_fp2t3=6.6)],
                         ids=["default", "options", "extra_vetoes"])
def test_report_decisions_equal_the_rtamt_monitor(cfg):
    mon = HeadMovementMonitor(cfg)
    th = thresholds_of(cfg)
    for seed in range(3):
        for kind in ({"event": None}, {}, {"nod": True}, {"polarity": -1.0}):
            x = window(seed, **kind)
            assert decide(np.array([measure_window(x, cfg)]), **th)[0] == mon.classify(x)


CONTRAST = RtamtConfig(down_veto_fp1fp2=None, down_veto_fp2t3=None)   # toy rows: contrast gate, one veto


def _toy(rest_hi, move_lo):
    """Rest contrast in [rest_hi-5, rest_hi], movement contrast in [move_lo, move_lo+20]."""
    rng = np.random.default_rng(0)
    rows, labels, names = [], [], []
    for s in range(6):
        for lab in ("STRAIGHT", "LEFT", "RIGHT", "DOWN"):
            for _ in range(40):
                moving = lab != "STRAIGHT"
                contrast = move_lo + rng.uniform(0, 20) if moving else rest_hi - rng.uniform(0, 5)
                nod = 2.0 if lab == "DOWN" else -3.0
                turn = 1.0 if lab == "LEFT" else -1.0
                rows.append((contrast, 0.5, 4.0, nod, 3.0, turn))
                labels.append(lab); names.append(f"s{s}")
    return np.array(rows), np.array(labels), np.array(names)


def test_selection_moves_a_threshold_the_data_clearly_rejects():
    values, labels, names = _toy(rest_hi=19.0, move_lo=21.0)   # start g = 5.0 is far too low
    th, _ = select_thresholds(values, labels, names, thresholds_of(CONTRAST))
    assert 19.0 < th["g"] <= 21.0
    assert (decide(values, **th) == labels).mean() == 1.0


def test_selection_keeps_a_threshold_that_is_already_fine():
    values, labels, names = _toy(rest_hi=4.5, move_lo=6.0)     # 5.0 already separates them
    start = thresholds_of(CONTRAST)
    th, _ = select_thresholds(values, labels, names, start)
    assert th == start


def test_parse_spec_and_saved_thresholds_round_trip(tmp_path):
    files = parse_spec("sub_01:1-3")
    assert [p.parent.name for p in files] == ["ses-01", "ses-02", "ses-03"]
    th = dict(g=8.0, d=0.2, q=2.7, n=-0.7, v=7.5)
    save_thresholds(tmp_path / "t.json", th, {"trained_on": ["x"]})
    assert thresholds_of(load_thresholds(tmp_path / "t.json")) == th
    assert json.loads((tmp_path / "t.json").read_text())["trained_on"] == ["x"]
