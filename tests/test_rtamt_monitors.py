"""Tests for the four RTAMT head-movement monitors.

They pin the properties the design depends on:
  * every specification is real STL that RTAMT parses, using past-time operators only;
  * the signals reproduce the physics measurements -- checked here against an independent
    numpy reference, so the tests do not depend on any retired module;
  * the default design (contrast gate, largest-peak direction) is unchanged, and the two options
    tested on 2026-09-16 and not adopted (CUSUM gate, SINCE direction) behave as documented;
  * RTAMT's online monitor (sample by sample, as on a live stream) and its offline evaluator
    agree, including for SINCE;
  * every signal is scale-free, so a per-subject gain cannot move a robustness value;
  * an unusable window returns INVALID instead of raising;
  * the stream wrapper emits its first verdict on exactly the 768th sample.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, lfilter

import monitoring
from monitoring.rtamt_monitors import (
    CLASSES,
    HeadMovementMonitor,
    StreamMonitor,
    class_formulas,
    satisfied_class,
)
from monitoring.rtamt_signals import (
    CHANNELS,
    FS,
    SIGNALS,
    InvalidWindow,
    RtamtConfig,
    check_window,
    compute_signals,
)

N = 3 * FS
I = {c: k for k, c in enumerate(CHANNELS)}
OPTIONS = RtamtConfig(gate_statistic="cusum", direction_rule="since")     # tested, not adopted
EXTRA_VETOES = RtamtConfig(down_veto_fp1fp2=7.1, down_veto_fp2t3=6.6)    # tested, not adopted


def window(seed: int = 0, event: float | None = 1.2, polarity: float = 1.0,
           nod: bool = False) -> np.ndarray:
    """Resting EEG, optionally with one lateral turn or one nod centred at ``event`` seconds."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 5.0, size=(len(CHANNELS), N))
    if event is not None:
        t = np.arange(N) / FS - event
        shape = 40.0 * (np.exp(-((t - 0.2) ** 2) / (2 * 0.08 ** 2))
                        - 0.6 * np.exp(-((t - 0.7) ** 2) / (2 * 0.25 ** 2)))
        if nod:
            for name in ("Fz", "Cz", "F3", "F4", "Fp1", "Fp2"):
                x[I[name]] += shape
        else:
            for name, gain in (("T3", 1.0), ("T4", -1.0), ("F3", 0.5), ("F4", -0.5)):
                x[I[name]] += polarity * gain * shape
    return x


def _b(lo, hi):
    return butter(2, [lo / (FS / 2), hi / (FS / 2)], btype="band")


def _mad(a, axis=-1):
    return 1.4826 * np.median(np.abs(a - np.median(a, axis=axis, keepdims=True)), axis=axis)


def _energy(x):
    band = lfilter(*_b(0.5, 3.0), x, axis=-1)
    return np.mean((band / (_mad(band)[:, None] + 1e-12)) ** 2, axis=0)


def _loudest(x, width):
    c = np.concatenate([[0.0], np.cumsum(_energy(x))])
    s = int(np.argmax(c[width:] - c[:-width]))
    return slice(s, s + width)


def reference(x, cfg: RtamtConfig):
    """The physics measurements, written directly and independently of the module."""
    e = _energy(x)
    c = np.concatenate([[0.0], np.cumsum(e)])
    blocks = c[128:] - c[:-128]
    contrast = blocks.max() / (blocks.min() + 1e-18)

    quiet = (blocks / 128).min()
    smooth = (c[32:] - c[:-32]) / 32
    g, best = 0.0, 0.0
    for value in np.log(smooth + 1e-18) - np.log(quiet + 1e-18) - cfg.cusum_kappa:
        g = max(0.0, g + value)
        best = max(best, g)
    cusum = best / FS

    shares = []
    for s in range(0, N - 128 + 1, 64):
        seg = x[:, s:s + 128] - x[:, s:s + 128].mean(axis=1, keepdims=True)
        p = np.abs(np.fft.rfft(seg * np.hanning(128), axis=1)) ** 2
        f = np.fft.rfftfreq(128, 1 / FS)
        shares.append(np.mean(p[:, (f >= 1) & (f < 4)].sum(1) / (p[:, (f >= .5) & (f <= 45)].sum(1)
                                                                  + 1e-18)))
    delta = max(shares) - min(shares)

    band = lfilter(*_b(0.5, 6.0), x, axis=-1)
    band = band - np.median(band, axis=1, keepdims=True)
    deflect = np.mean(np.abs(band).max(1) / (1.4826 * np.median(np.abs(band), 1) + 1e-12))
    sq = band ** 2
    midline = np.log((sq[[I["Fz"], I["Cz"]]].mean() + 1e-18) / (sq[[I["T3"], I["T4"]]].mean()
                                                                 + 1e-18))
    seg = band[:, _loudest(x, 256)]
    sym = np.log((np.mean((seg[I["F3"]] + seg[I["F4"]]) ** 2) + 1e-18)
                 / (np.mean((seg[I["F3"]] - seg[I["F4"]]) ** 2) + 1e-18))

    pr = lfilter(*_b(0.3, 4.0), x, axis=-1)

    def excursion(a, b):
        v = pr[I[a]] - pr[I[b]]
        v = v - np.median(v)
        return np.abs(v).max() / (1.4826 * np.median(np.abs(v)) + 1e-12)
    sideways = excursion("F4", "T3")
    side_fp1fp2, side_fp2t3 = excursion("Fp1", "Fp2"), excursion("Fp2", "T3")

    dr = lfilter(*_b(0.3, 6.0), x, axis=-1)
    u = (dr[I["F4"]] - dr[I["T3"]])[_loudest(x, 384)]
    u = u - np.median(u)
    peak_direction = -np.sign(u[int(np.argmax(np.abs(u)))]) or 1.0
    big = np.flatnonzero(np.abs(u) >= cfg.direction_level * np.abs(u).max())
    last_direction = -np.sign(u[big[-1]]) or 1.0          # the LAST large excursion decides
    return (contrast, cusum, delta, deflect, midline + sym, sideways, peak_direction, last_direction,
            side_fp1fp2, side_fp2t3)


@pytest.fixture(scope="module")
def monitor():
    return HeadMovementMonitor()


def test_four_specifications_one_per_class():
    assert set(class_formulas()) == set(CLASSES)


def test_package_exposes_exactly_the_four_class_monitors():
    for name in ("StraightMonitor", "DownMonitor", "LeftMonitor", "RightMonitor",
                 "HeadMovementMonitor", "StreamMonitor"):
        assert hasattr(monitoring, name)
    assert monitoring.HeadMovementMonitor is HeadMovementMonitor


def test_specifications_use_only_past_time_operators():
    """RTAMT's online monitor cannot evaluate bounded future operators on a live stream."""
    for cfg in (RtamtConfig(), OPTIONS):
        for formula in class_formulas(cfg).values():
            assert "eventually" not in formula and "always" not in formula
            assert "once" in formula or "historically" in formula


def test_default_design_is_contrast_gate_and_peak_direction():
    f = class_formulas()
    assert "contrast" in f["STRAIGHT"] and "cusum" not in f["STRAIGHT"]
    assert "(turn > 0)" in f["LEFT"] and "(turn < 0)" in f["RIGHT"] and "since" not in f["LEFT"]
    opt = class_formulas(OPTIONS)
    assert "cusum" in opt["STRAIGHT"] and "contrast" not in opt["STRAIGHT"]
    assert "since" in opt["LEFT"] and "since" in opt["RIGHT"] and "turn" not in opt["LEFT"]


def test_extra_nod_vetoes_are_off_by_default_and_can_be_switched_on():
    off = class_formulas()
    assert all("sideways_fp" not in formula for formula in off.values())
    assert "(historically[0:767](sideways <= 7.5))" in off["DOWN"]
    f = class_formulas(EXTRA_VETOES)
    assert "sideways_fp1fp2 <= 7.1" in f["DOWN"] and "sideways_fp2t3 <= 6.6" in f["DOWN"]
    assert "sideways_fp1fp2 > 7.1" in f["LEFT"] and "sideways_fp2t3 > 6.6" in f["RIGHT"]


@pytest.mark.parametrize("change,expected", [
    ({"nod": -0.6, "sideways_fp1fp2": 7.1}, "DOWN"),
    ({"nod": -0.6, "sideways_fp1fp2": 7.1 + 1e-5}, "LEFT"),
    ({"nod": -0.6, "sideways_fp2t3": 6.6}, "DOWN"),
    ({"nod": -0.6, "sideways_fp2t3": 6.6 + 1e-5}, "LEFT"),
])
def test_extra_veto_boundaries_follow_the_declared_semantics(change, expected):
    mon = HeadMovementMonitor(EXTRA_VETOES)
    values = dict(BASE)
    values.update(change)
    signals = {k: np.full(N, v) for k, v in values.items()}
    rho = {c: mon.monitors[c].robustness(signals) for c in CLASSES}
    assert satisfied_class(rho) == expected


def test_invalid_design_names_are_rejected():
    with pytest.raises(ValueError):
        RtamtConfig(gate_statistic="kurtosis")
    with pytest.raises(ValueError):
        RtamtConfig(direction_rule="vote")


def test_every_specification_parses_in_rtamt(monitor):
    signals = compute_signals(window(1))
    for c in CLASSES:
        assert np.isfinite(monitor.monitors[c].robustness(signals))


@pytest.mark.parametrize("cfg", [RtamtConfig(), OPTIONS], ids=["default", "options"])
def test_specifications_partition_the_space(cfg):
    """Exactly one class is satisfied, so the label is never ambiguous."""
    mon = HeadMovementMonitor(cfg)
    for seed in range(4):
        for kind in ({"event": None}, {"event": 1.2}, {"event": 1.2, "nod": True}):
            rho = mon.robustness(window(seed, **kind))
            assert len([c for c in CLASSES if rho[c] > 0]) <= 1
            assert satisfied_class(rho) in CLASSES


@pytest.mark.parametrize("seed", range(3))
def test_signals_reproduce_the_physics_measurements(seed):
    cfg = OPTIONS                 # every signal is computed under any design; OPTIONS sets h
    for x in (window(seed), window(seed, event=None), window(seed, nod=True),
              window(seed, polarity=-1.0)):
        s = compute_signals(x, cfg)
        (contrast, cusum, delta, deflect, nod, sideways, peak_dir, last_dir,
         side_fp1fp2, side_fp2t3) = reference(x, cfg)
        assert s["contrast"].max() == pytest.approx(contrast, rel=1e-9)
        assert s["cusum"].max() == pytest.approx(cusum, rel=1e-9, abs=1e-12)
        assert s["delta"].max() == pytest.approx(delta, abs=1e-12)
        assert s["deflect"][-1] == pytest.approx(deflect, rel=1e-9)
        assert s["nod"][-1] == pytest.approx(nod, abs=1e-9)
        assert s["sideways"].max() == pytest.approx(sideways, rel=1e-9)
        assert s["sideways_fp1fp2"].max() == pytest.approx(side_fp1fp2, rel=1e-9)
        assert s["sideways_fp2t3"].max() == pytest.approx(side_fp2t3, rel=1e-9)
        assert np.sign(s["turn"][-1]) == peak_dir
        lat = s["lateral"]
        idx = np.arange(N)
        last_left = np.max(np.where(lat <= -cfg.direction_level, idx, -1))
        last_right = np.max(np.where(lat >= cfg.direction_level, idx, -1))
        assert (1.0 if last_left > last_right else -1.0) == last_dir


def test_cusum_is_zero_before_its_first_block_and_never_negative():
    s = compute_signals(window(2))
    assert np.all(s["cusum"][:31] == 0) and np.all(s["cusum"] >= 0)


def test_cusum_rewards_sustained_energy_over_a_brief_spike():
    """Same peak amplitude: a 1 s deflection accumulates far more CUSUM than a 0.1 s one."""
    rng = np.random.default_rng(7)
    base = rng.normal(0.0, 5.0, size=(len(CHANNELS), N))
    t = np.arange(N) / FS
    brief, sustained = base.copy(), base.copy()
    brief += 60.0 * np.exp(-((t - 1.5) ** 2) / (2 * 0.03 ** 2))
    sustained += 60.0 * np.exp(-((t - 1.5) ** 2) / (2 * 0.30 ** 2))
    assert compute_signals(sustained)["cusum"].max() > 3 * compute_signals(brief)["cusum"].max()


def test_lateral_is_zero_outside_the_loudest_stretch_and_reaches_one():
    s = compute_signals(window(3))
    lat = s["lateral"]
    assert np.isclose(np.abs(lat).max(), 1.0)
    assert np.count_nonzero(lat) <= int(RtamtConfig().direction_seconds * FS)


BASE = dict(cusum=0.2, contrast=10.0, delta=.4, deflect=4.0, nod=-2.0, sideways=5.0, turn=1.0,
            lateral=-1.0, sideways_fp1fp2=3.0, sideways_fp2t3=3.0,
            background=0.0, background_valid=0.0)          # no stream history: the contrast gate decides


@pytest.mark.parametrize("change,expected", [
    ({"contrast": 5.0}, "LEFT"),
    ({"contrast": 5.0 - 1e-5}, "STRAIGHT"),
    ({"nod": -0.6, "sideways": 7.5}, "DOWN"),
    ({"nod": -0.6, "sideways": 7.5 + 1e-5}, "LEFT"),
    ({"nod": -0.6, "sideways_fp1fp2": 50.0}, "DOWN"),      # optional vetoes are off by default
    ({"turn": -1.0}, "RIGHT"),
    ({"cusum": 0.0, "lateral": 1.0}, "LEFT"),          # the default design ignores both
])
def test_threshold_boundaries_follow_the_declared_semantics(monitor, change, expected):
    """Zero robustness at a >= threshold passes; a strict < fails."""
    values = dict(BASE)
    values.update(change)
    signals = {k: np.full(N, v) for k, v in values.items()}
    rho = {c: monitor.monitors[c].robustness(signals) for c in CLASSES}
    assert satisfied_class(rho) == expected


@pytest.mark.parametrize("change,expected", [
    ({"cusum": 0.0684}, "LEFT"),
    ({"cusum": 0.0684 - 1e-5}, "STRAIGHT"),
    ({"nod": -0.6, "sideways": 7.5}, "DOWN"),
    ({"lateral": 1.0}, "RIGHT"),
    ({"contrast": 1.0, "turn": -1.0}, "LEFT"),        # the options ignore both
])
def test_option_boundaries_follow_the_declared_semantics(change, expected):
    mon = HeadMovementMonitor(OPTIONS)
    values = dict(BASE)
    values.update(change)
    signals = {k: np.full(N, v) for k, v in values.items()}
    rho = {c: mon.monitors[c].robustness(signals) for c in CLASSES}
    assert satisfied_class(rho) == expected


@pytest.mark.parametrize("lobes,expected", [
    ((-1.00, +0.97), "RIGHT"),   # a near tie: the later lobe decides
    ((+1.00, -0.97), "LEFT"),
    ((-1.00, +0.50), "LEFT"),    # one lobe dominates: same answer as the largest-peak rule
    ((+0.50, -1.00), "LEFT"),
    ((+1.00, -0.50), "RIGHT"),
])
def test_since_direction_lets_the_last_large_excursion_decide(lobes, expected):
    monitor = HeadMovementMonitor(OPTIONS)
    values = dict(BASE)
    signals = {k: np.full(N, v) for k, v in values.items()}
    lat = np.zeros(N)
    lat[200:230] = lobes[0]
    lat[400:430] = lobes[1]
    signals["lateral"] = lat
    rho = {c: monitor.monitors[c].robustness(signals) for c in CLASSES}
    assert satisfied_class(rho) == expected


@pytest.mark.parametrize("cfg", [RtamtConfig(), OPTIONS], ids=["default", "options"])
def test_lateral_polarity_decides_left_versus_right(cfg):
    mon = HeadMovementMonitor(cfg)
    a = mon.classify(window(3, polarity=1.0))
    b = mon.classify(window(3, polarity=-1.0))
    assert {a, b} <= {"LEFT", "RIGHT"} and a != b


@pytest.mark.parametrize("cfg", [RtamtConfig(), OPTIONS], ids=["default", "options"])
def test_online_monitor_matches_offline_evaluator(cfg):
    offline, online = HeadMovementMonitor(cfg), HeadMovementMonitor(cfg, online=True)
    for x in (window(5), window(5, polarity=-1.0)):
        a, b = offline.robustness(x), online.robustness(x)
        for c in CLASSES:
            assert a[c] == pytest.approx(b[c], abs=1e-9)
        assert offline.classify(x) == online.classify(x)


@pytest.mark.parametrize("cfg", [RtamtConfig(), OPTIONS], ids=["default", "options"])
@pytest.mark.parametrize("gain", [0.01, 3.0, 250.0])
def test_robustness_is_invariant_to_a_global_gain(cfg, gain):
    monitor = HeadMovementMonitor(cfg)
    x = window(2)
    base, scaled = monitor.robustness(x), monitor.robustness(x * gain)
    for c in CLASSES:
        assert scaled[c] == pytest.approx(base[c], rel=1e-6, abs=1e-9)


def test_stream_emits_first_verdict_on_exactly_the_768th_sample():
    stream = StreamMonitor()
    x = np.concatenate([window(4), window(4)], axis=1)
    out = stream.push_block(x)
    assert out[0][0] == N
    assert [k for k, _ in out] == [N, 2 * N]          # default: non-overlapping 3 s windows


def test_stream_verdict_equals_the_window_verdict(monitor):
    x = window(6)
    assert StreamMonitor().push_block(x) == [(N, monitor.classify(x))]


def test_dead_channel_is_invalid_not_a_crash(monitor):
    x = window(0)
    x[I["T4"]] = 0.0
    assert monitor.classify(x) == "INVALID"


def test_non_finite_sample_is_invalid_not_a_crash(monitor):
    x = window(0)
    x[2, 10] = np.nan
    assert monitor.classify(x) == "INVALID"


def test_check_window_rejects_wrong_shape():
    with pytest.raises(InvalidWindow):
        check_window(np.zeros((3, N)))
    with pytest.raises(InvalidWindow):
        check_window(np.zeros((len(CHANNELS), N // 2)))


def test_signal_names_are_the_declared_set():
    assert set(compute_signals(window(0))) == set(SIGNALS)


def test_auxiliary_directions_cannot_change_monitor_decisions(monitor):
    from monitoring.rtamt_signals import AUXILIARY_SIGNALS, CORE_SIGNALS
    signals = compute_signals(window(3))
    baseline = {c: m.robustness(signals) for c, m in monitor.monitors.items()}
    for name in AUXILIARY_SIGNALS:
        assert signals[name].shape == (N,)
        assert np.isfinite(signals[name]).all()
        assert np.all(signals[name][:383] == 0)
        signals[name] = np.full(N, -1e9)
    for c, m in monitor.monitors.items():
        assert set(m.variables) <= set(CORE_SIGNALS)
        assert m.robustness(signals) == baseline[c]
