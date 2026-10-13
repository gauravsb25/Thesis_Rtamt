# Calibration ceiling, polarity-free direction, and wave shape (18 Sep 2026)

Three questions were asked: how much would one known turn per session buy, can timing (lead/lag) replace
voltage polarity for direction, and can wave shape separate nods from turns without spatial energy ratios.

Baseline: the shipped monitor, **85.01 / 69.62 / 78.67**. Code: `features.py`, `run.py`; log `dev.log`.

## A. What per-session calibration would buy — the answer is: almost nothing

A session's sign is set either by its **first turn** (a realistic calibration; that movement is then excluded
from scoring) or by the **oracle**: the sign that is best for that session, which is the ceiling no
calibration can beat.

| split | monitor now | 1-turn calibration | oracle sign (ceiling) |
|---|---|---|---|
| sub_01 1–30 | 84.49% | 80.53% | 84.49% |
| sub_01 31–36 | 59.07% | 59.07% | 59.07% |
| sub_11 | 78.65% | 70.06% | **80.49%** |

Direction accuracy among the windows the monitor calls a turn:

| split | now | 1-turn | oracle |
|---|---|---|---|
| sub_01 1–30 | 92.2% | 83.2% | 92.2% |
| sub_01 31–36 | 100.0% | 100.0% | 100.0% |
| sub_11 | 73.5% | 53.9% | 77.5% |

(The baselines here differ slightly from the headline because the calibration movement of each session is
removed from scoring for all three columns.)

**Findings**

1. **Only 2 of 46 sessions are actually sign-inverted**: sub_11 ses-08 and ses-09. Every sub_01 session is
   correct as it is, which is why the oracle changes nothing for sub_01.
2. **Even a perfect per-session sign buys only +1.8 points on sub_11** (78.65 → 80.49) and +4 points of
   direction accuracy (73.5 → 77.5%). So the inversion is **not** the main defect: most of sub_11's
   direction errors are single windows inside sessions whose overall sign is already right.
3. **A realistic 1-turn calibration is worse than doing nothing** (sub_11 70.06%, direction 53.9%). The
   calibration turn is itself misread often enough that it flips sessions that were correct. Calibrating on
   one movement is unreliable; it would need several known turns, and even then the ceiling is +1.8.

**Conclusion:** per-session calibration is not worth breaking the "no per-session constants" rule for.

## B. Direction from timing instead of polarity — the signal is not there

Direction accuracy among turn windows, orientation fixed on sessions 1–30, no other fitting:

| direction rule | sub_01 1–30 | sub_01 31–36 | sub_11 |
|---|---|---|---|
| current: sign of the F4−T3 peak | 92.6% | 100.0% | 70.5% |
| peak-time difference F3 − F4 | 51.1% | 47.6% | 56.1% |
| peak-time difference T3 − T4 | 59.3% | 26.7% | 49.6% |
| envelope alignment lag, left against right | 59.1% | 53.5% | 49.3% |

All three are at or near chance everywhere, including on the design sessions where polarity works at 92.6%.
The median alignment quality of the envelope cross-correlation is 0.49, i.e. the two sides do not have a
sharp common shape to align. As rules inside the monitor:

| design | CV on sessions 1–30 | outcome |
|---|---|---|
| side from F3−F4 peak time | 85.01% (+0.00) | CV kept the current rule |
| side from T3−T4 peak time | 85.01% (+0.00) | CV kept the current rule |
| side from the envelope lag | 85.01% (+0.00) | CV kept the current rule |
| timing only when the polarity read is weak (tie-break) | 84.33% (−0.68) | fail |

At 256 Hz one sample is 3.9 ms, and the two sides of a head turn move within a few tens of milliseconds of
each other, so the peak time is dominated by noise rather than by the movement order.

## C. Nods against turns by wave shape — no separation

Signals from the pooled envelope of the loudest second: `rise` (10% → 90% before the peak), `fall` (peak →
10% after), `asym` = (fall − rise)/(fall + rise), `steep` = largest slope ÷ peak, `width` at half height.
Each was tried as an extra condition on DOWN, in both directions (≤ and ≥ a threshold).

| signal | CV on sessions 1–30 |
|---|---|
| asym | 84.94% (−0.07) |
| rise | 84.94% (−0.07) |
| fall | 83.87% (−1.14) |
| steep | 85.01% (+0.00, CV kept the current rule) |
| width | 85.01% (+0.00, CV kept the current rule) |

Nods and turns have the same envelope shape in these recordings; what differs is *where* on the head the
energy sits, which is what the existing `nod` and `sideways` signals already measure.

## Overall

Nothing here is adopted; the monitors are unchanged. The useful result is the ceiling: **fixing every
session's sign perfectly would raise sub_11 from 78.67 to about 80.5**, so sub_11's LEFT/RIGHT problem is
mostly window-level, not a per-session flip. Improving it needs a different kind of information (a third
subject to check against, or a motion or eye sensor), not a cleverer rule over the same eight channels.
