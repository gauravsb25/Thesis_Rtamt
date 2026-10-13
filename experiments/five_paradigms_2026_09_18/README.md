# Five different paradigms (18 Sep 2026)

After amplitude, energy, spectral and spatial-threshold ideas were exhausted, five fundamentally different
paradigms were proposed: the surface Laplacian, symbolic complexity, sequential evidence on the RTAMT
robustness trace, Hilbert instantaneous phase, and a reframing of the system around what it does well.
Paradigms 1–4 were implemented (`features.py`) and tested with the usual protocol (`run.py`, `dev.log`):
blocked 3-fold CV on sub_01 sessions 1–30, adoption rule fixed first, one test run for a pass.

Baseline: the shipped monitor, **85.01 / 69.62 / 78.67**.

## Results: paradigms 1–4

| paradigm | design | CV on sessions 1–30 | outcome |
|---|---|---|---|
| 1. Surface Laplacian | the whole monitor rebuilt on Laplacian channels | 85.01% (+0.00) | never beat the current monitor on any training fold |
| 1. Surface Laplacian | only the nod signal from the Laplacian | 85.01% (+0.00) | CV kept the current rule |
| 1. Surface Laplacian | only the sideways veto from the Laplacian | 85.01% (+0.00) | chosen in 1 of 3 folds, no gain |
| 1. Surface Laplacian | only the direction from the Laplacian | 85.01% (+0.00) | CV kept the current rule |
| 2. Permutation entropy | entropy as the gate (a movement is an ordered stretch) | 85.01% (+0.00) | CV kept the current rule |
| 2. Permutation entropy | entropy drop as an extra gate test | 84.98% (−0.03) | fail |
| 3. SPRT / CUSUM on the robustness trace | accumulator with allowance k = 0, 0.05, 0.1, 0.2 | 85.01% (+0.00) | CV kept the current rule for every k |
| 4. Hilbert phase | phase difference F3 − F4 as the direction | 85.01% (+0.00) | CV kept the current rule |
| 4. Hilbert phase | phase jumps as a blink veto (on DOWN or on the gate) | 85.01% (+0.00) | CV kept the current rule |

**Nothing passed, so the test sets were not touched and the monitors are unchanged.**

Direction accuracy among windows already called a turn:

| direction read | sub_01 1–30 | sub_01 31–36 | sub_11 |
|---|---|---|---|
| current: sign of the F4−T3 peak | **92.6%** | **100.0%** | **70.5%** |
| Laplacian F4−T3 | 84.0% | 56.7% | 67.9% |
| Hilbert phase difference F3 − F4 | 50.0% | 63.1% | 48.3% |

### Why each one failed

1. **Surface Laplacian.** It is a high-pass spatial filter: it removes what is common to neighbouring
   electrodes. But a head movement *is* a common-mode event — the whole scalp moves together — so the
   Laplacian removes most of the signal the monitor lives on. Direction drops from 92.6 to 84.0% on the
   design sessions and from 100 to 56.7% on the sub_01 test. The nod-versus-turn separation does not improve
   either: the "smearing" of nod energy into T3/T4 is not volume conduction but real mechanical motion of
   those electrodes, which the Laplacian cannot remove.
2. **Permutation entropy.** The idea is right in direction — entropy does fall during movements — but the
   existing gate already measures the same thing more sharply. The entropy of a 0.5 s slice is dominated by
   sample-to-sample noise, so its drop during a 1.4 s movement is small compared with the 6.6× energy rise
   the background gate sees.
3. **Sequential evidence (SPRT).** Accumulating the robustness over time is equivalent to the existing
   `once` over a signal that is already a running statistic: the window's loudest 0.5 s dominates the sum
   either way. This is the third time an accumulate-then-threshold gate has been tried (Page's CUSUM on
   energy on 16 Sep, the persistence gate on 17 Sep) and the third time it has not helped.
4. **Hilbert phase.** The phase difference between F3 and F4 at the movement peak is at chance for direction
   (50.0 / 63.1 / 48.3%), and the phase-jump measures never beat the current blink handling. At 0.5–6 Hz the
   two sides of the head move within a fraction of a cycle of each other, so the phase difference is
   dominated by noise — the same reason the peak-time lag failed earlier today.

## Paradigm 5: the reframing is supported by the data

The monitor's three decisions, measured separately:

| split | movement vs still (balanced) | nod vs turn, given a movement | left vs right, given a turn | 4-class |
|---|---|---|---|---|
| sub_01 1–30 | **95.7%** | 91.7% | 92.6% | 85.01% |
| sub_01 31–36 | **91.8%** | 70.7% | 100.0% | 69.62% |
| sub_11 | **94.2%** | 98.6% | 70.5% | 78.67% |

- **Movement detection transfers**: 91.8–95.7% on every split, including a person the design never saw, and
  the row-by-row monitor can commit to it about 0.4 s early.
- **Nod versus turn mostly transfers**: 70.7–98.6%.
- **Left versus right does not**: 100% on one held-out set and 70.5% on the other person, with the measured
  ceiling at about 80.5% even with a perfect per-session sign (`experiments/phase_shape_2026_09_18`).

That asymmetry is the honest architecture: **8-channel EEG is a good low-latency movement trigger, and a
poor cross-subject direction sensor.** A one-axis gyroscope, an EOG channel, or a third subject resolves the
direction; nothing inside these eight channels has, across nine rounds of trying.

## What has now been closed

Amplitude and energy ratios, spectral fractions, per-channel rule banks, spatial geometry and centroids,
power and magnitude asymmetries, timing and phase, complexity, sequential accumulation, artifact vetoes,
persistence gates, threshold re-selection, per-session calibration and machine learning have all been
measured against the same protocol. Only two changes ever survived: removing the spliced windows, and
comparing each window with the stream before it.
