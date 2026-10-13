# The 15 proposed methods, all implemented and measured (18 Sep 2026)

Fifteen methods were proposed in five groups: subject-adaptive normalisation, polarity-free direction,
nod-against-turn separation, artifact rejection, and STL structure. All fifteen were implemented as
unit-free signals (`features.py`) and tried as monitor changes (`run.py`), with the project's protocol:
blocked 3-fold CV on sub_01 sessions 1–30, adoption rule fixed first (CV gain ≥ +0.5 points and more
sessions better than worse), one test run only for what passes.

Baseline: the shipped monitor, **85.01 / 69.62 / 78.67**. Log: `dev.log`.

## Results

| # | method | what it changes | CV on sessions 1–30 | gain | sessions +/− |
|---|---|---|---|---|---|
| 1 | history median/IQR normalisation (`hist_z`) | gate | 85.02% | +0.01 | 9 / 1 |
| 2 | variance share, temporal group | DOWN | 83.94% | −1.07 | 0 / 3 |
| 2 | variance share, midline group | DOWN | 85.05% | +0.05 | 2 / 1 |
| 2 | largest single-channel share | DOWN | 85.01% | +0.00 | — |
| 3 | 1–4 Hz share of 0.5–30 Hz (largest slice) | gate | 85.00% | −0.01 | 9 / 1 |
| 3 | the same as a range across slices | gate | 85.01% | +0.00 | 3 / 0 |
| 4 | hemisphere power asymmetry | side | 85.01% | +0.00 | — |
| 5 | zero-lag correlation F3/F4 as a side | side | 85.01% | +0.00 | — |
| 5 | zero-lag correlation as a DOWN condition (F3/F4, T3/T4) | DOWN | 84.11% / 84.95% | −0.90 / −0.06 | 1/3, 2/2 |
| 6 | midline excursion imbalance | side | 85.01% | +0.00 | — |
| 7 | lateral energy centroid (and its shift) | side | 85.01% | +0.00 | — |
| 8 | common-mode against differential energy | DOWN | 83.97% | −1.04 | 0 / 3 |
| 9 | midline coherence ÷ temple coherence | DOWN | 84.65% | −0.36 | 3 / 4 |
| 10 | front-to-centre velocity gradient | DOWN | **85.27%** | **+0.26** | 2 / 1 |
| 11 | anterior-to-central decay (whole window / loudest 1 s) | DOWN | 84.74% / 85.01% | −0.27 / +0.00 | — |
| 12 | spectral flatness | gate | 84.84% | −0.17 | 1 / 1 |
| 13 | curvature against velocity | gate | 84.72% | −0.29 | 0 / 1 |
| 14 | time-bounded persistence gate | gate | (measured on 17 Sep) 85.54% at best | +0.53 | 11 / 17 |
| 15 | robustness-gated soft veto | DOWN | 85.01% | +0.00 | — |

Combinations of the best ones do not pass either:

| combination | CV | gain | sessions +/− |
|---|---|---|---|
| 1 + 10 | 85.28% | +0.27 | 9 / 2 |
| 1 + 2(midline) + 10 | 85.28% | +0.27 | 9 / 2 |
| 10 + 9 | 85.27% | +0.26 | 2 / 1 |
| 1 + 3 | 84.97% | −0.03 | 12 / 2 |

**Nothing reached the adoption rule, so the test sets were not touched and the monitors are unchanged.**

## The most informative measurement: polarity-free direction

Direction accuracy among the windows the monitor already calls a turn, with each signal's orientation fixed
on sub_01 sessions 1–30:

| direction read | sub_01 1–30 | sub_01 31–36 | sub_11 |
|---|---|---|---|
| current: sign of the F4−T3 peak | **92.6%** | **100.0%** | **70.5%** |
| hemisphere power asymmetry (method 4) | 57.5% | 54.5% | **5.0%** |
| zero-lag correlation F3/F4 (method 5) | 50.8% | 69.5% | 53.0% |
| midline excursion imbalance (method 6) | 53.0% | 35.3% | 47.3% |
| lateral centroid shift (method 7) | 55.2% | 57.8% | 16.7% |
| lateral centroid (method 7) | 59.0% | 57.2% | **5.0%** |

The two power-based reads are **5% correct on sub_11** — that is not noise, it is the opposite answer almost
every time. Orient them on sub_11 instead and they would be 95% right, while dropping to about 43% on sub_01.
This repeats the magnitude-imbalance result of the previous round: **the two subjects are mirror images in
every magnitude-based measure.** Power is indeed immune to a sign flip, but the quantity it measures points
the other way for the other person, so immunity to polarity does not give agreement between subjects.

Meanwhile the polarity-based read stays at 92.6% and 100% for sub_01 and 70.5% for sub_11, so it remains the
best available rule.

## Why each group failed

1. **Normalisation (1–3).** The signals are already unit-free: every one divides by the channel's own spread
   or by the recent background. Replacing that with a history IQR or a different band fraction changes which
   windows sit near the threshold, but adds no information (+0.01 and −0.01).
2. **Direction (4–7).** See above: magnitude-based reads are subject-mirrored.
3. **Nod against turn (8–10).** Common-mode ratio (−1.04), coherence ratio (−0.36) and velocity gradient
   (+0.26) all measure the same thing the existing `nod` signal measures, and none is sharper than it.
4. **Artifacts (11–13).** Blinks in the rest windows are rare (0.3%) and the residual ones look like small
   nods, so a decay or flatness or curvature threshold removes real nods together with them.
5. **STL structure (14–15).** Requiring persistence trades false alarms for missed movements roughly evenly;
   the soft veto never fired, because when the gate margin is large the sideways veto is rarely the deciding
   test anyway.
