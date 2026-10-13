# Four proposed methods, tested (18 Sep 2026)

Four ideas were proposed to attack the three known weaknesses: turns called DOWN on sub_01 31–36,
LEFT/RIGHT inversion on sub_11, and blink-driven false DOWNs. All four respect the project's rules (no
machine learning, unit-free ratios only, past-time STL). All four were implemented and measured with the
usual protocol: blocked 3-fold CV on sub_01 sessions 1–30, adoption rule fixed first (CV gain ≥ +0.5 points
and more sessions better than worse), then one test run.

Baseline: the shipped monitor with the stream background, **85.01 / 69.62 / 78.67**.

## The signals that were added — `features.py`

| signal | definition |
|---|---|
| `lateral_imbalance` | (|peak of F4−T4| − |peak of F3−T3|) ÷ (spread of F4−T4 + spread of F3−T3), in the loudest 1.5 s |
| `lat_signed` | the same with signed peaks (a magnitude difference alone cannot give a side) |
| `dir_frontal`, `dir_temporal` | signed peak of F4−F3 and of T4−T3 ÷ their own spread |
| `yaw_ratio` | [energy(T3−T4) + energy(F4−F3)] ÷ [energy(Fz) + energy(Cz)], 0.3–4 Hz, whole window |
| `yaw_ratio_loud` | the same in the loudest 1 s |
| `ocular_ratio`, `ocular_ratio_loud` | energy(Fp1+Fp2) ÷ energy(Fz+Cz), whole window and loudest 1 s |
| `bg_sustained_K` | largest value the background HOLDS for K samples (the STL `once(historically[0:K-1](background ≥ b))`) |
| `midline_deflect`, `midline_deflect4` | prominence restricted to Fz, Cz (and Fp1, Fp2) instead of all 8 channels |

## Cross-validation on sessions 1–30 — `dev.log`

| method | design | CV | gain | sessions +/− | verdict |
|---|---|---|---|---|---|
| 1. direction | majority of F4−T3, F4−F3, T4−T3 | 85.01% | +0.00 | 0 / 0 | CV kept the current rule |
| 1. direction | abstain when the pairs disagree | 85.01% | +0.00 | 0 / 0 | CV kept the current rule |
| 1. direction | signed imbalance alone | 85.01% | +0.00 | 0 / 0 | CV kept the current rule |
| 1. direction | magnitude imbalance alone | 85.01% | +0.00 | 0 / 0 | CV kept the current rule |
| 2. yaw veto | whole window | 84.28% | −0.73 | 1 / 4 | fail |
| **2. yaw veto** | **loudest 1 s** | **86.39%** | **+1.38** | **4 / 1** | **pass → tested** |
| 3. blink veto | ocular ratio, whole window | 84.74% | −0.27 | 0 / 1 | fail |
| 3. blink veto | ocular ratio, loudest 1 s | 85.01% | +0.00 | 0 / 0 | fail |
| 3. persistence | background must hold 0.125 s | 85.54% | +0.53 | 11 / 17 | fail (more sessions worse) |
| 3. persistence | hold 0.3 s / 0.5 s / 0.75 s | 85.10 / 85.14 / 84.15% | +0.09 / +0.13 / −0.85 | 13/15, 13/16, 5/15 | fail |
| 4. midline nods | Fz, Cz | 84.98% | −0.02 | 0 / 5 | fail |
| 4. midline nods | Fz, Cz, Fp1, Fp2 | 84.98% | −0.02 | 0 / 5 | fail |

## The one test run — `test.log`

Yaw veto in the loudest 1 s, threshold 3.67, fitted on all of sessions 1–30:

| split | current | with the yaw veto |
|---|---|---|
| sessions 1–30 (design) | 85.01% (L 83.9 R 80.0 D 78.2 S 97.9) | **86.39%** (L 87.2 R 82.7 D 77.8 S 97.9) |
| sub_01 31–36 (test) | 69.62% (L 59.0 R 64.5 D 57.7 S 97.3) | 69.03% (L 60.7 R 64.5 **D 53.7** S 97.3) |
| sub_11 (test) | 78.67% (L 56.1 R 74.7 D 88.4 S 95.5) | 74.81% (L 56.1 R 74.7 **D 73.0** S 95.5) |

**Not adopted.** It does what it promises on the design sessions — turns stop being called DOWN, LEFT
83.9 → 87.2 and RIGHT 80.0 → 82.7 — but it also throws real nods out of DOWN on both held-out sets
(57.7 → 53.7 and 88.4 → 73.0). sub_11's nods carry large temporal energy, so a yaw-based veto removes them.
This is the same failure as the head-geometry rule tested on 2026-09-17 (sub_11 77.00 → 67.77).

## Why the direction consensus cannot work — `direction_diagnostic.log`

Accuracy of each direction rule among the windows the monitor already calls a turn:

| direction rule | sub_01 1–30 | sub_01 31–36 | sub_11 |
|---|---|---|---|
| current: sign of the F4−T3 peak | 92.6% | 100.0% | 70.5% |
| frontal pair F4−F3 | 83.7% | 93.6% | 65.8% |
| temporal pair T4−T3 | 92.5% | 100.0% | 42.6% |
| signed imbalance (F4−T4 against F3−T3) | 75.8% | 85.6% | 54.6% |
| **magnitude imbalance** | **47.9%** | **41.7%** | **95.8%** |
| majority of the three pairs | 91.2% | 100.0% | 68.4% |

Two things follow:

1. **Consensus cannot repair an inversion.** On sub_11 the three pairs agree with each other in only 31% of
   turn windows, and the pairs that are strongest there (T4−T3) are the most inverted (42.6%). A majority
   vote of mostly-wrong voters is still wrong.
2. **A rule that fixes sub_11 exists, and it is the opposite of what sub_01 needs.** sub_11's turns are
   carried by the *size* of the electrode opposite the turn, not by polarity, so the magnitude imbalance is
   right 95.8% of the time there — and 47.9% on the design sessions, i.e. a coin toss. No single unit-free
   rule serves both people.
3. The inversion is per session, not per subject: the current rule is right in 100% of sub_11 ses-01 windows
   and 0% of ses-08.

## Conclusion

All four methods are rejected; the monitors are unchanged. The yaw veto is the only one that survived the
design sessions, and it fails the same way earlier spatial rules did. The direction diagnostic is the most
useful outcome: it shows that LEFT/RIGHT on sub_11 cannot be fixed by combining electrode pairs, because the
information there lives in a different quantity than in sub_01. That needs a third subject, or one known
turn per session to fix the polarity (per-session calibration, currently not allowed by the project rules).
