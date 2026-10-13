# Blinks, muscle, electrode and body artifacts in the 3-second windows (17 Sep 2026)

Question: do bumps caused by eye blinks or by body or hand movement sit in the windows, can they be
detected and separated, and does the monitor's accuracy change without them?

- Dataset: `extracted_npz_dataset_v2`.
- Monitor: the default RTAMT monitors, unchanged.
- Design: detector definitions and every look at the signals used sub_01 sessions 1–30 only. The test
  sets (sub_01 31–36, sub_11) were scored once, with the definitions fixed.
- Test-set reuse: they had already been used in earlier rounds.

The data is the ICA-cleaned recording (`ica_eeg`). It is low-pass filtered near 45 Hz, and channel 9
is T3−T4. There is no motion or EOG channel, so artifacts can only be found from the 8 EEG channels.

## Detectors (no learning, fixed definitions)

| name | what it catches | rule |
|---|---|---|
| BLINK | classic blink | Fp1+Fp2 bump ≥ 5 robust SD, 0.04–0.40 s at half height, same sign and similar size on Fp1/Fp2, ≥ 2.5× the size on Cz/T3/T4 and ≥ 1.3× on F3/F4 |
| BLINK-LIKE | residual blink left by ICA, at the moment the gate reacted to | largest Fp excursion in the loudest 0.5 s is narrow (≤ 0.125 s), same sign on ≥ 7 of 8 channels, Fp ≥ mean(F3,Fz,F4) |
| POP | electrode or cable jump (hand or body touching the headset) | one channel's sample-to-sample jump ≥ 10 robust SD and ≥ 3× the next channel |
| EMG | muscle burst (jaw, neck, body tension) | 20–45 Hz RMS in a 0.25 s slice ≥ 4× that channel's median |
| BIG | gross artifact | any channel > 100 µV (0.5–45 Hz) |

Code: `artifact_features.py`, `separate_and_score.py`. Every flagged window is listed in
`artifact_windows.csv`.

## What the false alarms are (sessions 1–30) — `burst_typing_dev.log`, `bump_shapes_dev.log`, `bump_averages_dev.png`, `false_alarm_examples_dev.png`

- **Not movement leftovers.** Rest windows more than 6 s from any movement raise false alarms as often
  (10.7%) as the straight-ahead task (11.8%).
- **Not classic artifacts.** BLINK, POP, EMG and BIG together are in 0.7% of the 1,534 false alarms, and in
  0.3% of correct rest windows.
- **They are small whole-head blink-shaped bumps.** Typically a dip, then a narrow peak (~0.08 s at
  half height), the same sign on nearly all channels, strongest at Fp1/Fp2 and weakest at Cz,
  12–26 µV. Fp1–Fp2 correlation is 0.96. This looks like blinks whose frontal dominance was flattened by the
  ICA cleaning. 995 of the 1,534 false alarms are called DOWN.
- **The same bump is in every kind of window.** The narrow, bilateral form is the loudest moment in
  64% of correct rest windows, 68% of false DOWNs, 33% of real DOWNs and 33% of turns (`blink_shape_dev.log`).
  False alarms differ from correct rest mainly by **size** (median 14.4 vs 9.4 µV). Real nods have
  a similar shape but are wider (0.14 s) and bigger (19 µV); the distributions overlap.
- **Turns are different.** T3 and T4 move oppositely (correlation −0.8), while false alarms and nods move
  together (+0.2 / +0.75).

## How often each detector fires

| detector | sessions 1–30 L / R / D / S | sub_01 31–36 L / R / D / S | sub_11 L / R / D / S |
|---|---|---|---|
| BLINK | 1.5 / 3.4 / 0.0 / 0.2 % | 0 / 0 / 0 / 0 % | 0 / 0 / 0.4 / 0 % |
| BLINK-LIKE | 31.6 / 27.2 / 24.3 / 51.5 % | 17.9 / 23.5 / 18.7 / 56.9 % | 24.6 / 33.0 / 21.4 / 21.7 % |
| POP | 0.8 / 0 / 6.8 / 0 % | 15.4 / 0 / 0 / 0 % | 0 / 0 / 0 / 0 % |
| EMG | 11.4 / 3.1 / 13.1 / 0.1 % | 7.7 / 0 / 0 / 0 % | 2.1 / 2.3 / 3.2 / 0 % |
| BIG | 9.6 / 1.2 / 2.3 / 0.0 % | 15.4 / 0 / 0 / 0 % | 92.0 / 64.3 / 11.2 / 9.1 % |

- **Classic artifacts are almost absent at rest** and appear mainly *inside movement windows*: POP on Cz
  during nods, EMG and BIG on T3/T4 during turns. These are produced by the head movement itself.
- **sub_01 31–36:** all POP/EMG/BIG windows are one LEFT turn group in ses-32 (T3/T4), and all are classified
  correctly.
- **sub_11:** BIG is T4 (239) and T3 (140) at ~170 µV in most turn windows. These are movement-locked
  swings of the side electrodes, not a body artifact and not a broken electrode (`experiments/t4_check_2026_09_17`).

## Accuracy with and without the flagged windows (4-class balanced)

| windows scored | sessions 1–30 | sub_01 31–36 | sub_11 |
|---|---|---|---|
| all (the honest number) | **82.74%** | **67.22%** | **77.00%** |
| without classic artifacts (BLINK/POP/EMG/BIG) | 82.96% (98% kept) | 65.36% (99% kept) | 88.17% (67% kept; 16% of LEFT, 36% of RIGHT windows kept) |
| only classic-artifact windows | 71.20% | 100% (18 windows) | 80.48% |
| without BLINK-LIKE | 83.46% (52% kept) | 70.84% (55% kept) | 76.76% (76% kept) |
| only BLINK-LIKE | 80.97% | 52.58% | 77.11% |
| without any flag | 82.80% (50% kept) | 69.16% (54% kept) | 87.37% (52% kept) |

- **Removing artifact windows barely changes accuracy on sub_01** (±0.2 points on sessions 1–30; −1.9 / +3.6 on
  31–36, pointing in opposite directions).
- **sub_11's apparent jump to 88% is not valid.** It comes from removing 84% of its LEFT and 64% of its RIGHT windows:
  those with the large T4/T3 turn swings.
- **BLINK-LIKE bumps at the loudest moment make turns a little harder** (sessions 1–30: LEFT 80.2 vs 85.2, RIGHT 75.2
  vs 81.0; sub_01 31–36: 21 LEFT windows from 4 movements, 1 correct). The effect is inconsistent on DOWN and on sub_11.

## Can the monitor use them? Artifact veto (flagged → STRAIGHT)

| veto | sessions 1–30 | sub_01 31–36 | sub_11 |
|---|---|---|---|
| none (current) | **82.74%** | **67.22%** | **77.00%** |
| BLINK | 82.57% | 67.22% | 76.91% |
| BLINK-LIKE | 67.77% | 62.52% | 63.12% |
| POP | 80.83% | 63.38% | 77.00% |
| EMG | 76.68% | 65.30% | 75.11% |
| BIG | 80.32% | 63.38% | 50.94% |
| any | 61.43% | 58.68% | 43.79% |

Blocked 3-fold CV on sessions 1–30 chose **no veto** in all three folds. Every veto removes more
real movements than false alarms, so the monitors are unchanged.

## Conclusion

1. **Blinks, muscle, electrode and body artifacts do exist, but not where expected.** Classic ones are almost
   absent from rest windows (ICA removed most blinks). They occur mainly inside movement windows, caused by the movement.
2. **Most false alarms are small residual blink-shaped bumps.** Such a bump is present in about half of all windows, and
   the same shape appears in real nods. With these 8 channels the difference is only size and width, and the
   distributions overlap.
3. **Excluding flagged windows does not make sub_01 accuracy meaningfully higher**, and on sub_11 it only removes the
   turns with large T4/T3 swings. The full-data numbers (82.74 / 67.22 / 77.00) remain the honest ones.
4. **A clean separation would need an EOG or motion (IMU) channel**, which this dataset does not have.
