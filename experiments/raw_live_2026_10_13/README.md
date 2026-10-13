# Raw-data techniques (2026-10-13)

**Question.** The monitor was only ever tuned and scored on ICA-cleaned data, but a live headset delivers raw
EEG with blinks (`D:/errp/eeg_cleaned_data/*/ses-*/pp_eeg`, filtered only). Can it be made to work on raw data?

**Result.** One technique passes the adoption rule: a **blink veto** (T2). It raises raw-data balanced accuracy
on both held-out sets and cuts false alarms by 35–76 %. Re-selecting the thresholds (T1) fails cross-validation on
both raw and cleaned data. Nothing in `monitoring/` was changed.

## Setup

Every window of `extracted_npz_dataset_v2` was cut from its run's CSV at the stored offset, from the raw AND the
cleaned recording (identical samples otherwise), with the run's real preceding 5 s as history (`measure.py`).
Two blink features from the loudest 0.5 s: `fp_dom` = log(Fp1,Fp2 energy / T3,T4 energy), each channel over its
own median (unit-free); `event_w` = width of the energy event at half height. Medians on sub_01/ses-05 raw:
rest +0.29, DOWN −1.15, LEFT −0.89 — a blink is forehead-dominant, a head movement drives the temples.

Protocol (`evaluate.py`, `rest.py`): design on sub_01 1–20, blocked CV 1–7 / 8–14 / 15–20, adopt only with a CV
gain ≥ +0.5 and more sessions better than worse; held-out sub_01 21–36 and sub_11 scored once.

## Results (four-class balanced accuracy; false alarms in brackets)

| data | technique | 1–20 (CV) | 21–36 | 31–36 | sub_11 | sessions |
|---|---|---|---|---|---|---|
| raw | baseline (current thresholds) | 75.66 (4238) | 77.67 (1282) | 62.66 (127) | 76.31 (166) | |
| raw | **T2 blink veto** (fp_dom ≥ 2.0 and width ≤ 0.3 s → STRAIGHT) | **78.94 (796)** | **81.35 (309)** | **65.18 (42)** | **77.94 (108)** | **+17 / −3** |
| raw | T1 thresholds re-selected | 73.50 (4601) | 79.43 (1500) | 66.87 (165) | 76.80 (179) | +9 / −11 |
| raw | T3 = T1 + T2 | 77.45 (1008) | 83.33 (482) | 69.92 (62) | 78.43 (121) | +12 / −8 |
| cleaned | baseline, real history | 85.46 (301) | 81.22 (122) | 69.98 (50) | 77.37 (72) | |
| cleaned | T1 thresholds re-selected | 82.18 (438) | 80.96 (113) | 70.02 (51) | 78.31 (53) | +8 / −10 |
| cleaned | T2 veto applied unchanged | 85.28 | 80.89 | 69.98 | 77.37 | |

T3 scores higher on the held-out sets than T2, but CV prefers T2 (78.94 vs 77.45, 17 vs 12 sessions better);
choosing T3 would mean choosing on the test sets. On cleaned data the veto is nearly inert (ICA removed the
blinks): −0.33 on 21–36 (5 DOWN windows), 0.00 on sub_11. The selected veto level (2.0) is the edge of the grid
searched; a wider grid must be re-selected on 1–20 before it is trusted.

## Live replay with the veto (`live_veto.py`)

The saved verdicts of the real StreamMonitor on raw runs (`experiments/live_replay_2026_10_07`), with the veto
applied to every movement verdict; command issued after 4 identical verdicts (1 s):

| raw, live | right | wrong | missed | false commands / min |
|---|---|---|---|---|
| sub_01 21–36, current | 78.6 % | 14.3 % | 7.1 % | 4.33 |
| sub_01 21–36, with veto | 77.8 % | 15.1 % | 7.1 % | **1.24** |
| sub_11, current | 71.6 % | 28.4 % | 0.0 % | 9.41 |
| sub_11, with veto | 72.6 % | 27.4 % | 0.0 % | **5.65** |

(sub_11 has about one minute of scored rest, so its rate is rough.)

## Also found this round

The labelled recordings carry 171 `BAD_noise` annotations (567 s, about as long as all movements). None of the
21,029 windows overlaps one: the noise was excluded upstream, when `labelled_epochs_ds.npz` was made, not by any
script in this repository. The monitor has therefore never been scored on marked noise.

```
python measure.py      # ~10 min: measurements_raw_ica.npz
python evaluate.py     # T1/T2/T3 on raw, T1 on cleaned (~40 min, mostly threshold grid searches)
python live_veto.py    # live replay with the veto (needs ../live_replay_2026_10_07/replay_verdicts.npy)
```
