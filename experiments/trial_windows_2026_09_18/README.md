> **The dataset this round built was deleted on 2026-09-18** (the user keeps only `extracted_npz_dataset_v2`).
> Rebuild it with `python -m _archive.pipeline.extract_trial_windows` if you want to re-run these scripts. The measurements below stay valid.

# One 3-second window per trial: 442 windows from the original recordings (18 Sep 2026)

The user asked for exactly one window per trial (442 trials), cut from the original recordings, with the
movement found by a new method that uses no `.fif` file and no marker code, then for the thresholds to be
trained on sessions 1–30 and the result tested. The earlier `extracted_npz_dataset_v3_csv` was deleted.

## The dataset — `_archive/pipeline/extract_trial_windows.py`, `extracted_npz_dataset_trials/`

- **Input:** only `dataset/eeg_cleaned_data/<subject>/<session>/ica_eeg/*_task-<task>_run-<nn>_eeg.csv`.
- **Label:** the task in the file name.
- **442 windows:** LEFT 146, RIGHT 150, DOWN 122, STRAIGHT 24 (one per trial).
  sub_01 sessions 1–30: 287 · sub_01 sessions 31–36: 55 · sub_11: 100. All 24 straight trials belong to
  sub_01, so **sub_11 has no rest trial at all**.

**Finding the movement from the signal alone.** A head movement moves every electrode at once; a blink
moves the two frontopolar channels and a bad contact moves one. The detector uses that difference:

1. band-pass 0.3–4 Hz, zero-phase;
2. divide each channel by its own robust spread over the whole trial (unit-free, so a firmly and a loosely
   sitting electrode count the same);
3. `score(t)` = **median over the eight channels**, smoothed over 0.25 s. A median needs half the channels
   to move at the same moment, so a one- or two-channel artifact cannot make a peak;
4. the movement is the largest score in the trial, ignoring 0.3 s at each end (filter edges);
5. its **core** is the surrounding stretch above 35% of the peak — low enough to hold both phases of a
   turn — capped at 2.5 s;
6. the window is the 3 s centred on the core, moved inside the trial if it would stick out.

A straight trial has no movement, so its window is the **quietest 3 s** of the trial.

**Quality of the extraction**

| | |
|---|---|
| core duration | median 1.13 s (q10 0.61, q90 1.76) |
| core strength (peak ÷ median score) | median 4.2 |
| whole core inside its window | **418 / 418** |
| stream before the window inside the trial | ≥ 2 s for 49% of trials, ≥ 1 s for 68% |

**Cross-check against the `.fif` annotations** (used only for this comparison, never for building):

| | all | left | right | down |
|---|---|---|---|---|
| annotated movement fully inside my window | 85% | 89% | 87% | 79% |
| at least half inside | 90% | 93% | 89% | 88% |
| not inside at all | 7% | | | |
| my peak within 0.5 s of the annotated centre | 70% | | | |
| within 1.0 s | 86% | | | |

## Training on sub_01 sessions 1–30 — `design.log`

Blocked 3-fold CV (sessions 1–10 / 11–20 / 21–30), gate thresholds first, then the nod thresholds.

| on sessions 1–30 | movement type | detected | direction | 4-class |
|---|---|---|---|---|
| shipped thresholds | 190/265 (71.7%) | 87.2% | 89.7% | 78.78% |
| shipped, no stream history | 191/265 (72.1%) | 89.1% | 89.6% | 78.09% |
| re-selected, cross-validated | 201/265 (75.8%) | 97.4% | 90.0% | **81.06%** |
| re-selected, fitted and scored here | 206/265 (77.7%) | 97.4% | 89.2% | 83.58% |

Re-selection wins on this dataset. The chosen values are much looser:
**b 3.0, g 5.0, d 0.16, q 2.0, n −0.6, v 5.75.**

## The one test run — `test.log`

| split | thresholds | movement type | detected | direction | 4-class |
|---|---|---|---|---|---|
| sub_01 1–30 (design) | shipped | 190/265 (71.7%) | 87.2% | 89.7% | 78.78% |
| | trained on 1–30 | 206/265 (77.7%) | 97.4% | 89.2% | 83.58% |
| sub_01 31–36 (test) | shipped | 20/53 (37.7%) | 64.2% | 93.8% | 54.19% |
| | trained on 1–30 | **29/53 (54.7%)** | 83.0% | 88.9% | **65.18%** |
| sub_11 (test) | shipped | 67/100 (67.0%) | 94.0% | 63.5% | 68.10% |
| | trained on 1–30 | **70/100 (70.0%)** | 100.0% | 63.8% | **70.79%** |

Per class on the test sets, trained thresholds: sub_01 31–36 LEFT 6/15, RIGHT 10/17, DOWN 13/21, STRAIGHT 2/2;
sub_11 LEFT 23/35, RIGHT 21/35, DOWN 26/30 (no rest trials).

## Why the package defaults were NOT changed — `v2_crosscheck.log`

This dataset contains almost only trials in which a movement happened (418 of 442), so a monitor tuned on it
becomes movement-happy: that costs nothing here and everything on a real stream. The same trained thresholds
on the window dataset, which has a realistic amount of rest:

| thresholds | sub_01 1–30 | sub_01 31–36 | sub_11 | false alarms |
|---|---|---|---|---|
| shipped | **85.01%** | **69.62%** | **78.67%** | 2.1 / 2.7 / 4.5% |
| trained on the 442 trials | 82.44% | 69.39% | 75.09% | **22.4 / 25.4 / 33.1%** |

So the shipped thresholds stay the default. The trained set is worth keeping for a **cued** interface, where
the system already knows a movement is coming and only the direction matters:

```python
RtamtConfig(background_ratio=3.0, gate_ratio=5.0, delta_range_min=0.16,
            quiet_min=2.0, down_threshold=-0.6, down_veto_prominence=5.75)
```

## What the 442-trial numbers say

1. **The new extraction works:** the movement is inside its window in 418/418 trials by the detector's own
   measure, and the independently annotated movement is fully inside in 85% of trials.
2. **One window per trial is a much harder test** than the window dataset: each trial counts once, and a single
   badly placed window is a whole trial lost. The same monitor scores 78.78% here and 85.01% on the window
   dataset for the design sessions.
3. **Direction is the same story as everywhere:** 89.7% right on sub_01 sessions 1–30 and 63.5% on sub_11.
4. **Training on 30 sessions helps on this dataset** (+11 points on sub_01 31–36, +2.7 on sub_11), entirely by
   detecting more movements (64% → 83%), and it does so by trading away rest accuracy that this dataset cannot
   measure.
