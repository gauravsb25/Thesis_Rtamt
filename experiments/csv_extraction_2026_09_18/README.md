> **Superseded on 18 Sep 2026.** The dataset this round built (`extracted_npz_dataset_v3_csv`, 11,137 windows)
> was deleted at the user's request: they asked for ONE 3-second window per trial (442 in total) and for the
> movement to be found from the signal alone, without the cue marker. See
> `experiments/trial_windows_2026_09_18/`. The measurements and conclusions below still describe this round.

# A second, independent extraction of the 3-second windows (18 Sep 2026)

The user asked for the windows to be cut again from the original recordings, without the `.fif` files and
without reusing the earlier extraction, then for the thresholds to be selected on sessions 1–30 and the
result tested, with both datasets scored at the end.

## What was used

- **Input:** only `dataset/eeg_cleaned_data/<subject>/<session>/ica_eeg/*_task-<task>_run-<nn>_eeg.csv`
  (8 channels, 256 Hz, plus a `marker` column).
- **Label:** the task in the file name.
- **Timing:** the marker codes are 1000 = recording start, 1001 = recording stop, **7006 = movement cue**.
  Every left run (146) and nearly every right run (150) has a cue; **no down run has one**, so the cue can
  only narrow the search. The cue is also loose: against the `.fif` annotation the onset sits within ±1.5 s
  of it in only 71% of runs.
- **Movement detector** (`_archive/pipeline/extract_windows_from_csv.py`), deliberately different from the monitor's gate:

| | monitor gate | extractor |
|---|---|---|
| band | 0.5–3 Hz | 0.3–4 Hz |
| statistic | squared energy, each channel divided by its own spread | sum of absolute deviation in µV |
| reference | quietest 0.5 s inside the window | the run's median level |
| filter | causal | zero-phase |
| search | — | left/right: ±4 s around the cue; down: the whole run |

- **Windows kept** (same rule as v2, so the two are comparable): a movement window holds the whole detected
  span with at least 0.25 s spare at both ends, taken every 0.1 s; a rest window stays at least 1 s away from
  the span, taken every 0.5 s.
- **Result:** `extracted_npz_dataset_v3_csv`, 11,137 windows (LEFT 2,423, RIGHT 2,278, DOWN 1,674,
  STRAIGHT 4,762). A movement was found in 407 of 418 movement runs.

**Caveat, stated plainly:** the movement windows are placed by a detector, so the monitor's *movement gate*
is flattered on them. The false-alarm rate on rest windows and the LEFT/RIGHT/DOWN split among movement
windows do not depend on the detector, and both are reported below.

## How the new timings compare with the old ones — `agreement.py`, `agreement.log`

The `.fif` timings are used here only for this comparison, never for building the dataset.

| detector | movement found | peak within 0.5 s | within 1.0 s | span overlap at least 50% | no overlap |
|---|---|---|---|---|---|
| 0.3–4 Hz (used) | 97% | 67% | 87% | 25% | 13% |
| 0.1–1.2 Hz (first try) | 94% | 47% | 65% | 35% | 23% |

The two extractions put the movement at a similar place but not the same place. A head movement goes out and
comes back, so a detector can lock onto the return phase, while the `.fif` annotation marks one phase.

## Selecting the thresholds on the new dataset — `select_dev.log`

Blocked 3-fold cross-validation on v3 sub_01 sessions 1–30: first the gate (background ratio b, contrast
fallback g, slow-wave d, deflection q), then the nod (n, v).

| | four-class balanced on v3 sessions 1–30 |
|---|---|
| shipped thresholds | **85.22%** |
| re-selected, cross-validated | 82.99% |
| re-selected, fitted and scored on the same sessions | 86.49% (optimistic) |

Re-selection is **worse** under cross-validation, so the shipped thresholds stay. The values it wanted
(b 5.5, g 5.5, d 0.18, q 2.6, n −0.6, v 6.5) let far more false alarms through (9.4% against 5.9%).

## Accuracy on both datasets — `select_test.log`

Four-class balanced accuracy. "type" = among the windows called a movement, how often the movement class is
right. "FA" = share of rest windows called a movement.

| dataset · thresholds | sub_01 1–30 | sub_01 31–36 | sub_11 |
|---|---|---|---|
| **v2 (.fif) · shipped** | **85.01%** (FA 2.1%) | **69.62%** (FA 2.7%) | **78.67%** (FA 4.5%) |
| **v3 (CSV-only) · shipped** | **85.22%** (FA 5.9%) | **64.94%** (FA 6.2%) | **72.25%** (FA 26.6%) |
| v2 · thresholds selected on v3 1–30 | 86.25% | 71.46% | 78.45% |
| v3 · thresholds selected on v3 1–30 | 86.49% | 64.59% | 70.71% |

Per class, shipped thresholds:

| | LEFT | RIGHT | DOWN | STRAIGHT |
|---|---|---|---|---|
| v3, sub_01 1–30 | 77.3% | 83.9% | 85.6% | 94.1% |
| v3, sub_01 31–36 | 46.7% | 60.9% | 58.4% | 93.8% |
| v3, sub_11 | 56.5% | 68.2% | 90.9% | 73.4% |
| v2, sub_01 1–30 | 83.9% | 80.0% | 78.2% | 97.9% |
| v2, sub_01 31–36 | 59.0% | 64.5% | 57.7% | 97.3% |
| v2, sub_11 | 56.1% | 74.7% | 88.4% | 95.5% |

## What this says — `diagnostics.log`

1. **The monitor does not depend on how the windows were cut.** On the design sessions the two datasets give
   85.22% and 85.01% with the same unchanged monitor.
2. **Re-training the thresholds on the new dataset does not help.** Cross-validation prefers the shipped values,
   and on the new test sets the re-selected ones are worse (64.59 / 70.71 against 64.94 / 72.25).
3. **LEFT versus RIGHT behaves the same in both datasets:** among turn windows called a turn, the direction is
   right in 92.9% (v3) against 92.6% (v2) of design windows, 96.6% against 100% on sub_01 31–36, and 65.8%
   against 70.5% on sub_11.
4. **The lower v3 test numbers come from its rest windows, not from the movements.** v3 has 4,762 rest windows
   against 17,243, and they are taken only from stretches near a movement, so they are the hardest ones. On
   sub_11 only 252 rest windows remain and 27% of them raise a false alarm. Rest windows from the 11 runs where
   no movement was found are worse than the others (12% against 6% on the design sessions), as expected: those
   runs contain a movement that was labelled rest.
5. **The same weaknesses appear in both extractions:** turns called DOWN on sub_01 31–36 (89 + 70 windows in v3)
   and LEFT/RIGHT swaps on sub_11 (213 + 169 windows in v3).

## Files

| file | what |
|---|---|
| `_archive/pipeline/extract_windows_from_csv.py` | the extractor (CSV only) |
| `extracted_npz_dataset_v3_csv/` | the new dataset, with `manifest.json` and per-window span fields |
| `agreement.py` | comparison with the `.fif` timings |
| `measure.py` | the monitor's measurements per window, with the stream before it |
| `select_and_test.py` | threshold selection, the CV table and the test run |
