# Are the extracted 3-second windows correct? — audit without the fif files (17 Sep 2026)

**Sources used:**
- `extracted_npz_dataset/*/ses-*/clean_epochs.npz` (the windows),
- the parent `dataset/.../labelled_epochs_ds.npz`,
- the per-run CSV exports in `dataset/.../ica_eeg/` (recordings and trial/cue markers).

**No `.fif` file was opened.** All 46 sessions, 23,164 windows. No monitor was changed.

## Verdict

The windows are real recorded EEG with correct shape, units and labels. **But 2,135 windows (9.2%)
are not continuous recordings:** each joins the end of one run to the start of a different run.

## What is correct

| check | result |
|---|---|
| shape (9 × 768), finite values, no flat channel | all windows |
| channel 9 = T3 − T4 | exact (worst 2.9e-11 V) |
| each window has its own mean removed | exact (worst 3e-14 µV) |
| duplicates | none |
| label matches the task (LEFT windows from left runs, …) | all windows |
| movement metadata: fully inside the window with 0.25 s margins | all movement windows |
| identical to the parent `labelled_epochs_ds.npz` row | all windows, difference exactly 0 V |
| label agrees with the parent label | 100% |
| every sample exists in the per-run CSV recordings | all windows |
| one movement per run, inside the run's trial begin/end markers | 384 movements, 0 exceptions |
| where a labelled movement shows a burst, the burst lies at the labelled time | all but 1 movement |

## What is wrong: spliced windows — `spliced_windows.csv`, `spliced_window_examples.png`

Each task file joins its runs back to back, and windows were cut across the joins. A spliced window
contains the end of one run and the start of another, recorded at a different time, with a step at
the join. The step is typically **13× the normal sample-to-sample change** (median 16 µV, max 81 µV).
A live headset stream never contains such a step.

| label | spliced windows |
|---|---|
| STRAIGHT | 1,942 of 19,192 (10.1%) |
| LEFT | 92 of 1,355 (6.8%) |
| RIGHT | 93 of 1,461 (6.4%) |
| DOWN | 0 of 1,156 |

By split: sessions 1–30 8.9%, sub_01 31–36 7.0%, sub_11 15.4%. The first check classified 2,127
windows; 8 more (1 in sub_01 ses-29, 7 in sub_11 ses-04) were splices of runs whose file order could
not be inferred, and were confirmed piece by piece.

**Effect on the unchanged monitor:**
- On sessions 1–30 it calls 9.7% of continuous STRAIGHT windows a movement but 21.6% of spliced
  ones, and 35% of splices with a large jump.
- Spliced windows are 9% of STRAIGHT but 19% of the false alarms (357 of 1,891).

| 4-class balanced | all windows | without spliced windows |
|---|---|---|
| sessions 1–30 | 82.30% | 82.74% |
| sub_01 31–36 | 66.28% | 67.22% |
| sub_11 | 76.40% | 77.00% |

## Smaller issue: labelled movements without a visible burst

Movement windows whose loudest 0.5 s is weaker than 90% of rest windows:

| split | LEFT | RIGHT | DOWN |
|---|---|---|---|
| sessions 1–30 | 11 of 92 | 9 of 92 | 17 of 68 |
| sub_01 31–36 | 1 of 11 | 1 of 16 | 3 of 10 |
| sub_11 | 4 of 31 | 3 of 34 | 5 of 30 |

This matches the earlier annotation check: some labels point at barely visible movements, nods most
often.

Timing note: movement onsets lie between −3.9 and +3.4 s from the run's `move_head_now` marker
(median −0.5 s). The marker is not a precise movement time.

## Correction: `extracted_npz_dataset_v2`

Built by `_archive/pipeline/remove_spliced_windows.py`. It keeps a window only if all 768 samples lie inside
**one** run of the per-run CSV recordings (exact match within 1e-10 V). No sample or label is
changed, and no fif file is read.

| | windows | LEFT | RIGHT | DOWN | STRAIGHT |
|---|---|---|---|---|---|
| extracted_npz_dataset | 23,164 | 1,355 | 1,461 | 1,156 | 19,192 |
| extracted_npz_dataset_v2 | **21,029** | 1,263 | 1,367 | 1,156 | 17,243 |

**The same audit re-run on v2** (`audit_summary_v2.txt`):
- 100% of windows lie inside one run; 0 spliced, 0 unexplained.
- Every structural check passes, and all windows are identical to their parent rows.
- All 384 movements are still present.

A separate check confirmed that all 21,029 v2 windows, labels and metadata are byte-identical to
their v1 rows (`v1_index`).

**Accuracy on v2 with the unchanged monitors** (`eval_v2.log`; RTAMT agreed on 40 of 40 sampled
windows per split):

| 4-class balanced | old dataset | **v2** | LEFT | RIGHT | DOWN | STRAIGHT |
|---|---|---|---|---|---|---|
| sub_01 sessions 1–30 (design) | 82.30% | **82.74%** | 83.6 | 79.4 | 77.5 | 90.3 |
| sub_01 sessions 31–36 (test) | 66.28% | **67.22%** | 59.0 | 64.5 | 57.7 | 87.7 |
| sub_11 (test) | 76.40% | **77.00%** | 56.1 | 74.7 | 88.4 | 88.8 |

Re-selecting the thresholds on v2 sessions 1–30 with the one-standard-error rule moves nothing
meaningful (d 0.208→0.210, q 2.72→2.726, +0.06 points), so the thresholds are kept.

## Does each v2 window contain its labelled head movement?

This check does not use the monitors (`content_audit.py`, `content_audit2.py`,
`v2_movement_content_check_final.csv`). All reference levels come from rest windows.

- **Presence:** is there a burst at the labelled time? "Clear" means stronger than 90% of rest
  windows; "weak" means between the rest median and that level; "none" means as quiet as ordinary
  rest.
- **Type:** does the waveform match the turn signature measured on the continuous recordings
  (sessions 1–30)? Two methods are used: a fixed-timing three-phase index, and shape correlation
  with ±0.15 s slack, on T3−T4 and F4−T3. A turn counts as "the other direction", and a nod as
  "turn-shaped", only when **both** methods agree. "Not confirmed" means the shape was too weak or
  unusual to judge; it is not a sign of an error.

| split | class | movements | burst clear / weak / none | direction confirmed / not confirmed / **opposite** | nods turn-shaped |
|---|---|---|---|---|---|
| sub_01 1–30 | LEFT | 92 | 81 / 11 / 0 | 57 / 33 / **2** | |
| sub_01 1–30 | RIGHT | 92 | 83 / 9 / 0 | 60 / 30 / **2** | |
| sub_01 1–30 | DOWN | 68 | 51 / 16 / **1** | | **1** |
| sub_01 31–36 | LEFT | 11 | 10 / 1 / 0 | 4 / 7 / 0 | |
| sub_01 31–36 | RIGHT | 16 | 15 / 1 / 0 | 8 / 8 / 0 | |
| sub_01 31–36 | DOWN | 10 | 7 / 3 / 0 | | **1** |
| sub_11 | LEFT | 31 | 27 / 4 / 0 | 13 / 17 / **1** | |
| sub_11 | RIGHT | 34 | 31 / 3 / 0 | 19 / 6 / **9** | |
| sub_11 | DOWN | 30 | 25 / 5 / 0 | | **12** |

- **Rest windows:** a movement-sized burst appears in 0.4% of sessions 1–30 windows, 0.4% of
  sessions 31–36 and 0% of sub_11. A turn-like shape appears in 5.1%, 6.3% and 13.2%; the threshold
  alone gives 5% by chance.
- **Presence verdict:** 383 of 384 labelled movements contain a burst at the labelled time.
- **sub_01 verdict:** labels agree with the EEG. Only 4 of 211 turns look like the opposite
  direction, and 2 of 78 nods look like turns.
- **sub_11 verdict:** these checks cannot confirm or reject direction or type. Its waveforms differ
  from sub_01's: in ses-03 both LEFT and RIGHT are inverted, and in ses-02, 05 and 10 both
  directions look LEFT-shaped. This fits its large movement-locked T4/T3 swings (not a broken electrode; `experiments/t4_check_2026_09_17`) and inverted sessions better
  than random label mistakes. The 22 flagged sub_11 movements are listed in the CSV for manual
  review.

## Files

- `audit_npz.py` — runs the audit.
- `audit_per_session.json` — full per-session results.
- `audit_summary.txt` — the printed summary.
- `splice_look.py` — monitor behaviour on splices and the example figure.
- `spliced_windows.csv` — every spliced window: session, window index, parent index, label, task,
  start sample, the two runs, join time and jump size.
