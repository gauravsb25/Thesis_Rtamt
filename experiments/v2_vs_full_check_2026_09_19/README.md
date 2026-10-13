# Why the full 39k corpus looked better than v2 in some cells (19 Sep 2026)

The user noticed that a few cells were higher on the full 39,269-window corpus than on the cleaned
`extracted_npz_dataset_v2`, e.g. DOWN on sub_01 sessions 31–36: 63.4% (full) against 57.7% (v2). If v2 only
removed bad windows, that should not happen. This check pairs every v2 window with the same samples in the full
corpus (`source_index`) and scores them like for like. Code `run.py`, log `results.log`.

## 1. The v2 data itself is intact

- Every one of the 21,029 v2 windows maps to a row of the full corpus.
- The samples are identical in 444 of 444 spot-checked windows (difference < 1e-10 V after the per-window mean
  removal both datasets apply).
- The labels agree in 21,029 of 21,029 windows.

So v2 is exactly a subset of the full corpus, with the same samples and the same labels.

## 2. The discrepancy comes from how v2's stream history is rebuilt

The background gate needs the stream before each window. For v2 that stream is rebuilt from **v2's own
windows** — but v2 dropped 18,240 windows, which leaves gaps in the stream:

| | windows whose background gate is active |
|---|---|
| history rebuilt from v2's windows | **74.3%** |
| the same windows, full continuous stream | **86.7%** |

So a quarter of v2's windows fall back to the older contrast gate, not because the data is bad but because
their history was thrown away with the dropped windows. Scoring the **same windows, same labels, same
thresholds** with the continuous history:

| split | v2's own history | full continuous history | verdicts changed (fixed / broken) |
|---|---|---|---|
| sub_01 1–30 | 85.01% | **85.83%** | 146 (103 / 24) |
| sub_01 31–36 | 69.62% | **71.19%** | 25 (12 / 0) |
| sub_11 | 78.67% | **79.48%** | 28 (18 / 5) |

The suspicious cell disappears: DOWN on sub_01 31–36 becomes **78/123 = 63.4%**, exactly the full-corpus
value. sub_11 LEFT moves from 56.1% to 57.2%; the remaining 1-point gap to the full corpus (58.2%) is far
inside that cell's confidence interval (29.6–70.1%).

## 3. What is still imperfect in v2 (known, measured)

- **Label noise from the annotations.** 5 of 108 nods look nearly empty; among rest windows called DOWN, at
  most ~6% could be unlabelled real movements (`experiments/label_content_check_2026_09_17`).
- **Annotation timing is not perfect.** An independent, annotation-free extraction put the movement peak
  within 0.5 s of the annotation in 70% of trials; the annotated movement was fully inside the independent
  window in 85% (`experiments/trial_windows_2026_09_18`).
- **Residual blinks** that the ICA cleaning did not fully remove sit in about half of all windows; they are
  the main source of false DOWNs (`experiments/artifact_check_2026_09_17`).
- **Two sub_11 sessions (08, 09) have an inverted direction signal**; whether that is physiology or how the
  cap sat cannot be decided from the data.
- **The splice removal is exact**: every kept window was matched sample by sample to a single recorded run.

## Recommendation

Score v2 with the **full continuous stream** as history. A live system always has the continuous stream, so
this is the more realistic evaluation, it removes the inconsistency the user spotted, and it needs no change to
the monitor. The headline would become **85.83 / 71.19 / 79.48** (from 85.01 / 69.62 / 78.67).
