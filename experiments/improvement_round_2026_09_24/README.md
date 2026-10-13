# Can the four monitors be made more accurate? (24 Sep 2026)

**Result: no change passed the adoption rule. The monitors are unchanged at 85.01 / 69.62 / 78.67.**

This round looked for improvements the earlier ~50 rounds had not tried, and measured why the remaining
errors resist. It produced new facts about the errors and two traps that future rounds should avoid.

Protocol as always: design on sub_01 sessions 1–30 with blocked 3-fold cross-validation (sessions 1–10 /
11–20 / 21–30), held-out sets sub_01 31–36 and sub_11. Several diagnostics printed held-out numbers
(listed in section 6). None was used to choose a change.

## 1. Tools built

- `scripts/feat.py` is a vectorised copy of the monitor's end-of-window measurements. It reproduces
  `measurements_v2.npz` to 5e-15 on all 21,029 windows in 27 s, and with it the RTAMT verdicts 21,029 / 21,029.
- `scripts/events.py` loads every run from the per-run CSVs (411 runs). Every stored window matches its CSV
  slice to 3e-5 µV.
- `scripts/harness.py` scores the monitor and any modification with the project's blocked 3-fold CV.
- `scripts/true_hist.py` gives every window its true preceding samples from its own run.

## 2. What the remaining errors are

### 2.1 Errors are all-or-nothing per movement

Each movement appears in about ten overlapping windows. A movement is almost always right in all of them
or wrong in all of them.

| split | movements | all windows right or all wrong |
|---|---|---|
| sub_01 1–30 | 252 | 221 (88%) |
| sub_01 31–36 | 37 | 29 (78%) |
| sub_11 | 95 | 85 (89%) |

So the held-out sets hold only 37 and 95 independent movements. A LEFT class of 11 movements moves by
about 9 points when one movement flips. Where a movement sits inside the window barely matters
(accuracy by movement position varies by a few points, without a trend).

### 2.2 Nod versus turn: perfect inside each session, but the gap moves

Inside every held-out session that has both classes, the `nod` signal separates turns from nods:
AUC 1.0 in 13 of 14 sessions (ses-36: 0.857). The fixed threshold −0.661 is simply in the wrong place for
sessions 32–35, whose turns sit at median −0.61, −0.69, +0.63 and −0.16.

| measurement | value |
|---|---|
| best per-session threshold, chosen with labels (ceiling of any per-session fix) | 88.65 on sessions 1–30 (+3.64) |
| label-free split of each session's own movements (Otsu, non-causal) | −0.38 to −2.04 |
| per-session threshold from ONE labelled nod and ONE labelled turn | 1–30 −1.02 (4 better / 8 worse); 31–36 +2.34; sub_11 −0.12 |
| correlation, session rest `nod` vs session turn `nod` (sessions 1–30) | 0.43; vs nod `nod`: 0.04 |
| `nod` minus the session's rest median, AUC | 1–30 0.964 → 0.972, 31–36 0.858 → 0.817, sub_11 0.999 → 0.733 |
| correlation, turn `nod` vs movement strength (log contrast), sessions 1–30 | −0.03 |

Turns and nods drift independently from session to session, so it is not a shared electrode offset.
Resting data does not predict the drift, and weak turns do not drift toward the resting value. One
calibration movement per class is too variable to find the gap. The within-session separation is real,
but nothing available to a label-free monitor locates it.

### 2.3 Left/right swaps on the design sessions are turn-and-return lobes

All 13 swapped design movements have a multi-lobed F4−T3 trace (`figures/swaps.png`). The turn and the
return make lobes of both signs with similar size, and the peak rule sometimes picks the return lobe.

Reading the FIRST lobe that reaches h × the largest instead (`scripts/first_lobe.py`; h = 1 is the shipped
peak rule), direction accuracy among windows called a turn:

| h | sessions 1–30 | sub_01 31–36 | sub_11 | swapped design movements now right |
|---|---|---|---|---|
| 1.0 (shipped) | 92.6 | 100.0 | 70.5 | 0 / 13 |
| 0.8 | 86.7 | 87.7 | 71.0 | 4 / 13 |
| 0.6 | 87.1 | 75.4 | 65.5 | 7 / 13 |
| 0.5 | 86.5 | 71.7 | 56.9 | 9 / 13 |

It repairs most swaps and breaks more turns than it repairs, on every split. sub_11's RIGHT turns read
correctly mostly because the middle lobe of its mirrored T3 swing is the largest (`figures/erp_pairs.png`),
which any earlier-lobe read loses. The first deflection after the annotated onset gives at most 61.5% on
sub_11 for every pair and window tested (`scripts/onset_sign.py`), so the "saccade leads the head" sign does
not transfer either.

### 2.4 The stream history is confounded with the class

Movements start about 2–3 s into their runs, so most movement windows have no usable history.

| windows without valid history | movement | rest |
|---|---|---|
| rebuilt history (the headline protocol) | 74.5% | 15.0% |
| true run history from the CSVs | 52.5% | 10.9% |

In practice the contrast gate judges the movements and the background gate judges the rest. Any change to
the no-history branch therefore acts mostly on movements. See trap 4.1.

With true run history (5 s buffer, as `StreamMonitor` keeps) the shipped monitor scores
**85.18 / 70.29 / 78.63**, essentially the headline.

## 3. Tested and rejected

"CV" is the gain on sessions 1–30 with thresholds re-selected inside each training fold, against the shipped
monitor. "Fair" compares against the shipped design with the same thresholds re-selected the same way.

| idea | result | why rejected |
|---|---|---|
| `nod` minus the rest value of the 4 s before the window | pooled AUC 0.955 → 0.926 (windows with history) | worse, and most movement windows have no history |
| veto DOWN when the direction read is strong (`|turn|` > t) | CV −0.17 (0 / 1 sessions) | no gain |
| slanted boundary `nod − a·sideways ≥ n` | CV −1.35; fair +0.19 | noise |
| nod from the loudest second / the event only | AUC 1–30 +0.003 to +0.005, 31–36 −0.004 to −0.006 | no gain |
| re-select b under true history | flat 85.0–85.3 for b 5.0–7.0; CV −1.23 | already optimal |
| per-channel within-window reference as the no-history gate (the background gate's form, referenced to the window's 10th-percentile block) | "each window alone": fair +1.02 (22 / 7) but vs shipped +0.51 (11 / 19). Movement recall at matched false alarms: 1–30 +0.9 to +1.4, sub_11 +1 to +6, 31–36 −2 to −16 | not dominant; loses 31–36 |
| direction from the first lobe reaching h × the largest | see 2.3 | worse on every split |
| shorter background settle / minimum history, cold or steady-state filter start (9 variants, two history sources) | design selection picks settle 0: 31–36 +1.4 to +2.5, sub_11 −1.4 to −1.7 | trades one held-out set for the other |
| label-free per-session nod threshold | −0.38 to −2.04 | see 2.2 |

## 4. Traps found

### 4.1 Lowering the contrast threshold looks like a +1.9 gain and is an artifact

Re-selecting only `g` under CV picks 3.75–4.5 instead of 7.81 and gains +1.88 on sessions 1–30. `train` never
finds this because its grid starts at `g = 5.0`. The gain comes from 2.4: without history, `g` judges mostly
movement windows. On sub_11, 66% of rest windows lack history, so its false alarms rise from 23 to 142 at
`g = 4.0` and its score falls to 75.30. Only 14 design sessions get better and 15 get worse. In a live stream
almost every window has history, so the gain would not exist there. **Do not widen the `g` grid below 5.0
without this check.**

### 4.2 CV candidates were compared with an in-sample baseline

Re-selecting even the single nod threshold `n` under the same CV scores 1.53 below the shipped 85.01. The
shipped thresholds were chosen on all of sessions 1–29, so the baseline carries an in-sample advantage of
about that size. Any candidate that re-selects a threshold starts ~1.5 points behind. Future rounds should
also report the baseline under the same CV (the "fair" column above).

## 5. What would still move the numbers

- **Nod versus turn on a new session** needs to know where that session's gap lies. The ceiling is +3.6 on
  the design sessions. A one-nod-plus-one-turn calibration does not reach it. A few movements per class per
  session might, but that breaks the no-per-session-constants rule.
- **Direction on sub_11** needs a third subject or another sensor. The polarity patterns of the two subjects
  are mirror images, so any read that fixes the design swaps breaks sub_11 (2.3).
- **The held-out sets** are too small to confirm gains below about 5 points per class (2.1). The history
  details in 2.4 alone move them by 1–3 points.

## 6. Held-out exposure in this round

Diagnostics that printed sub_01 31–36 or sub_11 numbers: true-history scoring, the `g` sweep (4.1), the
settle/minimum variants, the matched-false-alarm curves, the calibration measurement, within-session AUCs,
the first-deflection table and the median waveforms. None selected a change. Nothing was adopted.

## 7. Reproduce

From `scripts/`, in order: `meta.py`, `events.py` (writes `runs.pkl`, 40 MB), `verify_feat.py` (writes
`feat_base.npz` and checks it against the cache), `hist_feat.py`, `true_hist.py`, `bg_variants.py`. Then any
`*_eval.py` or analysis script. They read the project at `C:\Users\gaura\thesis_rtamt` and write their
caches next to themselves. The caches are not kept here.
