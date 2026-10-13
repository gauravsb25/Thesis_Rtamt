> **The dataset this round built was deleted on 2026-09-18** (the user keeps only `extracted_npz_dataset_v2`).
> Rebuild it with `python -m _archive.pipeline.build_preica_windows` if you want to re-run these scripts. The measurements below stay valid.

# Five different ideas for the monitors (17 Sep 2026)

**Setup.** Design data: sub_01 sessions 1–30, blocked 3-fold CV (sessions 1–10 / 11–20 / 21–30).

**Adoption rule, fixed before testing:** CV gain ≥ +0.5 points and more sessions better than worse.
Designs that passed were scored once on sub_01 sessions 31–36 and sub_11. Those sets had been used in
earlier rounds, so they are not fully fresh.

**Data.** `extracted_npz_dataset_v2`. For idea 3, `extracted_npz_dataset_v2_preica` has the same windows cut
from the pre-ICA recordings (`python -m _archive.pipeline.build_preica_windows`; every window verified against the
ICA recording, worst difference 2.9e-11 V).

**Stream history** (ideas 1 and 5) is rebuilt from the dataset's own windows: each window's removed mean
is restored from its overlap with earlier windows, which must agree to 1e-3 µV. Of the windows in
sessions 1–30, 79% have at least 2 s of stream before them.

## The ideas

| # | idea | signals (`features.py`, `preica_features.py`) |
|---|---|---|
| 1 | compare the window with the stream's recent past (radar CFAR) | `background` = per-channel 0.5–3 Hz energy of the 0.5 s ending at t ÷ that channel's median over the H s before the window; mean or median over channels |
| 2 | head geometry instead of data | per sample, least-squares fit of the 8 channels to fixed yaw (left-right x), pitch (front-back y) and common patterns from standard 10-20 positions (MNE); `geo_turn` = log(yaw/pitch energy); `geo_dir` = signed largest yaw coefficient |
| 3 | pre-ICA data for eye-movement direction | monitor signals on pre-ICA windows; horizontal EOG from Fp1−Fp2 and F3−F4 (largest slope, first slope, level shift); physics sign fixed in advance |
| 4 | UNSURE when close to a threshold | unit-free STL robustness margin along the decision path; UNSURE below m |
| 5 | consecutive windows must agree | monitor verdicts at the 10 positions 0.1–1.0 s before each window |

## Cross-validation on sessions 1–30 — `ideas_125_dev.log`, `ideas_combined_dev.log`, `idea3_dev.log`, `idea3_cv_dev.log`, `idea4_dev.log`

Current monitor: 82.74% (L 83.6, R 79.4, D 77.5, S 90.3).

| design | CV | gain | sessions + / − | verdict |
|---|---|---|---|---|
| **1: background replaces contrast (mean over channels, 4 s)** | **85.08%** | **+2.34** | **30 / 0** | pass |
| 1: background AND contrast gate (mean, 4 s) | 84.78% | +2.04 | 29 / 1 | pass |
| 1: other variants (2 s, 8 s, median, both) | 83.3–84.6% | +0.5 to +1.9 | 25–29 / 1–5 | pass |
| 1: background OR contrast (rescue) | 82.7–82.9% | ≤ +0.14 | 2–3 / 6–13 | fail |
| 2: nod OR (geo_turn ≤ t AND no sideways swing) | 83.46% | +0.72 | 3 / 1 | pass |
| 2: geometry nod alone / AND, geometry direction | 82.47–82.74% | ≤ 0 | — | fail |
| 3: monitor on pre-ICA windows (no fitting) | 76.90% | −5.84 | — | fail (rest 62%: blinks) |
| 3: direction from pre-ICA F4−T3 | 82.43% | −0.31 | 3 / 6 | fail |
| 3: eye-movement direction, physics sign | LEFT/RIGHT 24–61% (Fp1−Fp2), 31–87% (F3−F4) | — | — | fail |
| 5: all / majority of the last n verdicts agree | 82.74% | 0 (CV kept current) | — | fail |
| 1 + 2 combined | 85.80% | +3.06 | 29 / 1 | pass |
| 1 + 3 (pre-ICA direction) | 84.76% | +2.03 | 27 / 3 | pass, but below 1 alone |

Idea 3 detail:
- The eye signal behaves the same in ICA and pre-ICA data. The early fast slope of Fp1−Fp2 points against the
  physics sign in 76% of turns, so it is not an eye component that ICA removed.
- F4−T3 direction on pre-ICA windows was 94.0% vs 91.1% on ICA windows, but did not raise the four-class accuracy.

## The one test run — `ideas_test.log`, `idea4_test.log`

| design (fitted on all of sessions 1–30) | sessions 1–30 | sub_01 31–36 | sub_11 |
|---|---|---|---|
| current monitor | 82.74% | 67.22% | 77.00% |
| **1: stream background (b = 6.61, 4 s, mean)** | **85.01%** | **69.62%** | **78.67%** |
| 2: geometry nod rescue (t = −1.37) | 83.84% | 66.79% | 67.77% |
| 1 + 2 | 86.11% | 69.19% | 69.45% |

Idea 1 per class (L / R / D / S):

| split | current | with idea 1 |
|---|---|---|
| sessions 1–30 | 83.6 / 79.4 / 77.5 / 90.3 | 83.9 / 80.0 / 78.2 / 97.9 |
| sub_01 31–36 | 59.0 / 64.5 / 57.7 / 87.7 | 59.0 / 64.5 / 57.7 / 97.3 |
| sub_11 | 56.1 / 74.7 / 88.4 / 88.8 | 56.1 / 74.7 / 88.4 / 95.5 |

- **Idea 1 improved both test sets** and was adopted.
- **Idea 2 broke sub_11:** 64 turn windows were called DOWN (LEFT 56.1 → 48.7, RIGHT 74.7 → 45.2). Not adopted.

Idea 4 (UNSURE), on 1 + 2 with m = 0.14 chosen on sessions 1–30 (every class ≥ 80% decided):

| split | decided L / R / D / S | balanced on decided | wrong movement answers |
|---|---|---|---|
| sessions 1–30 | 90 / 91 / 81 / 97% | 86.11 → 88.08% | 22.3 → 15.3% |
| sub_01 31–36 | 80 / 83 / 50 / 95% | 69.19 → 68.08% | 34.0 → 32.0% |
| sub_11 | 89 / 71 / 86 / 91% | 69.45 → 73.02% | 33.5 → 28.1% |

Idea 4 is mixed: fewer wrong movement answers everywhere, but lower balanced accuracy on sub_01 31–36. Not adopted.

## What changed in the monitors

`background` and `background_valid` were added as RTAMT signals. The gate is now:

```
LOUD = once[0:767](background >= 6.61) or (historically[0:767](background_valid < 0.5) and once[0:767](contrast >= 7.81))
```

- **API:** `HeadMovementMonitor.classify(window, history)`. `StreamMonitor` passes the previous 5 s automatically.
- **Without history** the monitor is exactly the previous one (`python project_code.py evaluate`:
  82.74 / 67.22 / 77.00).
- **With history** the package gives 85.01 / 69.62 / 78.67 (`module_evaluate.log`). RTAMT agreed on 92/92 sampled windows.
- **Tests:** `tests/test_stream_background.py` (24).
