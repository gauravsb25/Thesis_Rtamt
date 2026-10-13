# Improvement round, 16 September 2026 — CUSUM gate and SINCE direction (not adopted)

**Outcome in one line.** Two new STL techniques improved the four monitors on the design sessions
(82.30% → 83.50%) but not on the held-out sessions (sub_01 31–36: 66.28% → 65.05%; sub_11:
76.40% → 75.48%). The package default therefore stays the previous monitor; both techniques remain
available as `RtamtConfig` options so this comparison can be reproduced through the real code.

## Protocol

| role | data | windows | movements |
|---|---|---|---|
| design (the only data looked at while designing) | sub_01 sessions 1–30 | 20,373 | 252 |
| test, scored **once** after the design was frozen | sub_01 sessions 31–36 | 1,365 | 37 |
| test, scored **once** after the design was frozen | sub_11 sessions 1–10 | 1,426 | 95 |

- No machine learning. The monitors are RTAMT STL formulas; thresholds are numbers in fixed rules,
  chosen by sweeping them and counting correct windows.
- No per-subject or per-session constants.
- Every accuracy is produced by the four monitors alone.
- Metric: 4-class balanced accuracy (mean of LEFT/RIGHT/DOWN/STRAIGHT recall).
- `common.py` refuses to load a test session unless `FINAL_TEST=1` is set.
- Design choices were cross-validated on three contiguous session blocks (1–10, 11–20, 21–30):
  thresholds chosen on two blocks, scored on the third, so a technique had to survive
  between-session drift.

## What the design data showed (sessions 1–30)

- Baseline 82.30%. Errors come per movement, not per window: 48 of 252 movements misread
  (15 turns called DOWN, 13 LEFT/RIGHT swaps, 10 nods missed, 5 nods called a turn, 5 turns missed).
- In the `straight` task files, where nothing is annotated, 11.6% of windows are still called DOWN:
  these false DOWNs are brief resting events, not unannotated head movements.
- The nod evidence drifts because the **nods themselves** change between sessions (session spread of
  DOWN medians 0.75/1.04 nats for balance/symmetry, lateral 0.42/0.53); resting windows do not track
  it (correlation 0.07), so no within-window normaliser can remove it.
- LEFT/RIGHT errors sit on near-tied lobes: 96.5% correct when the smaller F4–T3 lobe is under half
  the larger, 68.9% when it is within 15%.
- The recordings are low-pass filtered near 45 Hz; the 20–45 Hz (EMG) band carries no safe direction
  information (best LEFT/RIGHT AUC 0.705, on T4).

## Techniques tried (cross-validated on sessions 1–30; refit baseline 80.55%)

| family | best result | verdict |
|---|---|---|
| IIR filter initial conditions (`lfilter_zi`) | 77.53 (−3.02) | rejected |
| spatial whitening (Mahalanobis energy, covariance of the quietest 1–1.5 s, shrinkage 0–1) | 80.88 (+0.33, diagonal only); full covariance −0.75 to −1.47 | rejected |
| derived-component gates (F4−T3, (Fp1+Fp2)/2−Cz, common mode, Fp1−Fp2) | −1.33 to −1.61; as an OR route −0.03 to −0.46 | rejected |
| PC1 coherent-energy contrast | −0.85 | rejected |
| median pre-filter to suppress blinks (0.3/0.5 s, centred/causal) | gate −3.9 to −22, nod −0.2 to −1.9, direction −4.2 to −12.4 | rejected |
| DOWN evidence: F3/F4 correlation | +0.93 at 1.0 s but −0.9/−1.0 at 0.75/1.5 s | rejected (unstable) |
| DOWN evidence: PCA topography, global common mode, excess-energy balance, per-channel activation, vertical/horizontal activation, Fp1/Fp2 correlation | −2.2 to −7.6 | rejected |
| DOWN evidence: T3/T4 correlation | +1.26 | excluded: depends on T4 alone |
| two-tier gate / two-tier DOWN / confidence-aware arbitration (7 secondary evidences), built on the CUSUM gate | −0.60 to +0.02 against the CUSUM gate alone | rejected |
| signed common-mode shape (sign, order, polarity agreement) | AUC 0.43–0.75 | rejected |
| ordinal symmetry (sign agreement, Spearman) | ≤ Pearson | rejected |
| biphasic order reads (argmax/argmin order, ramp, fixed-gap STL order) | best 86.7% vs peak 90.4% LEFT/RIGHT | rejected |
| **CUSUM gate** (Page's CUSUM of log-energy above the quietest 0.5 s, κ = 2 nats) | +0.4 to +0.73; better in 20–22 of 30 sessions; not an edge artifact | **selected on design data** |
| **SINCE direction** (`(not (lateral >= h)) since (lateral <= -h)`, h = 0.95) | +0.69 with thresholds frozen; movements 4 fixed / 1 broken | **selected on design data** |

With the one-standard-error rule inside each fold, CUSUM + SINCE scored 82.85% against 81.72%, better
in all three folds. The CUSUM threshold started at the contrast gate's false-alarm rate (0.0726) and the
one-standard-error rule moved it to 0.0684. The real module reproduced the selected design on all 20,373
design windows (0 differing decisions, `dev_verify.log`).

## Final test (run once; `final_test.log`)

| 4-class balanced | sessions 1–30 (design) | **sub_01 31–36** | **sub_11** |
|---|---|---|---|
| previous monitor (contrast gate, peak) | 82.30% | **66.28%** | **76.40%** |
| CUSUM gate only | 82.81% | 66.08% | 76.16% |
| SINCE direction only | 82.99% | 65.26% | 75.72% |
| CUSUM gate + SINCE direction | 83.50% | 65.05% | 75.48% |

- SINCE broke one RIGHT turn in each held-out set (sub_01 ses-35, sub_11 ses-06) and fixed none.
  Together with the design data that is 4 movements fixed and 3 broken: no evidence it helps.
- CUSUM changed 23 held-out windows on sub_01 31–36 (11 fixed, 12 broken) and 22 on sub_11
  (8 fixed, 12 broken): a wash.
- RTAMT agreed with the scored decisions on 160 of 160 sampled test windows.
- **Not adopted.** The package default is unchanged: formulas identical, and 48 real windows gave
  identical labels with zero robustness difference.

Where the previous monitor loses on the held-out data:
- **sub_01 31–36:** LEFT→DOWN 46/127, RIGHT→DOWN 53/195, DOWN→STRAIGHT 39/123. This is the
  nod-evidence drift again, larger in these late sessions.
- **sub_11:** LEFT→RIGHT 62/187 and RIGHT→LEFT 51/221, mostly the two polarity-inverted sessions
  (ses-08 48.7%, ses-09 57.2%).

Neither can be fixed from a single 3-second window with these electrodes. The first needs
session-level (stream) adaptation; the second needs labels or another subject.

## Files

- `common.py` — split guard and metrics. `cv.py` — cross-validated cascade scorer.
- `ship.py` — vectorised copy of the shipped measurements (matches the monitor exactly).
- `look_waveforms.py`, `feats1.py`, `atlas.py`, `events.py`, `events_zoom.py` — diagnosis.
- `cands.py`, `cands2.py`, `medf.py`, `dir1.py`, `tiebreak.py`, `since_dir.py`, `edge_check.py`
  — candidate statistics.
- `run_cv1.py` … `run_cv4.py` with `cv1.log` … `cv4.log` — cross-validation rounds.
- `cv5.log`, `cv6.log`, `cv7.log` — rounds run inline (SINCE in the cascade, confidence-aware
  arbitration, quiet-normalised CUSUM). Their code is not saved separately.
- `final_select.py` / `.log` — final design comparison with the one-standard-error rule.
- `dev_verify.py` / `.log` — the module reproduces the selected design on the design sessions.
- `final_test.py` / `.log` — the one-time test.

Rerunning writes caches next to the scripts. `cache_dev.npz` alone is about 470 MB.
