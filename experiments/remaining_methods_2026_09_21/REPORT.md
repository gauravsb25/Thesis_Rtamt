# What can help: independent comparison, 21 September 2026

**Outcome: continuous history reproduces a modest improvement. None of the six learned replacements improves both external populations. Keep the original monitor.**

## New classifier tests

All figures in this first table are balanced accuracy (%), the mean recall across four classes. Training fit is not validation. External populations have been examined before and are retrospective tests.

| Method | Training fit | Nested session validation | External sub_01 | External sub_11 |
|---|---:|---:|---:|---:|
| Original RTAMT | 85.01 (development baseline) | 85.01 (fixed comparator) | 69.62 | 78.67 |
| original_ridge | 85.43 | 84.72 | 67.77 | 65.44 |
| operator_ridge | 89.65 | 82.44 | 65.61 | 51.68 |
| operator_lda | 93.41 | 86.88 | 65.75 | 62.54 |
| nonlinear_ridge | 92.05 | 83.04 | 65.78 | 53.20 |
| hybrid_gate | 93.60 | 89.38 | 69.87 | 62.86 |
| polarity_invariant | 92.36 | 69.15 | 52.28 | 55.79 |

Ordinary external accuracy and counts:

| Method | sub_01 correct / total | Accuracy | sub_11 correct / total | Accuracy |
|---|---:|---:|---:|---:|
| Original RTAMT | 1081 / 1269 | 85.19% | 1006 / 1200 | 83.83% |
| original_ridge | 987 / 1269 | 77.78% | 736 / 1200 | 61.33% |
| operator_ridge | 1007 / 1269 | 79.35% | 665 / 1200 | 55.42% |
| operator_lda | 1067 / 1269 | 84.08% | 837 / 1200 | 69.75% |
| nonlinear_ridge | 1013 / 1269 | 79.83% | 660 / 1200 | 55.00% |
| hybrid_gate | 1089 / 1269 | 85.82% | 862 / 1200 | 71.83% |
| polarity_invariant | 970 / 1269 | 76.44% | 754 / 1200 | 62.83% |

## What each method tested

1. **Original-signal ridge:** learns a heavily regularized combination of the eight current signals.
2. **Operator ridge:** learns combinations from the 1,100-feature bank plus the original signals, with training-only selection of 32 or 128 features.
3. **Shrinkage LDA:** classifies using class means and a regularized shared covariance matrix.
4. **Nonlinear ridge:** uses 256 fixed-seed random Fourier features to approximate nonlinear combinations of 32 selected signals.
5. **Hybrid gate:** preserves the original movement/no-movement decision and learns LEFT/RIGHT/DOWN. This was the best development family but loses 15.81 balanced-accuracy points on sub_11.
6. **Polarity-invariant inputs:** removes original turn polarity and keeps magnitude/timing summaries. This loses useful direction information and fails badly.
7. **Corrected response rule:** separately reruns the original single-rule search after fixing the no-trigger case.
8. **Uncertainty rejection:** freezes margin thresholds from training and reports accuracy alongside retained coverage.
9. **Continuous history:** independently reruns the previous paired-window evaluation on cached full-history measurements.

Six classifier families used 22 fixed parameter configurations and 138 fits, taking 54.0 seconds for the recorded model/evaluation run. Feature extraction was reused from the existing bank, not rerun or included in that time. These are small matrix-based models, not long neural-network trainings. Separate response and continuous-history checks took additional time.

## Corrected response result

Original feature: 85.16% development CV; corrected feature: 84.37%; original monitor: 85.01%. The correction fixes the stated semantics but does not improve this classifier search. No external tuning or evaluation was needed for this failed family.

## Continuous history: the useful change reproduced

The same windows and labels were paired to the larger recording cache. Labels agree on all 21,029 windows; 444 waveform spot checks agree. This reruns cached measurement evaluation, not all raw filtering or RTAMT trace evaluation. The existing fingerprint validates the original cache; the historical full-history cache has weaker provenance and merits a fresh raw-stream run before publication.

| Split | Old balanced | Continuous-history balanced | Old ordinary | Continuous-history ordinary | Correct with continuous history |
|---|---:|---:|---:|---:|---:|
| Development | 85.01% | 85.83% | 95.47% | 95.89% | 17798 / 18560 |
| External sub_01 | 69.62% | 71.19% | 85.19% | 86.13% | 1093 / 1269 |
| External sub_11 | 78.67% | 79.48% | 83.83% | 84.92% | 1019 / 1200 |

This supplies missing past context; it does not change the monitor formulas or justify claiming 90%. A live implementation must maintain past samples and reset history at genuine recording boundaries. Do not silently replace published v2-history results with this different evaluation protocol.

## Abstention does not solve the problem

For the hybrid at the training 25th-percentile margin threshold, sub_01 reaches 91.97% ordinary accuracy but retains only 67.69% of windows (790/859 accepted predictions correct). Its accepted balanced accuracy is 79.21%. On sub_11 it retains 62.50%, with 72.13% ordinary and 63.99% balanced accuracy. This is not a >90% full-coverage monitor. All thresholds and per-class accepted counts are in results.json. Margins are uncalibrated model scores, not confidence probabilities.

## Remaining avenues and their status

| Avenue | Evidence/status | What would constitute a fair next test |
|---|---|---|
| Continuous recording history | Modest improvement reproduced above | Recompute from raw recordings and verify live history continuity |
| More EEG users and sessions | Not testable with the current two-user dataset | Train on multiple users; reserve entirely new users and sessions before feature selection |
| Consistent electrode placement/reference and recorded acquisition quality | Plausible acquisition investigation; cause of polarity changes is unresolved | Record placement/reference metadata and repeat sessions without label-based sign flipping |
| Better movement labels and independent timing review | Prior audits indicate timing ambiguity; no corrected ground truth supplied | Review entire recordings with a fixed labeling policy, blind to candidate errors |
| Small labeled user calibration | Different deployment objective; earlier personalized model was rejected | Only revisit with explicit acceptance of calibration and evaluate later sessions chronologically |
| Session normalization/domain adaptation | Earlier methods did not establish transferable gains | Fit adaptation from permitted past or unlabeled calibration data only, then test new users |
| More temporal operators, channel pairs, fallback chains | Extensively tested previously; corrected response also fails here | New hypothesis and fixed validation protocol; avoid repeatedly choosing from exposed external scores |
| Other nonlinear models/deep learning | Some earlier configurations failed; this is not proof all architectures fail | More independent recordings and users before larger model searches |
| Event detection plus command confirmation | Earlier command study reduced false commands, with extra delay | Evaluate event recall, false commands/minute and latency; do not substitute these for window accuracy |
| Reject uncertain windows | Tested above; coverage loss and poor new-user performance | Predefine acceptable rejection/delay costs, use development calibration only |
| Alternative sensors | Outside current EEG-only scope | Not tested or proposed as required here |

## Validation limits and isolation

Feature ranking, centering, scaling, parameter selection and model fitting use training folds only. Entire sessions stay together. Models were frozen before external metrics were calculated. Original thresholds and bank designs were developed in earlier work, so nested validation does not undo historical design exposure. The six-family comparison is not a new-user generalization guarantee or an exhaustive search of all methods.

Class/session weighting reduces class and session dominance but does not make overlapping windows independent. Reported uncertainty resamples sessions. A pooled interval containing gains is not proof of transfer to a new person; the external failure is material. Preprocessing within complete windows is not necessarily sample-by-sample causal.

Input and production-source hash changes during the classifier run: []. No shared caches, production monitors or other experiment files were written. Source snapshots and frozen models are experimental artifacts only. The source snapshot signals.py has its original extraction entry point; do not run it directly.

Reproduce: `python -B experiments\remaining_methods_2026_09_21\run.py` (from project root). Full confusion matrices, recalls, paired confidence intervals, fit selections and predictions are saved locally.
