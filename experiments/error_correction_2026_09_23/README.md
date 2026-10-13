# Targeted corrections to the EEG monitor

**Measured outcome:** nested balanced accuracy improved from 85.01% to 88.76%. Later same-subject sessions improved from 69.62% to 71.50%, but the other subject fell from 78.67% to 67.69%. The 90% target and adoption criteria were not met. The production monitor remains unchanged. Training fit was 99.99% balanced accuracy; this is not evidence of generalization. See RESULTS.md for all counts and failures.

This experiment addresses missed movements and confusion between turns and nods. It compares three architectures, each with and without added spectral/shape features:

- **Direct nonlinear classifier:** a control for whether the existing decision boundaries are too rigid.
- **Staged classifier:** separate decisions for movement versus straight, nod versus turn, then left versus right. Each decision learns its own evidence and receives its own balanced training weights.
- **Correction classifier:** four alternative answers plus KEEP. KEEP probability is assigned to the existing monitor's answer. This lets the model preserve a correct decision instead of learning every case from scratch.

The simple feature set contains eight existing measurements, six threshold margins and the existing answer. The combined set has 324 features, including relative bandpower, ordered bandpower changes, and prior phase/shape measurements. It contains no identity, annotation, marker, run index or task predictor. Whole sessions, including all overlapping windows from their recordings, remain together.

The method family is [histogram gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html). Random-window early stopping is disabled. Settings and nested selection are fixed in PROTOCOL.md. This is ML-assisted research, not a claim that boosted trees are STL formulas.

## Reproduce from thesis_rtamt

Dependencies are isolated under this folder, rather than installed globally:

```powershell
python -m pip install --target experiments/error_correction_2026_09_23/vendor --no-deps scikit-learn==1.9.1 joblib==1.6.0 threadpoolctl==3.7.0 cloudpickle==3.1.2
python -B experiments/error_correction_2026_09_23/benchmark.py
python -B experiments/error_correction_2026_09_23/report.py
```

Verify that the frozen candidate reproduces every saved prediction:

```powershell
python -B experiments/error_correction_2026_09_23/predict_saved.py
```

This replay operates on cached features; it is not a new live EEG interface. All 21,029 predictions reproduced exactly in verification. Four implementation tests passed. The unchanged baseline fingerprint and input/fold audits are saved alongside the results.

NumPy and SciPy must already be available. The benchmark uses the earlier cached spectral, shape and monitor measurements in this workspace and asserts row/session alignment. It does not recompute or overwrite those files.

Print completed results without retraining:

```powershell
python -B experiments/error_correction_2026_09_23/report.py
```

Run implementation checks:

```powershell
python -B -m pytest experiments/error_correction_2026_09_23/test_benchmark.py -q -p no:cacheprovider
```

## Evidence files

- RESULTS.md and results.json: nested validation, training fit, external comparisons and class counts.
- predictions.npz: every baseline, nested, final and ablation prediction, plus truth/session/trial metadata.
- outer*_selection.json: inner selection scores, separately for each outer fold.
- final_selection.json and pretest.json: final choice made before external scoring.
- diagnostics.json: paired session-bootstrap interval and counts of errors fixed versus correct answers broken.
- run.log: actual fitting progress and elapsed times.
- candidate.pkl: experimental final fitted model, if a candidate rather than baseline was selected. It is not automatically installed as a production monitor.

The existing monitor has historical access to development data, and the external sets have been exposed in earlier experiments. Neither nested folds nor this experiment make the historical data blind again. Results are retrospective. Confidence intervals resample whole sessions from fixed predictions; they do not cover all uncertainty about a new subject.
