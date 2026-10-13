# New signal representations, not another channel sweep

This independent experiment implements and tests three different representations:

1. **Joint bandpower:** relative power within channels and spatial power shares across electrodes, in six bands: 1–4, 4–8, 8–13, 13–20, 20–30 and 30–45 Hz. There are 96 simultaneous features rather than one threshold at a time.
2. **Ordered spectral dynamics:** 144 additional features describing early-to-middle and early-to-late log-power changes and the range of power within each window. This preserves whether power rises or falls. It is not called ERD because there is no independent pre-event rest baseline in this feature extraction.
3. **Filter-bank common spatial patterns (CSP):** learn electrode combinations whose band-limited variance differs by class. Filters are fitted separately in each training fold, never on held-out recordings. See [MNE's time-frequency CSP example](https://mne.tools/stable/auto_examples/decoding/decoding_csp_timefreq.html) for the established method family. This implementation uses NumPy/SciPy, one-versus-rest regularized covariance, and the two extreme generalized eigenvectors per class and band.

These are tested alone, jointly with existing monitor measurements, and as direction classifiers behind the existing movement gate. A model using only the existing measurements is an essential control: it checks whether any gain actually comes from the new signals.

## Validation and limitations

- All 21,029 windows are included; no difficult examples are dropped. Only the first eight EEG channels are used. Markers, movement annotations, subject identity, session identity and task/run identity are not predictor features.
- Development uses sub_01 sessions 1–30. Three folds hold out entire blocks of ten sessions. Overlapping windows from a recording never cross a fold boundary. Within fitting, classes have equal weight, and recordings within each class have equal total weight.
- Seven fixed configurations, each with one fixed shrinkage LDA classifier, make 21 validation fits. There is no large neural-network training claim and no search for a favorable random window split.
- Select on development movement macro recall first, then balanced accuracy. Adoption additionally requires at least +0.5 movement points, no balanced-accuracy loss, more sessions helped than harmed, and no loss on either external split. The winner failed the development gate. Its external evaluation is explicitly diagnostic, not justification to adopt it.
- External splits are sub_01 sessions 31–36 and sub_11. They have been inspected in previous work; this is retrospective evidence, not a blind benchmark. Only the development-selected winner was scored there.
- Spectrograms use one-second complete frames with 0.25-second steps. Covariances use causal fourth-order bandpass filters, initialized at zero, with the first second discarded. Residual boundary transients, particularly in the lowest band, remain a limitation. Decisions use the complete three-second window, not instantaneous predictions.
- The first two feature families are scale-invariant up to numerical floors. Trace-normalized covariance removes global amplitude but is not invariant to separately scaling individual electrodes.
- These classifiers are research comparisons. Production RTAMT files, thresholds, shared caches and other experiments were not modified. A successful classifier would still require a separate, honestly labelled monitor integration and evaluation.
- Failed configurations do not establish an information-theoretic ceiling. Relative beta power is not by itself proof of neural origin. Earlier reports making either inference should not be treated as established findings.

## Run from thesis_rtamt

Print saved results without retraining:

```powershell
python -B experiments/spectral_patterns_2026_09_23/report.py
```

Reproduce the fixed experiment (several minutes for extraction; cached features speed subsequent runs):

```powershell
python -B experiments/spectral_patterns_2026_09_23/run.py
python -B experiments/spectral_patterns_2026_09_23/report.py
```

Check the signal-processing implementation:

```powershell
python -B -m pytest experiments/spectral_patterns_2026_09_23/test_spectral.py -q -p no:cacheprovider
```

Requires NumPy, SciPy and, for checks only, pytest. These were already available in this workspace.

## Files

- `RESULTS.md`: all measured development results, external diagnostic results and per-class correct/total counts.
- `results.json`: numerical results and confusion matrices, per-fold metrics and adoption decision.
- `selection.json`: selection written before external scoring.
- `oof_predictions.npz`: saved validation predictions with recording and session identifiers.
- `features.npz`, `inputs.json`: extracted features and source-file manifest. If data or feature extraction changes, regenerate the cache; it is not a general automatic cache invalidation system.
- `run.log`: extraction and evaluation output from the actual run.
- `test_spectral.py`: synthetic frequency/order/scale tests, recording-weight tests, and recovery of known covariance discrimination on held-out synthetic examples.

The tests check implementation behavior; they are not evidence of EEG classification accuracy. See RESULTS.md for that evidence.
