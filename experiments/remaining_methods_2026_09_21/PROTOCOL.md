# Frozen protocol before running

Scope: finite comparison of six classifier families and an abstention analysis using existing EEG.
This is not an exhaustive test of all conceivable algorithms. No production or other experiment edits.

Inputs: original eight measurements, genuine RTAMT cached labels, and existing 1,100-feature operator
bank. Check fingerprint, row alignment, finite values, and arithmetic baseline agreement. Record hashes.
Correct response features to zero when max(signal)<2 (no trigger, vacuous implication). Keep original
bank on disk unchanged. Use snapshot code provenance and record any concurrent source changes.

Families: ridge on original signals; ridge on operator bank; shrinkage LDA on operator bank;
random Fourier nonlinear ridge on operator bank; hybrid preserving original movement gate with
learned three-way direction; polarity-invariant ridge using magnitude/timing features only.
Each bank includes original eight features except polarity-invariant, which excludes original turn.
All learned models are experimental ML classifiers, not automatically native RTAMT specifications.

Fit class-balanced weights with equal session weight inside each class. Training-only standardization
and supervised feature ranking. Small fixed hyperparameter grids; no test-derived tuning.
Development: sub_01 sessions 1-30. Three blocked outer folds (ten sessions each). Within each outer
training set, choose parameters by two inner blocked folds. Full model parameters are chosen by
three-fold development CV. Report training fit separately from nested out-of-fold validation.
All families are frozen before any external scores are calculated. External splits: sub_01 31-36,
sub_11 1-10. These have been examined historically; call these retrospective external evaluations.

Report ordinary accuracy, balanced accuracy, correct/total, per-class recall, confusion matrices,
session changes, paired session bootstrap intervals (2,000 replicates) and external predictions.
No independent-window significance tests. Bootstrap intervals are descriptive, not selection-adjusted.
Potential candidate requires >=0.5 point nested CV gain, more sessions improved than harmed, then
improvement on both external splits; no automatic production adoption even then.
Abstention: fixed training-derived margin quantiles, print coverage and class counts; never call
accuracy on accepted windows full-coverage accuracy. No personal calibration or new sensors in this run.
