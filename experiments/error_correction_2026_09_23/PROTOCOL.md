# Frozen experiment plan

Target: at least 90% four-class balanced accuracy on nested session validation and both existing external splits. No promise that this is attainable. External splits are historically exposed, so not blind.

Current development errors include 176 movement windows called STRAIGHT and 143 LEFT/RIGHT windows called DOWN. Try:

1. Nonlinear direct classification (control).
2. Hierarchical movement / nod / direction classifiers, allowing different evidence at each decision.
3. Explicit error correction with a fifth KEEP class that preserves the current monitor answer.

For each, compare monitor measurements and threshold margins alone against those plus joint spectral power, temporal spectral changes and previously extracted shape/phase features. This distinguishes architectural gains from added-feature gains. No per-person constants or identity predictors.

Six fixed configurations. Each has 160 histogram boosting iterations, seven leaves, L2=10, leaf minimum=35, learning rate=.06, fixed seed23. No random-window early stopping. Equal class weights and equal recording weights within each class. Dependencies isolated under this experiment's vendor directory.

Three outer blocks of ten source sessions measure performance. Within each training block, two inner folds of alternating five-session groups choose configuration by balanced accuracy, subject to no movement-recall loss versus the fixed baseline. Baseline is a selectable option. All windows in a recording stay together. Existing baseline thresholds and historical feature choices are already development-informed; nested validation does not erase that history.

All six outer predictions are retained as exploratory ablations, but the reported pipeline uses only inner-selected choices. The final configuration is chosen through development-only inner folds, then written to pretest.json before external evaluation. Do not select a different model after seeing external results. No adoption unless nested balanced accuracy improves and both external sets show no loss in balanced accuracy or movement recall.

This tests ML-assisted corrections, permitted by the user's request to try any methods. It does not present those models as pure STL monitors. Production monitoring files remain untouched pending evidence of improvement.
