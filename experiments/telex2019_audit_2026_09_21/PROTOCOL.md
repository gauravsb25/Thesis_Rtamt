# TeLEx 2019-inspired bounded learner: preregistered experiment

Reference: Jha et al., FMSD 2019, https://susmitjha.github.io/papers/fmsd19.pdf (Sections 4–5).
Scope is a bounded reimplementation, NOT an exact reproduction of every paper heuristic.
Implement predicate tightness P(r)=1/(r+exp(-beta*r))-exp(-r), F/G time contraction/expansion,
max-min tightness across positive traces, and augmentation using subformula robustness signals.
Enumerate a finite temporal-bound grid, optimize predicate thresholds numerically, and build nested
F/G, bounded U, negations and AND/OR. Numerical threshold refinement is limited to shortlisted
templates. Bound formula size and depth. Exact min/max reductions, not a globally smooth solver.
Use closed sample-index intervals to match RTAMT, including for G (paper notation varies).

Two modes: positive-only ranking (no negative labels in formula fitting/ranking); supervised
extension (negative examples rank consistent formulas). Report both; multiclass decoding/validation
are supervised in both. No claim of global optimum, full 2019 replication, or exhaustive syntax.

Dataset: original 21,029 windows, 256 Hz, eight EEG channels, volts converted to microvolts.
Label-independent preprocessing: causal .3–6 Hz bandpass; per-window robust normalization; decimate
to 32 Hz after filtering; eight channels, all 28 differences, eight positive energy traces.
Completed-window normalization is not prefix causal. Time origin stays at window start: no label-
dependent recentering, class-specific preprocessing, or test-statistic normalization.

Independent trial key = subject/session/task/run, never run number alone. Sample one window from
each selected recording, assign each selected recording to only one class in a balanced subset.
Round-robin sessions; equal 8 or 16 examples per class, seeds 11 and 29. If there are too few
distinct recordings, reduce all classes equally and record the achieved count. Do not oversample
overlapping windows. Every discovery index, trial key and class is recorded. Entire sessions remain
together in train/validation. Sampling uses labels only inside the current training fold.

Three outer blocks of ten sub_01 sessions; two inner blocks choose sample budget, formula depth
(1/2/3) and direct four-class versus original movement gate + learned LEFT/RIGHT/DOWN.
Average fixed seeds for selection; choose neither seed using test scores. Baseline is eligible.
Main metric mean LEFT/RIGHT/DOWN recall over ALL actual movement windows, including misses called
STRAIGHT. Selection disallows more than a one-point STRAIGHT-recall loss in inner validation.
Report ordinary and four-class balanced accuracy, per-class counts, confusion matrices, fit versus
validation, subset stability, and trial/session counts. Formula scores are normalized by training
discovery robustness spread; argmax provides deterministic class choice even if none/all satisfy.
That decoding rule is an explicit extension, not part of original positive-example TeLEx.

Freeze the final model before scoring historically inspected external sub_01 31–36 and sub_11.
No test-driven sign reversal, parameter edits or adoption. Production remains unchanged unless a
validated candidate demonstrates benefit without material per-class damage. Verify STL evaluator
against RTAMT on toy and actual traces, including nesting and boundaries; save runnable raw-window
experimental monitor, ASTs and readable formulas. All writes stay in this directory.
