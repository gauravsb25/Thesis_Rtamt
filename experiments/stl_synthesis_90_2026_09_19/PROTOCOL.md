# Frozen search plan (before this experiment's results)

Target: >=90% four-class balanced accuracy, all windows, no abstention. Report development
cross-validation separately from held-out same-subject sessions and the second subject.

Use only sub_01 sessions 1–30 for formula selection. Baseline and existing feature definitions
have prior design exposure; the held-out sets have historical exposure too. Do not claim blind
validation. Do not alter the production monitors unless a frozen rule passes both held-out sets.

Build new scale-free temporal signals for all 28 electrode pairs: sustained positive/negative
excursions, ordered biphasic and triphasic robustness, and signed channel-trajectory circulation.
Combine these with existing scale-free features, never metadata or labels as predictors.

Search explicit Boolean STL formula structures, with up to 1, 2, or 3 added predicates per stage,
for movement, nod, and direction. This is supervised STL structure/parameter synthesis, not
hand-written rules and not an independent guarantee against overfitting. No neural network,
decision tree, or other separately trained classifier is used to make decisions.

Outer validation: session blocks 1–10, 11–20, 21–30. Inside each training fold, use two inner
folds whose validation subsets comprise alternating five-session blocks (by session index)
to choose predicate budget from {1,2,3}. Feature thresholds use inner-training data only.
Choose the smallest budget on ties. Rank final budgets from inner-validation results, never from
held-out subjects. Candidate feature screening and formula search stay inside each fit.

Baseline is an available outcome. Penalize each added predicate by 0.20 percentage points during
training optimization. A beam search explores AND/OR formulas; conjunction/disjunction with
complementary predicates supports exceptions and alternative sufficient conditions.

Before external evaluation require >=0.5 point pooled outer-validation gain, more development
sessions improved than harmed, and positive gain in at least two of the three validation blocks.
Freeze one full-development fitted specification and hashes before evaluating it once on both
held-out sets. Keep only if neither held-out split loses accuracy. A development result >=90%
alone does not establish the requested cross-subject target.

Verify optimized feature operators and exported formulas against RTAMT, and report paired
session-bootstrap intervals. Completed-window analysis is not a validated early/live monitor.
