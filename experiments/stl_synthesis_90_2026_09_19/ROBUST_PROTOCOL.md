# Development-only continuation, fixed before running the balanced variants

The first 703-signal search has finished. Its held-out results are now known. This continuation
will not score those sets again or use them as an optimization criterion.

Hypothesis: per-window class weighting gives movements with many valid overlapping windows and
large sessions disproportionate influence. Test equal weight per session-within-class and equal
weight per original recording-within-class, retaining all windows in evaluation. This is training
weighting, not exclusion or replacement of the four-class balanced-accuracy target.

Keep the same 703 features and AND/OR grammar. Test both gate-nod-direction and nod-direction-gate
optimization order, with predicate budgets 1, 2, and 3. Include all three original window-weighted
fits as options. Select among these 15 configurations using inner session validation only, and
score the selected pipeline on the three outer session blocks. Baseline is eligible.

All group metadata is used only to assign training weights and evaluate groups; it is never an
input to a formula. Reuse cached fits with identical training folds and configuration. The new
search has development-result exposure from preceding rounds; it is exploratory, even though
each particular fit excludes its validation sessions. No claim of untouched validation is made.

Report ordinary four-class balanced accuracy on all 18,560 development windows, per-session
results, per-recording-within-class metrics, and paired session-bootstrap uncertainty. Export and
verify the frozen formula for inspection, but leave production unchanged.
