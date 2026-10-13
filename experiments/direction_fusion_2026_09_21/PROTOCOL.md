# Direction-focused fusion protocol, fixed before execution

Preserve the original movement gate and STRAIGHT predictions exactly. Evaluate all true LEFT,
RIGHT and DOWN windows, including movements missed as STRAIGHT; headline is mean of their three
recalls, not accuracy conditional on the detector already being right. Also show per-class counts,
ordinary accuracy, four-class balanced accuracy and false movement counts.

Test five architectures: nod-only specialist, turn-only specialist, nested nod-then-turn specialists,
agreement between a three-way expert and the nested specialists, and baseline-prior score fusion.
Specialists use class/session-weighted ridge, training-only standardized/ranked operator features,
32 or 128 selected features, regularization 100. No new subject labels or subject-specific signs.
Confidence filtering uses training-score margin quantiles 0, .5 and .75. Prior fusion gives the
baseline class an added score of .25, .5 or 1. Baseline itself is always an available selection.

Select using development movement macro recall with no individual movement recall falling more
than one point below baseline in inner validation. Three ten-session outer folds, two blocked
inner folds; select the final configuration using all three development folds. Select architecture
inside the outer folds too to report the whole selection procedure, not just the best family.
Freeze all configurations before external scoring. Historically inspected external splits are
retrospective, not blind tests. No tuning from their new scores and no automatic production adoption.
Adoption requires positive movement-macro change on both external populations and no movement
class losing more than one point on either; report confidence intervals and counts regardless.

No changes outside this experiment directory. Reuse the prior verified feature/cache-loading code
as a recorded snapshot; source hashes and cache fingerprint must match. Classifiers are ML/STL
hybrids, not a claim that linear scoring is a native RTAMT temporal operator. If rejected, retain
the executable experimental monitor for reproducibility but leave production unchanged.
