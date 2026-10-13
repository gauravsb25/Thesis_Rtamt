# Reporting the learned challenger even when baseline wins

Fixed before inspecting external learned-model results: also report the highest-scoring eligible
learned configuration from each development selection table, excluding baseline fallback.
Apply this inside outer folds as well as for final fitting, so learned-only nested validation is
not confused with validation of a pipeline that chose the baseline. Never select using external
results. Keep this forced challenger explicitly separate from the recommended selection procedure.
Freeze both modes' full-development formulas before any challenger external scoring.
This reporting supplement does not change learning, sample selection, feature preparation or grids.
