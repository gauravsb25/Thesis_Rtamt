# Targeted error correction: measured results

**90% balanced accuracy achieved across required evaluations: False. Adoption criteria passed: False.**

No production monitor was changed by this experiment. These are ML research candidates, not pure STL formulas.
Nested validation chooses models using inner training folds, then measures on separate outer sessions. The model and feature choice are therefore part of the evaluated procedure.

| Evaluation | Correct / total | Ordinary accuracy | Balanced accuracy | Movement macro recall |
|---|---:|---:|---:|---:|
| Development baseline | 17719/18560 | 95.47% | 85.01% | 80.70% |
| Nested validation | 18036/18560 | 97.18% | 88.76% | 85.29% |
| Final training fit (not validation) | 18554/18560 | 99.97% | 99.99% | 100.00% |
| sub01_later baseline | 1081/1269 | 85.19% | 69.62% | 60.39% |
| sub01_later candidate | 1095/1269 | 86.29% | 71.50% | 62.59% |
| sub11 baseline | 1006/1200 | 83.83% | 78.67% | 73.08% |
| sub11 candidate | 783/1200 | 65.25% | 67.69% | 72.83% |

Paired session-bootstrap 95% interval for nested balanced-accuracy gain: **+0.92 to +6.72 percentage points**. This resamples fixed predictions and does not include uncertainty from refitting or historical method selection.

## Exact class counts

| Evaluation | LEFT | RIGHT | DOWN | STRAIGHT |
|---|---:|---:|---:|---:|
| Development baseline | 805/959 (83.94%) | 770/963 (79.96%) | 585/748 (78.21%) | 15559/15890 (97.92%) |
| Nested validation | 826/959 (86.13%) | 830/963 (86.19%) | 625/748 (83.56%) | 15755/15890 (99.15%) |
| Final training fit (not validation) | 959/959 (100.00%) | 963/963 (100.00%) | 748/748 (100.00%) | 15884/15890 (99.96%) |
| sub01_later baseline | 69/117 (58.97%) | 118/183 (64.48%) | 71/123 (57.72%) | 823/846 (97.28%) |
| sub01_later candidate | 72/117 (61.54%) | 112/183 (61.20%) | 80/123 (65.04%) | 831/846 (98.23%) |
| sub11 baseline | 105/187 (56.15%) | 165/221 (74.66%) | 252/285 (88.42%) | 484/507 (95.46%) |
| sub11 candidate | 124/187 (66.31%) | 137/221 (61.99%) | 257/285 (90.18%) | 265/507 (52.27%) |

## Nested choices

| Outer held-out sessions | Inner-selected method | Baseline BA | Candidate BA |
|---|---|---:|---:|
| 1-10 | staged:combined | 83.98% | 87.18% |
| 11-20 | direct:combined | 83.51% | 89.18% |
| 21-30 | direct:combined | 87.14% | 89.69% |

Final development-only selection: `staged:combined`. Frozen before external scoring.

## Exploratory ablations (not substitutes for the nested result)

| Fixed model | Outer balanced accuracy | Movement recall |
|---|---:|---:|
| direct:monitor | 80.85% | 75.49% |
| direct:combined | 88.89% | 85.43% |
| staged:monitor | 80.76% | 75.14% |
| staged:combined | 88.87% | 85.50% |
| correct:monitor | 82.81% | 78.17% |
| correct:combined | 86.48% | 82.28% |

Post-hoc direct-classifier ablation without the 240 spectral features: **87.43% balanced accuracy**, versus **88.89%** with them. This diagnostic was not used to change model selection or the external candidate.

## Corrections and regressions in nested validation

| True class | Existing errors fixed | Previously correct decisions broken |
|---|---:|---:|
| LEFT | 55 | 34 |
| RIGHT | 83 | 23 |
| DOWN | 66 | 26 |
| STRAIGHT | 256 | 60 |

## Scope

The actual benchmark took 37.4 minutes, excluding dependency setup. It used 48 inner configuration fits, 18 outer configuration fits and one final configuration fit if selected. Staged configurations fit three separate classifiers each; every classifier uses 160 boosting iterations.
Previously exposed external data cannot establish blind generalization. Source-session feature design and baseline thresholds also predate this experiment. Nested splitting prevents new window leakage but does not erase prior exposure.
Dependencies are installed only in this experiment/vendor folder. The fixed settings and selection rule are recorded in PROTOCOL.md.
A failed result does not prove 90% is impossible. A favorable training or single-fold score does not meet the target.
