# Substantial STL synthesis search toward 90% — 19 September 2026

**The 90% criterion was not reached on pooled session-held-out validation or on either external
split.** The strongest complete pipeline reached **87.26% development balanced accuracy**, up from
85.01%. It reached 90.18% in one validation block and 91.21% on training fit; neither is a 90%
generalization result. The production monitor is unchanged.

## What was done

- Combined **703 signal measurements**, including **280 new temporal signals** for all 28 electrode
  pairs, with prior scale-free spectral, shape, spatial, history, and monitor measurements.
- New signals include sustained signed excursions, ordered two- and three-lobe STL robustness,
  and signed circulation of each two-channel trajectory. No subject/session identity, run index,
  marker, annotation, or label is a predictor.
- Searched explicit AND/OR specifications for three stages: movement, nod-versus-turn, and direction.
  Each stage can retain the existing rule, replace it, add an exception, or add a sufficient condition.
  A four-candidate beam searches up to three new predicates per stage. A 0.20-point-per-predicate
  training penalty discourages unnecessary complexity. Thresholds and feature screening use
  training data only.
- Compared window-, session-, and recording-balanced training objectives and two stage-fitting
  orders. The expanded search contains **15 configurations**; the final continuation examines
  **105 configuration/stage-subset options plus baseline**.
- Completed **100 saved training fits**, with their explicit formulas retained for inspection.
- Used two inner session folds to select configurations and three outer ten-session blocks to
  evaluate them. No overlapping windows from the same session cross a fold boundary.
- Exported three frozen experimental specifications and checked each against real RTAMT on over
  1,100 development windows. The optimizer's Boolean decisions matched RTAMT exactly on those checks.

This is **supervised STL structure/parameter synthesis**: the formulas are fitted using labels,
not manually written. No separately trained neural network or decision-tree classifier supplies
the decisions. Keeping an explicit formula does not eliminate statistical overfitting.

## Results on all 18,560 development windows

Metric throughout: four-class balanced accuracy. No abstention or removal of difficult windows.

| Pipeline | Outer-session validation | Training fit, not validation |
|---|---:|---:|
| Existing monitor | 85.01% | — |
| First synthesis search | **87.26%** | 91.21% |
| Inner-selected run/session balancing and fitting order | 86.52% | 92.73% |
| Inner-selected subsets of changed stages | 86.54% | 89.61% |

First search, by outer validation block:

| Sessions held out | Existing | New |
|---|---:|---:|
| 1–10 | 83.98% | 86.60% |
| 11–20 | 83.51% | 84.49% |
| 21–30 | 87.14% | **90.18%** |

The first search improves 17 sessions, harms 11, and ties 2. Its pooled gain is **+2.26 points**;
a paired session bootstrap gives a gain interval of **−1.49 to +5.54 points**. This resamples fixed
out-of-fold predictions; it does not refit selection or measure uncertainty on another person.

| Class, development | Existing recall | First search recall |
|---|---:|---:|
| LEFT | 83.94% | 85.92% |
| RIGHT | 79.96% | 83.59% |
| DOWN | 78.21% | 83.16% |
| STRAIGHT | 97.92% | 96.38% |

Balanced accuracy improves while the raw number of errors increases: 203 windows are corrected
and 357 become wrong, largely because STRAIGHT is much more numerous. This is a real trade-off,
not a contradiction; both facts matter for a monitor that must avoid false alarms.

## One frozen external evaluation

The first search passed the preset development eligibility rule: >=0.5-point gain, more sessions
improved than harmed, and improvement in at least two outer blocks. Its full-development formula
was saved and hashed before a single evaluation on both external splits.

| Split | Existing | Frozen new formula | Change |
|---|---:|---:|---:|
| sub_01 sessions 31–36 | 69.62% | **73.94%** | +4.33 points |
| sub_11 | **78.67%** | 72.43% | −6.24 points |

It is **not adopted** because it loses on sub_11. The later balancing and pruning searches were
development-only and did not evaluate these external splits again.

The main transfer failure is specific: sub_11 STRAIGHT recall falls from **95.46% to 80.08%**.
DOWN recall falls from 88.42% to 81.75%; RIGHT falls from 74.66% to 70.14%, while LEFT rises
slightly from 56.15% to 57.75%. The added movement rescue is too permissive on the other subject.
On later sessions of sub_01, DOWN recall rises from 57.72% to 76.42%, explaining much of the gain.

## What the selected formula actually says

The first full-development fit chooses one predicate change per stage. Let `MOVE0` be the
existing movement decision and `LEFT0` the existing direction sign:

```text
MOVE = MOVE0 OR (Cz power share in 0.5–1 Hz >= 0.0239700761)
NOD  = loud-window yaw energy ratio < 3.139851904
LEFT_DIRECTION = LEFT0 AND (ordered Fp1−Fp2 evidence >= −1.651470196)

STRAIGHT = NOT MOVE
DOWN     = MOVE AND NOD
LEFT     = MOVE AND NOT NOD AND LEFT_DIRECTION
RIGHT    = MOVE AND NOT NOD AND NOT LEFT_DIRECTION
```

The yaw ratio is an earlier project's signal: temporal and frontal differential energy divided
by midline energy in the loudest second. Its earlier single-veto failure did not prevent it
fitting development well in this larger combination.

The new ordered evidence compares opposite waveforms. At 32 Hz, with pair signal `x` normalized
by its robust spread:

```text
P = historically[0:1](x >= 0)
N = historically[0:1](-x >= 0)
R(x) = robustness of once[0:62](P AND once[2:32](N)), evaluated at sample 95
ordered evidence = R(x) − R(−x)
```

This requires a sustained lobe preceded by a sustained opposite lobe within one second, and
excludes incomplete history. The difference of two robustness values is a derived signal;
the class specification tests that signal. The exported JSON has exact executable predicates.

## Which changes really helped?

A post-hoc development ablation of the first outer-fold fits gives:

| Retained stage changes | Balanced accuracy |
|---|---:|
| Gate only | **87.75%** |
| Nod only | 84.83% |
| Direction only | 84.77% |
| Gate + nod | 87.37% |
| Gate + direction | 87.51% |
| All three | 87.26% |

This identifies movement detection as the strongest component. **87.75% is an exploratory ablation,
not a separately inner-selected pipeline score.** When stage subsets were subsequently selected
inside the inner folds, the full procedure scored 86.54%. Looking at the validation results and
keeping the best subset afterward would exaggerate how reliably we can select it.

The extra temporal information was usable by the search, but adding a nested direction predicate
did not itself produce the overall gain. More complex formulas were not consistently better.

## Verification and limitations

- First, balanced, and pruned frozen formulas matched RTAMT on **1,183**, **1,178**, and **1,184**
  development windows, respectively, covering each class and changed decisions.
- All 280 new signals passed positive common-gain scaling checks on random and real windows;
  polarity transformation and finiteness checks also passed. Three optimized-search/AST checks passed.
- The earlier nested-operator tests check bounded extrema against brute force and representative
  two-/three-lobe robustness against RTAMT. These are reused by this feature extractor.
- Signal arrays are float32. When exporting an exactly representable predicate threshold, its
  numerical boundary is placed between adjacent float32 values to preserve Boolean decisions
  without ambiguous zero-robustness ties.
- These are completed-window experiments. Whole-window normalization and aggregate measurements
  do not establish an early causal implementation. No live latency claim is made.
- Only two subjects exist. Prior studies exposed the baseline, feature definitions, and researchers
  to these datasets; further development searches are exploratory. The external sets have also
  been reused historically. No result here is claimed to be blind cross-subject validation.

**Conclusion: more work found a measurable development improvement and a same-subject held-out
improvement, but did not establish 90% four-class accuracy or improve transfer to the second subject.**
Neither the training scores above 90% nor one favorable validation block should be substituted
for the requested criterion. This is not proof that 90% is impossible.

## Reproduction and saved evidence

```powershell
python experiments/stl_synthesis_90_2026_09_19/features.py
python experiments/stl_synthesis_90_2026_09_19/run_search.py
python experiments/stl_synthesis_90_2026_09_19/run_robust.py
python experiments/stl_synthesis_90_2026_09_19/run_pruned.py
python experiments/stl_synthesis_90_2026_09_19/audit.py
```

Search plans: PROTOCOL.md, ROBUST_PROTOCOL.md, PRUNING_PROTOCOL.md. Results: results.json,
robust_results.json, pruned_results.json, audit.json. All out-of-fold predictions, per-fit ASTs,
feature caches, logs, and frozen specifications are retained. Cached identical fits are reused.
Re-running run_search.py reproduces its fixed external evaluation; it should not be used to tune
against that evaluation. Production files and the original data were not modified.
