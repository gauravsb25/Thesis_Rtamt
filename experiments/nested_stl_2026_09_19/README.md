# New signals and nested STL: measured follow-up, 19 September 2026

The user's proposal was tested with **145 candidate rules including the baseline**, grouped into five
families. These are original extensions motivated by the project's own results; no literature search
was used in this round. The shipped monitors were not changed.

## What the earlier work establishes—and what it does not

The successful stream-background gate shows that reference choice matters. Earlier persistence,
phase, timing, magnitude, spatial, and raw-signal learning experiments often improved development
performance without transferring. This motivates testing **the order of sustained events**, rather
than another isolated peak or whole-window statistic.

Some earlier reports make stronger claims than their experiments justify. Failure of the tested
methods does **not** establish an information-theoretic accuracy ceiling, prove that no combination
can work, or prove that all subjects require incompatible rules. With two subjects, one used during
earlier design, we cannot establish a universal limitation or universal subject invariance.
Likewise, a poor covariance classifier does not prove that whitening necessarily removes all
movement-to-rest energy information. Those interpretations need separate evidence.

## Protocol

Data: extracted_npz_dataset_v2, **18,560 development windows**, sub_01 sessions 1–30.
Three validation blocks: sessions 1–10, 11–20, 21–30. Candidate parameters and family selection use
only the other two blocks. Whole sessions stay together. No labels, markers, movement annotations,
subject identifiers, or session identifiers enter the signal computations or classification rules.
Labels select a finite set of STL parameters on development data, as in the existing project.

The pre-existing baseline was previously designed on these sessions: this is an incremental
comparison, not a fresh unbiased validation of the baseline. Acceptance requires at least **0.5
percentage point** out-of-fold gain and more sessions improved than harmed. The combined search
must pass before one frozen candidate is tested on held-out data. **It did not pass; neither held-out
split was extracted or scored in this round.** Prior published test results informed the motivation,
so future testing on those same subjects would still not be blind validation.

## New signals and formulas

Raw channels are filtered causally at 0.3–6 Hz, sampled at 32 Hz, median-centered, and divided by
their own robust within-window spread. Pair direction uses normalized F4−T3. All durations below
are in **32-Hz samples**, not the production monitor's 256-Hz samples.

1. **Distributed burst:** fourth-largest absolute normalized channel value. A threshold requires
   at least four channels to support activity. Test sustained runs of 2, 4, or 8 samples, both as
   an additional veto and a rescue within the existing loudness gate.
2. **Quiet then burst:** a four-sample quiet period followed 8–32 samples later by sustained
   distributed activity. This tests temporal context, not just activity strength.
3. **Sustained midline evidence:** log ratio of Fz/Cz mean squared normalized amplitude to the
   median squared amplitude of F3/F4/T3/T4. Test as a replacement nod discriminator or extra veto.
4. **Two sustained opposite lobes:** compare robustness of negative-then-positive and
   positive-then-negative pair waveforms.
5. **Three sustained alternating lobes:** compare negative-positive-negative with its inverse.

For normalized pair signal `x`, persistence `d`, and maximum lag `g`, define:

```text
P = historically[0:d-1](x >= 0)
N = historically[0:d-1](-x >= 0)
two = P and once[d:g](N)
three = N and once[d:g](two)
R_two(x) = robustness of once[0:95-(g+d-1)](two), evaluated at sample 95
R_three(x) = robustness of once[0:95-(2*g+d-1)](three), at sample 95
direction_evidence = R(x) - R(-x)
```

The outer interval excludes incomplete history. Test `d` in {2,4,8}, `g` in {16,32}, both
orientations, and evidence margins in {0,1,2}. Within the margin retain the existing direction
rule. The features are computed with exact min/max robustness operations; six representative
comparisons against real RTAMT matched. The robustness difference is a derived signal on which
the final direction predicate acts, not itself a single atomic raw-voltage predicate.

This is a **completed-window research prototype**. Within-window centering/scaling use the full
window, so the prototype does not establish causal early decisions or match the production
row-by-row implementation. Positive common amplitude rescaling is normalized away, but arbitrary
channel-specific gain changes or polarity inversions are not guaranteed invariances.

## Results

Four-class balanced accuracy; all new scores below are out-of-fold development results.

| Family | Accuracy | Gain, points | Sessions better / worse |
|---|---:|---:|---:|
| Existing monitor | 85.0064% | — | — |
| Distributed burst | 85.2860% | +0.2796 | 7 / 13 |
| Quiet then burst | 85.1705% | +0.1641 | 5 / 8 |
| Sustained midline evidence | 85.0064% | 0 | 0 / 0 |
| Two opposite lobes | **85.4984%** | **+0.4921** | **7 / 3** |
| Three alternating lobes | 84.9282% | −0.0782 | 0 / 2 |
| Selection across all families inside each fold | **85.4984%** | **+0.4921** | **7 / 3** |

The two-lobe candidate selected a one-second maximum lag and evidence margin 1 in all folds;
persistence was 2, 4, and 2 samples. It corrected **38** windows and introduced **19** errors.
These windows overlap and are not independent movements.

| Class | Baseline recall | Nested-rule recall |
|---|---:|---:|
| LEFT | 83.94% | 82.79% |
| RIGHT | 79.96% | 83.07% |
| DOWN | 78.21% | 78.21% |
| STRAIGHT | 97.92% | 97.92% |

A 4,000-replicate paired session-cluster bootstrap gives a gain interval of **−0.36 to +1.38
points**. This resamples fixed out-of-fold predictions; it does not repeat parameter selection,
account fully for the history of searches, or measure uncertainty on a new subject.

The +0.4921 gain is below +0.5, despite both rounding to +0.5 at one decimal. We do not move the
threshold after seeing the result. This is a promising development observation, **not an adopted
improvement or demonstrated cross-subject solution**.

## Correlation with earlier methods

Development-only Spearman correlations relate the new evidence to cached existing measurements.
The two-lobe evidence has weak marginal correlations with the existing signed peak (−0.092),
envelope lag (−0.027), and envelope width (−0.012). It is therefore not simply a monotonic copy
of those measurements, but weak correlation alone does not prove useful independent information.
Distributed-burst strength correlates with envelope width at −0.411, connecting it to the earlier
shape experiments. These correlations include the many rest windows; they should not be read as
class-conditional or cross-subject relationships. Exact outputs are in diagnostics.json.

## Practical conclusion

Ordered sustained lobes deserve further independent validation; merely adding a third lobe did
not help. The current evidence supports neither giving up on richer STL nor claiming it solves
subject variation. The most useful next validation would freeze the two-lobe hypothesis before
collecting independent sessions/subjects, and report LEFT/RIGHT separately as well as balanced
accuracy. Repeatedly searching the existing test subjects would not settle the question.

Reproduce with:

```powershell
python experiments/nested_stl_2026_09_19/run.py
python experiments/nested_stl_2026_09_19/diagnostics.py
```

Outputs: run.log, results.json, diagnostics.json, development_features.npz, oof_predictions.npz.
Verification: ten brute-force bounded-past comparisons, twelve robustness homogeneity checks,
and six real RTAMT comparisons passed. No production code was edited.
