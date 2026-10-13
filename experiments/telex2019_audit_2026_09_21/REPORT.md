# TeLEx-style structure learning on the EEG dataset

**Implemented, executed and verified. No production accuracy improvement was established; your original monitor remains unchanged.**

This is a bounded TeLEx-2019-inspired implementation plus a separately labelled supervised extension. It is not an exact replication of the paper or a test of every possible STL structure.

## What was implemented

The implementation uses predicate tightness, temporal contraction/expansion, positive-trace max-min threshold fitting, and subformula-robustness augmentation for nesting. These mechanisms follow Sections 4–5 of [Jha et al. (2019)](https://susmitjha.github.io/papers/fmsd19.pdf). The paper learns properties from positive examples; four-class decoding is an extension.

Practical limits: seven initial temporal intervals on a 32 Hz grid; shortlisted scalar threshold optimization; beam width five; up to three search levels and fourteen AST nodes. The grammar includes F/G, bounded until, Boolean combinations and negated augmented formulas. Until is explored over shortlisted subformulas rather than all initial predicate pairs. Exact min/max evaluation is piecewise smooth, so this is not a reproduction of the paper’s full numerical optimization or a global optimum claim. A decreasing robustness-band heuristic uses a bounded fallback when its band is empty.

Two variants were kept separate:

- **Positive-only formula learner:** uses the target-class positives for parameter fitting and structure ranking.
- **Supervised structure selection:** also uses the other discovery classes to reject non-discriminating formulas. Every chosen class formula must still satisfy all its discovery positives.

Both variants use labelled development validation and a specified multiclass decoder; neither is an entirely unsupervised classifier. The decoder averages normalized standard STL robustness across two fixed seeds and selects the largest score. If none or multiple class formulas are satisfied, argmax still assigns a class. Score normalization uses only discovery data. Fitting a formula that accepts all positive examples is not the same as classifying all those examples correctly.

## Sampling and data handling

The dataset contains 21,029 windows from 411 recording keys and two subjects. Each discovery subset contains exactly 8 or 16 examples per class: 32 or 64 windows total. Each is from a distinct subject/session/task/run recording; no overlapping windows from one recording are counted as separate discovery trials. Seeds 11 and 29 are fixed and averaged, not cherry-picked.

Training sessions are round-robin sampled. Sessions, including all their overlapping windows, remain together in validation splits. Three outer ten-session blocks and two inner blocks select subset size, search level and classifier layout. The layouts are four-class learned formulas, or the original movement gate followed by learned LEFT/RIGHT/DOWN formulas. All 52 recorded sampling manifests passed equal-count, unique-trial and training-session membership checks.

Raw 256 Hz windows are filtered at .3–6 Hz and reduced to 96 time points at 32 Hz. Inputs include eight normalized channels, all 28 normalized pair differences and eight normalized log-energy traces. Per-window normalization uses the entire completed window. Annotation timing is not used to recenter input, and this is not a prefix-causal monitor.

Small-subset formula discovery is followed by validation on other sessions. This implementation does **not** add counterexamples iteratively to the discovery set; that remains a separately testable extension. No raw test samples or labels select thresholds, structures or seeds. The external populations have been inspected historically, so the evaluations are retrospective rather than blind.

## Actual learned-model results

The primary score below is mean LEFT/RIGHT/DOWN recall, counting missed movements labelled STRAIGHT as errors. STRAIGHT does not contribute to this score. The learned-only rows force a learned challenger rather than hiding its result behind baseline fallback.

| Monitor | Nested development validation | External sub_01 | External sub_11 |
|---|---:|---:|---:|
| Original | 80.70% | 60.39% | 73.08% |
| Learned positive | 30.34% | 27.84% | 35.99% |
| Learned supervised | 45.49% | 41.23% | 42.51% |

The deployment selection procedure chose the original baseline in both modes, including every outer fold. Consequently its accuracy is exactly unchanged. This does not mean the learned formulas matched the baseline: their separate results above are substantially worse.

| Monitor | External sub_01 ordinary / balanced | Correct / total | External sub_11 ordinary / balanced | Correct / total |
|---|---:|---:|---:|---:|
| Original | 85.19% / 69.62% | 1081/1269 | 83.83% / 78.67% | 1006/1200 |
| positive | 74.55% / 45.20% | 946/1269 | 62.08% / 50.86% | 745/1200 |
| supervised | 78.17% / 55.24% | 992/1269 | 64.58% / 55.75% | 775/1200 |

## Class-by-class external recall

| Monitor | sub_01 LEFT | RIGHT | DOWN | sub_11 LEFT | RIGHT | DOWN |
|---|---:|---:|---:|---:|---:|---:|
| Original | 58.97% | 64.48% | 57.72% | 56.15% | 74.66% | 88.42% |
| positive | 23.08% | 36.07% | 24.39% | 24.06% | 36.20% | 47.72% |
| supervised | 47.01% | 32.79% | 43.90% | 39.04% | 53.39% | 35.09% |

## Fit versus validation

| Variant | Discovery movement recall, seed 11 / 29 | Full development fit | Nested validation |
|---|---:|---:|---:|
| positive | 25.00% / 33.33% | 34.03% | 30.34% |
| supervised | 75.00% / 79.17% | 47.66% | 45.49% |

The final learned challengers both selected 8 examples per class, search level 2, and the original movement gate. Seed-specific external outcomes are preserved in challenger_results.json; neither seed was selected using external scores. Paired session-bootstrap intervals are in audit_summary.json and are descriptive, not selection-adjusted.

## Verification and effort

- 52 four-class learner runs, or 208 class-specific structure searches.
- 2,496 numerical threshold refinements and 560,335 candidate evaluations including reranking. These are bounded searches, not that many independent methods.
- 64 formula/real-trace comparisons against genuine RTAMT; full valid-horizon robustness agrees.
- All 42,058 exported challenger predictions reproduced; 8 raw-window/history adapter checks passed.
- Synthetic tests cover strict trace horizons, tightness sign/boundary and RTAMT agreement for nesting, Boolean logic and bounded until.
- Main search/evaluation: 215.1 seconds. Explicit challenger fitting/evaluation: 24.8 seconds. Raw preprocessing and verification are additional. Production fingerprint remained unchanged.

## Files and commands

`FORMULAS.md` contains the actual frozen formulas. Their named signals are defined by data.py. `monitor.py` supplies a raw-window adapter; `LearnedSTLMonitor()` follows the selected baseline fallback, while `LearnedSTLMonitor(challenger=True)` exposes the rejected learned experiment explicitly.

```powershell
python -B experiments\telex2019_audit_2026_09_21\data.py
python -B experiments\telex2019_audit_2026_09_21\run.py
python -B experiments\telex2019_audit_2026_09_21\challenger.py
python -B -m pytest experiments\telex2019_audit_2026_09_21\test_logic.py -q -p no:cacheprovider
python -B experiments\telex2019_audit_2026_09_21\verify.py
python -B experiments\telex2019_audit_2026_09_21\report.py
```

Conclusion: the bounded implementation runs correctly but does not produce a better monitor on these data. It does not establish that every TeLEx search would fail. Increasing nesting, changing objectives, or adding counterexample refinement would be new experiments requiring fresh validation.
