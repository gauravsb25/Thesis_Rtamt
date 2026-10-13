# The old classify.py's rules and onset timing, tested against the current monitor (23 Sep 2026)

`_archive/pipeline/classify.py` carries two things the current monitor does not use: a set of
threshold rules, and **onset timing** (when each electrode first "goes high"). Both were tested here.

Definitions copied verbatim from `_archive/lib/features.py` and `classify.py`, so a feature cannot
mean something different here than it did there. Code: `features.py`, `cv.py`.

## The rules: most cannot transfer, by construction

Of the 17 rules in `SELECTED_RULES`, most use **absolute microvolt thresholds** —
`T3T4_peak_signed >= 63.677`, `Cz_ptp <= 40.058`, `Fp1_bp_delta <= 8.452`, `Fp1_rms >= 10.824`.
That file's own comment records that sub_11's raw amplitude "runs ~2-2.6x hotter than sub_01's",
which is exactly why this project uses only unit-free ratios. Those rules cannot move between
subjects and were not tested further.

Five of its features **are** unit-free and were tested: `T3T4_skew`, `T3_normpeak`, `Cz/T3_ratio`,
`T4_spec_centroid` (a frequency in Hz) and `temporal~central_gonsetlag`.

> Also worth knowing before comparing numbers: classify.py's 82% is under **oracle windowing** — its
> own docstring says the window was selected using the trial's true class label. It is not
> comparable to the monitor's 85.01 / 69.62 / 78.67 on unlabelled windows.

## The onset claim is TRUE, and it is the rare cross-subject-consistent effect

classify.py states: *"Cz fires ~110ms AFTER the temporal pair on DOWN but BEFORE it on LEFT/RIGHT
(consistent on both subjects)."* Measured on the current dataset, over the windows the monitor calls
a movement (`lag = temporal onset - Cz onset`; negative means the temples fired first):

| split | DOWN | LEFT/RIGHT | difference |
|---|---|---|---|
| sub_01 1-30 | −0.083 s | +0.062 s | **−0.146 s** |
| sub_01 31-36 | −0.062 s | +0.018 s | **−0.080 s** |
| sub_11 | −0.049 s | +0.131 s | **−0.180 s** |

**The claim holds on all three splits, and the effect is largest on sub_11.** Four onset features are
sign-consistent across all three: `lag_temporal~central`, `lag_temporal~midline`,
`lag_frontal~temporal`, `lag_frontopolar~temporal`. Consistency like this is rare in this project —
almost every other feature flips or vanishes between subjects.

The single-pair lateralisation lag `lag_T3_T4` is **not** consistent (−0.004 / +0.008 / −0.039),
matching classify.py's own admission that single-pair onset lags are weak.

## But it adds nothing, because the nod signal already does the job far better

How well each separates DOWN from turns on its own, at its best threshold:

| | sub_01 1-30 | sub_11 |
|---|---|---|
| `lag_temporal~central` | 77.3% | 65.1% |
| **the monitor's `nod` signal** | **91.6%** | **98.4%** |

The two are only weakly related (correlation −0.321 on movement windows), so the timing really is
*independent* information — it is simply much weaker. Offered to the monitor as an extra DOWN
condition it changes nothing at all:

| design | CV gain | sessions |
|---|---|---|
| `lag_temporal~central` alone, role nod+ | **+0.00** | 0 / 0 |
| `lag_temporal~central` alone, role nod− | **+0.00** | 0 / 0 |

No threshold on it improved even the training folds, so the search never used it.

## All 18 features, all five roles

2,700 candidates per fold, blocked 3-fold CV on sub_01 sessions 1-30 (baseline 85.01%):

| role | CV gain | sessions +/− | verdict |
|---|---|---|---|
| gate+ | −0.37 | 0 / 2 | fail |
| gate− | +0.26 | 9 / 11 | fail (both halves) |
| nod+ | −1.04 | 0 / 3 | fail |
| nod− | +0.00 | 2 / 1 | fail |
| dir | −1.61 | 3 / 8 | fail |
| **all roles together** | **−0.41** | **1 / 10** | **fail** |

Each fold picked a different winner (`Cz/T3_ratio`, then `T4_spec_centroid`, then `Cz/T3_ratio`) —
the usual signature of a search finding noise. Nothing passed, so no test run was made and the
monitor is unchanged.

## Conclusion

The onset timing is **not** a dead end intellectually: it is one of very few measurements in this
project that behaves the same way in both subjects, and the old file's claim about it is accurate.
It fails only because the monitor's `nod` signal answers the same question much better (98.4% vs
65.1% on sub_11).

That is worth a sentence in the thesis: *the nod-versus-turn distinction is visible in two
independent ways — an energy topography and a timing order — and both transfer across subjects,
which is why DOWN is the movement class that generalises best.*

The rules are a different story: they are per-subject amplitude thresholds selected under oracle
windowing, and they have nothing to offer this monitor.
