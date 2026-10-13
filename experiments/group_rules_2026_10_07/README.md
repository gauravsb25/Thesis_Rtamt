# Electrode-group rules, rule combinations and context switching (2026-10-07)

**Result: nothing adopted.** Every combination that improved sub_01 cross-validation lost on sub_11, most of
them badly. The production monitor is unchanged.

## What was tested

Everything is window-only (no stream history) and computed fresh from the raw windows of
`extracted_npz_dataset_v2`. No earlier result was reused; the baseline thresholds were re-selected in every fold.

- **Groups.** Every non-empty subset of Fp1 Fp2 F3 Fz F4 T3 Cz T4 (255), and every pair of disjoint groups
  (3,025). This includes left Fp1+F3+T3 vs right Fp2+F4+T4, front Fp1+Fp2+F3+Fz+F4 vs T3+Cz+T4, midline Fz+Cz vs
  temples T3+T4, T3 vs T4, F3 vs F4, Fp1 vs Fp2 and the current F4 vs T3.
- **Measurements per pair**, in 0.3–6 Hz and 0.5–3 Hz: signed peak of the group difference (µV and per-channel
  normalised), its prominence, energy log-ratio (µV and normalised), common-mode/differential ratio,
  correlation, peak-time lag. 48,400 rules for the nod and direction decisions.
- **Per group** for detection: contrast, deflection, slow-wave rise. 765 rules.
- **Combinations:** single best rule, AND / OR of 2–3 rules, k-of-n votes (3–25 rules), baseline AND one rule
  (veto), baseline OR one rule (rescue), **regime switching** (one window property chooses which rule is on:
  `(c and rule1) or (not c and rule2)`), **confidence cascade** (a rule is switched off for a window that sits
  near its threshold; the next rule decides).
- **Protocol:** selection on sub_01 sessions 1–20 with blocked 3-fold CV (1–7 / 8–14 / 15–20); null control =
  the same search with labels shuffled inside each session; held-out sub_01 21–36 and sub_11 scored once.

## Per decision (cross-validated, sub_01 1–20; class-balanced score)

| decision | baseline | best family over all rules | best of your groupings | shuffled-label null |
|---|---|---|---|---|
| detect | 90.88 | vote of 15: 92.65 | all 8, vote of 3: 93.67 | 75.1 (real best 93.57) |
| nod | 86.91 | baseline OR one rule: 90.13 | T3 vs T4, vote of 3: 92.16 | 68.0 (real 93.96) |
| direction | 89.11 | baseline AND one rule: 90.60 | none beats baseline (F4 vs T3 vote 87.26) | 59.6 (real 92.23) |

Regime switching scored 91.24 / 81.98 / 81.97 and the cascade 91.48 / 84.67 / 83.04 (detect / nod / direction).
The single best of all 48,404 rules lost to the baseline on nod and direction: picking from that many rules
selects luck. Full table: `cv_log.txt`, `cv_report.json`.

## End to end (four-class balanced accuracy)

| combination | CV 1–20 | 21–36 | 31–36 | sub_11 | sub_11 false alarms |
|---|---|---|---|---|---|
| **baseline** (window-only, re-selected) | 78.01 | 78.85 | 66.34 | 75.55 | 73/507 |
| detect: vote of 15 | 79.75 | 78.57 | 65.01 | 74.05 | 120 |
| detect: all 8, vote of 3 | 80.20 | 78.14 | 63.37 | 72.13 | 126 |
| nod: baseline OR one rule | 80.32 | 80.22 | 66.61 | 65.44 | 73 |
| nod: vote of 3 | 80.90 | 80.78 | 69.20 | 49.66 | 73 |
| nod: T3 vs T4, vote of 3 | 80.81 | 76.63 | 61.41 | 59.61 | 73 |
| nod: T3 vs T4, single | 80.20 | 79.16 | 67.00 | 52.22 | 73 |
| direction: baseline AND one rule | 79.25 | 76.27 | 66.34 | 66.68 | 73 |
| detect all 8 + nod T3–T4 | 83.00 | 76.21 | 58.81 | 56.41 | 126 |
| detect all 8 + nod T3–T4 + direction AND | 84.01 | 74.15 | 59.63 | 49.11 | 126 |
| detect vote 15 + nod OR + direction AND | 83.63 | 77.42 | 65.28 | 57.96 | 120 |

## Why it fails on sub_11 (`perclass.py`)

| sub_11 | turns called DOWN | correct LEFT | correct STRAIGHT |
|---|---|---|---|
| baseline | 4 | 106/187 | 434/507 |
| nod: vote of 3 | 258 | 64 | 434 |
| nod: T3 vs T4 vote | 140 | 87 | 434 |
| nod: baseline OR one rule | 160 | 79 | 434 |
| direction: baseline AND one rule | 4 | 32 | 434 |
| detect: all 8 vote | 4 | 105 | 381 |

Rules learned on sub_01 read sub_11's turns as nods: in sub_11 one temple swings by hundreds of µV during a turn,
so group comparisons that separate sub_01's turns from nods put sub_11's turns on the nod side. The 21–36 gains
of the nod rules (up to +1.9, +2.9 on 31–36) do not survive a second person.

## Reproduce

```
python features.py          # ~5 min, writes cache/ (733 MB of pair features, gitignored)
python group_select.py      # per-decision CV, null control, your groupings -> cv_log.txt, cv_report.json
python e2e.py "NOD=vote_3" "DETECT=vote_3@all8,NOD=vote_3@T3T4"   # end-to-end CV + held-out once
python perclass.py          # per-class held-out counts
```
