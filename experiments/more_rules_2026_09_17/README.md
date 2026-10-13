# More signals and more rules inside the monitors (17 Sep 2026)

User question: why only a few signals and rules per monitor, and why not many more, especially for
STRAIGHT? Tested with `more_rules.py`. Each rule is one signal against one threshold (an STL atom). A class
holds when at least k of its n rules hold, which is expressible in STL as a disjunction of conjunctions.
Signals are the per-channel and per-pair versions of the monitor's own measures (`components_v2.npz`,
identical to the monitor).

## Designs

| design | rules | thresholds |
|---|---|---|
| `gate_kofn` | movement if ≥ k of 27 rules fire: contrast, delta, deflect for the window and for each of the 8 channels | each rule fires above the p-quantile of rest windows |
| `gate_extra` | current gate AND ≥ k of the 24 per-channel rules (current STRAIGHT rules + many extra STRAIGHT rules) | same |
| `down_kofn` | DOWN if ≥ k of 13 nod rules: nod, balance, symmetry, Fp symmetry, no sideways swing on 9 T4-free pairs | each rule's best DOWN-vs-turn split |

Selection: blocked 3-fold CV on sub_01 sessions 1–30. Adoption rule fixed before testing: CV gain ≥ +0.5
points, more sessions better than worse, then the test sets once.

## Results (four-class balanced)

| design | sessions 1–30 CV | sessions better / worse | sub_01 31–36 | sub_11 |
|---|---|---|---|---|
| current monitor | 82.74% | — | **67.22%** | **77.00%** |
| `gate_kofn` (p 0.97, k 3) | **84.93% (+2.19)** | 20 / 9 | 64.47% (−2.75) | 68.29% (−8.71; STRAIGHT 88.8 → 40.0) |
| `gate_extra` (p 0.95, k 4) | 83.46% (+0.72) | 24 / 6 | 64.03% (−3.19; DOWN 57.7 → 43.1) | 77.29% (+0.29) |
| `down_kofn` | 82.74% (+0.00, CV kept the current rule) | 0 / 0 | not tested | not tested |

**Neither passing design survived the test sets. The monitors are unchanged.**

Why (diagnostic on the rest windows): every per-channel threshold encodes sub_01's resting level on that
channel. The rules were set to fire in 3% of sessions 1–30 rest windows, but they fire in up to 11% of rest
windows on sub_01 sessions 31–36 and 24–47% on sub_11. The slow-wave (delta) rules on Fp2, F4, Cz and Fp1
are worst, because sub_11's resting EEG has more slow-wave power. More rules mean more such numbers, so the
monitor fits the design sessions better and new sessions and subjects worse. The three window-level rules
average over channels, which is why they transfer better.
