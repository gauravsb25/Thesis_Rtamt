# Why each window is misclassified, and what the hints are worth (17 Sep 2026)

Dataset: `extracted_npz_dataset_v2`. Monitor: the default RTAMT monitors (unchanged).
New rules were discovered and chosen on **sub_01 sessions 1–30 only**. Sessions 31–36 and sub_11
were used only once, to test the one rule that passed.

## 1. Every wrong window traced to its test and electrodes — `wrong_windows.csv`, `wrong_movements.csv`

`components.py` rebuilds each monitor measure channel by channel and pair by pair. For all 21,029
windows it reproduces the monitor's six values exactly (relative difference 0). `attribution.py`
then records, for each of the **2,554 wrong windows** (91 wrong movements):
- which STL test sent it to the wrong class, and by how much;
- which electrodes drove that test;
- which other electrodes or signals pointed to the true class.

| sessions 1–30 (18,560 windows, 2,057 wrong) | windows | cause | hints seen |
|---|---|---|---|
| rest → movement (false alarm) | 1,534 | all three gate tests passed | 41% of the false DOWNs are bursts < 0.25 s; 6% blink-like (Fp1/Fp2) |
| movement → STRAIGHT | 187 | contrast too low (72% alone) | most show the burst on 6–8 single channels |
| turn called a nod | 149 | nod ≥ −0.661 and F4−T3 veto did not fire | 40% T3/T4 move oppositely; some show big Fp1−Fp2 / Fp2−T3 swings |
| LEFT ↔ RIGHT | 122 | F4−T3's largest excursion had the wrong sign | other pairs mostly agree with the wrong read |
| nod called a turn | 65 | nod < −0.661, mostly its F3/F4 symmetry part | 31% no sideways swing on other pairs; 18% Fp1/Fp2 common mode |

Test sessions (`attribution_summary.txt`):
- **sub_01 31–36:** 269 wrong windows. Largest groups: turns called nods (94) and false DOWNs (94).
- **sub_11:** 228 wrong windows. Largest group: LEFT/RIGHT swaps (113), mostly its inverted sessions.

## 2. Each hint turned into a rule and scored — `hint_rules.py`, `hint_rules*.log`

Sessions 1–30, thresholds chosen by 3-fold blocked cross-validation (sessions 1–10 / 11–20 /
21–30). Adoption rule, fixed before testing: positive cross-validated gain with a stable threshold,
more movements fixed than broken, sessions better ≥ worse, and no T4 dependency.

| rule family | result | verdict |
|---|---|---|
| A false-alarm rescue (short burst, 1–2 channels, one channel dominant, blink-like, edge) | CV +0.00; e.g. short bursts fix 296 false alarms but break 412 movement windows | rejected |
| B missed-movement rescue (strong single channel / k channels) | CV +0.8–0.95, fixes 5–7 movements, but +290–470 false alarms; sessions 8–12 better vs 18–22 worse | rejected |
| C turn-called-nod rescue, single pairs | Fp1−Fp2 +0.46, Fp2−T3 +0.38; T3/T4, F3/F4 and others ≤ 0 | combined → C5 |
| **C5 extra nod vetoes: a nod must not swing on Fp1−Fp2 (> 7.1) or Fp2−T3 (> 6.6)** | nested CV +0.84, 4 movements fixed / 0 broken, sessions 6/6 | **passed → tested** |
| D nod-called-turn rescue | all ≤ 0 | rejected |
| E direction rescue (other pairs disagree, tied lobes) | all < 0 | rejected |

## 3. The one test — `verify_vetoes_*.log`

C5 was implemented as STL. DOWN gains `historically(sideways_fp1fp2 <= 7.1)` and
`historically(sideways_fp2t3 <= 6.6)`; the turns gain the matching `once(... > ...)` terms. The
module reproduced the selected rule on all sessions 1–30 windows (0 differ), and RTAMT agreed on
40/40 sampled windows per split.

| 4-class balanced | without | with extra vetoes |
|---|---|---|
| sessions 1–30 (design) | 82.74% | 83.86% (LEFT 83.6→86.9, RIGHT 79.4→82.0, DOWN 77.5→76.2) |
| sub_01 31–36 (test) | 67.22% | **68.04%** (RIGHT 64.5→69.4, DOWN 57.7→56.1) |
| sub_11 (test) | 77.00% | **74.72%** (DOWN 88.4→79.3: 26 nod windows broken, 0 fixed) |

**Not adopted.** It helps sub_01 but breaks sub_11's nods, which carry large Fp1−Fp2 / Fp2−T3
swings (see the v2 content check: 12 of 30 sub_11 nods looked turn-shaped). The vetoes stay
available, off by default:
`RtamtConfig(down_veto_fp1fp2=7.1, down_veto_fp2t3=6.6)`.
