# Do DOWN windows contain a nod, and do STRAIGHT windows contain movement? (17 Sep 2026)

`check.py`, log `check.log`. The check uses the 3-second npz windows only. Each run was rebuilt by joining its
overlapping windows (every window is a verbatim slice of its run), so one bump is counted once, not once per window.

**Movement size:** RMS over the 8 channels of the 0.5–3 Hz signal in a 0.5 s block (µV). It is compared
with the session's rest background and with the session's typical real nod.

**How the dataset was built:** by the "contained" rule. A DOWN window holds the whole annotated nod with at least 0.25 s
to spare at both edges; a STRAIGHT window has no annotated movement within 1 s. Labels therefore follow the annotations exactly.

## A. DOWN windows called STRAIGHT

| | sub_01 1–30 | sub_01 31–36 | sub_11 |
|---|---|---|---|
| annotated nods | 68 | 10 | 30 |
| nod size / background, median | 4.1× | 2.8× | 4.9× |
| blank-looking (< 1.5×) | 2 | 0 | 3 (none missed) |
| weak (1.5–2.5×) | 6 | 1 | 4 |
| missed windows (called STRAIGHT) | 103 | 39 | 29 |
| … from blank or weak nods | 69 | 1 | 21 |
| … from clear nods | 34 | 38 | 8 |
| gate test that failed (windows) | contrast 92, delta 27, deflect 11 | contrast 32, delta 18, deflect 2 | contrast 28, delta 1, deflect 2 |

- **The DOWN windows are not blank.** Almost every nod is visible; only 5 of 108 look nearly empty.
- **Sessions 1–30 and sub_11:** most missed windows come from a few weak nods, barely bigger than the background.
- **sub_01 31–36:** the misses come from four clear nods (2.5–3×). Their windows also contain other activity of similar size,
  so the "loudest ½ s vs quietest ½ s" contrast test fails.
- Figure: `missed_down_examples.png`, with the annotated nod shaded.

## B. STRAIGHT windows called DOWN

| | sub_01 1–30 | sub_01 31–36 | sub_11 |
|---|---|---|---|
| false-DOWN windows | 995 | 94 | 51 |
| separate bumps behind them | 175 (median 3 windows each) | 17 | 12 |
| bump < half a typical nod | 103 bumps (530 windows) | 5 (31) | 12 (51) |
| bump 0.5–1× a typical nod | 68 (441) | 10 (28) | 0 |
| bump ≥ a typical nod | 4 (24) | 2 (35) | 0 |
| bumps > 3 s from any annotated movement or in the straight task | 158 of 175 | 14 of 17 | 8 of 12 |

- **Most false DOWNs are small bumps**: the residual blink-shaped whole-head bumps described in `experiments/artifact_check_2026_09_17`.
  They are not a hidden head movement.
- **One bump can make many windows wrong**, because windows overlap in 0.1 s steps.
- **Only 6 bumps are as big as a typical nod** and may be small unannotated head movements or strong blinks:
  - sub_01 ses-12 left run 5 at 0.4 s
  - ses-13 left run 6 at 18.3 s
  - ses-14 straight run 5 at 0.5 s
  - ses-29 straight run 5 at 13.8 s
  - ses-32 left run 1 at 13.8 s
  - ses-36 straight run 3 at 8.7 s
- They cover 59 windows, so at most about 6% of the false-DOWN windows could be label problems.
- Figures: `false_down_largest.png`, `false_down_typical.png`.
