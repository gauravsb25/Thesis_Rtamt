# Round 2 on top of the stream-background gate (17 Sep 2026)

Starting point: the shipped monitor with stream history (85.01% on sub_01 sessions 1–30).

Errors that remain:

| error | sessions 1–30 | sub_01 31–36 | sub_11 |
|---|---|---|---|
| turn called DOWN | 143 | 94 | 4 |
| DOWN called STRAIGHT | 98 | 39 | 29 |
| LEFT/RIGHT swapped | 126 | 0 | 113 |

**Protocol:** blocked 3-fold CV on sessions 1–30. Adoption rule, fixed before testing: CV gain ≥ +0.5 points
and more sessions better than worse. `features.py` uses the package code (`compute_signals` with history from
`monitoring.stream_history`) and reproduces 85.01%.

## Results — `round2_dev.log`

| design | CV | gain | sessions + / − |
|---|---|---|---|
| **idea 1: DOWN vs turn from channels normalised by their own stream background** | | | |
| nod computed on background-normalised channels | 84.42% | −0.58 | 0 / 3 |
| sideways swing of the background-normalised F4−T3 pair | 85.35% | +0.34 | 2 / 2 |
| both | 83.42% | −1.59 | 2 / 9 |
| temporal ÷ midline background ratios (replace / AND) | 85.01% / 84.61% | 0 / −0.40 | — |
| background-normalised nod OR / AND current nod | 84.93% / 84.61% | −0.08 / −0.40 | — |
| **idea 2: re-select thresholds under the new gate** | | | |
| delta d, deflect q, background b jointly | 84.69% | −0.32 | 13 / 13 |
| nod n, sideways v jointly | 83.45% | −1.56 | 2 / 6 |
| both | 83.13% | −1.88 | 11 / 16 |

**No design passed**, so the test sets were not used and the monitor thresholds are unchanged.

- **Idea 1:** normalising each channel by its own stream level does not make the nod/turn pattern more separable than
  the window-internal measures already are.
- **Idea 2:** the current thresholds are still the best choice under the new gate.

## Idea 3 (engineering): `RowStreamMonitor` uses the stream background — done

- It keeps the last 5 s of rows before each window. At the window start it computes each channel's background from
  them, then updates the 0.5 s energy row by row.
- At row 768 it equals `HeadMovementMonitor().classify(window, history)` exactly (tests in `tests/test_row_stream.py`).
- `RowStreamMonitor(history=...)` seeds a history; `clear()` forgets it.
