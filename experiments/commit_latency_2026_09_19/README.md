# The early-commit table, re-measured with the current gate (19 Sep 2026)

The published early-commit numbers were measured on 17 September, **before** the stream-background gate was
adopted, so they no longer described the shipped monitor. This round recomputes them.

**Method.** Every window of `extracted_npz_dataset_v2` is pushed through `RowStreamMonitor` with the same
stream history the window monitor receives, taking a verdict every 32 rows (0.125 s). A commitment of k
seconds means the same movement class held for the last ⌈k·256/32⌉ verdicts; STRAIGHT is never committed
early; a window that never commits keeps its verdict at row 768, which equals `HeadMovementMonitor`.
Code `run.py`, raw verdicts `verdicts.npz`, log `results.log` (9m51s over 21,029 windows).

## Result

Four-class balanced accuracy, and the median time from the movement's start inside the window to the
decision (over correctly classified movement windows):

| decision | sub_01 1–30 | sub_01 31–36 | sub_11 |
|---|---|---|---|
| wait for row 768 (default) | **85.01%** · 2.22 s | 69.62% · 2.20 s | 78.67% · 2.31 s |
| commit after 1.5 s | 84.95% · 2.03 s | 69.60% · 2.06 s | 78.67% · 2.15 s |
| **commit after 1.0 s** | 83.92% · 1.79 s | **69.93%** · 1.85 s | **79.57%** · 1.76 s |
| commit after 0.75 s | 80.89% · 1.56 s | **69.98%** · 1.67 s | 78.97% · 1.50 s |
| commit after 0.5 s | 74.85% · 1.24 s | 67.53% · 1.31 s | 77.53% · 1.23 s |

## What changed against the old table

| | old (before the background gate) | new (current monitor) |
|---|---|---|
| wait for row 768 | 82.74 / 67.22 / 77.00 | 85.01 / 69.62 / 78.67 |
| commit after 1.0 s | 81.97 / 67.75 / 77.83 | 83.92 / **69.93** / **79.57** |
| cost on the design sessions | −0.77 | −1.09 |
| effect on the held-out sets | +0.53 / +0.83 | **+0.31 / +0.90** |

The conclusion is the same but now stronger: **committing after 1 second of a stable class decides about
0.45 s sooner and is slightly better on both held-out sets.** Waiting for the full window is still best on
the sessions the design saw. A 0.75 s commitment is the fastest setting that does not lose on either
held-out set (69.98 and 78.97, deciding 0.55–0.8 s sooner).

Note the bootstrap intervals measured the same day: differences of ±1 point are inside the noise, so the
honest statement is "no worse, and about half a second earlier", not "better".

## Files updated

`monitoring/row_stream.py` (docstring) and `monitoring/README.md` now carry this table instead of the
17 September one.
