# Offline RTAMT monitor (25 Sep 2026)

**Result: an offline monitor that judges each window against the rest of its own recording scores
86.73 / 71.80 / 79.37, against the live monitor's 85.01 / 69.62 / 78.67.** It improves all three splits and
cuts sub_11 false alarms from 23 to 0. It is a separate monitor, `monitoring/offline_monitor.py`. The live
monitor is unchanged.

**Verified through RTAMT.** `python project_code.py evaluate --offline` ran the four offline specifications on all
21,029 windows (20 min, 6 processes). The verdicts equal the design arithmetic on 21,029 / 21,029 windows, and the
table below comes from those RTAMT verdicts. 68 new tests pass (`tests/test_offline_monitor.py`).

| class | sub_01 1–30 (design) | sub_01 31–36 (test) | sub_11 (test) |
|---|---|---|---|
| LEFT | 830 / 959 (86.55%) | 69 / 117 (58.97%) | 108 / 187 (57.75%) |
| RIGHT | 787 / 963 (81.72%) | 120 / 183 (65.57%) | 163 / 221 (73.76%) |
| DOWN | 606 / 748 (81.02%) | 81 / 123 (65.85%) | 245 / 285 (85.96%) |
| STRAIGHT | 15,513 / 15,890 (97.63%) | 819 / 846 (96.81%) | 507 / 507 (100.00%) |
| **balanced** | **86.73%** | **71.80%** | **79.37%** |

## Why an offline monitor helps

The live monitor may only use the window and at most 5 s before it. The recordings are short runs, and the
movement comes 2–3 s after a run starts. So 74.5% of movement windows, but only 15% of rest windows, have no
usable history. The live monitor therefore judges most movements with its weaker fallback, the window's own
loudest-over-quietest contrast (`experiments/improvement_round_2026_09_24`, section 2.4).

Offline, the whole recording is known. The offline monitor compares every window with the rest of its own
recording, before and after it. Every window, movement or rest, early or late in its run, gets the same
stream-background test.

## The monitor

`background(t)` = mean over the eight channels of (0.5–3 Hz energy of the 0.5 s ending at t) ÷ (that channel's
median energy over the rest of the recording). Blocks ending in the 0.5 s before the window or inside it are
left out of the median. The filter is causal and starts in steady state.

```
MOVE     = eventually[0:767](background >= 4.95) and eventually[0:767](delta >= 0.208)
           and eventually[0:767](deflect >= 2.72)
STRAIGHT = always[0:767](background < 4.95) or always[0:767](delta < 0.208) or always[0:767](deflect < 2.72)
DOWN     = MOVE and (nod >= -0.661) and always[0:767](sideways <= 7.54)
LEFT     = MOVE and ((nod < -0.661) or eventually[0:767](sideways > 7.54)) and (turn > 0)
RIGHT    = MOVE and ((nod < -0.661) or eventually[0:767](sideways > 7.54)) and (turn < 0)
```

These are RTAMT offline specifications with future-time operators, read at t = 0. `delta`, `deflect`,
`sideways`, `nod` and `turn` are the live monitor's signals of the same window. Only the loudness test is new.

## How it was chosen

Design on sub_01 sessions 1–30 only; the held-out sets were scored once, after the design was frozen.

| step | result |
|---|---|
| four offline background variants (whole run or run minus window; causal or zero-phase filter), CV on 1–30 | run minus window, causal: 86.59 CV (18 sessions better / 9 worse); the others 85.48–86.25 |
| threshold b, blocked 3-fold CV | 4.83 / 5.21 / 4.95 |
| b on all of 1–30 (in-sample) | 4.95, flat from 4.75 (86.66) to 5.25 (86.66) |
| run-referenced `sideways` or `deflect`, label-free session nod references | ≤ +0.4 or worse: not added |

## Held-out result (scored once)

| split | live monitor | offline monitor | sessions better / worse |
|---|---|---|---|
| sub_01 1–30 (design) | 85.01 (L 83.9 R 80.0 D 78.2 S 97.9) | 86.73 (L 86.5 R 81.7 D 81.0 S 97.6) | 19 / 8 |
| sub_01 31–36 | 69.62 (L 59.0 R 64.5 D 57.7 S 97.3) | 71.80 (L 59.0 R 65.6 D 65.9 S 96.8) | 4 / 1 |
| sub_11 | 78.67 (L 56.1 R 74.7 D 88.4 S 95.5) | 79.37 (L 57.8 R 73.8 D 86.0 S 100.0) | 4 / 2 |

The gain is detection: movements the contrast fallback missed are now caught (design LEFT → STRAIGHT 42 → 17
windows, DOWN → STRAIGHT 98 → 70). The nod and direction errors are unchanged; they are not loudness problems.

## Limits

- **Offline only.** It uses samples after the window. A live system does not have them. Use it to analyse
  recordings, not for live control.
- **One recording per run.** The reference comes from the window's own run file (median length 8.6 s). A longer
  continuous recording would give a steadier reference.
- The held-out sets have been used in many earlier rounds and hold only 37 and 95 independent movements.

## Run it

```
python project_code.py evaluate --offline      # every window through RTAMT, cached (~15 min the first time)
python -m pytest -q tests/test_offline_monitor.py
```
