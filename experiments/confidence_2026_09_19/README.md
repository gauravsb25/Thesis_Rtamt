# How certain are the numbers? (19 Sep 2026)

Every accuracy in this project is an average over a handful of recording sessions. This round asks how much
each number would move if the sessions had been different ones from the same person. Sessions are resampled
with replacement 2,000 times inside each split, and the 2.5% / 97.5% percentiles are reported. Windows are
never resampled on their own — they overlap by 0.1 s and are not independent. Code: `run.py`, log `results.log`.

## Results (the shipped monitor, with the stream background)

| measure | sub_01 1–30 (30 sessions) | sub_01 31–36 (6 sessions) | sub_11 (10 sessions) |
|---|---|---|---|
| **four-class balanced** | **85.01%** [81.9, 89.2] | **69.62%** [66.5, 80.2] | **78.67%** [66.6, 83.2] |
| LEFT | 83.94% [76.4, 90.7] | 58.97% [43.8, 100.0] | 56.15% [29.6, 70.1] |
| RIGHT | 79.96% [71.6, 87.6] | 64.48% [31.0, 85.6] | 74.66% [50.3, 86.6] |
| DOWN | 78.21% [69.7, 90.0] | 57.72% [51.5, 62.5] | 88.42% [76.3, 100.0] |
| STRAIGHT | 97.92% [97.1, 98.6] | 97.28% [91.6, 99.5] | 95.46% [95.2, 99.0] |
| **movement vs still** | **95.66%** [94.5, 97.0] | **91.78%** [85.7, 95.1] | **94.20%** [92.2, 99.3] |
| nod vs turn | 91.66% [87.6, 94.9] | 70.68% [58.3, 80.1] | 98.60% [96.7, 100.0] |
| left vs right | 92.59% [88.2, 95.7] | 100.00% [100.0, 100.0] | 70.50% [48.2, 81.0] |

## What this changes

1. **The detection claim is solid.** Movement versus still is 91.8–95.7% with intervals of a few points on
   every split. This is the number the thesis should lead with, and it survives the uncertainty.
2. **The per-class held-out numbers are very uncertain.** LEFT on sub_01 31–36 is 59% with an interval from
   44% to 100%: those six sessions contain only a handful of left turns. Any statement of the form "LEFT is
   worse than RIGHT here" is not supported.
3. **sub_11's direction is 70.5% [48.2, 81.0].** Its lower end touches chance, so even the claim "the second
   person is better than a coin toss at direction" is weak — which is consistent with everything else this
   week and strengthens the argument for a second sensor rather than better rules.
4. **Every improvement we rejected was inside the noise anyway.** The design-session intervals are about
   ±3.5 points, so the +0.2 to +1.4 point gains that several ideas showed on sessions 1–30 could never have
   been distinguished from chance there. The session-wise consistency condition in the adoption rule (more
   sessions better than worse) turns out to have been the part doing the real work.
5. **The two adopted changes are still meaningful**, because they were measured as paired differences on the
   same windows (false alarms 1534 → 331 on the design sessions), not as two independent averages.

## How to report a number in the thesis

Quote the interval, the number of sessions and the number of windows behind it, for example:
"69.6% (95% CI 66.5–80.2, 6 sessions, 1,269 windows)". Where a class has fewer than about 10 movements in a
split, report the counts (69/117) instead of a percentage alone.
