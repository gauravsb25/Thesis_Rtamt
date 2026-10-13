# Rest-state adaptation, Riemannian geometry, and the STL+ML cascade (19 Sep 2026)

Seven methods were proposed. Three could be tested with the data in hand and were; four cannot be tested
without new recordings or more subjects. Baseline: the shipped monitor **85.01 / 69.62 / 78.67**
(direction among turn windows 92.6 / 100.0 / 70.5).

| proposed method | status |
|---|---|
| 1. resting-state adaptation (unsupervised) | tested — fails |
| 2. Riemannian covariance / tangent space | tested — far below the monitors |
| 5. cascade: STL trigger + ML direction | tested — ties the rule, no gain |
| 3. multi-source domain adaptation | impossible: one source subject (sub_01); "multi-source" needs several |
| 4. PEFT / LoRA with 1-turn calibration | 1-turn calibration already measured as harmful (sub_11 78.65 → 70.06); a deep net has 442 movements to learn from |
| 6. EOG / IMU / eye tracking | no such channel was recorded |
| 7. third subject | no data |

## Method 1: adapt to the session's own rest state — `features.py`, `dev.log`

The rest reference is built **without labels**: the quietest 20% of a session's windows are pooled, which is
what "sit still for 20 seconds" would give. Each monitor measurement is then scaled by that rest level
instead of by the window's own level.

| rule | CV on sessions 1–30 | gain | sessions +/− |
|---|---|---|---|
| rest-referenced energy as the gate | 84.00% | −1.00 | 4 / 16 |
| current gate AND rest-referenced energy | 84.38% | −0.63 | 11 / 2 |
| deflection against the rest spread | 84.69% | −0.31 | 7 / 8 |
| nod against the rest spread | 85.01% | +0.00 | — |
| sideways against the rest spread | 85.25% | +0.25 | 2 / 2 |
| direction from the rest-referenced pair | 85.01% | +0.00 | — |

Nothing passes. The reason is that the monitor already compares each window with the stream just before it
(the background gate adopted on 17 Sep), which is the same idea over 4 seconds instead of a session. Going
from a 4-second reference to a session-long one adds nothing and loses responsiveness.

## Method 2: Riemannian covariance and tangent space

Each window's 8×8 covariance is whitened by the session's rest covariance and mapped to the tangent space
(36 features). These features are scale-free and polarity-free by construction, which is exactly what the
direction problem seemed to need.

| features, 5-fold grouped CV inside sessions 1–30 | 4-class | direction |
|---|---|---|
| the monitors | 85.01% | 92.6% |
| 87 hand-made signals (19 Sep study) | 87.58% | 90.2% |
| **tangent space, logistic regression** | **38.33%** | 61.6% |
| **tangent space, random forest** | **34.71%** | 48.2% |
| tangent space + hand-made signals, logistic | 75.54% | 80.9% |
| tangent space + hand-made signals, random forest | 83.29% | 90.5% |

The tangent-space features are not merely weaker — adding them makes the hand-made signals worse (83.29
against 87.58), i.e. they are noise for this problem. Three reasons:

1. **A head movement is a transient, not a stationary pattern.** Covariance over 3 seconds is dominated by
   the 1.6 s of ordinary EEG around the 1.4 s movement. Riemannian methods earn their reputation on
   sustained oscillatory tasks (motor imagery), which this is not.
2. **Whitening by rest removes the amplitude information the monitors live on.** The entire gate is "this
   window is much louder than the background"; after whitening, that ratio is gone.
3. **Covariance discards the sign — including the sign that works.** Polarity invariance was the attraction,
   but for sub_01 the polarity is exactly what carries direction (92.6% and 100%).

## Method 5: the cascade (STL decides, a model re-decides LEFT vs RIGHT)

Stage 2 trained only on the 1,701 turn windows of sessions 1–30, 5-fold grouped CV:

| features | model | direction | 4-class |
|---|---|---|---|
| monitor signals | logistic regression | 92.4% | 84.93% |
| monitor signals | random forest | 91.6% | 84.56% |
| monitor + rest-normalised | logistic regression | 92.7% | 85.06% |
| tangent space | logistic regression | 61.9% | 71.45% |
| tangent + monitor signals | random forest | 92.7% | 85.06% |
| **the monitor's own rule (`turn > 0`)** | — | **92.6%** | **85.01%** |

The best model matches the one-line rule to within 0.1 point on the design sessions. And on the held-out
sets, measured in the ceiling study of the same week, direction models score 83.0% / 79.4% on sub_01 31–36
where the rule scores 100%. So the cascade costs interpretability and gains nothing.

## Conclusion

All three testable methods fail, and the two untestable ones that matter (a third subject, an extra sensor)
are the two the previous rounds also pointed to. The cascade result is worth keeping for the thesis: a
learned direction stage, given the same signals, does not beat `turn > 0`.
