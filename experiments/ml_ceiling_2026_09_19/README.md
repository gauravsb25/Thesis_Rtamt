# Machine learning as a ceiling estimator (19 Sep 2026)

The question was not "can we ship a model" but "how much is in these windows at all". Three models were
given **87 signals per window** — every measurement computed in every round of this project — and asked to
beat the hand-written STL monitors. The monitors themselves are unchanged and remain machine-learning free.

**Protocol.** Development: sub_01 sessions 1–30, 5-fold grouped cross-validation where whole *sessions* are
held out (windows overlap by 0.1 s, so a window-level split would leak). Held-out sets scored once, with the
models trained on all of sessions 1–30. Balanced accuracy throughout; class weights balanced.
Models: logistic regression, random forest (300 trees), gradient boosting. Code: `run.py`; logs `dev.log`, `test.log`.

## Inside the design sessions, a model is better

| 5-fold grouped CV, sub_01 sessions 1–30 | 4-class | direction only |
|---|---|---|
| **the monitors** | 85.01% | **92.6%** |
| logistic regression | 83.84% | 85.9% |
| **random forest** | **87.58%** | 90.2% |
| gradient boosting | 85.21% | 88.5% |

A random forest gains about 2.6 points at the 4-class task on the subject it was developed on — but is
already *worse* than the simple rule at direction (90.2% against 92.6%).

## On new data, no model beats the monitors

| trained on sessions 1–30, scored once | 4-class, sub_01 31–36 | 4-class, sub_11 | direction, 31–36 | direction, sub_11 |
|---|---|---|---|---|
| **the monitors** | **69.62%** | **78.67%** | **100.0%** | **70.5%** |
| logistic regression | 74.60% | 39.99% | 85.2% | 42.0% |
| random forest | 65.69% | 78.27% | 83.0% | 66.4% |
| gradient boosting | 61.65% | 71.27% | 79.4% | 75.6% |

- **No model wins on both held-out sets.** Logistic regression beats the monitors on more sub_01 sessions
  (74.60%) and then collapses on the second person (39.99%). The random forest matches the monitors on
  sub_11 (78.27 against 78.67) and loses 4 points on sub_01 31–36.
- **Average over the two held-out sets:** monitors 74.1%, random forest 72.0%, gradient boosting 66.5%,
  logistic regression 57.3%.
- **Direction is worse for every model on the sub_01 test set** (79–85% against 100%), and only gradient
  boosting is better on sub_11 (75.6 against 70.5) — at the cost of the other set.

## What the models lean on

Random forest importances, 4-class: `rob_max` 5.2%, `dir_temporal` 5.1%, `yaw_ratio_loud` 5.0%, `turn` 4.9%,
`phase_t` 4.5%, `sprt_0.0` 4.5%, `yaw_ratio` 3.9%, `nod` 2.5%, `cm_diff` 2.5%.
For LEFT versus RIGHT: `turn` 16.1% and 15.9% (the two copies of the monitor's own signal),
`dir_temporal` 14.4%, `lap_turn` 6.8%, `dir_frontal` 6.7%, `lat_signed` 5.1%.

The models rediscover the monitor's own quantities. The extra ones they use — `yaw_ratio`, `dir_temporal` —
are exactly the signals that were tested as rules this week and failed on the held-out sets for the same
reason the models do: they help one subject and hurt the other.

## Conclusion

1. **There is a little more information inside a subject** than the current rules extract (87.58 against
   85.01 within sessions 1–30), so a *per-user* system could be tuned further.
2. **There is no more information across subjects.** With 87 signals and unconstrained learners, nothing
   beats six hand-written rules on both held-out sets. That is the strongest available evidence that the
   ~80% cross-subject ceiling is a property of 8-channel EEG, not of the STL formulation.
3. **The monitors remain the better choice**: equal or better on new people, readable, and they run live.

This is the empirical support for the "impossibility" argument and for the two-sensor architecture: use EEG
as the movement trigger, and take direction from a second sensor or from a calibrated per-user setup.
