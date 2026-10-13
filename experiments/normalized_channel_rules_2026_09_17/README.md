# Normalised per-channel STL rules in the current monitors (17 Sep 2026)

User request: add the original channels as STL rules, normalised so that amplitude differences between
subjects (such as sub_11's large T4 swings) cannot matter. Design data: sub_01 sessions 1–30, blocked 3-fold
CV. Adoption rule fixed before testing: CV gain ≥ +0.5 points and more sessions better than worse. Only
passing designs go to the test sets.

## Normalised channel signals — `channel_signals.py`

| signal | normalisation |
|---|---|
| `lc[c]` | loudest ÷ quietest 0.5 s of channel c's energy, divided by the channel's own spread (log) |
| `lp[c]` | largest excursion ÷ the channel's own spread (log) |
| `dr[c]` | largest ÷ median slow-wave share of the channel (log) |
| `rel_*[c]` | the above minus the median of the 8 channels in the same window |
| `top[c]` | share of the burst energy on channel c (log, relative to the 8-channel mean) |
| `agree` | channels with the majority sign at the burst peak (sign only) |
| `coherence` | mean correlation between the 8 channels in the loudest 0.5 s (shape only) |

Each rule family uses **one threshold shared by all 8 channels**.

## Does the normalisation work? — `normalisation_check.py`

Threshold set to fire on 10% of rest windows in sessions 1–30; the same threshold is then applied to the other data.

| signal | rest 1–30 | rest sub_01 31–36 | rest sub_11 | movements 1–30 | movements sub_11 |
|---|---|---|---|---|---|
| raw amplitude (µV, not normalised) | 10% | 16% | 14% (T4 46%) | 59% | 78% |
| old slow-wave rule (max − min share, used in the 27-rule gate) | 10% | 17% | **58%** | 72% | 90% |
| `lc` (own spread) | 10% | 12% | 12% | 64% | 71% |
| `lp` (own spread) | 10% | 14% | 10% | 51% | 47% |
| `dr` (own median share) | 10% | 8% | 1% | 4% | 0% |
| `rel_lc` / `rel_lp` / `rel_dr` | 10% / 10% / 10% | 10% / 10% / 9% | 12% / 10% / 7% | 17% / 17% / 8% | 20% / 16% / 4% |

- **Raw amplitude does not transfer**, and the old slow-wave rule is what broke the 27-rule gate on sub_11.
- **Normalising by the channel's own spread (`lc`, `lp`) transfers**: 10–14% at rest everywhere, and still
  informative during movements.
- **The other normalisations transfer but carry almost no movement information** (`dr`, `rel_*`).

## Rules added to the current monitors — `channel_rules.py`, `kofn_normalised.py`

| design | CV on sessions 1–30 | gain | sessions better / worse |
|---|---|---|---|
| current monitor | 82.74% | — | — |
| gate AND ≥ k channels with `lc` ≥ t | 82.78% | +0.04 | 24 / 5 |
| gate AND ≥ k channels with `lp` ≥ t | 82.35% | −0.39 | 11 / 3 |
| gate OR ≥ k channels with `lc` ≥ t (rescue) | 83.01% | +0.27 | 9 / 18 |
| gate OR ≥ k channels with `lp` ≥ t | 82.74% | 0 | 3 / 21 |
| `dr` versions, one-channel-alone veto (`rel_*`) | 82.74% | 0 (CV kept current) | — |
| nod AND/OR sign agreement ≥ k | 82.74% | 0 (CV kept current) | — |
| nod AND coherence ≥ t | 82.74% | 0 (CV kept current) | — |
| nod OR coherence ≥ t | 82.40% | −0.34 | 0 / 1 |
| nod AND no channel dominates (`top`) | 82.74% | 0 | — |
| k-of-19 gate (3 window + 8 `lc` + 8 `lp`, shared thresholds) | 82.47% | −0.26 | 10 / 20 |
| current gate AND ≥ k of 16 `lc`/`lp` channel rules | 82.51% | −0.23 | 17 / 9 |

**No design passed, so nothing was scored on the test sets and the monitors are unchanged.**

- **Sign agreement and coherence separate nods from turns** (8 vs 5 channels; 0.59 vs 0.13), but the current nod
  signal already contains that information.
- **The earlier 27-rule gate gained +2.19 only through its un-normalised, channel-specific slow-wave
  thresholds**. That is the part that fitted sub_01 and failed on sub_11. With proper normalisation the gain disappears.
