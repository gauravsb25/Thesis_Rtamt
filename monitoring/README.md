# monitoring — four RTAMT head-movement monitors

One monitor per head movement. Each is a Signal Temporal Logic specification evaluated by
[RTAMT](https://github.com/nickovic/rtamt) (v0.4.10, bundled under `.runtime/rtamt`).

| file | what it is |
|---|---|
| `rtamt_signals.py` | turns one 3-second window (8 × 768, microvolts) into six per-sample signals |
| `rtamt_monitors.py` | the four STL specifications, `HeadMovementMonitor`, `StreamMonitor` |
| `rtamt_persistence.py` | optional RTAMT confirmation of repeated decisions on a live stream |
| `live_session.py` | timestamped live-stream session: gap handling + optional confirmation |
| `threshold_report.py` | why each threshold has its value (sub_01 sessions 1–29 only) |
| `stream_history.py` | rebuilds the stream before each stored window from the dataset's own windows |
| `row_stream.py` | the row-by-row monitor: a verdict after every incoming sample, equal to the window monitor at row 768 |
| `offline_monitor.py` | the OFFLINE monitor for finished recordings: each window judged against the rest of its own recording (86.24 / 81.52 / 78.86 on sub_01 1–20 / 21–36 / sub_11) |

A detailed, plain-English guide to every file, block by block, is in the project's `README.md`.

## Why these thresholds

```
python -m monitoring.threshold_report       # explain the current thresholds (first run measures, then cached)
python -m monitoring.threshold_report --select --train sub_11 --test sub_01:30-36   # choose on other data
python -m monitoring.threshold_report --select --save my_thresholds.json           # save a choice
python -m monitoring.threshold_report --gate cusum --direction since               # the tested options
```

`--select` only prints (or saves) a choice; it never changes the monitor's thresholds.

For each of the five thresholds it prints the rule, how it separates the classes, and a sweep of
4-class accuracy on sub_01 sessions 1–29. Every chosen value lies within 0.5 points of the best
value there; the gate g (7.81) sits 0.49 points below its best (6.63) and lets fewer rest windows
through (16.3% vs 24.5%). Sessions 30–36 and sub_11 are never read.

## See the four monitors

```
python -m monitoring                      # the four formulas + a demo on real windows
python -m monitoring --session sub_01/ses-30
```

```python
from monitoring import HeadMovementMonitor

m = HeadMovementMonitor()
m.formulas            # {"STRAIGHT": "...", "DOWN": "...", "LEFT": "...", "RIGHT": "..."}
m.classify(window)    # "LEFT" | "RIGHT" | "DOWN" | "STRAIGHT" | "INVALID"
m.classify(window, history)   # history: (8, n) uV just before the window -> stream-background gate
m.robustness(window)  # RTAMT robustness per class; > 0 means that specification is satisfied
m.explain(window)     # label + robustness + formulas
```

Each class on its own:

```python
from monitoring import StraightMonitor, DownMonitor, LeftMonitor, RightMonitor, compute_signals

signals = compute_signals(window)
LeftMonitor().robustness(signals)          # one RTAMT robustness value
LeftMonitor().robustness_trace(signals)    # robustness at every sample of the window
```

## The four specifications

`compute_signals()` also returns diagnostic direction signals `turn_fp2_t3`,
`turn_t3_t4`, `turn_fp1_f4`, `turn_vote5`, `turn_mean5`, and `turn_consensus`.
They remain available through `HeadMovementMonitor.signals(window)` but do not
participate in any monitor formula. LEFT/RIGHT still use the original F4–T3 `turn`.

```
LOUD     = once[0:767](background >= 8.0)
           or (historically[0:767](background_valid < 0.5) and once[0:767](contrast >= 5.0))
MOVE     = LOUD and once[0:767](delta >= 0.16) and once[0:767](deflect >= 2.8)
STRAIGHT = (historically[0:767](background < 8.0)
            and (once[0:767](background_valid >= 0.5) or historically[0:767](contrast < 5.0)))
           or historically[0:767](delta < 0.16) or historically[0:767](deflect < 2.8)
DOWN     = MOVE and (nod >= -0.6) and historically[0:767](sideways <= 7.5)
LEFT     = MOVE and ((nod < -0.6) or once[0:767](sideways > 7.5)) and (turn > 0)
RIGHT    = MOVE and ((nod < -0.6) or once[0:767](sideways > 7.5)) and (turn < 0)
```

Past-time operators only: RTAMT's online monitor cannot evaluate a bounded `eventually` on a
live stream. The four specifications partition the space, so exactly one is satisfied.

`background` is the stream-background gate (adopted 2026-09-17). For each channel, it divides the
0.5–3 Hz energy of the 0.5 s ending at t by that channel's median energy over the 4 s before the
window, then averages over channels. It is unit-free. `background_valid` is 1 when at least 1 s of
earlier stream exists (after a 1 s filter settle). Without it — the first window of a stream, or
`classify(window)` without `history` — LOUD is the original contrast gate and the monitor is
exactly the previous monitor.

## Live streaming

```python
from monitoring import StreamMonitor
stream = StreamMonitor()               # one verdict per non-overlapping 3-second window
for column in samples:                 # (8,) microvolts, one every 1/256 s
    label = stream.push(column)        # None until a verdict is due
```

First verdict on exactly the 768th sample. `StreamMonitor` keeps the last 5 s before each window
and passes them as `history`, so from the second window on the stream-background gate is used. A
dead channel or a non-finite sample gives `INVALID`, never an exception. About 0.36 s of compute per 3-second window (offline evaluator);
`HeadMovementMonitor(online=True)` feeds RTAMT sample by sample and gives identical robustness.

**Censored background (option, off by default, 2026-10-07).** The background is the median of the last
4 s, so a movement that fills most of them becomes the background, and a second movement soon after it is
called STRAIGHT. `StreamMonitor(RtamtConfig(background_censor=True))` (also `RowStreamMonitor`) leaves the
loud 0.5-s blocks of windows already judged LEFT/RIGHT/DOWN out of the background and looks up to 20 s back
for clean stream. On joined real runs, second movements right rise 186 → 251 of 318, but single-run false
alarms rise 348 → 1214 (sub_01) and 43 → 66 (sub_11), so it stays off. With no mask,
`classify(window, history)` is unchanged; `classify(window, history, history_usable=mask)` applies it.

## Current default (since 2026-10-07): thresholds selected on sub_01 sessions 1–20

b 8.0, g 5.0, d 0.16, q 2.8, n −0.6, v 7.5 (offline b 5.75). Four-class balanced accuracy,
`python project_code.py evaluate`:

| | sub_01 1–20 (design) | sub_01 21–36 (test) | sub_11 (test) |
|---|---|---|---|
| rules only (each window alone) | 79.45% | 78.69% | 74.50% |
| **live, with history** | **86.13%** | **83.31%** | **78.15%** |
| offline (whole recording) | 86.24% | 81.52% | 78.86% |
| live false alarms | 408 / 11,592 | 162 / 5,144 | 97 / 507 |

g = 5.0 is the floor of the search grid: without history about 30% of rest windows become false alarms.
The previous default (selected on sessions 1–30) gave 81.76% / 78.67% live on 21–36 / sub_11, with 66 and
23 false alarms. The sections below describe that previous version and its experiments.

## Accuracy with the stream background (PREVIOUS default, thresholds from sessions 1–30)

`python project_code.py evaluate` (before 2026-10-07). The history before each window is rebuilt from the dataset's own
windows (`stream_history.py`). The gate ratio 6.61 was chosen by blocked cross-validation on sub_01
sessions 1–30; the test sets were then scored once.

| class | sub_01 sessions 1–30 | sub_01 test (ses 31–36) | sub_11 |
|---|---|---|---|
| LEFT | 83.9% (805/959) | 59.0% (69/117) | 56.1% (105/187) |
| RIGHT | 80.0% (770/963) | 64.5% (118/183) | 74.7% (165/221) |
| DOWN | 78.2% (585/748) | 57.7% (71/123) | 88.4% (252/285) |
| STRAIGHT | 97.9% (15559/15890) | 97.3% (823/846) | 95.5% (484/507) |
| **4-class** | **85.0%** | **69.6%** | **78.7%** |

The gain is in rest windows: false alarms fall from 9.7% to 2.1% (sessions 1–30), 12.3% to 2.7% and
11.2% to 4.5%. The three movement classes are unchanged or slightly better. RTAMT itself agreed with
the evaluator on all 92 windows checked. Record and the other four ideas tested the same day:
`experiments/five_ideas_2026_09_17/README.md`.

## Accuracy without history: every window alone (extracted_npz_dataset_v2)

`extracted_npz_dataset_v2` is `extracted_npz_dataset` without the 2,135 windows that had been cut
across the join between two separate runs (`_archive/pipeline/remove_spliced_windows.py`; audit and content
check in `experiments/npz_audit_2026_09_17`). Design data: sub_01 sessions 1–30. Held-out: sub_01
sessions 31–36 and sub_11.

| class | sub_01 sessions 1–30 | sub_01 test (ses 31–36) | sub_11 |
|---|---|---|---|
| LEFT | 83.6% (802/959) | 59.0% (69/117) | 56.1% (105/187) |
| RIGHT | 79.4% (765/963) | 64.5% (118/183) | 74.7% (165/221) |
| DOWN | 77.5% (580/748) | 57.7% (71/123) | 88.4% (252/285) |
| STRAIGHT | 90.3% (14356/15890) | 87.7% (742/846) | 88.8% (450/507) |
| **4-class** | **82.7%** | **67.2%** | **77.0%** |

On the old `extracted_npz_dataset`, with the spliced windows, the same monitor scored 82.3%, 66.3%
and 76.4% on the same splits (70.1% on sub_01 sessions 30–36). Re-running the threshold selection on
v2 sessions 1–30 moves no threshold meaningfully (d 0.208→0.210, q 2.72→2.726, +0.06 points), so
the thresholds are unchanged. They were selected on sub_01 sessions 1–29. **sub_11 is not a fully blind
test**: no parameter was fitted on it, but it was used as an acceptance check during design and
decided the electrode pairs (F4−T3, F3/F4 rather than T3−T4, because its T4 swings by hundreds of µV during left turns; `experiments/t4_check_2026_09_17`).
A design built from sub_01 alone scores 58.1% on sub_11. A third subject is needed for a blind
estimate.

## Row by row: a verdict at every incoming EEG row (`row_stream.py`)

`HeadMovementMonitor` can only answer after a whole 3-second window, because some signals use
whole-window statistics. `RowStreamMonitor` reads the same data one row at a time:
- **causal filtering:** filters keep their state between rows;
- **prefix signals:** every signal is the monitor's measurement on rows 0…t (loudest 0.5 s so far over
  quietest so far, largest excursion so far over its spread so far, …);
- **RTAMT online:** the four class specifications are evaluated on every row.

At row 768 its verdict and robustness equal `HeadMovementMonitor` exactly. It runs at about 1–2.5 ms
per row, inside the 3.9 ms that 256 Hz allows.

It also uses the stream background. It keeps the last 5 s of rows from earlier windows; at each window
start it computes every channel's background level from them and then updates the 0.5 s energy row
by row. At row 768 it therefore equals `HeadMovementMonitor().classify(window, history)` with those
rows as history. The first window of a stream uses the contrast gate. `RowStreamMonitor(history=samples)`
seeds a history, and `clear()` forgets it.

```python
from monitoring import RowStreamMonitor
rs = RowStreamMonitor()                    # or RowStreamMonitor(commit_seconds=1.0)
for row in rows:                           # (8,) microvolts, 256 per second
    v = rs.push(row)                       # v.label (anytime), v.robustness, v.committed, v.row
```

Optional early commitment (`commit_seconds`) is the RTAMT specification
`historically[0:K-1](<class specification>)`. It commits to LEFT/RIGHT/DOWN once that class has held
for K rows; STRAIGHT is only given at row 768. Four-class balanced accuracy and median decision time
after movement onset:

Measured with the stream-background gate and the PREVIOUS (1–30) thresholds (re-measured 2026-09-19,
`experiments/commit_latency_2026_09_19`; not re-measured with the 1–20 default):

| decision | sessions 1–30 | sub_01 31–36 | sub_11 |
|---|---|---|---|
| row 768 (default) | 85.01% · 2.22 s | 69.62% · 2.20 s | 78.67% · 2.31 s |
| `commit_seconds=1.5` | 84.95% · 2.03 s | 69.60% · 2.06 s | 78.67% · 2.15 s |
| `commit_seconds=1.0` | 83.92% · 1.79 s | **69.93%** · 1.85 s | **79.57%** · 1.76 s |
| `commit_seconds=0.75` | 80.89% · 1.56 s | **69.98%** · 1.67 s | 78.97% · 1.50 s |
| `commit_seconds=0.5` | 74.85% · 1.24 s | 67.53% · 1.31 s | 77.53% · 1.23 s |

Row-wise reading does not make the monitor more accurate: its full-window verdict is the same. It
lets an application decide earlier. With `commit_seconds=1.0` decisions come about 0.45 s sooner and are
slightly better on both held-out sets (+0.31 and +0.90), costing 1.1 points on the design sessions.

## Error analysis of 2026-09-17: extra nod vetoes (tested, not adopted)

Every wrong v2 window was traced to the STL test and electrodes that caused it, and every hint of
the true class was turned into a rule and cross-validated on sub_01 sessions 1–30
(`experiments/error_attribution_2026_09_17`). One rule passed: a nod must not show a large sideways
swing on **Fp1−Fp2** or **Fp2−T3** either. It scored 82.74% → 83.86% on sessions 1–30 and 67.22% →
68.04% on sub_01 31–36, but 77.00% → 74.72% on sub_11, whose nods swing sideways. So it is off by
default. Enable it with `RtamtConfig(down_veto_fp1fp2=7.1, down_veto_fp2t3=6.6)`.

## Two options tested on 2026-09-16 and not adopted

Designed on sub_01 sessions 1–30 only, then scored once on sub_01 sessions 31–36 and sub_11
(record: `experiments/improvement_round_2026_09_16/README.md`).

- **CUSUM gate** (`gate_statistic="cusum"`) — `once(cusum >= 0.0684)` replaces `once(contrast >= g)`.
  `cusum` is Page's cumulative-sum change detector on the log-energy above the window's quietest 0.5 s,
  with a 2-nat allowance, so the energy must rise *and* stay up.
- **SINCE direction** (`direction_rule="since"`) — `(not (lateral >= 0.95)) since (lateral <= -0.95)`
  for LEFT, mirrored for RIGHT. The **last** F4−T3 excursion reaching 95% of the largest decides,
  so near-tied lobes go to the later lobe.

| 4-class balanced | sessions 1–30 (design) | sub_01 31–36 | sub_11 |
|---|---|---|---|
| default (contrast gate, peak direction) | 82.30% | **66.28%** | **76.40%** |
| CUSUM gate + SINCE direction | 83.50% | 65.05% | 75.48% |

The gain on the design sessions did not transfer. SINCE broke one RIGHT turn in each held-out set and
fixed none. CUSUM changed 23 held-out windows on sub_01 (11 fixed, 12 broken) and 22 on sub_11 (8
fixed, 12 broken). So the defaults stay. To try the options:

```python
from monitoring import HeadMovementMonitor, RtamtConfig
HeadMovementMonitor(RtamtConfig(gate_statistic="cusum", direction_rule="since"))
```
