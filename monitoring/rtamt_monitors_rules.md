# The four head-movement monitors — the rules, written out

Companion to [`rtamt_monitors.py`](rtamt_monitors.py) (live) and [`offline_monitor.py`](offline_monitor.py) (offline).
Markdown files are not part of the result-cache fingerprint, so editing this file never forces a recompute.

Print the rules exactly as RTAMT reads them:

```
python -m monitoring
```

---

## Where the rules live in the code

Line numbers as of 2026-10-14.

| File | Function | Lines | What |
|---|---|---|---|
| [`rtamt_monitors.py`](rtamt_monitors.py#L181-L219) | `class_formulas()` | 181–219 | the four **live** rules |
| | STRAIGHT | [215](rtamt_monitors.py#L215) | |
| | DOWN | [216](rtamt_monitors.py#L216) | |
| | LEFT | [217](rtamt_monitors.py#L217) | |
| | RIGHT | [218](rtamt_monitors.py#L218) | |
| [`offline_monitor.py`](offline_monitor.py#L162-L178) | `offline_formulas()` | 162–178 | the four **offline** rules |
| [`rtamt_signals.py`](rtamt_signals.py#L174-L188) | `class RtamtConfig` | 174–188 | the threshold values |
| [`../project_code.py`](../project_code.py#L269-L275) | `predict()` | 269–275 | the same rules as plain arithmetic (same answer as RTAMT on all 21,029 windows) |

---

## 1. The thresholds

Selected on sub_01 sessions 1–20 (the default since 2026-10-07). Every threshold is a ratio or a share; none is in
microvolts.

| Symbol | Value | Name in `RtamtConfig` | What it asks |
|:---:|---:|---|---|
| **b** | 8.0 | `background_ratio` | Is the window 8× louder than the 4 s before it? |
| **g** | 5.0 | `gate_ratio` | (no history) Is the loudest 0.5 s at least 5× the quietest 0.5 s? |
| **d** | 0.16 | `delta_range_min` | Did the slow-wave (1–4 Hz) share rise by at least 0.16? |
| **q** | 2.8 | `quiet_min` | Did the electrodes swing (mean prominence ≥ 2.8)? |
| **n** | −0.6 | `down_threshold` | Is the nod evidence high enough for a nod? |
| **v** | 7.5 | `down_veto_prominence` | Largest F4–T3 sideways swing a nod may have |
| **b<sub>off</sub>** | 5.75 | `OfflineConfig.background_ratio` | (offline) Is the window 5.75× louder than the rest of the recording? |

---

## 2. The live rules — `rtamt_monitors.py`

Read once the window's last sample has arrived.

| Notation | Meaning |
|---|---|
| `[0:767]` | the whole 3-second window (768 samples at 256 Hz) |
| `once[0:767](x)` | x was true at **some** moment of the window |
| `historically[0:767](x)` | x was true at **every** moment of the window |

### Building blocks

```
LOUD      =   once[0:767](background >= 8.0)                         with history
              or ( historically[0:767](background_valid < 0.5)         no history:
                   and once[0:767](contrast >= 5.0) )                  window vs itself

MOVE      =   LOUD
              and once[0:767](delta   >= 0.16)                        slow waves rose
              and once[0:767](deflect >= 2.8)                         electrodes swung

NOT_NOD   =   (nod < -0.6)  or  once[0:767](sideways > 7.5)
```

### The four rules

```
STRAIGHT  =   not MOVE, written out:
              ( historically[0:767](background < 8.0)
                and ( once[0:767](background_valid >= 0.5)
                      or historically[0:767](contrast < 5.0) ) )
              or historically[0:767](delta   < 0.16)
              or historically[0:767](deflect < 2.8)

DOWN      =   MOVE  and  (nod >= -0.6)  and  historically[0:767](sideways <= 7.5)

LEFT      =   MOVE  and  NOT_NOD  and  (turn > 0)

RIGHT     =   MOVE  and  NOT_NOD  and  (turn < 0)
```

Exactly one of the four is true for every window, so the answer is always unique.

---

## 3. The offline rules — `offline_monitor.py`

The whole recording is known. Same rules, with two differences:

- loudness is judged against the **rest of the recording** (before and after the window), with b = 5.75;
- future-time operators, read at the window's first sample: `eventually[0:767](x)` = x at **some** moment,
  `always[0:767](x)` = x at **every** moment.

```
MOVE      =   eventually[0:767](background >= 5.75)
              and eventually[0:767](delta   >= 0.16)
              and eventually[0:767](deflect >= 2.8)

STRAIGHT  =   always[0:767](background < 5.75)
              or always[0:767](delta   < 0.16)
              or always[0:767](deflect < 2.8)

DOWN      =   MOVE  and  (nod >= -0.6)  and  always[0:767](sideways <= 7.5)

LEFT      =   MOVE  and  ( (nod < -0.6) or eventually[0:767](sideways > 7.5) )  and  (turn > 0)

RIGHT     =   MOVE  and  ( (nod < -0.6) or eventually[0:767](sideways > 7.5) )  and  (turn < 0)
```

---

## 4. The same rules in plain words

Three questions, asked in order:

| | Question | Rule | Answer |
|:---:|---|---|---|
| **Q1** | Did the head move? | The window must be **loud**, the slow waves must **rise**, and the electrodes must **swing**. If any one fails… | **STRAIGHT** |
| **Q2** | Was it a nod? | Nod evidence at least −0.6, and never a sideways swing above 7.5 | **DOWN** |
| **Q3** | Which side? | The sign of the largest F4–T3 deflection: positive | **LEFT** |
| | | negative | **RIGHT** |

---

## 5. What each signal means

Computed in [`rtamt_signals.py`](rtamt_signals.py), `compute_signals`.

| Signal | Meaning |
|---|---|
| `background` | loudness of the 0.5 s ending now ÷ each channel's level in the 4 s before the window |
| `background_valid` | 1 if at least 1 s of earlier stream was available, else 0 |
| `contrast` | loudness of the 0.5 s ending now ÷ the window's quietest 0.5 s |
| `delta` | how far the 1–4 Hz share has risen above the window's lowest 0.5-s slice |
| `deflect` | how far the 8 electrodes have swung, each over its own spread (averaged) |
| `nod` | midline-versus-temporal balance + F3/F4 symmetry (high = nod-like) |
| `sideways` | the F4–T3 swing now, over that pair's own spread |
| `turn` | sign of the largest F4–T3 swing in the loudest 1.5 s (+ LEFT, − RIGHT) |

### The signals in simple words

**1. Contrast**
It cuts the window into half-second pieces and measures how much energy each piece has. It then divides each piece's energy by the energy of the quietest half-second. If some piece is at least 5 times louder than the quietest piece, the window counts as "loud". This test is only used when there are no earlier seconds to compare with.

**2. Delta**
It cuts the window into half-second pieces and asks of each one: what share of its power is slow (1–4 Hz)? It then subtracts the share of the calmest piece. If the slow share rose by at least 0.16, the head probably moved. Head movement makes slow waves, so a rise means movement. If it never rises, the window looks straight.

**3. Deflect**
For each electrode it finds the biggest swing in the window and divides it by that electrode's normal swing size. It then takes the average over all 8 electrodes. If the average is at least 2.8, the electrodes really swung.

**4. Nod**
It adds two checks together:

- Do the middle electrodes (Fz, Cz) move about as much as the temple electrodes (T3, T4)? In a nod, everything moves. In a turn, the temples move much more.
- In the loudest second, do F3 and F4 move together, in the same direction? In a nod they do. In a turn they move in opposite directions.

If the sum is at least −0.6, it looks like a nod.

**5. Sideways**
At every moment it measures the left-right swing between F4 and T3 and divides it by that pair's normal swing size. If it ever goes above 7.5, there was a big sideways swing, so the window cannot be a nod.

**6. Turn**
It finds the loudest 1.5 seconds of the window. Inside that part it finds the single biggest swing of F4 − T3, flips its sign, and divides it by its normal size. If the result is positive, the head turned LEFT; if negative, RIGHT. The sign is flipped because in a left turn T3 rises, so F4 − T3 goes down.

**Background**
It compares how loud the window is now with how loud each electrode was in the 4 seconds before the window. If the window is at least 8 times louder, it counts as "loud". When those earlier seconds exist, this test replaces contrast.

---

> **Keep in step with the code:** if a threshold or a rule changes in the `.py` files, update this file.
> `python -m monitoring` always shows the rules the code really uses.




















---

## 6. The three kinds of monitor

All three use the same four rules and the same signals. They differ in one thing only: how they decide whether
the window was "loud".

| | 1. Rules only (each window judged alone) | 2. Live monitor (with stream history) — your main monitor | 3. Offline monitor (whole recording known) |
|---|---|---|---|
| **What it sees** | The monitor sees only the 3-second window and nothing else. | It works like a real headset: data arrives second by second, and the monitor remembers the 4 seconds before the window. | For finished recordings only. It can see the whole recording, before and after the window. |
| **How it decides "loud"** | It compares the window with itself: is the loudest half-second at least 5 times its quietest half-second? | Is this window at least 8 times louder than those earlier seconds? If there are no earlier seconds yet, for example at the very start, it falls back to the rules-only test. | Is this window at least 5.75 times louder than the rest of the recording? |
| **Strength** | — | Why it's better: a small blink after a calm stretch is compared with what came just before, not with the calmest moment, so it causes far fewer false alarms. | The fewest false alarms (zero on sub_11). |
| **Weakness** | In a very calm window, even a small eye blink looks big next to the calm part, so it raises false alarms. About 30% of still windows are wrongly called movements. | — | It needs the future part of the recording, so it can't run live on a headset. It also misses more nods, so on sub_01 21–36 it scores below the live monitor. |
| **Accuracy** (sub_01 1–20 / 21–36 / sub_11) | 79.45 / 78.69 / 74.50 | **86.13 / 83.31 / 78.15** — this is what `show_accuracy.py` shows | 86.24 / 81.52 / 78.86 |

---

## 7. Accuracy of the monitors

Measured on `extracted_npz_dataset_v2` (21,029 windows of 3 s) with genuine RTAMT verdicts
(`python project_code.py evaluate` and `evaluate --offline`, verified 2026-10-07, cache fingerprint `b64ea2fddb50`).
Thresholds selected on **sub_01 sessions 1–20**; **sub_01 sessions 21–36** and **sub_11** were not used to choose anything.

- **Balanced accuracy** = the average of the four class accuracies. Guessing gives 25%. It is the main number,
  because 82% of the windows are STRAIGHT (always answering STRAIGHT would score 82% overall but 25% balanced).
- **Overall accuracy** = correct windows ÷ all windows.
- **False alarm** = a STRAIGHT (still) window that the monitor called a movement.

### 7.1 Summary — balanced accuracy

| Technique | sub_01 sessions 1–20 (design) | sub_01 sessions 21–36 (test) | sub_11 (test, second person) | All 21,029 windows |
|---|---:|---:|---:|---:|
| **Live monitor, with stream history** (the default) | **86.13%** | **83.31%** | **78.15%** | **84.80%** |
| Rules only (each window judged alone) | 79.45% | 78.69% | 74.50% | 78.88% |
| Offline monitor (whole recording known) | 86.24% | 81.52% | 78.86% | 83.50% |

### 7.2 Live monitor, with stream history (the default)

| Class | sub_01 sessions 1–20 (design) | sub_01 sessions 21–36 (test) | sub_11 (test) |
|---|---:|---:|---:|
| LEFT | 556 / 639 (87.01%) | 341 / 437 (78.03%) | 111 / 187 (59.36%) |
| RIGHT | 488 / 621 (78.58%) | 434 / 525 (82.67%) | 170 / 221 (76.92%) |
| DOWN | 399 / 484 (82.44%) | 293 / 387 (75.71%) | 272 / 285 (95.44%) |
| STRAIGHT | 11,184 / 11,592 (96.48%) | 4,982 / 5,144 (96.85%) | 410 / 507 (80.87%) |
| **Total correct / total** | **12,627 / 13,336** | **6,050 / 6,493** | **963 / 1,200** |
| Overall accuracy | 94.68% | 93.18% | 80.25% |
| **Balanced accuracy** | **86.13%** | **83.31%** | **78.15%** |
| False alarms | 408 / 11,592 | 162 / 5,144 | 97 / 507 |

All splits combined: 19,640 / 21,029 correct (93.39% overall, 84.80% balanced).

### 7.3 Rules only — each window judged alone (no stream history)

| Class | sub_01 sessions 1–20 (design) | sub_01 sessions 21–36 (test) | sub_11 (test) |
|---|---:|---:|---:|
| LEFT | 556 / 639 (87.01%) | 348 / 437 (79.63%) | 111 / 187 (59.36%) |
| RIGHT | 488 / 621 (78.58%) | 434 / 525 (82.67%) | 170 / 221 (76.92%) |
| DOWN | 399 / 484 (82.44%) | 307 / 387 (79.33%) | 272 / 285 (95.44%) |
| STRAIGHT | 8,088 / 11,592 (69.77%) | 3,762 / 5,144 (73.13%) | 336 / 507 (66.27%) |
| **Total correct / total** | **9,531 / 13,336** | **4,851 / 6,493** | **889 / 1,200** |
| Overall accuracy | 71.47% | 74.71% | 74.08% |
| **Balanced accuracy** | **79.45%** | **78.69%** | **74.50%** |
| False alarms | 3,504 / 11,592 | 1,382 / 5,144 | 171 / 507 |

All splits combined: 15,271 / 21,029 correct (72.62% overall, 78.88% balanced).
Without history the loudness test is g = 5.0 (the lowest value the threshold search allows), so about
30% of still windows become false alarms.

### 7.4 Offline monitor — the whole recording known (b = 5.75)

| Class | sub_01 sessions 1–20 (design) | sub_01 sessions 21–36 (test) | sub_11 (test) |
|---|---:|---:|---:|
| LEFT | 556 / 639 (87.01%) | 340 / 437 (77.80%) | 111 / 187 (59.36%) |
| RIGHT | 480 / 621 (77.29%) | 434 / 525 (82.67%) | 165 / 221 (74.66%) |
| DOWN | 400 / 484 (82.64%) | 259 / 387 (66.93%) | 232 / 285 (81.40%) |
| STRAIGHT | 11,359 / 11,592 (97.99%) | 5,077 / 5,144 (98.70%) | 507 / 507 (100.00%) |
| **Total correct / total** | **12,795 / 13,336** | **6,110 / 6,493** | **1,015 / 1,200** |
| Overall accuracy | 95.94% | 94.10% | 84.58% |
| **Balanced accuracy** | **86.24%** | **81.52%** | **78.86%** |
| False alarms | 233 / 11,592 | 67 / 5,144 | 0 / 507 |

All splits combined: 19,920 / 21,029 correct (94.73% overall, 83.50% balanced). The offline monitor uses
samples recorded after each window, so it is for analysing finished recordings, not for live control. It has
the fewest false alarms but misses more nods, so it is below the live monitor on sub_01 sessions 21–36.

### 7.5 Honest reading of these numbers

| Check | Result |
|---|---|
| Cross-validated estimate on the design sessions (blocks 1–7 / 8–14 / 15–20) | **82.93%** balanced, against 86.13% when scored on the same sessions that chose the thresholds |
| Previous default (thresholds chosen on sessions 1–30), same test sets, live | 81.76% on sub_01 21–36 (66 false alarms), 78.67% on sub_11 (23 false alarms) |
| Previous default, same test sets, rules only | 79.57% on sub_01 21–36, 77.00% on sub_11 |
| Independent movements in the test sets | 126 in sub_01 21–36, 95 in sub_11 (windows overlap: one movement appears in ~10 windows) |

### 7.6 Deciding earlier — row-by-row monitor (`row_stream.py`, `commit_seconds`)

Balanced accuracy · median time from the movement's start to the decision · false alarms. A commitment of k s
means one movement class held for k s; STRAIGHT is never committed early.

| Decision | sub_01 1–20 (design) | sub_01 21–36 (test) | sub_11 (test) |
|---|---|---|---|
| **Wait for the whole window (default)** | **86.13%** · 2.21 s · 408 | **83.31%** · 2.22 s · 162 | 78.15% · 2.31 s · 97 |
| Commit after 1.5 s | 86.09% · 2.01 s · 408 | 83.26% · 2.01 s · 162 | 78.15% · 2.11 s · 97 |
| Commit after 1.0 s | 84.70% · 1.75 s · 413 | 82.18% · 1.78 s · 162 | **78.99%** · 1.73 s · 98 |
| Commit after 0.75 s | 81.16% · 1.52 s · 420 | 81.34% · 1.54 s · 163 | 78.41% · 1.47 s · 101 |
| Commit after 0.5 s | 73.95% · 1.13 s · 430 | 74.87% · 1.07 s · 165 | 74.30% · 1.15 s · 104 |

At the end of the window the row-by-row monitor gives the same verdict as the window monitor on all
21,029 windows.

---

## 8. The three monitors in short

All three monitors use the same STL rules. Only the loudness test differs: what the window is compared with
(itself, the 4 seconds before it, or the rest of its run) and that test's threshold (5.0, 8.0 or 5.75). Every
other test — delta ≥ 0.16, deflect ≥ 2.8, nod ≥ −0.6, sideways ≤ 7.5, and the sign of turn — and its threshold
is identical in all three. The offline monitor writes the same rules with future-time operators
(`eventually` / `always`), which mean the same as `once` / `historically` over a finished 3-second window.

The dataset has 21,029 three-second windows. Rules only judges a window by itself. Live judges a window together
with up to 4 seconds before it, taken from the same run, as a real stream would provide them. Offline finds the
run the window came from (its cleaned CSV recording) and compares the window's energy with the normal energy of
the rest of that run, before and after the window. Every window is judged separately, and the accuracy counts all
21,029 answers.
