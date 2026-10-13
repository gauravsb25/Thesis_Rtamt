# EEG head-movement monitors with RTAMT: the complete guide

This project decides, from 3 seconds of 8-channel EEG, whether a person turned the head **LEFT**, turned it
**RIGHT**, nodded **DOWN**, or kept it **STRAIGHT**. The decision is made by four Signal Temporal Logic (STL)
rules evaluated by the RTAMT library. No machine learning takes part in the decision.

## How to run this project, step by step

Run the steps **in this order**. Each step needs the ones before it. All commands are run from the project
folder in a terminal (PowerShell on Windows, or any shell on Linux/macOS).

| Step | What you do | Command | Takes | Needs the dataset? |
|---|---|---|---|---|
| 0 | Get the code | `git clone https://github.com/gauravsb25/Thesis_Rtamt.git` then `cd Thesis_Rtamt` | seconds | no |
| 1 | Install Python 3.13 and the packages | `python -m pip install -r requirements.txt` | 1–2 min | no |
| 2 | Check that the code works | `python -m pytest -q tests` | ~3 min | no (see below) |
| 3 | See the four monitor rules | `python -m monitoring` | seconds | no |
| 4 | Put the dataset in place | (copy the folders, see below) | — | — |
| 5 | Run the live monitor on every window | `python project_code.py evaluate` | 60–90 min the first time | **yes** |
| 6 | Run the offline monitor on every window | `python project_code.py evaluate --offline` | ~20 min the first time | **yes** |
| 7 | Show the accuracy report | `python show_accuracy.py` | seconds | **yes** (uses step 5) |

**Step 1, install.** Use Python 3.13 (tested with 3.13.13, 64-bit). Do **not** install RTAMT with pip: the project
uses the copy bundled in `.runtime/rtamt`, which the code finds by itself.

**Step 2, tests.** 179 automatic checks of the monitors (rules, signals, live and offline versions). Without the
dataset, 1 test fails and 1 is skipped, because those two read the dataset; everything else must pass.

**Step 3, the rules.** Prints the four STL specifications (LEFT, RIGHT, DOWN, STRAIGHT) with the threshold values
filled in. With the dataset present, it also classifies one real window of each class.

**Step 4, the dataset.** The EEG data is **not** in this repository (it is too large). Ask the author for it and
copy it into the project folder so that these two folders exist:

| Folder | What it is | Needed for |
|---|---|---|
| `extracted_npz_dataset_v2/<subject>/<session>/clean_epochs.npz` | the 21,029 three-second windows with their labels | steps 5–7 |
| `dataset/eeg_cleaned_data/<subject>/<session>/ica_eeg/*.csv` | the original continuous recordings (one CSV per run) | step 6 (offline monitor) |

Subjects: `sub_01` (sessions 1–36) and `sub_11` (sessions 1–10).

**Step 5, the live monitor.** Runs the real RTAMT monitor on all 21,029 windows and prints a table of correct /
total per class for: sub_01 sessions 1–20 (used to choose the thresholds), sub_01 sessions 21–36 (test) and
sub_11 (test, a second person). The results are saved, so running it again takes seconds. Add `--detail` to also
see each window judged without history ("rules only").

**Step 6, the offline monitor.** The same, for the offline monitor, which compares each window with the rest of its
recording. Prints its table next to the live monitor's.

**Step 7, the report.** A short accuracy summary for presenting. `python show_accuracy.py --details --verify-live`
also re-checks 12 windows directly through RTAMT.

**Optional, after step 5:**

| Command | What it does |
|---|---|
| `python project_code.py train` | re-selects the six thresholds on sub_01 sessions 1–20 (grid search + cross-validation) |
| `python project_code.py train30` | the same on sessions 1–30, the previous version, for comparison |
| `python project_code.py windows` | writes `windows.csv`: one row per window with its label, the answer and the margins |
| `python project_code.py evaluate --fast` | the same table as step 5 in seconds, using plain arithmetic (identical answers) |

**Expected result of step 5** (balanced accuracy, guessing = 25%): **86.13%** on sub_01 1–20, **83.31%** on sub_01
21–36, **78.15%** on sub_11.

## Running the monitors on your own subjects

The monitors are fixed: their rules and thresholds were chosen on sub_01 sessions 1–20 and are **not** changed for a
new subject. A new subject is simply scored with them, exactly like sub_11 was. The process has four steps.

**1. Put the subject's data in the lab's folder format.** One folder per session, named `ses-01`, `ses-02`, …:

```
dataset/eeg_cleaned_data/sub_12/ses-01/
    labelled_epochs_ds.npz               the lab's 3-s sliding windows: `epochs` (n, channels, 768) in volts, `labels`
    ica_eeg/augmented_task-left_raw.fif   the ICA-cleaned recording of each task, with ACT_left / ACT_right /
    ica_eeg/augmented_task-right_raw.fif  ACT_down annotations marking each movement (one file per task)
    ica_eeg/augmented_task-down_raw.fif
    ica_eeg/sub-12_ses-01_task-left_run-01_eeg.csv    one CSV per run: columns Fp1 Fp2 F3 Fz F4 T3 Cz T4 (volts)
    ...
```

Requirements: the same 8 electrodes (Fp1, Fp2, F3, Fz, F4, T3, Cz, T4), 256 Hz, ICA-cleaned like the existing data.
The `.fif` files need `mne` (in `requirements.txt`).

**2. Cut the subject's clean 3-second windows.**

```powershell
python project_code.py extract sub_12
```

For every session it places each of the lab's windows in its recording, labels it from the `ACT_` annotations,
and keeps only clean windows: a movement window must hold the **whole** movement with 0.25 s to spare at both
edges, a rest window must be at least 1 s away from any movement, and every window must lie inside one recorded
run. It prints, per session and class, how many windows were kept. The result goes to
`extracted_npz_dataset_v2/sub_12/`. (This is exactly how the existing dataset was made: re-extracting sub_11 this
way reproduces its 1,200 windows identically.)

**3. Run the monitors on every window.**

```powershell
python project_code.py evaluate            # the live monitor (recomputed because the dataset changed: ~1 h+)
python project_code.py evaluate --offline  # optional: the offline monitor
```

**4. Print the accuracy table, one row per subject.**

```powershell
python project_code.py subjects            # live monitor
python project_code.py subjects --offline  # offline monitor
```

Each row gives, per class, correct / total windows and the percentage, then the four-class **balanced accuracy**
(the average of the four class percentages; guessing gives 25%) and the false alarms (still windows called a
movement). sub_01 is split into its design sessions (1–20) and its test sessions (21–36); every other subject is a
test subject, scored on all its sessions.

**Reading the result.** Compare each new subject's balanced accuracy with sub_11 (78.15%), the only other subject
the monitors were not designed on. Expect variation between people: direction (LEFT/RIGHT) transferred worst from
sub_01 to sub_11, detection (moved or not) transferred well.

## About this guide

This README is written so that someone who has never seen the code can understand all of it. Read it top to
bottom the first time. The code itself is kept free of long comments; this guide is where each block is
explained, with the line numbers to find it.

## Installing and running after a fresh clone (author's own setup)

The code is on GitHub; the data is not (it is too large). After a reset, restore both like this.

**1. Clone the repository.**

```powershell
git clone https://github.com/gauravsb25/Thesis_Rtamt.git
cd Thesis_Rtamt
```

**2. Install Python and the packages.** Tested with Python 3.13.13 (64-bit, Windows 11).

```powershell
python -m pip install -r requirements.txt
```

Do **not** `pip install rtamt`: PyPI only has 0.3.5. The monitors use the bundled copy in `.runtime/rtamt`
(RTAMT 0.4.10 from GitHub master, patched for Python 3.13), which `monitoring` imports automatically. Keep
that folder.

**3. Copy the data back from the backup**, into exactly these places:

| folder | what it is | needed for |
|---|---|---|
| `dataset/eeg_cleaned_data/<subject>/<session>/` | the original recordings: `ica_eeg/*.csv`, `ica_eeg/*.fif` (with the `ACT_*` and `BAD_noise` annotations) and `labelled_epochs_ds.npz` | the offline monitor, live replays, rebuilding datasets |
| `extracted_npz_dataset_v2/<subject>/<session>/clean_epochs.npz` (+ its `README.md`, `manifest.json`) | THE dataset: 21,029 windows | every accuracy table |
| `D:/errp/eeg_cleaned_data/<subject>/<session>/pp_eeg/*.csv` | the raw (pre-ICA) recordings, in microvolts | only the raw-data experiments (`experiments/live_replay_2026_10_07`, `experiments/raw_live_2026_10_13`); if you put them elsewhere, change the `RAW` path at the top of those scripts |

Subjects: `sub_01` (36 sessions) and `sub_11` (10 sessions).

**4. Check it works, then rebuild the result caches.** The caches (`*.npz`) are not on GitHub. They are
rebuilt automatically, because each one is tied to a fingerprint of the code and the dataset files.

```powershell
python -m pytest -q tests                  # 179 tests, about 2 minutes (before the data is restored: 1 fails, 1 skips -- both read the dataset)
python -m monitoring                       # prints the four STL rules and classifies a few real windows
python project_code.py evaluate            # genuine RTAMT verdicts for all windows: 60-90 min the first time
python project_code.py evaluate --offline  # the offline monitor: about 20 min the first time
python show_accuracy.py                    # the accuracy report (works once evaluate has run)
```

**Not on GitHub** (restore from the backup if needed): the datasets above, all `*.npz` / `*.npy` caches,
experiment caches (they are rebuilt by each experiment's own scripts), `telex2019_paper/`, and `_archive/`
apart from its `README.md`, `pipeline/` (the scripts that build the datasets) and `lib/`.

## Contents

0. [Quick start](#0-quick-start)
1. [Results at a glance](#1-results-at-a-glance)
2. [The ideas, in plain words](#2-the-ideas-in-plain-words)
3. [Reading order for all documents](#3-reading-order-for-all-documents)
4. [The code, file by file](#4-the-code-file-by-file)
5. [How one number in a table is produced](#5-how-one-number-in-a-table-is-produced)
6. [Where each technique lives](#6-where-each-technique-lives)
7. [Try it yourself](#7-try-it-yourself)
8. [The folders](#8-the-folders)
9. [Glossary](#9-glossary)

---

## 0. Quick start

Open PowerShell in the project folder.

```powershell
cd C:\Users\gaura\thesis_rtamt
python show_accuracy.py                          # the accuracy report (live monitor)
python show_accuracy.py --details --verify-live  # + confusion matrices + a fresh RTAMT recheck of 12 windows
python -m monitoring                             # print the four STL rules and classify real windows
python project_code.py evaluate --detail         # live monitor: with history, and each window alone
python project_code.py evaluate --offline        # the offline monitor
python -m pytest -q tests                        # 179 automatic tests
```

Save the report:

```powershell
python show_accuracy.py --json reports/accuracy.json
python show_accuracy.py --details --no-color | Out-File -Encoding utf8 reports/accuracy.txt
```

`show_accuracy.py` reads genuine RTAMT verdicts from `monitor_labels_v2.npz`, and only if that file was made by
exactly the current code, settings and dataset (its "fingerprint", see section 5). If the code changed, it
refuses and asks you to run `python project_code.py evaluate`, which recomputes every verdict (about 60–90
minutes). Nothing in a table is typed in by hand; every percentage is counted from saved RTAMT verdicts.

Requirements: Python, NumPy and SciPy (pytest for the tests); see `requirements.txt` and
[Installing and running after a fresh clone](#installing-and-running-after-a-fresh-clone). RTAMT is bundled in
`.runtime/rtamt`; keep that folder.

---

## 1. Results at a glance

**Four-class balanced accuracy** is the average of the four class accuracies. Guessing gives 25%. It is the
main number because 82% of the windows are STRAIGHT: a program that always says STRAIGHT would get 82% plain
accuracy but only 25% balanced.

**Default since 2026-10-07: thresholds selected on sub_01 sessions 1–20** (b 8.0, g 5.0, d 0.16, q 2.8,
n −0.6, v 7.5; offline b 5.75). Design = sessions 1–20; tests = sub_01 sessions 21–36 and sub_11.

| Technique | sub_01 sessions 1–20 (design) | sub_01 sessions 21–36 (test) | sub_11 (test, second person) |
|---|---|---|---|
| Rules only: each window judged alone | 79.45% | 78.69% | 74.50% |
| **Live monitor: with 5 s of stream history** (the current monitor) | **86.13%** | **83.31%** | **78.15%** |
| Offline monitor: whole recording known | 86.24% | 81.52% | 78.86% |

Per class, correct windows / all windows of that class:

| Class | Rules only 1–20 | Live 1–20 | Offline 1–20 | Rules only 21–36 | Live 21–36 | Offline 21–36 | Rules only sub_11 | Live sub_11 | Offline sub_11 |
|---|---|---|---|---|---|---|---|---|---|
| LEFT | 556/639 | 556/639 | 556/639 | 348/437 | 341/437 | 340/437 | 111/187 | 111/187 | 111/187 |
| RIGHT | 488/621 | 488/621 | 480/621 | 434/525 | 434/525 | 434/525 | 170/221 | 170/221 | 165/221 |
| DOWN | 399/484 | 399/484 | 400/484 | 307/387 | 293/387 | 259/387 | 272/285 | 272/285 | 232/285 |
| STRAIGHT | 8,088/11,592 | 11,184/11,592 | 11,359/11,592 | 3,762/5,144 | 4,982/5,144 | 5,077/5,144 | 336/507 | 410/507 | 507/507 |
| **Balanced** | **79.45** | **86.13** | **86.24** | **78.69** | **83.31** | **81.52** | **74.50** | **78.15** | **78.86** |
| False alarms | 3,504 | 408 | 233 | 1,382 | 162 | 67 | 171 | 97 | 0 |

**The cost of this default.** g = 5.0 is the lowest value the threshold search allows. With stream history
it rarely decides; without history (rules only, or the first seconds of a stream) about 30% of rest windows
become false alarms. Live false alarms are 2.5–4 times the previous default's.

**Previous default (thresholds selected on sessions 1–30, used until 2026-10-07)**, scored on the same
test sets: live 81.76% on 21–36 and 78.67% on sub_11 (66 and 23 false alarms); rules only 79.57% and
77.00%. On the old split it gave 85.01 / 69.62 / 78.67 (live), 82.74 / 67.22 / 77.00 (rules only) and
86.73 / 71.80 / 79.37 (offline, b 4.95). Reproduce: `python project_code.py train30`.

What the rows mean is explained in section 2.10. In short:

- **The live monitor is the current monitor.** It can run on a live headset.
- **The offline monitor is for finished recordings only.** It uses samples recorded after each window. With the
  1–20 thresholds it removes the most false alarms but misses more nods, so it is below the live monitor on
  sub_01 21–36.
- **Rules only is the same monitor without any history.** It is the weakest, mainly because of false alarms.

Honesty notes: sessions 1–20 were used to choose the thresholds, so their score is optimistic. The two test sets
have been looked at in many earlier rounds, so they are "retrospective", not blind. Windows overlap (one
movement appears in about ten windows), so the test sets hold only 126 (21–36) and 95 (sub_11) independent movements.

---

## 2. The ideas, in plain words

Read this section before any code.

### 2.1 The data

- A cap holds **8 electrodes**: Fp1, Fp2, F3, Fz, F4, T3, Cz, T4.
  - Fp = forehead, F = front, C = top of the head, T = temple.
  - Odd numbers are on the left, even numbers on the right, z is the middle line.
  - So **T3 = left temple** and **T4 = right temple**.
- Each electrode gives **256 numbers per second** (256 Hz). The numbers are voltages in microvolts (µV).
- The program looks at **3 seconds at a time**: 3 × 256 = **768 numbers per electrode**. One **window** is a table
  of 8 rows × 768 columns.
- Every window has its true answer written next to it, its **label**: LEFT, RIGHT, DOWN or STRAIGHT.
- The dataset `extracted_npz_dataset_v2` has **21,029 windows**: LEFT 1,263, RIGHT 1,367, DOWN 1,156, STRAIGHT 17,243.
- Windows are cut every 0.1 s, so one real movement appears in about ten overlapping windows.
- Two people were recorded: `sub_01` on 36 days (sessions) and `sub_11` on 10 days.
- The original recordings are CSV files in `dataset/eeg_cleaned_data/<person>/<session>/ica_eeg/`, one file per
  short run (about 7–15 seconds) containing one movement.

### 2.2 What a head movement looks like in the signal

The monitors do **not** read thoughts. When the head moves, three things happen together: the neck muscles pull,
the eyes move, and the electrodes and wires shift on the skin. Together they make a **slow, large bump** in the
signal, much bigger than ordinary brain activity.

- **A turn** moves the left and right sides of the head in **opposite** directions, because the head rotates.
  *Example:* in a LEFT turn, T3 (left temple) goes up while T4 (right temple) goes down.
- **A nod** moves the whole head the **same** way. The middle electrodes (Fz, Cz) move as much as the temples.
- **Left versus right** is told by **which way the bump points**, its sign.

### 2.3 Filters

A **band-pass filter** keeps only wiggles of a certain speed, like the bass or treble knob on a radio. A
**0.5–3 Hz** filter keeps wiggles that repeat 0.5 to 3 times per second, which is the speed of a head movement,
and removes slow drift and fast noise.

| Name in the code | Keeps | Used for |
|---|---|---|
| `_BAND_DETECT` | 0.5–3 Hz | finding loud moments (the "did it move?" test) |
| `_BAND_CLASS` | 0.5–6 Hz | channel swings and the nod evidence |
| `_BAND_DIRECTION` | 0.3–6 Hz | reading left or right |
| `_BAND_PROMINENCE` | 0.3–4 Hz | the sideways-swing check |

All filters are **causal**: the value at a moment uses only earlier samples. That is what lets the same code run
on a live stream.

### 2.4 Energy, spread and ratios

- **Energy** of a stretch of signal is the average of its squared values. Bigger wiggles give more energy.
  *Example:* wiggles of ±2 µV give energy about 4; wiggles of ±10 µV give about 100.
- **Spread** is the size of the normal wiggle. The code uses a robust version: the median distance from the
  middle, times 1.4826. A single huge spike cannot inflate it.
- **Ratio**. Almost every measurement divides something by something of the same kind, such as loud energy ÷
  quiet energy, or a swing ÷ the normal wiggle.
  - *Why:* one person's signals can be ten times bigger than another's, simply because the cap sits differently.
    In a ratio, the ×10 appears on top and bottom and cancels.
  - That is why no rule ever compares a signal with a fixed microvolt value. The tests check that multiplying a
    whole window by 250 changes nothing.

### 2.5 The measurements ("signals")

From one window the code computes these. Each is a list of 768 numbers, one per moment of the window. The last
column gives the middle half of the values the rules actually read (the maximum, or for nod and turn the last
value) over the whole dataset.

| Signal | The question it answers | Middle half of values in the dataset |
|---|---|---|
| `contrast` | Is some half-second much louder than the **quietest** half-second of the **same window**? | rest 3.4–6.4; movements 12–47 |
| `delta` | Did the slow waves (1–4 Hz) rise, compared with the calmest part of the window? | rest 0.16–0.24; movements 0.31–0.47 |
| `deflect` | Did the electrodes really swing? Each channel's biggest swing ÷ its normal spread, averaged over the 8 channels | rest 2.8–3.2; movements 3.2–5.0 |
| `nod` | Do the middle electrodes move as much as the temples, and do F3 and F4 move together? | turns −4.1 to −1.7; nods +0.2 to +2.2 |
| `sideways` | Size of the biggest left–right swing (F4 minus T3) ÷ its normal size | nods 2.5–3.7; turns 4.0–6.3 |
| `turn` | Which way did the biggest F4 − T3 swing point, in the loudest 1.5 s? | LEFT +2.2 to +4.0; RIGHT −3.6 to −2.2 |
| `background` (live) | Is this half-second louder than the same electrodes were in the **4 s before** the window? | rest 1.5–3.1; movements 15–57 |
| `background` (offline) | Is it louder than the **rest of the whole recording**, before and after the window? | rest 1.5–2.6; movements 9–42 |

No single signal separates everything; each rule combines several. For example, many rest windows pass the
deflect test on their own, but very few pass it **and** the loudness **and** the slow-wave test together.

### 2.6 STL: the small language the rules are written in

**Signal Temporal Logic** makes statements about a signal over time. The words used in this project:

| Word | Meaning | Looks | Used by |
|---|---|---|---|
| `once[0:767](x >= 5)` | at **some** moment in the window, x was at least 5 | backwards from the end | live monitor |
| `historically[0:767](x < 5)` | at **every** moment in the window, x was below 5 | backwards from the end | live monitor |
| `eventually[0:767](x >= 5)` | at some moment in the window | forwards from the start | offline monitor |
| `always[0:767](x < 5)` | at every moment in the window | forwards from the start | offline monitor |
| `and`, `or`, `not` | as in English | | all |

`[0:767]` means the 768 samples of the window. `once` and `eventually` ask the same question from opposite ends.
The live monitor must look backwards because a live stream has no future. The offline monitor has the finished
recording, so it may look forwards.

*Example:* `delta` over a window is 0.05, 0.10, 0.42, 0.30, … Then `once(delta >= 0.16)` is **true**, because
0.42 ≥ 0.16 at one moment, and `historically(delta < 0.16)` is **false**.

### 2.7 RTAMT and robustness

**RTAMT** is the library that reads an STL rule and checks it against the signals. It is bundled in
`.runtime/rtamt` (version 0.4.10). It does not answer only yes or no. It returns **robustness**, a number that
says how clearly the rule holds:

| Piece of a rule | Robustness | Example |
|---|---|---|
| `x >= k` | x − k | contrast 25, k = 5.0 → **+20.00**, clearly true |
| `x < k` | k − x | contrast 4.2, k = 5.0 → **+0.80**, just true |
| `A and B` | the **smaller** of A and B | +20.00 and −0.30 → −0.30: false, because B fails |
| `A or B` | the **larger** of A and B | +20.00 or −0.30 → +20.00: true |
| `once` / `eventually` | the **largest** value over time | |
| `historically` / `always` | the **smallest** value over time | |

**Positive robustness means the rule holds**; negative means it fails. The size is the margin. That is how
`project_code.py` can reproduce RTAMT with simple minimum and maximum arithmetic (section 4.5, `predict`).

### 2.8 The four rules and the three questions

The live monitor's rules (with its thresholds filled in):

```
LOUD     = once(background >= 8.0)   or   (no history  and  once(contrast >= 5.0))
MOVE     = LOUD  and  once(delta >= 0.16)  and  once(deflect >= 2.8)
STRAIGHT = not MOVE
DOWN     = MOVE  and  (nod >= -0.6)  and  historically(sideways <= 7.5)
LEFT     = MOVE  and  ((nod < -0.6) or once(sideways > 7.5))  and  (turn > 0)
RIGHT    = MOVE  and  ((nod < -0.6) or once(sideways > 7.5))  and  (turn < 0)
```

1. **Did the head move?** It must be loud, the slow waves must rise, and the electrodes must swing. If any of the
   three fails, the answer is STRAIGHT.
2. **Was it a nod?** Nod evidence high enough, and never a big sideways swing, means DOWN.
3. **Left or right?** Otherwise the sign of `turn` decides.

Exactly one of the four rules is true for any window, so there is always exactly one answer. The tests check this.

**Worked example 1, a left turn.** Contrast 25 ≥ 5.0, delta 0.40 ≥ 0.16, deflect 4.2 ≥ 2.8, so the head moved.
Nod −3.5 < −0.6, so it is not a nod. Turn +3.3 > 0, so the answer is **LEFT**.
In the rules-only version, the robustness of LEFT = min(contrast margin, delta margin, deflect margin, not-nod
margin, turn) = min(20.00, 0.24, 1.40, 2.90, 3.3) = **+0.24**. It is positive, so LEFT holds, and its weakest part
was delta (0.40 − 0.16 = 0.24): a slightly weaker slow-wave rise would have made it STRAIGHT.

**Worked example 2, a real nod from the dataset** (`sub_01/ses-05`, window 10, true label DOWN). Contrast 53.6,
delta 0.40 and deflect 3.80 all pass, so the head moved. Nod +1.40 ≥ −0.6 and sideways 3.1 never passed 7.5,
so the answer is **DOWN**, which is correct. Section 7 shows how to print these numbers yourself.

**Worked example 3, a false alarm.** A small leftover eye blink in a very calm window: contrast 12, delta 0.25 and
deflect 3.0 all pass, because next to the calm part the blink looks big. Nod +0.5 and sideways 3, so the answer is
**DOWN**. That is wrong; the person was still. Comparing with the seconds before (live) or the whole recording
(offline) catches most of these (section 2.10).

### 2.9 Thresholds, training and testing

- A **threshold** is a number inside a rule: 8.0, 5.0, 0.16, 2.8, −0.6, 7.5, and 5.75 offline (default since
  2026-10-07; before that 6.61, 7.81, 0.208, 2.72, −0.661, 7.54 and 4.95, chosen on sessions 1–30).
- **Training** means choosing these numbers: try many values and keep those with the best balanced accuracy on
  the **training (design) sessions**, sub_01 sessions 1–20.
- **Testing** means scoring the finished monitor on sessions it never used: sub_01 sessions 21–36 and all of
  sub_11.
- **Cross-validation** is a stricter check inside training. Split sessions 1–20 into three blocks (1–7, 8–14,
  15–20). Choose the numbers on two blocks, score on the third, and repeat three times. An idea is adopted only if
  it wins on sessions that did not choose its numbers, improves more sessions than it harms, and does not lose on
  the test sets.
- A **false alarm** is a still (STRAIGHT) window that the monitor calls a movement.

### 2.10 The three techniques

All three use the **same four rules**. They differ only in how the first test, "was it loud?", is measured.

| Technique | "Loud" means | Runs live? | Where |
|---|---|---|---|
| **Rules only** (each window alone) | the loudest half-second is ≥ 5.0 × the **quietest half-second of the same window** | yes | `HeadMovementMonitor().classify(window)` |
| **Live, with history** | the loudest half-second is ≥ 8.0 × the level of the **4 s just before** the window; with less than 1 s of usable history it falls back to rules only | yes | `HeadMovementMonitor().classify(window, history)` |
| **Offline** | the loudest half-second is ≥ 5.75 × the level of the **rest of the whole recording**, before and after | no | `OfflineHeadMovementMonitor().classify(recording, start)` |

*Example.* A person chews for a few seconds, and then a small bump appears.
- **Rules only** compares the bump with the calmest part of its own window, where it looks big, so it says
  "moved".
- **Live** compares it with the busy chewing seconds just before, where it looks ordinary, so it says STRAIGHT.
- **Offline** compares it with the whole recording and also says STRAIGHT.

*Why the offline monitor wins on this dataset.* Each recording is a short run, and the movement comes 2–3 s after
it starts. The live background needs 2 s of history: 1 s for the filter to settle and 1 s to measure. So 74.5% of
movement windows have too little history and fall back to rules only. The offline monitor also uses the seconds
**after** the window, so every window gets a proper comparison. On a real, continuous live stream this handicap
would not exist.

### 2.11 What is still hard

- **Turn versus nod in later sessions of sub_01.** Inside each session the nod measurement separates turns from
  nods almost perfectly, but where the dividing line falls moves from session to session.
- **Left versus right on sub_11.** Its movement artifact is a mirror image of sub_01's, so the direction sign is
  unreliable.
- **Only two people.** A third person is needed for a truly blind number.

Details: `experiments/improvement_round_2026_09_24/README.md`.

---

## 3. Reading order for all documents

Everything needed is in this repository; this README is the main document.

| # | File | What it gives you |
|---|---|---|
| 1 | **This README, "How to run" and sections 1–2** | How to run it, the results, and every idea in plain words |
| 2 | `monitoring/rtamt_monitors_rules.md` | The four rules and thresholds, the three kinds of monitor, all accuracy tables |
| 3 | `monitoring/signals.md` | The seven signals in depth, with their formulas |
| 4 | **This README, section 4** | Every code file, block by block |
| 5 | `experiments/experiments.md` | Every experiment in plain words: the idea, the result, kept or not |
| 6 | `experiments/<folder>/README.md` | The full record of one experiment (method, numbers, scripts) |

---

## 4. The code, file by file

Read the files in this order; each needs only the ones before it. Line numbers are for the files as they are on
14 October 2026.

| # | File | Lines | Role |
|---|---|---|---|
| 1 | `monitoring/rtamt_signals.py` | 541 | One window → the measurements |
| 2 | `monitoring/rtamt_monitors.py` | 490 | The four rules and RTAMT: **the live monitor** |
| 3 | `monitoring/stream_history.py` | 110 | Rebuilds "the seconds before a window" from the dataset |
| 4 | `monitoring/offline_monitor.py` | 277 | **The offline monitor** |
| 5 | `project_code.py` | 670 | Runs the monitors on the dataset, scores them, chooses thresholds |
| 6 | `show_accuracy.py` | 110 | The accuracy report for a presentation |
| 7 | `monitoring/row_stream.py` | 367 | Live: an answer after every incoming sample |
| 8 | `monitoring/live_session.py` | 91 | Live headset wrapper: timestamps, gaps |
| 9 | `monitoring/rtamt_persistence.py` | 49 | Optional "say it twice" confirmation |
| 10 | `monitoring/threshold_report.py` | 462 | A tool that explains and re-chooses thresholds |
| 11 | `monitoring/__init__.py`, `__main__.py` | 64, 6 | Entry points |
| 12 | `tests/` | 8 files | 179 automatic checks |

If you only have one hour, read files 1 and 2. That is the monitor; everything else runs or reports it.

---

### 4.1 `monitoring/rtamt_signals.py`: from raw EEG to measurements

**Lines 1–106: the description.** Plain text. It lists every signal and states which STL operator recovers which
measurement. Worth reading once.

**Lines 108–114: imports.** `numpy` handles arrays of numbers. `scipy.signal.butter` designs a filter and
`lfilter` applies it causally.

**Lines 116–133: constants.**
```python
FS = 256                                   # 128: samples per second
WINDOW_SAMPLES = 3 * FS                    # 130: 768 samples = 3 s
CHANNELS = ("Fp1","Fp2","F3","Fz","F4","T3","Cz","T4")   # 133: the row order of every window
_IDX = {c: i for i, c in enumerate(CHANNELS)}            # 135: _IDX["T3"] is 5
```
- `CORE_SIGNALS` (120) lists the signals the rules may read.
- `AUXILIARY_SIGNALS` (122) are extra direction signals for inspection only; no rule reads them.
- `_DIRECTION_PAIRS`, `_VOTE_PAIRS`, `_MEAN_PAIRS` (125–132) feed only those extra signals.
- `_TINY = 1e-12` (133) is a tiny number that avoids an exact zero.

**Lines 136–143: the four filters.** `_band(lo, hi)` (136) builds a 2nd-order Butterworth band-pass filter. Lines
165–168 create the four filters of section 2.3.

**Lines 146–203: `RtamtConfig`, every number in one place.** A frozen dataclass: once made it cannot change;
build a new one with `dataclasses.replace(RtamtConfig(), gate_ratio=7.0)`.

| Line | Field | Value | Meaning |
|---|---|---|---|
| 174 | `gate_ratio` | 5.0 | contrast threshold: the rules-only "loud" test |
| 176 | `delta_range_min` | 0.16 | how much the slow-wave share must rise |
| 178 | `quiet_min` | 2.8 | how much the electrodes must swing (deflect) |
| 179 | `down_threshold` | −0.6 | nod threshold |
| 180 | `down_veto_prominence` | 7.5 | the biggest sideways swing a nod may have |
| 188 | `background_ratio` | 8.0 | the live "loud" test, against history |
| 189 | `background_seconds` | 4.0 | how much history the background uses |
| 190 | `background_min_seconds` | 1.0 | less usable history than this means "no history" |
| 191 | `background_settle_seconds` | 1.0 | the first second of history is skipped while the filter settles |

The other fields hold options that were tested and rejected: the CUSUM gate, the SINCE direction rule, and two extra
nod vetoes. They are off by default and remain only for reproducibility. `__post_init__` (195–203) refuses
impossible settings.

**Lines 208–209: `InvalidWindow`.** The error raised for a window that cannot be judged. The monitor turns it into
the answer `INVALID` instead of crashing.

**Lines 212–225: `check_window`, the doorman.**
- 215–216: the window must be 8 rows.
- 217–218: it must have at least 768 samples.
- 219: only the first 768 are kept.
- 220–221: every number must be finite (no NaN or infinity).
- 222–224: no electrode may be completely flat.

*Example:* if the Cz wire falls off and Cz reads 0.0 everywhere, the answer is INVALID, not a guess.

**Lines 228–233: `_pooled_energy`, "how much is happening, moment by moment".**
1. Filter every channel to 0.5–3 Hz.
2. Divide each channel by its own robust spread, so a naturally large channel cannot dominate.
3. Square, and average over the 8 channels.

The result is one number per sample: about 1 at rest, 20 or more during a movement.

**Lines 236–239: `_block_sums`.** The total energy in every block of `width` samples, computed with a running total
(`cumsum`), which is fast. With width 128 it gives the energy of every half-second.

**Lines 242–258: `_running_loudest`.** For every moment, where the loudest block **seen so far** starts. At the last
sample this is the loudest block of the whole window. The "so far" version is what past-time STL needs.

**Lines 291–332: `stream_background`, the live history test.** Inputs: the window (`raw`) and `history`, the samples
just before it.
- 307–308: no history, or the feature switched off, gives all zeros and "not valid", which means rules only.
- 310–313: history must be 8 rows of finite numbers.
- 321: `start` = where the window begins inside history + window.
- 314–320: an optional `usable` mask (the censored background of 2026-10-07; off by default, see `background_positions`, 261–274).
- 322: `background_positions` skips the first second of history (filter settling) and keeps at most the last 4 s.
- 323–324: less than 1 s of usable history gives "not valid".
- 326: filter history and window **together**, so the filter runs continuously into the window.
- 328–329: the energy of every half-second, per channel.
- 330: each channel's **reference** = its median energy over the usable history (every 8th sample).
- 331: `background` = for each moment of the window, each channel's energy ÷ its reference, averaged over the 8
  channels.

*Example:* in the quiet history T3's reference is 4. During a turn T3's energy is 60, so its ratio is 15. The
average over the 8 channels might be 9, which is ≥ 8.0, so the window is loud.

**Lines 335–454: `compute_signals`, the main function.** Called once per window; returns a dictionary of signals,
each 768 values long.

- **343–347, setup.** Check the window (344). Compute pooled energy (346). Compute the background, all zeros if no
  history is given (347).
- **349–353, `contrast`.** Every half-second's energy ÷ the quietest half-second's energy. Its maximum is what
  rules only compares with 5.0.
  *Example:* quietest half-second 2, loudest 50 → contrast 25.
- **354–364, `cusum`.** A tested option, off by default. It counts only energy that rises **and stays up**.
- **365–383, `delta`.**
  1. Cut the window into half-second slices, stepping a quarter second.
  2. For each slice, an FFT gives the share of power that is slow (1–4 Hz), averaged over channels.
  3. Subtract the lowest share of any slice in the window.

  *Example:* calm slices 0.20, a movement slice 0.62 → delta reaches 0.42.
  *Note:* a half-second FFT has 2 Hz resolution, so "1–4 Hz" is in practice the 2 Hz bin.
- **384–388, `deflect`.** Filter 0.5–6 Hz and centre each channel. Take each channel's running biggest swing ÷ its
  normal spread, and average over the 8 channels.
  *Example:* T3 normally wiggles ±3 µV and swings to 15 µV → 5 for T3.
- **389–407, `nod`.** Two parts, added at line 406:
  - *balance* = log(energy of Fz and Cz ÷ energy of T3 and T4). A turn makes the temples huge, so this is very
    negative; a nod moves everything, so it is near zero.
  - *symmetry* = log(energy of F3+F4 ÷ energy of F3−F4) in the loudest second. If F3 and F4 move together (a
    nod), the sum is big and the value positive; if they move oppositely (a turn), the difference is big and the
    value negative.
- **408–419, `sideways`.** Filter 0.3–4 Hz, take F4 − T3, and divide its size at every moment by its normal size
  (the `excursion` helper, 411–414). A value above 7.5 anywhere means a clear left–right swing, which vetoes
  DOWN. The same is computed for the pairs Fp1−Fp2 and Fp2−T3, which only the rejected extra vetoes read.
- **420–435, `turn`.** Filter 0.3–6 Hz and take F4 − T3. In the loudest 1.5 s seen so far, find the single biggest
  swing, flip its sign, and divide by its normal size (432). Positive means LEFT, negative means RIGHT.
  *Why the minus sign:* in a LEFT turn T3 rises, so F4 − T3 falls, giving a negative peak; minus that is positive,
  which is LEFT.
- **436–445, `lateral`.** Only for the rejected SINCE option.
- **446–454, result.** Collect everything; add diagnostic direction signals that no rule reads.

**Lines 457–481: `_direction_bank`.** The same biggest-swing reading on eight other electrode pairs, plus a vote
and a mean. Inspection only.

---

### 4.2 `monitoring/rtamt_monitors.py`: the four rules and RTAMT (the live monitor)

**Lines 1–136: the description.** The formulas, what they mean, measured accuracy, live streaming.

**Lines 161 and 162.** `CLASSES` lists the four answers. `LAST = 767` is the last sample index; the rules are read
there, so they judge the whole window.

**Lines 165–178: `import_rtamt`.** Uses an installed RTAMT, otherwise the copy bundled in `.runtime/rtamt`.

**Lines 181–219: `class_formulas`, where the rules are written.** Builds the four rules as text with the
thresholds filled in. Run `python -m monitoring` to print them.
- 184–187: which loudness statistic to use; contrast by default.
- 190: `w = "[0:767]"`, the whole window.
- 191–192: **rules-only loudness**, `once(contrast >= 5.0)`, and its opposite.
- 193–198: **the live technique.** If a background threshold exists, "loud" becomes
  `once(background >= 8.0) or (historically(background_valid < 0.5) and once(contrast >= 5.0))`. This single
  `or` is the entire difference between "live with history" and "rules only".
- 199: MOVE = loud **and** `once(delta >= 0.16)` **and** `once(deflect >= 2.8)`.
- 200–206: the sideways veto for DOWN; the extra vetoes are added only if switched on.
- 207: "not a nod" = `nod < −0.6` **or** a big sideways swing.
- 208–213: direction: `turn > 0` is LEFT, `turn < 0` is RIGHT.
- 214–219: the four finished rules. STRAIGHT is exactly "not MOVE".

**Lines 222–267: `ClassMonitor`, one rule handed to RTAMT.**
- 227–232, `__init__`: store the formula and find which signal names it uses (230).
- 234–242, `_spec`: create an RTAMT specification, declare every signal as a variable, parse the formula.
- 244–250, `robustness_trace`: the **offline evaluator** — all 768 samples at once; robustness at every sample.
- 252–258, `robustness_online`: the **online evaluator** — samples one by one, as a live stream would. The tests
  check both give the same answer.
- 260–264, `robustness`: the value at the **last** sample, which judges the whole window.

**Lines 270–287: the four class monitors.** `StraightMonitor`, `DownMonitor`, `LeftMonitor`, `RightMonitor` differ
only in which rule (label) they hold.

**Lines 294–305: `satisfied_class`, picking the answer.** If STRAIGHT's robustness is positive the answer is
STRAIGHT; else if DOWN's is at least 0, DOWN; else LEFT if LEFT's is at least 0, otherwise RIGHT. Exactly one rule
holds, so this simply finds it; the order matters only at an exact zero.

**Lines 308–347: `HeadMovementMonitor`, the object you use.**
```python
from monitoring import HeadMovementMonitor
monitor = HeadMovementMonitor()
monitor.classify(window)            # rules only: the window alone
monitor.classify(window, history)   # live: with the samples just before the window
monitor.robustness(window)          # the four robustness values
monitor.formulas                    # the four rules as text
```
- 319–320, `signals`: calls `compute_signals`.
- 332–341, `explain`: checks the window, computes the signals, asks RTAMT for all four robustness values, and picks
  the answer. A bad window gives INVALID instead of an exception.
- 343–345, `classify`: just the label.

**Lines 350–444: `StreamMonitor`, feeding it live.** Push one sample (8 numbers) at a time.
- 368–390: it keeps a buffer of the last 3 s plus 5 s of history.
- 403–432, `push`: shift the buffer and add the sample. After 768 samples, and then every `decide_every` samples
  (default 768), classify the latest 3 s, passing the earlier samples as history (416–428).

**Lines 455–485: `main`.** The demo behind `python -m monitoring`: prints the four rules and classifies one real
window of each class.

---

### 4.3 `monitoring/stream_history.py`: where "history" comes from in the dataset

The dataset stores separate 3-s windows, not continuous recordings. To test the live technique, this file rebuilds
the seconds before each window from the dataset's own overlapping windows. No other file is read.

- **Line 30:** overlapping windows must agree to 0.001 µV, or the rebuild stops with an error.
- **Lines 33–35, `default_history_seconds`:** 1 s settle + 4 s background = 5 s.
- **Lines 38–69, `rebuild_runs`:**
  - For each recording run, place every window at its position (`run_offset`) and glue the overlaps into one long
    signal.
  - Each stored window had its own average subtracted, so the code restores it from the overlap with windows
    already placed.
  - It records where the rebuilt signal has gaps (`stretch_start`).
- **Lines 72–90, `windows_with_history`:** for each window, yields the window plus up to 5 s of continuous rebuilt
  signal before it, or `None` if nothing continuous precedes it.

*Consequence:* only 25.5% of movement windows get enough rebuilt history, against 85% of rest windows. This is why
the live technique mostly helps the rest windows in this dataset.

---

### 4.4 `monitoring/offline_monitor.py`: the offline technique

**Lines 1–71: the description.** Why offline helps, the background signal, the four offline rules, the results.

**Lines 94–95.** `LAST = 767`; `OFFLINE_SIGNALS` lists the six signals the offline rules read.

**Lines 98–111: `OfflineConfig`.** `background_ratio = 5.75` (103) is the only new threshold. `live` carries the
live monitor's other thresholds unchanged.

**Lines 114–122: `load_recording`.** Reads one run's CSV file (columns Fp1…T4 in volts) and returns 8 × n
microvolts.

**Lines 125–135: `recording_energy`.** Filters the **whole recording** 0.5–3 Hz, starting the filter "already
settled" (129–130) so there is no artificial jump at the start, then the energy of every half-second per channel.

**Lines 138–159: `offline_background`, the offline loudness signal for one window.**
- 152: leave out blocks that end in the half-second before the window or inside it, so the window is compared with
  the **rest** of its recording, before and after.
- 153–154: take every 8th block; if fewer than 16 remain (a very short recording), use the whole recording.
- 156: each channel's reference = the median of those energies.
- 158: for each moment of the window, each channel's energy ÷ its reference, averaged over channels.

*Example:* a turn at second 2 of a 9-s run is compared with seconds 0–1 **and** 5–9.

**Lines 162–178: `offline_formulas`.** The four rules with `eventually` and `always`, read at t = 0. The logic is
the live rules'; only the loudness test differs.

**Lines 181–243: `OfflineHeadMovementMonitor`.**
```python
from monitoring.offline_monitor import OfflineHeadMovementMonitor, load_recording
m = OfflineHeadMovementMonitor()
rec = load_recording("dataset/eeg_cleaned_data/sub_01/ses-05/ica_eeg/sub-01_ses-05_task-right_run-06_eeg.csv")
m.classify(rec, 512)     # the window rec[:, 512:1280]
```
- 189–201, `_parsed`: parse the four rules once with RTAMT's offline evaluator.
- 203–216, `signals`: cut the window from the recording, compute the live signals with `compute_signals` (210), and
  add the offline background. `nod` and `turn` are held constant over the window (214–215), so reading them at
  t = 0 gives their end-of-window value.
- 218–221, `robustness_from_signals`: RTAMT evaluates each rule; `[0][1]` is the robustness at t = 0 (221).
- 227–237, `explain` and `classify`: the label, or INVALID.
- 239–243, `classify_recording`: many windows of one recording, computing the recording's energy only once.

**Lines 246–250: `dataset_recording_path`.** Builds the CSV path from session, task and run.

---

### 4.5 `project_code.py`: running and scoring everything

**Lines 1–46: the description.** Every command, the datasets, the splits.

**Lines 63–77: paths and constants.**
- `DATA` (69) is the dataset folder, and `RECORDINGS` holds the CSV recordings.
- There are three cache files: `measurements_v2.npz`, `monitor_labels_v2.npz` and `monitor_labels_offline_v2.npz`.
- `DESIGN_LAST` (75) = 20: the design sessions are sub_01 1–20. `FOLDS` (76) are the cross-validation blocks.
- `KEYS` (77) are the measurements.

**Lines 80–87: `monitor_thresholds` and `SHIPPED`.** Reads the thresholds from `RtamtConfig`, so every table
follows the monitor.

**Lines 90–104: `fingerprint`.** A code made from every `monitoring/*.py` file, the settings and the dataset. If
any of them changes, old saved results are ignored and recomputed, so a table can never show numbers made by
older code.

**Section 1, MEASURE (lines 108–266).**
- `_session` (110–124): for every window of one session, the one number each rule reads — the **maximum** of
  contrast, background, delta, deflect and sideways, and the **last** value of nod and turn.
- `_cached` (127–134): returns a saved file only if its fingerprint matches.
- `measure` (137–155): all sessions in 6 parallel processes → `measurements_v2.npz`.
- `_classify_session` (158–170): the **real** `HeadMovementMonitor` on every window, twice: with history (live) and
  alone (rules only).
- `monitor_labels` (173–196): all sessions, 60–90 minutes → `monitor_labels_v2.npz`. **The accuracy tables are
  counted from these genuine RTAMT verdicts.**
- `offline_fingerprint` (199–207): the fingerprint plus the offline settings and the CSV files.
- `_offline_session` (210–224): the offline monitor on every window of one session; each run's CSV is read once.
- `offline_labels` (227–244): all sessions, about 20 minutes → `monitor_labels_offline_v2.npz`.
- `evaluate_offline` (247–266): prints the offline table and a live-versus-offline comparison.

**Section 2, THE DECISION (lines 270–368).**
- `predict` (272–278): the four rules as plain arithmetic:
  ```python
  loud = background >= b  if history is valid  else  contrast >= g      # line 293
  move = loud & (delta >= d) & (deflect >= q)
  nod  = (nod >= n) & (sideways <= v)
  answer = STRAIGHT if not move, else DOWN if nod, else LEFT if turn > 0, else RIGHT
  ```
  It gives exactly RTAMT's answers on all 21,029 windows, in milliseconds instead of an hour. Threshold searches
  use it.
- `balanced` (281–283): the average of the four class accuracies.
- `report` (286–291): one line per split. `splits` (294–298): design and the two tests.
- `results_table` (301–337): the big table of correct / total.
- `evaluate` (340–368): `python project_code.py evaluate`; with `--detail` it also prints the rules-only table.

**Section 3, TRAIN (lines 372–442).**
- `search` (374–396): tries every combination of b, g, d, q on a grid (383) and keeps the best on the training
  sessions, then n and v (391). Its docstring warns not to lower the g grid below 5.0: that "gain" is an
  artifact of this dataset's missing history.
- `train` (399–411): cross-validation — choose on two blocks, score the third, three times.
- `train_first_30` (414–430): the previous default, thresholds re-selected on sessions 1–30, for comparison
  (`python project_code.py train30`).
- `test` (433–442): scores the held-out sets, only when `FINAL_TEST=1` is set, so it cannot happen by accident.

**Sections 4–5 (lines 445–584): comments only.**
- Section 4 describes how each dataset was built.
- Section 5 lists **every rejected idea with the numbers it produced**, so nothing is tried twice. Section 5j
  (540) is the latest round.

**Section 6, ONE ROW PER WINDOW (lines 585–663).**
- `class_robustness` (588–607): the four robustness values with min/max arithmetic.
- `windows` (610–663): writes `windows.csv` with truth, answer, margins and every measurement. Sort by `margin` to
  see the least certain decisions.

**Section 7, NEW SUBJECTS (lines 666–885).**
- `extract_session` (750–831): cuts one session into clean windows, in two steps: place each of the lab's windows in
  its recording and label it from the `ACT_` annotations (whole movement inside with 0.25 s to spare; rest at least
  1 s from any movement), then keep only windows that lie inside one recorded run.
- `extract` (834–855): `python project_code.py extract sub_12`, every session of one subject.
- `subjects` (858–884): `python project_code.py subjects`, one accuracy row per subject.

**Lines 887–909: `main`.** Reads the command word (`evaluate`, `measure`, `windows`, `train`, `train30`, `test`,
`extract`, `subjects`) and runs it.

---

### 4.6 `show_accuracy.py`: the report for a presentation

It only **presents** results; it never decides anything.
- `summarize` (7–18): confusion matrix, accuracy, balanced accuracy and per-class counts.
- `verify_live` (20–40): re-runs the real monitor now on one window per class per split (9 windows) and stops
  with an error if any answer differs from the cache.
- `main` (42–101): refuses to run if the cache does not match the current code, then prints the coloured tables.
  Options: `--details` (confusion matrices), `--verify-live`, `--json FILE`, `--no-color`.

---

### 4.7 `monitoring/row_stream.py`: an answer after every sample

The window monitor can only answer after all 768 samples. This file answers after **every** sample, using only
what has arrived. At sample 768 its answer equals `HeadMovementMonitor`'s exactly.

- **88–111, `row_formulas`:** the same rules without `once`/`historically`, because the "so far" signals already
  contain the running maximum.
- **114–122, `RowVerdict`:** what one answer contains — the sample number, label, the four robustness values, and
  any early decision.
- **125–134, `_Filter`:** a filter that remembers its state between samples, so filtering one sample at a time
  equals filtering the whole window.
- **146–170, `__init__`:** settings and optional history.
- **172–203, `clear` and `reset`:** `clear` forgets everything; `reset` starts a new window. `reset` builds the RTAMT
  **online** specifications and, with `commit_seconds`, early-decision rules of the form
  `historically[0:K-1](that class's rule)`.
- **216–252, background:** the reference level at the window start, and an update after every sample.
- **254–299, `prefix_signals`:** every measurement on the samples received so far.
- **301–353, `push`:** one sample in — filter, update, ask RTAMT (online), return the answer.

*Early commitment example:* with `commit_seconds=1.0`, if LEFT's rule has held for a full second of samples, the
monitor commits to LEFT about 0.45 s sooner than waiting for the full window, at about the same accuracy.

---

### 4.8 `monitoring/live_session.py`: a real headset

Wraps `StreamMonitor` for rows arriving with timestamps.
- `push(row, timestamp)` (46–86) rejects bad rows, and resets if time goes backwards or jumps (a gap), because the
  3-s history is then broken.
- It reports a status — WARMUP, WAITING, PENDING, CONFIRMED, INVALID or GAP_RESET — plus the compute time and
  whether it fit the time budget.

---

### 4.9 `monitoring/rtamt_persistence.py`: "say it twice"

An optional filter. A label is **confirmed** only after it repeats k times in a row. This is checked by the RTAMT
rule `historically[0:k-1](evidence > 0)`, where evidence is +1 for the class that was said and −1 for the others.

*Example with k = 2:* the answers LEFT, LEFT confirm LEFT; the answers LEFT, DOWN confirm nothing.

---

### 4.10 `monitoring/threshold_report.py`: explains and re-chooses thresholds

A tool, not part of the decision. `python -m monitoring.threshold_report` explains the current thresholds.

| Lines | Function | What it does |
|---|---|---|
| 135–145 | `measure_window` | The numbers the rules read for one window |
| 195–213 | `decide` | The four rules as min/max robustness arithmetic |
| 237–273 | `select_thresholds` | The **one-standard-error rule**: move a threshold only if the gain is bigger than its uncertainty across sessions, then take the value nearest the current one; repeat until nothing moves |
| 308–348 | `explain` | For each threshold, how many movement or rest windows pass it, and a sweep of nearby values |
| 296–305 | `cross_check` | Proves its arithmetic equals the real RTAMT monitor on 60 sampled windows |
| 356–433 | `main` | Command line: explain by default, `--select`, `--train` / `--test`, `--save` |

---

### 4.11 `monitoring/__init__.py` and `__main__.py`

- `__init__.py` lists what `from monitoring import …` gives you: `HeadMovementMonitor`, `StreamMonitor`,
  `RowStreamMonitor`, `RtamtConfig`, `compute_signals` and others.
- The offline monitor is imported directly: `from monitoring.offline_monitor import OfflineHeadMovementMonitor`.
- `__main__.py` makes `python -m monitoring` run the demo.

---

### 4.12 `tests/`: 179 automatic checks

Run `python -m pytest -q tests`. The tests use synthetic windows with known answers; accuracy is measured on the
dataset, not inferred from the tests.

| File | What it checks |
|---|---|
| `test_rtamt_monitors.py` | Every rule parses in RTAMT; exactly one class is true; the signals equal an independent calculation written inside the test; thresholds behave correctly exactly at their boundaries; online equals offline evaluation; multiplying a window by 0.01, 3 or 250 changes nothing; bad windows give INVALID; the stream answers on exactly the 768th sample |
| `test_stream_background.py` | The live background equals an independent calculation; with no history the monitor is exactly rules only; the same bump is a movement after a quiet stretch and rest after an equally loud one; history rebuilt from the dataset continues each window exactly |
| `test_offline_monitor.py` | The offline rules use only future operators and parse; RTAMT equals the min/max arithmetic; the offline background equals an independent calculation and is unit-free; a burst is a movement in a quiet recording and rest in a busy one; windows at the edge of a recording work |
| `test_row_stream.py` | The row-by-row monitor equals the window monitor at sample 768; it never uses a later sample; early commitment works and never commits to STRAIGHT early |
| `test_rtamt_persistence.py` | The "say it twice" confirmation equals an independent check; a time gap restarts it |
| `test_threshold_report.py` | The threshold tool agrees with RTAMT; it moves a clearly bad threshold and keeps a good one; saving and loading work |
| `test_accuracy_report.py` | Balanced accuracy is not fooled by the many STRAIGHT windows; a missing class gets no four-class score |
| `test_censored_background.py` | The optional censored background (2026-10-07, off by default): no mask or an all-true mask is exactly the original rule; a movement already judged is left out of the background; it equals an independent calculation; with censoring off both stream monitors behave exactly as before |

---

## 5. How one number in a table is produced

Take "LEFT 556/639" for the live monitor on sessions 1–20:

1. **The recordings.** `dataset/eeg_cleaned_data/…/*.csv`, one per run.
2. **The windows.** `extracted_npz_dataset_v2/<person>/<session>/clean_epochs.npz` stores 3-s windows cut from those
   runs, with their labels. There are 639 LEFT windows in sessions 1–20.
3. **The history.** `stream_history.windows_with_history` gives each window the seconds before it, if available.
4. **The signals.** `rtamt_signals.compute_signals(window, history=…)` computes every measurement.
5. **The rules.** `rtamt_monitors.HeadMovementMonitor.classify` asks RTAMT for the robustness of the four rules and
   picks the one that holds.
6. **The cache.** `project_code.monitor_labels` does steps 3–5 for all 21,029 windows, twice (with history, and
   alone), and saves the verdicts with the fingerprint in `monitor_labels_v2.npz`.
7. **The count.** `project_code.results_table` counts LEFT windows answered LEFT: 556 of 639.

The **fingerprint** ties the numbers to the code. It is a hash of every `monitoring/*.py` file, the thresholds and
the dataset. Change anything — even a comment in a monitor file — and the saved verdicts are no longer trusted:
`show_accuracy.py` refuses, and `python project_code.py evaluate` recomputes them.

The offline monitor follows the same path, with `offline_monitor.py` in steps 4–5 and the whole CSV recording in
place of the history. Its verdicts are saved in `monitor_labels_offline_v2.npz`.

---

## 6. Where each technique lives

| Technique | The "loud" test is in | Switched on by | Scored by |
|---|---|---|---|
| **Rules only** | `rtamt_signals.py` 365–372 (contrast), used in `rtamt_monitors.py` 183 and the `or` branch of 190–191 | `classify(window)` with no history | `project_code.py` `_classify_session`, stored as `alone` |
| **Live, with history** | `rtamt_signals.py` 303–344 (`stream_background`), used in `rtamt_monitors.py` 187–193 | `classify(window, history)`; history comes from `stream_history.py` (dataset) or `StreamMonitor` (live) | `project_code.py` `_classify_session`, stored as `with_history` |
| **Offline** | `offline_monitor.py` 128–169, used in 181–183 | `OfflineHeadMovementMonitor().classify(recording, start)` | `project_code.py` `_offline_session` (`evaluate --offline`) |
| **Choosing thresholds** | `project_code.py` `search` / `train`; `threshold_report.py` `select_thresholds` | `python project_code.py train` | blocked cross-validation on sessions 1–20 |

The rest of every rule — delta, deflect, nod, sideways, turn — is **the same for all three techniques**:
`rtamt_signals.py` lines 386–485.

---

## 7. Try it yourself

Print the measurements of one real window and the monitor's answer:

```python
import numpy as np
from monitoring import HeadMovementMonitor
d = np.load("extracted_npz_dataset_v2/sub_01/ses-05/clean_epochs.npz")
window = d["epochs"][10, :8] * 1e6          # stored in volts; the monitor wants microvolts
m = HeadMovementMonitor()
s = m.signals(window)
print(d["labels"][10], m.classify(window))                                  # DOWN DOWN
print("contrast", s["contrast"].max(), "delta", s["delta"].max(), "deflect", s["deflect"].max())
print("nod", s["nod"][-1], "sideways", s["sideways"].max(), "turn", s["turn"][-1])
print(m.robustness(window))                 # the four robustness values; only DOWN's is >= 0
```

Output for that window: contrast 53.6, delta 0.40, deflect 3.80, nod +1.40, sideways 3.1, turn −2.2, which is a
correct DOWN.

See the rules, and run the same window through the offline monitor:

```python
print(m.formulas["LEFT"])
from monitoring.offline_monitor import OfflineHeadMovementMonitor, load_recording, dataset_recording_path
rec = load_recording(dataset_recording_path("dataset/eeg_cleaned_data", "sub_01/ses-05",
                                            str(d["task"][10]), int(d["run"][10])))
print(OfflineHeadMovementMonitor().classify(rec, int(d["run_offset"][10])))
```

---

## 8. The folders

| Folder or file | What it holds |
|---|---|
| `monitoring/` | The monitors (the deliverable); section 4 |
| `project_code.py`, `show_accuracy.py` | Running, scoring and presenting |
| `tests/` | 179 automatic checks |
| `dataset/eeg_cleaned_data/` | The original recordings: one CSV per run (ICA-cleaned) |
| `extracted_npz_dataset_v2/` | The 21,029 windows used everywhere |
| `monitor_labels_v2.npz`, `monitor_labels_offline_v2.npz` | Saved RTAMT verdicts (fingerprint-checked) |
| `measurements_v2.npz` | Saved measurements for fast arithmetic |
| `experiments/` | One folder per round of work: what was tried, logs, figures |
| `reports/` | Saved accuracy reports |
| `_archive/` | Everything retired: old rounds, the TeLEx era, original extraction scripts |
| `.runtime/` | The bundled RTAMT library; keep it |

---

## 9. Glossary

| Word | Meaning |
|---|---|
| EEG | Recording small voltages on the head |
| Electrode / channel | One sensor on the cap; eight here |
| Sample | One reading of all 8 electrodes; 256 per second |
| Window | 3 seconds = 8 × 768 samples |
| Run / recording | One short CSV recording with one movement |
| Session | One recording day |
| Filter / band-pass | Keeps wiggles of a chosen speed range |
| Energy | The average of squared values: how strong a wiggle is |
| Spread | The size of the normal wiggle (robust) |
| Signal (in this code) | A measurement with one value per sample of the window |
| STL | Signal Temporal Logic, the language of the rules |
| RTAMT | The library that checks STL rules |
| Robustness | How clearly a rule holds: positive means it holds, and the size is the margin |
| Threshold | A number inside a rule |
| Training / design | Choosing thresholds, on sub_01 sessions 1–20 |
| Cross-validation | Choosing on some sessions and scoring on others |
| Test / held-out | Sessions never used to choose anything |
| Balanced accuracy | The average of the four class accuracies (guessing = 25%) |
| False alarm | A still window called a movement |
| History | The samples just before a window (live technique) |
| Offline | Using the whole finished recording, including samples after the window |
| Fingerprint | A hash that ties saved results to the exact code, settings and dataset |
| ICA | A cleaning method applied to the data before this project, mainly to remove eye blinks |
