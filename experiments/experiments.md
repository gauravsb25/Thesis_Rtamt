# All experiments, in simple words

For every folder in `experiments/`: **the idea** (what was tried, and why), **what happened**, and whether it was
kept. Each folder has its own `README.md` (or `REPORT.md`) with all the details and numbers.

**How to read this**

- **✅ Kept** = it is now part of the project. **❌ Not kept** = tested, did not help, nothing changed.
  **🔍 Check** = a test of the data or the code, not an attempt to raise accuracy.
- Almost every experiment ran **before 2026-10-07**, when the thresholds were chosen on sub_01 sessions 1–30 and
  the test sets were sub_01 31–36 and sub_11. Its numbers are on that old split. The live monitor then scored
  **85.01 / 69.62 / 78.67** (design / sub_01 test / sub_11), so most results below are compared with that.
- **The fair-test rule** used everywhere: an idea is chosen only on the design sessions; it must win under
  cross-validation there (at least +0.5 points, and more sessions better than worse); only then is it scored once
  on the test sets, and it is kept only if it does not lose there.
- **Why so many ideas failed:** most helped sub_01 but broke sub_11. With only two people, a rule that fits one
  person's head often does not fit the other's.

---

## 1. Checks on the data

#### `npz_audit_2026_09_17` 🔍
**Idea:** Before trusting any accuracy, check that the 3-second windows are really what they claim to be: right
shape, right units, right labels, and cut from one continuous recording.
**What happened:** The windows and labels are correct, but **2,135 windows (9.2%) were glued together from two
different recordings**, with an artificial jump at the join that looks like a movement. They were removed, which
made `extracted_npz_dataset_v2` (21,029 windows), the dataset used today.

#### `label_content_check_2026_09_17` 🔍
**Idea:** When the monitor calls a still window DOWN, maybe the label is wrong and there really was a nod. So look
inside DOWN windows (is there a real nod?) and inside STRAIGHT windows (is there hidden movement?).
**What happened:** The labels hold up. DOWN windows contain real nods. Most false DOWNs are small leftover blink
bumps; only about 6 nod-sized bumps might be missing a label.

#### `artifact_check_2026_09_17` 🔍
**Idea:** Maybe the false alarms come from eye blinks, muscle twitches, electrode pops or body movement. Detect
those artifacts, then see whether removing or vetoing them helps.
**What happened:** Such artifacts are almost absent in still windows, because the data is ICA-cleaned. The false
alarms are small blink-shaped bumps that look like nods. Removing or vetoing them did not help.

#### `t4_check_2026_09_17` 🔍
**Idea:** sub_11's T4 electrode swings to about 150 µV during LEFT turns, and earlier notes called it "broken". Check
whether it misbehaves all the time (broken) or only during turns (real).
**What happened:** **Not broken.** T4 is normal at rest; the big swing comes in almost every LEFT turn, and T3
mirrors it in RIGHT turns. It is how this person's head and electrodes move.

#### `continuous_ica_insights_2026_09_16` 🔍
**Idea:** Look at the full continuous recordings (not 3-s windows) to see what a turn and a nod really look like
over time, and check whether the movement annotations sit where the movement is.
**What happened:** A turn has a three-part mirror-image shape (accelerate out, reverse, stop); nods have no fixed
shape. 27% of the hard movements have almost empty annotations. Ideas built from this (a "doublet" direction rule,
a per-session nod threshold) were tested and not kept.

#### `v2_vs_full_check_2026_09_19` 🔍
**Idea:** A few numbers were higher on the full 39k corpus than on the cleaned v2 dataset, which should not happen
if v2 only removed bad windows. Pair every v2 window with the same samples in the full corpus and find out why.
**What happened:** The data is identical. The difference came from v2's rebuilt "seconds before each window" (the
history) having gaps, not from the windows themselves.

#### `labelled_epochs_ds_monitor_2026_09_19` 🔍
**Idea:** Run the unchanged monitor on all 39,269 original windows, to see how it does on the windows v2 threw away
(movements cut off at the window edge, rest right next to a movement).
**What happened:** 80.56% balanced on the windows v2 kept, but only **68.93%** on the 18,240 it removed. The cleaned
dataset's accuracy is the easier case.

#### `volts_units_2026_09_23` 🔍
**Idea:** Every signal is a ratio, so giving the monitor volts instead of microvolts should change nothing. Test
whether that is really true in the code.
**What happened:** 195 of 21,029 answers change in volts, because tiny safety numbers in the code (like 1e-12) are
no longer tiny at that scale. **Always give the monitor microvolts.**

---

## 2. Other ways to cut the windows

#### `csv_extraction_2026_09_18` ❌
**Idea:** Cut the 3-s windows again from scratch, from the original CSV files only, with a different movement
detector, to see whether the results depend on how the windows were cut.
**What happened:** The unchanged monitor scored about the same (85.22 / 64.94 / 72.25), so they don't. Re-choosing
the thresholds on this new set failed. The dataset was deleted afterwards.

#### `trial_windows_2026_09_18` ❌
**Idea:** Instead of many overlapping windows, take exactly **one** 3-s window per trial (442 in total), finding the
movement from the signal alone (no markers, no annotation files).
**What happened:** A harder test: 78.78 / 54.19 / 68.10. Thresholds trained on it raised false alarms badly on the
normal dataset. The dataset was deleted afterwards.

---

## 3. Changes that were kept ✅

#### `five_ideas_2026_09_17` ✅ (idea 1 only)
**Idea:** Five different ideas:
1. **Compare the window with the seconds before it** (like a radar compares an echo with the recent background),
   instead of only with its own calmest moment.
2. Use the known **positions of the electrodes on the head** to tell a turn from a nod.
3. Use the **uncleaned (pre-ICA) data** to read eye movements as a direction clue.
4. Answer **UNSURE** when the decision is very close to a threshold.
5. Only accept an answer when **several consecutive windows agree**.

**What happened:** **Idea 1 was adopted**: the stream-background gate. False alarms fell from about 1 in 10 still
windows to 1 in 40, and accuracy rose from 82.74 / 67.22 / 77.00 to **85.01 / 69.62 / 78.67**. Ideas 2–5 failed:
the head geometry broke sub_11, the uncleaned data was full of blinks, and the other two gave no gain.

#### `row_stream_2026_09_17` ✅
**Idea:** A headset sends one row of data at a time. Make the monitor give an answer after **every row**, using only
the rows received so far, instead of waiting for the 3-s window to finish.
**What happened:** Built as `RowStreamMonitor`. At the end of the window its answer is exactly the normal
monitor's, and it can decide about 0.4 s earlier.

#### `commit_latency_2026_09_19` ✅
**Idea:** With the row-by-row monitor, "commit" to LEFT/RIGHT/DOWN as soon as that answer has stayed the same for
k seconds. Measure how much speed this buys and how much accuracy it costs.
**What happened:** Committing after 1 s decides about 0.45 s sooner at about the same accuracy (83.92 / 69.93 /
79.57).

#### `offline_monitor_2026_09_25` ✅
**Idea:** For **finished** recordings, compare each window with the **whole rest of its recording** (before and
after it), not only with the few seconds before it.
**What happened:** 86.73 / 71.80 / 79.37, better on all three, and **no false alarms on sub_11**. Kept as a separate
offline monitor; it cannot run live because it needs the part of the recording after the window.

#### `confidence_2026_09_19` 🔍
**Idea:** With few sessions, an accuracy number could easily be a few points off. Resample the sessions 2,000 times
to see how much each number would move with different sessions from the same person.
**What happened:** 85.01 [81.9–89.2], 69.62 [66.5–80.2], 78.67 [66.6–83.2]. Differences under about 1 point are
noise.

---

## 4. Improvements that were tried and not kept ❌

#### `improvement_round_2026_09_16` ❌
**Idea:** Two smarter STL rules:
- a **CUSUM gate**: a movement must make the energy rise **and stay up**, so a brief blink cannot pass;
- a **SINCE direction rule**: when a turn has two equal-sized swings, let the **last** big swing decide the side,
  not the biggest one.

**What happened:** Better on the design sessions (82.30 → 83.50) but worse on both test sets. Kept only as
switched-off options.

#### `error_attribution_2026_09_17` ❌
**Idea:** Trace every wrong window to the exact rule and electrodes that caused the mistake. Then look for clues
pointing to the right answer, and turn each clue into a new rule.
**What happened:** All 2,554 wrong windows traced. One rule passed (an extra "no sideways swing on Fp1−Fp2 or
Fp2−T3" check before calling a nod), but it broke sub_11 (77.00 → 74.72). Switched off.

#### `more_rules_2026_09_17` ❌
**Idea:** Each class uses only a few rules. Use **many**: 27 rules (contrast, delta and deflect for the whole window
**and for each of the 8 electrodes**), and call it a movement when at least k of them fire (voting).
**What happened:** Better in cross-validation (+2.19) but worse on the tests (64.47 / 68.29). The per-electrode
thresholds fit one person only.

#### `normalized_channel_rules_2026_09_17` ❌
**Idea:** Add each electrode as its own rule, but divide each one by **its own normal size**, so a person with
bigger or smaller signals (like sub_11's large T4) cannot fool it.
**What happened:** The scaling itself works (rest windows look alike for both people), but no electrode rule added
accuracy.

#### `round2_background_2026_09_17` ❌
**Idea:** Two follow-ups after the stream-background gate: (1) also judge **nod versus turn** against each
electrode's own background level; (2) **re-choose the other thresholds**, now that the gate had changed.
**What happened:** Neither passed cross-validation: the old thresholds were still the best. (The row-by-row monitor
was updated to use the stream background in this round.)

#### `fifteen_methods_2026_09_18` ❌
**Idea:** Fifteen small new signals, in five groups:
- **adapting to each person** (normalise by the history);
- **direction without the sign of F4−T3** (left/right energy balance, correlation, energy centre);
- **nod versus turn** (how energy is shared across electrodes, front-to-back spread, coherence);
- **artifacts** (how flat or curved the signal is);
- **STL structure** (the movement must persist for a time).

**What happened:** All 15 failed cross-validation (best +0.27). For direction, the two people behave like mirror
images.

#### `five_paradigms_2026_09_18` ❌
**Idea:** Four completely different ways of looking at the signal, plus a reframing:
1. **Surface Laplacian**: each electrode minus its neighbours, to sharpen where activity is;
2. **Permutation entropy**: a movement makes the signal more orderly than random noise;
3. **Sequential evidence**: add up the RTAMT margin over time before deciding;
4. **Hilbert phase**: read the timing (phase) of the waves instead of their size;
5. reframe the system around what it does best: detecting *that* the head moved.

**What happened:** All four testable ones gave +0.00. Detecting **that** the head moved works for both people
(95.7 / 91.8 / 94.2%); telling the **direction** does not transfer.

#### `proposed_methods_2026_09_18` ❌
**Idea:** Four methods aimed at the three known weaknesses:
1. **Direction from several electrode pairs** (majority vote, or abstain when they disagree);
2. **Yaw veto**: if the left-right energy is large compared with the midline, it cannot be a nod;
3. **Blink veto**: if the forehead electrodes dominate, it is a blink, not a nod;
4. **Midline nods**: read the nod only from Fz and Cz.

**What happened:** Only the yaw veto passed cross-validation (+1.38), and then it broke both test sets (sub_11 DOWN
88.4 → 73.0%). All four rejected.

#### `phase_shape_2026_09_18` ❌
**Idea:** Three questions: (1) if each session were **calibrated** with one known turn, how much would it help?
(2) Can **timing** (which electrode reacts first) replace the sign of F4−T3 for direction? (3) Can the **shape** of
the wave tell a nod from a turn?
**What happened:** Even a *perfect* per-session calibration adds only +1.8 on sub_11, and a realistic one-turn
calibration makes it worse. Timing and shape rules are at chance.

#### `nested_stl_2026_09_19` ❌
**Idea:** Use STL to describe the **order** of events, not just their size: "a quiet stretch, then a burst",
"two opposite swings one after the other", "three alternating swings", "midline activity that lasts". 145 such
rules in five families.
**What happened:** The best family ("two opposite swings") gained +0.49, just under the +0.5 bar, so it did not
pass, and the test sets were not even scored.

#### `stl_synthesis_90_2026_09_19` ❌
**Idea:** A large automatic search for STL rules reaching 90%: 703 measurements (280 of them new, for all 28
electrode pairs) combined with AND/OR into rules for each decision.
**What happened:** Best 87.26% on unseen design sessions (91.21% on the sessions it was fitted on). 90% was not
reached anywhere that counts.

#### `onset_rules_2026_09_23` ❌
**Idea:** An older script (`classify.py`) used fixed rules and **onset timing** (when each electrode first "goes
high"). Test both against the current monitor.
**What happened:** Most old rules use fixed microvolt limits, so they cannot work for another person. Onset timing
is real, but the nod signal already does better; it adds +0.00.

#### `microvolt_thresholds_2026_09_23` ❌
**Idea:** The monitor uses ratios (each signal divided by something from the same recording). What if the signals
were **plain microvolts**, with thresholds fitted directly on them?
**What happened:** Similar on sub_01 (+1.1), but **sub_11 fell 78.67 → 67.59 (−11.1)**. This is the evidence for
why the monitor uses ratios: they work for a person whose signals are bigger or smaller.

#### `improvement_round_2026_09_24` ❌
**Idea:** One more broad search for accuracy, plus a study of **why the remaining errors resist** every fix.
**What happened:** Nothing passed. The errors are all-or-nothing per movement (a movement is either right in all
its windows or wrong in all of them), and a lower contrast threshold only looks good because of a flaw in how the
history is rebuilt.

#### `operator_audit_2026_09_21` 🔍
**Idea:** Check that RTAMT's operators (`once`, `historically`, `since`, …) give the same results offline, online,
and after converting future-time rules to past-time rules, before trusting any rule built from them.
**What happened:** A compatibility check of 40 formulas: the operators behave as expected. A related sweep of all
operators over all electrodes and pairs found none that adds +0.5.

#### `telex2019_audit_2026_09_21` ❌
**Idea:** Instead of writing the rules by hand, **learn** them automatically from the data with the TeLEx method
(the method this repository started from).
**What happened:** The learning engine works, but its learned rules lose badly to the hand-written monitors (45–55%
against 69.62% on the sub_01 test).

#### `group_rules_2026_10_07` ❌
**Idea:** Use **groups of electrodes** instead of single ones (e.g. left side vs right side, front vs back, midline
vs temples): all 255 groups and 3,025 group pairs, combined with AND / OR / voting.
**What happened:** Up to 84.01 in cross-validation, but sub_11 fell as low as 49.11.

#### `rtamt_accuracy_2026_09_30` ❌
**Idea:** Nine more methods, among them: **shapelets** (short typical pieces of a movement wave, searched for in the
window), **wavelet scattering** (energy at many time scales), **waveform templates** (compare with an average
movement), and **longer context** (more history or the whole recording).
**What happened:** None passed. The nine experiments were moved to `_archive/rejected_2026_09_30/`.

---

## 5. Machine learning, used only as a comparison

The monitors themselves never use machine learning. These rounds asked: **could a learned model do better on the
same data?** If a model had won clearly, the hand-written rules would have something to learn from.

#### `ml_ceiling_2026_09_19` ❌
**Idea:** Give three models (logistic regression, random forest, gradient boosting) **all 87 signals** ever
computed in the project, to estimate how much information the windows hold at all.
**What happened:** A random forest reached 87.58% within sub_01, above the monitors' 85.01%, but **no model beat the
monitors on both test sets**. The models learn one person's head.

#### `ml_raw_2026_09_19` 🔍
**Idea:** Save features computed directly from the raw signal, for the machine-learning comparison above.
**What happened:** Data only (`features_raw.npz`); no separate result.

#### `riemann_rest_2026_09_19` ❌
**Idea:** Three methods from the brain-computer-interface field:
1. **Adapt to each session's rest**: judge movement against how this session looks when still;
2. **Riemannian geometry**: describe each window by how its electrodes vary together (covariance);
3. **Cascade**: the STL monitor detects a movement, then a model decides its direction.

**What happened:** Rest adaptation +0.25, Riemannian far below the monitors (38.3%), and the cascade only ties the
rule.

#### `remaining_methods_2026_09_21` ❌
**Idea:** Six learned replacements for the monitor's decision (linear and non-linear models on the monitor's own
signals and on RTAMT outputs, one that ignores the sign for direction, and a hybrid with the STL gate).
**What happened:** Several looked better on the design sessions, but none improved both test sets; on sub_11 most
fell to 52–65%. Keep the original monitor.

#### `direction_fusion_2026_09_21` ❌
**Idea:** Keep the monitor, but add **specialist models**: one that only decides nod versus turn, one that only
decides left versus right, then combine their answers with the monitor's.
**What happened:** Five combinations were tested; the best-looking one (88.33% inside sub_01) fell to 58.14% on
sub_11.
None could replace the monitor.

#### `error_correction_2026_09_23` ❌
**Idea:** Let a model **correct** the monitor: it either keeps the monitor's answer or replaces it, aimed at missed
movements and nods confused with turns.
**What happened:** sub_01 improved (69.62 → 71.50) but **sub_11 fell 78.67 → 67.69**. Not kept.

#### `spectral_patterns_2026_09_23` ❌
**Idea:** Describe each window by its **frequency content** (how much power in each band from 1 to 45 Hz, how it
changes during the window, and how electrodes share it), then classify with a simple linear model.
**What happened:** The best version failed the adoption rule on the test sets.

---

## 6. Testing it like a real headset

#### `live_replay_2026_10_07` 🔍
**Idea:** Stop judging pre-cut windows. **Stream every recording sample by sample** through the real live monitor,
exactly as a headset would deliver it, on both the cleaned EEG and the **raw** EEG (with blinks). Score each
movement (right command, wrong command, missed) and count false commands per minute of rest.
**What happened:** Movements are found (1–7% missed). But on **raw** EEG it gives **4–11 false commands per
minute**, mostly from blinks. Waiting for several matching answers before acting (about 1 s) fixes most wrong
commands on cleaned data.

#### `raw_live_2026_10_13` ✅ (passed; not yet in the monitor)
**Idea:** Make the monitor usable on raw EEG. Add a **blink veto**: if a "movement" is very short (≤ 0.3 s) and its
energy sits mostly on the forehead electrodes (Fp1, Fp2), it is a blink, so answer STRAIGHT.
**What happened:** It passed: better on both test sets (+3.7 / +1.6) and false commands down from 4.33 to **1.24
per minute** on sub_01. Not added to `monitoring/` yet.

---

**In one sentence:** out of about 40 experiments, the ideas that lasted are the **stream-background gate**, the
**row-by-row monitor**, the **offline monitor** and, for raw data, the **blink veto**. Almost everything else
improved sub_01 and broke sub_11, which only a third recorded person could settle.
