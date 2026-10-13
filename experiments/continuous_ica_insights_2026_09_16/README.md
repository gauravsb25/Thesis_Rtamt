# What the continuous ICA recordings show (sub_01 sessions 1–30 only)

Source: `dataset/eeg_cleaned_data/sub_01/ses-XX/ica_eeg/augmented_task-*_raw.fif`, with the `ACT_*`
annotations. Used here **only to understand the signals and check labels**. The monitors still work on
3-second windows only. Sessions 31–36 and sub_11 were not opened.

## Facts about the folder

- `task-*_raw.fif` is the per-run CSVs joined together, identical to 1e-12 V, in a different run
  order.
- `augmented_*` is the same recording with `ACT_left/right/down` movement annotations added.
- Each run contains one movement. The `move_head_now` marker is often up to 1.4 s *after* the
  annotated movement onset, so the `ACT_*` onsets are the reliable timing.
- Your `visuals/ica_eeg/*_diff_eeg.png` plots show the recording before cleaning (red: large DC
  offsets and drifts) against the cleaned signal (green). Blink-like frontal dips are strongly
  reduced by the cleaning.

## 1. Movement signatures — `movement_signatures_sessions_1-30.png`

Averages locked to the movement onset: 85 LEFT, 81 RIGHT, 66 DOWN, 90 rest snippets. Thin lines are
single-session means.

- **Turns are three-phase mirror images.** On T3−T4, LEFT goes + (0.2 s) − + (1.1 s), about ±20 µV.
  RIGHT is the mirror, − + −. F4−T3 shows the same, inverted. The strongest phase keeps its sign in
  29–30 of 30 sessions for RIGHT and 24–26 of 30 for LEFT.
- **The shape matches head acceleration** during a turn-and-return: accelerate out, reverse, stop.
  With the eye components cleaned out, the lateral signal behaves like a motion signature. Its size
  and polarity depend on how the headset sits, which fits the drift seen between sessions.
- **Nods have no consistent time-locked waveform.** Only 14–22 of 29 sessions agree on sign. Energy
  (not sign) is the right nod evidence, and nods are the weakest events.

## 2. A rule built from that shape — `doublet.py`, `doublet_windows.py`

"Two same-sign peaks 0.7–1.1 s apart" is expressible in past-time STL as
`once((x >= h) and once[a:b](x >= h))`.

| LEFT/RIGHT accuracy | largest peak (current rule) | same-sign doublet |
|---|---|---|
| continuous recordings, exact timing (F4−T3) | 87.3% | **91.6%** |
| continuous recordings, exact timing (T3−T4) | 88.6% | **94.0%** |
| your 3-s windows, monitor's causal filter | **90.4%** | 82.4% at best |
| your 3-s windows, zero-phase filter in the window | 86.8% | 90.0% |

The advantage exists in continuous data but is lost once the data is cut into independent 3-second
windows and each window is filtered alone. **Not worth changing the monitor for.** An analytic
cosine acceleration template (`accel_template.py`) was also worse.

## 3. Label check — `annotation_energy_sessions_1-30.csv`, `annotation_check_examples.png`

For each annotation: the strongest movement energy within ±4 s, and how strong the energy is inside
the annotated interval compared with the recording's median.

- 94% of annotations have their energy peak within 0.75 s of the annotation centre.
- **22 of 251 are flagged:** 16 with stronger energy outside the annotated interval, plus 6 that are
  weak (under 5× the median). DOWN is weak more often (10% vs 2–3% for turns).
- **13 of the monitor's 48 error movements (27%) are flagged,** against 4% of correctly read ones.
  That includes 7 of the 10 "missed nods" and 6 "turns called DOWN or STRAIGHT".
- In the example figure, ses-23 RIGHT (19.3 s) and ses-14 LEFT (23.5 s) show no visible movement
  inside the annotation at all.

So part of the measured error is label quality, not the monitor. Excluding flagged annotations from
an evaluation must follow a rule fixed *before* looking at the monitor's output (like this energy
rule), and must be reported.

## 4. Follow-up: could any of this update the monitors? (design sessions 1–30; nothing adopted)

The bar was a gain that holds per movement and per session, not only per window. A direction rule
that looked good only on windows (SINCE: 4 movements fixed, 1 broken) had already failed on held-out
data.

| idea | result on sessions 1–30 | verdict |
|---|---|---|
| **Session-adaptive DOWN threshold** (Otsu / Kittler–Illingworth on the session's own nod scores, no labels; causal in true recording order from `run_order.py`) | static 82.30; the oracle per-session threshold (uses labels) 85.81; unsupervised at best 82.6–83.6, with neighbouring settings 72–82 | rejected: only 2–3 nods per session, too few to place a threshold without labels (`adapt1.log`, `adapt2.log`) |
| **Turn-signature strength** (sign-free doublet) as turn-vs-nod evidence | AUC 0.859 vs sideways 0.881 and nod 0.962 | rejected (`turn_signature.log`) |
| nod score + turn-signature strength | AUC 0.962 → at most 0.971; "nod − log(sideways)" from existing signals already 0.969 and gave +0.13 in the full monitor | rejected |
| detrend + zero-phase low-pass for direction (replacing the 0.3 Hz high-pass) | doublet 90.5% windows / 91.3% movements vs peak 90.4% / 89.7%; 15 movements fixed, 12 broken | rejected: a wash |
| peak + doublet, resolving disagreements by the more decisive rule | 92.2% windows / 91.8% movements; 6 movements fixed, 2 broken; sessions 8 better / 2 worse | rejected: evidence too thin (the same regime as SINCE) |

Useful by-product for a safety-critical use: the peak and doublet rules **agree on 85.4%** of turn
windows and are then **97.4% correct**; when they disagree each is right about half the time.
A monitor that answers only when both agree would be much safer for direction, at the cost of
abstaining on about 15% of turns.

`run_order.py` maps every window to its recording run (exact sample matching of the per-run CSVs
inside the task fifs), so a session can be replayed in true time order.

## Running the scripts

`doublet_windows.py` imports `common.py` and `ship.py` from `../improvement_round_2026_09_16`, so
copy them next to it or add that folder to `PYTHONPATH`. `signatures.py` writes `signatures_dev.npz`
next to itself; the plotting scripts read it.
