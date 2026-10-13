# Live replay of the real monitor (2026-10-07)

Every recorded run streamed sample by sample through `monitoring.StreamMonitor` (real RTAMT, current default
thresholds selected on sub_01 1–20, censoring off), a verdict every 0.25 s, on the ICA-cleaned recordings and on
the raw pre-ICA recordings (`D:/errp/eeg_cleaned_data/*/ses-*/pp_eeg`, filtered only), which is what a headset
delivers. Scored per movement (the command the system would act on) and per minute of rest.

**Main result.** Detection works live (1–7% of movements missed), and waiting for agreement fixes much of the
wrong-command problem. But on RAW data the monitor issues 4–11 false commands per minute, because the blinks
that ICA removed offline are back. The offline ICA cannot be reproduced from these 8 channels.

## Act on the first verdict vs wait for k identical verdicts (k × 0.25 s)

| rule | data | split | right | wrong | missed | delay | false commands / min |
|---|---|---|---|---|---|---|---|
| first verdict | cleaned | sub_01 21–36 | 72.2% | 26.2% | 1.6% | 1.15 s | 5.05 |
| first verdict | cleaned | sub_11 | 55.8% | 42.1% | 2.1% | 0.62 s | 16.9 |
| 4 in a row (1 s) | cleaned | sub_01 21–36 | 78.6% | 14.3% | 7.1% | 2.02 s | 0.82 |
| 4 in a row (1 s) | cleaned | sub_11 | 73.7% | 24.2% | 2.1% | 1.64 s | 2.82 |
| 4 in a row (1 s) | **raw** | sub_01 21–36 | 78.6% | 14.3% | 7.1% | 1.98 s | **4.33** |
| 4 in a row (1 s) | **raw** | sub_11 | 71.6% | 28.4% | 0.0% | 1.64 s | **9.41** |
| 6 in a row (1.5 s) | raw | sub_01 21–36 | 75.4% | 14.3% | 10.3% | 2.46 s | 3.81 |

All k: `replay_policies.txt`. First-verdict-only run: `replay_log_first_verdict.txt`. The window-dataset scores
hid both effects because they score only windows that contain a whole movement, on cleaned data. Rest time is
short (sub_11: about 1 minute), so the sub_11 false-command rates are rough.

## Can the cleaning be done live?

A least-squares map from the 8 raw channels to the 8 cleaned channels, fitted on half the runs of a session,
leaves 25–90% of each cleaned channel unexplained (Fp1: correlation 0.41 between raw and cleaned). There is no
time shift (best lag 0), and a lagged (FIR) fit barely helps. The offline cleaning used information these 8
electrodes do not contain, so no fixed filter on them reproduces it.

## What follows

- The "wait for k identical verdicts" rule already exists: `LiveStreamSession(confirmations=4, decide_every=64)`.
- Live use needs its own artifact handling (more electrodes or EOG for online ICA/ASR, or a blink veto designed
  and tested on the RAW data); the monitors were only ever designed and scored on cleaned data.

```
python live_replay.py      # ~40 min the first time (saves replay_verdicts.npy), then prints every k instantly
```
