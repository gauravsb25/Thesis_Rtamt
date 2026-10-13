# Is T4 faulty in sub_11? (17 Sep 2026)

Earlier notes called sub_11's T4 electrode "faulty" because it reaches ~150 µV in LEFT windows.
The user asked whether the swing might instead be common to every trial. It is: the "faulty" label was wrong.
`t4_check.py`, log `t4_check.log`, figure `t3_t4_time_locked.png`. Descriptive check on both subjects
and all sessions; nothing was fitted and no monitor changed.

## Findings

1. **T4 is normal at rest in most runs.**
   - Rest SD is 6–9 µV, like the other channels and like sub_01.
   - It still follows its neighbour (T4–Cz correlation 0.34).
   - A broken electrode would misbehave all the time.
2. **The big swing is present in almost every LEFT turn and time-locked to it.**
   - T4 exceeds 100 µV in 26 of 27 LEFT movements (median peak 429 µV), in every session.
   - It has the same three-phase turn shape each time: out, reverse, back (+170 / −100 / +170 µV at 0.4 / 0.8 / 1.5 s).
   - Each movement correlates r = 0.55 with the average of the others.
3. **It is mirrored on the other side for RIGHT turns.**
   - T3 exceeds 100 µV in 70% of RIGHT movements (median 124 µV), with the same three-phase shape, r = 0.76.
   - The electrode on the side opposite the turn swings. T4 swings about 3× more than T3, so it probably sits looser or is more exposed.
4. **It is too large to be brain activity.**
   - sub_01's turn swings are ~45 µV, and nods in both subjects stay near 35–45 µV.
   - This is a mechanical (electrode/skin) movement artifact. It is systematic and carries direction information.
5. **The polarity pattern differs from sub_01.**
   - In sub_01, T3 and T4 go in opposite directions (~±10 µV).
   - In sub_11, both go positive first and the side opposite the turn dominates, so T3−T4 has the opposite sign in the two subjects.
   - This, not a broken electrode, is why T3−T4 direction rules don't transfer.
6. **One real contact/tension problem:** only in the LEFT-task runs of sessions 04, 05 and 06.
   - Even before the movement, T4 has an SD of 30–63 µV and 20–45 Hz power 8–11 µV (others ~1).
   - In other runs of the same sessions T4 is normal.
   - So it is run-specific (possibly posture or muscle tension while waiting to turn left), not a permanent fault.

Docs that said "faulty T4" were corrected: `monitoring/README.md`, `monitoring/rtamt_signals.py` docstring,
`docs/pure_rtamt_monitors.tex`, and the artifact-check and npz-audit READMEs.
