# Row-by-row monitoring (17 Sep 2026)

The monitors now also run on EEG arriving row by row (`monitoring/row_stream.py`, `RowStreamMonitor`).

- Every signal is a prefix statistic: the monitor's measurement on rows 0..t only. At row 768 all
  prefix values equal the window monitor's measurements exactly (relative difference 0 on all 18,560
  windows of sessions 1-30), so decisions there are identical.
- `prefix_eval.py` computes the prefix signals every 1/16 s. In the first version the direction read
  needed a full 1.5 s block; the second (`_v2`, the one implemented) reads the loudest stretch
  available so far, which changes only early verdicts.
- The module was checked against the analysis on 40 windows: anytime labels agreed at 1640/1640
  checkpoints and committed labels at 40/40.
- RTAMT's online monitor evaluates the four class specifications at every row; about 1-2.5 ms per row.
- Early commitment = `historically[0:K-1](class specification)`.

| decision | sessions 1-30 | sub_01 31-36 | sub_11 |
|---|---|---|---|
| row 768 (default) | 82.74% · 2.22 s | 67.22% · 2.20 s | 77.00% · 2.31 s |
| commit after 1.0 s | 81.97% · 1.83 s | 67.75% · 1.87 s | 77.83% · 1.80 s |
| commit after 0.75 s | 79.79% · 1.62 s | 67.73% · 1.74 s | 77.00% · 1.55 s |
| commit after 0.5 s | 73.73% · 1.31 s | 65.87% · 1.45 s | 75.94% · 1.27 s |

Times are the median delay from movement onset to the decision, for correctly classified movements.
The held-out sessions had been used for earlier decisions; here they could not change any default,
because the default verdict is identical by construction. Logs: `commit_dev.log` (first version),
`commit_dev_v2.log`, `commit_test.log`.
