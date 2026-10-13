"""Live replay WITH the blink veto, on the raw recordings.

Re-uses the saved verdicts of the real StreamMonitor (experiments/live_replay_2026_10_07/replay_verdicts.npy,
one every 0.25 s) and turns a movement verdict into STRAIGHT when its 3-s window is forehead-dominant and short
(the veto selected on sub_01 1-20: fp_dom >= 2.0 and event width <= 0.3 s). Scored exactly as live_replay.py.
"""
import sys
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "live_replay_2026_10_07"))
import numpy as np
from measure import blink_features, load_raw
import live_replay as L

A, WID = 2.0, 0.3

def apply(row):
    ses, task, run, src, move, verdicts = row
    if src != "raw":
        return row
    x = load_raw(ses, task, run)
    out = []
    for end, lab in verdicts:
        if lab not in ("STRAIGHT", "INVALID"):
            w = x[:, end - L.W:end]
            fp, width = blink_features(w - w.mean(axis=1, keepdims=True))
            if fp >= A and width <= WID:
                lab = "STRAIGHT"
        out.append((end, lab))
    return (ses, task, run, "raw+veto", move, out)

if __name__ == "__main__":
    rows = list(np.load(HERE.parent / "live_replay_2026_10_07" / "replay_verdicts.npy", allow_pickle=True))
    raw = [tuple(r) for r in rows if r[3] == "raw"]
    with Pool(6) as pool:
        vetoed = pool.map(apply, raw)
    allrows = [tuple(r) for r in rows] + vetoed
    for k in (1, 2, 4):
        for src in ("raw", "raw+veto"):
            for split in ("sub_01 1-20", "sub_01 21-36", "sub_11"):
                rs = [r for r in allrows if r[3] == src and L.split_of(r[0]) == split]
                res = [L.score(r[5], r[4], k) for r in rs]
                ev = [x[0] for x in res if x[0]]
                n = len(ev); right = sum(e[1] == "right" for e in ev); wrong = sum(e[1] == "wrong" for e in ev)
                missed = sum(e[1] == "missed" for e in ev)
                rest = sum(x[1] for x in res); false = sum(x[2] for x in res)
                minutes = rest * L.STEP / L.FS / 60
                print(f"k={k} {src:9s} {split:13s} right {100*right/n:5.1f}%  wrong {100*wrong/n:5.1f}%  missed {100*missed/n:5.1f}%"
                      f"  false commands {false:4d} = {false/minutes:5.2f}/min")
        print()
