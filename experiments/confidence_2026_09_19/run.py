"""How certain are the headline numbers? Bootstrap confidence intervals over SESSIONS.

Every accuracy in this project is an average over a handful of recording sessions, so the honest question is
how much it would move with different sessions from the same person. Sessions are resampled with
replacement (2000 times) inside each split, and the 2.5% and 97.5% percentiles are reported. Windows are
never resampled on their own: they overlap by 0.1 s and are not independent.

Reported for each split: the four-class balanced accuracy, each class on its own, and the three decisions
(movement vs still, nod vs turn, left vs right).

usage: python run.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\gaura\thesis_rtamt")
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
CL = ["LEFT", "RIGHT", "DOWN", "STRAIGHT"]
P = dict(b=6.61, g=7.81, d=0.208, q=2.72, n=-0.661, v=7.54)
DRAWS = 2000

M = dict(np.load(ROOT / "measurements_v2.npz", allow_pickle=True))
LAB, SES = M["label"].astype(str), M["session"].astype(str)
VALID = M["background_valid"] >= 0.5
LOUD = np.where(VALID, M["background"] >= P["b"], M["contrast"] >= P["g"])
MOVE = LOUD & (M["delta"] >= P["d"]) & (M["deflect"] >= P["q"])
NOD = (M["nod"] >= P["n"]) & (M["sideways"] <= P["v"])
PRED = np.where(~MOVE, "STRAIGHT", np.where(NOD, "DOWN", np.where(M["turn"] > 0, "LEFT", "RIGHT")))
DEV = np.array([s.startswith("sub_01") and int(s[-2:]) <= 30 for s in SES])
SPLITS = (("sub_01 sessions 1-30", DEV), ("sub_01 sessions 31-36", np.char.startswith(SES, "sub_01") & ~DEV),
          ("sub_11", np.char.startswith(SES, "sub_11")))


def statistics(mask):
    """The seven numbers, computed on whatever windows `mask` selects."""
    out = {}
    for k in CL:
        m = mask & (LAB == k)
        out[k] = np.mean(PRED[m] == k) if m.any() else np.nan
    out["balanced"] = np.nanmean([out[k] for k in CL])
    rest, move = mask & (LAB == "STRAIGHT"), mask & (LAB != "STRAIGHT")
    out["movement vs still"] = 0.5 * (np.mean(PRED[rest] == "STRAIGHT") + np.mean(PRED[move] != "STRAIGHT")) \
        if rest.any() and move.any() else np.nan
    both = mask & MOVE & (LAB != "STRAIGHT")
    out["nod vs turn"] = np.mean((PRED[both] == "DOWN") == (LAB[both] == "DOWN")) if both.any() else np.nan
    turn = mask & np.isin(LAB, ["LEFT", "RIGHT"]) & MOVE & ~NOD
    out["left vs right"] = np.mean(PRED[turn] == LAB[turn]) if turn.any() else np.nan
    return out


def main():
    rng = np.random.default_rng(0)
    order = ["balanced"] + CL + ["movement vs still", "nod vs turn", "left vs right"]
    for name, mask in SPLITS:
        sessions = np.unique(SES[mask])
        point = statistics(mask)
        draws = {k: [] for k in order}
        for _ in range(DRAWS):
            chosen = rng.choice(sessions, size=len(sessions), replace=True)
            resampled = np.zeros(len(LAB), bool)
            counts = {}
            for s in chosen:                      # a session drawn twice counts twice
                counts[s] = counts.get(s, 0) + 1
            values = {k: [] for k in order}
            for s, times in counts.items():
                stats = statistics(mask & (SES == s))
                for k in order:
                    values[k].extend([stats[k]] * times)
            for k in order:
                draws[k].append(np.nanmean(values[k]))
        print(f"\n{name}   ({len(sessions)} sessions, {int(mask.sum())} windows)")
        print(f"   {'measure':22s}{'value':>9s}{'95% interval':>22s}")
        for k in order:
            lo, hi = np.nanpercentile(draws[k], [2.5, 97.5])
            print(f"   {k:22s}{100 * point[k]:8.2f}%   [{100 * lo:6.2f}, {100 * hi:6.2f}]")


if __name__ == "__main__":
    main()
