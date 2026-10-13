"""Fit microvolt thresholds by the project's own protocol, then compare with the shipped monitor.

Structure is identical to the shipped monitor -- only the units change:

    move   = (uv_loud >= T1) and (uv_delta >= T2) and (uv_deflect >= T3)
    nod    = (uv_nod  >= T4) and (uv_sideways <= T5)
    side   = turn > 0                       (unchanged: a sign test, immune to scaling)

The five thresholds are chosen by coordinate descent on balanced accuracy, fitted on the training
folds only. Blocked 3-fold CV gives the honest design-set number; the thresholds are then fitted on
all of sessions 1-30 and scored once on the two held-out sets.
usage: python select.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\gaura\thesis_rtamt")
HERE = Path(__file__).resolve().parent
CL = ["LEFT", "RIGHT", "DOWN", "STRAIGHT"]
FOLDS = [[f"sub_01/ses-{s:02d}" for s in range(a, a + 10)] for a in (1, 11, 21)]
KEYS = ["uv_loud", "uv_delta", "uv_deflect", "uv_nod", "uv_sideways"]

U = dict(np.load(HERE / "uv.npz", allow_pickle=True))
M = dict(np.load(ROOT / "measurements_v2.npz", allow_pickle=True))
LAB, SES = M["label"].astype(str), M["session"].astype(str)
assert (U["session"].astype(str) == SES).all()
Y = np.array([CL.index(v) for v in LAB])
TURN = M["turn"]
sess = np.array([int(s[-2:]) if s.startswith("sub_01") else -1 for s in SES])
DEV = (sess >= 1) & (sess <= 30)
T1M = np.char.startswith(SES, "sub_01") & (sess > 30)
S11 = np.char.startswith(SES, "sub_11")

# the shipped unit-free monitor, for comparison
P = dict(b=6.61, g=7.81, d=0.208, q=2.72, n=-0.661, v=7.54)
VALID = M["background_valid"] >= 0.5
SHIPPED = np.where(
    ~((np.where(VALID, M["background"] >= P["b"], M["contrast"] >= P["g"]))
      & (M["delta"] >= P["d"]) & (M["deflect"] >= P["q"])), 3,
    np.where((M["nod"] >= P["n"]) & (M["sideways"] <= P["v"]), 2,
             np.where(TURN > 0, 0, 1)))


def predict(t):
    move = (U["uv_loud"] >= t[0]) & (U["uv_delta"] >= t[1]) & (U["uv_deflect"] >= t[2])
    nod = (U["uv_nod"] >= t[3]) & (U["uv_sideways"] <= t[4])
    return np.where(~move, 3, np.where(nod, 2, np.where(TURN > 0, 0, 1)))


def bal(pred, m):
    return 100.0 * float(np.mean([(pred[m & (Y == k)] == k).mean()
                                  for k in range(4) if (m & (Y == k)).any()]))


def fit(train, rounds=4, n=25):
    grids = [np.quantile(U[k][train], np.linspace(0.02, 0.98, n)) for k in KEYS]
    t = [float(np.median(U[k][train])) for k in KEYS]
    best = bal(predict(t), train)
    for _ in range(rounds):
        improved = False
        for i in range(5):
            for v in grids[i]:
                trial = list(t)
                trial[i] = float(v)
                a = bal(predict(trial), train)
                if a > best + 1e-9:
                    best, t, improved = a, trial, True
        if not improved:
            break
    return t, best


def main():
    print("How different are the two subjects in raw microvolts?")
    print(f"   {'measurement':14s}{'sub_01 median':>16s}{'sub_11 median':>16s}{'ratio':>9s}")
    for k in KEYS:
        a, b = np.median(U[k][np.char.startswith(SES, "sub_01")]), np.median(U[k][S11])
        print(f"   {k:14s}{a:16.3f}{b:16.3f}{(b / a if a else np.nan):8.2f}x")

    print("\nblocked 3-fold CV on sub_01 sessions 1-30 (thresholds fitted inside each fold):")
    cv = SHIPPED.copy()
    for j in range(3):
        held = np.isin(SES, FOLDS[j])
        t, score = fit(DEV & ~held)
        cv[held] = predict(t)[held]
        print(f"   fold {j + 1}: " + "  ".join(f"{k}={v:.4g}" for k, v in zip(KEYS, t))
              + f"   (train {score:.2f}%)", flush=True)
    d = [bal(cv, SES == s) - bal(SHIPPED, SES == s) for s in np.unique(SES[DEV])]
    up, down = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
    print(f"\n   microvolt monitor CV {bal(cv, DEV):.2f}%   "
          f"shipped {bal(SHIPPED, DEV):.2f}%   ({bal(cv, DEV) - bal(SHIPPED, DEV):+.2f})"
          f"   sessions +{up}/-{down}")

    t, score = fit(DEV)
    print("\nthresholds fitted on ALL of sessions 1-30:")
    for k, v in zip(KEYS, t):
        print(f"   {k:14s} {v:.4g}")
    pred = predict(t)
    print(f"\n{'split':24s}{'shipped (unit-free)':>22s}{'microvolt thresholds':>23s}{'change':>10s}")
    for name, m in (("sub_01 1-30 (design)", DEV), ("sub_01 31-36 (TEST)", T1M), ("sub_11 (TEST)", S11)):
        a, b = bal(SHIPPED, m), bal(pred, m)
        print(f"{name:24s}{a:21.2f}%{b:22.2f}%{b - a:+9.2f}")

    print("\nper class:")
    for name, m in (("sub_01 1-30", DEV), ("sub_01 31-36", T1M), ("sub_11", S11)):
        s1 = " ".join(f"{CL[k][0]} {100 * (SHIPPED[m & (Y == k)] == k).mean():5.1f}" for k in range(4))
        s2 = " ".join(f"{CL[k][0]} {100 * (pred[m & (Y == k)] == k).mean():5.1f}" for k in range(4))
        print(f"   {name:14s} shipped   {s1}")
        print(f"   {'':14s} microvolt {s2}")


if __name__ == "__main__":
    main()
