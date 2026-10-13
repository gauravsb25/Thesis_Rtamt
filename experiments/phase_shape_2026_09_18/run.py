"""Three requested experiments on top of the shipped monitor (85.01 / 69.62 / 78.67).

A. CALIBRATION CEILING. One known turn at the start of a session sets the sign of `turn` for that whole
   session. Reported next to the ORACLE sign (the best possible per-session sign), which is the ceiling.
   The calibration movement is excluded from scoring. This breaks the project's "no per-session constants"
   rule on purpose: it measures what such a rule would be worth.
B. POLARITY-FREE DIRECTION. Side taken from timing instead of voltage sign: peak-time difference F3-F4 or
   T3-T4, or the lag that aligns the left and right envelopes. Blocked 3-fold CV on sub_01 sessions 1-30.
C. NOD AGAINST TURN BY WAVE SHAPE. rise, fall, asymmetry, steepness and width of the movement envelope, as
   an extra condition on DOWN (or as a rescue of turns wrongly called DOWN). Same CV.

usage: python run.py               FINAL_TEST=1 python run.py test DESIGN[,DESIGN...]
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\gaura\thesis_rtamt")
HERE = Path(__file__).resolve().parent
CL = ["LEFT", "RIGHT", "DOWN", "STRAIGHT"]
P = dict(b=6.61, g=7.81, d=0.208, q=2.72, n=-0.661, v=7.54)
FOLDS = [[f"sub_01/ses-{s:02d}" for s in range(a, a + 10)] for a in (1, 11, 21)]

M = dict(np.load(ROOT / "measurements_v2.npz", allow_pickle=True))
F = dict(np.load(HERE / "features.npz", allow_pickle=True))
LAB, SES = M["label"].astype(str), M["session"].astype(str)
assert (F["session"].astype(str) == SES).all() and (F["label"].astype(str) == LAB).all()
DEV = np.array([s.startswith("sub_01") and int(s[-2:]) <= 30 for s in SES])
T1 = np.char.startswith(SES, "sub_01") & ~DEV
S11 = np.char.startswith(SES, "sub_11")
VALID = M["background_valid"] >= 0.5
LOUD = np.where(VALID, M["background"] >= P["b"], M["contrast"] >= P["g"])
MOVE = LOUD & (M["delta"] >= P["d"]) & (M["deflect"] >= P["q"])
NOD = (M["nod"] >= P["n"]) & (M["sideways"] <= P["v"])
SIDE = np.where(M["turn"] > 0, "LEFT", "RIGHT")
TURNWIN = np.isin(LAB, ["LEFT", "RIGHT"])
MOVEMENT_ID = np.array([f"{s}|{t}|{r}|{o // 64}" for s, t, r, o in
                        zip(SES, F["task"].astype(str), F["run"], F["onset"])])


def assemble(move, nod, side):
    return np.where(~move, "STRAIGHT", np.where(nod, "DOWN", side))


BASE = assemble(MOVE, NOD, SIDE)


def bal(pred, m):
    return 100 * float(np.mean([np.mean(pred[m & (LAB == k)] == k) for k in CL if (m & (LAB == k)).any()]))


def cls_text(pred, m):
    return " ".join(f"{k[0]} {100 * np.mean(pred[m & (LAB == k)] == k):5.1f}" for k in CL)


# ---------------------------------------------------------------- A. calibration ceiling
def calibration(kind):
    """Per-session sign for `turn`: 'one_turn' uses the session's first turn, 'oracle' uses the best sign.

    Returns (predictions, mask of windows to score) -- the calibration movement is not scored.
    """
    side = SIDE.copy()
    scored = np.ones(len(LAB), bool)
    for session in np.unique(SES):
        here = SES == session
        turns = here & TURNWIN & MOVE & ~NOD           # windows the monitor calls a turn
        if not turns.any():
            continue
        if kind == "oracle":
            sign = 1.0 if np.mean(SIDE[turns] == LAB[turns]) >= 0.5 else -1.0
        else:
            first = sorted(set(MOVEMENT_ID[here & TURNWIN]))[0]
            calib = here & (MOVEMENT_ID == first)
            use = calib & MOVE & ~NOD
            if not use.any():
                use = calib
            sign = 1.0 if np.mean(SIDE[use] == LAB[use]) >= 0.5 else -1.0
            scored[calib] = False                      # the calibration movement cannot be scored
        if sign < 0:
            side[here] = np.where(SIDE[here] == "LEFT", "RIGHT", "LEFT")
    return assemble(MOVE, NOD, side), scored


def report_calibration():
    print("A. What one known turn per session would buy (this breaks the no-per-session rule on purpose)\n")
    print(f"   {'split':16s}{'monitor now':>14s}{'1-turn calibration':>21s}{'oracle sign (ceiling)':>24s}")
    one, scored = calibration("one_turn")
    oracle, _ = calibration("oracle")
    for name, m in (("sub_01 1-30", DEV), ("sub_01 31-36", T1), ("sub_11", S11)):
        k = m & scored
        print(f"   {name:16s}{bal(BASE, k):13.2f}%{bal(one, k):20.2f}%{bal(oracle, k):23.2f}%")
    for name, m in (("sub_01 1-30", DEV), ("sub_01 31-36", T1), ("sub_11", S11)):
        k = m & scored & TURNWIN & MOVE & ~NOD
        print(f"   {name:16s} direction among turn windows: now {100 * np.mean(SIDE[k] == LAB[k]):5.1f}%"
              f"  1-turn {100 * np.mean(one[k] == LAB[k]):5.1f}%  oracle {100 * np.mean(oracle[k] == LAB[k]):5.1f}%"
              f"   ({int(k.sum())} windows)")
    flipped = [s for s in np.unique(SES) if (lambda t: t.any() and np.mean(SIDE[t] == LAB[t]) < 0.5)
               (( SES == s) & TURNWIN & MOVE & ~NOD)]
    print(f"\n   sessions whose sign is inverted: {len(flipped)} of {len(np.unique(SES))} -> {flipped}")


# ---------------------------------------------------------------- B and C. new rules
def grid(x, train, n=40):
    v = x[train & np.isfinite(x)]
    return np.unique(np.quantile(v, np.linspace(0.02, 0.98, n)))


def orientation(signal, train):
    lat = train & TURNWIN & MOVE & ~NOD
    return float(np.sign(np.median(signal[lat & (LAB == "LEFT")]) - np.median(signal[lat & (LAB == "RIGHT")])) or 1.0)


def candidates(design, train):
    if design == "B_lag_tiebreak":
        o = orientation(F["lag_env"], train)
        for t in grid(np.abs(M["turn"]), train & MOVE & ~NOD, 20):
            side = np.where(np.abs(M["turn"]) >= t, SIDE, np.where(F["lag_env"] * o >= 0, "LEFT", "RIGHT"))
            yield f"use timing when |turn| < {t:.3g}", MOVE, NOD, side
    elif design.startswith("B_"):
        key = {"B_lag_f3f4": "lag_f3f4", "B_lag_t3t4": "lag_t3t4", "B_lag_env": "lag_env"}[design]
        o = orientation(F[key], train)
        yield f"orientation {o:+.0f}", MOVE, NOD, np.where(F[key] * o >= 0, "LEFT", "RIGHT")
    elif design.startswith("C_"):
        key = design.split("_", 1)[1]
        x = F[key]
        for t in grid(x, train & MOVE, 40):
            yield f"DOWN needs {key} <= {t:.4g}", MOVE, NOD & (x <= t), SIDE
            yield f"DOWN needs {key} >= {t:.4g}", MOVE, NOD & (x >= t), SIDE
    else:
        raise ValueError(design)


def select(designs, train):
    move, nod, side, chosen = MOVE, NOD, SIDE, []
    for design in designs:
        best, name = bal(assemble(move, nod, side), train), "current"
        state = (move, nod, side)
        for label, m_, n_, s_ in candidates(design, train):
            m2, n2, s2 = move, (n_ if design.startswith("C_") else nod), (s_ if design.startswith("B_") else side)
            a = bal(assemble(m2, n2, s2), train)
            if a > best + 1e-9:
                best, name, state = a, label, (m2, n2, s2)
        move, nod, side = state
        chosen.append(f"{design}: {name}")
    return chosen, assemble(move, nod, side)


DESIGNS = ["B_lag_f3f4", "B_lag_t3t4", "B_lag_env", "B_lag_tiebreak",
           "C_asym", "C_rise", "C_fall", "C_steep", "C_width"]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if os.environ.get("FINAL_TEST") != "1":
            raise SystemExit("needs FINAL_TEST=1")
        for spec in sys.argv[2].split(","):
            chosen, pred = select(spec.split("+"), DEV)
            print(f"\n{spec}: fitted on all of sessions 1-30 -> {chosen}")
            for name, m in (("sessions 1-30", DEV), ("sub_01 31-36", T1), ("sub_11", S11)):
                print(f"   {name:14s} current {bal(BASE, m):6.2f}% ({cls_text(BASE, m)})  ->  "
                      f"{bal(pred, m):6.2f}% ({cls_text(pred, m)})")
        return
    report_calibration()
    print(f"\nB and C. baseline on sessions 1-30: {bal(BASE, DEV):.2f}%  ({cls_text(BASE, DEV)})\n")
    print("   direction accuracy among turn windows (no fitting, orientation from sessions 1-30):")
    for key in ("lag_f3f4", "lag_t3t4", "lag_env"):
        o = orientation(F[key], DEV)
        pred = np.where(F[key] * o >= 0, "LEFT", "RIGHT")
        cells = []
        for m in (DEV, T1, S11):
            k = m & TURNWIN & MOVE & ~NOD
            cells.append(f"{100 * np.mean(pred[k] == LAB[k]):6.1f}%")
        print(f"      {key:12s}" + "".join(cells) + f"   (current rule: 92.6% / 100.0% / 70.5%)")
    print(f"      lag_env alignment quality: median {np.median(F['lag_env_q']):.2f}\n")
    for design in DESIGNS:
        cv, chosen = BASE.copy(), []
        for k in range(3):
            held = np.isin(SES, FOLDS[k])
            names, pred = select(design.split("+"), DEV & ~held)
            cv[held] = pred[held]
            chosen.append(names)
        d = [bal(cv, SES == s) - bal(BASE, SES == s) for s in np.unique(SES[DEV])]
        up, down = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
        gain = bal(cv, DEV) - bal(BASE, DEV)
        print(f"{design:16s} CV {bal(cv, DEV):6.2f}% ({gain:+5.2f}) sessions +{up}/-{down}  {cls_text(cv, DEV)}"
              f"  {'PASS' if gain >= 0.5 and up > down else ''}")
        print(f"                 {chosen}", flush=True)


if __name__ == "__main__":
    main()
