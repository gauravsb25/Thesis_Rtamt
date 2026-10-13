"""The five paradigms, measured with the project's protocol.

Baseline: the shipped monitor (85.01 / 69.62 / 78.67). Blocked 3-fold CV on sub_01 sessions 1-30;
adoption rule fixed first (CV gain >= +0.5 and more sessions better than worse); one test run for a pass.

P1 Laplacian    the whole monitor on Laplacian channels; or only its nod, only its direction, or its
                sideways veto, put into the current monitor
P2 entropy      permutation entropy as the gate (a movement is an ORDERED stretch) or as an extra gate test
P3 SPRT         accumulated robustness instead of the instantaneous one: MOVE when the accumulator passes A
P4 phase        Hilbert phase difference as the direction; phase jumps as a blink veto on DOWN
P5 is not a rule but a reframing; `paradigm5()` prints the numbers it needs.

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
OTHER = (M["delta"] >= P["d"]) & (M["deflect"] >= P["q"])
MOVE = LOUD & OTHER
NOD = (M["nod"] >= P["n"]) & (M["sideways"] <= P["v"])
SIDE = np.where(M["turn"] > 0, "LEFT", "RIGHT")
TURNWIN = np.isin(LAB, ["LEFT", "RIGHT"])


def assemble(move, nod, side):
    return np.where(~move, "STRAIGHT", np.where(nod, "DOWN", side))


BASE = assemble(MOVE, NOD, SIDE)


def bal(pred, m):
    return 100 * float(np.mean([np.mean(pred[m & (LAB == k)] == k) for k in CL if (m & (LAB == k)).any()]))


def cls_text(pred, m):
    return " ".join(f"{k[0]} {100 * np.mean(pred[m & (LAB == k)] == k):5.1f}" for k in CL)


def grid(x, train, n=40):
    v = x[train & np.isfinite(x)]
    return np.unique(np.quantile(v, np.linspace(0.02, 0.98, n)))


def orientation(signal, train):
    lat = train & TURNWIN & MOVE & ~NOD
    return float(np.sign(np.median(signal[lat & (LAB == "LEFT")]) - np.median(signal[lat & (LAB == "RIGHT")])) or 1.0)


def candidates(design, train):
    if design == "P1_all":                        # the whole monitor on Laplacian channels
        for gate in grid(F["lap_contrast"], train, 25):
            for nod_t in grid(F["lap_nod"], train & MOVE, 15):
                move = (F["lap_contrast"] >= gate) & (F["lap_delta"] >= P["d"]) & (F["lap_deflect"] >= P["q"])
                nod = (F["lap_nod"] >= nod_t) & (F["lap_sideways"] <= P["v"])
                yield f"gate {gate:.3g} nod {nod_t:.3g}", move, nod, np.where(F["lap_turn"] > 0, "LEFT", "RIGHT")
    elif design == "P1_nod":
        for t in grid(F["lap_nod"], train & MOVE):
            yield f"lap_nod >= {t:.4g}", MOVE, (F["lap_nod"] >= t) & (M["sideways"] <= P["v"]), SIDE
    elif design == "P1_sideways":
        for t in grid(F["lap_sideways"], train & MOVE):
            yield f"lap_sideways <= {t:.4g}", MOVE, (M["nod"] >= P["n"]) & (F["lap_sideways"] <= t), SIDE
    elif design == "P1_side":
        o = orientation(F["lap_turn"], train)
        yield f"orientation {o:+.0f}", MOVE, NOD, np.where(F["lap_turn"] * o > 0, "LEFT", "RIGHT")
    elif design in ("P2_gate", "P2_extra"):
        key = "pe_min" if design == "P2_gate" else "pe_drop"
        for t in grid(F[key], train):
            rule = (F[key] <= t) if design == "P2_gate" else (F[key] >= t)
            move = (rule & OTHER) if design == "P2_gate" else (LOUD & OTHER & rule)
            yield f"{key} {'<=' if design == 'P2_gate' else '>='} {t:.4g}", move, NOD, SIDE
    elif design.startswith("P3_sprt"):
        key = f"sprt_{design.split('_')[-1]}"
        for t in grid(F[key], train, 50):
            yield f"{key} >= {t:.4g}", (F[key] >= t) & OTHER, NOD, SIDE
    elif design == "P4_side":
        o = orientation(F["phase_diff"], train)
        yield f"orientation {o:+.0f}", MOVE, NOD, np.where(F["phase_diff"] * o >= 0, "LEFT", "RIGHT")
    elif design in ("P4_jump", "P4_jump_ratio", "P4_rough"):
        key = {"P4_jump": "phase_jump", "P4_jump_ratio": "phase_jump_ratio", "P4_rough": "phase_rough"}[design]
        for t in grid(F[key], train & MOVE):
            yield f"DOWN needs {key} <= {t:.4g}", MOVE, NOD & (F[key] <= t), SIDE
            yield f"gate needs {key} <= {t:.4g}", MOVE & (F[key] <= t), NOD, SIDE
    else:
        raise ValueError(design)


def select(designs, train):
    move, nod, side, chosen = MOVE, NOD, SIDE, []
    for design in designs:
        best, name, state = bal(assemble(move, nod, side), train), "current", (move, nod, side)
        for label, m_, n_, s_ in candidates(design, train):
            a = bal(assemble(m_, n_, s_), train)
            if a > best + 1e-9:
                best, name, state = a, label, (m_, n_, s_)
        move, nod, side = state
        chosen.append(f"{design}: {name}")
    return chosen, assemble(move, nod, side)


DESIGNS = ["P1_all", "P1_nod", "P1_sideways", "P1_side", "P2_gate", "P2_extra",
           "P3_sprt_0.0", "P3_sprt_0.05", "P3_sprt_0.1", "P3_sprt_0.2",
           "P4_side", "P4_jump", "P4_jump_ratio", "P4_rough"]


def paradigm5():
    """The numbers the reframing needs: movement detection is strong, direction is not."""
    print("\nP5. What the system is actually good at (the reframing)\n")
    print(f"   {'split':16s}{'MOVE vs STRAIGHT':>20s}{'nod vs turn':>14s}{'left vs right':>16s}{'4-class':>10s}")
    for name, m in (("sub_01 1-30", DEV), ("sub_01 31-36", T1), ("sub_11", S11)):
        rest, move = m & (LAB == "STRAIGHT"), m & (LAB != "STRAIGHT")
        detect = 0.5 * (np.mean(BASE[rest] == "STRAIGHT") + np.mean(BASE[move] != "STRAIGHT"))
        both = m & MOVE & (LAB != "STRAIGHT")
        nod_ok = np.mean((BASE[both] == "DOWN") == (LAB[both] == "DOWN"))
        lat = m & TURNWIN & MOVE & ~NOD
        side_ok = np.mean(BASE[lat] == LAB[lat])
        print(f"   {name:16s}{100 * detect:19.1f}%{100 * nod_ok:13.1f}%{100 * side_ok:15.1f}%{bal(BASE, m):9.2f}%")
    print("   (movement detection is balanced over the two classes; nod-vs-turn and left-vs-right are"
          " measured only on windows the monitor already calls a movement)")


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
    print(f"baseline on sessions 1-30: {bal(BASE, DEV):.2f}%  ({cls_text(BASE, DEV)})\n")
    k = DEV & TURNWIN & MOVE & ~NOD
    for key in ("lap_turn", "phase_diff"):
        o = orientation(F[key], DEV)
        pred = np.where(F[key] * o >= 0, "LEFT", "RIGHT")
        cells = "".join(f"{100 * np.mean(pred[m & TURNWIN & MOVE & ~NOD] == LAB[m & TURNWIN & MOVE & ~NOD]):7.1f}%"
                        for m in (DEV, T1, S11))
        print(f"   direction from {key:12s}" + cells)
    print(f"   direction from {'F4-T3 (current)':12s}" + "".join(
        f"{100 * np.mean(SIDE[m & TURNWIN & MOVE & ~NOD] == LAB[m & TURNWIN & MOVE & ~NOD]):7.1f}%"
        for m in (DEV, T1, S11)) + "\n")
    for design in (sys.argv[1].split(",") if len(sys.argv) > 1 else DESIGNS):
        cv, chosen = BASE.copy(), []
        for k in range(3):
            held = np.isin(SES, FOLDS[k])
            names, pred = select(design.split("+"), DEV & ~held)
            cv[held] = pred[held]
            chosen.append(names[-1].split(": ", 1)[1])
        d = [bal(cv, SES == s) - bal(BASE, SES == s) for s in np.unique(SES[DEV])]
        up, down = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
        gain = bal(cv, DEV) - bal(BASE, DEV)
        print(f"{design:14s} CV {bal(cv, DEV):6.2f}% ({gain:+5.2f}) sessions +{up}/-{down}  {cls_text(cv, DEV)}"
              f"  {'PASS' if gain >= 0.5 and up > down else ''}")
        print(f"               folds: {chosen}", flush=True)
    paradigm5()


if __name__ == "__main__":
    main()
