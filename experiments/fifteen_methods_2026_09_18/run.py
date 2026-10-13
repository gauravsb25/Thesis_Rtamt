"""All 15 proposed methods as monitor changes, with the project's protocol.

Baseline: the shipped monitor (85.01 / 69.62 / 78.67). Selection: blocked 3-fold CV on sub_01 sessions 1-30.
Adoption rule, fixed first: CV gain >= +0.5 points and more sessions better than worse; only then one test run.

Each method changes exactly one part of the decision:
  gate    (methods 1, 3, 12, 13, 14)   what counts as "a movement happened"
  nod     (methods 2, 5, 8, 9, 10, 11, 15)  DOWN against LEFT/RIGHT
  side    (methods 4, 5, 6, 7)         which way the turn went

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
SCALE = {k: 1.4826 * np.median(np.abs(M[k][DEV] - np.median(M[k][DEV]))) for k in ("delta", "nod")}
# how far the gate was from its thresholds, unit-free (method 15)
GATE_MARGIN = np.minimum.reduce([
    np.where(VALID, np.log(np.maximum(M["background"], 1e-9) / P["b"]), np.log(M["contrast"] / P["g"])),
    (M["delta"] - P["d"]) / SCALE["delta"], np.log(M["deflect"] / P["q"])])


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


SIDE_KEYS = {"m4_power_asym": "power_asym", "m5_phase_side": "phase_f", "m6_mag_imbalance": "mag_imbalance",
             "m7_centroid": "centroid_shift", "m7_centroid_abs": "centroid"}
NOD_KEYS = {"m2_share_temporal": "rel_share_temporal", "m2_share_midline": "rel_share_midline",
            "m2_share_max": "rel_share_max", "m5_phase_nod_f": "phase_f", "m5_phase_nod_t": "phase_t",
            "m8_cm_diff": "cm_diff", "m9_coh_ratio": "coh_ratio", "m10_vel_gradient": "vel_gradient",
            "m11_ocular_decay": "ocular_decay", "m11_ocular_decay_loud": "ocular_decay_loud"}
GATE_KEYS = {"m3_delta_share": "delta_share", "m3_delta_share_range": "delta_share_range",
             "m12_flatness": "flatness", "m13_curvature": "curvature"}


def candidates(design, train):
    if design in SIDE_KEYS:                      # methods 4-7: a different way to pick the side
        key = SIDE_KEYS[design]
        o = orientation(F[key], train)
        yield f"orientation {o:+.0f}", MOVE, NOD, np.where(F[key] * o >= 0, "LEFT", "RIGHT")
    elif design in NOD_KEYS:                     # methods 2, 5, 8-11: an extra condition on DOWN
        x = F[NOD_KEYS[design]]
        for t in grid(x, train & MOVE):
            yield f"DOWN needs {NOD_KEYS[design]} <= {t:.4g}", MOVE, NOD & (x <= t), SIDE
            yield f"DOWN needs {NOD_KEYS[design]} >= {t:.4g}", MOVE, NOD & (x >= t), SIDE
    elif design in GATE_KEYS:                    # methods 3, 12, 13: an extra condition on the gate
        x = F[GATE_KEYS[design]]
        for t in grid(x, train):
            yield f"gate needs {GATE_KEYS[design]} >= {t:.4g}", LOUD & OTHER & (x >= t), NOD, SIDE
            yield f"gate needs {GATE_KEYS[design]} <= {t:.4g}", LOUD & OTHER & (x <= t), NOD, SIDE
    elif design == "m1_hist_z":                  # method 1: gate on the history's median/IQR scale
        valid = F["hist_valid"] >= 0.5
        for t in grid(F["hist_z"], train & valid):
            loud = np.where(valid, F["hist_z"] >= t, LOUD)
            yield f"hist_z >= {t:.4g}", loud & OTHER, NOD, SIDE
            yield f"hist_z >= {t:.4g} AND the current gate", np.where(valid, (F["hist_z"] >= t) & LOUD, LOUD) & OTHER, NOD, SIDE
    elif design == "m15_soft_veto":              # method 15: apply the sideways veto only when the gate is unsure
        for eps in grid(GATE_MARGIN, train & MOVE, 30):
            nod = (M["nod"] >= P["n"]) & ((M["sideways"] <= P["v"]) | (GATE_MARGIN >= eps))
            yield f"skip the sideways veto when the gate margin >= {eps:.3g}", MOVE, nod, SIDE
    else:
        raise ValueError(design)


def select(designs, train):
    move, nod, side, chosen = MOVE, NOD, SIDE, []
    for design in designs:
        best, name, state = bal(assemble(move, nod, side), train), "current", (move, nod, side)
        for label, m_, n_, s_ in candidates(design, train):
            state2 = (m_ if not np.array_equal(m_, MOVE) else move,
                      n_ if not np.array_equal(n_, NOD) else nod,
                      s_ if not np.array_equal(s_, SIDE) else side)
            a = bal(assemble(*state2), train)
            if a > best + 1e-9:
                best, name, state = a, label, state2
        move, nod, side = state
        chosen.append(f"{design}: {name}")
    return chosen, assemble(move, nod, side)


DESIGNS = ["m1_hist_z"] + list(NOD_KEYS) + list(GATE_KEYS) + list(SIDE_KEYS) + ["m15_soft_veto"]


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
    print(f"baseline on sessions 1-30: {bal(BASE, DEV):.2f}%  ({cls_text(BASE, DEV)})")
    print("\ndirection accuracy of the polarity-free reads, among turn windows (orientation from 1-30):")
    for design, key in SIDE_KEYS.items():
        o = orientation(F[key], DEV)
        pred = np.where(F[key] * o >= 0, "LEFT", "RIGHT")
        cells = []
        for m in (DEV, T1, S11):
            k = m & TURNWIN & MOVE & ~NOD
            cells.append(f"{100 * np.mean(pred[k] == LAB[k]):7.1f}%")
        print(f"   {key:16s}" + "".join(cells))
    k = DEV & TURNWIN & MOVE & ~NOD
    print(f"   {'current F4-T3':16s}" + "".join(
        f"{100 * np.mean(SIDE[m & TURNWIN & MOVE & ~NOD] == LAB[m & TURNWIN & MOVE & ~NOD]):7.1f}%"
        for m in (DEV, T1, S11)))
    print()
    for design in (sys.argv[1].split(",") if len(sys.argv) > 1 else DESIGNS):
        cv, chosen = BASE.copy(), []
        for k in range(3):
            held = np.isin(SES, FOLDS[k])
            names, pred = select(design.split("+"), DEV & ~held)
            cv[held] = pred[held]
            chosen.append(names[0].split(": ", 1)[1] if names else "")
        d = [bal(cv, SES == s) - bal(BASE, SES == s) for s in np.unique(SES[DEV])]
        up, down = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
        gain = bal(cv, DEV) - bal(BASE, DEV)
        print(f"{design:22s} CV {bal(cv, DEV):6.2f}% ({gain:+5.2f}) sessions +{up}/-{down}  {cls_text(cv, DEV)}"
              f"  {'PASS' if gain >= 0.5 and up > down else ''}")
        print(f"                       folds: {chosen}", flush=True)


if __name__ == "__main__":
    main()
