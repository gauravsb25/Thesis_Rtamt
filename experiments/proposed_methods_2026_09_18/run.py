"""The four proposed methods, tested with the project's protocol.

Baseline: the shipped monitor with the stream background (85.01 / 69.62 / 78.67).
Selection: blocked 3-fold CV on sub_01 sessions 1-30. Adoption rule, fixed first: CV gain >= +0.5 points
and more sessions better than worse; only then the held-out sets are scored once.

M1 direction (multi-pair consensus)
   M1_vote3        side = majority sign of F4-T3, F4-F3, T4-T3 (each pair's polarity fixed on the training sessions)
   M1_confirm      keep the current side when lat_signed agrees; otherwise fall back to it (variant: abstain)
   M1_abstain      when the pairs disagree the window is called STRAIGHT instead of guessing a side
   M1_imbalance    side = sign of lat_signed alone
   M1_magnitude    side = sign of lateral_imbalance (the magnitude version) alone
M2 turns called DOWN
   M2_yaw / M2_yaw_loud     DOWN also needs yaw_ratio <= t (whole window / loudest 1 s)
M3 blinks called DOWN
   M3_ocular / M3_ocular_loud   DOWN also needs ocular_ratio <= t
   M3_sustain_<K>               the gate needs the background to HOLD for K samples, not just touch the level
M4 weak nods
   M4_midline / M4_midline4     MOVE also fires when the midline prominence passes its own threshold
M5 the combination of whatever passed

usage: python run.py            FINAL_TEST=1 python run.py test DESIGN[,DESIGN...]
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

M = dict(np.load(ROOT / "measurements_v2.npz", allow_pickle=True))      # the shipped measurements
F = dict(np.load(HERE / "features.npz", allow_pickle=True))             # the proposed signals
LAB = M["label"].astype(str)
SES = M["session"].astype(str)
assert (F["session"].astype(str) == SES).all()
DEV = np.array([s.startswith("sub_01") and int(s[-2:]) <= 30 for s in SES])
VALID = M["background_valid"] >= 0.5
LOUD = np.where(VALID, M["background"] >= P["b"], M["contrast"] >= P["g"])
REST_TESTS = (M["delta"] >= P["d"])
MOVE = LOUD & REST_TESTS & (M["deflect"] >= P["q"])
NOD = (M["nod"] >= P["n"]) & (M["sideways"] <= P["v"])
SIDE = np.where(M["turn"] > 0, "LEFT", "RIGHT")


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
    """+1 if the signal is larger for LEFT than for RIGHT on the training sessions."""
    lat = train & np.isin(LAB, ["LEFT", "RIGHT"])
    return float(np.sign(np.median(signal[lat & (LAB == "LEFT")]) - np.median(signal[lat & (LAB == "RIGHT")])) or 1.0)


def candidates(design, train):
    turn_ok = train & np.isin(LAB, ["LEFT", "RIGHT"]) & MOVE
    if design == "M1_vote3":
        o_f, o_t = orientation(F["dir_frontal"], turn_ok), orientation(F["dir_temporal"], turn_ok)
        votes = np.sign(M["turn"]) + np.sign(F["dir_frontal"] * o_f) + np.sign(F["dir_temporal"] * o_t)
        side = np.where(votes >= 0, "LEFT", "RIGHT")
        yield f"orientations frontal {o_f:+.0f} temporal {o_t:+.0f}", MOVE, NOD, side
    elif design in ("M1_confirm", "M1_abstain"):
        o = orientation(F["lat_signed"], turn_ok)
        agree = np.sign(M["turn"]) == np.sign(F["lat_signed"] * o)
        if design == "M1_confirm":
            yield f"orientation {o:+.0f}", MOVE, NOD, np.where(agree, SIDE, SIDE)      # identical to the current rule
        else:
            yield f"orientation {o:+.0f}", MOVE & (NOD | agree), NOD, SIDE             # disagreement -> STRAIGHT
    elif design in ("M1_imbalance", "M1_magnitude"):
        key = "lat_signed" if design == "M1_imbalance" else "lateral_imbalance"
        o = orientation(F[key], turn_ok)
        yield f"orientation {o:+.0f}", MOVE, NOD, np.where(F[key] * o > 0, "LEFT", "RIGHT")
    elif design.startswith("M2_"):
        key = "yaw_ratio_loud" if design.endswith("loud") else "yaw_ratio"
        for t in grid(F[key], train & MOVE):
            yield f"t={t:.4g}", MOVE, NOD & (F[key] <= t), SIDE
    elif design.startswith("M3_ocular"):
        key = "ocular_ratio_loud" if design.endswith("loud") else "ocular_ratio"
        for t in grid(F[key], train & MOVE):
            yield f"t={t:.4g}", MOVE, NOD & (F[key] <= t), SIDE
    elif design.startswith("M3_sustain"):
        K = int(design.split("_")[-1])
        bg, ct = F[f"bg_sustained_{K}"], F[f"contrast_sustained_{K}"]
        for scale in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
            loud = np.where(VALID, bg >= P["b"] * scale, ct >= P["g"] * scale)
            move = loud & REST_TESTS & (M["deflect"] >= P["q"])
            yield f"K={K} level={scale:.1f}x", move, NOD, SIDE
    elif design.startswith("M4_"):
        key = "midline_deflect4" if design.endswith("4") else "midline_deflect"
        for t in grid(F[key], train):
            move = LOUD & REST_TESTS & ((M["deflect"] >= P["q"]) | (F[key] >= t))
            yield f"t={t:.4g}", move, NOD, SIDE
    else:
        raise ValueError(design)


def select(designs, train):
    move, nod, side, chosen = MOVE, NOD, SIDE, []
    for design in designs:
        best, best_name = bal(assemble(move, nod, side), train), "current"
        best_state = (move, nod, side)
        for name, m_, n_, s_ in candidates(design, train):
            # each design changes only its own part, on top of what earlier designs chose
            m2 = m_ if design.startswith(("M3_sustain", "M4_")) else move
            n2 = n_ if design.startswith(("M2_", "M3_ocular")) else nod
            s2 = s_ if design.startswith("M1_") else side
            if design == "M1_abstain":
                m2 = m_
            a = bal(assemble(m2, n2, s2), train)
            if a > best + 1e-9:
                best, best_name, best_state = a, name, (m2, n2, s2)
        move, nod, side = best_state
        chosen.append(f"{design}: {best_name}")
    return chosen, assemble(move, nod, side)


DESIGNS = ["M1_vote3", "M1_abstain", "M1_imbalance", "M1_magnitude", "M2_yaw", "M2_yaw_loud",
           "M3_ocular", "M3_ocular_loud"] + [f"M3_sustain_{K}" for K in (32, 76, 128, 192)] + \
          ["M4_midline", "M4_midline4"]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if os.environ.get("FINAL_TEST") != "1":
            raise SystemExit("needs FINAL_TEST=1")
        for spec in sys.argv[2].split(","):
            chosen, pred = select(spec.split("+"), DEV)
            print(f"\n{spec}: fitted on all of sessions 1-30 -> {chosen}")
            for text, m in (("sessions 1-30", DEV), ("sub_01 31-36", np.char.startswith(SES, "sub_01") & ~DEV),
                            ("sub_11", np.char.startswith(SES, "sub_11"))):
                print(f"   {text:14s} current {bal(BASE, m):6.2f}% ({cls_text(BASE, m)})  ->  "
                      f"{bal(pred, m):6.2f}% ({cls_text(pred, m)})")
        return
    print(f"baseline (shipped monitor), sessions 1-30: {bal(BASE, DEV):.2f}%  ({cls_text(BASE, DEV)})\n")
    for design in (sys.argv[1].split(",") if len(sys.argv) > 1 else DESIGNS):
        cv, chosen = BASE.copy(), []
        for k in range(3):
            held = np.isin(SES, FOLDS[k])
            names, pred = select(design.split("+"), DEV & ~held)
            cv[held] = pred[held]
            chosen.append(names)
        d = [bal(cv, SES == s) - bal(BASE, SES == s) for s in np.unique(SES[DEV])]
        up, down = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
        gain = bal(cv, DEV) - bal(BASE, DEV)
        flag = "PASS" if gain >= 0.5 and up > down else ""
        print(f"{design:18s} CV {bal(cv, DEV):6.2f}% ({gain:+5.2f}) sessions +{up}/-{down}  {cls_text(cv, DEV)}  {flag}")
        print(f"                   {chosen}", flush=True)


if __name__ == "__main__":
    main()
