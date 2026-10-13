"""Methods 1, 2 and 5: rest-state adaptation, Riemannian tangent space, and the STL+ML cascade.

Baseline: the shipped monitor, 85.01 / 69.62 / 78.67 (direction among turn windows 92.6 / 100.0 / 70.5).

A. RULES from rest-normalised signals (method 1), with the usual protocol: blocked 3-fold CV on sub_01
   sessions 1-30, adoption rule fixed first, one test run for a pass. NOTE these rules use the session's own
   rest state, which the project's rules otherwise avoid; it is unsupervised (no labels), so it would be
   deployable as "sit still for 20 seconds at the start".
B. MODELS on Riemannian tangent-space features (method 2), as a ceiling estimate: session-grouped 5-fold CV
   inside sessions 1-30, then one held-out scoring. Compared with yesterday's models on the 87 hand-made
   signals and with the monitors.
C. CASCADE (method 5): the STL monitor decides MOVE / STRAIGHT and nod vs turn; a model decides LEFT vs
   RIGHT only for the windows the monitor calls a turn.

usage: python run.py               FINAL_TEST=1 python run.py test
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
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
TURN = np.isin(LAB, ["LEFT", "RIGHT"])
TAN = np.column_stack([F[k] for k in sorted(F) if k.startswith("tan_")])
REST = np.column_stack([F[k] for k in ("rest_deflect", "rest_nod", "rest_sideways", "rest_turn", "rest_energy")])


def assemble(move, nod, side):
    return np.where(~move, "STRAIGHT", np.where(nod, "DOWN", side))


BASE = assemble(MOVE, NOD, SIDE)


def bal(pred, m, truth=None):
    truth = LAB if truth is None else truth
    return 100 * float(np.mean([np.mean(pred[m & (truth == k)] == k) for k in np.unique(truth[m])]))


def cls_text(pred, m):
    return " ".join(f"{k[0]} {100 * np.mean(pred[m & (LAB == k)] == k):5.1f}" for k in CL)


def grid(x, train, n=40):
    v = x[train & np.isfinite(x)]
    return np.unique(np.quantile(v, np.linspace(0.02, 0.98, n)))


def candidates(design, train):
    if design == "A_gate":                      # rest-referenced energy as the gate
        for t in grid(F["rest_energy"], train):
            yield f"rest_energy >= {t:.4g}", (F["rest_energy"] >= t) & OTHER, NOD, SIDE
    elif design == "A_gate_and":
        for t in grid(F["rest_energy"], train):
            yield f"gate and rest_energy >= {t:.4g}", MOVE & (F["rest_energy"] >= t), NOD, SIDE
    elif design == "A_deflect":
        for t in grid(F["rest_deflect"], train):
            yield f"rest_deflect >= {t:.4g}", LOUD & (M["delta"] >= P["d"]) & (F["rest_deflect"] >= t), NOD, SIDE
    elif design == "A_nod":
        for t in grid(F["rest_nod"], train & MOVE):
            yield f"rest_nod >= {t:.4g}", MOVE, (F["rest_nod"] >= t) & (M["sideways"] <= P["v"]), SIDE
    elif design == "A_sideways":
        for t in grid(F["rest_sideways"], train & MOVE):
            yield f"rest_sideways <= {t:.4g}", MOVE, (M["nod"] >= P["n"]) & (F["rest_sideways"] <= t), SIDE
    elif design == "A_side":
        yield "sign of rest_turn", MOVE, NOD, np.where(F["rest_turn"] > 0, "LEFT", "RIGHT")
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


def models():
    return {"logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
            "random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                                    n_jobs=-1, random_state=0)}


def cv_scores(X, mask, label, groups):
    out = {}
    splitter = GroupKFold(n_splits=5)
    for name, model in models().items():
        pred = np.empty(int(mask.sum()), dtype=object)
        Xd, yd, gd = X[mask], label[mask], groups[mask]
        for train, test in splitter.split(Xd, yd, groups=gd):
            model.fit(Xd[train], yd[train])
            pred[test] = model.predict(Xd[test])
        out[name] = bal(np.array(pred, dtype=str), np.ones(len(yd), bool), yd)
    return out


def main():
    test_mode = len(sys.argv) > 1 and sys.argv[1] == "test"
    if test_mode and os.environ.get("FINAL_TEST") != "1":
        raise SystemExit("needs FINAL_TEST=1")
    print(f"baseline: 85.01 / 69.62 / 78.67 (direction 92.6 / 100.0 / 70.5); tangent features {TAN.shape[1]}\n")

    if not test_mode:
        print("A. rules from the session's rest state, blocked 3-fold CV on sessions 1-30")
        for design in ("A_gate", "A_gate_and", "A_deflect", "A_nod", "A_sideways", "A_side"):
            cv, chosen = BASE.copy(), []
            for k in range(3):
                held = np.isin(SES, FOLDS[k])
                names, pred = select([design], DEV & ~held)
                cv[held] = pred[held]
                chosen.append(names[0].split(": ", 1)[1])
            d = [bal(cv, SES == s) - bal(BASE, SES == s) for s in np.unique(SES[DEV])]
            up, down = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
            gain = bal(cv, DEV) - bal(BASE, DEV)
            print(f"   {design:12s} CV {bal(cv, DEV):6.2f}% ({gain:+5.2f}) sessions +{up}/-{down}  {cls_text(cv, DEV)}"
                  f"  {'PASS' if gain >= 0.5 and up > down else ''}  folds: {chosen}", flush=True)

        print("\nB. models on the Riemannian tangent space, 5-fold grouped CV inside sessions 1-30")
        four = cv_scores(TAN, DEV, LAB, SES)
        direction = cv_scores(TAN, DEV & TURN, LAB, SES)
        for name in four:
            print(f"   {name:22s} 4-class {four[name]:6.2f}%   direction {direction[name]:6.1f}%")
        print("   (the monitors: 85.01% and 92.6%; yesterday's 87 hand-made signals: 87.58% and 90.2%)")
        print("\n   tangent space + the 87 hand-made signals together:")
        big = np.column_stack([TAN, REST, np.column_stack([M[k] for k in
              ("contrast", "background", "delta", "deflect", "nod", "sideways", "turn")])])
        four = cv_scores(big, DEV, LAB, SES)
        direction = cv_scores(big, DEV & TURN, LAB, SES)
        for name in four:
            print(f"   {name:22s} 4-class {four[name]:6.2f}%   direction {direction[name]:6.1f}%")
        return

    print("B. tangent-space models trained on sessions 1-30, scored once")
    print(f"   {'model':22s}{'4-class 31-36':>15s}{'4-class sub_11':>16s}{'direction 31-36':>17s}{'direction sub_11':>18s}")
    for name, model in models().items():
        model.fit(TAN[DEV], LAB[DEV])
        pred = model.predict(TAN)
        dmodel = models()[name]
        dmodel.fit(TAN[DEV & TURN], LAB[DEV & TURN])
        dpred = dmodel.predict(TAN)
        print(f"   {name:22s}{bal(pred, T1):14.2f}%{bal(pred, S11):15.2f}%"
              f"{bal(dpred, T1 & TURN):16.1f}%{bal(dpred, S11 & TURN):17.1f}%")

    print("\nC. cascade: the STL monitor decides, a model only re-decides LEFT vs RIGHT")
    for name, model in models().items():
        for tag, X in (("tangent space", TAN), ("rest-normalised + monitor signals",
                        np.column_stack([REST, np.column_stack([M[k] for k in
                            ("contrast", "background", "delta", "deflect", "nod", "sideways", "turn")])]))):
            m = models()[name]
            train = DEV & MOVE & ~NOD & TURN
            m.fit(X[train], LAB[train])
            side = np.where(m.predict(X) == "LEFT", "LEFT", "RIGHT")
            pred = assemble(MOVE, NOD, side)
            print(f"   {name:20s} {tag:34s} " + "  ".join(
                f"{text} {bal(pred, msk):6.2f}%" for text, msk in
                (("sessions 1-30", DEV), ("31-36", T1), ("sub_11", S11))))
    print(f"   {'monitor (no model)':20s} {'':34s} " + "  ".join(
        f"{text} {bal(BASE, msk):6.2f}%" for text, msk in
        (("sessions 1-30", DEV), ("31-36", T1), ("sub_11", S11))))


if __name__ == "__main__":
    main()
