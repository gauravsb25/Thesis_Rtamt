"""Do the onset-timing or unit-free features from the old classify.py improve the monitor?

Blocked 3-fold CV on sub_01 sessions 1-30. Adoption rule fixed first: CV gain >= +0.5 points AND
more sessions better than worse. A pass earns exactly one test run.

Roles: gate+/gate- (movement detection), nod+/nod- (DOWN), dir (LEFT/RIGHT).
usage: python cv.py            python cv.py --null 8       FINAL_TEST=1 python cv.py test
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
ROLES = ["gate+", "gate-", "nod+", "nod-", "dir"]
if "--role" in sys.argv:
    ROLES = [sys.argv[sys.argv.index("--role") + 1]]
NLEVEL = 15

F = dict(np.load(HERE / "onset.npz", allow_pickle=True))
M = dict(np.load(ROOT / "measurements_v2.npz", allow_pickle=True))
LAB, SES = M["label"].astype(str), M["session"].astype(str)
assert (F["session"].astype(str) == SES).all()
FEATS = sorted(k for k in F if k not in ("session", "label"))
X = np.stack([np.nan_to_num(F[k].astype(float), nan=0.0) for k in FEATS])
Y = np.array([CL.index(v) for v in LAB])

sess = np.array([int(s[-2:]) if s.startswith("sub_01") else -1 for s in SES])
DEV = (sess >= 1) & (sess <= 30)
T1 = np.char.startswith(SES, "sub_01") & (sess > 30)
S11 = np.char.startswith(SES, "sub_11")
VALID = M["background_valid"] >= 0.5
LOUD = np.where(VALID, M["background"] >= P["b"], M["contrast"] >= P["g"])
REST = (M["delta"] >= P["d"]) & (M["deflect"] >= P["q"])
MOVE = LOUD & REST
NOD = (M["nod"] >= P["n"]) & (M["sideways"] <= P["v"])
NOSWING = M["sideways"] <= P["v"]
LEFTISH = M["turn"] > 0
TURNLAB = np.isin(Y, [0, 1])


def assemble(move, nod, leftish):
    return np.where(~move, 3, np.where(nod, 2, np.where(leftish, 0, 1)))


BASE = assemble(MOVE, NOD, LEFTISH)


def idxs(mask):
    return [np.flatnonzero(mask & (Y == k)) for k in range(4)]


def bal(pred, ix):
    return 100.0 * float(np.mean([(pred[i] == k).mean() for k, i in enumerate(ix) if i.size]))


def apply_role(role, f, t):
    if role == "gate+":
        return assemble(MOVE & (f >= t), NOD, LEFTISH)
    if role == "gate-":
        return assemble(MOVE | (REST & (f >= t)), NOD, LEFTISH)
    if role == "nod+":
        return assemble(MOVE, NOD & (f >= t), LEFTISH)
    if role == "nod-":
        return assemble(MOVE, NOD | ((f >= t) & NOSWING), LEFTISH)
    if role == "dir":
        return assemble(MOVE, NOD, f >= t)
    raise ValueError(role)


def search(x, train):
    ix = idxs(train)
    base = bal(BASE, ix)
    best = (0.0, None, None, None)
    for j, name in enumerate(FEATS):
        for sign in (1.0, -1.0):
            g = x[j] * sign
            for role in ROLES:
                sub = train & (MOVE if role.startswith(("nod", "dir")) else np.ones(len(Y), bool))
                if role == "dir":
                    sub = sub & ~NOD & TURNLAB
                if sub.sum() < 30:
                    continue
                for t in np.unique(np.quantile(g[sub], np.linspace(0.05, 0.95, NLEVEL))):
                    a = bal(apply_role(role, g, t), ix)
                    if a - base > best[0] + 1e-9:
                        best = (a - base, f"{name}{'' if sign > 0 else ' (neg)'}", role, t)
    return best


def cv_gain(x, verbose=False):
    cv = BASE.copy()
    for k in range(3):
        held = np.isin(SES, FOLDS[k])
        gain, name, role, t = search(x, DEV & ~held)
        if name is None:
            continue
        j = FEATS.index(name.replace(" (neg)", ""))
        g = x[j] * (-1.0 if "(neg)" in name else 1.0)
        cv[held] = apply_role(role, g, t)[held]
        if verbose:
            print(f"   fold {k + 1}: {name} | {role} | {t:.4g}   (train gain {gain:+.2f})", flush=True)
    d = [bal(cv, idxs(SES == s)) - bal(BASE, idxs(SES == s)) for s in np.unique(SES[DEV])]
    return (bal(cv, idxs(DEV)) - bal(BASE, idxs(DEV)),
            sum(v > 1e-9 for v in d), sum(v < -1e-9 for v in d), cv)


def shuffled(rng):
    x = X.copy()
    for s in np.unique(SES):
        i = np.flatnonzero(SES == s)
        x[:, i] = x[:, rng.permutation(i)]
    return x


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if os.environ.get("FINAL_TEST") != "1":
            raise SystemExit("needs FINAL_TEST=1")
        gain, name, role, t = search(X, DEV)
        j = FEATS.index(name.replace(" (neg)", ""))
        g = X[j] * (-1.0 if "(neg)" in name else 1.0)
        pred = apply_role(role, g, t)
        print(f"fitted on all of sessions 1-30: {name} in role {role}, threshold {t:.4g}")
        for lbl, m in (("sessions 1-30", DEV), ("sub_01 31-36", T1), ("sub_11", S11)):
            ix = idxs(m)
            c1 = " ".join(f"{CL[k][0]} {100 * (BASE[i] == k).mean():5.1f}" for k, i in enumerate(ix) if i.size)
            c2 = " ".join(f"{CL[k][0]} {100 * (pred[i] == k).mean():5.1f}" for k, i in enumerate(ix) if i.size)
            print(f"   {lbl:14s} shipped {bal(BASE, ix):6.2f}% ({c1})  ->  {bal(pred, ix):6.2f}% ({c2})")
        return

    if "--null" in sys.argv:
        reps = int(sys.argv[sys.argv.index("--null") + 1])
        rng = np.random.default_rng(7)
        g = [cv_gain(shuffled(rng))[0] for _ in range(reps)]
        print(f"NOISE FLOOR from {reps} shuffles: mean {np.mean(g):+.2f}, max {np.max(g):+.2f}")
        return

    print(f"{len(FEATS)} features x {len(ROLES)} roles x 2 signs x {NLEVEL} thresholds "
          f"= {len(FEATS) * len(ROLES) * 2 * NLEVEL:,} candidates per fold")
    print(f"shipped monitor on sessions 1-30: {bal(BASE, idxs(DEV)):.2f}%\n")
    gain, up, down, cv = cv_gain(X, verbose=True)
    ok = gain >= 0.5 and up > down
    print(f"\nCV {bal(cv, idxs(DEV)):.2f}%  ({gain:+.2f})  sessions +{up}/-{down}  "
          f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
