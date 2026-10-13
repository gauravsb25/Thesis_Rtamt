"""Sweep 704 RTAMT-operator features through every role in the monitor, with a noise floor.

704 features = 44 signals (8 channels, 28 channel pairs, 8 channel energies) x 16 operator
reductions (once, historically, bounded once over thirds, rise counts, since, dwell, first/last
crossing, biphasic order and separation). verify.py proves each reduction equals real RTAMT.

ROLES each feature is tried in
  gate+   the movement gate also needs the feature      MOVE and (f >= t)
  gate-   the gate also accepts a loud feature          MOVE or  (rest-checks and (f >= t))
  nod+    DOWN also needs the feature                   NOD and (f >= t)
  nod-    DOWN when the nod rule holds OR the feature   NOD or   (f >= t and no big swing)
  dir     LEFT/RIGHT from the sign of the feature       (orientation fixed on training only)
  dirtie  the feature decides only where |turn| is weak

THE NOISE FLOOR -- why this experiment needs one

Searching 704 features x 6 roles x 15 thresholds means ~63,000 candidates per fold. The best of
63,000 noise variables will beat the baseline by SOME margin even when every feature is
meaningless. So the same search is repeated on data whose features have been shuffled within each
session (the labels, the monitor and the class balance are untouched, only the feature-to-window
link is broken). The best CV gain from those runs is the gain this search produces from luck
alone. A real candidate has to beat it, not just beat +0.5.

usage: python run.py               python run.py --null 10
       FINAL_TEST=1 python run.py test "<feature>|<role>"
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
ROLES = ["gate+", "gate-", "nod+", "nod-", "dir", "dirtie"]
NLEVEL = 15

sys.path.insert(0, str(HERE))
from signals import OPS as OPDOC                                          # noqa: E402

M = dict(np.load(ROOT / "measurements_v2.npz", allow_pickle=True))
B = dict(np.load(ROOT / "experiments/operator_sweep_2026_09_21/bank.npz", allow_pickle=True))
LAB, SES = M["label"].astype(str), M["session"].astype(str)
assert (B["session"].astype(str) == SES).all() and (B["label"].astype(str) == LAB).all()
FEATS = sorted(k for k in B if k not in ("session", "label"))
X = np.stack([B[k].astype(np.float64) for k in FEATS])          # (n_features, n_windows)

Y = np.searchsorted(np.array(CL), LAB)                          # ints, but CL is not sorted:
Y = np.array([CL.index(v) for v in LAB])
DEV = np.array([s.startswith("sub_01") and int(s[-2:]) <= 30 for s in SES])
T1 = np.char.startswith(SES, "sub_01") & ~DEV
S11 = np.char.startswith(SES, "sub_11")
VALID = M["background_valid"] >= 0.5
LOUD = np.where(VALID, M["background"] >= P["b"], M["contrast"] >= P["g"])
REST = (M["delta"] >= P["d"]) & (M["deflect"] >= P["q"])
MOVE = LOUD & REST
NOD = (M["nod"] >= P["n"]) & (M["sideways"] <= P["v"])
LEFTISH = M["turn"] > 0
NOSWING = M["sideways"] <= P["v"]
WEAK = np.abs(M["turn"])


def assemble(move, nod, leftish):
    return np.where(~move, 3, np.where(nod, 2, np.where(leftish, 0, 1)))


BASE = assemble(MOVE, NOD, LEFTISH)


def bal_idx(mask):
    """index arrays per class, for a fast balanced accuracy"""
    return [np.flatnonzero(mask & (Y == k)) for k in range(4)]


def bal(pred, idx):
    return 100.0 * float(np.mean([(pred[i] == k).mean() for k, i in enumerate(idx) if i.size]))


def apply_role(role, f, t, orient=1.0):
    if role == "gate+":
        return assemble(MOVE & (f >= t), NOD, LEFTISH)
    if role == "gate-":
        return assemble(MOVE | (REST & (f >= t)), NOD, LEFTISH)
    if role == "nod+":
        return assemble(MOVE, NOD & (f >= t), LEFTISH)
    if role == "nod-":
        return assemble(MOVE, NOD | ((f >= t) & NOSWING), LEFTISH)
    if role == "dir":
        return assemble(MOVE, NOD, (f * orient) > 0)
    if role == "dirtie":
        return assemble(MOVE, NOD, np.where(WEAK >= t, LEFTISH, (f * orient) > 0))
    raise ValueError(role)


def search(x, train, idx=None, report=None, only=None):
    """best (gain, feature, role, threshold, orientation) over every candidate, fitted on `train`

    ``only``: restrict the search to one operator (the part after '|' in a feature name), so each
    operator can be judged on its own instead of inside one 198,000-candidate search.
    """
    idx = idx or bal_idx(train)
    base = bal(BASE, idx)
    lat = train & np.isin(Y, [0, 1]) & MOVE & ~NOD
    best = (0.0, None, None, None, 1.0)
    weak_grid = np.quantile(WEAK[train & MOVE & ~NOD], np.linspace(0.05, 0.95, NLEVEL))
    for j, name in enumerate(FEATS):
        if only is not None and name.split("|")[1] != only:
            continue
        f = x[j]
        for sign in (1.0, -1.0):
            g = f * sign
            for role in ROLES:
                if role == "dir":
                    o = float(np.sign(np.median(g[lat & (Y == 0)]) - np.median(g[lat & (Y == 1)])) or 1.0)
                    cand = [(None, bal(apply_role(role, g, None, o), idx), o)]
                elif role == "dirtie":
                    o = float(np.sign(np.median(g[lat & (Y == 0)]) - np.median(g[lat & (Y == 1)])) or 1.0)
                    cand = [(t, bal(apply_role(role, g, t, o), idx), o) for t in weak_grid]
                else:
                    sub = train & (MOVE if role.startswith("nod") else np.ones(len(Y), bool))
                    grid = np.quantile(g[sub], np.linspace(0.05, 0.95, NLEVEL))
                    cand = [(t, bal(apply_role(role, g, t), idx), 1.0) for t in np.unique(grid)]
                for t, a, o in cand:
                    if a - base > best[0] + 1e-9:
                        best = (a - base, f"{name}{'' if sign > 0 else ' (negated)'}", role, t, o)
                        if report is not None:
                            report.append(best)
            if role == "dir":
                pass
    return best


def cv_gain(x, verbose=False, only=None):
    """blocked 3-fold CV gain of the best candidate chosen inside each fold"""
    cv = BASE.copy()
    picks = []
    for k in range(3):
        held = np.isin(SES, FOLDS[k])
        train = DEV & ~held
        gain, name, role, t, o = search(x, train, only=only)
        if name is None:
            picks.append("current")
            continue
        j = FEATS.index(name.replace(" (negated)", ""))
        g = x[j] * (-1.0 if "negated" in name else 1.0)
        cv[held] = apply_role(role, g, t, o)[held]
        picks.append(f"{name}|{role}|{t if t is None else round(float(t), 4)}")
        if verbose:
            print(f"   fold {k + 1}: {picks[-1]}  (train gain {gain:+.2f})", flush=True)
    d = [bal(cv, bal_idx(SES == s)) - bal(BASE, bal_idx(SES == s)) for s in np.unique(SES[DEV])]
    idx = bal_idx(DEV)
    return bal(cv, idx) - bal(BASE, idx), sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d), picks


def shuffled(rng):
    """features shuffled within each session: base monitor, labels and balance all unchanged"""
    x = X.copy()
    for s in np.unique(SES):
        i = np.flatnonzero(SES == s)
        x[:, i] = x[:, rng.permutation(i)]
    return x


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if os.environ.get("FINAL_TEST") != "1":
            raise SystemExit("needs FINAL_TEST=1")
        name, role, t = sys.argv[2].split("|")
        j = FEATS.index(name.replace(" (negated)", ""))
        g = X[j] * (-1.0 if "negated" in name else 1.0)
        lat = DEV & np.isin(Y, [0, 1]) & MOVE & ~NOD
        o = float(np.sign(np.median(g[lat & (Y == 0)]) - np.median(g[lat & (Y == 1)])) or 1.0)
        pred = apply_role(role, g, None if t == "None" else float(t), o)
        print(f"\n{name} in role {role}, threshold {t}: fitted on all of sessions 1-30")
        for lbl, m in (("sessions 1-30", DEV), ("sub_01 31-36", T1), ("sub_11", S11)):
            i = bal_idx(m)
            cls = " ".join(f"{CL[k][0]} {100 * (pred[x] == k).mean():5.1f}" for k, x in enumerate(i) if x.size)
            cls0 = " ".join(f"{CL[k][0]} {100 * (BASE[x] == k).mean():5.1f}" for k, x in enumerate(i) if x.size)
            print(f"   {lbl:14s} current {bal(BASE, i):6.2f}% ({cls0})  ->  {bal(pred, i):6.2f}% ({cls})")
        return

    if "--combo" in sys.argv:
        # xor and iff between TWO different signals. Neither adds expressive power -- both are
        # definable from and/or/not, which the monitors already use -- but the monitors only ever
        # conjoin single predicates, so these combination shapes are untested.
        # A small PRE-SPECIFIED set, not a data-chosen one, to keep the search honest.
        want = [f"ch_{c}|{op}" for c in ("Fp1", "Fp2", "F3", "Fz", "F4", "T3", "Cz", "T4")
                for op in ("absmax", "run2")]
        want += [f"pr_{p}|{op}" for p in ("T3T4", "F4T3", "F3T4", "F3F4")
                 for op in ("absmax", "run2", "ford")]
        cols = [FEATS.index(w) for w in want if w in FEATS]
        print(f"{len(cols)} signals -> {len(cols) * (len(cols) - 1) // 2} pairs x (xor, iff) "
              f"x (nod+, nod-) x 5x5 thresholds\n")
        cv = BASE.copy()
        picks = []
        for k in range(3):
            held = np.isin(SES, FOLDS[k])
            train = DEV & ~held
            idx = bal_idx(train)
            base = bal(BASE, idx)
            best = (0.0, None)
            for ai in range(len(cols)):
                fa = X[cols[ai]]
                ga = np.quantile(fa[train & MOVE], np.linspace(0.15, 0.85, 5))
                for bi in range(ai + 1, len(cols)):
                    fb = X[cols[bi]]
                    gb = np.quantile(fb[train & MOVE], np.linspace(0.15, 0.85, 5))
                    for ta in ga:
                        pa = fa >= ta
                        for tb in gb:
                            pb = fb >= tb
                            for opn, pred in (("xor", pa ^ pb), ("iff", pa == pb)):
                                for role in ("nod+", "nod-"):
                                    n_ = (NOD & pred) if role == "nod+" else (NOD | (pred & NOSWING))
                                    a = bal(assemble(MOVE, n_, LEFTISH), idx)
                                    if a - base > best[0] + 1e-9:
                                        best = (a - base, (cols[ai], ta, cols[bi], tb, opn, role,
                                                           assemble(MOVE, n_, LEFTISH)))
            if best[1] is None:
                picks.append("current")
                continue
            ca, ta, cb, tb, opn, role, pred = best[1]
            cv[held] = pred[held]
            picks.append(f"{FEATS[ca]}>={ta:.3g} {opn} {FEATS[cb]}>={tb:.3g} [{role}]")
            print(f"   fold {k + 1}: {picks[-1]}  (train gain {best[0]:+.2f})", flush=True)
        d = [bal(cv, bal_idx(SES == s)) - bal(BASE, bal_idx(SES == s)) for s in np.unique(SES[DEV])]
        up, down = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
        i = bal_idx(DEV)
        print(f"\nxor/iff CV gain {bal(cv, i) - bal(BASE, i):+.2f}  sessions +{up}/-{down}")
        return

    if "--per-op" in sys.argv:
        ops = sorted({k.split("|")[1] for k in FEATS})
        print(f"each operator judged ALONE: 44 signals x {len(ROLES)} roles x 2 signs x {NLEVEL} "
              f"thresholds = {44 * len(ROLES) * 2 * NLEVEL:,} candidates per fold\n")
        print(f"{'operator':9s} {'CV gain':>8s} {'sess +/-':>9s}  what it stands for in STL")
        rows = []
        for op in ops:
            g, up, down, _ = cv_gain(X, only=op)
            rows.append((g, op, up, down))
            print(f"{op:9s} {g:+8.2f} {f'+{up}/-{down}':>9s}  {OPDOC.get(op, '')}", flush=True)
        print("\nranked:")
        for g, op, up, down in sorted(rows, reverse=True):
            print(f"   {g:+6.2f}  {op:9s} (+{up}/-{down})")
        return

    if "--null" in sys.argv:
        reps = int(sys.argv[sys.argv.index("--null") + 1])
        # --op X calibrates the SMALL per-operator search (7,920 candidates), which overfits far
        # less than the full one, so it needs its own floor.
        only = sys.argv[sys.argv.index("--op") + 1] if "--op" in sys.argv else None
        rng = np.random.default_rng(7)
        gains = []
        if only:
            print(f"noise floor for the per-operator search, operator '{only}'")
        for r in range(reps):
            g, up, down, _ = cv_gain(shuffled(rng), only=only)
            gains.append(g)
            print(f"null {r + 1:2d}/{reps}: best-of-search CV gain {g:+.2f}  (sessions +{up}/-{down})",
                  flush=True)
        a = np.array(gains)
        print(f"\nNOISE FLOOR from {reps} shuffles: mean {a.mean():+.2f}, max {a.max():+.2f}, "
              f"95th pct {np.percentile(a, 95):+.2f}")
        print("A real candidate must beat the MAX, not just the adoption rule.")
        return

    print(f"{len(FEATS)} features x {len(ROLES)} roles x 2 signs x {NLEVEL} thresholds "
          f"= {len(FEATS) * len(ROLES) * 2 * NLEVEL:,} candidates per fold")
    print(f"baseline on sessions 1-30: {bal(BASE, bal_idx(DEV)):.2f}%\n")
    print("blocked 3-fold CV, the winner chosen independently inside each fold:")
    gain, up, down, picks = cv_gain(X, verbose=True)
    print(f"\nCV gain {gain:+.2f}  sessions +{up}/-{down}")

    print("\nthe strongest single candidates fitted on ALL of sessions 1-30 (in-sample, for reading only):")
    report = []
    search(X, DEV, report=report)
    for g, name, role, t, o in report[-12:][::-1]:
        print(f"   {g:+6.2f}  {name:34s} {role:7s} t={t if t is None else round(float(t), 4)}")


if __name__ == "__main__":
    main()
