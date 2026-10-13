"""Electrode-group rules, rule combinations and context switching -- window-only (no stream history).

Three decisions, as in the monitor: DETECT (moved or STRAIGHT), NOD (DOWN or a turn), DIRECTION (LEFT or RIGHT).
For each decision, every family below is fitted on sub_01 sessions 1-20 with blocked 3-fold CV
(1-7 / 8-14 / 15-20); nothing is copied from earlier rounds, the baseline thresholds are re-selected in
every fold. Each rule is "feature >= t" or "feature <= t" on one unit-free window feature.

    baseline    the current window-only rule (contrast/delta/deflect; nod/sideways; sign of turn)
    single      the best single rule of the whole pool
    and_k/or_k  AND / OR of the k best mutually different rules (k = 2, 3)
    vote_n      majority of the n best mutually different rules (n = 3, 5, 9, 15, 25)
    base_and    baseline AND one extra rule (a veto);  base_or: baseline OR one extra rule (a rescue)
    regime      CONTEXT SWITCHING: one window property splits the windows in two; each side gets its own
                best rule (or the baseline refitted there) -- (c and rule1) or (not c and rule2)
    cascade     CONFIDENCE SWITCHING: ranked rules; a rule is switched off for a window that lies close
                to its threshold (normalised margin < c) and the next rule decides
Null control: the same single-rule search with labels shuffled inside each session.
Scores are class-balanced exactly as in the four-class balanced accuracy.
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import numpy as np
import project_code as P
import features as FEAT

RNG = np.random.default_rng(7)
f = P.measure()
LAB, SES = f["label"], f["session"]
N = len(LAB)
NUM = np.array([int(s[-2:]) for s in SES])
S01 = np.char.startswith(SES, "sub_01")
TRAIN, TEST01, TEST11 = S01 & (NUM <= 20), S01 & (NUM >= 21), np.char.startswith(SES, "sub_11")
FOLD = np.full(N, -1)
for k, sessions in enumerate(P.FOLDS):
    FOLD[np.isin(SES, sessions)] = k

G = np.load(HERE / "cache" / "group_features.npy")
PF = np.load(HERE / "cache" / "pair_features.npy", mmap_mode="r")
PROWS = np.load(HERE / "cache" / "pair_rows.npy")
GNAMES, PNAMES = FEAT.group_feature_names(), FEAT.pair_feature_names()


# ------------------------------------------------------------------ rule search
def class_weights(lab):
    classes = sorted(set(lab))
    w = np.zeros(len(lab))
    for c in classes:
        w[lab == c] = 1.0 / (len(classes) * (lab == c).sum())
    return w


def best_rules(X, y, w, chunk=1500):
    """Best threshold and polarity for every column: (objective, threshold, polarity)."""
    n, F = X.shape
    obj, thr, pol = np.empty(F), np.empty(F), np.empty(F)
    wp, wn = w * y, w * ~y
    Wp, Wn = wp.sum(), wn.sum()
    for a in range(0, F, chunk):
        Xc = np.asarray(X[:, a:a + chunk], dtype=np.float64)
        order = np.argsort(Xc, axis=0, kind="stable")
        xs = np.take_along_axis(Xc, order, 0)
        cp = np.vstack([np.zeros((1, Xc.shape[1])), np.cumsum(wp[order], axis=0)])
        cn = np.vstack([np.zeros((1, Xc.shape[1])), np.cumsum(wn[order], axis=0)])
        ge = Wp - cp + cn                       # x >= t is positive, split after k smallest
        le = cp + Wn - cn                       # x <= t is positive
        valid = np.ones_like(ge, dtype=bool)
        valid[1:-1] = xs[1:] > xs[:-1]          # no split between equal values
        ge[~valid], le[~valid] = -1, -1
        kg, kl = ge.argmax(axis=0), le.argmax(axis=0)
        cols = np.arange(Xc.shape[1])
        use_ge = ge[kg, cols] >= le[kl, cols]
        k = np.where(use_ge, kg, kl)
        lo = np.where(k > 0, xs[np.maximum(k - 1, 0), cols], xs[0, cols] - 1)
        hi = np.where(k < n, xs[np.minimum(k, n - 1), cols], xs[-1, cols] + 1)
        obj[a:a + chunk] = np.where(use_ge, ge[kg, cols], le[kl, cols])
        thr[a:a + chunk] = (lo + hi) / 2
        pol[a:a + chunk] = np.where(use_ge, 1, -1)
    return obj, thr, pol


def apply(x, t, p):
    return x >= t if p > 0 else x <= t


def score(pred, y, w):
    return float(np.sum(w * (pred == y)))


def diverse(X, order, thr, pol, k, tr, limit=0.95):
    chosen, preds = [], []
    for j in order:
        pj = apply(X[tr, j], thr[j], pol[j])
        if all(np.mean(pj == q) < limit for q in preds):
            chosen.append(j)
            preds.append(pj)
        if len(chosen) == k:
            break
    return chosen


# ------------------------------------------------------------------ stages
class Stage:
    def __init__(self, name, rows, y, X, names, base_cols, base_kind, regime_cols):
        self.name, self.rows, self.y, self.X, self.names = name, rows, y, X, names
        self.lab, self.ses = LAB[rows], SES[rows]
        self.fold, self.train = FOLD[rows], TRAIN[rows]
        self.base_cols, self.base_kind, self.regime_cols = base_cols, base_kind, regime_cols

    # baseline: re-selected thresholds of the current rule
    def fit_baseline(self, tr, w):
        X, y = self.X, self.y
        if self.base_kind == "sign":
            return ("sign",)
        if self.base_kind == "detect":
            c, d, q = (X[:, j] for j in self.base_cols)
            grid = (np.arange(3.0, 11.01, 0.5), np.arange(0.08, 0.361, 0.02), np.arange(2.0, 3.61, 0.1))
            C = np.array([c[tr] >= g for g in grid[0]])
            D = np.array([d[tr] >= g for g in grid[1]])
            Q = np.array([q[tr] >= g for g in grid[2]])
            best = (-1, None)
            for i, cc in enumerate(C):
                for j, dd in enumerate(D):
                    cd = cc & dd
                    s = ((cd[None, :] & Q) == y[tr][None, :]).astype(float) @ w[tr]
                    m = int(np.argmax(s))
                    if s[m] > best[0]:
                        best = (s[m], (grid[0][i], grid[1][j], grid[2][m]))
            return ("detect", best[1])
        nod, side = (X[:, j] for j in self.base_cols)
        best = (-1, None)
        V = np.array([side[tr] <= v for v in np.arange(5.0, 11.01, 0.25)])
        for n in np.arange(-2.0, 1.01, 0.1):
            s = ((nod[tr] >= n)[None, :] & V) == y[tr][None, :]
            s = s.astype(float) @ w[tr]
            m = int(np.argmax(s))
            if s[m] > best[0]:
                best = (s[m], (n, np.arange(5.0, 11.01, 0.25)[m]))
        return ("nod", best[1])

    def predict_baseline(self, model, idx):
        X = self.X
        if model[0] == "sign":
            return X[idx, self.base_cols[0]] > 0
        if model[0] == "detect":
            g, d, q = model[1]
            c0, c1, c2 = self.base_cols
            return (X[idx, c0] >= g) & (X[idx, c1] >= d) & (X[idx, c2] >= q)
        n, v = model[1]
        return (X[idx, self.base_cols[0]] >= n) & (X[idx, self.base_cols[1]] <= v)


def fit_family(st, fam, tr, w, rank, pool=None):
    """Fit one family on training rows tr; returns a predict(idx) function and a description."""
    X, y = st.X, st.y
    obj, thr, pol = rank
    order = np.argsort(-obj) if pool is None else pool[np.argsort(-obj[pool])]
    if fam == "baseline":
        m = st.fit_baseline(tr, w)
        return (lambda idx: st.predict_baseline(m, idx)), f"baseline {m}"
    if fam == "single":
        j = order[0]
        return (lambda idx: apply(X[idx, j], thr[j], pol[j])), f"{st.names[j]} {'>=' if pol[j] > 0 else '<='} {thr[j]:.4g}"
    if fam[:3] in ("and", "or_", "vot"):
        kind, k = fam.split("_")
        js = diverse(X, order[:3000], thr, pol, int(k), tr)
        need = {"and": len(js), "or": 1, "vote": len(js) // 2 + 1}[kind]

        def predict(idx, js=js, need=need):
            votes = sum(apply(X[idx, j], thr[j], pol[j]).astype(int) for j in js)
            return votes >= need
        return predict, f"{fam}: " + "; ".join(st.names[j] for j in js)
    if fam in ("base_and", "base_or"):
        bm = st.fit_baseline(tr, w)
        b = st.predict_baseline(bm, tr)
        best = (-1, None)
        for j in order[:300]:
            r = apply(X[tr, j], thr[j], pol[j])
            s = score(b & r if fam == "base_and" else b | r, y[tr], w[tr])
            if s > best[0]:
                best = (s, j)
        j = best[1]

        def predict(idx, j=j, bm=bm):
            r = apply(X[idx, j], thr[j], pol[j])
            b = st.predict_baseline(bm, idx)
            return b & r if fam == "base_and" else b | r
        return predict, f"{fam}: baseline {bm} with {st.names[j]}"
    if fam == "regime":
        top = order[:2000]
        best = (-1, None)
        for r in st.regime_cols:
            for q in (0.25, 0.5, 0.75):
                cut = np.quantile(X[tr, r], q)
                total, parts = 0.0, []
                for side in (X[tr, r] >= cut, X[tr, r] < cut):
                    t = tr[side]
                    if len(t) < 30 or len(set(y[t])) < 2:
                        total = -1
                        break
                    o, th, po = best_rules(X[np.ix_(t, top)], y[t], w[t])
                    m = int(np.argmax(o))
                    bm = st.fit_baseline(t, w)
                    bs = score(st.predict_baseline(bm, t), y[t], w[t])
                    if bs >= o[m]:
                        parts.append(("base", bm))
                        total += bs
                    else:
                        parts.append(("rule", (top[m], th[m], po[m])))
                        total += o[m]
                if total > best[0]:
                    best = (total, (r, cut, parts))
        r, cut, parts = best[1]

        def predict(idx, r=r, cut=cut, parts=parts):
            out = np.zeros(len(idx), bool)
            for side, part in zip((X[idx, r] >= cut, X[idx, r] < cut), parts):
                if part[0] == "base":
                    out[side] = st.predict_baseline(part[1], idx[side])
                else:
                    j, t, p = part[1]
                    out[side] = apply(X[idx[side], j], t, p)
            return out
        desc = [("baseline" if p[0] == "base" else st.names[p[1][0]]) for p in parts]
        return predict, f"regime: if {st.names[r]} >= {cut:.4g} use {desc[0]}, else {desc[1]}"
    if fam == "cascade":
        best = (-1, None)
        for n in (2, 3, 5):
            js = diverse(X, order[:3000], thr, pol, n, tr)
            iqr = [np.subtract(*np.quantile(X[tr, j], [0.75, 0.25])) + 1e-12 for j in js]
            for c in (0.25, 0.5, 1.0):
                def predict(idx, js=js, iqr=iqr, c=c):
                    out = apply(X[idx, js[0]], thr[js[0]], pol[js[0]])
                    done = np.zeros(len(idx), bool)
                    for j, s in zip(js, iqr):
                        m = pol[j] * (X[idx, j] - thr[j]) / s
                        sure = ~done & (np.abs(m) >= c)
                        out[sure] = m[sure] >= 0
                        done |= sure
                    return out
                s = score(predict(tr), y[tr], w[tr])
                if s > best[0]:
                    best = (s, (predict, f"cascade n={n} c={c}: " + "; ".join(st.names[j] for j in js)))
        return best[1]
    raise ValueError(fam)


FAMILIES = ["baseline", "single", "and_2", "and_3", "or_2", "or_3", "vote_3", "vote_5", "vote_9", "vote_15",
            "vote_25", "base_and", "base_or", "regime", "cascade"]


def cv(st, fams, pool=None):
    """Pooled out-of-fold predictions on sessions 1-20 for every family."""
    idx_all = np.flatnonzero(st.train)
    w_all = np.zeros(len(st.rows))
    w_all[idx_all] = class_weights(st.lab[idx_all])
    preds = {fam: np.zeros(len(st.rows), bool) for fam in fams}
    for k in range(3):
        tr = np.flatnonzero(st.train & (st.fold != k))
        va = np.flatnonzero(st.train & (st.fold == k))
        w = np.zeros(len(st.rows))
        w[tr] = class_weights(st.lab[tr])
        Xp = st.X[tr] if pool is None else st.X[tr][:, pool]
        o, t, p = best_rules(Xp, st.y[tr], w[tr])
        if pool is not None:
            O, T, Pp = (np.full(st.X.shape[1], -1.0) for _ in range(3))
            O[pool], T[pool], Pp[pool] = o, t, p
            o, t, p = O, T, Pp
        for fam in fams:
            fn, _ = fit_family(st, fam, tr, w, (o, t, p), pool)
            preds[fam][va] = fn(va)
    out = {}
    base = preds.get("baseline")
    for fam in fams:
        s = score(preds[fam][idx_all], st.y[idx_all], w_all[idx_all])
        better = worse = 0
        if base is not None and fam != "baseline":
            for ses in np.unique(st.ses[idx_all]):
                m = idx_all[st.ses[idx_all] == ses]
                wm = class_weights(st.lab[m])
                d = score(preds[fam][m], st.y[m], wm) - score(base[m], st.y[m], wm)
                better += d > 1e-9
                worse += d < -1e-9
        out[fam] = dict(cv=100 * s, better=int(better), worse=int(worse))
    return out, preds


def final_fit(st, fam, pool=None):
    tr = np.flatnonzero(st.train)
    w = np.zeros(len(st.rows))
    w[tr] = class_weights(st.lab[tr])
    Xp = st.X[tr] if pool is None else st.X[tr][:, pool]
    o, t, p = best_rules(Xp, st.y[tr], w[tr])
    if pool is not None:
        O, T, Pp = (np.full(st.X.shape[1], -1.0) for _ in range(3))
        O[pool], T[pool], Pp[pool] = o, t, p
        o, t, p = O, T, Pp
    return fit_family(st, fam, tr, w, (o, t, p), pool)


def shuffled_max(st, reps=5):
    tr = np.flatnonzero(st.train)
    w = class_weights(st.lab[tr])
    real = best_rules(st.X[tr], st.y[tr], w)[0].max()
    fake = []
    for _ in range(reps):
        y = st.y[tr].copy()
        lab = st.lab[tr].copy()
        for s in np.unique(st.ses[tr]):
            m = np.flatnonzero(st.ses[tr] == s)
            perm = RNG.permutation(m)
            y[m], lab[m] = y[perm], lab[perm]
        fake.append(best_rules(st.X[tr], y, class_weights(lab))[0].max())
    return 100 * real, [100 * x for x in fake]


# ------------------------------------------------------------------ build the three stages
def col(name):
    return f[name].astype(np.float32)


def build():
    det_X = np.hstack([G, np.stack([col("contrast"), col("delta"), col("deflect")], axis=1)])
    det_names = GNAMES + ["BASE contrast", "BASE delta", "BASE deflect"]
    nb = len(GNAMES)
    det_regime = [nb, nb + 2] + [det_names.index(n) for n in ("g_contrast|T3+T4", "g_contrast|Fp1+Fp2",
                                                            "g_contrast|Fz+Cz", "g_deflect|T3+T4")]
    det = Stage("DETECT", np.arange(N), LAB != "STRAIGHT", det_X, det_names, [nb, nb + 1, nb + 2], "detect", det_regime)

    pf = np.asarray(PF)
    extra = np.stack([col("nod")[PROWS], col("sideways")[PROWS], col("turn")[PROWS],
                      np.abs(col("turn")[PROWS])], axis=1)
    mov_X = np.hstack([pf, extra])
    del pf
    mov_names = PNAMES + ["BASE nod", "BASE sideways", "BASE turn", "BASE |turn|"]
    nb = len(PNAMES)
    regime_names = ["eratio_raw|dir|T3+T4|Fp1+Fp2+F3+Fz+F4+Cz", "eratio_raw|dir|Fp1+F3+T3|Fp2+F4+T4",
                    "eratio_raw|dir|Fp1+Fp2+F3+Fz+F4|T3+Cz+T4", "eratio_raw|dir|Fz+Cz|T3+T4"]
    lookup = {n: i for i, n in enumerate(mov_names)}

    def find(n):
        if n in lookup:
            return lookup[n]
        t, b, a, c = n.split("|")
        return lookup[f"{t}|{b}|{c}|{a}"]
    regime = [nb, nb + 1, nb + 3] + [find(n) for n in regime_names]
    nod = Stage("NOD", PROWS, LAB[PROWS] == "DOWN", mov_X, mov_names, [nb, nb + 1], "nod", regime)
    turns = np.flatnonzero(LAB[PROWS] != "DOWN")
    dir_ = Stage("DIRECTION", PROWS[turns], LAB[PROWS[turns]] == "LEFT", mov_X[turns], mov_names, [nb + 2], "sign",
                 regime)
    return det, nod, dir_, find


USER_GROUPINGS = {
    "left Fp1+F3+T3 vs right Fp2+F4+T4": ("Fp1+F3+T3", "Fp2+F4+T4"),
    "front Fp1+Fp2+F3+Fz+F4 vs T3+Cz+T4": ("Fp1+Fp2+F3+Fz+F4", "T3+Cz+T4"),
    "midline Fz+Cz vs temples T3+T4": ("Fz+Cz", "T3+T4"),
    "T3 vs T4": ("T3", "T4"),
    "F3 vs F4": ("F3", "F4"),
    "Fp1 vs Fp2": ("Fp1", "Fp2"),
    "F4 vs T3 (current direction pair)": ("F4", "T3"),
    "left+midline Fp1+F3+T3+Fz+Cz vs right Fp2+F4+T4": ("Fp1+F3+Fz+T3+Cz", "Fp2+F4+T4"),
}
USER_GROUPS = {"left Fp1+F3+T3": "Fp1+F3+T3", "right Fp2+F4+T4": "Fp2+F4+T4", "midline Fz+Cz": "Fz+Cz",
               "front Fp1+Fp2+F3+Fz+F4": "Fp1+Fp2+F3+Fz+F4", "T3+Cz+T4": "T3+Cz+T4", "all 8": "Fp1+Fp2+F3+Fz+F4+T3+Cz+T4"}


def grouping_pool(names, a, b):
    out = []
    for i, n in enumerate(names):
        parts = n.split("|")
        if len(parts) == 4 and {parts[2], parts[3]} == {a, b}:
            out.append(i)
    return np.array(out)


def group_pool(names, g):
    return np.array([i for i, n in enumerate(names) if n.split("|")[-1] == g and n.startswith("g_")])


if __name__ == "__main__":
    t0 = time.time()
    det, nod, dir_, find = build()
    report = {}
    for st in (det, nod, dir_):
        print(f"\n=== {st.name}: {int(st.train.sum())} training rows, {st.X.shape[1]} features "
              f"({time.time() - t0:.0f} s)", flush=True)
        res, preds = cv(st, FAMILIES)
        report[st.name] = {"families": res}
        for fam, r in res.items():
            print(f"   {fam:10s} CV {r['cv']:6.2f}   sessions better {r['better']:2d} / worse {r['worse']:2d}", flush=True)
        real, fake = shuffled_max(st)
        report[st.name]["null"] = {"real_best_train": real, "shuffled_best_train": fake}
        print(f"   null control: best single rule on training, real {real:.2f} vs shuffled "
              + ", ".join(f"{x:.2f}" for x in fake), flush=True)
        user = {}
        if st.name == "DETECT":
            for label, g in USER_GROUPS.items():
                pool = group_pool(st.names, g)
                r, _ = cv(st, ["single", "vote_3"], pool)
                user[label] = r
                print(f"   your group {label:28s} single CV {r['single']['cv']:6.2f}  vote3 {r['vote_3']['cv']:6.2f}", flush=True)
        else:
            for label, (a, b) in USER_GROUPINGS.items():
                pool = grouping_pool(st.names, a, b)
                r, _ = cv(st, ["single", "vote_3", "or_2", "and_2"], pool)
                user[label] = r
                print(f"   your grouping {label:40s} single {r['single']['cv']:6.2f}  vote3 {r['vote_3']['cv']:6.2f}"
                      f"  or2 {r['or_2']['cv']:6.2f}  and2 {r['and_2']['cv']:6.2f}", flush=True)
        report[st.name]["user"] = user
    (HERE / "cv_report.json").write_text(json.dumps(report, indent=1))
    print(f"\nsaved cv_report.json ({time.time() - t0:.0f} s)")
