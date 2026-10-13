"""End-to-end four-class result of chosen families: blocked CV on sub_01 1-20, then the held-out sets ONCE.

    python e2e.py DETECT=<family> NOD=<family> DIRECTION=<family> [...more combos]
Each argument group is one combination; unspecified stages use the baseline.
"""
import sys

import numpy as np
from group_select import (FOLD, LAB, SES, TEST01, TEST11, TRAIN, NUM, S01, build, class_weights, fit_family,
                          best_rules, group_pool, grouping_pool, P)

# family@pool: restrict the rule pool to one of your groups / groupings
POOLS = {"all8": "Fp1+Fp2+F3+Fz+F4+T3+Cz+T4", "left": "Fp1+F3+T3", "right": "Fp2+F4+T4", "back": "T3+Cz+T4",
         "T3T4": ("T3", "T4"), "F3F4": ("F3", "F4"), "LR": ("Fp1+F3+T3", "Fp2+F4+T4"), "F4T3": ("F4", "T3"),
         "Fp1Fp2": ("Fp1", "Fp2")}


def fit(st, fam, tr):
    w = np.zeros(len(st.rows))
    w[tr] = class_weights(st.lab[tr])
    if fam == "baseline":
        return fit_family(st, fam, tr, w, (np.zeros(st.X.shape[1]),) * 3)
    pool = None
    if "@" in fam:
        fam, key = fam.split("@")
        spec = POOLS[key]
        pool = group_pool(st.names, spec) if isinstance(spec, str) else grouping_pool(st.names, *spec)
        o, t, p = best_rules(st.X[tr][:, pool], st.y[tr], w[tr])
        rank = tuple(np.full(st.X.shape[1], -1.0) for _ in range(3))
        for a, b in zip(rank, (o, t, p)):
            a[pool] = b
        return fit_family(st, fam, tr, w, rank, pool)
    return fit_family(st, fam, tr, w, best_rules(st.X[tr], st.y[tr], w[tr]))


def assemble(stages, choice, train_mask_rows):
    """Four-class predictions for every window, stages fitted on rows where train_mask_rows(st) is true;
    returns (pred over all windows, description)."""
    det, nod, dir_ = stages
    pred = np.full(len(LAB), "STRAIGHT", dtype=object)
    desc = {}
    moved = np.zeros(len(LAB), bool)
    out = {}
    for st in stages:
        tr = np.flatnonzero(train_mask_rows(st))
        fn, d = fit(st, choice.get(st.name, "baseline"), tr)
        out[st.name] = fn
        desc[st.name] = d
    return out, desc


def predict_all(stages, fns, which):
    det, nod, dir_ = stages
    pred = np.full(len(LAB), "STRAIGHT", dtype=object)
    rows = np.flatnonzero(which)
    moved = fns["DETECT"](rows)
    pred[rows[moved]] = "MOVE"                           # a rest window called a movement: wrong whatever class
    pos = {r: i for i, r in enumerate(nod.rows)}
    mv = [r for r in rows[moved] if r in pos]
    if mv:
        idx = np.array([pos[r] for r in mv])
        is_nod = fns["NOD"](idx)
        pred[np.array(mv)[is_nod]] = "DOWN"
        dpos = {r: i for i, r in enumerate(dir_.rows)}
        turn_rows = np.array(mv)[~is_nod]
        for r in turn_rows:
            pred[r] = "TURN?"                            # a DOWN window called a turn: wrong whatever side
        known = np.array([r for r in turn_rows if r in dpos])
        if len(known):
            left = fns["DIRECTION"](np.array([dpos[r] for r in known]))
            pred[known] = np.where(left, "LEFT", "RIGHT")
    return pred


def balanced(pred, m):
    return 100 * np.mean([np.mean(pred[m & (LAB == c)] == c) for c in ("LEFT", "RIGHT", "DOWN", "STRAIGHT")
                          if (m & (LAB == c)).any()])


def false_alarms(pred, m):
    rest = m & (LAB == "STRAIGHT")
    return int((pred[rest] != "STRAIGHT").sum()), int(rest.sum())


def run(stages, choice, test=True):
    cvpred = np.full(len(LAB), "STRAIGHT", dtype=object)
    for k in range(3):
        fns, _ = assemble(stages, choice, lambda st, k=k: st.train & (st.fold != k))
        held = TRAIN & (FOLD == k)
        p = predict_all(stages, fns, held)
        cvpred[held] = p[held]
    line = f"CV 1-20 {balanced(cvpred, TRAIN):6.2f} (false alarms {false_alarms(cvpred, TRAIN)[0]})"
    better = worse = 0
    if test:
        fns, desc = assemble(stages, choice, lambda st: st.train)
        p = predict_all(stages, fns, ~TRAIN)
        for name, m in (("21-36", TEST01), ("31-36", S01 & (NUM >= 31)), ("sub_11", TEST11)):
            fa = false_alarms(p, m)
            line += f" | {name} {balanced(p, m):6.2f} (FA {fa[0]}/{fa[1]})"
        return line, desc, cvpred
    return line, None, cvpred


if __name__ == "__main__":
    stages = build()[:3]
    combos = [dict(a.split("=") for a in arg.split(",")) for arg in sys.argv[1:]] or [{}]
    base_line, _, base_cv = run(stages, {})
    print("baseline (window-only, thresholds re-selected):", base_line, flush=True)
    for choice in combos:
        if not choice:
            continue
        line, desc, cvp = run(stages, choice)
        better = worse = 0
        for s in np.unique(SES[TRAIN]):
            m = TRAIN & (SES == s)
            d = balanced(cvp, m) - balanced(base_cv, m)
            better += d > 1e-9
            worse += d < -1e-9
        print(f"\n{choice}: {line} | CV sessions better {better} / worse {worse}")
        for k, v in desc.items():
            print(f"   {k}: {v}")
