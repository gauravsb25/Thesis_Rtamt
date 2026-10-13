"""Three techniques on RAW data (what a headset gives), same rules, same protocol as project_code.py.

    T1  thresholds re-selected on raw windows of sub_01 1-20 (project_code.search, blocked CV 1-7/8-14/15-20)
    T2  blink veto: a window judged a movement becomes STRAIGHT when its loudest 0.5 s is forehead-dominant
        (fp_dom >= a) and short (event_w <= w); a, w selected on the training folds
    T3  T1 + T2 together
Baseline: the current default thresholds (selected on ICA-cleaned sub_01 1-20). Every window has its REAL
preceding 5 s of recording as history. Adoption rule: CV gain >= +0.5 and more sessions better than worse,
then the held-out sets (sub_01 21-36, sub_11) once.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import numpy as np
import project_code as P

M = np.load(HERE / "measurements_raw_ica.npz", allow_pickle=True)
LAB, SES = M["label"].astype(str), M["session"].astype(str)
NUM = np.array([int(s[-2:]) for s in SES])
S01 = np.char.startswith(SES, "sub_01")
DEV = S01 & (NUM <= 20)
SPLITS = (("CV/fit 1-20", DEV), ("sub_01 21-36", S01 & (NUM >= 21)), ("sub_01 31-36", S01 & (NUM >= 31)),
          ("sub_11", np.char.startswith(SES, "sub_11")))
A_GRID = np.round(np.arange(-1.0, 2.01, 0.1), 2)
W_GRID = (0.15, 0.2, 0.3, 0.4, np.inf)


def source(src):
    f = {k: M[f"{src}_{k}"] for k in P.KEYS + ("fp_dom", "event_w")}
    f.update(label=LAB, session=SES, dev=DEV)
    return f


def veto(f, pred, a, w):
    out = pred.copy()
    out[(pred != "STRAIGHT") & (f["fp_dom"] >= a) & (f["event_w"] <= w)] = "STRAIGHT"
    return out


def pick_veto(f, p, train):
    base = P.predict(f, p)
    best = (P.balanced(f, base, train), None)
    for a in A_GRID:
        for w in W_GRID:
            s = P.balanced(f, veto(f, base, a, w), train)
            if s > best[0] + 1e-9:
                best = (s, (a, w))
    return best[1]


def run(f, technique):
    """Out-of-fold predictions on 1-20 and the final model fitted on all of 1-20."""
    def fit(train):
        p = P.search(f, train) if technique in ("T1", "T3") else dict(P.SHIPPED)
        v = pick_veto(f, p, train) if technique in ("T2", "T3") else None
        return p, v

    def apply(model):
        p, v = model
        pred = P.predict(f, p)
        return veto(f, pred, *v) if v else pred

    cv = np.full(len(LAB), "STRAIGHT", dtype=object)
    for fold in P.FOLDS:
        held = np.isin(SES, fold)
        cv[held] = apply(fit(DEV & ~held))[held]
    final = fit(DEV)
    return cv, apply(final), final


def line(f, name, cv, pred, base_cv=None):
    out = [f"{name:34s}"]
    for split, m in SPLITS:
        pr = cv if split.startswith("CV") else pred
        rest = m & (LAB == "STRAIGHT")
        out.append(f"{split} {P.balanced(f, pr, m):6.2f} (FA {int((pr[rest] != 'STRAIGHT').sum())})")
    if base_cv is not None:
        better = worse = 0
        for s in np.unique(SES[DEV]):
            d = P.balanced(f, cv, SES == s) - P.balanced(f, base_cv, SES == s)
            better += d > 1e-9
            worse += d < -1e-9
        out.append(f"sessions +{better}/-{worse}")
    return " | ".join(out)


if __name__ == "__main__":
    for src in ("raw", "ica"):
        f = source(src)
        base = P.predict(f)
        print(f"\n===== {src.upper()} recordings, real history")
        print(line(f, "baseline (current thresholds)", base, base))
        for tech in ("T2", "T1", "T3") if src == "raw" else ("T1",):
            cv, pred, model = run(f, tech)
            print(line(f, {"T1": "T1 thresholds re-selected", "T2": "T2 blink veto",
                           "T3": "T3 re-selected + blink veto"}[tech], cv, pred, base), flush=True)
            print(f"   final model: thresholds {model[0]}, veto (fp_dom >=, width <=) {model[1]}", flush=True)
            for split, m in SPLITS[1:]:
                cells = "  ".join(f"{c} {int((pred[m & (LAB == c)] == c).sum())}/{int((m & (LAB == c)).sum())}"
                                  for c in P.CLASSES)
                print(f"   {split:13s} {cells}")
