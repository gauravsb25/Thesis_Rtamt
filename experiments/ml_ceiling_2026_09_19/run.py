"""How much is IN the windows? Machine learning used as a CEILING ESTIMATOR, not as a monitor.

The shipped monitors stay machine-learning free. Here a set of models is given every signal computed in
every round (about 60 per window) and asked how well it can do. The point is the comparison:

  * if the models cannot beat the monitors across subjects, the information is not in these 8 channels,
    which is the strongest form of the project's conclusion;
  * if they can, the rules are the limitation, and what the model uses can be turned into an STL rule.

PROTOCOL (the same discipline as everywhere else)
  * development: sub_01 sessions 1-30 only, 5-fold GROUPED cross-validation -- whole sessions are held out,
    never single windows, because neighbouring windows overlap by 0.1 s and would leak;
  * the two held-out sets (sub_01 31-36, sub_11) are scored once, with models trained on all of 1-30;
  * balanced accuracy everywhere, and a direction-only score on the turn windows;
  * models: logistic regression, random forest, gradient boosting, all with balanced class weights where
    available.

usage: python run.py               FINAL_TEST=1 python run.py test
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(r"C:\Users\gaura\thesis_rtamt")
HERE = Path(__file__).resolve().parent
CL = ["LEFT", "RIGHT", "DOWN", "STRAIGHT"]
SOURCES = {
    "base": ROOT / "measurements_v2.npz",
    "round2": ROOT / "_archive/experiment_scripts/round2_background_2026_09_17/features.npz",
    "proposed": ROOT / "experiments/proposed_methods_2026_09_18/features.npz",
    "phase": ROOT / "experiments/phase_shape_2026_09_18/features.npz",
    "fifteen": ROOT / "experiments/fifteen_methods_2026_09_18/features.npz",
    "paradigms": ROOT / "experiments/five_paradigms_2026_09_18/features.npz",
}
SKIP = {"session", "label", "index", "task", "run", "offset", "onset", "dev", "valid"}


def load():
    names, columns = [], []
    reference = None
    for tag, path in SOURCES.items():
        data = dict(np.load(path, allow_pickle=True))
        session = data["session"].astype(str)
        if reference is None:
            reference, label = session, data["label"].astype(str)
        else:
            assert (session == reference).all(), tag
        for key, value in sorted(data.items()):
            if key in SKIP or value.dtype.kind not in "fiu" or value.ndim != 1:
                continue
            name = key if tag == "base" else f"{tag}.{key}"
            if name not in names:
                names.append(name)
                columns.append(value.astype(float))
    X = np.column_stack(columns)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, label, reference, names


def models():
    return {
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                                n_jobs=-1, random_state=0),
        "gradient boosting": GradientBoostingClassifier(random_state=0),
    }


def balanced(truth, pred):
    return 100 * float(np.mean([np.mean(pred[truth == k] == k) for k in np.unique(truth)]))


def main():
    X, label, session, names = load()
    dev = np.array([s.startswith("sub_01") and int(s[-2:]) <= 30 for s in session])
    t1 = np.char.startswith(session, "sub_01") & ~dev
    s11 = np.char.startswith(session, "sub_11")
    turn = np.isin(label, ["LEFT", "RIGHT"])
    print(f"{X.shape[1]} signals per window, {len(label)} windows "
          f"(design {int(dev.sum())}, sub_01 31-36 {int(t1.sum())}, sub_11 {int(s11.sum())})")
    print("monitors for comparison: 4-class 85.01 / 69.62 / 78.67 ; direction 92.6 / 100.0 / 70.5\n")

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if os.environ.get("FINAL_TEST") != "1":
            raise SystemExit("needs FINAL_TEST=1")
        print("trained on all of sub_01 sessions 1-30, scored once on the held-out sets\n")
        print(f"{'model':22s}{'4-class 31-36':>15s}{'4-class sub_11':>16s}{'direction 31-36':>17s}{'direction sub_11':>18s}")
        for name, model in models().items():
            model.fit(X[dev], label[dev])
            pred = model.predict(X)
            cells = [balanced(label[t1], pred[t1]), balanced(label[s11], pred[s11])]
            direction = []
            for m in (t1, s11):
                k = m & turn
                dmodel = models()[name]
                dmodel.fit(X[dev & turn], label[dev & turn])
                direction.append(balanced(label[k], dmodel.predict(X[k])))
            print(f"{name:22s}{cells[0]:14.2f}%{cells[1]:15.2f}%{direction[0]:16.1f}%{direction[1]:17.1f}%")
        return

    print("5-fold grouped cross-validation inside sub_01 sessions 1-30 (whole sessions held out)\n")
    print(f"{'model':22s}{'4-class':>10s}{'direction only':>17s}")
    splitter = GroupKFold(n_splits=5)
    for name, model in models().items():
        pred = np.empty(int(dev.sum()), dtype=object)
        Xd, yd, gd = X[dev], label[dev], session[dev]
        for train, test in splitter.split(Xd, yd, groups=gd):
            model.fit(Xd[train], yd[train])
            pred[test] = model.predict(Xd[test])
        four = balanced(yd, np.array(pred, dtype=str))
        td = turn[dev]
        dpred = np.empty(int(td.sum()), dtype=object)
        Xt, yt, gt = Xd[td], yd[td], gd[td]
        for train, test in splitter.split(Xt, yt, groups=gt):
            model.fit(Xt[train], yt[train])
            dpred[test] = model.predict(Xt[test])
        print(f"{name:22s}{four:9.2f}%{balanced(yt, np.array(dpred, dtype=str)):16.1f}%", flush=True)

    forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                    n_jobs=-1, random_state=0).fit(X[dev], label[dev])
    order = np.argsort(forest.feature_importances_)[::-1][:12]
    print("\nwhat the random forest leans on (4-class, trained on sessions 1-30):")
    for i in order:
        print(f"   {names[i]:34s}{100 * forest.feature_importances_[i]:5.1f}%")
    forest_dir = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                        n_jobs=-1, random_state=0).fit(X[dev & turn], label[dev & turn])
    order = np.argsort(forest_dir.feature_importances_)[::-1][:8]
    print("\nwhat it leans on for LEFT versus RIGHT:")
    for i in order:
        print(f"   {names[i]:34s}{100 * forest_dir.feature_importances_[i]:5.1f}%")


if __name__ == "__main__":
    main()
