"""The two parts of evaluate.py that were interrupted: T3 on raw, T1 on ICA-cleaned."""
from evaluate import source, run, line, P, SPLITS, LAB
for src, tech, name in (("raw", "T3", "T3 re-selected + blink veto"), ("ica", "T1", "T1 thresholds re-selected (ICA)")):
    f = source(src)
    base = P.predict(f)
    if src == "ica":
        print(line(f, "baseline ICA (current thresholds)", base, base), flush=True)
    cv, pred, model = run(f, tech)
    print(line(f, name, cv, pred, base), flush=True)
    print(f"   final model: thresholds {model[0]}, veto {model[1]}", flush=True)
