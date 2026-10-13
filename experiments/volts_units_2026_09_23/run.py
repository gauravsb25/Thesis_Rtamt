"""Run the monitor on the dataset in VOLTS -- i.e. without the x1e6 conversion -- and compare.

`project_code.py` feeds the monitor `epochs[:, :8] * 1e6`, so it has only ever seen microvolts.
Every signal is a ratio, so a constant factor should cancel exactly. The one thing that can break
that is floating point: the signals carry small absolute guards (`+1e-12`, `+1e-18`) to avoid
dividing by zero, and at volt scale the numbers are a million times smaller, so those guards are no
longer negligible in the same way.

This computes the monitor's eight measurements for every window twice -- once in microvolts, once in
volts -- applies the shipped thresholds to both, and reports how many verdicts differ.
usage: python run.py
"""
import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\gaura\thesis_rtamt")
HERE = Path(__file__).resolve().parent
CL = ["LEFT", "RIGHT", "DOWN", "STRAIGHT"]
KEYS = ("contrast", "background", "background_valid", "delta", "deflect", "nod", "sideways", "turn")
P = dict(b=6.61, g=7.81, d=0.208, q=2.72, n=-0.661, v=7.54)


def session_job(path):
    import sys as _s
    if str(ROOT) not in _s.path:
        _s.path.insert(0, str(ROOT))
    from monitoring.rtamt_signals import RtamtConfig, compute_signals
    from monitoring.stream_history import windows_with_history
    cfg = RtamtConfig()
    data = np.load(path, allow_pickle=True)
    n = len(data["labels"])
    out = {f"{u}_{k}": np.zeros(n) for u in ("uv", "v") for k in KEYS}
    for i, window, history in windows_with_history(path, config=cfg):
        # windows_with_history already returns microvolts; volts = the same numbers / 1e6
        for tag, factor in (("uv", 1.0), ("v", 1e-6)):
            s = compute_signals(window * factor, cfg,
                                None if history is None else history * factor)
            for k in KEYS:
                out[f"{tag}_{k}"][i] = s[k].min() if k == "background_valid" else s[k].max() \
                    if k in ("contrast", "background", "delta", "deflect", "sideways") else s[k][-1]
    p = Path(path)
    out["session"] = np.array([f"{p.parent.parent.name}/{p.parent.name}"] * n)
    out["label"] = data["labels"].astype(str)
    return out


def predict(f, tag):
    valid = f[f"{tag}_background_valid"] >= 0.5
    loud = np.where(valid, f[f"{tag}_background"] >= P["b"], f[f"{tag}_contrast"] >= P["g"])
    move = loud & (f[f"{tag}_delta"] >= P["d"]) & (f[f"{tag}_deflect"] >= P["q"])
    nod = (f[f"{tag}_nod"] >= P["n"]) & (f[f"{tag}_sideways"] <= P["v"])
    return np.where(~move, "STRAIGHT", np.where(nod, "DOWN",
                    np.where(f[f"{tag}_turn"] > 0, "LEFT", "RIGHT")))


def main():
    cache = HERE / "both.npz"
    if cache.exists():
        f = dict(np.load(cache, allow_pickle=True))
    else:
        files = sorted(glob.glob(str(ROOT / "extracted_npz_dataset_v2" / "*" / "ses-*" / "clean_epochs.npz")),
                       key=lambda p: (p.split(os.sep)[-3], p.split(os.sep)[-2]))
        print(f"computing the monitor's signals twice for {len(files)} sessions ...", flush=True)
        with Pool(6) as pool:
            parts = pool.map(session_job, files)
        f = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez(cache, **f)

    lab, ses = f["label"].astype(str), f["session"].astype(str)
    uv, v = predict(f, "uv"), predict(f, "v")
    sess = np.array([int(s[-2:]) if s.startswith("sub_01") else -1 for s in ses])
    splits = [("sub_01 1-30 (design)", (sess >= 1) & (sess <= 30)),
              ("sub_01 31-36 (TEST)", np.char.startswith(ses, "sub_01") & (sess > 30)),
              ("sub_11 (TEST)", np.char.startswith(ses, "sub_11"))]

    def bal(pred, m):
        return 100.0 * float(np.mean([(pred[m & (lab == c)] == c).mean() for c in CL
                                      if (m & (lab == c)).any()]))

    print(f"\nverdicts that differ between microvolts and volts: "
          f"{int((uv != v).sum())} of {len(uv)}")
    print("\nlargest relative difference in each measurement:")
    for k in KEYS:
        a, b = f[f"uv_{k}"], f[f"v_{k}"]
        scale = np.maximum(np.abs(a), 1e-30)
        print(f"   {k:18s} {np.abs(a - b).max() / np.maximum(scale.max(), 1e-30):.3e}")

    print(f"\n{'split':24s}{'microvolts':>14s}{'volts':>12s}{'change':>10s}")
    for name, m in splits:
        a, b = bal(uv, m), bal(v, m)
        print(f"{name:24s}{a:13.2f}%{b:11.2f}%{b - a:+9.2f}")


if __name__ == "__main__":
    main()
