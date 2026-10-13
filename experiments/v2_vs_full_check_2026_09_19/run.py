"""Why does the full 39k corpus score HIGHER than v2 in a few cells? A like-for-like check.

Every v2 window records `source_index`, its row in the session's labelled_epochs_ds.npz, so each v2 window
can be paired with the SAME samples in the full corpus. For those paired windows the monitor is scored twice:

  A. with the stream history rebuilt from v2's own windows (the published v2 numbers), and
  B. with the stream history of the full, continuous 39k corpus.

The samples, the labels (v2's) and the thresholds are identical, so any difference comes only from the
history. The rest of the gap between the datasets comes from WHICH windows each one contains.

It also checks that the pairing is exact (same samples) and that the labels agree.
usage: python run.py
"""
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\gaura\thesis_rtamt")
CL = ["LEFT", "RIGHT", "DOWN", "STRAIGHT"]
P = dict(b=6.61, g=7.81, d=0.208, q=2.72, n=-0.661, v=7.54)


def predict(f):
    valid = f["background_valid"] >= 0.5
    loud = np.where(valid, f["background"] >= P["b"], f["contrast"] >= P["g"])
    move = loud & (f["delta"] >= P["d"]) & (f["deflect"] >= P["q"])
    nod = (f["nod"] >= P["n"]) & (f["sideways"] <= P["v"])
    return np.where(~move, "STRAIGHT", np.where(nod, "DOWN", np.where(f["turn"] > 0, "LEFT", "RIGHT")))


def main():
    v2 = dict(np.load(ROOT / "measurements_v2.npz", allow_pickle=True))
    full = dict(np.load(ROOT / "experiments/labelled_epochs_ds_monitor_2026_09_19/measurements_39k.npz", allow_pickle=True))
    full_key = {(s, int(r)): i for i, (s, r) in enumerate(zip(full["session"].astype(str), full["row"]))}
    v2_session = v2["session"].astype(str)

    # pair every v2 window with its row in the full corpus, and check the samples are the same
    pair = np.zeros(len(v2_session), int)
    same_samples = label_agree = 0
    checked = 0
    for s in np.unique(v2_session):
        d = np.load(ROOT / "extracted_npz_dataset_v2" / s / "clean_epochs.npz", allow_pickle=True)
        src = d["source_index"].astype(int)
        idx = np.flatnonzero(v2_session == s)
        for j, k in enumerate(idx):
            pair[k] = full_key[(s, int(src[j]))]
        full_file = np.load(ROOT / "dataset/eeg_cleaned_data" / s / "labelled_epochs_ds.npz", allow_pickle=True)
        for j in range(0, len(src), 50):                       # spot-check the samples
            a = d["epochs"][j, :8]
            b = full_file["epochs"][src[j], :8]
            b = b - b.mean(axis=1, keepdims=True)
            same_samples += int(np.abs(a - b).max() < 1e-10)
            checked += 1
    full_label = np.array([str(x).upper() for x in full["label"][pair]])
    label_agree = int(np.sum(full_label == v2["label"].astype(str)))
    print(f"pairing: {len(pair)} v2 windows -> full-corpus rows; identical samples in {same_samples}/{checked} checked;"
          f" the two label sets agree on {label_agree}/{len(pair)} windows")
    print(f"usable stream history (background gate active): v2's own history "
          f"{100 * np.mean(v2['background_valid'] >= 0.5):.1f}%  |  full continuous stream "
          f"{100 * np.mean(full['background_valid'][pair] >= 0.5):.1f}%\n")

    label = v2["label"].astype(str)
    pa = predict(v2)                                      # A: v2's own history
    paired = {k: full[k][pair] for k in ("contrast", "background", "background_valid", "delta", "deflect",
                                          "nod", "sideways", "turn")}
    pb = predict(paired)                                  # B: the full stream's history, same windows

    dev = np.array([s.startswith("sub_01") and int(s[-2:]) <= 30 for s in v2_session])
    splits = (("sub_01 1-30", dev), ("sub_01 31-36", np.char.startswith(v2_session, "sub_01") & ~dev),
              ("sub_11", np.char.startswith(v2_session, "sub_11")))
    print("the SAME v2 windows and labels, scored with two different histories")
    print(f"{'':14s}{'history':26s}{'balanced':>10s}" + "".join(f"{k:>18s}" for k in CL))
    for name, m in splits:
        for tag, pred in (("A: rebuilt from v2", pa), ("B: full continuous stream", pb)):
            bal = 100 * np.mean([np.mean(pred[m & (label == k)] == k) for k in CL])
            cells = "".join(f"{int(((pred == k) & m & (label == k)).sum()):>7d}/{int((m & (label == k)).sum()):<5d}"
                            f"{100 * np.mean(pred[m & (label == k)] == k):5.1f}%" for k in CL)
            print(f"{name:14s}{tag:26s}{bal:9.2f}%{cells}")
        changed = m & (pa != pb)
        print(f"{'':14s}windows whose verdict changes: {int(changed.sum())} "
              f"(fixed {int((changed & (pb == label)).sum())}, broken {int((changed & (pa == label)).sum())})\n")


if __name__ == "__main__":
    main()
