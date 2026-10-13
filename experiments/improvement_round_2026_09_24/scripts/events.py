"""One record per movement run: full continuous run (uV), onset/duration in run samples, label, session.
Also verify the dataset windows are exact slices of the CSV runs (after mean removal)."""
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
ROOT = Path(r"C:\Users\gaura\thesis_rtamt")
DATA = ROOT / "extracted_npz_dataset_v2"
SRC = ROOT / "dataset" / "eeg_cleaned_data"
CH = ["Fp1","Fp2","F3","Fz","F4","T3","Cz","T4"]
runs = {}
maxerr = 0.0
for p in sorted(DATA.glob("*/ses-*/clean_epochs.npz")):
    d = dict(np.load(p, allow_pickle=True))
    subj, ses = p.parent.parent.name, p.parent.name
    lab = d["labels"].astype(str); task = d["task"].astype(str); run = d["run"].astype(int)
    ro = d["run_offset"].astype(int); mo = d["movement_offset"].astype(float); md = d["movement_duration"].astype(float)
    for (t, r) in sorted(set(zip(task, run))):
        idx = np.flatnonzero((task == t) & (run == r))
        sub = subj.replace("_", "-")
        csv = SRC / subj / ses / "ica_eeg" / f"{sub}_{ses}_task-{t}_run-{r:02d}_eeg.csv"
        x = pd.read_csv(csv)
        sig = x[CH].to_numpy().T * 1e6
        marker = x["marker"].to_numpy()
        mv = idx[lab[idx] != "STRAIGHT"]
        onset = dur = np.nan; mlabel = "NONE"
        if len(mv):
            ons = ro[mv] + mo[mv] * 256
            onset = float(np.median(ons)); dur = float(np.median(md[mv]))
            assert np.ptp(ons) < 2, (csv, np.ptp(ons))
            mlabel = lab[mv][0]
            assert (lab[mv] == mlabel).all()
        # verify a few windows are exact slices
        for k in idx[:: max(1, len(idx)//5)]:
            w = d["epochs"][k, :8] * 1e6
            s = sig[:, ro[k]:ro[k] + 768]
            e = np.abs((s - s.mean(1, keepdims=True)) - w).max()
            maxerr = max(maxerr, e)
        cue = np.flatnonzero(marker == 7006)
        runs[f"{subj}/{ses}|{t}|{r}"] = dict(sig=sig.astype(np.float32), onset=onset, dur=dur, label=mlabel, task=t,
                                             session=f"{subj}/{ses}", cue=(int(cue[0]) if len(cue) else -1),
                                             n_windows=len(idx), labels_present=sorted(set(lab[idx])))
print("runs", len(runs), "max slice error uV", maxerr)
import pickle
pickle.dump(runs, open("runs.pkl", "wb"))
from collections import Counter
print(Counter(v["label"] for v in runs.values()))
print(Counter((v["session"][:6], v["label"]) for v in runs.values()))
