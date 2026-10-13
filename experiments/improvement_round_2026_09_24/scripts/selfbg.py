"""Within-window per-channel background: each channel's 0.5 s block energy over its own q-quantile block energy
inside the window, averaged over channels; max over time. Same filter/energy as the monitor's background."""
import numpy as np
from pathlib import Path
from scipy.signal import lfilter
from feat import B_DET, block_sums
DATA = Path(r"C:\Users\gaura\thesis_rtamt\extracted_npz_dataset_v2")
res = {f"selfbg_q{q}": [] for q in (10, 25, 50)}
for p in sorted(DATA.glob("*/ses-*/clean_epochs.npz")):
    X = np.load(p)["epochs"][:, :8] * 1e6
    det = lfilter(*B_DET, X, axis=-1)
    e = block_sums(det**2, 128) / 128              # (n, 8, 641) block energies, element k = block ending k+127
    for q in (10, 25, 50):
        ref = np.percentile(e[:, :, ::8], q, axis=-1, keepdims=True) + 1e-12
        res[f"selfbg_q{q}"].append(np.mean(e / ref, axis=1).max(-1))
np.savez("feat_selfbg.npz", **{k: np.concatenate(v) for k, v in res.items()})
print("ok")
