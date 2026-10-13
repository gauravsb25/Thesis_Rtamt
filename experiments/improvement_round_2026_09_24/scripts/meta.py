"""Gather per-window metadata (session, task, run, offsets, movement position) for all windows, in project order."""
import numpy as np
from pathlib import Path
ROOT = Path(r"C:\Users\gaura\thesis_rtamt")
DATA = ROOT / "extracted_npz_dataset_v2"
rows = {k: [] for k in ("session","label","task","run","run_offset","start_sample","movement_offset","movement_duration","source_index")}
for p in sorted(DATA.glob("*/ses-*/clean_epochs.npz")):
    d = np.load(p, allow_pickle=True)
    n = len(d["labels"])
    rows["session"] += [f"{p.parent.parent.name}/{p.parent.name}"] * n
    rows["label"] += list(d["labels"].astype(str))
    for k in ("task","run","run_offset","start_sample","movement_offset","movement_duration","source_index"):
        rows[k] += list(d[k])
out = {k: np.array(v) for k, v in rows.items()}
np.savez(Path(__file__).parent / "meta.npz", **out)
print({k: v.shape for k, v in out.items()})
