import sys; sys.path.insert(0, r"C:\Users\gaura\thesis_rtamt")
import numpy as np, time
from pathlib import Path
from feat import features, load_session
DATA = Path(r"C:\Users\gaura\thesis_rtamt\extracted_npz_dataset_v2")
f = dict(np.load(r"C:\Users\gaura\thesis_rtamt\measurements_v2.npz", allow_pickle=True))
t0=time.time(); allf = []
for p in sorted(DATA.glob("*/ses-*/clean_epochs.npz")):
    d, X = load_session(p)
    allf.append(features(X))
F = {k: np.concatenate([a[k] for a in allf]) for k in allf[0]}
print("time", time.time()-t0)
for k in ("contrast","delta","deflect","nod","sideways","turn"):
    a, b = F[k], f[k]
    rel = np.abs(a-b)/(np.abs(b)+1e-9)
    print(f"{k:9s} max abs diff {np.abs(a-b).max():.3e}  max rel {rel.max():.3e}  sign agree {np.mean(np.sign(a)==np.sign(b)) if k=='turn' else ''}")
np.savez("feat_base.npz", **F)
