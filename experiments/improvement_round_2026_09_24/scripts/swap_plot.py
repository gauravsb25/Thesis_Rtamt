import pickle, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import harness as H
f = H.load(); base = H.predict(f); lab = f["label"]
m = dict(np.load("meta.npz", allow_pickle=True)); F = dict(np.load("feat_base.npz"))
runs = pickle.load(open("runs.pkl","rb"))
key = np.array([f"{s}|{t}|{r}" for s,t,r in zip(m['session'], m['task'], m['run'])])
b, a = butter(2, [0.3/128, 6/128], btype="band")
CH = ["Fp1","Fp2","F3","Fz","F4","T3","Cz","T4"]; I = {c:i for i,c in enumerate(CH)}
from collections import Counter
sw, ok = [], []
for k in np.unique(key[f["dev"] & np.isin(lab, ["LEFT","RIGHT"])]):
    idx = np.flatnonzero((key == k) & (lab != "STRAIGHT"))
    pr = Counter(base[idx]).most_common(1)[0][0]; tr = lab[idx[0]]
    if {pr, tr} == {"LEFT","RIGHT"}: sw.append((k, idx))
    elif pr == tr: ok.append((k, idx))
rng = np.random.default_rng(0); pick = [ok[i] for i in rng.choice(len(ok), 5, replace=False)]
rows = sw + pick
fig, axes = plt.subplots(len(rows), 1, figsize=(13, 1.9*len(rows)), sharex=True)
for ax, (k, idx) in zip(axes, rows):
    v = runs[k]; x = lfilter(b, a, v["sig"].astype(float), axis=-1)
    on = int(round(v["onset"])); lo = max(0, on-256); hi = min(x.shape[1], on+2*256)
    t = (np.arange(lo, hi)-on)/256
    ax.plot(t, x[I["F4"], lo:hi]-x[I["T3"], lo:hi], lw=1.4, label="F4-T3")
    ax.plot(t, x[I["T3"], lo:hi]-x[I["T4"], lo:hi], lw=1.0, alpha=.7, label="T3-T4")
    ax.plot(t, x[I["Fp1"], lo:hi]-x[I["Fp2"], lo:hi], lw=1.0, alpha=.7, label="Fp1-Fp2")
    # loud 1.5 s block of the middle window of this movement (in run coordinates)
    j = idx[len(idx)//2]; s15 = int(F["loud15_start"][j]) + int(m["run_offset"][j])
    ax.axvspan((s15-on)/256, (s15+384-on)/256, color="orange", alpha=.12)
    ax.axvline(0, c="k", lw=.5); ax.axvline(v["dur"]/256, c="k", lw=.5, ls=":")
    ax.set_title(f"{k} true {lab[idx[0]]} pred {Counter(base[idx]).most_common(1)[0][0]}  median turn {np.median(f['turn'][idx]):.2f}", fontsize=8)
    ax.grid(alpha=.3)
axes[0].legend(fontsize=7, ncol=3)
plt.tight_layout(); plt.savefig("swaps.png", dpi=60)
print(len(sw), "swapped movements")
