import pickle, numpy as np, matplotlib, sys
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
runs = pickle.load(open("runs.pkl","rb")); mv = pickle.load(open("movements.pkl","rb"))
ses = sys.argv[1]
CH = ["Fp1","Fp2","F3","Fz","F4","T3","Cz","T4"]
b, a = butter(2, [0.3/128, 6/128], btype="band")
rows = [r for r in mv if r[0].startswith(ses)]
fig, axes = plt.subplots(len(rows), 1, figsize=(14, 2.3*len(rows)), sharex=True)
for ax, r in zip(axes, rows):
    v = runs[r[0]]; x = lfilter(b, a, v["sig"].astype(float), axis=-1)
    on = int(round(v["onset"])); lo = max(0, on-256); hi = min(x.shape[1], on+2*256)
    t = (np.arange(lo, hi)-on)/256
    for ci, ch in enumerate(CH): ax.plot(t, x[ci, lo:hi], lw=1.1, ls="-" if ci < 5 else "--", label=ch)
    ax.axvline(0, c="k", lw=.5); ax.axvline(v["dur"]/256, c="k", lw=.5, ls=":")
    ax.set_title(f"{r[0]} true {r[1]} pred {r[2]}  nod {r[4]:.2f} bal {r[5]:.2f} sym {r[6]:.2f} side {r[7]:.2f} turn {r[8]:.2f}", fontsize=9)
    ax.grid(alpha=.3)
axes[0].legend(ncol=8, fontsize=7)
plt.tight_layout(); plt.savefig(f"ses_{ses.replace('/','_')}.png", dpi=62)
