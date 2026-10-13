import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
runs = pickle.load(open("runs.pkl","rb"))
CH = ["Fp1","Fp2","F3","Fz","F4","T3","Cz","T4"]
b, a = butter(2, [0.3/128, 6/128], btype="band")
pre, post = int(1.0*256), int(2.5*256)
ep = {}
for k, v in runs.items():
    if v["label"] == "NONE": continue
    on = int(round(v["onset"])); sig = v["sig"].astype(float)
    if on - pre < 0 or on + post > sig.shape[1]: 
        # pad by edge
        pass
    x = filtfilt(b, a, sig, axis=-1)
    seg = np.full((8, pre+post), np.nan)
    lo, hi = max(0, on-pre), min(sig.shape[1], on+post)
    seg[:, lo-(on-pre):hi-(on-pre)] = x[:, lo:hi]
    seg -= np.nanmean(seg[:, :int(0.5*256)], axis=1, keepdims=True)   # baseline -1..-0.5 s
    sub = v["session"][:6]; ses = int(v["session"][-2:])
    grp = sub if sub=="sub_11" else ("sub_01dev" if ses<=30 else "sub_01test")
    ep.setdefault((grp, v["label"]), []).append(seg)
t = (np.arange(pre+post)-pre)/256
fig, axes = plt.subplots(3, 3, figsize=(18, 13), sharex=True)
for i, grp in enumerate(["sub_01dev","sub_01test","sub_11"]):
    for j, c in enumerate(["LEFT","RIGHT","DOWN"]):
        ax = axes[i, j]
        E = np.array(ep.get((grp, c), []))
        if not len(E): continue
        mu = np.nanmedian(E, axis=0)
        for ci, ch in enumerate(CH):
            ax.plot(t, mu[ci], label=ch, lw=1.4, ls="-" if ci<5 else "--")
        ax.set_title(f"{grp} {c} (n={len(E)}) median, 0.3-6 Hz zero-phase")
        ax.axvline(0, color="k", lw=0.5); ax.grid(alpha=0.3)
axes[0,0].legend(ncol=4, fontsize=8)
plt.tight_layout(); plt.savefig("erp_channels.png", dpi=70)
# pairs
fig, axes = plt.subplots(3, 3, figsize=(18, 13), sharex=True)
pairs = [("F4","T3"), ("T3","T4"), ("F3","F4"), ("Fp1","Fp2")]
for i, grp in enumerate(["sub_01dev","sub_01test","sub_11"]):
    for j, c in enumerate(["LEFT","RIGHT","DOWN"]):
        ax = axes[i, j]
        E = np.array(ep.get((grp, c), []))
        if not len(E): continue
        for (p, q) in pairs:
            d = E[:, CH.index(p)] - E[:, CH.index(q)]
            ax.plot(t, np.nanmedian(d, axis=0), label=f"{p}-{q}", lw=1.5)
        cm = E[:, CH.index("F3")] + E[:, CH.index("F4")]
        ax.plot(t, np.nanmedian(cm, axis=0)/2, label="(F3+F4)/2", lw=1.5, ls=":")
        ax.set_title(f"{grp} {c} pairs"); ax.axvline(0, color="k", lw=0.5); ax.grid(alpha=0.3)
axes[0,0].legend(fontsize=8)
plt.tight_layout(); plt.savefig("erp_pairs.png", dpi=70)
print("ok")
