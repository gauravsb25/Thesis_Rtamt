"""Vectorised re-implementation of the monitor's end-of-window measurements + extra diagnostics.
Everything computed from the window alone (no history), exactly like compute_signals."""
import numpy as np
from scipy.signal import butter, lfilter
FS = 256; N = 768
CH = ("Fp1","Fp2","F3","Fz","F4","T3","Cz","T4"); I = {c:i for i,c in enumerate(CH)}
def band(lo, hi): return butter(2, [lo/(FS/2), hi/(FS/2)], btype="band")
B_DET, B_CLS, B_DIR, B_PROM = band(0.5,3.0), band(0.5,6.0), band(0.3,6.0), band(0.3,4.0)

def block_sums(e, w):
    c = np.concatenate([np.zeros(e.shape[:-1]+(1,)), np.cumsum(e, axis=-1)], axis=-1)
    return c[..., w:] - c[..., :-w]

def features(X):
    """X: (n, 8, 768) uV. Returns dict of per-window arrays."""
    n = X.shape[0]
    out = {}
    det = lfilter(*B_DET, X, axis=-1)
    spread = 1.4826*np.median(np.abs(det - np.median(det, axis=-1, keepdims=True)), axis=-1, keepdims=True) + 1e-12
    energy = np.mean((det/spread)**2, axis=1)            # (n, 768) pooled
    b05 = block_sums(energy, 128)
    out["contrast"] = b05.max(1) / (b05.min(1) + 1e-18)
    # delta
    width, step = 128, 64
    taper = np.hanning(width); freq = np.fft.rfftfreq(width, 1/FS)
    ind = (freq>=1)&(freq<4); intot = (freq>=0.5)&(freq<=45)
    starts = np.arange(0, N-width+1, step)
    segs = np.stack([X[:, :, s:s+width] for s in starts], axis=1)   # n, S, 8, w
    segs = segs - segs.mean(-1, keepdims=True)
    P = np.abs(np.fft.rfft(segs*taper, axis=-1))**2
    share = np.mean(P[..., ind].sum(-1) / (P[..., intot].sum(-1)+1e-18), axis=-1)  # n, S
    out["delta"] = share.max(1) - share.min(1)
    # deflect
    cls = lfilter(*B_CLS, X, axis=-1)
    cls = cls - np.median(cls, axis=-1, keepdims=True)
    sp = 1.4826*np.median(np.abs(cls), axis=-1) + 1e-12
    out["deflect"] = np.mean(np.abs(cls).max(-1)/sp, axis=1)
    # nod
    sq = cls**2
    mid = sq[:, [I["Fz"], I["Cz"]]].sum(1).mean(-1); tmp = sq[:, [I["T3"], I["T4"]]].sum(1).mean(-1)
    out["balance"] = np.log((mid+1e-18)/(tmp+1e-18))
    b1 = block_sums(energy, 256); s1 = b1.argmax(1)
    idx = s1[:, None] + np.arange(256)[None]
    F3 = np.take_along_axis(cls[:, I["F3"]], idx, 1); F4 = np.take_along_axis(cls[:, I["F4"]], idx, 1)
    out["symmetry"] = np.log((np.mean((F3+F4)**2, 1)+1e-18)/(np.mean((F3-F4)**2, 1)+1e-18))
    out["nod"] = out["balance"] + out["symmetry"]
    out["loud1_start"] = s1
    # sideways
    prom = lfilter(*B_PROM, X, axis=-1)
    v = prom[:, I["F4"]] - prom[:, I["T3"]]; v = v - np.median(v, -1, keepdims=True)
    out["sideways"] = np.abs(v).max(-1) / (1.4826*np.median(np.abs(v), -1) + 1e-12)
    # turn
    dr = lfilter(*B_DIR, X, axis=-1)
    pair = dr[:, I["F4"]] - dr[:, I["T3"]]
    b15 = block_sums(energy, 384); s15 = b15.argmax(1)
    idx = s15[:, None] + np.arange(384)[None]
    seg = np.take_along_axis(pair, idx, 1); seg = seg - np.median(seg, -1, keepdims=True)
    pk = seg[np.arange(n), np.abs(seg).argmax(1)]
    t = -pk/(1.4826*np.median(np.abs(seg), -1)+1e-12)
    out["turn"] = np.where(t==0, 1e-12, t)
    out["loud15_start"] = s15
    return out

def load_session(path):
    d = dict(np.load(path, allow_pickle=True))
    return d, d["epochs"][:, :8]*1e6
