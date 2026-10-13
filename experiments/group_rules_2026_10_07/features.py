"""Window-only electrode-GROUP features (no stream history), for every window of extracted_npz_dataset_v2.

Electrodes: Fp1 Fp2 F3 Fz F4 T3 Cz T4. A GROUP is any non-empty subset (255). A COMPARISON is any pair of
disjoint non-empty groups (A, B): 3,025 unordered pairs; this includes left {Fp1,F3,T3} vs right {Fp2,F4,T4},
front {Fp1,Fp2,F3,Fz,F4} vs {T3,Cz,T4}, midline {Fz,Cz} vs temples {T3,T4}, and T3 vs T4.

Per comparison, in two bands (0.3-6 Hz "dir", 0.5-3 Hz "det"), eight unit-free measurements:
    peak_raw   signed largest excursion of mean(A) - mean(B) (uV) in the loudest 1.5 s, over its own spread
    peak_z     the same after each channel is divided by its own spread
    prom_raw   largest |mean(A) - mean(B)| over the whole window, over its own spread
    eratio_raw log(energy of A / energy of B) in the loudest 1 s
    eratio_z   the same on spread-normalised channels
    cm_ratio   log(common-mode / differential energy) of mean(A), mean(B) in the loudest 1 s
    corr       correlation of mean(A) and mean(B) in the loudest 1.5 s
    lag        time of the largest |mean(A)| minus that of |mean(B)| in the loudest 1.5 s (s)
Pair features are computed for MOVEMENT windows only (they serve the nod and direction decisions).

Per single group, for ALL windows (the "did it move" decision):
    g_contrast log(loudest / quietest 0.5-s energy), spread-normalised channels, 0.5-3 Hz
    g_deflect  mean over the group of each channel's largest excursion / its spread, 0.5-6 Hz
    g_delta    rise of the group's 1-4 Hz power share across 0.5-s slices
All filters are the monitor's causal 2nd-order Butterworth filters. Nothing uses earlier windows.
"""
import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.signal import lfilter
from monitoring.rtamt_signals import (CHANNELS, FS, _BAND_CLASS, _BAND_DETECT, _BAND_DIRECTION, _block_sums,
                                      _pooled_energy)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
DATA = ROOT / "extracted_npz_dataset_v2"
W = 3 * FS
SUBSETS = [s for r in range(1, 9) for s in itertools.combinations(range(8), r)]          # 255
SUBSET_ID = {s: k for k, s in enumerate(SUBSETS)}
AVG = np.zeros((len(SUBSETS), 8))
for k, s in enumerate(SUBSETS):
    AVG[k, list(s)] = 1.0 / len(s)
PAIRS = [(SUBSET_ID[a], SUBSET_ID[b]) for a in SUBSETS for b in SUBSETS
         if SUBSET_ID[a] < SUBSET_ID[b] and not set(a) & set(b)]                          # 3025
PA = np.array([p[0] for p in PAIRS])
PB = np.array([p[1] for p in PAIRS])
PAIR_TYPES = ("peak_raw", "peak_z", "prom_raw", "eratio_raw", "eratio_z", "cm_ratio", "corr", "lag")
BANDS = {"dir": _BAND_DIRECTION, "det": _BAND_DETECT}
GROUP_TYPES = ("g_contrast", "g_deflect", "g_delta")


def subset_name(k):
    return "+".join(CHANNELS[i] for i in SUBSETS[k])


def pair_feature_names():
    return [f"{t}|{b}|{subset_name(a)}|{subset_name(c)}" for b in BANDS for t in PAIR_TYPES for a, c in PAIRS]


def group_feature_names():
    return [f"{t}|{subset_name(k)}" for t in GROUP_TYPES for k in range(len(SUBSETS))]


def _mad(a, axis=-1):
    return 1.4826 * np.median(np.abs(a - np.median(a, axis=axis, keepdims=True)), axis=axis) + 1e-12


def group_features(x):
    det = lfilter(*_BAND_DETECT, x, axis=-1)
    det = det - np.median(det, axis=1, keepdims=True)
    z = det / _mad(det)[:, None]
    e = AVG @ (z ** 2)                                           # (255, 768) group energy
    blocks = np.stack([_block_sums(row, FS // 2) for row in e])
    contrast = np.log((blocks.max(axis=1) + 1e-18) / (blocks.min(axis=1) + 1e-18))
    cls = lfilter(*_BAND_CLASS, x, axis=-1)
    cls = cls - np.median(cls, axis=1, keepdims=True)
    prominence = np.abs(cls).max(axis=1) / (1.4826 * np.median(np.abs(cls), axis=1) + 1e-12)
    deflect = AVG @ prominence
    width, step = FS // 2, FS // 4
    taper = np.hanning(width)
    freq = np.fft.rfftfreq(width, 1.0 / FS)
    slow, total = (freq >= 1.0) & (freq < 4.0), (freq >= 0.5) & (freq <= 45.0)
    shares = []
    for s in range(0, W - width + 1, step):
        seg = x[:, s:s + width] - x[:, s:s + width].mean(axis=1, keepdims=True)
        p = np.abs(np.fft.rfft(seg * taper, axis=1)) ** 2
        shares.append(p[:, slow].sum(axis=1) / (p[:, total].sum(axis=1) + 1e-18))
    g = AVG @ np.array(shares).T                                  # (255, slices)
    delta = g.max(axis=1) - g.min(axis=1)
    return np.concatenate([contrast, deflect, delta]).astype(np.float32)


def pair_features(x):
    energy = _pooled_energy(x)
    s15 = int(np.argmax(_block_sums(energy, 3 * FS // 2)))
    s1 = int(np.argmax(_block_sums(energy, FS)))
    out = []
    for band in BANDS.values():
        y = lfilter(*band, x, axis=-1)
        y = y - np.median(y, axis=1, keepdims=True)
        z = y / _mad(y)[:, None]
        m_raw, m_z = AVG @ y, AVG @ z
        d_raw, d_z = m_raw[PA] - m_raw[PB], m_z[PA] - m_z[PB]
        feats = {}
        for name, d in (("peak_raw", d_raw), ("peak_z", d_z)):
            seg = d[:, s15:s15 + 3 * FS // 2]
            seg = seg - np.median(seg, axis=1, keepdims=True)
            peak = seg[np.arange(len(seg)), np.argmax(np.abs(seg), axis=1)]
            feats[name] = peak / (1.4826 * np.median(np.abs(seg), axis=1) + 1e-12)
        c = d_raw - np.median(d_raw, axis=1, keepdims=True)
        feats["prom_raw"] = np.abs(c).max(axis=1) / (1.4826 * np.median(np.abs(c), axis=1) + 1e-12)
        e_raw = AVG @ np.mean(y[:, s1:s1 + FS] ** 2, axis=1)
        e_z = AVG @ np.mean(z[:, s1:s1 + FS] ** 2, axis=1)
        feats["eratio_raw"] = np.log((e_raw[PA] + 1e-18) / (e_raw[PB] + 1e-18))
        feats["eratio_z"] = np.log((e_z[PA] + 1e-18) / (e_z[PB] + 1e-18))
        ga, gb = m_raw[PA][:, s1:s1 + FS], m_raw[PB][:, s1:s1 + FS]
        feats["cm_ratio"] = np.log((np.mean((ga + gb) ** 2, axis=1) + 1e-18) / (np.mean((ga - gb) ** 2, axis=1) + 1e-18))
        ga, gb = m_raw[PA][:, s15:s15 + 3 * FS // 2], m_raw[PB][:, s15:s15 + 3 * FS // 2]
        ga, gb = ga - ga.mean(axis=1, keepdims=True), gb - gb.mean(axis=1, keepdims=True)
        feats["corr"] = (ga * gb).sum(axis=1) / (np.sqrt((ga ** 2).sum(axis=1) * (gb ** 2).sum(axis=1)) + 1e-18)
        feats["lag"] = (np.argmax(np.abs(ga), axis=1) - np.argmax(np.abs(gb), axis=1)) / FS
        out.extend(feats[t] for t in PAIR_TYPES)
    return np.concatenate(out).astype(np.float32)


def session(path):
    d = np.load(path, allow_pickle=True)
    x = d["epochs"][:, :8].astype(float) * 1e6
    labels = d["labels"].astype(str)
    g = np.stack([group_features(w) for w in x])
    move = np.flatnonzero(labels != "STRAIGHT")
    p = np.stack([pair_features(x[k]) for k in move]) if len(move) else np.zeros((0, len(PAIRS) * 16), np.float32)
    return g, p, move


if __name__ == "__main__":
    files = sorted(DATA.glob("*/ses-*/clean_epochs.npz"))
    if len(sys.argv) > 1 and sys.argv[1] == "--time":
        import time
        t = time.time()
        g, p, move = session(str(files[0]))
        print(f"one session: {len(g)} windows, {len(move)} movement, {time.time() - t:.1f} s; "
              f"group {g.shape}, pair {p.shape}")
        sys.exit()
    with Pool(6) as pool:
        parts = pool.map(session, [str(f) for f in files])
    sizes = np.cumsum([0] + [len(p[0]) for p in parts])
    G = np.concatenate([p[0] for p in parts])
    MOVE = np.concatenate([p[2] + sizes[i] for i, p in enumerate(parts)])
    PF = np.concatenate([p[1] for p in parts])
    np.save(CACHE / "group_features.npy", G)
    np.save(CACHE / "pair_features.npy", PF)
    np.save(CACHE / "pair_rows.npy", MOVE)
    print(f"saved: group {G.shape}, pair {PF.shape} ({PF.nbytes / 1e6:.0f} MB)")
