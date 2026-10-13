"""Resting-state adaptation and Riemannian covariance features (extracted_npz_dataset_v2).

The session's own REST REFERENCE is built without any label: the 20% of the session's windows with the
lowest movement energy are pooled. That is what 10-30 s of "sit still" at the start of a session would give,
so it is deployable; it does use data from the session, which the project's rules otherwise avoid, and that
is stated wherever the results are reported.

METHOD 1 -- resting-state normalisation. The monitor's own measurements, but every channel divided by its
level in the rest reference instead of by its level inside the window:
    rest_deflect   mean over channels of (largest excursion / that channel's rest spread)
    rest_nod       midline-versus-temporal balance + F3/F4 symmetry, on rest-normalised channels
    rest_sideways  largest F4-T3 excursion / that pair's rest spread
    rest_turn      signed largest F4-T3 deflection / that pair's rest spread
    rest_energy    window energy / rest energy, pooled over channels (a rest-referenced gate)

METHOD 2 -- Riemannian tangent space. For each window the 8x8 covariance C of the 0.3-6 Hz signal is
whitened by the session's rest covariance R and mapped to the tangent space:
    S = logm(R^-1/2 C R^-1/2),  features = the 36 upper-triangular entries (tan_00 ... tan_77)
Whitening by R is exactly the "align every subject to its own rest state" idea; the logarithm makes the
result a vector that ordinary methods can use. Scale changes cancel; a sign flip of a channel leaves the
covariance unchanged, so these features are polarity-invariant by construction.

usage: python features.py
"""
import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, logm
from scipy.signal import lfilter

ROOT = Path(r"C:\Users\gaura\thesis_rtamt")
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
QUIET_SHARE = 0.20            # the quietest fifth of the session's windows becomes the rest reference


def whiten(reference):
    values, vectors = eigh(reference)
    values = np.maximum(values, 1e-12)
    return vectors @ np.diag(values ** -0.5) @ vectors.T


def session_job(path):
    from monitoring.rtamt_signals import _BAND_CLASS, _BAND_DIRECTION, _BAND_PROMINENCE, _IDX, FS, RtamtConfig, _pooled_energy
    cfg = RtamtConfig()
    data = np.load(path, allow_pickle=True)
    epochs = data["epochs"][:, :8] * 1e6
    n = len(epochs)
    bands = {"cls": _BAND_CLASS, "pro": _BAND_PROMINENCE, "dir": _BAND_DIRECTION}
    filtered = {k: lfilter(b[0], b[1], epochs, axis=-1) for k, b in bands.items()}
    energy = np.array([float(np.mean(_pooled_energy(x))) for x in epochs])

    # ---- the rest reference: the quietest windows of this session, chosen without labels
    quiet = np.argsort(energy)[:max(int(QUIET_SHARE * n), 5)]
    rest_cls = filtered["cls"][quiet].transpose(1, 0, 2).reshape(8, -1)
    rest_pro = filtered["pro"][quiet].transpose(1, 0, 2).reshape(8, -1)
    rest_dir = filtered["dir"][quiet].transpose(1, 0, 2).reshape(8, -1)
    spread_cls = 1.4826 * np.median(np.abs(rest_cls - np.median(rest_cls, axis=1, keepdims=True)), axis=1) + 1e-12
    rest_energy = float(np.mean(np.mean(rest_cls ** 2, axis=1)))

    def pair_spread(sig, a, b):
        v = sig[_IDX[a]] - sig[_IDX[b]]
        return 1.4826 * float(np.median(np.abs(v - np.median(v)))) + 1e-12
    spread_pair_pro = pair_spread(rest_pro, "F4", "T3")
    spread_pair_dir = pair_spread(rest_dir, "F4", "T3")

    reference = np.cov(filtered["dir"][quiet].transpose(1, 0, 2).reshape(8, -1)) + 1e-9 * np.eye(8)
    W = whiten(reference)
    rows = []
    triu = np.triu_indices(8)
    for k in range(n):
        cls = filtered["cls"][k] - np.median(filtered["cls"][k], axis=1, keepdims=True)
        row = {}
        row["rest_deflect"] = float(np.mean(np.abs(cls).max(axis=1) / spread_cls))
        scaled = cls / spread_cls[:, None]
        sq = scaled ** 2
        balance = np.log((sq[[_IDX["Fz"], _IDX["Cz"]]].mean() + 1e-18) / (sq[[_IDX["T3"], _IDX["T4"]]].mean() + 1e-18))
        f3, f4 = scaled[_IDX["F3"]], scaled[_IDX["F4"]]
        symmetry = float(np.log((np.mean((f3 + f4) ** 2) + 1e-18) / (np.mean((f3 - f4) ** 2) + 1e-18)))
        row["rest_nod"] = float(balance) + symmetry
        v = filtered["pro"][k][_IDX["F4"]] - filtered["pro"][k][_IDX["T3"]]
        row["rest_sideways"] = float(np.abs(v - np.median(v)).max() / spread_pair_pro)
        u = filtered["dir"][k][_IDX["F4"]] - filtered["dir"][k][_IDX["T3"]]
        u = u - np.median(u)
        row["rest_turn"] = float(-u[int(np.argmax(np.abs(u)))] / spread_pair_dir)
        row["rest_energy"] = float(np.mean(np.mean(cls ** 2, axis=1)) / (rest_energy + 1e-18))

        cov = np.cov(filtered["dir"][k]) + 1e-9 * np.eye(8)
        tangent = logm(W @ cov @ W)
        tangent = np.real((tangent + tangent.T) / 2)
        for i, (a, b) in enumerate(zip(*triu)):
            row[f"tan_{a}{b}"] = float(tangent[a, b])
        rows.append(row)
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    path = Path(path)
    out["session"] = np.array([f"{path.parent.parent.name}/{path.parent.name}"] * n)
    out["label"] = data["labels"].astype(str)
    return out


if __name__ == "__main__":
    files = sorted(glob.glob(str(ROOT / "extracted_npz_dataset_v2" / "*" / "ses-*" / "clean_epochs.npz")),
                   key=lambda p: (p.split(os.sep)[-3], p.split(os.sep)[-2]))
    with Pool(6) as pool:
        parts = pool.map(session_job, files)
    res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez(HERE / "features.npz", **res)
    print("windows", len(res["session"]), "| features", len([k for k in res if k not in ("session", "label")]))
