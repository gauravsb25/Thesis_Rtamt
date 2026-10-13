"""Build extracted_npz_dataset_v2_preica: the SAME windows as extracted_npz_dataset_v2, cut from the
pre-ICA recordings (pp_eeg: filtered only, eye and muscle components NOT removed).

Every v2 window is a verbatim slice of one ICA run CSV (run, run_offset). For each window this script
  1. loads that run's ica_eeg CSV and checks that the slice, with its own mean removed (as in v2),
     equals the v2 epoch (worst sample <= 1e-10 V),
  2. loads the pp_eeg CSV with the identical filename, checks it has the same length and markers,
  3. cuts the same 768 samples from it and removes their mean, exactly as v2 does.
A window that fails any check stops the build. Labels, timing fields and order are copied unchanged,
so every accuracy on the two datasets is window-for-window comparable.

usage: python -m pipeline.build_preica_windows --source D:/errp/eeg_cleaned_data
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "extracted_npz_dataset_v2"
OUT = ROOT / "extracted_npz_dataset_v2_preica"
CH = ["Fp1", "Fp2", "F3", "Fz", "F4", "T3", "Cz", "T4"]


def run_csv(folder, session_dir, task, run):
    hits = sorted(folder.glob(f"*task-{task}_run-{int(run):02d}_eeg.csv"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{folder}: task {task} run {run}: {len(hits)} files")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="D:/errp/eeg_cleaned_data")
    args = ap.parse_args()
    src = Path(args.source)
    manifest = {"source": str(src), "sessions": {}}
    worst_all = 0.0
    for f in sorted(V2.glob("sub_*/ses-*/clean_epochs.npz")):
        sub, ses = f.parent.parent.name, f.parent.name
        d = dict(np.load(f, allow_pickle=True))
        n = len(d["labels"])
        out = np.zeros((n, 9, 768))
        cache = {}
        worst = 0.0
        for key in sorted(set(zip(d["task"].astype(str), d["run"].astype(int)))):
            ica_path = run_csv(src / sub / ses / "ica_eeg", ses, *key)
            pp_path = src / sub / ses / "pp_eeg" / ica_path.name
            ica = pd.read_csv(ica_path)
            pp = pd.read_csv(pp_path)
            if len(ica) != len(pp) or not np.array_equal(ica["marker"].to_numpy(), pp["marker"].to_numpy()):
                raise ValueError(f"{pp_path}: length or markers differ from {ica_path}")
            cache[key] = (ica[CH].to_numpy(), pp[CH].to_numpy())
        for k in range(n):
            ica, pp = cache[(str(d["task"][k]), int(d["run"][k]))]
            o = int(d["run_offset"][k])
            ref = ica[o:o + 768].T
            diff = np.abs(ref - ref.mean(axis=1, keepdims=True) - d["epochs"][k, :8]).max()
            worst = max(worst, diff)
            if diff > 1e-10:
                raise ValueError(f"{sub}/{ses} window {k}: ICA slice differs from v2 by {diff:.3g} V")
            seg = pp[o:o + 768].T * 1e-6                        # pp_eeg is in microvolts; store volts like v2
            seg = seg - seg.mean(axis=1, keepdims=True)          # v2 windows have their own mean removed
            out[k, :8] = seg
            out[k, 8] = seg[CH.index("T3")] - seg[CH.index("T4")]
        worst_all = max(worst_all, worst)
        d["epochs"] = out
        (OUT / sub / ses).mkdir(parents=True, exist_ok=True)
        np.savez(OUT / sub / ses / "clean_epochs.npz", **d)
        manifest["sessions"][f"{sub}/{ses}"] = {"windows": n, "worst_ica_match_V": worst}
        print(f"{sub}/{ses}: {n} windows, ICA slice match worst {worst:.2e} V")
    manifest["worst_ica_match_V"] = worst_all
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (OUT / "README.md").write_text(
        "# extracted_npz_dataset_v2_preica\n\n"
        "The same 21,029 windows as `extracted_npz_dataset_v2` (same order, labels and timing fields), cut from the\n"
        "pre-ICA `pp_eeg` recordings instead of the ICA-cleaned `ica_eeg` ones. Every window's position was verified\n"
        f"against the ICA recording (worst difference {worst_all:.1e} V). Eye and muscle components are NOT removed.\n"
        "Each window has its own per-channel mean removed, as in v2.\n"
        "Built by `python -m pipeline.build_preica_windows`.\n")


if __name__ == "__main__":
    main()
