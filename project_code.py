"""ALL THE PROJECT'S WORKING CODE IN ONE FILE.

Read this first. Everything that is ACTIVE below runs on the one dataset that is kept,
`extracted_npz_dataset_v2`. Everything that was tried and NOT adopted is kept as COMMENTED code with
the numbers it produced, so nothing has to be searched for and nothing is repeated by mistake.

    python project_code.py evaluate      the results table: runs the REAL monitor (RTAMT) on every window.
                                         ~90 min the first time, then instant; recomputed automatically
                                         whenever monitoring/, the thresholds or the dataset change
    python project_code.py evaluate --detail   + the table for each window judged alone
    python project_code.py evaluate --fast     seconds: the arithmetic equivalent (follows threshold
                                         changes, NOT formula changes)
    python project_code.py evaluate --force    ignore the cache and recompute
    python project_code.py evaluate --offline  the OFFLINE monitor (monitoring/offline_monitor.py): every window is
                                         judged against the rest of its own recording, before and after it.
                                         ~20 min the first time, then cached. 86.73 / 71.80 / 79.37 (RTAMT, all windows)
    python project_code.py measure       cache the monitors' measurements for every window
    python project_code.py train         re-select the thresholds on sub_01 sessions 1-20 (blocked CV)
    python project_code.py train30       the previous default for comparison: thresholds selected on sessions 1-30
    FINAL_TEST=1 python project_code.py test    score the held-out sets once
    python project_code.py extract sub_12  cut a NEW subject's recordings into clean 3-s windows
    python project_code.py subjects      one accuracy table per subject (add --offline for the offline monitor)

WHAT THE MONITORS ARE:  monitoring/ (the deliverable, imported by 179 tests)
    rtamt_signals.py    one 3-second window -> the signals the specifications read
    rtamt_monitors.py   the four STL specifications, HeadMovementMonitor, StreamMonitor
    row_stream.py       a verdict at every incoming row
    stream_history.py   the stream before each stored window, rebuilt from the dataset itself
    threshold_report.py why each threshold has its value
    offline_monitor.py  the OFFLINE monitor: each window judged against its whole recording

THE DATASETS  (only the first one exists now; the others are rebuilt by the commented sections below)
    extracted_npz_dataset_v2        ACTIVE. 21,029 windows of 3 s (LEFT 1263, RIGHT 1367, DOWN 1156,
                                    STRAIGHT 17243), cut around the annotated movements, without the
                                    2,135 windows that had been spliced across two recordings.
                                    Accuracy: 85.01 / 69.62 / 78.67 (see `evaluate`).
    extracted_npz_dataset           the same windows including the spliced ones (82.30 / 66.28 / 76.40)
    extracted_npz_dataset_v2_preica the same windows cut from the pre-ICA recordings (blinks not removed)
    extracted_npz_dataset_v3_csv    windows cut from the raw CSVs with a cue marker + detector
    extracted_npz_dataset_trials    ONE window per trial, 442 in total, movement found from the signal
    The full original scripts are preserved under `_archive/pipeline/`.

THE SPLITS used everywhere: design = sub_01 sessions 1-20; held-out = sub_01 sessions 21-36 and sub_11.
(Default since 2026-10-07. Before that the design was sessions 1-30 and the thresholds
 b 6.61, g 7.81, d 0.208, q 2.72, n -0.661, v 7.54, offline b 4.95; see `train_first_30` below.)
"""

# ================================================================================================
# ================================================================================================
from __future__ import annotations

import hashlib
import itertools
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from monitoring.rtamt_signals import RtamtConfig, compute_signals            # noqa: E402
from monitoring.stream_history import windows_with_history                   # noqa: E402

DATA = ROOT / "extracted_npz_dataset_v2"
RECORDINGS = ROOT / "dataset" / "eeg_cleaned_data"      # the per-run CSVs the offline monitor reads
CACHE = ROOT / "measurements_v2.npz"
LABELS_CACHE = ROOT / "monitor_labels_v2.npz"
OFFLINE_LABELS_CACHE = ROOT / "monitor_labels_offline_v2.npz"
CLASSES = ["LEFT", "RIGHT", "DOWN", "STRAIGHT"]
DESIGN_LAST = 20
FOLDS = [[f"sub_01/ses-{s:02d}" for s in range(a, z + 1)] for a, z in ((1, 7), (8, 14), (15, 20))]
KEYS = ("contrast", "background", "background_valid", "delta", "deflect", "nod", "sideways", "turn")


def monitor_thresholds(config: RtamtConfig | None = None) -> dict:
    """The thresholds, read from the monitor itself (monitoring/rtamt_signals.py, RtamtConfig)."""
    cfg = config or RtamtConfig()
    return dict(b=cfg.background_ratio, g=cfg.gate_ratio, d=cfg.delta_range_min, q=cfg.quiet_min,
                n=cfg.down_threshold, v=cfg.down_veto_prominence)


SHIPPED = monitor_thresholds()       # never typed in here: change RtamtConfig and every table follows


def fingerprint() -> str:
    """A hash of everything that decides a verdict: the monitor's code, its settings, and the dataset.

    Any edit to a file in monitoring/, to the thresholds, or to the dataset changes it, and the cached
    results are then recomputed automatically.
    """
    h = hashlib.sha256()
    for path in sorted((ROOT / "monitoring").glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    h.update(repr(RtamtConfig()).encode())
    for path in sorted(DATA.glob("*/ses-*/clean_epochs.npz")):
        stat = path.stat()
        h.update(f"{path.relative_to(DATA)}|{stat.st_size}|{int(stat.st_mtime)}".encode())
    return h.hexdigest()[:12]


# =====================================================================================================
# 1. MEASURE -- the monitors' own measurements for every window, with the stream before it
# =====================================================================================================
def _session(path):
    cfg = RtamtConfig()
    labels = np.load(path, allow_pickle=True)["labels"].astype(str)
    out = {k: np.zeros(len(labels)) for k in KEYS}
    for k, window, history in windows_with_history(path, config=cfg):
        s = compute_signals(window, cfg, history)
        for key, value in (("contrast", s["contrast"].max()), ("background", s["background"].max()),
                           ("background_valid", s["background_valid"].max()), ("delta", s["delta"].max()),
                           ("deflect", s["deflect"].max()), ("nod", s["nod"][-1]),
                           ("sideways", s["sideways"].max()), ("turn", s["turn"][-1])):
            out[key][k] = value
    path = Path(path)
    out["label"] = labels
    out["session"] = np.array([f"{path.parent.parent.name}/{path.parent.name}"] * len(labels))
    return out


def _cached(path: Path, stamp: str):
    """The cache's contents if it was made by the current monitor, else None."""
    if not path.exists():
        return None
    f = dict(np.load(path, allow_pickle=True))
    if str(f.get("fingerprint", np.array(""))) != stamp:
        return None
    return f


def measure(force=False):
    """Compute (and cache) the measurements of every window. About 5 minutes the first time.

    The cache is thrown away and rebuilt whenever the monitor's code, settings or the dataset change.
    """
    stamp = fingerprint()
    f = None if force else _cached(CACHE, stamp)
    if f is None:
        print(f"measuring every window with the current monitor ({stamp}) ...", flush=True)
        files = sorted(DATA.glob("*/ses-*/clean_epochs.npz"))
        with Pool(6) as pool:
            parts = pool.map(_session, [str(p) for p in files])
        f = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez(CACHE, fingerprint=np.array(stamp), **f)
        print(f"saved {CACHE.name}: {len(f['label'])} windows")
    f["label"] = f["label"].astype(str)
    f["session"] = f["session"].astype(str)
    f["dev"] = np.array([s.startswith("sub_01") and int(s[-2:]) <= DESIGN_LAST for s in f["session"]])
    return f


def _classify_session(path):
    """Run the REAL monitor (monitoring.HeadMovementMonitor, RTAMT) on every window of one session."""
    from monitoring import HeadMovementMonitor
    cfg = RtamtConfig()
    monitor = HeadMovementMonitor(cfg)
    data = np.load(path, allow_pickle=True)
    with_history, alone = [], []
    for k, window, history in windows_with_history(path, config=cfg):
        with_history.append(monitor.classify(window, history))
        alone.append(monitor.classify(window))
    path = Path(path)
    return (np.array(with_history), np.array(alone), data["labels"].astype(str),
            np.array([f"{path.parent.parent.name}/{path.parent.name}"] * len(alone)))


def monitor_labels(force=False):
    """The verdict of the production monitor on every window, with and without the stream history.

    This calls HeadMovementMonitor.classify -- the four RTAMT specifications themselves -- so the table
    always reflects the monitor as it is in monitoring/. About 90 minutes the first time (6 processes);
    cached afterwards, and recomputed automatically whenever the monitor or the dataset changes.
    """
    stamp = fingerprint()
    f = None if force else _cached(LABELS_CACHE, stamp)
    if f is None:
        files = sorted(DATA.glob("*/ses-*/clean_epochs.npz"))
        print(f"running the monitor on every window of {DATA.name} ({len(files)} sessions, "
              f"monitor fingerprint {stamp}); this takes a while the first time ...", flush=True)
        start = time.time()
        with Pool(6) as pool:
            parts = pool.map(_classify_session, [str(p) for p in files])
        f = dict(with_history=np.concatenate([p[0] for p in parts]), alone=np.concatenate([p[1] for p in parts]),
                 label=np.concatenate([p[2] for p in parts]), session=np.concatenate([p[3] for p in parts]))
        np.savez(LABELS_CACHE, fingerprint=np.array(stamp), **f)
        print(f"done in {(time.time() - start) / 60:.1f} min; saved {LABELS_CACHE.name}\n")
    for key in ("with_history", "alone", "label", "session"):
        f[key] = f[key].astype(str)
    f["dev"] = np.array([s.startswith("sub_01") and int(s[-2:]) <= DESIGN_LAST for s in f["session"]])
    return f


def offline_fingerprint() -> str:
    """fingerprint() plus the offline thresholds and the CSV recordings the offline monitor reads."""
    from monitoring.offline_monitor import OfflineConfig
    h = hashlib.sha256(fingerprint().encode())
    h.update(repr(OfflineConfig()).encode())
    for path in sorted(RECORDINGS.glob("*/ses-*/ica_eeg/*.csv")):
        stat = path.stat()
        h.update(f"{path.relative_to(RECORDINGS)}|{stat.st_size}|{int(stat.st_mtime)}".encode())
    return h.hexdigest()[:12]


def _offline_session(path):
    """Run the OFFLINE RTAMT monitor on every window of one session, reading each run's CSV recording once."""
    from monitoring.offline_monitor import OfflineHeadMovementMonitor, dataset_recording_path, load_recording
    monitor = OfflineHeadMovementMonitor()
    data = np.load(path, allow_pickle=True)
    task, run = data["task"].astype(str), data["run"].astype(int)
    start = data["run_offset"].astype(int)
    path = Path(path)
    session = f"{path.parent.parent.name}/{path.parent.name}"
    verdicts = np.empty(len(task), dtype=object)
    for t, r in sorted(set(zip(task, run))):
        idx = np.flatnonzero((task == t) & (run == r))
        recording = load_recording(dataset_recording_path(RECORDINGS, session, t, r))
        verdicts[idx] = monitor.classify_recording(recording, start[idx])
    return verdicts.astype(str), data["labels"].astype(str), np.array([session] * len(task))


def offline_labels(force=False):
    """Verdicts of the OFFLINE monitor (monitoring/offline_monitor.py, RTAMT) on every window. ~15 min, cached."""
    stamp = offline_fingerprint()
    f = None if force else _cached(OFFLINE_LABELS_CACHE, stamp)
    if f is None:
        files = sorted(DATA.glob("*/ses-*/clean_epochs.npz"))
        print(f"running the OFFLINE monitor on every window ({len(files)} sessions, fingerprint {stamp}) ...", flush=True)
        start = time.time()
        with Pool(6) as pool:
            parts = pool.map(_offline_session, [str(p) for p in files])
        f = dict(offline=np.concatenate([p[0] for p in parts]), label=np.concatenate([p[1] for p in parts]),
                 session=np.concatenate([p[2] for p in parts]))
        np.savez(OFFLINE_LABELS_CACHE, fingerprint=np.array(stamp), **f)
        print(f"done in {(time.time() - start) / 60:.1f} min; saved {OFFLINE_LABELS_CACHE.name}\n")
    for key in ("offline", "label", "session"):
        f[key] = f[key].astype(str)
    f["dev"] = np.array([s.startswith("sub_01") and int(s[-2:]) <= DESIGN_LAST for s in f["session"]])
    return f


def evaluate_offline(force=False):
    """The results table of the OFFLINE monitor (whole recording known), next to the live monitor's."""
    from monitoring.offline_monitor import OfflineConfig
    f = offline_labels(force=force)
    cfg = OfflineConfig()
    intro = (f"OFFLINE monitor (monitoring/offline_monitor.py) on {DATA.name}: every window is compared with the rest\n"
             f"of its own recording, before AND after it. Uses samples a live system would not have yet.\n"
             f"Verdicts from: OfflineHeadMovementMonitor, i.e. four RTAMT offline specifications.  "
             f"Fingerprint: {offline_fingerprint()}.\n"
             f"Thresholds: offline background {cfg.background_ratio}, delta {cfg.live.delta_range_min}, deflect "
             f"{cfg.live.quiet_min}, nod {cfg.live.down_threshold}, sideways {cfg.live.down_veto_prominence}.\n")
    results_table(f, f["offline"], intro)
    if LABELS_CACHE.exists():
        live = dict(np.load(LABELS_CACHE, allow_pickle=True))
        same = (live["label"].astype(str) == f["label"]).all() and (live["session"].astype(str) == f["session"]).all()
        if same:
            print("\nFor comparison, the LIVE monitor (with stream history), balanced accuracy:")
            for name, m in splits(f):
                print(f"   {name:32s} live {balanced(f, live['with_history'].astype(str), m):6.2f}%   "
                      f"offline {balanced(f, f['offline'], m):6.2f}%")


# =====================================================================================================
# 2. THE DECISION -- the four specifications, as plain arithmetic (RTAMT gives the same answer)
# =====================================================================================================
def predict(f, p=None, history=True):
    p = p or SHIPPED
    valid = (f["background_valid"] >= 0.5) if history else np.zeros(len(f["label"]), bool)
    loud = np.where(valid, f["background"] >= p["b"], f["contrast"] >= p["g"])
    move = loud & (f["delta"] >= p["d"]) & (f["deflect"] >= p["q"])
    nod = (f["nod"] >= p["n"]) & (f["sideways"] <= p["v"])
    return np.where(~move, "STRAIGHT", np.where(nod, "DOWN", np.where(f["turn"] > 0, "LEFT", "RIGHT")))


def balanced(f, pred, m):
    return 100 * float(np.mean([np.mean(pred[m & (f["label"] == k)] == k) for k in CLASSES
                                if (m & (f["label"] == k)).any()]))


def report(f, pred, m, name):
    cells = " | ".join(f"{k} {int(((pred == k) & m & (f['label'] == k)).sum())}/{int((m & (f['label'] == k)).sum())}"
                       f" ({100 * np.mean(pred[m & (f['label'] == k)] == k):.1f}%)" for k in CLASSES)
    rest = m & (f["label"] == "STRAIGHT")
    print(f"   {name:34s} balanced {balanced(f, pred, m):6.2f}%  | {cells} | false alarms "
          f"{int((pred[rest] != 'STRAIGHT').sum())}/{int(rest.sum())}")


def splits(f):
    dev = f["dev"]
    return (("sub_01 sessions 1-20 (design)", dev),
            ("sub_01 sessions 21-36 (test)", np.char.startswith(f["session"], "sub_01") & ~dev),
            ("sub_11 (test)", np.char.startswith(f["session"], "sub_11")))


def results_table(f, pred, title):
    """Print the per-class table: correct / total (percentage) per split, then overall and balanced."""
    columns = (("Training/design: sub_01 sessions 1-20", f["dev"]),
               ("Test: sub_01 sessions 21-36", np.char.startswith(f["session"], "sub_01") & ~f["dev"]),
               ("Test: sub_11", np.char.startswith(f["session"], "sub_11")))
    label = f["label"]

    def cell(m):
        correct, total = int(((pred == label) & m).sum()), int(m.sum())
        return f"{correct:,} / {total:,} ({100 * correct / total:.2f}%)"

    rows = [(k, [cell(m & (label == k)) for _, m in columns]) for k in CLASSES]
    rows.append(("Total correct / total", [f"{int(((pred == label) & m).sum()):,} / {int(m.sum()):,}" for _, m in columns]))
    rows.append(("Overall accuracy", [f"{100 * np.mean(pred[m] == label[m]):.2f}%" for _, m in columns]))
    rows.append(("Balanced accuracy", [f"{balanced(f, pred, m):.2f}%" for _, m in columns]))

    header = ["Class"] + [name for name, _ in columns]
    width = [max(len(header[0]), max(len(r[0]) for r in rows))]
    width += [max(len(header[i + 1]), max(len(r[1][i]) for r in rows)) for i in range(len(columns))]
    line = "+" + "+".join("-" * (w + 2) for w in width) + "+"

    def show(cells):
        print("| " + " | ".join(c.ljust(w) for c, w in zip(cells, width)) + " |")

    print(title)
    print(line)
    show(header)
    print(line)
    for i, (name, cells) in enumerate(rows):
        if i == len(CLASSES):                          # separate the class rows from the totals
            print(line)
        show([name] + cells)
    print(line)
    everything = np.ones(len(label), bool)
    correct = int((pred == label).sum())
    print(f"\nAcross all splits combined: {correct:,} / {len(label):,} correct "
          f"({100 * correct / len(label):.2f}% overall; {balanced(f, pred, everything):.2f}% balanced accuracy).")


def evaluate(detail=False, fast=False, force=False):
    """The results table of the production monitor on extracted_npz_dataset_v2.

    Every number is computed at run time. By default the verdicts come from running
    monitoring.HeadMovementMonitor.classify (the RTAMT specifications themselves) on every window, so any
    change to the monitor's formulas, thresholds or signals changes the table. `fast` uses the arithmetic
    equivalent of the formulas on cached measurements instead: seconds instead of minutes, and it follows
    threshold changes, but NOT changes to the formulas themselves.
    """
    cfg = RtamtConfig()
    p = monitor_thresholds(cfg)
    source = ("arithmetic equivalent of the formulas (--fast)" if fast
              else "HeadMovementMonitor.classify, i.e. the RTAMT specifications")
    intro = (f"These are the current production monitor's results, with stream history enabled, on {DATA.name}.\n"
             f"Each cell shows correct / total (percentage). The recent experimental formulas have not replaced "
             f"this monitor.\n"
             f"Verdicts from: {source}.  Monitor fingerprint: {fingerprint()}.\n"
             f"Thresholds read from the monitor: background {p['b']}, contrast {p['g']}, delta {p['d']}, "
             f"deflect {p['q']}, nod {p['n']}, sideways {p['v']}.\n")
    if fast:
        f = measure(force=force)
        with_history, alone = predict(f, p), predict(f, p, history=False)
    else:
        f = monitor_labels(force=force)
        with_history, alone = f["with_history"], f["alone"]
    results_table(f, with_history, intro)
    if detail:                                         # the same monitor judging every window on its own
        print()
        results_table(f, alone, "For comparison: each window judged alone (no stream history, the contrast gate).\n")


# =====================================================================================================
# 3. TRAIN -- re-select the thresholds on the design sessions (blocked 3-fold CV), then test once
# =====================================================================================================
def search(f, train, params=None):
    """Gate thresholds first (b, g, d, q), then the nod thresholds (n, v).

    Keep the g grid at 5.0 and above. Below it, CV "finds" g ~ 4 (+1.9 on sessions 1-30), but only because
    74.5% of movement windows and 15% of rest windows lack stream history here, so g judges mostly movements.
    sub_11 false alarms then rise 23 -> 142. See experiments/improvement_round_2026_09_24, trap 4.1.
    """
    best = dict(params or SHIPPED)
    top = (-1.0, best)
    for b, g, d, q in itertools.product(np.arange(3.0, 12.1, 0.5), np.arange(5.0, 11.1, 0.5),
                                        np.arange(0.08, 0.361, 0.02), np.arange(2.0, 3.61, 0.1)):
        p = dict(best, b=round(float(b), 2), g=round(float(g), 2), d=round(float(d), 3), q=round(float(q), 2))
        a = balanced(f, predict(f, p), train)
        if a > top[0] + 1e-9:
            top = (a, p)
    best = top[1]
    top = (-1.0, best)
    for n, v in itertools.product(np.arange(-2.0, 1.01, 0.1), np.arange(5.0, 11.01, 0.25)):
        p = dict(best, n=round(float(n), 2), v=round(float(v), 2))
        a = balanced(f, predict(f, p), train)
        if a > top[0] + 1e-9:
            top = (a, p)
    return top[1]


def train():
    """Blocked 3-fold CV on sub_01 sessions 1-20 (blocks 1-7, 8-14, 15-20)."""
    f = measure()
    dev = f["dev"]
    cv = predict(f).copy()
    for k in range(3):
        held = np.isin(f["session"], FOLDS[k])
        chosen = search(f, dev & ~held)
        cv[held] = predict(f, chosen)[held]
        print(f"   fold {k + 1}: {chosen}")
    report(f, cv, dev, "cross-validated selection")
    report(f, predict(f), dev, "shipped thresholds")
    print(f"   selected on all of sessions 1-20: {search(f, dev)}")


PREVIOUS_1_30 = dict(b=6.61, g=7.81, d=0.208, q=2.72, n=-0.661, v=7.54)


def train_first_30():
    """Re-select thresholds on sub_01 sessions 1-30; score both settings on sub_01 21-36 / 31-36 and sub_11."""
    f = measure()
    num = np.array([int(s[-2:]) for s in f["session"]])
    s01 = np.char.startswith(f["session"], "sub_01")
    fit = s01 & (num <= 30)
    chosen = search(f, fit, PREVIOUS_1_30)
    print(f"   selected on sub_01 sessions 1-30: {chosen}")
    print(f"   default (selected on 1-20):        {SHIPPED}")
    for name, m in (("sub_01 sessions 21-36", s01 & (num >= 21)), ("sub_01 sessions 31-36", s01 & (num >= 31)),
                    ("sub_11", np.char.startswith(f["session"], "sub_11"))):
        print(name)
        report(f, predict(f, chosen), m, "trained on 1-30")
        report(f, predict(f), m, "default (trained on 1-20)")


def test():
    """Score the held-out sets once. Set FINAL_TEST=1 to confirm you mean it."""
    if os.environ.get("FINAL_TEST") != "1":
        raise SystemExit("the test sets are scored only with FINAL_TEST=1")
    f = measure()
    chosen = search(f, f["dev"])
    for name, m in splits(f):
        print(name)
        report(f, predict(f), m, "shipped thresholds")
        report(f, predict(f, chosen), m, "selected on sessions 1-20")


# =====================================================================================================
# 4. DATASET EXTRACTION -- four methods. The ACTIVE dataset is extracted_npz_dataset_v2, already built.
#    Full original scripts: _archive/pipeline/. Uncomment one block to rebuild that dataset.
# =====================================================================================================
#
# --- 4a. extracted_npz_dataset_v2 (THE dataset in use) ------------------------------------------------
# Built in two steps from the recordings and their ACT_* annotations:
#     python -m pipeline.extract_clean_windows      # _archive/pipeline/extract_clean_windows.py
#     python -m pipeline.remove_spliced_windows     # _archive/pipeline/remove_spliced_windows.py
# Step 1 keeps a window only if the WHOLE annotated movement lies inside it with 0.25 s to spare, and a
# rest window only if no movement comes within 1 s. Step 2 drops the 2,135 windows whose samples came
# from two different recordings joined end to end (they doubled the false alarms: 21.6% against 9.7%).
# Result: 21,029 windows; accuracy 85.01 / 69.62 / 78.67.
#
# --- 4b. extracted_npz_dataset_v2_preica (same windows, blinks NOT removed) ---------------------------
# from pipeline.build_preica_windows import main as build_preica      # _archive/pipeline/
# build_preica(["--source", "D:/errp/eeg_cleaned_data"])
# Every window is verified against the ICA recording (worst difference 2.9e-11 V) and then cut from the
# pre-ICA file. Used once, to look for an eye-movement direction signal: there is none (the early slope of
# Fp1-Fp2 points against the physics in 76% of turns, in the cleaned data too). The monitors score 76.90%
# on it, because the uncleaned blinks raise false alarms (rest 62%).
#
# --- 4c. extracted_npz_dataset_v3_csv (cut from the raw CSVs, cue marker + detector) ------------------
# from pipeline.extract_windows_from_csv import main as build_csv      # _archive/pipeline/
# build_csv([])
# Labels from the file name; marker 7006 is the movement cue (left/right runs only, no down run has one);
# the movement is found by a 0.3-4 Hz detector, deliberately different from the monitor's gate. 11,137
# windows. The unchanged monitor scores 85.22 / 64.94 / 72.25 -- the same picture as v2, which is the point
# of the exercise: the result does not depend on how the windows were cut. Deleted afterwards.
#
# --- 4d. extracted_npz_dataset_trials (ONE window per trial, 442) ------------------------------------
# from pipeline.extract_trial_windows import main as build_trials      # _archive/pipeline/
# build_trials([])
# No marker and no annotation at all: band-pass 0.3-4 Hz, divide each channel by its own spread, take the
# MEDIAN over the eight channels (so a blink or one bad electrode cannot make a peak), and centre the 3 s
# on that movement. A straight trial gives its quietest 3 s. The annotated movement lands fully inside in
# 85% of trials. Monitors: 78.78 / 54.19 / 68.10 (one decision per trial is a harder test than per window).
# Thresholds trained on it (b 3.0, g 5.0, d 0.16, q 2.0, n -0.6, v 5.75) win there but cause 22-33% false
# alarms on v2, because that dataset has 418 movement trials against 24 rest trials.


# =====================================================================================================
# 5. TRIED AND NOT ADOPTED -- kept so that no idea is tested twice. Records: experiments/<round>/README.md
# =====================================================================================================
#
# --- 5a. CUSUM gate and SINCE direction (2026-09-16) --------------------------------------------------
# HeadMovementMonitor(RtamtConfig(gate_statistic="cusum", direction_rule="since"))
# Design sessions 83.50% against 82.30%, held out 65.05 / 75.48 against 66.28 / 76.40. Both still exist in
# monitoring/rtamt_signals.py as options, off by default.
#
# --- 5b. extra nod vetoes on Fp1-Fp2 and Fp2-T3 (2026-09-17) -----------------------------------------
# HeadMovementMonitor(RtamtConfig(down_veto_fp1fp2=7.1, down_veto_fp2t3=6.6))
# Design 83.86%, sub_01 test 68.04%, but sub_11 fell 77.00 -> 74.72 (its nods do swing sideways).
#
# --- 5c. artifact detectors: blink, muscle, electrode jump, very large signal (2026-09-17) ------------
# def blink_flag(x):                       # full code: _archive/experiment_scripts/artifact_check_2026_09_17/
#     b = filtfilt(*butter(2, [0.5 / 128, 10 / 128], "band"), x, axis=-1)
#     fp = 0.5 * (b[0] + b[1]) - np.median(0.5 * (b[0] + b[1]))
#     z = np.abs(fp) / (1.4826 * np.median(np.abs(fp)) + 1e-12)
#     ...                                  # >= 5 SD, 0.04-0.40 s wide, same sign on Fp1/Fp2, fading backwards
# Classic artifacts sit in 0.3% of rest windows and explain under 1% of false alarms; most false alarms are
# small residual blink-shaped bumps that look like small nods. Removing flagged windows: 82.74 -> 82.96.
# Every veto ("artifact -> STRAIGHT") lost accuracy; cross-validation chose no veto 3 times out of 3.
#
# --- 5d. many rules per monitor: k of 27 (2026-09-17) ------------------------------------------------
# rules = np.column_stack([contrast, delta, deflect] + [per-channel contrast, delta, deflect])
# move = (rules >= threshold_at_rest_quantile).sum(1) >= k
# CV +2.19 on the design sessions, then 64.47 / 68.29 held out. The per-channel slow-wave thresholds were
# the problem: set to fire on 3% of rest windows here, they fire on 24-47% of sub_11's.
#
# --- 5e. normalised per-channel rules and sign-only rules (2026-09-17) --------------------------------
# lc = log(loudest 0.5 s / quietest 0.5 s of the channel)     # transfers: fires on 10-14% of rest everywhere
# agree = channels sharing the majority sign at the burst peak  # DOWN 8 of 8 against turns 5 of 8
# coherence = mean correlation between the channels             # DOWN 0.59 against turns 0.13
# Normalising by each channel's own spread does make a rule transfer between subjects, but none of the 16
# designs added accuracy: cross-validation kept the current nod rule every time.
#
# --- 5f. head geometry from the electrode positions (2026-09-17) -------------------------------------
# P = np.column_stack([x_coordinate, y_coordinate, np.ones(8)])    # yaw, pitch, common; MNE standard_1020
# coefficients = np.linalg.pinv(P) @ window_in_the_loudest_second
# geo_turn = log(yaw energy / pitch energy)
# Passed the design sessions (+0.72) and broke sub_11 (77.00 -> 67.77: its turns became nods).
#
# --- 5g. consecutive windows must agree, and "UNSURE" near a threshold (2026-09-17) ------------------
# historically[0:n](same class)                         # no gain at all
# UNSURE when the normalised STL margin < 0.14          # fewer wrong movement answers on both test sets,
#                                                       # but lower accuracy on sub_01 31-36; not adopted
#
# --- 5h. re-selecting the thresholds after the background gate (2026-09-17, 2026-09-18) --------------
# `train` above does exactly this. On v2 it keeps the shipped values; on the CSV dataset the selection was
# worse under CV (82.99 against 85.22); on the 442 trials it wins there and fails everywhere else (5d-like).
#
# --- 5i. machine learning on top of the monitors (earlier round) -------------------------------------
# 27 models and 3 hybrids on the four robustness values. None beat the plain rules on held-out data.
#
# --- 5j. Improvement round (2026-09-24). Record + scripts: experiments/improvement_round_2026_09_24/ ----------------
# Nothing adopted. Errors are all-or-nothing per movement, so the held-out sets are 37 and 95 movements.
# nod - rest value of the 4 s before the window      pooled AUC 0.955 -> 0.926
# nod - session rest median                          AUC 0.964/0.858/0.999 -> 0.972/0.817/0.733
# veto DOWN when |turn| > t                          CV -0.17 (0/1 sessions)
# nod - a*sideways >= n (slanted boundary)           CV -1.35 (+0.19 against a baseline re-selected the same way)
# nod from the loudest second / event only           AUC +0.005 on 1-30, -0.006 on 31-36
# first F4-T3 lobe reaching h x the largest          direction 92.6/100/70.5 -> 86.5/71.7/56.9 (h 0.5)
# per-channel within-window gate (10th pct block)    matched false alarms: 31-36 recall -2 to -16
# background settle/minimum/filter-start variants    31-36 +1.4..+2.5 but sub_11 -1.4..-1.7
# label-free per-session nod split                   -0.38 to -2.04
# ONE labelled nod + ONE labelled turn per session   1-30 -1.02, 31-36 +2.34, sub_11 -0.12 (breaks the rules anyway)
# Inside each held-out session nod separates turns from nods (AUC 1.0 in 13 of 14); only the gap's position
# moves. A per-session threshold chosen with labels would give 88.65 on 1-30: the ceiling of any such fix.
#
# --- 5k. Rejected-methods round (2026-09-30). Summary: experiments/rtamt_accuracy_2026_09_30; code in _archive/rejected_2026_09_30
# Joint 8-channel shapelets (depth-3/5 trees exported to RTAMT)   outer CV 80.70 live / 83.50 offline (vs 85.01 / 86.73)
# Second-order wavelet scattering rules                          outer CV 82.65 live / 83.10 offline
# Scattering four-class corrections                              31-36 +2.8 but sub_11 78.67 -> 69.39 (false DOWNs)
# Waveform templates, true CSV history, warmed filters, offline crops, robust nod energy: inner CV kept the baseline
#
# --- 5l. Censored stream background (2026-10-07; an OPTION, off by default) ---------------------------
# RtamtConfig(background_censor=True): StreamMonitor / RowStreamMonitor leave the loud 0.5-s blocks of windows
# they already judged a movement out of the background (radar-style censored CFAR). On joined real runs it
# rescues a second movement 1.3-4 s after the first (186 -> 251 of 318 right), but single-run false alarms rise
# 348 -> 1214 (sub_01) and 43 -> 66 (sub_11): the stream after a movement is louder than before it.
# Censoring whole windows: 254 of 318 but 1979 false alarms. Stored-window results are unaffected.
#
# --- 5m. Electrode-group rules, rule combinations, context switching (2026-10-07). NOT adopted ----------
# Record + scripts: experiments/group_rules_2026_10_07/. Window-only (no stream history). All 3,025 pairs of
# disjoint electrode groups x 8 measurements x 2 bands = 48,400 rules for nod/direction, 765 single-group rules
# for detection; families single / AND / OR / k-of-n vote / baseline AND (veto) / baseline OR (rescue) /
# regime switching / confidence cascade; selected on sub_01 1-20 (blocked CV), null = shuffled labels.
# Baseline (window-only, re-selected): CV 78.01, 21-36 78.85, sub_11 75.55. Best CV combo 84.01 -> 74.15 / 49.11.
# Every combination that gained on sub_01 lost on sub_11: nod rules turn sub_11's turns into DOWN (4 -> 258).
# Regime switching and the cascade lost already in CV (nod 81.98, direction 81.97 vs 86.91 / 89.11).
#
# --- 5n. RAW (pre-ICA) data techniques (2026-10-13). Record: experiments/raw_live_2026_10_13/ ------------
# Blink veto PASSED for raw data (not yet in monitoring/): movement -> STRAIGHT when the loudest 0.5 s is
# forehead-dominant (log Fp/T energy >= 2.0) and short (<= 0.3 s). Raw, real history: CV 75.66 -> 78.94 (+17/-3),
# 21-36 77.67 -> 81.35, sub_11 76.31 -> 77.94; false alarms 1282 -> 309 and 166 -> 108. Live replay, 1 s agreement:
# false commands 4.33 -> 1.24 /min (21-36), 9.41 -> 5.65 (sub_11). On cleaned data nearly inert (-0.33 / 0.00).
# Thresholds re-selected on raw: CV 73.50 (+9/-11), rejected; on cleaned: CV 82.18 vs 85.46 (+8/-10), rejected.


# =====================================================================================================
# 6. ONE ROW PER WINDOW -- for error analysis: what was it, what did the monitor say, and how surely
# =====================================================================================================
def class_robustness(f, p=None):
    """The four specifications' RTAMT robustness for every window, from the cached measurements.

    RTAMT's quantitative semantics are min for `and`, max for `or`, max over time for `once` and min
    over time for `historically`. The measurements are already reduced the way each signal's own
    temporal operator reduces it, so each formula's robustness is a min/max of (signal - threshold)
    terms. `windows --verify` checks this against the real HeadMovementMonitor.
    """
    p = p or SHIPPED
    bg, bv = f["background"], f["background_valid"]
    loud = np.maximum(bg - p["b"], np.minimum(0.5 - bv, f["contrast"] - p["g"]))
    quiet = np.minimum(p["b"] - bg, np.maximum(bv - 0.5, p["g"] - f["contrast"]))
    move = np.minimum.reduce([loud, f["delta"] - p["d"], f["deflect"] - p["q"]])
    not_nod = np.maximum(p["n"] - f["nod"], f["sideways"] - p["v"])
    return {
        "LEFT": np.minimum.reduce([move, not_nod, f["turn"]]),
        "RIGHT": np.minimum.reduce([move, not_nod, -f["turn"]]),
        "DOWN": np.minimum.reduce([move, f["nod"] - p["n"], p["v"] - f["sideways"]]),
        "STRAIGHT": np.maximum.reduce([quiet, p["d"] - f["delta"], p["q"] - f["deflect"]]),
    }


def windows(out="windows.csv", verify=False):
    """Write one row per window: truth, verdict, the four robustness values and every signal."""
    f = measure()
    labels = monitor_labels()                      # the REAL monitor's verdicts (RTAMT)
    p = SHIPPED
    rho = class_robustness(f, p)

    if verify:                                     # the composed robustness against real RTAMT
        from monitoring import HeadMovementMonitor
        monitor = HeadMovementMonitor(RtamtConfig())
        worst, seen = 0.0, 0
        for path in sorted(DATA.glob("*/ses-*/clean_epochs.npz"))[:2]:
            for k, window, history in windows_with_history(path, config=RtamtConfig()):
                got = monitor.robustness(window, history)
                one = {key: np.array([f[key][seen]]) for key in KEYS}
                mine = class_robustness(one, p)
                worst = max(worst, max(abs(float(mine[c][0]) - got[c]) for c in CLASSES))
                seen += 1
                if seen >= 60:
                    break
            if seen >= 60:
                break
        print(f"verified {seen} windows against real RTAMT: worst disagreement {worst:.3e}")

    stack = np.stack([rho[c] for c in CLASSES])
    margin = stack.max(axis=0)                     # the winning specification's robustness
    order = np.sort(stack, axis=0)
    gap = order[-1] - order[-2]                    # how far ahead the winner is
    split = np.where(f["dev"], "design",
                     np.where(np.char.startswith(f["session"], "sub_11"), "test_sub_11", "test_sub_01"))
    columns = {
        "index": np.arange(len(f["label"])), "session": f["session"], "split": split,
        "true": f["label"], "predicted": labels["with_history"],
        "correct": (labels["with_history"] == f["label"]).astype(int),
        "predicted_no_history": labels["alone"],
        "margin": margin, "runner_up_gap": gap,
    }
    columns.update({f"rho_{c}": rho[c] for c in CLASSES})
    columns.update({k: f[k] for k in KEYS})
    parts = np.stack([f["background"] - p["b"], f["contrast"] - p["g"],
                      f["delta"] - p["d"], f["deflect"] - p["q"]])
    columns["weakest_move_term"] = np.array(["background", "contrast", "delta", "deflect"])[parts.argmin(axis=0)]

    path = ROOT / out
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(columns) + "\n")
        rows = zip(*[np.asarray(v).astype(str) for v in columns.values()])
        fh.write("\n".join(",".join(r) for r in rows) + "\n")
    print(f"wrote {len(f['label']):,} rows to {path}")
    print("columns:", ", ".join(columns))
    wrong = columns["correct"] == 0
    print(f"\n{int(wrong.sum()):,} wrong windows. Median winning margin: "
          f"{np.median(margin[~wrong]):.3f} when correct, {np.median(margin[wrong]):.3f} when wrong.")
    print("Sort by `margin` ascending to see the monitor's least certain decisions first.")


# =====================================================================================================
# 7. NEW SUBJECTS -- cut the 3-s windows from a subject's recordings, then score every subject
# =====================================================================================================
# The two steps that built extracted_npz_dataset_v2 (originally _archive/pipeline/extract_clean_windows.py,
# --mode contained, then _archive/pipeline/remove_spliced_windows.py), in one command and for any subject:
#     python project_code.py extract sub_12
# Input, per session (the lab's format): dataset/eeg_cleaned_data/<subject>/<session>/
#     labelled_epochs_ds.npz             the lab's 3-s sliding windows (key `epochs`, volts; key `labels`)
#     ica_eeg/augmented_task-*_raw.fif   the recordings with ACT_left / ACT_right / ACT_down annotations
#     ica_eeg/*_task-<task>_run-<NN>_eeg.csv   one CSV per run (columns Fp1..T4, volts)
# Output: extracted_npz_dataset_v2/<subject>/<session>/clean_epochs.npz, ready for `evaluate`.
EXTRACT_MARGIN = 0.25        # a movement must clear both window edges by this many seconds
EXTRACT_GUARD = 1.0          # a rest window must be this far from any movement
EXTRACT_MATCH = 0.999        # a window is placed in its recording only if it matches at least this well
EXTRACT_TOL = 1e-10          # volts: a window must equal 768 consecutive samples of one CSV run
EEG_CHANNELS = ["Fp1", "Fp2", "F3", "Fz", "F4", "T3", "Cz", "T4"]


def _act_label(description):
    value = str(description).strip()
    match = re.fullmatch(r"ACT(\d*)_(LEFT|RIGHT|DOWN)", value, re.I)
    if match:
        return match.group(2).upper()
    if value.upper().startswith("ACT"):
        raise ValueError(f"unrecognised ACT-like annotation: {description!r}")
    return None


def _locate(template, signal):
    from scipy.signal import fftconvolve
    centred = template - template.mean()
    n = len(centred)
    numerator = fftconvolve(signal, centred[::-1], mode="valid")
    total = fftconvolve(signal, np.ones(n), mode="valid")
    square = fftconvolve(signal ** 2, np.ones(n), mode="valid")
    variance = np.maximum(square - total ** 2 / n, 1e-12)
    r = (numerator - total * centred.mean()) / np.sqrt(variance * float((centred ** 2).sum()))
    k = int(np.argmax(r))
    return k, float(r[k])


def _place(epochs_fp1, signals):
    out, task, base_i, base_start = [], None, 0, 0
    for i, template in enumerate(epochs_fp1):
        matched = False
        if task is not None:
            guess = base_start + int(round((i - base_i) * 0.1 * 256))
            signal = signals[task]
            if 0 <= guess <= len(signal) - 768:
                r = float(np.corrcoef(signal[guess:guess + 768], template)[0, 1])
                if r > EXTRACT_MATCH:
                    out.append((task, guess, r))
                    matched = True
        if not matched:
            best = (None, -1, -2.0)
            for name, signal in signals.items():
                k, r = _locate(template, signal)
                if r > best[2]:
                    best = (name, k, r)
            out.append(best)
            task, base_i, base_start = best[0], i, best[1]
    return out


def _in_run(window, run, k):
    if k < 0 or k + 768 > run.shape[1]:
        return False
    seg = run[:, k:k + 768]
    return float(np.abs((seg - seg.mean(axis=1, keepdims=True)) - window).max()) < EXTRACT_TOL


def _find_run(window, runs):
    dw = np.diff(window[0])
    for key, run in runs.items():
        if run.shape[1] < 768:
            continue
        dr = np.diff(run[0])
        for k in np.flatnonzero(np.abs(dr[:run.shape[1] - 768 + 1] - dw[0]) < EXTRACT_TOL):
            if (abs(dr[k + 1] - dw[1]) < EXTRACT_TOL and abs(dr[k + 2] - dw[2]) < EXTRACT_TOL
                    and _in_run(window, run, int(k))):
                return key, int(k)
    return None, None


def extract_session(session_dir, out_file):
    """Cut one session's clean windows; returns per-class counts (source windows, kept windows)."""
    import csv
    import mne
    mne.set_log_level("ERROR")
    session_dir = Path(session_dir)
    data = np.load(session_dir / "labelled_epochs_ds.npz", allow_pickle=True)
    raw_labels = np.array([str(x).upper() for x in data["labels"]])
    usable = np.isin(raw_labels, CLASSES)
    epochs, npz_labels, source_index = data["epochs"][usable], raw_labels[usable], np.flatnonzero(usable)
    signals, acts = {}, {}
    for fif in sorted((session_dir / "ica_eeg").glob("augmented_task-*_raw.fif")):
        task = fif.name.split("task-")[1].split("_")[0]
        raw = mne.io.read_raw_fif(fif, preload=True)
        signals[task] = raw.get_data(picks=["Fp1"])[0] * 1e6
        acts[task] = [(float(o), float(du), _act_label(de)) for de, o, du in
                      zip(raw.annotations.description, raw.annotations.onset, raw.annotations.duration)
                      if _act_label(de) is not None]
    if not signals:
        raise SystemExit(f"{session_dir}: no ica_eeg/augmented_task-*_raw.fif found")
    # Step 1: place every window in its recording and label it from the annotations ("contained" rule).
    placed = _place(epochs[:, 0] * 1e6, signals)
    n = len(epochs)
    keep, truth = np.zeros(n, bool), np.array(["STRAIGHT"] * n, dtype="<U8")
    onset, duration, inside = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i, (task, start, r) in enumerate(placed):
        if task is None or r <= EXTRACT_MATCH:
            continue
        t0, t1 = start / 256, start / 256 + 3.0
        over = [(o, du, c, min(t1, o + du) - max(t0, o)) for o, du, c in acts[task] if o < t1 and o + du > t0]
        if over:
            o, du, c, ov = max(over, key=lambda z: z[3])
            onset[i], duration[i], inside[i] = o - t0, du, ov
        near = [a for a in acts[task] if a[0] - EXTRACT_GUARD < t1 and a[0] + a[1] + EXTRACT_GUARD > t0]
        if len(over) == 1:
            o, du, c, _ = over[0]
            if o - t0 >= EXTRACT_MARGIN and o + du - t0 <= 3.0 - EXTRACT_MARGIN:
                keep[i], truth[i] = True, c
        elif not near:
            keep[i], truth[i] = True, "STRAIGHT"
    task_of = np.array([p[0] or "" for p in placed])
    start_of = np.array([p[1] for p in placed], dtype=int)
    k1 = np.flatnonzero(keep)
    # Step 2: keep only windows that lie entirely inside ONE run of the per-run CSV recordings.
    runs = {}
    for path in (session_dir / "ica_eeg").glob("*_eeg.csv"):
        m = re.search(r"task-(\w+?)_run-(\d+)_eeg", path.name)
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            cols = [header.index(c) for c in EEG_CHANNELS]
            runs[(m.group(1), int(m.group(2)))] = np.array([[float(r[c]) for c in cols] for r in reader if r]).T
    run_of, offset_of = np.full(len(k1), -1), np.full(len(k1), -1)
    for t in np.unique(task_of[k1]):
        of_task = {key: run for key, run in runs.items() if key[0] == t}
        order = np.flatnonzero(task_of[k1] == t)
        order = order[np.argsort(start_of[k1][order], kind="stable")]
        previous = None
        for j in order:
            window, key, k = epochs[k1[j], :8], None, None
            if previous is not None:
                pkey, pk, ps = previous
                guess = pk + int(start_of[k1[j]] - ps)
                if _in_run(window, of_task[pkey], guess):
                    key, k = pkey, guess
            if key is None:
                key, k = _find_run(window, of_task)
            if key is None:
                previous = None
                continue
            run_of[j], offset_of[j] = key[1], k
            previous = (key, k, start_of[k1[j]])
    k2 = np.flatnonzero(run_of >= 0)
    rows = k1[k2]
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_file, epochs=epochs[rows], labels=truth[rows], npz_label=npz_labels[rows],
                        source_index=source_index[rows], movement_offset=onset[rows],
                        movement_duration=duration[rows], movement_inside=inside[rows], task=task_of[rows],
                        start_sample=start_of[rows], mode=np.array(["contained"]), min_overlap=np.array([0.3]),
                        margin=np.array([EXTRACT_MARGIN]), guard=np.array([EXTRACT_GUARD]),
                        run=run_of[k2], run_offset=offset_of[k2], v1_index=k2)
    return {c: (int((npz_labels == c).sum()), int((truth[rows] == c).sum())) for c in CLASSES}


def extract(subject, source=None, out=None, force=False):
    """`python project_code.py extract sub_12`: cut every session of one subject into clean windows."""
    source = Path(source or RECORDINGS) / subject
    out = Path(out or DATA) / subject
    sessions = sorted(p.parent for p in source.glob("ses-*/labelled_epochs_ds.npz"))
    if not sessions:
        raise SystemExit(f"no {source}/ses-*/labelled_epochs_ds.npz found")
    print(f"{subject}: {len(sessions)} sessions -> {out}")
    print(f"{'session':10s}" + "".join(f"{c:>14s}" for c in CLASSES) + "   (kept / in the lab's windows)")
    totals = {c: [0, 0] for c in CLASSES}
    for ses in sessions:
        target = out / ses.name / "clean_epochs.npz"
        if target.exists() and not force:
            raise SystemExit(f"{target} exists; add --force to overwrite")
        counts = extract_session(ses, target)
        for c in CLASSES:
            totals[c][0] += counts[c][0]
            totals[c][1] += counts[c][1]
        print(f"{ses.name:10s}" + "".join(f"{counts[c][1]:>7d} / {counts[c][0]:<5d}" for c in CLASSES))
    print(f"{'total':10s}" + "".join(f"{totals[c][1]:>7d} / {totals[c][0]:<5d}" for c in CLASSES))
    print("\nNext: python project_code.py evaluate   (recomputes, because the dataset changed)\n"
          "      python project_code.py subjects   (one accuracy table per subject)")


def subjects(offline=False):
    """`python project_code.py subjects [--offline]`: the accuracy of every subject in the dataset."""
    f = offline_labels() if offline else monitor_labels()
    pred = f["offline"] if offline else f["with_history"]
    label, session = f["label"], f["session"]
    subject = np.array([s.split("/")[0] for s in session])
    num = np.array([int(s.split("-")[-1]) for s in session])
    groups = []
    for s in sorted(set(subject)):
        m = subject == s
        if s == "sub_01":
            groups += [(f"sub_01 sessions 1-{DESIGN_LAST} (design)", m & (num <= DESIGN_LAST)),
                       (f"sub_01 sessions {DESIGN_LAST + 1}-36 (test)", m & (num > DESIGN_LAST))]
        else:
            groups.append((f"{s} (all {len(set(session[m]))} sessions)", m))
    print(f"{'OFFLINE' if offline else 'LIVE'} monitor, every subject in {DATA.name}. "
          f"Thresholds were chosen on sub_01 sessions 1-{DESIGN_LAST} only.\n")
    print(f"{'subject':34s}" + "".join(f"{c:>22s}" for c in CLASSES) + f"{'balanced':>10s}{'false alarms':>15s}")
    for name, m in groups:
        cells = []
        for c in CLASSES:
            mc = m & (label == c)
            cells.append(f"{int((pred[mc] == c).sum())}/{int(mc.sum())} ({100 * np.mean(pred[mc] == c):.1f}%)"
                         if mc.any() else "-")
        rest = m & (label == "STRAIGHT")
        fa = f"{int((pred[rest] != 'STRAIGHT').sum())}/{int(rest.sum())}"
        print(f"{name:34s}" + "".join(f"{x:>22s}" for x in cells) + f"{balanced(f, pred, m):9.2f}%{fa:>15s}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0] if argv else "evaluate"
    if command == "measure":
        measure(force=True)
    elif command == "evaluate" and "--offline" in argv:
        evaluate_offline(force="--force" in argv)
    elif command == "evaluate":
        evaluate(detail="--detail" in argv, fast="--fast" in argv, force="--force" in argv)
    elif command == "windows":
        windows(verify="--verify" in argv)
    elif command == "train":
        train()
    elif command == "train30":
        train_first_30()
    elif command == "test":
        test()
    elif command == "extract":
        extract(argv[1], force="--force" in argv)
    elif command == "subjects":
        subjects(offline="--offline" in argv)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
