"""LIVE REPLAY: every recorded run streamed sample by sample through the real StreamMonitor (RTAMT), as a
headset would deliver it, on the ICA-cleaned recordings AND on the raw (pre-ICA, filtered-only) recordings.

A verdict every 0.25 s, each about the 3 s ending at that sample (current default monitor, censoring off).
Scored the way a live user experiences it:
    movement   the FIRST non-STRAIGHT verdict whose window touches the movement (or up to 3 s after it) is the
               command the system would act on: right class / wrong class / missed (none); delay from movement
               onset to that verdict
    rest       verdicts whose 3-s window lies >= 1 s away from any movement; a run of consecutive non-STRAIGHT
               rest verdicts counts as ONE false command; reported per minute of rest scanned
Each run is streamed on its own (runs are separate recordings), so the first 1-2 s after a run starts has no
background yet and the monitor uses the contrast gate there -- as a live system would right after switch-on.
"""
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from monitoring import StreamMonitor
from monitoring.rtamt_signals import CHANNELS, FS
from monitoring.offline_monitor import load_recording, dataset_recording_path

DATA = ROOT / "extracted_npz_dataset_v2"
ICA = ROOT / "dataset" / "eeg_cleaned_data"
RAW = Path("D:/errp/eeg_cleaned_data")
W, STEP = 3 * FS, FS // 4


def load_raw(session, task, run):
    subject, ses = session.split("/")
    p = RAW / subject / ses / "pp_eeg" / f"{subject.replace('_', '-')}_{ses}_task-{task}_run-{int(run):02d}_eeg.csv"
    with open(p, newline="") as fh:
        r = csv.reader(fh)
        head = next(r)
        cols = [head.index(c) for c in CHANNELS]
        return np.array([[float(row[c]) for c in cols] for row in r if row]).T      # already microvolts


def runs_of(path):
    d = np.load(path, allow_pickle=True)
    t, r, o = d["task"].astype(str), d["run"].astype(int), d["run_offset"].astype(int)
    mo, md = d["movement_offset"].astype(float), d["movement_duration"].astype(float)
    out = []
    for key in sorted(set(zip(t, r))):
        m = (t == key[0]) & (r == key[1]) & np.isfinite(mo)
        if key[0] == "straight":
            out.append((key, None))
        elif m.any():
            on = float(np.median(o[m] + mo[m] * FS))
            out.append((key, (on, on + float(np.median(md[m])) * FS, key[0].upper())))
    return out


def replay(x, move):
    """Stream one run; returns event outcome and rest statistics."""
    verdicts = StreamMonitor(decide_every=STEP).push_block(x)              # [(end sample, label)]
    event = None
    if move:
        on, off, cls = move
        first = next(((end, lab) for end, lab in verdicts
                      if lab not in ("STRAIGHT", "INVALID") and end > on and end - W < off + 3 * FS), None)
        if first is None:
            event = (cls, "missed", None)
        else:
            event = (cls, "right" if first[1] == cls else f"wrong:{first[1]}", (first[0] - on) / FS)
    rest_n = false_cmds = false_verdicts = 0
    prev_false = False
    for end, lab in verdicts:
        start = end - W
        if move and not (end < move[0] - FS or start > move[1] + FS):
            prev_false = False
            continue
        rest_n += 1
        is_false = lab not in ("STRAIGHT", "INVALID")
        false_verdicts += is_false
        false_cmds += is_false and not prev_false
        prev_false = is_false
    return event, rest_n, false_verdicts, false_cmds


def session(path):
    p = Path(path)
    ses = f"{p.parent.parent.name}/{p.parent.name}"
    rows = []
    for (task, run), move in runs_of(path):
        for source in ("ica", "raw"):
            try:
                x = load_recording(dataset_recording_path(ICA, ses, task, run)) if source == "ica" else load_raw(ses, task, run)
            except FileNotFoundError:
                continue
            if x.shape[1] < W + STEP:
                continue
            verdicts = StreamMonitor(decide_every=STEP).push_block(x)
            rows.append((ses, task, run, source, move, [(int(e), str(l)) for e, l in verdicts]))
    return rows


def split_of(ses):
    n = int(ses[-2:])
    return "sub_11" if ses.startswith("sub_11") else ("sub_01 1-20" if n <= 20 else "sub_01 21-36")


def score(verdicts, move, k):
    """Commands under the rule 'act when the same movement class holds for k verdicts in a row'."""
    fired, streak, prev = [], 0, None
    for end, lab in verdicts:
        streak = streak + 1 if lab == prev else 1
        prev = lab
        if lab not in ("STRAIGHT", "INVALID") and streak == k:
            fired.append((end, lab))
    event = None
    if move:
        on, off, cls = move
        first = next(((e, l) for e, l in fired if e > on and e - W < off + 3 * FS), None)
        event = (cls, "missed", None) if first is None else (cls, "right" if first[1] == cls else "wrong", (first[0] - on) / FS)
    rest = [(e, l) for e, l in verdicts if not move or e < move[0] - FS or e - W > move[1] + FS]
    false = [e for e, l in fired if not move or e < move[0] - FS or e - W > move[1] + FS]
    return event, len(rest), len(false)


def table(rows, k):
    for source in ("ica", "raw"):
        for split in ("sub_01 1-20", "sub_01 21-36", "sub_11"):
            rs = [r for r in rows if r[3] == source and split_of(r[0]) == split]
            res = [score(r[5], r[4], k) for r in rs]
            ev = [x[0] for x in res if x[0]]
            n = len(ev)
            right = sum(e[1] == "right" for e in ev)
            wrong = sum(e[1] == "wrong" for e in ev)
            missed = sum(e[1] == "missed" for e in ev)
            delay = np.median([e[2] for e in ev if e[1] == "right"])
            rest = sum(x[1] for x in res)
            false = sum(x[2] for x in res)
            minutes = rest * STEP / FS / 60
            print(f"k={k} {source:3s} {split:13s} right {right:3d}/{n} ({100 * right / n:5.1f}%)  wrong {100 * wrong / n:5.1f}%"
                  f"  missed {100 * missed / n:5.1f}%  delay {delay:4.2f} s  false commands {false:4d} = {false / minutes:5.2f}/min")


if __name__ == "__main__":
    out = HERE / "replay_verdicts.npy"
    if out.exists():
        rows = list(np.load(out, allow_pickle=True))
    else:
        files = sorted(DATA.glob("*/ses-*/clean_epochs.npz"))
        with Pool(6) as pool:
            rows = [r for part in pool.map(session, [str(f) for f in files]) for r in part]
        np.save(out, np.array(rows, dtype=object), allow_pickle=True)
    for k in (1, 2, 3, 4, 6, 8):
        table(rows, k)
        print()
