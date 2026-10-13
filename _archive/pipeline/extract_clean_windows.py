"""Build extracted_npz_dataset: keep only windows whose label the signal can support.

WHY THIS EXISTS

``labelled_epochs_ds.npz`` slides a 3-second window forward in 0.1 s steps and gives a window
a movement's label whenever the two overlap by roughly 0.4 s. One movement therefore produces
about 35 windows carrying the same label, holding anything from a sliver of the movement to
all of it. Only 46% of the parent's movement windows contain the whole event, and the damage
falls almost entirely on laterality, which is read from the polarity of one brief deflection
that is simply absent when clipped:

    energy peak position     4-class     LEFT    RIGHT     DOWN
    dead centre               0.759      0.705    0.690    0.774
    at the window edge        0.644      0.494    0.504    0.765

DOWN barely moves, because it is read from an energy distribution across the head, which
survives a partial view.

HOW A WINDOW'S POSITION IS KNOWN -- NOTHING IS INFERRED

An earlier version of this script GUESSED where the movement sat inside each window from the
window's position in its run of identical labels. Auditing that against the recordings showed
the guess rested on false assumptions: it assumed a movement lasts 0.5 s (the truth is a
median of 1.38 s, and NONE is under 0.5 s) and that every label run is 36 windows (only 19%
are; they run from 11 to 92, because nearby movements merge into one run and runs are cut at
task-file boundaries).

So there is no guessing now. Every npz epoch is 768 raw samples copied verbatim out of one of
its session's task recordings, so normalised cross-correlation against that recording peaks in
exactly one place. All 39,269 windows of this corpus place at correlation 1.000000; a window
that failed to place would be dropped, never guessed. The ``ACT_`` annotations of the located
recording then give the movement's real onset, real duration and real class.

WHAT IS KEPT -- two selection modes

``--mode overlap`` (default). The rule asked for by the user:

    movement window   exactly one annotated movement overlaps the window, and at least
                      ``--min-overlap`` seconds of it are inside. Default 0.30 s.
    straight window   NO annotated movement intrudes into the 3 seconds at all -- not even
                      by one sample.

``--mode contained``. The stricter rule:

    movement window   the whole movement lies inside the window with at least ``--margin``
                      seconds of clearance at both ends. Movements longer than
                      ``3 - 2*margin`` seconds cannot satisfy this and contribute nothing.
    straight window   no movement comes within ``--guard`` seconds of the window.

Labels always come from the ANNOTATION, never from the npz. Both ACT_<class> and numbered
ACT<digits>_<class> annotations are recognized. The source contains ACT1_right in
sub_01/ses-22/task-right; the previous ACT_-only parser ignored that movement and incorrectly
relabelled 34 RIGHT epochs as STRAIGHT. Existing extracts made by that version must be
rebuilt before use. Unknown ACT-like annotations fail instead of silently becoming rest.

WHAT THIS IS NOT

A filtered subset. Accuracy on it is NOT comparable to accuracy on the full corpus and must
never be quoted as if it were. The full corpus is the deployment-realistic headline; this
subset answers the narrower question "how well does the monitor do when the movement is
genuinely in the window". Quote both and state the exclusion. A manifest and a README are
written beside the data so the filter can never be forgotten.

Usage
-----
    python pipeline/extract_clean_windows.py
    python pipeline/extract_clean_windows.py --min-overlap 0.30 --dry-run
    python pipeline/extract_clean_windows.py --mode contained --margin 0.25 --guard 1.0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.signal import fftconvolve

FS = 256
WINDOW_SAMPLES = 768
WINDOW_SECONDS = WINDOW_SAMPLES / FS
CLASSES = ("LEFT", "RIGHT", "DOWN", "STRAIGHT")
MATCH_MIN = 0.999           # a window is placed only if it matches at least this well
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from lib.annotation_labels import parse_act_label
SOURCE = os.path.join(ROOT, "dataset", "eeg_cleaned_data")
OUT_ROOT = os.path.join(ROOT, "extracted_npz_dataset")


def locate(template: np.ndarray, signal: np.ndarray):
    """Start index of ``template`` inside ``signal``, by normalised cross-correlation."""
    centred = template - template.mean()
    n = len(centred)
    numerator = fftconvolve(signal, centred[::-1], mode="valid")
    total = fftconvolve(signal, np.ones(n), mode="valid")
    square = fftconvolve(signal ** 2, np.ones(n), mode="valid")
    variance = np.maximum(square - total ** 2 / n, 1e-12)
    r = (numerator - total * centred.mean()) / np.sqrt(variance * float((centred ** 2).sum()))
    k = int(np.argmax(r))
    return k, float(r[k])


def place(epochs_ch0: np.ndarray, signals: dict):
    """Locate every epoch: returns (task, start_sample, correlation) per epoch.

    Consecutive epochs are 0.1 s apart, so the next position is usually predictable. The
    prediction is verified against the signal and a full search runs whenever it fails, which
    is what happens at a task boundary.
    """
    out = []
    task = None
    base_i = base_start = 0
    for i, template in enumerate(epochs_ch0):
        matched = False
        if task is not None:
            guess = base_start + int(round((i - base_i) * 0.1 * FS))
            signal = signals[task]
            if 0 <= guess <= len(signal) - WINDOW_SAMPLES:
                window = signal[guess:guess + WINDOW_SAMPLES]
                if float(np.corrcoef(window, template)[0, 1]) > MATCH_MIN:
                    out.append((task, guess,
                                float(np.corrcoef(window, template)[0, 1])))
                    matched = True
        if not matched:
            best = (None, -1, -2.0)
            for name, signal in signals.items():
                k, r = locate(template, signal)
                if r > best[2]:
                    best = (name, k, r)
            out.append(best)
            task, base_i, base_start = best[0], i, best[1]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=("overlap", "contained"), default="overlap",
                        help="'overlap': keep a movement window if at least --min-overlap "
                             "seconds of the movement is inside, and a straight window only "
                             "if no movement intrudes at all. 'contained': the whole "
                             "movement must fit with --margin clearance.")
    parser.add_argument("--min-overlap", type=float, default=0.30,
                        help="[overlap mode] seconds of movement a window must contain")
    parser.add_argument("--margin", type=float, default=0.25,
                        help="[contained mode] seconds the movement must clear both edges by")
    parser.add_argument("--guard", type=float, default=1.0,
                        help="[contained mode] seconds a straight window must be clear by")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be kept without writing anything")
    args = parser.parse_args()

    import mne
    mne.set_log_level("ERROR")

    files = sorted(glob.glob(os.path.join(SOURCE, "*", "*", "labelled_epochs_ds.npz")))
    if not files:
        raise SystemExit(f"no source npz under {SOURCE}")
    if args.mode == "overlap":
        rule = (f"movement: at least {args.min_overlap}s of the movement inside; "
                f"straight: no movement at all inside the 3 s")
    else:
        rule = (f"movement: whole movement inside with {args.margin}s clearance; "
                f"straight: {args.guard}s clear of any movement")
    print(f"{len(files)} source sessions   mode={args.mode}")
    print(f"rule -- {rule}")
    print(f"{'session':18s}{'in':>7s}{'out':>7s}{'kept':>6s}   "
          + "".join(f"{c:>12s}" for c in CLASSES))

    manifest = []
    total_in = {c: 0 for c in CLASSES}
    total_out = {c: 0 for c in CLASSES}
    worst_match = 1.0
    unusable_moves = 0
    all_moves = 0
    for path in files:
        parts = os.path.normpath(path).split(os.sep)
        subject, session = parts[-3], parts[-2]
        data = np.load(path, allow_pickle=True)
        labels_raw = np.array([str(x).upper() for x in data["labels"]])
        usable = np.isin(labels_raw, CLASSES)
        epochs = data["epochs"][usable]
        npz_labels = labels_raw[usable]
        source_index = np.flatnonzero(usable)

        signals, annotations = {}, {}
        pattern = os.path.join(SOURCE, subject, session, "ica_eeg", "augmented_task-*_raw.fif")
        for fif in sorted(glob.glob(pattern)):
            name = os.path.basename(fif).split("task-")[1].split("_")[0]
            raw = mne.io.read_raw_fif(fif, preload=True)
            signals[name] = raw.get_data(picks=["Fp1"])[0] * 1e6
            annotations[name] = []
            for de, o, du in zip(raw.annotations.description, raw.annotations.onset,
                                 raw.annotations.duration):
                label = parse_act_label(str(de))
                if label is None:
                    continue
                if not str(de).upper().startswith("ACT_"):
                    print(f"annotation alias in {fif}: {de} -> ACT_{label.lower()}")
                annotations[name].append((float(o), float(du), label))
        if not signals:
            print(f"{subject + '/' + session:18s}   no fif -- session skipped entirely")
            continue
        for acts in annotations.values():
            for _, du, _ in acts:
                all_moves += 1
                if args.mode == "contained" and du > WINDOW_SECONDS - 2 * args.margin:
                    unusable_moves += 1

        placed = place(epochs[:, 0] * 1e6, signals)
        worst_match = min(worst_match, min(p[2] for p in placed))

        n = len(epochs)
        keep = np.zeros(n, dtype=bool)
        truth = np.array(["STRAIGHT"] * n, dtype="<U8")
        onset = np.full(n, np.nan)
        duration = np.full(n, np.nan)
        inside = np.full(n, np.nan)
        task_of = np.array([p[0] or "" for p in placed])
        start_of = np.array([p[1] for p in placed], dtype=int)

        for i, (task, start, r) in enumerate(placed):
            if task is None or r <= MATCH_MIN:
                continue                                   # never guess a placement
            t0 = start / FS
            t1 = t0 + WINDOW_SECONDS
            overlapping = [(o, du, c, min(t1, o + du) - max(t0, o))
                           for o, du, c in annotations[task] if o < t1 and o + du > t0]
            if overlapping:
                o, du, c, ov = max(overlapping, key=lambda z: z[3])
                onset[i], duration[i], inside[i] = o - t0, du, ov
            if args.mode == "overlap":
                if len(overlapping) == 1 and overlapping[0][3] >= args.min_overlap:
                    keep[i], truth[i] = True, overlapping[0][2]
                elif not overlapping:
                    keep[i], truth[i] = True, "STRAIGHT"
            else:
                near = [(o, du, c) for o, du, c in annotations[task]
                        if o - args.guard < t1 and o + du + args.guard > t0]
                if len(overlapping) == 1:
                    o, du, c, _ = overlapping[0]
                    if (o - t0 >= args.margin
                            and o + du - t0 <= WINDOW_SECONDS - args.margin):
                        keep[i], truth[i] = True, c
                elif not near:
                    keep[i], truth[i] = True, "STRAIGHT"

        per = {c: (int((npz_labels == c).sum()), int((keep & (truth == c)).sum()))
               for c in CLASSES}
        for c in CLASSES:
            total_in[c] += per[c][0]
            total_out[c] += per[c][1]
        print(f"{subject + '/' + session:18s}{n:7d}{int(keep.sum()):7d}{keep.mean():6.0%}   "
              + "".join(f"{per[c][1]}/{per[c][0]:>5}".rjust(12) for c in CLASSES))

        manifest.append(dict(
            subject=subject, session=session, source=os.path.relpath(path, ROOT),
            windows_in=int(n), windows_out=int(keep.sum()),
            match_min=float(min(p[2] for p in placed)),
            per_class={c: dict(before=per[c][0], after=per[c][1]) for c in CLASSES}))
        if args.dry_run:
            continue

        out_dir = os.path.join(OUT_ROOT, subject, session)
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(out_dir, "clean_epochs.npz"),
            epochs=epochs[keep], labels=truth[keep],
            npz_label=npz_labels[keep],
            source_index=source_index[keep],
            movement_offset=onset[keep],
            movement_duration=duration[keep],
            movement_inside=inside[keep],
            task=task_of[keep], start_sample=start_of[keep],
            mode=np.array([args.mode]),
            min_overlap=np.array([args.min_overlap]),
            margin=np.array([args.margin]), guard=np.array([args.guard]))

    grand_in, grand_out = sum(total_in.values()), sum(total_out.values())
    print()
    print(f"{'TOTAL':18s}{grand_in:7d}{grand_out:7d}{grand_out / max(grand_in, 1):6.0%}   "
          + "".join(f"{total_out[c]}/{total_in[c]:>5}".rjust(12) for c in CLASSES))
    print(f"worst placement correlation across the corpus: {worst_match:.6f}")
    if args.mode == "contained":
        print(f"movements too long to fit with this margin: {unusable_moves}/{all_moves}")
    if args.dry_run:
        print("\ndry run: nothing written")
        return

    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(os.path.join(OUT_ROOT, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(mode=args.mode, min_overlap=args.min_overlap,
                       margin=args.margin, guard=args.guard, rule=rule,
                       window_seconds=WINDOW_SECONDS, step_seconds=0.1,
                       label_source="fif ACT_ annotations; windows placed by "
                                    "cross-correlation at >= %.3f" % MATCH_MIN,
                       worst_match=worst_match,
                       totals=dict(before=total_in, after=total_out),
                       sessions=manifest), fh, indent=2)
    with open(os.path.join(OUT_ROOT, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(f"""# extracted_npz_dataset

A FILTERED subset of `dataset/eeg_cleaned_data/*/*/labelled_epochs_ds.npz`. Labels and
geometry come from the `ACT_` annotations of the source recordings, not from the npz and not
from any model of where a movement sits.

Generated by `pipeline/extract_clean_windows.py`, mode `{args.mode}`.

## The rule

{rule}

## How a window's position is known

Each npz epoch is 768 raw samples taken verbatim from one of its session's task recordings,
so normalised cross-correlation against that recording peaks in exactly one place. Every
window here placed at correlation >= {worst_match:.6f}. A window that failed to place would be
dropped, never guessed. Its real movement onset, duration and class then come straight from
the annotations.

## Counts

| class | in source | kept | |
|---|---|---|---|
""")
        for c in CLASSES:
            frac = total_out[c] / max(total_in[c], 1)
            fh.write(f"| {c} | {total_in[c]} | {total_out[c]} | {frac:.0%} |\n")
        fh.write(f"| **total** | **{grand_in}** | **{grand_out}** | "
                 f"**{grand_out / max(grand_in, 1):.0%}** |\n")
        fh.write("""
## Fields

| field | meaning |
|---|---|
| `epochs` | (n, 9, 768) as in the source, unmodified |
| `labels` | the class, **from the annotation** |
| `npz_label` | what the source npz called it, for comparison |
| `source_index` | the row this came from in the source npz |
| `movement_offset` | seconds into the window at which the movement begins |
| `movement_duration` | the annotated duration of that movement |
| `movement_inside` | seconds of that movement actually inside the window |
| `task`, `start_sample` | where the window sits in its source recording |

## Reporting rule

Accuracy on this subset is **not** comparable to accuracy on the full corpus and must never
be quoted as if it were. The full corpus is the deployment-realistic headline; this subset
answers the narrower question *"how well does the monitor do when the movement is genuinely
in the window"*. Quote both, and state the exclusion.
""")
    print(f"\nwritten to {OUT_ROOT}")
    print("manifest.json and README.md record the rule, the placement quality and the counts")


if __name__ == "__main__":
    main()
