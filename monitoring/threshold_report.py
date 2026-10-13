"""Choose, explain and test the five thresholds on any sessions of any subject.

    python -m monitoring.threshold_report                     # explain the current thresholds
    python -m monitoring.threshold_report --select            # choose them on sub_01 ses 1-29
    python -m monitoring.threshold_report --select --train sub_11 --test sub_01:30-36
    python -m monitoring.threshold_report --select --train sub_01:1-22 --test sub_01:23-29 sub_11
    python -m monitoring.threshold_report --select --save my_thresholds.json
    python -m monitoring.threshold_report --gate cusum --direction since     # the tested options

Data specs: ``sub_01:1-29``, ``sub_01:1-22,25``, ``sub_11`` (all of its sessions).

The four RTAMT specifications test five dimensionless thresholds (g, d, q, n, v; see
:class:`monitoring.rtamt_signals.RtamtConfig`). g is the contrast ratio ``gate_ratio`` for the
default gate and the CUSUM threshold ``cusum_min`` for ``--gate cusum``. The SINCE direction
level h (``--direction since``) is structural and is not selected here. Nothing here is machine
learning: a threshold is a number in a fixed rule, chosen by sweeping it and counting correct
windows.

SELECTION RULE (``--select``, the one-standard-error rule):
  1. start from the current thresholds;
  2. sweep one threshold with the other four fixed; the best value is the one with the highest
     4-class balanced accuracy on the training windows;
  3. move only if the gain over the current value is larger than its standard error, measured
     over training sessions (the paired per-session differences); then take, among the values
     within one standard error of the best, the one nearest the current value;
  4. repeat for g, d, q, n, v until nothing moves.
Sessions are the independent units because windows of one session are strongly correlated.
``--rule max`` takes the plain best value instead.

Every window is measured with the monitor's own :func:`compute_signals`, and the rules are
scored exactly as RTAMT scores them at the last sample (``once`` = maximum so far,
``historically`` = minimum so far, ``and`` = minimum, ``or`` = maximum, and the SINCE direction
formula is true for LEFT exactly when the last excursion reaching h of the largest was negative).
Sampled training windows are cross-checked against the real RTAMT monitor before anything is
reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import numpy as np

from .rtamt_monitors import HeadMovementMonitor, class_formulas
from .rtamt_signals import CHANNELS, RtamtConfig, compute_signals

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "extracted_npz_dataset_v2"
CACHE = ROOT / ".cache" / "measurements"
CLASSES = ("LEFT", "RIGHT", "DOWN", "STRAIGHT")
MEASURES = ("gate", "delta", "deflect", "nod", "sideways", "direction", "sideways_fp1fp2", "sideways_fp2t3")
KEYS = ("g", "d", "q", "n", "v", "v1", "v2")
COLUMN = dict(g=0, d=1, q=2, n=3, v=4, v1=6, v2=7)
THRESHOLDS_FIELDS = ("gate_ratio", "cusum_min", "delta_range_min", "quiet_min", "down_threshold",
                     "down_veto_prominence", "down_veto_fp1fp2", "down_veto_fp2t3")
MEASUREMENT_VERSION = "8 values: gate delta deflect nod sideways direction sideways_fp1fp2 sideways_fp2t3"


def fields(config: RtamtConfig):
    """Which RtamtConfig field each threshold key sets, for this design."""
    out = dict(g="cusum_min" if config.gate_statistic == "cusum" else "gate_ratio",
               d="delta_range_min", q="quiet_min", n="down_threshold", v="down_veto_prominence")
    if config.down_veto_fp1fp2 is not None:
        out["v1"] = "down_veto_fp1fp2"
    if config.down_veto_fp2t3 is not None:
        out["v2"] = "down_veto_fp2t3"
    return out


def threshold_text(config: RtamtConfig):
    """key: (measure, rule text, what it means)."""
    if config.gate_statistic == "cusum":
        g = ("gate", "once(cusum >= g)",
             f"CUSUM of log(energy / quietest 0.5 s) beyond {config.cusum_kappa:g} nats, in nat-seconds "
             "(did the energy rise AND stay up?)")
    else:
        g = ("gate", "once(contrast >= g)",
             "loudest 0.5 s over quietest 0.5 s of the window (was the window lopsided in time?)")
    return {
        "g": g,
        "d": ("delta", "once(delta >= d)",
              "spread of the slow-wave share across 0.5 s slices (did the spectrum slow down?)"),
        "q": ("deflect", "once(deflect >= q)",
              "mean channel excursion over its own spread (did the electrodes deflect?)"),
        "n": ("nod", "nod >= n",
              "midline-vs-temporal energy + F3/F4 common-mode symmetry, in nats (nod or turn?)"),
        "v": ("sideways", "historically(sideways <= v)",
              "largest F4-T3 excursion over its spread (was there a big sideways swing?)"),
        "v1": ("sideways_fp1fp2", "historically(sideways_fp1fp2 <= v1)",
               "largest Fp1-Fp2 excursion over its spread (a sideways eye movement across the forehead?)"),
        "v2": ("sideways_fp2t3", "historically(sideways_fp2t3 <= v2)",
               "largest Fp2-T3 excursion over its spread (a sideways swing on the second diagonal pair?)"),
    }


def parse_spec(spec: str):
    """'sub_01:1-30' / 'sub_01:1-22,25' / 'sub_11' -> sorted list of clean_epochs.npz paths."""
    subject, _, sessions = spec.partition(":")
    folder = DATA / subject
    available = {int(re.search(r"(\d+)", p.parent.name).group(1)): p
                 for p in folder.glob("ses-*/clean_epochs.npz")}
    if not available:
        raise SystemExit(f"no sessions found for '{spec}' under {folder}")
    if sessions in ("", "all"):
        wanted = sorted(available)
    else:
        wanted = []
        for part in sessions.split(","):
            a, _, b = part.partition("-")
            wanted += list(range(int(a), int(b or a) + 1))
    missing = [s for s in wanted if s not in available]
    if missing:
        raise SystemExit(f"'{spec}': session(s) {missing} not found")
    return [available[s] for s in sorted(set(wanted))]


def since_direction(lateral, level):
    """+1 (LEFT) if the last excursion reaching ``level`` was negative, else -1 (RIGHT).

    Boolean value of ``(not (lateral >= h)) since (lateral <= -h)`` at the last sample.
    """
    idx = np.arange(len(lateral))
    last_right = int(np.max(np.where(lateral >= level, idx, -1)))
    last_left = int(np.max(np.where(lateral <= -level, idx, -1)))
    return 1.0 if last_left > last_right else -1.0


def measure_window(window, config: RtamtConfig | None = None):
    """The eight values the specifications read at the last sample of one window."""
    cfg = config or RtamtConfig()
    s = compute_signals(window, cfg)
    gate = s["cusum"].max() if cfg.gate_statistic == "cusum" else s["contrast"].max()
    if cfg.direction_rule == "since":
        direction = since_direction(s["lateral"], cfg.direction_level)
    else:
        direction = s["turn"][-1]
    return (gate, s["delta"].max(), s["deflect"][-1], s["nod"][-1], s["sideways"].max(), direction,
            s["sideways_fp1fp2"].max(), s["sideways_fp2t3"].max())


def _measurement_key(config: RtamtConfig):
    """Everything about the design that changes a measurement (thresholds do not)."""
    geometry = {k: v for k, v in asdict(config).items() if k not in THRESHOLDS_FIELDS}
    geometry["measurement_version"] = MEASUREMENT_VERSION
    return hashlib.sha1(json.dumps(geometry, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _cache_path(path, config):
    dataset = path.parent.parent.parent.name
    return CACHE / f"{dataset}_{path.parent.parent.name}_{path.parent.name}_{_measurement_key(config)}.npz"


def _fingerprint(path):
    st = path.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def _measure_file(path, config):
    d = np.load(path, allow_pickle=True)
    x = d["epochs"][:, :len(CHANNELS)] * 1e6                   # stored in volts
    values = np.array([measure_window(w, config) for w in x])
    labels = np.array([str(v).upper() for v in d["labels"]])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_cache_path(path, config), values=values, labels=labels,
                        key=_fingerprint(path))
    return values, labels


def load_measurements(files, recompute=False, config: RtamtConfig | None = None):
    """(values (n,8), labels (n,), session names (n,)) for these files, cached per session."""
    cfg = config or RtamtConfig()
    todo = [p for p in files if recompute or not _cache_path(p, cfg).exists()
            or str(np.load(_cache_path(p, cfg))["key"]) != _fingerprint(p)]
    if todo:
        print(f"measuring {len(todo)} session(s) with compute_signals (cached afterwards) ...",
              flush=True)
        from multiprocessing import Pool
        with Pool(max(1, min(len(todo), (os.cpu_count() or 2) - 1))) as pool:
            pool.map(partial(_measure_file, config=cfg), todo)
    parts = [np.load(_cache_path(p, cfg), allow_pickle=True) for p in files]
    values = np.concatenate([c["values"] for c in parts])
    labels = np.concatenate([c["labels"] for c in parts])
    names = np.concatenate([[f"{p.parent.parent.name}/{p.parent.name}"] * len(c["labels"])
                            for p, c in zip(files, parts)])
    return values, labels, names


def decide(values, g, d, q, n, v, v1=None, v2=None):
    """The four specifications' robustness at the last sample, then satisfied_class.

    v1 / v2 are the Fp1-Fp2 and Fp2-T3 nod vetoes (columns 7 and 8); None means switched off.
    """
    values = np.asarray(values, dtype=float)
    gate, delta, deflect, nod, sideways, direction = values[:, :6].T
    move = np.minimum(np.minimum(gate - g, delta - d), deflect - q)
    no_swing = v - sideways
    if v1 is not None:
        no_swing = np.minimum(no_swing, v1 - values[:, 6])
    if v2 is not None:
        no_swing = np.minimum(no_swing, v2 - values[:, 7])
    down_ev = np.minimum(nod - n, no_swing)
    straight = -move
    down = np.minimum(move, down_ev)
    left = np.minimum(np.minimum(move, -down_ev), direction)
    return np.where(straight > 0, "STRAIGHT",
                    np.where(down >= 0, "DOWN", np.where(left >= 0, "LEFT", "RIGHT")))


def balanced(labels, pred):
    return float(np.mean([np.mean(pred[labels == c] == c) for c in CLASSES if (labels == c).any()]))


def per_session(labels, pred, names):
    return np.array([balanced(labels[names == s], pred[names == s]) for s in np.unique(names)])


def thresholds_of(config: RtamtConfig):
    return {k: getattr(config, f) for k, f in fields(config).items()}


def config_with(th, base: RtamtConfig | None = None):
    base = base or RtamtConfig()
    return replace(base, **{f: float(th[k]) for k, f in fields(base).items() if k in th})


def _sig4(x):
    return float(f"{x:.4g}")


def select_thresholds(values, labels, names, start, rule="1se", passes=5):
    """Choose g, d, q, n, v on these windows. Returns (thresholds, trace lines)."""
    if rule == "1se" and len(np.unique(names)) < 2:
        raise SystemExit("the one-standard-error rule needs at least two training sessions")
    th, trace = dict(start), []
    for p in range(passes):
        moved = False
        for k in [k for k in KEYS if k in th]:
            x = values[:, COLUMN[k]]
            qs = np.quantile(x, np.linspace(0.01, 0.995, 200))
            grid = np.unique(np.r_[qs, (qs[1:] + qs[:-1]) / 2, th[k]])
            preds = [decide(values, **{**th, k: t}) for t in grid]
            acc = np.array([balanced(labels, pr) for pr in preds])
            i_cur = int(np.flatnonzero(grid == th[k])[0])
            i_best = int(acc.argmax())
            gain = acc[i_best] - acc[i_cur]
            if rule == "max":
                se = 0.0
                new = grid[i_best] if gain > 1e-12 else th[k]
            else:
                diff = per_session(labels, preds[i_best], names) - per_session(labels, preds[i_cur], names)
                se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
                if gain > se:
                    ok = grid[acc >= acc[i_best] - se]
                    new = ok[np.argmin(np.abs(ok - th[k]))]
                else:
                    new = th[k]
            new = _sig4(new) if not np.isclose(new, th[k]) else th[k]
            verdict = "keep" if np.isclose(new, th[k]) else f"move to {new:g}"
            trace.append(f"  pass {p+1}  {k}: current {th[k]:<9g} {100*acc[i_cur]:6.2f}%   best "
                         f"{grid[i_best]:<9.4g} {100*acc[i_best]:6.2f}%   gain {100*gain:5.2f} pts"
                         f"   SE {100*se:5.2f} pts   -> {verdict}")
            if not np.isclose(new, th[k]):
                th[k], moved = float(new), True
        if not moved:
            break
    return th, trace


DESIGN_FIELDS = ("gate_statistic", "direction_rule", "direction_level", "cusum_kappa",
                 "cusum_smooth_seconds", "cusum_reference_seconds")


def save_thresholds(path, th, meta, config: RtamtConfig | None = None):
    cfg = config or RtamtConfig()
    design = {k: getattr(cfg, k) for k in DESIGN_FIELDS}
    design["down_veto_fp1fp2"] = cfg.down_veto_fp1fp2
    design["down_veto_fp2t3"] = cfg.down_veto_fp2t3
    Path(path).write_text(json.dumps({**design, **{f: th[k] for k, f in fields(cfg).items() if k in th},
                                      **meta}, indent=2))


def load_thresholds(path) -> RtamtConfig:
    """RtamtConfig from a file written by ``--save``; unknown keys are ignored."""
    data = json.loads(Path(path).read_text())
    known = set(asdict(RtamtConfig()))
    return replace(RtamtConfig(), **{k: v for k, v in data.items() if k in known})


def cross_check(files, th, config: RtamtConfig, k=60, seed=0):
    """Scored decisions == the real RTAMT monitor on randomly sampled windows."""
    rng = np.random.default_rng(seed)
    cfg = config_with(th, config)
    mon, same = HeadMovementMonitor(cfg), 0
    for _ in range(k):
        d = np.load(files[rng.integers(len(files))], allow_pickle=True)
        x = d["epochs"][rng.integers(len(d["epochs"])), :len(CHANNELS)] * 1e6
        same += decide(np.array([measure_window(x, cfg)]), **th)[0] == mon.classify(x)
    return same, k


def explain(values, labels, th, config: RtamtConfig):
    mv, rest = labels != "STRAIGHT", labels == "STRAIGHT"
    down, lat = labels == "DOWN", np.isin(labels, ["LEFT", "RIGHT"])
    texts, names = threshold_text(config), fields(config)
    for key in [k for k in KEYS if k in th]:
        measure, rule, meaning = texts[key]
        x = values[:, MEASURES.index(measure)]
        c = th[key]
        print(f"\n[{key}] {names[key]} = {c:g}    rule: {rule}")
        print(f"    {meaning}")
        if key in "gdq":
            print(f"    separation: {100*np.mean(x[mv] >= c):5.1f}% of movement windows pass, "
                  f"{100*np.mean(x[rest] >= c):5.1f}% of rest windows pass")
        elif key == "n":
            print(f"    separation: {100*np.mean(x[down] >= c):5.1f}% of DOWN windows reach n, "
                  f"{100*np.mean(x[lat] >= c):5.1f}% of LEFT/RIGHT windows do")
        else:
            print(f"    separation: {100*np.mean(x[lat] > c):5.1f}% of LEFT/RIGHT windows exceed {key}, "
                  f"{100*np.mean(x[down] > c):5.1f}% of DOWN windows do")
        score = lambda t: balanced(labels, decide(values, **{**th, key: t}))
        lo, hi = np.quantile(x, [0.02, 0.995])
        fine = np.unique(np.r_[np.linspace(lo, hi, 400), c])
        fs = np.array([score(t) for t in fine])
        best = fine[fs.argmax()]
        plateau = fine[fs >= fs.max() - 0.005]
        if key == "n":
            coarse = c + np.array([-1.5, -1.0, -0.6, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6, 1.0])
        else:
            coarse = c * np.array([0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5])
        print("    sweep (other thresholds fixed)       4-class")
        for t in np.unique(np.r_[np.round(coarse, 4), best]):
            mark = "  <-- chosen" if np.isclose(t, c) else ("  <-- best" if np.isclose(t, best) else "")
            s = score(t)
            print(f"      {key} = {t:9.4f}   {100 * s:6.2f}%  {'#' * max(0, int(round((s - 0.5) * 100)))}{mark}")
        inside = plateau.min() <= c <= plateau.max()
        print(f"    best {best:.4f} ({100*fs.max():.2f}%); within 0.5 points of best: "
              f"{plateau.min():.4f} .. {plateau.max():.4f}; chosen {'inside' if inside else 'OUTSIDE'}"
              f" ({100*(fs.max() - score(c)):.2f} points below best)")
        if key in "gdq" and not np.isclose(best, c):
            print(f"    rest windows passing this test: {100*np.mean(x[rest] >= c):.1f}% at the chosen "
                  f"{c:g} vs {100*np.mean(x[rest] >= best):.1f}% at the best {best:.4g}")


def classwise(labels, pred):
    return "  ".join(f"{c} {int(((labels == c) & (pred == c)).sum())}/{int((labels == c).sum())}"
                     for c in CLASSES if (labels == c).any())


def main(argv=None):
    ap = argparse.ArgumentParser(description="Choose, explain and test the five thresholds.")
    ap.add_argument("--train", nargs="+", default=["sub_01:1-29"],
                    help="sessions to explain/choose on (default sub_01:1-29)")
    ap.add_argument("--test", nargs="*", default=[], help="sessions to evaluate the rules on")
    ap.add_argument("--select", action="store_true", help="choose the thresholds on --train")
    ap.add_argument("--rule", choices=("1se", "max"), default="1se",
                    help="selection rule: one-standard-error (default) or plain maximum")
    ap.add_argument("--gate", choices=("cusum", "contrast"), default=None,
                    help="gate statistic (default: the configuration's, contrast)")
    ap.add_argument("--direction", choices=("since", "peak"), default=None,
                    help="direction rule (default: the configuration's, peak)")
    ap.add_argument("--start", help="JSON of starting thresholds (default: the current ones)")
    ap.add_argument("--save", help="write the chosen thresholds to this JSON file")
    ap.add_argument("--recompute", action="store_true", help="re-measure every window")
    args = ap.parse_args(argv)

    base_cfg = load_thresholds(args.start) if args.start else RtamtConfig()
    if args.gate:
        base_cfg = replace(base_cfg, gate_statistic=args.gate)
    if args.direction:
        base_cfg = replace(base_cfg, direction_rule=args.direction)

    train_files = [p for spec in args.train for p in parse_spec(spec)]
    test_sets = [(spec, parse_spec(spec)) for spec in args.test]
    all_files = sorted(set(train_files) | {p for _, fs in test_sets for p in fs})
    load_measurements(all_files, args.recompute, base_cfg)   # measure anything missing, once
    values, labels, names = load_measurements(train_files, config=base_cfg)

    th = thresholds_of(base_cfg)
    line = "=" * 96
    print(line)
    print(f"DESIGN: gate = {base_cfg.gate_statistic}, direction = {base_cfg.direction_rule}"
          + (f" (h = {base_cfg.direction_level:g})" if base_cfg.direction_rule == "since" else ""))
    counts = "  ".join(f"{c} {int((labels == c).sum())}" for c in CLASSES)
    print(f"TRAINING DATA: {' '.join(args.train)}   ({len(np.unique(names))} sessions, "
          f"{len(labels)} windows: {counts})")
    if args.select:
        print(f"SELECTION ({'one-standard-error rule' if args.rule == '1se' else 'plain maximum'}),"
              f" starting from {'the current thresholds' if not args.start else args.start}")
        th, trace = select_thresholds(values, labels, names, th, args.rule)
        print("\n".join(trace))
        print("chosen: " + "  ".join(f"{k}={th[k]:g}" for k in KEYS if k in th))
    else:
        print("EXPLAINING the current thresholds (add --select to choose them on the training data)")
    same, k = cross_check(train_files, th, base_cfg)
    print(f"check: these decisions equal the real RTAMT monitor on {same}/{k} sampled windows")
    pred = decide(values, **th)
    print(f"4-class balanced accuracy on the training data: {100 * balanced(labels, pred):.2f}%"
          f"   ({classwise(labels, pred)})")
    print(line)
    explain(values, labels, th, base_cfg)

    cfg = config_with(th, base_cfg)
    print("\n" + line)
    print("THE FOUR RULES (RTAMT STL, evaluated at the last sample of each 3-second window)")
    print(line)
    for name, f in class_formulas(cfg).items():
        print(f"{name}:\n    {f}\n")
    if test_sets:
        print("EVALUATION on data not used for choosing:")
        for spec, fs in test_sets:
            v, lab, _ = load_measurements(fs, config=base_cfg)
            p = decide(v, **th)
            overlap = " (OVERLAPS the training data)" if set(fs) & set(train_files) else ""
            print(f"  {spec:14s} 4-class {100 * balanced(lab, p):6.2f}%   {classwise(lab, p)}{overlap}")
    if args.save:
        save_thresholds(args.save, th, {"trained_on": args.train, "rule": args.rule if args.select
                                        else "current"}, base_cfg)
        print(f"\nsaved to {args.save}. Use it with:\n"
              f"    from monitoring import HeadMovementMonitor\n"
              f"    from monitoring.threshold_report import load_thresholds\n"
              f"    monitor = HeadMovementMonitor(load_thresholds('{args.save}'))")
    else:
        names_ = fields(base_cfg)
        print("\nPython: HeadMovementMonitor(RtamtConfig(" + ", ".join(
            [f"gate_statistic={base_cfg.gate_statistic!r}", f"direction_rule={base_cfg.direction_rule!r}"]
            + [f"{names_[k]}={th[k]:g}" for k in KEYS if k in th]) + "))")


if __name__ == "__main__":
    main()
