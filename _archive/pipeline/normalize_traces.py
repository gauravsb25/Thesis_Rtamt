"""
normalize_traces.py -- build processed_norm_data/ from processed_traces/.

WHY
---
Between-subject gain is the pipeline's documented failure mode: sub-11's EEG
runs hotter than sub-01's (measured on T3 over the 3-second windows: 1.60x on
peak, 1.27x on MAD; pipeline/classify.py records up to 2-2.6x on raw peaks).
Any monitor with an absolute microvolt threshold therefore misfires the moment
it meets a subject whose gain it was not fitted on.

The fix applied here is to normalise ONCE, at the source, so every stage
downstream -- windowing, mining, thresholds, and the live monitor -- sees one
common scale instead of each re-inventing its own normalisation.

WHAT IS COMPUTED
----------------
Per (subject, session) and per channel, over that session's ENTIRE recording:

    centre = median over all samples in the session
    spread = median absolute deviation (MAD) over all samples

and each channel is written out as (v - centre) / spread. The statistics are
robust (median/MAD, not mean/std) so that a single artefact cannot define the
session's scale.

The session is the unit, not the subject, because amplifier gain and electrode
impedance are re-established at every setup -- two sessions from one subject
are no more comparable than two subjects.

NO LABELS ARE USED
------------------
centre and spread come from the raw signal alone. Normalising the session that
is being tested is therefore unsupervised calibration, not label leakage: a
deployed monitor obtains exactly these two numbers from a short baseline
recording before it starts, which is standard practice for a BCI.

WINDOWING COMMUTES WITH THIS
----------------------------
(v - centre)/spread is affine and time-invariant, and centre/spread are
constant across the session. So normalising the full trace and then cutting a
3-second window gives BIT-IDENTICAL output to cutting the window first and
then normalising it. That is what lets rtamt_layer/mine_monitors.py apply the
same session scale to the existing 3SEC_EXTRACTS windows and still be exactly
equivalent to windowing this corpus -- benchmarks stay comparable.

Run:
    python pipeline/normalize_traces.py
    python pipeline/normalize_traces.py --check      # verify commutativity
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.paths import bootstrap; bootstrap()
from lib import paths

import argparse
import json

import numpy as np
import pandas as pd

from lib.corpus_layout import list_corpus_csvs, parse_sub_ses

SOURCE = paths.at('3SEC_EXTRACTS')
DEST = paths.at('processed_norm_data')
SCALES_JSON = os.path.join(DEST, 'session_scales.json')

CHANNELS = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'T3', 'Cz', 'T4']
# Columns carried through untouched: time is an index, marker is categorical,
# and T3T4 / T3-T4 are derived from channels that are themselves normalised.
PASSTHROUGH = ['time', 'marker']


def session_scales(root=SOURCE):
    """{('sub/ses'): {channel: (centre, spread)}} over each session's full data."""
    pooled = {}
    for rel in list_corpus_csvs(root):
        if 'Logs' in rel:
            continue
        sub, ses = parse_sub_ses(rel)
        if sub is None:
            continue
        key = '%s_%s/ses-%s' % ('sub', sub, ses)
        df = pd.read_csv(os.path.join(root, rel))
        d = pooled.setdefault(key, {})
        for ch in CHANNELS:
            if ch in df.columns:
                d.setdefault(ch, []).append(df[ch].values.astype(float))

    scales = {}
    for key, chans in pooled.items():
        scales[key] = {}
        for ch, arrs in chans.items():
            v = np.concatenate(arrs)
            centre = float(np.median(v))
            spread = float(np.median(np.abs(v - centre)))
            if spread < 1e-9:                      # flat channel: leave it alone
                spread = float(v.std()) or 1.0
            scales[key][ch] = (centre, spread)
    return scales


def apply_scale(df, scale):
    """Normalised copy of `df`. Derived columns are rebuilt, not rescaled."""
    out = pd.DataFrame()
    for c in PASSTHROUGH:
        if c in df.columns:
            out[c] = df[c]
    for ch in CHANNELS:
        if ch in df.columns and scale and ch in scale:
            centre, spread = scale[ch]
            out[ch] = (df[ch].values.astype(float) - centre) / spread
        elif ch in df.columns:
            out[ch] = df[ch]
    # rebuild derived columns from the normalised channels so they stay
    # consistent rather than carrying the old microvolt scale
    if 'T3' in out.columns and 'T4' in out.columns:
        out['T3T4'] = out['T3'] - out['T4']
        out['T3-T4'] = out['T3'] - out['T4']
    return out


def build(source=SOURCE, dest=DEST, verbose=True):
    scales = session_scales(source)
    os.makedirs(dest, exist_ok=True)
    with open(SCALES_JSON.replace(DEST, dest), 'w') as fh:
        json.dump({k: {c: list(v) for c, v in d.items()}
                   for k, d in scales.items()}, fh, indent=1)

    n = 0
    for rel in list_corpus_csvs(source):
        if 'Logs' in rel:
            continue
        sub, ses = parse_sub_ses(rel)
        key = 'sub_%s/ses-%s' % (sub, ses)
        df = pd.read_csv(os.path.join(source, rel))
        out = apply_scale(df, scales.get(key))
        target = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        out.to_csv(target, index=False)
        n += 1
        if verbose and n % 100 == 0:
            print('  %d/%d' % (n, 442))
    if verbose:
        print('wrote %d normalised traces -> %s' % (n, dest))
        print('session scales -> %s' % SCALES_JSON.replace(DEST, dest))
    return scales, n


def check_commutes(scales):
    """Confirm normalise-then-window == window-then-normalise, bit for bit.

    This is what justifies mine_monitors.py applying the session scale to the
    pre-cut 3SEC_EXTRACTS windows instead of re-cutting this corpus.
    """
    rels = [r for r in list_corpus_csvs(SOURCE) if 'Logs' not in r][:8]
    worst = 0.0
    for rel in rels:
        sub, ses = parse_sub_ses(rel)
        sc = scales.get('sub_%s/ses-%s' % (sub, ses))
        df = pd.read_csv(os.path.join(SOURCE, rel))
        a = apply_scale(df, sc).iloc[100:868]                 # normalise, then cut
        b = apply_scale(df.iloc[100:868], sc)                 # cut, then normalise
        for ch in CHANNELS:
            if ch in a.columns:
                worst = max(worst, float(np.max(np.abs(
                    a[ch].values - b[ch].values))))
    print('commutativity check over %d traces: worst |diff| = %.3e' % (len(rels), worst))
    print('EXACT' if worst == 0.0 else ('within float error' if worst < 1e-12
                                        else 'MISMATCH -- investigate'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    print('normalising %s -> %s' % (SOURCE, DEST))
    scales, n = build()
    ks = sorted(scales)
    print('\n%d sessions scaled. Example spreads (MAD, microvolts):' % len(ks))
    for k in ks[:3] + ks[-2:]:
        row = ', '.join('%s=%.2f' % (c, v[1]) for c, v in
                        sorted(scales[k].items())[:4])
        print('  %-16s %s' % (k, row))
    if args.check:
        check_commutes(scales)


if __name__ == '__main__':
    main()
