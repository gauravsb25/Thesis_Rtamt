"""
Compiles the set of channel relationships that TeLEX's STL specification
mining actually considered and validated, so the rule-builder can restrict
its search to physiologically/formally motivated candidates instead of an
unconstrained combinatorial sweep over every possible channel pair.

Two sources are combined:
  1. formulas.txt + formulas_v2.txt -- every channel name and channel-pair
     token that appears in any tried STL template (this defines "TeLEX
     considered this relationship at all").
  2. formula_finalresults/grouped_synthesis_v2_results.csv -- the one
     synthesis run that was done correctly (class-grouped, multi-subject,
     verified against held-out other-class traces -- see
     run_synthesis_grouped.py). For each class, relationships that came
     back `satisfied=True` with a low `neg_satisfied_frac` (holds for the
     whole class, rarely true of other classes) are flagged as
     "TeLEX-confirmed" for that specific class and are prioritized first.

This module has no side effects other than printing a report when run
directly; import `get_candidate_relationships()` from other scripts.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.paths import bootstrap; bootstrap()
from lib import paths

import os
import re
import pandas as pd
from lib.constants import CLASSES

CHANNELS = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'T3', 'Cz', 'T4']


def _extract_relationships_from_formula_text(text):
    """Return the set of single channels and (a, b) channel pairs mentioned
    in a block of STL template text, e.g. 'F[0,5]((T3 > th?..) & (T4 < ..))'
    or '{Cz/T3}' or 'T3-T4'."""
    singles = set()
    pairs = set()

    # explicit {A-B} / {A/B} infix terms
    for m in re.finditer(r'\{?\s*([A-Za-z0-9]+)\s*[-/]\s*([A-Za-z0-9]+)\s*\}?', text):
        a, b = m.group(1), m.group(2)
        if a in CHANNELS and b in CHANNELS:
            pairs.add(tuple(sorted((a, b))))

    # any standalone channel mention
    for ch in CHANNELS:
        if re.search(rf'\b{ch}\b', text):
            singles.add(ch)

    return singles, pairs


def relationships_from_formula_files(*paths):
    singles, pairs = set(), set()
    for path in paths:
        try:
            with open(path, 'r') as f:
                text = f.read()
        except FileNotFoundError:
            continue
        s, p = _extract_relationships_from_formula_text(text)
        singles |= s
        pairs |= p
    return singles, pairs


# A mined formula counts as CONFIRMED for a class only if it held for that
# class while failing on most of the held-out other-class trials. Lower =
# stricter = fewer, more discriminative pairs survive.
NEG_MAX = float(os.environ.get("NEG_MAX", 0.6))


def relationships_confirmed_by_grouped_synthesis(csv_path, neg_satisfied_max=None):
    neg_satisfied_max = NEG_MAX if neg_satisfied_max is None else neg_satisfied_max
    """Per class, which single channels / pairs appeared in a synthesized
    formula that was satisfied for the whole sampled class AND held for
    <= neg_satisfied_max of the held-out other-class sample (i.e. actually
    discriminative, not just universally true)."""
    confirmed = {c: {'singles': set(), 'pairs': set()} for c in CLASSES}
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return confirmed

    df = df[(df['satisfied'] == True) & (df['neg_satisfied_frac'] <= neg_satisfied_max)]
    for _, row in df.iterrows():
        cls = row['class']
        if cls not in CLASSES:
            continue
        s, p = _extract_relationships_from_formula_text(str(row['formula']))
        confirmed[cls]['singles'] |= s
        confirmed[cls]['pairs'] |= p
    return confirmed


def get_candidate_relationships():
    """Returns (all_singles, all_pairs, per_class_confirmed) where
    per_class_confirmed[cls] = {'singles': set, 'pairs': set} are the
    relationships TeLEX's grouped synthesis actually confirmed as
    discriminative for that specific class."""
    # ICA-ONLY SOURCES. Everything here comes from clean_final_icaeeg_results/,
    # the TeLEX runs performed on the ica_eeg corpus. The older results under
    # formula_finalresults/ were mined from pre-ICA (pp_eeg) traces and must
    # not feed the classifier -- that mismatch is what this rewiring fixes.
    all_singles, all_pairs = relationships_from_formula_files(
        paths.at("synthesis", "formulas_ALL_run_synthesis_grouped.txt"))
    per_class = relationships_confirmed_by_grouped_synthesis(
        paths.at("clean_final_icaeeg_results", "grouped_synthesis_all_grouped_results.csv"))
    # union in whatever the per-class confirmation surfaced too, in case a
    # relationship only ever showed up inside a synthesized (not template) formula
    for cls in CLASSES:
        all_singles |= per_class[cls]['singles']
        all_pairs |= per_class[cls]['pairs']
    return all_singles, all_pairs, per_class


if __name__ == '__main__':
    all_singles, all_pairs, per_class = get_candidate_relationships()
    print("=" * 90)
    print("ALL channels TeLEX's templates ever referenced:")
    print(" ", sorted(all_singles))
    print("\nALL channel pairs TeLEX's templates ever referenced:")
    print(" ", sorted(all_pairs))
    print("=" * 90)
    for cls in CLASSES:
        print(f"\n{cls}: TeLEX-CONFIRMED (grouped synthesis, satisfied for whole class,")
        print("       held for <=60% of held-out other-class sample):")
        print("  singles:", sorted(per_class[cls]['singles']))
        print("  pairs  :", sorted(per_class[cls]['pairs']))
