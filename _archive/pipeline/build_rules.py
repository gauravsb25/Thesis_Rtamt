"""
Data-driven rule builder for classify.py, constrained to relationships
TeLEX's STL specification mining actually validated.

Why this exists: the previous SELECTED_RULES in classify.py were fit on a
98-file subset and then applied to the current 442-file, 2-subject dataset,
where per-subject median EEG peak amplitude differs by ~2-2.6x (sub-11 runs
"hotter" than sub-01 on every channel -- see analyze_full_dataset.py output).
Absolute microvolt thresholds tuned on one subject's scale silently fail on
the other's, which is the single biggest reason for the wall of UNCERTAIN
and X results in the classify.py log.

This script does greedy forward feature selection PER CLASS using grouped
(by subject+session) K-fold cross-validation, so a feature only gets picked
if it actually generalizes across sessions/subjects, not just because it
happens to separate classes in-sample. Per the project's direction: TeLEX
picks WHICH channel relationships are eligible (see telex_relationships.py,
which mines formulas.txt/formulas_v2.txt/formula_finalresults/ for every
channel and channel-pair TeLEX's synthesis actually touched or confirmed),
and this script fits and cross-validates the exact numeric thresholds fresh
on the current 3-second-window data -- it does not reuse TeLEX's own
synthesized numbers, which were computed on differently-sized windows.

Output: prints the selected SELECTED_RULES / MIN_FRAC dict (ready to paste
into classify.py) plus CV and in-sample accuracy numbers.

WHICH NUMBER TO QUOTE
---------------------
This script reports "MEAN CV accuracy" around 82%. That figure validates the
THRESHOLDS and MIN_FRAC per fold, but NOT the choice of which rules to use:
greedy_select scores candidates with cv_bacc_for_ruleset(), which averages over
all five folds, and the final number is reported on those same folds.

tools/nested_cv.py removes that leakage by re-running the entire selection
inside each fold. It reports ~72%. Its inner control reproduces ~82% on the
folds the selection saw, confirming the ~10-point gap is selection optimism
rather than a difference in implementation.

Quote the nested figure for generalisation. Run:

    python tools/nested_cv.py

"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.paths import bootstrap; bootstrap()
from lib import paths

import re
import os
import numpy as np

import lib.subject_norm as subject_norm
import pandas as pd

from pipeline.telex_relationships import get_candidate_relationships
from lib.constants import CLASSES

FEAT_TABLE = paths.results("full_feature_table.csv")
MAX_RULES_PER_CLASS = int(os.environ.get("MAX_RULES", 7))
MIN_CANDIDATES = int(os.environ.get("MIN_CANDIDATES", 0))
MIN_USEFUL_WORST_BACC = float(os.environ.get("MIN_WB", 0.50))
N_FOLDS = 5
SEED = 42
# Build easiest-separable classes first so harder classes' feature search
# accounts for the fact that STRAIGHT/DOWN votes are already "claimed".
CLASS_BUILD_ORDER = ['STRAIGHT', 'DOWN', 'RIGHT', 'LEFT']

CALIB_CSV = paths.results("subject_calibration.csv")

df = pd.read_csv(FEAT_TABLE)
df['group'] = df['subject'] + "_" + df['session'].astype(str)
non_feature_cols = {'file', 'label', 'subject', 'session', 'overall_peak', 'overall_rms', 'group'}
feature_cols = [c for c in df.columns if c not in non_feature_cols]

labels = df['label'].values
N = len(df)

# ----------------------------------------------------------------------
# grouped K-fold split (deterministic, by subject_session so a fold never
# splits a single recording session across train/test)
# ----------------------------------------------------------------------
# CV_GROUPING=session (default) keeps every trial of a recording session on
# the same side of the split. CV_GROUPING=trial shuffles individual trials
# instead -- the naive split. It exists only to demonstrate how much a naive
# split inflates the score; never quote a number produced with it.
CV_GROUPING = os.environ.get("CV_GROUPING", "session")

rng = np.random.RandomState(SEED)
if CV_GROUPING == "session":
    groups = sorted(df['group'].unique())
    rng.shuffle(groups)
    fold_lists = np.array_split(groups, N_FOLDS)
    fold_of_group = {g: i for i, glist in enumerate(fold_lists) for g in glist}
    df['fold'] = df['group'].map(fold_of_group)
elif CV_GROUPING == "trial":
    idx = rng.permutation(len(df))
    df['fold'] = 0
    for f, chunk in enumerate(np.array_split(idx, N_FOLDS)):
        df.loc[df.index[chunk], 'fold'] = f
else:
    raise ValueError(f"unknown CV_GROUPING {CV_GROUPING!r}")
fold_arr = df['fold'].values

# ----------------------------------------------------------------------
# TeLEX-endorsed relationship filter -- a pairwise feature is only eligible
# pair. Single-channel features are always eligible (every one of the 8
# channels appears in some TeLEX template).
# ----------------------------------------------------------------------
_TELEX_SINGLES, _TELEX_PAIRS, _TELEX_PER_CLASS = get_candidate_relationships()
_PAIR_RE = re.compile(r'^([A-Za-z0-9]+)[-/]([A-Za-z0-9]+)_(netdiff|absmeandiff|ratio|onsetlag)$')


# TELEX_FILTER=0 disables the channel-pair restriction entirely. Used to
# measure the ceiling: how good could the classifier get if TeLEX endorsed
# EVERY channel pair? That bounds what any re-run of the synthesis can buy.
TELEX_FILTER = os.environ.get("TELEX_FILTER", "1") not in ("0", "", "false", "False")


# ALLOWED_PAIRS lets a specific channel-pair subset be tested, e.g.
#   ALLOWED_PAIRS="T3-T4,Cz-T3,Fz-T4,Fp1-Fp2"
# Empty means use whatever telex_relationships.py endorses.
_AP = os.environ.get("ALLOWED_PAIRS", "").strip()
ALLOWED_PAIRS = None
if _AP:
    ALLOWED_PAIRS = {tuple(sorted(p.split("-"))) for p in _AP.split(",") if "-" in p}


def telex_allows(col):
    if not TELEX_FILTER:
        return True
    if ALLOWED_PAIRS is not None:
        m = _PAIR_RE.match(col)
        if not m:
            return True
        a, b, _ = m.groups()
        return tuple(sorted((a, b))) in ALLOWED_PAIRS
    m = _PAIR_RE.match(col)
    if not m:
        return True  # single-channel feature -- always eligible
    a, b, _ = m.groups()
    return tuple(sorted((a, b))) in _TELEX_PAIRS


print(f"TeLEX-endorsed channel pairs (candidate pool restricted to these for pairwise "
      f"features): {sorted(_TELEX_PAIRS)}\n")

# ----------------------------------------------------------------------
# pull the pooled threshold/direction/worst_sub_bacc already computed by
# analyze_full_dataset.py for each class, restricted to TeLEX-endorsed
# relationships, as the candidate pool
# ----------------------------------------------------------------------
candidate_pools = {}
for cls in CLASSES:
    cand_df = pd.read_csv(paths.results(f"full_feature_analysis_{cls}.csv"))
    strict = cand_df[cand_df['worst_subject_bacc'] >= MIN_USEFUL_WORST_BACC]
    # A hard class can be left with almost nothing by a fixed cutoff -- LEFT
    # already keeps only 123 of 425 candidates, and its weakest SELECTED rule
    # clears 0.56 by 0.027. Adding subjects lowers every worst-subject score,
    # so a fixed bar starves the hard classes first. Guarantee a floor: if the
    # cutoff leaves fewer than MIN_CANDIDATES, keep the best MIN_CANDIDATES
    # instead. Selection is still cross-validated, so a weak candidate only
    # enters the rule set if it earns its place on held-out folds.
    if len(strict) < MIN_CANDIDATES:
        strict = cand_df.nlargest(MIN_CANDIDATES, 'worst_subject_bacc')
    cand_df = strict
    cand_df = cand_df[cand_df['feature'].apply(telex_allows)]
    cand_df = cand_df.sort_values('worst_subject_bacc', ascending=False)
    candidate_pools[cls] = cand_df.to_dict('records')

# precompute numpy value arrays per feature (NaN-safe)
values_by_col = {c: df[c].values.astype(float) for c in feature_cols}


def fires(col, thresh, direction):
    v = values_by_col[col]
    with np.errstate(invalid='ignore'):
        if direction == 'above':
            f = v >= thresh
        else:
            f = v <= thresh
    f = np.where(np.isnan(v), False, f)
    return f


# ----------------------------------------------------------------------
# RULE WEIGHTING
# A rule's vote weight was originally its worst-subject balanced accuracy,
# which lives in [0.5, 0.95] -- so the strongest rule outvotes the weakest by
# under 2x even though one is near-perfect and the other near-chance.
# "advantage" rescales to how far above chance the rule actually is,
# 2*(bacc-0.5), which spreads the same rules over [0, 0.9] and lets a strong
# rule dominate a marginal one. Set WEIGHT_MODE to compare.
# ----------------------------------------------------------------------
WEIGHT_MODE = os.environ.get("WEIGHT_MODE", "advantage")

# HOW A CANDIDATE RULE IS SCORED ACROSS THE CV FOLDS
#   mean            average fold score. Rewards fitting the folds, which is
#                   why it carried ~11 points of selection optimism.
#   mean_minus_std  average minus the spread. A rule that is good on average
#                   but erratic across folds is penalised, so the search
#                   prefers rules that behave the SAME on every session --
#                   which is what has to survive on an unseen subject.
#   min             the worst fold. Strictest; slightly below mean_minus_std.
#
# Measured by nested CV, where the whole selection re-runs inside each fold:
#   mean            70.3% honest / 81.4% inner  -> 11.1 points of optimism
#   min             71.8% honest / 78.6% inner  ->  6.8
#   mean_minus_std  72.2% honest / 79.2% inner  ->  7.0   <- default
SELECT_CRITERION = os.environ.get("SELECT_CRITERION", "mean")


def rule_weight(bacc):
    b = float(bacc)
    if WEIGHT_MODE == "worst_bacc":
        return b
    if WEIGHT_MODE == "advantage":
        return max(0.0, 2.0 * (b - 0.5))
    if WEIGHT_MODE == "advantage_sq":
        return max(0.0, 2.0 * (b - 0.5)) ** 2
    if WEIGHT_MODE == "logodds":
        b = min(max(b, 0.5001), 0.9999)
        return float(np.log(b / (1.0 - b)))
    if WEIGHT_MODE == "uniform":
        return 1.0
    raise ValueError(f"unknown WEIGHT_MODE {WEIGHT_MODE!r}")


def composite_score(rules):
    """weighted-vote fraction-of-max-weight score, vectorized over all rows"""
    score = np.zeros(N)
    total_w = sum(w for _, _, _, w in rules)
    for col, thresh, direction, w in rules:
        score += np.where(fires(col, thresh, direction), w, 0.0)
    return score / total_w if total_w > 0 else score


def best_min_frac_bacc(score, is_class, idx):
    """search MIN_FRAC grid on idx subset, return (best_bacc, best_frac)"""
    s = score[idx]
    ic = is_class[idx]
    n_pos, n_neg = ic.sum(), (~ic).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.5
    best_bacc, best_frac = 0.0, 0.5
    for frac in np.linspace(0.05, 0.95, 37):
        pred = s >= frac
        recall = (pred & ic).sum() / n_pos
        spec = ((~pred) & (~ic)).sum() / n_neg
        bacc = (recall + spec) / 2
        if bacc > best_bacc:
            best_bacc, best_frac = bacc, frac
    return best_bacc, best_frac


# THRESHOLD LEAKAGE
# Thresholds come from full_feature_analysis_*.csv, which analyze_full_dataset.py
# fits on the WHOLE corpus -- including the rows that later serve as a test fold.
# Only MIN_FRAC was refit per fold, so the reported CV was optimistic: each rule's
# cut point had already seen the data it is scored on.
# LEAKFREE_THRESH=1 refits every threshold on the training rows of each fold.
LEAKFREE_THRESH = os.environ.get("LEAKFREE_THRESH", "1") not in ("0", "", "false", "False")


def fit_threshold(col, is_class, idx):
    """Best threshold/direction for one feature, using only rows in idx."""
    v = values_by_col[col][idx]
    ic = is_class[idx]
    ok = ~np.isnan(v)
    v, ic = v[ok], ic[ok]
    n_pos, n_neg = ic.sum(), (~ic).sum()
    if n_pos == 0 or n_neg == 0 or len(v) == 0:
        return None, 'above'
    best = (0.5, None, 'above')
    uniq = np.unique(v)
    if len(uniq) > 120:
        uniq = np.quantile(uniq, np.linspace(0, 1, 120))
    for t in uniq:
        above = v >= t
        b1 = ((above & ic).sum() / n_pos + ((~above) & (~ic)).sum() / n_neg) / 2
        if b1 > best[0]:
            best = (b1, t, 'above')
        b2 = (((~above) & ic).sum() / n_pos + (above & (~ic)).sum() / n_neg) / 2
        if b2 > best[0]:
            best = (b2, t, 'below')
    return best[1], best[2]


def composite_score_refit(rules, cls, train_idx):
    """Same weighted vote, but every threshold refitted on train_idx only."""
    is_class = (labels == cls)
    score = np.zeros(N)
    total_w = sum(w for _, _, _, w in rules)
    for col, _thr, _dir, w in rules:
        t, d = fit_threshold(col, is_class, train_idx)
        if t is None:
            continue
        score += np.where(fires(col, t, d), w, 0.0)
    return score / total_w if total_w > 0 else score


def cv_bacc_for_ruleset(rules, cls):
    """grouped 5-fold CV: fit MIN_FRAC on 4 folds, score held-out fold"""
    is_class = (labels == cls)
    score = composite_score(rules)
    fold_baccs = []
    for f in range(N_FOLDS):
        train_idx = np.where(fold_arr != f)[0]
        test_idx = np.where(fold_arr == f)[0]
        fold_score = composite_score_refit(rules, cls, train_idx) if LEAKFREE_THRESH else score
        _, frac = best_min_frac_bacc(fold_score, is_class, train_idx)
        s_test = fold_score[test_idx]
        ic_test = is_class[test_idx]
        n_pos, n_neg = ic_test.sum(), (~ic_test).sum()
        if n_pos == 0 or n_neg == 0:
            continue
        pred = s_test >= frac
        recall = (pred & ic_test).sum() / n_pos
        spec = ((~pred) & (~ic_test)).sum() / n_neg
        fold_baccs.append((recall + spec) / 2)
    if not fold_baccs:
        return 0.5
    if SELECT_CRITERION == "min":
        return float(np.min(fold_baccs))
    if SELECT_CRITERION == "mean_minus_std":
        return float(np.mean(fold_baccs) - np.std(fold_baccs))
    return float(np.mean(fold_baccs))


def greedy_select(cls):
    is_class = (labels == cls)
    candidates = candidate_pools[cls]
    selected = []
    selected_cols = set()
    current_cv = 0.5
    for _ in range(MAX_RULES_PER_CLASS):
        best_gain = 0.0
        best_cand = None
        best_cv = current_cv
        for cand in candidates:
            col = cand['feature']
            if col in selected_cols:
                continue
            thresh, direction, w = cand['threshold'], cand['direction'], rule_weight(cand['worst_subject_bacc'])
            trial_rules = selected + [(col, thresh, direction, w)]
            cv = cv_bacc_for_ruleset(trial_rules, cls)
            gain = cv - current_cv
            if gain > best_gain:
                best_gain = gain
                best_cand = (col, thresh, direction, w)
                best_cv = cv
        if best_cand is None or best_gain < 0.004:
            break
        selected.append(best_cand)
        selected_cols.add(best_cand[0])
        current_cv = best_cv
    return selected, current_cv


# OBJECTIVE MISMATCH
# greedy_select maximises one-vs-rest balanced accuracy for a class in
# isolation, but the reported metric is MULTICLASS accuracy. LEFT and RIGHT
# compete for the same trials, so a rule that improves LEFT's own bacc can
# still cost accuracy by stealing RIGHT trials. This pass refines the assembled
# rule set against the objective that is actually reported.
# TESTED AND REJECTED. Refining against multiclass accuracy raises what
# build_rules.py reports from 82.7% to 86.6%, but nested cross-validation shows
# it delivers NOTHING on held-out sessions: 71.9% without it, 71.8% with it.
# The +4 points are the search fitting the folds it is scored on. Kept so the
# experiment is reproducible; do not enable it for a reported result.
# Verify with:  python tools/nested_cv.py --refine
MULTICLASS_REFINE = os.environ.get("MULTICLASS_REFINE", "0") not in ("0", "", "false", "False")
REFINE_POOL = int(os.environ.get("REFINE_POOL", 40))


def multiclass_cv(rules_by_class):
    """Grouped CV accuracy of the whole assembled classifier, MIN_FRAC per fold."""
    accs = []
    for f in range(N_FOLDS):
        tr = np.where(fold_arr != f)[0]
        te = np.where(fold_arr == f)[0]
        if len(te) == 0:
            continue
        mf = {}
        for c in CLASSES:
            sc = composite_score(rules_by_class[c])
            _, frac = best_min_frac_bacc(sc, labels == c, tr)
            mf[c] = frac
        acc, _, _ = evaluate_multiclass(rules_by_class, mf, te)
        accs.append(acc)
    return float(np.mean(accs)) if accs else 0.0


def refine_multiclass(rules_by_class, max_rounds=4):
    """Coordinate ascent: add / drop / swap one rule at a time, keeping any
    change that improves MULTICLASS CV accuracy."""
    cur = {c: list(v) for c, v in rules_by_class.items()}
    best = multiclass_cv(cur)
    print("")
    print(f"Multiclass refinement -- starting CV {best*100:.1f}%")
    for rnd in range(max_rounds):
        improved = False
        for cls in CLASSES:
            have = {r[0] for r in cur[cls]}
            # try ADDING one candidate
            for cand in candidate_pools[cls][:REFINE_POOL]:
                col = cand['feature']
                if col in have or cand['threshold'] is None:
                    continue
                trial = {c: list(v) for c, v in cur.items()}
                trial[cls] = cur[cls] + [(col, cand['threshold'], cand['direction'],
                                          rule_weight(cand['worst_subject_bacc']))]
                sc = multiclass_cv(trial)
                if sc > best + 1e-6:
                    best, cur, improved = sc, trial, True
                    print(f"   + {cls:<9} {col:<32} -> {best*100:.1f}%")
                    have = {r[0] for r in cur[cls]}
            # try DROPPING one rule
            for r in list(cur[cls]):
                if len(cur[cls]) <= 1:
                    break
                trial = {c: list(v) for c, v in cur.items()}
                trial[cls] = [x for x in cur[cls] if x[0] != r[0]]
                sc = multiclass_cv(trial)
                if sc > best + 1e-6:
                    best, cur, improved = sc, trial, True
                    print(f"   - {cls:<9} {r[0]:<32} -> {best*100:.1f}%")
        if not improved:
            break
    print(f"Multiclass refinement -- final CV {best*100:.1f}%")
    return cur


def evaluate_multiclass(rules_by_class, min_frac_by_class, idx=None):
    if idx is None:
        idx = np.arange(N)
    scores = {c: composite_score(rules_by_class[c]) * sum(w for *_, w in rules_by_class[c]) for c in CLASSES}
    totals = {c: sum(w for *_, w in rules_by_class[c]) for c in CLASSES}
    fracs = {c: (scores[c] / totals[c] if totals[c] > 0 else np.zeros(N)) for c in CLASSES}
    winners = []
    for i in idx:
        qualifying = {c: fracs[c][i] for c in CLASSES if fracs[c][i] >= min_frac_by_class[c]}
        # Same fallback as classify.py: if nothing clears MIN_FRAC, take the
        # highest-scoring class rather than refusing. Refusal scores as an
        # error regardless, so this can only help -- and it keeps the CV
        # number measuring the same decision rule the classifier applies.
        winners.append(max(qualifying, key=qualifying.get) if qualifying
                       else max(CLASSES, key=lambda c: fracs[c][i]))
    winners = np.array(winners)
    true = labels[idx]
    overall_acc = (winners == true).mean()
    per_class = {}
    for c in CLASSES:
        m = true == c
        if m.sum() == 0:
            continue
        per_class[c] = (winners[m] == c).mean()
    uncertain_rate = (winners == "UNCERTAIN").mean()
    return overall_acc, per_class, uncertain_rate


ALL_CHANNELS = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'T3', 'Cz', 'T4']


def channels_used(rules):
    used = set()
    for col, *_ in rules:
        for ch in ALL_CHANNELS:
            if ch in col:
                used.add(ch)
    return used


def full_pipeline_cv_accuracy(rules_by_class):
    """Grouped 5-fold CV of the WHOLE assembled multiclass classifier
    (MIN_FRAC refit per fold, per class) -- the real metric that matters,
    since a change that looks harmless in isolation can still destabilize
    the argmax competition between classes."""
    accs = []
    for f in range(N_FOLDS):
        train_idx = np.where(fold_arr != f)[0]
        test_idx = np.where(fold_arr == f)[0]
        fold_min_frac = {}
        for c in CLASSES:
            score = composite_score(rules_by_class[c])
            _, frac = best_min_frac_bacc(score, labels == c, train_idx)
            fold_min_frac[c] = frac
        acc, _, _ = evaluate_multiclass(rules_by_class, fold_min_frac, test_idx)
        accs.append(acc)
    return float(np.mean(accs)), float(np.std(accs))


def coverage_pass(rules_by_class, tolerance=0.003):
    """If a channel is unused anywhere in the assembled ruleset, try adding
    the best candidate feature involving it as a corroborating vote -- but
    only accept the change if it's verified against the FULL multiclass
    pipeline's grouped CV accuracy (not just the isolated per-class binary
    score), since the argmax competition between classes can be destabilized
    by a change that looks harmless class-by-class. This is a redundancy/
    robustness addition, not a required feature -- if no channel placement
    survives full-pipeline validation, that channel is reported as
    genuinely redundant given the other 7."""
    used = set()
    for rules in rules_by_class.values():
        used |= channels_used(rules)
    missing = [c for c in ALL_CHANNELS if c not in used]
    if not missing:
        return rules_by_class, [], []
    base_mean, base_std = full_pipeline_cv_accuracy(rules_by_class)
    added_log, rejected_log = [], []
    for ch in missing:
        best = None  # (cls, cand, mean, std)
        for cls in CLASSES:
            existing_cols = {r[0] for r in rules_by_class[cls]}
            for cand in candidate_pools[cls][:15]:
                col = cand['feature']
                if ch not in col or col in existing_cols:
                    continue
                trial_rules_by_class = dict(rules_by_class)
                trial_rules_by_class[cls] = rules_by_class[cls] + [
                    (col, cand['threshold'], cand['direction'], cand['worst_subject_bacc'])]
                mean, std = full_pipeline_cv_accuracy(trial_rules_by_class)
                if mean >= base_mean - tolerance and std <= base_std + 0.02:
                    if best is None or mean > best[2]:
                        best = (cls, cand, mean, std)
        if best is not None:
            cls, cand, mean, std = best
            col, thresh, direction, w = cand['feature'], cand['threshold'], cand['direction'], rule_weight(cand['worst_subject_bacc'])
            rules_by_class[cls] = rules_by_class[cls] + [(col, thresh, direction, w)]
            added_log.append((ch, cls, col, thresh, direction, w, mean, std))
            base_mean, base_std = mean, std
        else:
            rejected_log.append(ch)
    return rules_by_class, added_log, rejected_log


def main():
    rules_by_class = {}
    min_frac_by_class = {}

    print("=" * 90)
    print("GREEDY, CROSS-SESSION-VALIDATED RULE SELECTION")
    print("=" * 90)
    for cls in CLASS_BUILD_ORDER:
        selected, cv_bacc = greedy_select(cls)
        rules_by_class[cls] = selected
        print(f"\n{cls}: CV balanced-accuracy = {cv_bacc:.3f}")
        for col, thresh, direction, w in selected:
            print(f"    ('{col}', {round(float(thresh),3)}, '{direction}', {round(float(w),3)}),")

    # reorder back to canonical class order for output/classify.py
    rules_by_class = {c: rules_by_class[c] for c in CLASSES}

    if MULTICLASS_REFINE:
        rules_by_class = refine_multiclass(rules_by_class)
        rules_by_class = {c: rules_by_class[c] for c in CLASSES}

    print("\n" + "=" * 90)
    print("ALL-CHANNEL COVERAGE PASS (validated against full multiclass pipeline CV)")
    print("=" * 90)
    rules_by_class, added_log, rejected_log = coverage_pass(rules_by_class)
    if added_log:
        for ch, cls, col, thresh, direction, w, mean, std in added_log:
            print(f"  + added {col!r} to {cls} for {ch} coverage "
                  f"(thr={round(float(thresh),3)} {direction}, w={round(float(w),3)}, "
                  f"full CV after={mean*100:.1f}%+/-{std*100:.1f}%)")
    if rejected_log:
        print(f"  channels with NO safe placement found (every candidate degraded full-pipeline CV): {rejected_log}")
    if not added_log and not rejected_log:
        print("  All 8 channels already represented; nothing to add.")

    for cls in CLASSES:
        is_class = (labels == cls)
        score = composite_score(rules_by_class[cls])
        _, frac = best_min_frac_bacc(score, is_class, np.arange(N))
        # 3 dp, not 2: the search grid steps by 0.025, so rounding to 2
        # decimals turns 0.325 into 0.32 and lowers the bar for that class.
        # build_rules then evaluated the un-rounded value while printing the
        # rounded one, reporting ~1.4 points more than classify.py delivers.
        min_frac_by_class[cls] = round(float(frac), 3)

    print("\n" + "=" * 90)
    print("FINAL VALIDATION")
    print("=" * 90)

    overall_acc, per_class, uncertain_rate = evaluate_multiclass(rules_by_class, min_frac_by_class)
    print(f"\nPooled in-sample accuracy: {overall_acc*100:.1f}%   UNCERTAIN rate: {uncertain_rate*100:.1f}%")
    for c, a in per_class.items():
        print(f"  {c:10}: {a*100:.1f}%")

    for sub in df['subject'].unique():
        idx = np.where(df['subject'].values == sub)[0]
        acc, pc, unc = evaluate_multiclass(rules_by_class, min_frac_by_class, idx)
        print(f"\n[{sub}] accuracy: {acc*100:.1f}%   UNCERTAIN rate: {unc*100:.1f}%")
        for c, a in pc.items():
            print(f"  {c:10}: {a*100:.1f}%")

    # honest grouped CV of the FULL assembled multiclass classifier
    print("\nGrouped 5-fold CV of the assembled multiclass classifier (MIN_FRAC refit per fold):")
    fold_accs = []
    for f in range(N_FOLDS):
        train_idx = np.where(fold_arr != f)[0]
        test_idx = np.where(fold_arr == f)[0]
        fold_min_frac = {}
        for c in CLASSES:
            score = composite_score(rules_by_class[c])
            _, frac = best_min_frac_bacc(score, labels == c, train_idx)
            fold_min_frac[c] = frac
        acc, pc, unc = evaluate_multiclass(rules_by_class, fold_min_frac, test_idx)
        fold_accs.append(acc)
        print(f"  fold {f}: acc={acc*100:.1f}%  n={len(test_idx)}  per_class={ {k: round(v,2) for k,v in pc.items()} }")
    print(f"  MEAN CV accuracy: {np.mean(fold_accs)*100:.1f}% (+/- {np.std(fold_accs)*100:.1f}%)")

    print("\n" + "=" * 90)
    print("PYTHON DICT FOR classify.py")
    print("=" * 90)
    print("SELECTED_RULES = {")
    for cls in CLASSES:
        print(f"    '{cls}': [")
        for col, thresh, direction, w in rules_by_class[cls]:
            print(f"        ('{col}', {round(float(thresh), 6)}, '{direction}', {round(float(w), 3)}),")
        print("    ],")
    print("}")
    print("\nMIN_FRAC = {")
    for c in CLASSES:
        print(f"    '{c}': {min_frac_by_class[c]},")
    print("}")


if __name__ == "__main__":
    main()

