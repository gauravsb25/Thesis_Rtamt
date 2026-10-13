"""
Grouped K-fold splitting, without sklearn.

The pipeline is rule-based end to end -- no models are fitted anywhere -- so
the only thing the project ever needed from sklearn was a splitter. Carrying a
machine-learning dependency for that invites the misreading that a model is
involved somewhere, so the splitter lives here instead.

Folds are formed over GROUPS, never individual trials. A group is one
(subject, session): all of its trials go to the same side of every split.
Trials within a session share a person, a day and an electrode placement, so
splitting them apart would measure "can it recognise this session again"
rather than "can it handle a new session".

`grouped_folds` mirrors sklearn's GroupKFold closely enough to drop in.
`stratified_grouped_folds` additionally tries to spread each class evenly
across folds, which matters here because STRAIGHT is only 24 of 442 trials and
a naive split can leave a fold with none.
"""

import numpy as np


def grouped_folds(groups, n_folds=5, seed=42):
    """Yield (train_idx, test_idx) with every group wholly on one side."""
    groups = np.asarray(groups)
    uniq = sorted(set(groups.tolist()))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    assign = {g: i % n_folds for i, g in enumerate(uniq)}
    fold_of = np.array([assign[g] for g in groups])
    for f in range(n_folds):
        te = np.where(fold_of == f)[0]
        tr = np.where(fold_of != f)[0]
        if len(te):
            yield tr, te


def stratified_grouped_folds(labels, groups, n_folds=5, seed=42):
    """Grouped folds that also balance class counts.

    Groups are placed one at a time into whichever fold currently holds the
    fewest trials of that group's majority class. Greedy, deterministic given
    the seed, and enough to stop a rare class landing entirely in one fold.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    uniq = sorted(set(groups.tolist()))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)

    classes = sorted(set(labels.tolist()))
    counts = {f: {c: 0 for c in classes} for f in range(n_folds)}
    assign = {}
    # biggest groups first, so the hardest placements happen while there is
    # still room to balance
    for g in sorted(uniq, key=lambda g: -(groups == g).sum()):
        m = groups == g
        gl = labels[m]
        vals, cnt = np.unique(gl, return_counts=True)
        major = vals[np.argmax(cnt)]
        f = min(range(n_folds), key=lambda k: (counts[k][major], sum(counts[k].values())))
        assign[g] = f
        for c, n in zip(vals, cnt):
            counts[f][c] += int(n)

    fold_of = np.array([assign[g] for g in groups])
    for f in range(n_folds):
        te = np.where(fold_of == f)[0]
        tr = np.where(fold_of != f)[0]
        if len(te):
            yield tr, te
