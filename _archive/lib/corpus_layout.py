"""
Shared helpers for the subject/session folder layout.

Pipeline outputs mirror the raw dataset's hierarchy:

    dataset/eeg_cleaned_data/sub_01/ses-01/ica_eeg/sub-01_ses-01_task-left_run-01_eeg.csv
    processed_traces/        sub_01/ses-01/        sub-01_ses-01_task-left_run-01_eeg.csv
    3SEC_EXTRACTS/           sub_01/ses-01/        sub-01_ses-01_task-left_run-01_eeg_3sec.csv

Note the two naming conventions, which come from the raw dataset and are kept
as-is: directories use underscores (`sub_01`), filenames use hyphens (`sub-01`).

Every reader in the pipeline goes through `list_corpus_csvs`, which walks
recursively and therefore works whether a folder is nested or still flat.
"""

import os
import re

# sub-01_ses-01_task-left_run-01_eeg.csv -> ('01', '01')
_SUB_SES = re.compile(r"sub-(\d+)_ses-(\d+)", re.IGNORECASE)


def parse_sub_ses(filename):
    """('01', '01') from a corpus filename, or (None, None) if unparseable."""
    m = _SUB_SES.search(os.path.basename(filename))
    return (m.group(1), m.group(2)) if m else (None, None)


def session_subpath(filename):
    """Relative 'sub_01/ses-01' for a corpus filename, '' if unparseable.

    Files that don't carry a subject/session land at the corpus root rather
    than being dropped, so nothing goes missing silently.
    """
    sub, ses = parse_sub_ses(filename)
    if sub is None:
        return ""
    return os.path.join(f"sub_{sub}", f"ses-{ses}")


def output_path(root, filename, makedirs=True):
    """Full path where `filename` belongs under `root`, creating the session dir."""
    directory = os.path.join(root, session_subpath(filename))
    if makedirs:
        os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, os.path.basename(filename))


def list_sessions(root):
    """Sorted ['sub_01/ses-01', ...] for every session folder holding CSVs."""
    sessions = set()
    for rel in list_corpus_csvs(root):
        parent = os.path.dirname(rel)
        if parent:
            sessions.add(parent)
    return sorted(sessions)


def corpus_path(root, name):
    """Full path to `name` under `root`, wherever it lives.

    Prefers the nested sub_XX/ses-YY location, falls back to a flat `root`.
    Lets callers that key everything by basename keep doing so.
    """
    nested = output_path(root, name, makedirs=False)
    if os.path.exists(nested):
        return nested
    return os.path.join(root, os.path.basename(name))


def list_corpus_csvs(root, skip_dirs=("Logs",)):
    """Sorted CSV paths relative to `root`, recursing into session folders.

    Returns paths relative to `root` so callers can keep using
    `os.path.join(root, f)`. Use `os.path.basename(f)` when parsing the label.
    Works on both the nested layout and a legacy flat folder.
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in filenames:
            if name.lower().endswith(".csv"):
                full = os.path.join(dirpath, name)
                found.append(os.path.relpath(full, root))
    return sorted(found)
