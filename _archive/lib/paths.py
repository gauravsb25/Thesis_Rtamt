"""
Project root and data-folder locations, resolved absolutely.

Scripts live in pipeline/, tools/ and telex/ but the data folders sit at the
repository root. Before this module every path was relative to the working
directory, so a script only worked when launched from the root -- `cd pipeline
&& python classify.py` silently looked for `pipeline/3SEC_EXTRACTS` and found
nothing.

Everything here is anchored to this file's own location, so a script runs
correctly from anywhere:

    python pipeline/classify.py
    cd pipeline && python classify.py
    python /absolute/path/to/pipeline/classify.py

`bootstrap()` additionally puts the repository root on sys.path so
`from features import ...` resolves. Every executable script calls it in
its first three lines; the library modules under lib/ do not need to, because
they are only ever imported by something that already has.
"""

import os
import sys

# lib/paths.py -> lib/ -> repository root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bootstrap():
    """Put the repository root on sys.path. Safe to call repeatedly."""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    return PROJECT_ROOT


def at(*parts):
    """Absolute path to something under the repository root."""
    return os.path.join(PROJECT_ROOT, *parts)


# ---------------------------------------------------------------------------
# Data folders. Raw input first, then each pipeline stage's output.
# ---------------------------------------------------------------------------
DATASET = at("dataset", "eeg_cleaned_data")          # raw recordings, never written
PROCESSED_TRACES = at("processed_traces")            # stage 1: marker-bounded trials
EXTRACTS = at("3SEC_EXTRACTS")                       # stage 2: 3-second windows
EXTRACTS_FLAT = at("3secFlat")                       # same windows, one folder, for TeLEX
EXTRACTS_ORACLE = at("fif_3sec_windows")             # annotation-centred windows (upper bound)
RESULTS = at("CLASSIFICATION_RESULT")                # every analysis + classification table
RTAMT_RESULTS = at("RTAMT_RESULTS")                  # RTAMT-layer outputs, kept separate

# TeLEX synthesis
TELEX_RESULTS = at("clean_final_icaeeg_results")     # the ICA synthesis runs
FORMULA_RESULTS = at("formula_finalresults")         # older synthesis output
FORMULAS_DIR = at("telex")                           # formula .txt files live beside the scripts


def results(*parts):
    """Path inside CLASSIFICATION_RESULT/, creating the folder on demand."""
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, *parts)


def rtamt_results(*parts):
    """Path inside RTAMT_RESULTS/, creating the folder on demand.

    The RTAMT layer writes robustness matrices, window statistics and mined
    monitor sets. Those are intermediate artefacts of the specification work,
    not classification tables, and mixing them into CLASSIFICATION_RESULT made
    that folder hard to read. They live here instead.
    """
    os.makedirs(RTAMT_RESULTS, exist_ok=True)
    return os.path.join(RTAMT_RESULTS, *parts)
