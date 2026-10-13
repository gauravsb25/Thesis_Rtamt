# _archive — everything retired (18 Sep 2026)

Nothing here is used by the monitors, the tests or `project_code.py`. **The whole folder can be deleted**
if you do not want the history. It was moved, not deleted, so that the thesis can still cite it.

| folder | what it is | why it is here |
|---|---|---|
| `pipeline/` | the original dataset scripts: `extract_clean_windows.py`, `remove_spliced_windows.py` (these two built `extracted_npz_dataset_v2`), `build_preica_windows.py`, `extract_windows_from_csv.py`, `extract_trial_windows.py`, and the TeLEx-era preprocessing | their code is summarised in section 4 of `project_code.py`; run them from here if a dataset has to be rebuilt |
| `experiment_scripts/` | the scripts of the rounds kept in `experiments/` | the rounds keep their README, logs and figures; the code moved here |
| `experiments/` | ~30 older rounds (left_rescue, pair_bank, non_ml_3sec, streaming_research_v1, sub01_rtamt_v1/v2, three_second_window_v1, …) | superseded by the RTAMT monitors; kept for provenance |
| `artifacts/` | the TeLEx mining output (traces, pairs, signals, synthesis results), ~1.3 GB | the monitors no longer use TeLEx; the thesis chapter may still cite these runs |
| `TeLEX/`, `synthesis/`, `features/`, `clean_final_icaeeg_results/`, `streaming_research/` | the TeLEx tool and its results | same |
| `monitoring/evaluate.py` | the accuracy tool | now section 1-2 of `project_code.py` (`python project_code.py evaluate`) |
| `_removed_monitors/` | the retired Python rule monitors and RTAMT 0.3.5 | replaced by the four STL specifications |
| `tests/test_annotation_labels.py` | the test of the archived annotation parser | it imports `lib/` and `streaming_research/`, both archived |
| `lib/`, `maintenance/`, `older_archive/` | path helpers and layout scripts of the old tree | the tree they described no longer exists |
| `rejected_2026_09_30/` | nine rounds of 30 Sep 2026 (joint shapelets, learned STL trees, scattering, waveform templates, CSV/filter/offline context, robust nod); summary kept in `experiments/rtamt_accuracy_2026_09_30` | none passed the adoption rule; production monitors unchanged |

## Rebuilding a dataset from here

```
python _archive/pipeline/extract_clean_windows.py     # then remove_spliced_windows.py -> extracted_npz_dataset_v2
python _archive/pipeline/extract_trial_windows.py     # 442 windows, one per trial
python _archive/pipeline/build_preica_windows.py --source D:/errp/eeg_cleaned_data
python _archive/pipeline/extract_windows_from_csv.py  # windows cut from the raw CSVs
```

They import each other through `pipeline.` names, so run them from a copy inside the project root, or add
`_archive` to `PYTHONPATH`. What each dataset is, and what the monitors scored on it, is written at the top
of `project_code.py`.
