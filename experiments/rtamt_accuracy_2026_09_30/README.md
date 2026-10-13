> **Moved (30 September 2026).** The nine experiment folders this summary describes were rejected and moved to `_archive/rejected_2026_09_30/`: joint_shapelets, learned_stl_rules, live_csv_context, live_filter_context, offline_context, robust_nod, scattering_rules, scattering_full_decision, waveform_templates. Their scripts resolve paths relative to themselves, so they still run from there.

# New RTAMT accuracy experiments: online and offline

No tested replacement passed the adoption rule. The production monitor code, thresholds, datasets and shared caches remain unchanged. The 90% balanced-accuracy target was not reached.

## Genuinely different methods

The archived experiments were checked before choosing new methods. DTW, ordinary Haar wavelets, CSP, Riemannian covariance, onset timing and phase rules had already been tried. The new work was:

1. **Joint-channel shapelets.** Learn short patterns lasting 0.25, 0.5 or 1 second across all eight channels. Search for each pattern anywhere inside the three-second window. Each training fold learns its own patterns, preventing test information from entering pattern discovery. This adapts the [shapelet method of Ye and Keogh](https://www.cs.ucr.edu/~eamonn/shaplet.pdf); it is different from the existing complete-event waveform templates.
2. **Second-order wavelet scattering.** First measure band envelopes, then measure their slower modulation. This retains envelope structure that a single average bandpower loses. The implementation uses a finite two-layer wavelet/modulus cascade based on [Anden and Mallat's Deep Scattering Spectrum](https://arxiv.org/abs/1304.6763). Its three-second support, normalization and channel choices are adaptations to this dataset.
3. **Four-class scattering corrections.** A separate follow-up allows the learned rules to recover missed movements and reject false alarms. It adds the existing monitor's predicted class as four indicators, never the true label. Unlike the first two studies, this can change STRAIGHT decisions.

Small depth-3 and depth-5 trees fit explicit threshold rules. Every frozen tree path is exported into actual RTAMT AND/OR specifications. No trained classifier object is called in the exported decision. These are learned Boolean STL predicates over window features, not the TeLEx2019 algorithm and not newly discovered temporal operators.

## Validation results

All 21,029 windows were counted. Source sessions 1-30 were split into three entire ten-session outer blocks. Inner validation chose the family and depth using only the corresponding training sessions. Overlapping windows never crossed the training/validation boundary. Training weights give each true class equal weight and each recording within a class equal weight.

| Fixed candidate | Online outer-validation BA | Offline outer-validation BA |
|---|---:|---:|
| Current fixed monitor | 85.01% | 86.73% |
| Joint shapelets, depth 3 | 80.70% | 83.50% |
| Joint shapelets, depth 5 | 80.67% | 83.35% |
| Second-order scattering, depth 3 | 82.65% | 83.10% |
| Second-order scattering, depth 5 | 80.61% | 81.07% |

Both studies included controls using morphology alone and first-order features. Neither new representation beat its matching existing monitor. Nested selection retained the existing monitor in every outer fold and in the final source-only selection. The unchanged external scores therefore describe the selected baseline, not a successful new classifier.

The full four-class follow-up selected a first-order depth-3 rule. Second-order features failed to provide an additional gain. Its source fitting scores were 86.77% online and 88.03% offline; these are in-sample results and are not test accuracy.

| Full correction evaluation | Existing online | New online | Existing offline | New offline |
|---|---:|---:|---:|---:|
| Nested source validation | 85.01% | 84.99% | 86.73% | 86.36% |
| Later sub_01 sessions | 69.62% | 72.43% | 71.80% | 73.71% |
| sub_11 | 78.67% | 69.39% | 79.37% | 68.56% |

The later-session improvements came mainly from detecting additional DOWN windows. On sub_11, the online rule increased correctly detected DOWN windows from 252 to 267 but reduced correctly recognized STRAIGHT windows from 484 to 269. The offline rule increased correct DOWN from 245 to 264 but reduced correct STRAIGHT from 507 to 254. These false DOWN alarms are why the replacements were rejected.

Keeping the original online gate frozen caps later-session balanced accuracy at 88.26%, even if every detected movement's identity were corrected perfectly. This is a bound on that particular unchanged detector, not on the information in EEG or on other possible methods. The four-class test explicitly removed this restriction.

## Earlier checks in this round

Twelve causal filter/doublet configurations and twenty-four completed-recording morphology configurations were also tested. Their selection-procedure balanced accuracies were 84.93% online and 86.54% offline, below their matching baselines. An offline candidate improved sub_11 to 80.25% but worsened later sub_01 to 69.72%.

Genuine past CSV samples were separately checked against reconstructed NPZ history. The same existing formulas with correctly aligned CSV past scored 85.18% / 70.29% / 78.63%, compared with 85.01% / 69.62% / 78.67% under the published reconstructed-history protocol. This is a change of input context, not a reliable improvement from a new rule. The learned causal-context rule added no external gain.

All external datasets have been examined during earlier research, so these are retrospective results. Completed-window online decisions were checked for future independence and against streamed raw CSV samples. Input EEG was already ICA-cleaned; this work does not establish live ICA preprocessing or early row-by-row event accuracy.

## Reproduce and inspect

From `C:\Users\gaura\thesis_rtamt`, print the complete comparison and class counts:

```powershell
python -B experiments/rtamt_accuracy_2026_09_30/report.py
```

Each experiment has its protocol, numerical results, frozen specifications, saved predictions and verification evidence:

- `experiments/joint_shapelets_2026_09_30/`
- `experiments/scattering_rules_2026_09_30/`
- `experiments/scattering_full_decision_2026_09_30/`
- `experiments/live_filter_context_2026_09_30/`
- `experiments/live_csv_context_2026_09_30/`
- `experiments/offline_context_2026_09_30/`

The helper in `experiments/learned_stl_rules_2026_09_30/` exports learned trees and tests actual RTAMT threshold-boundary behavior. Its pinned training dependencies are isolated under that experiment, leaving system Python and existing NumPy/SciPy untouched.
