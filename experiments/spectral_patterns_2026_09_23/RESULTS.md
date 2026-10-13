# Spectral-pattern experiment: measured results

**No candidate met the adoption rule. Production monitors were not changed.**

All percentages below are measured. Development means predictions for sessions excluded from fitting, not training accuracy.
Movement macro is the average recall for LEFT, RIGHT and DOWN, including movements incorrectly called STRAIGHT.

| Development method | Correct / total | Ordinary accuracy | Balanced accuracy | Movement macro |
|---|---:|---:|---:|---:|
| current_monitor | 17719/18560 | 95.47% | 85.01% | 80.70% |
| joint_bandpower | 17083/18560 | 92.04% | 69.61% | 60.28% |
| ordered_bandpower | 17536/18560 | 94.48% | 76.15% | 68.52% |
| filterbank_csp | 14825/18560 | 79.88% | 62.04% | 54.62% |
| monitor_lda_control | 17348/18560 | 93.47% | 84.64% | 81.03% |
| monitor_plus_spectral | 17864/18560 | 96.25% | 83.33% | 77.96% |
| hybrid_csp | 17051/18560 | 91.87% | 67.80% | 57.76% |
| hybrid_ordered | 17290/18560 | 93.16% | 73.64% | 65.54% |

The winner under the predeclared movement-first ranking was `monitor_lda_control`, the control using only existing monitor measurements. Its 0.33-point movement gain was below the 0.5-point requirement and its balanced accuracy fell. It was not eligible for adoption.
Only that frozen winner was scored externally, as a diagnostic, despite failing the adoption gate. No spectral candidate was selected using external results. These external sets have been exposed in previous project experiments and are not blind tests.

| Split / model | Correct / total | Ordinary accuracy | Balanced accuracy | Movement macro |
|---|---:|---:|---:|---:|
| Training fit / selected control | 17438/18560 | 93.95% | 85.78% | 82.44% |
| sub01_later / current_monitor | 1081/1269 | 85.19% | 69.62% | 60.39% |
| sub01_later / monitor_lda_control | 1037/1269 | 81.72% | 66.42% | 57.51% |
| sub11 / current_monitor | 1006/1200 | 83.83% | 78.67% | 73.08% |
| sub11 / monitor_lda_control | 904/1200 | 75.33% | 74.68% | 75.77% |

## Class counts

| Split / model | LEFT | RIGHT | DOWN | STRAIGHT |
|---|---:|---:|---:|---:|
| Development / current_monitor | 805/959 (83.94%) | 770/963 (79.96%) | 585/748 (78.21%) | 15559/15890 (97.92%) |
| Development / joint_bandpower | 550/959 (57.35%) | 467/963 (48.49%) | 561/748 (75.00%) | 15505/15890 (97.58%) |
| Development / ordered_bandpower | 588/959 (61.31%) | 609/963 (63.24%) | 606/748 (81.02%) | 15733/15890 (99.01%) |
| Development / filterbank_csp | 402/959 (41.92%) | 515/963 (53.48%) | 512/748 (68.45%) | 13396/15890 (84.30%) |
| Development / monitor_lda_control | 835/959 (87.07%) | 797/963 (82.76%) | 548/748 (73.26%) | 15168/15890 (95.46%) |
| Development / monitor_plus_spectral | 711/959 (74.14%) | 713/963 (74.04%) | 641/748 (85.70%) | 15799/15890 (99.43%) |
| Development / hybrid_csp | 460/959 (47.97%) | 424/963 (44.03%) | 608/748 (81.28%) | 15559/15890 (97.92%) |
| Development / hybrid_ordered | 583/959 (60.79%) | 591/963 (61.37%) | 557/748 (74.47%) | 15559/15890 (97.92%) |
| sub01_later / current_monitor | 69/117 (58.97%) | 118/183 (64.48%) | 71/123 (57.72%) | 823/846 (97.28%) |
| sub01_later / monitor_lda_control | 50/117 (42.74%) | 120/183 (65.57%) | 79/123 (64.23%) | 788/846 (93.14%) |
| sub11 / current_monitor | 105/187 (56.15%) | 165/221 (74.66%) | 252/285 (88.42%) | 484/507 (95.46%) |
| sub11 / monitor_lda_control | 108/187 (57.75%) | 170/221 (76.92%) | 264/285 (92.63%) | 362/507 (71.40%) |

## Interpretation

- Temporal ordering helps relative to static bandpower in this fixed comparison, but neither beats the existing monitor.
- Monitor plus spectral features has the best ordinary development accuracy (96.25%), but worse movement recognition. The STRAIGHT majority makes ordinary accuracy a poor selection target.
- The filter-bank CSP implementation learns spatial filters from training sessions only; it is not another sweep over hand-picked electrode differences.
- These are seven specific model/feature configurations, not proof that all frequency-domain methods fail. Only two subjects are available.
- Development comparisons reuse historically inspected sessions. Method selection adds optimism; fresh recordings are needed for a genuinely prospective evaluation.
- These research classifiers are not newly validated RTAMT monitors. No classifier is disguised as a temporal-logic rule.
