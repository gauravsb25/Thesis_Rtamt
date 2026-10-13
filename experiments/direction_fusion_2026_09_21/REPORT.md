# Combining the original monitor with direction specialists

**Result: five combinations implemented and tested; none qualifies as a replacement.** Production monitors were untouched.

The primary score is mean LEFT/RIGHT/DOWN recall. STRAIGHT contributes nothing to this score. Every actual movement window is counted, including those wrongly called STRAIGHT. The movement gate is preserved exactly. This makes comparisons about direction classification rather than exploiting class imbalance.

| Architecture | Development nested validation | External sub_01 | External sub_11 |
|---|---:|---:|---:|
| Original monitor | 80.70% | 60.39% | 73.08% |
| nod_specialist | 82.57% | 61.30% | 68.56% |
| turn_specialist | 81.70% | 54.74% | 56.03% |
| nested_specialists | 85.15% | 60.98% | 49.23% |
| agreement | 88.33% | 59.87% | 58.14% |
| prior_fusion | 85.84% | 61.28% | 59.27% |
| selected_pipeline | 88.33% | 59.87% | 58.14% |

## Architectures

1. **Nod specialist:** keep movement detection and the original LEFT/RIGHT polarity; let a learned DOWN-versus-turn expert change the nod decision when its margin clears a fixed threshold.
2. **Turn specialist:** preserve original STRAIGHT and DOWN decisions; use a learned LEFT/RIGHT expert for turns, falling back to original polarity below its margin threshold.
3. **Nested specialists:** movement gate → confident DOWN-versus-turn decision → confident LEFT-versus-RIGHT decision. Fall back to the original stage when confidence is insufficient.
4. **Agreement:** change the original direction only when the nested binary specialists and an independent three-way classifier agree and satisfy their margin thresholds. Their training data and feature pool overlap, so agreement does not imply statistically independent evidence.
5. **Prior fusion:** combine learned three-way scores with a fixed preference for the original monitor label.

Model margins are not calibrated probabilities. These are learned hybrid monitors, not native RTAMT-only formulas. Raw score fusion was limited to a fixed grid chosen on development data.

## Per-class external recall

| Architecture | sub_01 LEFT | RIGHT | DOWN | sub_11 LEFT | RIGHT | DOWN |
|---|---:|---:|---:|---:|---:|---:|
| Original | 58.97% | 64.48% | 57.72% | 56.15% | 74.66% | 88.42% |
| nod_specialist | 53.85% | 63.39% | 66.67% | 48.66% | 74.21% | 82.81% |
| turn_specialist | 45.30% | 61.20% | 57.72% | 56.15% | 23.53% | 88.42% |
| nested_specialists | 55.56% | 68.85% | 58.54% | 48.66% | 22.17% | 76.84% |
| agreement | 47.86% | 71.58% | 60.16% | 60.96% | 27.15% | 86.32% |
| prior_fusion | 57.26% | 70.49% | 56.10% | 64.71% | 23.98% | 89.12% |
| selected_pipeline | 47.86% | 71.58% | 60.16% | 60.96% | 27.15% | 86.32% |

## Why not merge the apparent winners into production?

The nod specialist improves sub_01 DOWN recall from 57.72% to 66.67%, but reduces LEFT recall from 58.97% to 53.85%. On sub_11 it reduces both LEFT and DOWN. The agreement method raises sub_11 LEFT recall from 56.15% to 60.96%, but cuts RIGHT from 74.66% to 27.15%. These are material tradeoffs, not across-the-board improvements.

We cannot choose which expert to trust using the true label of each new window. Choosing a different specialist after seeing each external class score would tune on the test data. Confidence and agreement were tested as deployable choices; they did not resolve the transfer failure.

## Selection and reproducibility

There were 30 fixed fusion configurations (five architectures × two feature counts × three confidence/prior settings), plus baseline fallback. 42 unique specialist fits were reused across combinations. Training/evaluation took 8.4 seconds using the existing feature bank; no neural networks or raw feature extraction were run in that time.

Selection used nested whole-session folds, maximizing movement macro recall subject to no individual movement class losing more than one point in inner validation. The selected_pipeline row selects architecture inside each outer fold too. All external configurations were frozen before their scores were calculated. Repeated prior project exploration means these are retrospective external evaluations, not new blind tests.

Training scores, per-class counts/confusion matrices and paired session-bootstrap confidence intervals are in results.json. Some experiments intentionally failed: their serialized models are for reproducibility, not deployment recommendations.

The saved fusion_monitor.py reproduced all 126,174 saved final predictions (six exports × 21,029 windows), with exact preservation of STRAIGHT decisions. Shared input/source hash changes during the experiment: [].

Reproduce from the project root:

```powershell
python -B experiments\direction_fusion_2026_09_21\run.py
python -B experiments\direction_fusion_2026_09_21\verify_and_report.py
```

`fusion_monitor.py` accepts the current monitor verdict and a complete feature dictionary in the frozen manifest. It is an experimental completed-window feature-level wrapper, not an independently validated raw-EEG/streaming pipeline. Keep the original production monitor.
