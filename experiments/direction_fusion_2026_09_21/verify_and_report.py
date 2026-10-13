import sys, json
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
from fusion_monitor import DirectionFusionMonitor
M=dict(np.load(ROOT/'measurements_v2.npz',allow_pickle=True))
B=dict(np.load(ROOT/'experiments/operator_sweep_2026_09_21/bank.npz',allow_pickle=True))
for k in B:
    if k.endswith('|resp'): B[k]=np.where(B[k.replace('|resp','|max')]<2,0,B[k])
data={**B,**M}
p=np.load(HERE/'predictions.npz'); d=json.loads((HERE/'results.json').read_text())
verified=0
for family in d['families']:
    monitor=DirectionFusionMonitor(f'{family}_monitor.json')
    x=np.column_stack([data[k] for k in monitor.manifest.get('feature_names',[])]) if monitor.family!='baseline' else None
    got=monitor.predict_many(x,p['baseline'])
    assert np.array_equal(got,p[family+'_final']),family
    assert np.array_equal(got==3,p['baseline']==3)
    verified+=len(got)
lines=['# Combining the original monitor with direction specialists','',
'**Result: five combinations implemented and tested; none qualifies as a replacement.** Production monitors were untouched.','',
'The primary score is mean LEFT/RIGHT/DOWN recall. STRAIGHT contributes nothing to this score. Every actual movement window is counted, including those wrongly called STRAIGHT. The movement gate is preserved exactly. This makes comparisons about direction classification rather than exploiting class imbalance.','',
'| Architecture | Development nested validation | External sub_01 | External sub_11 |',
'|---|---:|---:|---:|',
f"| Original monitor | {d['baseline']['development']['movement_macro']:.2f}% | {d['baseline']['external_sub01']['movement_macro']:.2f}% | {d['baseline']['external_sub11']['movement_macro']:.2f}% |"]
for name,v in d['families'].items():
    lines.append(f"| {name} | {v['nested_validation']['movement_macro']:.2f}% | {v['external']['external_sub01']['metrics']['movement_macro']:.2f}% | {v['external']['external_sub11']['metrics']['movement_macro']:.2f}% |")
lines+=['','## Architectures','',
'1. **Nod specialist:** keep movement detection and the original LEFT/RIGHT polarity; let a learned DOWN-versus-turn expert change the nod decision when its margin clears a fixed threshold.',
'2. **Turn specialist:** preserve original STRAIGHT and DOWN decisions; use a learned LEFT/RIGHT expert for turns, falling back to original polarity below its margin threshold.',
'3. **Nested specialists:** movement gate → confident DOWN-versus-turn decision → confident LEFT-versus-RIGHT decision. Fall back to the original stage when confidence is insufficient.',
'4. **Agreement:** change the original direction only when the nested binary specialists and an independent three-way classifier agree and satisfy their margin thresholds. Their training data and feature pool overlap, so agreement does not imply statistically independent evidence.',
'5. **Prior fusion:** combine learned three-way scores with a fixed preference for the original monitor label.',
'','Model margins are not calibrated probabilities. These are learned hybrid monitors, not native RTAMT-only formulas. Raw score fusion was limited to a fixed grid chosen on development data.',
'','## Per-class external recall','',
'| Architecture | sub_01 LEFT | RIGHT | DOWN | sub_11 LEFT | RIGHT | DOWN |',
'|---|---:|---:|---:|---:|---:|---:|']
rows=[('Original',{'external':{k:{'metrics':v} for k,v in d['baseline'].items() if k.startswith('external')}})]+list(d['families'].items())
for name,v in rows:
    values=[v['external'][s]['metrics']['recall'][c] for s in ('external_sub01','external_sub11') for c in ('LEFT','RIGHT','DOWN')]
    lines.append('| '+name+' | '+' | '.join(f'{x:.2f}%' for x in values)+' |')
lines+=['','## Why not merge the apparent winners into production?','',
'The nod specialist improves sub_01 DOWN recall from 57.72% to 66.67%, but reduces LEFT recall from 58.97% to 53.85%. On sub_11 it reduces both LEFT and DOWN. The agreement method raises sub_11 LEFT recall from 56.15% to 60.96%, but cuts RIGHT from 74.66% to 27.15%. These are material tradeoffs, not across-the-board improvements.',
'','We cannot choose which expert to trust using the true label of each new window. Choosing a different specialist after seeing each external class score would tune on the test data. Confidence and agreement were tested as deployable choices; they did not resolve the transfer failure.',
'','## Selection and reproducibility','',
f"There were 30 fixed fusion configurations (five architectures × two feature counts × three confidence/prior settings), plus baseline fallback. {d['fits']} unique specialist fits were reused across combinations. Training/evaluation took {d['elapsed_seconds']:.1f} seconds using the existing feature bank; no neural networks or raw feature extraction were run in that time.",
'','Selection used nested whole-session folds, maximizing movement macro recall subject to no individual movement class losing more than one point in inner validation. The selected_pipeline row selects architecture inside each outer fold too. All external configurations were frozen before their scores were calculated. Repeated prior project exploration means these are retrospective external evaluations, not new blind tests.',
'','Training scores, per-class counts/confusion matrices and paired session-bootstrap confidence intervals are in results.json. Some experiments intentionally failed: their serialized models are for reproducibility, not deployment recommendations.',
'',f"The saved fusion_monitor.py reproduced all {verified:,} saved final predictions (six exports × 21,029 windows), with exact preservation of STRAIGHT decisions. Shared input/source hash changes during the experiment: {d['concurrent_changes']}.",
'','Reproduce from the project root:',
'','```powershell','python -B experiments\\direction_fusion_2026_09_21\\run.py','python -B experiments\\direction_fusion_2026_09_21\\verify_and_report.py','```',
'','`fusion_monitor.py` accepts the current monitor verdict and a complete feature dictionary in the frozen manifest. It is an experimental completed-window feature-level wrapper, not an independently validated raw-EEG/streaming pipeline. Keep the original production monitor.']
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Verified',verified,'predictions and all STRAIGHT decisions; report written.')
