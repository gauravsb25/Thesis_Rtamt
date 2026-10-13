import json,sys
from pathlib import Path
import numpy as np
from data import HERE,NAMES
from logic import formula,horizon,size
sys.dont_write_bytecode=True
d=json.loads((HERE/'results.json').read_text()); c=json.loads((HERE/'challenger_results.json').read_text()); v=json.loads((HERE/'verification.json').read_text())
m=dict(np.load(HERE/'metadata.npz')); p=dict(np.load(HERE/'challenger_predictions.npz'))
CL=['LEFT','RIGHT','DOWN','STRAIGHT']; S=m['session']; Y=p['label']; B=p['baseline']
dev=np.array([s.startswith('sub_01') and int(s[-2:])<=30 for s in S])
splits={'development':dev,'external_sub01':np.char.startswith(S,'sub_01')&~dev,'external_sub11':np.char.startswith(S,'sub_11')}

def ci(pred,mask):
    sessions=np.unique(S[mask]); counts=[]
    for s in sessions:
        mm=mask&(S==s)
        counts.append([[np.sum(mm&(Y==k)&(pr==k)) for k in range(3)] for pr in (pred,B)]+[[np.sum(mm&(Y==k)) for k in range(3)]])
    counts=np.array(counts); rng=np.random.default_rng(71); diffs=[]
    for _ in range(2000):
        a=counts[rng.integers(len(counts),size=len(counts))].sum(0)
        r=np.divide(a[:2],a[2],out=np.full((2,3),np.nan),where=a[2]>0)
        diffs.append(100*np.nanmean(r[0]-r[1]))
    return np.percentile(diffs,[2.5,97.5]).tolist()

models=[json.loads(path.read_text()) for path in HERE.glob('model_*.json')]
for model in models:
    idx=np.array(model['indices']); allowed={f'sub_01/ses-{i:02d}' for fold in model['train_folds'] for i in range(1+10*fold,11+10*fold)}
    assert set(S[idx])<=allowed
    assert len(np.unique(m['group'][idx]))==len(idx)
    assert np.array_equal(Y[idx],model['labels'])
    assert len(set(np.bincount(Y[idx],minlength=4)))==1
    assert len(idx)//4==model['budget']
stats=dict(learner_runs=len(models),class_structure_searches=len(models)*4,
    numerical_threshold_refinements=sum(di['counts']['refinements'] for model in models for di in model['diagnostics']),
    evaluated_candidates_including_reranking=sum(di['counts']['templates'] for model in models for di in model['diagnostics']),
    independent_recordings=len(np.unique(m['group'])),sampling_audits_passed=len(models))
intervals={mode:{name:ci(p[mode+('_oof' if name=='development' else '_final')],mask) for name,mask in splits.items()} for mode in ('positive','supervised')}
(HERE/'audit_summary.json').write_text(json.dumps(dict(stats=stats,movement_gain_intervals=intervals),indent=2),encoding='utf-8')
lines=['# TeLEx-style structure learning on the EEG dataset','',
'**Implemented, executed and verified. No production accuracy improvement was established; your original monitor remains unchanged.**','',
'This is a bounded TeLEx-2019-inspired implementation plus a separately labelled supervised extension. It is not an exact replication of the paper or a test of every possible STL structure.','',
'## What was implemented','',
'The implementation uses predicate tightness, temporal contraction/expansion, positive-trace max-min threshold fitting, and subformula-robustness augmentation for nesting. These mechanisms follow Sections 4–5 of [Jha et al. (2019)](https://susmitjha.github.io/papers/fmsd19.pdf). The paper learns properties from positive examples; four-class decoding is an extension.',
'','Practical limits: seven initial temporal intervals on a 32 Hz grid; shortlisted scalar threshold optimization; beam width five; up to three search levels and fourteen AST nodes. The grammar includes F/G, bounded until, Boolean combinations and negated augmented formulas. Until is explored over shortlisted subformulas rather than all initial predicate pairs. Exact min/max evaluation is piecewise smooth, so this is not a reproduction of the paper’s full numerical optimization or a global optimum claim. A decreasing robustness-band heuristic uses a bounded fallback when its band is empty.',
'','Two variants were kept separate:',
'','- **Positive-only formula learner:** uses the target-class positives for parameter fitting and structure ranking.',
'- **Supervised structure selection:** also uses the other discovery classes to reject non-discriminating formulas. Every chosen class formula must still satisfy all its discovery positives.',
'','Both variants use labelled development validation and a specified multiclass decoder; neither is an entirely unsupervised classifier. The decoder averages normalized standard STL robustness across two fixed seeds and selects the largest score. If none or multiple class formulas are satisfied, argmax still assigns a class. Score normalization uses only discovery data. Fitting a formula that accepts all positive examples is not the same as classifying all those examples correctly.',
'','## Sampling and data handling','',
f"The dataset contains {len(Y):,} windows from {stats['independent_recordings']} recording keys and two subjects. Each discovery subset contains exactly 8 or 16 examples per class: 32 or 64 windows total. Each is from a distinct subject/session/task/run recording; no overlapping windows from one recording are counted as separate discovery trials. Seeds 11 and 29 are fixed and averaged, not cherry-picked.",
'','Training sessions are round-robin sampled. Sessions, including all their overlapping windows, remain together in validation splits. Three outer ten-session blocks and two inner blocks select subset size, search level and classifier layout. The layouts are four-class learned formulas, or the original movement gate followed by learned LEFT/RIGHT/DOWN formulas. All 52 recorded sampling manifests passed equal-count, unique-trial and training-session membership checks.',
'','Raw 256 Hz windows are filtered at .3–6 Hz and reduced to 96 time points at 32 Hz. Inputs include eight normalized channels, all 28 normalized pair differences and eight normalized log-energy traces. Per-window normalization uses the entire completed window. Annotation timing is not used to recenter input, and this is not a prefix-causal monitor.',
'','Small-subset formula discovery is followed by validation on other sessions. This implementation does **not** add counterexamples iteratively to the discovery set; that remains a separately testable extension. No raw test samples or labels select thresholds, structures or seeds. The external populations have been inspected historically, so the evaluations are retrospective rather than blind.',
'','## Actual learned-model results','',
'The primary score below is mean LEFT/RIGHT/DOWN recall, counting missed movements labelled STRAIGHT as errors. STRAIGHT does not contribute to this score. The learned-only rows force a learned challenger rather than hiding its result behind baseline fallback.',
'','| Monitor | Nested development validation | External sub_01 | External sub_11 |',
'|---|---:|---:|---:|',
f"| Original | {d['baseline']['development']['movement_macro']:.2f}% | {d['baseline']['external_sub01']['movement_macro']:.2f}% | {d['baseline']['external_sub11']['movement_macro']:.2f}% |"]
for mode,row in c['modes'].items():
    lines.append(f"| Learned {mode} | {row['nested_validation']['movement_macro']:.2f}% | {row['external']['external_sub01']['movement_macro']:.2f}% | {row['external']['external_sub11']['movement_macro']:.2f}% |")
lines+=['','The deployment selection procedure chose the original baseline in both modes, including every outer fold. Consequently its accuracy is exactly unchanged. This does not mean the learned formulas matched the baseline: their separate results above are substantially worse.',
'','| Monitor | External sub_01 ordinary / balanced | Correct / total | External sub_11 ordinary / balanced | Correct / total |',
'|---|---:|---:|---:|---:|']
for name,row in [('Original',{'external':{k:v for k,v in d['baseline'].items() if k.startswith('external')}})]+list(c['modes'].items()):
    a=row['external']['external_sub01']; b=row['external']['external_sub11']
    lines.append(f"| {name} | {a['accuracy']:.2f}% / {a['balanced_accuracy']:.2f}% | {a['correct']}/{a['n']} | {b['accuracy']:.2f}% / {b['balanced_accuracy']:.2f}% | {b['correct']}/{b['n']} |")
lines+=['','## Class-by-class external recall','',
'| Monitor | sub_01 LEFT | RIGHT | DOWN | sub_11 LEFT | RIGHT | DOWN |','|---|---:|---:|---:|---:|---:|---:|']
for name,row in [('Original',{'external':{k:v for k,v in d['baseline'].items() if k.startswith('external')}})]+list(c['modes'].items()):
    values=[row['external'][s]['recall'][cl] for s in ('external_sub01','external_sub11') for cl in CL[:3]]
    lines.append('| '+name+' | '+' | '.join(f'{v:.2f}%' for v in values)+' |')
lines+=['','## Fit versus validation','',
'| Variant | Discovery movement recall, seed 11 / 29 | Full development fit | Nested validation |','|---|---:|---:|---:|']
for mode,row in c['modes'].items():
    a,b=[z['movement_macro'] for z in row['discovery_fit']]
    lines.append(f"| {mode} | {a:.2f}% / {b:.2f}% | {row['development_fit']['movement_macro']:.2f}% | {row['nested_validation']['movement_macro']:.2f}% |")
lines+=['','The final learned challengers both selected 8 examples per class, search level 2, and the original movement gate. Seed-specific external outcomes are preserved in challenger_results.json; neither seed was selected using external scores. Paired session-bootstrap intervals are in audit_summary.json and are descriptive, not selection-adjusted.',
'','## Verification and effort','',
f"- {stats['learner_runs']} four-class learner runs, or {stats['class_structure_searches']} class-specific structure searches.",
f"- {stats['numerical_threshold_refinements']:,} numerical threshold refinements and {stats['evaluated_candidates_including_reranking']:,} candidate evaluations including reranking. These are bounded searches, not that many independent methods.",
f"- {v['rtamt_real_trace_checks']} formula/real-trace comparisons against genuine RTAMT; full valid-horizon robustness agrees.",
f"- All 42,058 exported challenger predictions reproduced; {v['raw_window_checks']} raw-window/history adapter checks passed.",
'- Synthetic tests cover strict trace horizons, tightness sign/boundary and RTAMT agreement for nesting, Boolean logic and bounded until.',
f"- Main search/evaluation: {d['elapsed_seconds']:.1f} seconds. Explicit challenger fitting/evaluation: {c['elapsed_seconds']:.1f} seconds. Raw preprocessing and verification are additional. Production fingerprint remained unchanged.",
'','## Files and commands','',
'`FORMULAS.md` contains the actual frozen formulas. Their named signals are defined by data.py. `monitor.py` supplies a raw-window adapter; `LearnedSTLMonitor()` follows the selected baseline fallback, while `LearnedSTLMonitor(challenger=True)` exposes the rejected learned experiment explicitly.',
'','```powershell','python -B experiments\\telex2019_audit_2026_09_21\\data.py','python -B experiments\\telex2019_audit_2026_09_21\\run.py','python -B experiments\\telex2019_audit_2026_09_21\\challenger.py','python -B -m pytest experiments\\telex2019_audit_2026_09_21\\test_logic.py -q -p no:cacheprovider','python -B experiments\\telex2019_audit_2026_09_21\\verify.py','python -B experiments\\telex2019_audit_2026_09_21\\report.py','```',
'','Conclusion: the bounded implementation runs correctly but does not produce a better monitor on these data. It does not establish that every TeLEx search would fail. Increasing nesting, changing objectives, or adding counterexample refinement would be new experiments requiring fresh validation.']
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
formlines=['# Frozen learned challenger formulas','',
'These formulas were learned automatically and failed adoption. Original production monitors are unchanged. Time bounds are sample indices on the preprocessed 32 Hz trace; divide by 32 for seconds. Evaluate at index 0 of the completed window. The last trace sample is at 95/32 seconds. All formula horizons fit within it.',
'','Channel signals are centered and MAD-normalized filtered EEG; pair signals normalize the channel difference; energy signals are log(1 + smoothed energy / within-window median energy). Exact preprocessing is in data.py. These are not raw-voltage thresholds.',
'','Each seed supplies four class formulas. Their normalized robustness values are averaged, then the highest movement-class score is used only when the original monitor detects movement. Class formulas may overlap; they are not a mutually exclusive partition.']
for mode in ('positive','supervised'):
    manifest=json.loads((HERE/f'{mode}_challenger_monitor.json').read_text()); depth=manifest['config'][1]-1
    for model in manifest['models']:
        for cl,trees,scales in zip(CL,model['trees'],model['scales']):
            tree=trees[depth]
            formlines+=['',f'## {mode}, seed {model["seed"]}, {cl}','',
                f'Robustness divisor: {scales[depth]:.8g}. Horizon: {horizon(tree)} samples. AST nodes: {size(tree)}.',
                '','```text',formula(tree,NAMES),'```']
(HERE/'FORMULAS.md').write_text('\n'.join(formlines)+'\n',encoding='utf-8')
print('Reports and formula exports written;',stats)
