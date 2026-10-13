"""Create auditable tables and paired session-bootstrap uncertainty."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
CL=['LEFT','RIGHT','DOWN','STRAIGHT']

def session_interval(y,baseline,pred,session,mask,repeats=2000):
    # Resample whole sessions, not strongly overlapping windows. Conditional
    # on saved predictions: does not refit models or quantify selection bias.
    totals=[]; correct=[]; basecorrect=[]
    for s in np.unique(session[mask]):
        idx=mask&(session==s)
        totals.append([np.sum(idx&(y==c)) for c in range(4)])
        correct.append([np.sum(idx&(y==c)&(pred==c)) for c in range(4)])
        basecorrect.append([np.sum(idx&(y==c)&(baseline==c)) for c in range(4)])
    totals=np.array(totals); correct=np.array(correct); basecorrect=np.array(basecorrect)
    rng=np.random.default_rng(2309); gains=[]
    for _ in range(repeats):
        draw=rng.integers(len(totals),size=len(totals)); n=totals[draw].sum(0)
        if np.any(n==0): continue
        gains.append(100*np.mean((correct[draw].sum(0)-basecorrect[draw].sum(0))/n))
    return np.quantile(gains,[.025,.975]).tolist()

def main():
    r=json.loads((HERE/'results.json').read_text()); p=dict(np.load(HERE/'predictions.npz'))
    y=p['y']; dev=p['dev']; base=p['baseline']; nested=p['nested']; ses=p['session']
    diagnostics={'development_gain_ci':session_interval(y,base,nested,ses,dev),'errors':{}}
    for k,c in enumerate(CL):
        mask=dev&(y==k)
        diagnostics['errors'][c]={'fixed':int(np.sum(mask&(base!=y)&(nested==y))),
            'broken':int(np.sum(mask&(base==y)&(nested!=y)))}
    (HERE/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    lines=['# Targeted error correction: measured results','',
        f"**90% balanced accuracy achieved across required evaluations: {r['achieved_90']}. Adoption criteria passed: {r['adopt']}.**",'',
        'No production monitor was changed by this experiment. These are ML research candidates, not pure STL formulas.',
        'Nested validation chooses models using inner training folds, then measures on separate outer sessions. The model and feature choice are therefore part of the evaluated procedure.','',
        '| Evaluation | Correct / total | Ordinary accuracy | Balanced accuracy | Movement macro recall |',
        '|---|---:|---:|---:|---:|']
    entries=[('Development baseline',r['baseline_development']),('Nested validation',r['nested_validation']),('Final training fit (not validation)',r['training_fit'])]
    for split,v in r['external'].items():
        entries.extend([(split+' baseline',v['baseline']),(split+' candidate',v['candidate'])])
    for name,v in entries:
        lines.append(f"| {name} | {v['correct']}/{v['n']} | {v['accuracy']:.2f}% | {v['ba']:.2f}% | {v['movement']:.2f}% |")
    low,high=diagnostics['development_gain_ci']
    lines+=['',f'Paired session-bootstrap 95% interval for nested balanced-accuracy gain: **{low:+.2f} to {high:+.2f} percentage points**. This resamples fixed predictions and does not include uncertainty from refitting or historical method selection.','',
        '## Exact class counts','', '| Evaluation | LEFT | RIGHT | DOWN | STRAIGHT |','|---|---:|---:|---:|---:|']
    for name,v in entries:
        cells=[f'{row[k]}/{sum(row)} ({v["recall"][k]:.2f}%)' for k,row in enumerate(v['confusion'])]
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['','## Nested choices','', '| Outer held-out sessions | Inner-selected method | Baseline BA | Candidate BA |','|---|---|---:|---:|']
    for f in r['folds']:
        start=f['fold']*10+1
        lines.append(f"| {start}-{start+9} | {f['selected']} | {f['baseline']['ba']:.2f}% | {f['candidate']['ba']:.2f}% |")
    lines+=['',f"Final development-only selection: `{r['selected']}`. Frozen before external scoring.",'',
        '## Exploratory ablations (not substitutes for the nested result)','',
        '| Fixed model | Outer balanced accuracy | Movement recall |','|---|---:|---:|']
    for name,v in r['exploratory_ablation'].items():
        lines.append(f"| {name} | {v['ba']:.2f}% | {v['movement']:.2f}% |")
    if (HERE/'spectral_ablation.json').exists():
        a=json.loads((HERE/'spectral_ablation.json').read_text())['no_spectral_direct']
        lines += ['',f"Post-hoc direct-classifier ablation without the 240 spectral features: **{a['ba']:.2f}% balanced accuracy**, versus **{r['exploratory_ablation']['direct:combined']['ba']:.2f}%** with them. This diagnostic was not used to change model selection or the external candidate."]
    lines+=['','## Corrections and regressions in nested validation','', '| True class | Existing errors fixed | Previously correct decisions broken |','|---|---:|---:|']
    for name,v in diagnostics['errors'].items(): lines.append(f"| {name} | {v['fixed']} | {v['broken']} |")
    lines+=['','## Scope','',
        f"The actual benchmark took {r['elapsed_seconds']/60:.1f} minutes, excluding dependency setup. It used 48 inner configuration fits, 18 outer configuration fits and one final configuration fit if selected. Staged configurations fit three separate classifiers each; every classifier uses 160 boosting iterations.",
        'Previously exposed external data cannot establish blind generalization. Source-session feature design and baseline thresholds also predate this experiment. Nested splitting prevents new window leakage but does not erase prior exposure.',
        'Dependencies are installed only in this experiment/vendor folder. The fixed settings and selection rule are recorded in PROTOCOL.md.',
        'A failed result does not prove 90% is impossible. A favorable training or single-fold score does not meet the target.','']
    (HERE/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('='*88); print('TARGETED ERROR CORRECTION | WHOLE-SESSION VALIDATION'); print('='*88)
    for name,v in entries:
        print(f"{name:42} BA {v['ba']:6.2f}%   L/R/D {v['movement']:6.2f}%   {v['correct']}/{v['n']}")
    print(f"90% target achieved: {r['achieved_90']} | Adoption passed: {r['adopt']}")
    print('Detailed results:',HERE/'RESULTS.md')

if __name__=='__main__': main()
