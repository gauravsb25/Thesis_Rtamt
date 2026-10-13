"""Uncertainty, error overlap, and development-only feature correlation audit."""
import json
import numpy as np
from scipy.stats import spearmanr
from run import HERE, ROOT, CL, past, sequence, verify

def main():
    rng=np.random.default_rng(1909)
    x=rng.normal(size=(3,96))
    for a,b in ((0,1),(0,3),(2,16),(4,32),(8,32)):
        for minimum in (False,True):
            actual=past(x,a,b,minimum)
            expected=np.full_like(x,np.inf if minimum else -np.inf)
            for t in range(a,96):
                seg=x[:,max(0,t-b):t-a+1]
                expected[:,t]=(np.min if minimum else np.max)(seg,axis=1)
            np.testing.assert_allclose(actual,expected)
    for d in (2,4,8):
        for g in (16,32):
            for triple in (False,True):
                np.testing.assert_allclose(sequence(10*x,d,g,triple),10*sequence(x,d,g,triple))
    verify()
    z=dict(np.load(HERE/'oof_predictions.npz')); y=z['label']; ses=z['session']; p=z['all_families']; base=z['baseline']
    sessions=np.unique(ses)
    counts=[]; correct=[]; old=[]
    for s in sessions:
        counts.append([np.sum((ses==s)&(y==c)) for c in CL])
        correct.append([np.sum((ses==s)&(y==c)&(p==c)) for c in CL])
        old.append([np.sum((ses==s)&(y==c)&(base==c)) for c in CL])
    counts=np.array(counts); correct=np.array(correct); old=np.array(old)
    boot=[]
    for _ in range(4000):
        ids=rng.integers(0,len(sessions),len(sessions)); n=counts[ids].sum(axis=0)
        boot.append(100*np.mean((correct[ids].sum(axis=0)-old[ids].sum(axis=0))/n))
    out={'operator_checks':'10 brute-force past extrema comparisons, 12 homogeneity comparisons, 6 RTAMT comparisons passed',
         'session_cluster_bootstrap_gain_95_percent':np.quantile(boot,[.025,.975]).tolist(),
         'bootstrap_note':'Resamples 30 development sessions of fixed OOF predictions; does not rerun selection and does not quantify new-subject generalization.',
         'fixed_errors':int(np.sum((base!=y)&(p==y))), 'introduced_errors':int(np.sum((base==y)&(p!=y))),
         'per_class':{str(c):{'baseline':100*float(np.mean(base[y==c]==c)),'nested':100*float(np.mean(p[y==c]==c))} for c in CL}}
    f=dict(np.load(HERE/'development_features.npz'))
    m=dict(np.load(ROOT/'measurements_v2.npz',allow_pickle=True)); dev=np.isin(m['session'],sessions)
    prior=dict(np.load(ROOT/'experiments'/'phase_shape_2026_09_18'/'features.npz',allow_pickle=True))
    out['spearman_development']={key:{k:float(spearmanr(f[key],v[dev]).statistic) for k,v in {**{k:m[k] for k in ('turn','nod','background')},**{k:prior[k] for k in ('lag_env','rise','fall','width')}}.items()} for key in ('double_2_32','triple_4_32','burst_4','nodsync_4')}
    (HERE/'diagnostics.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
