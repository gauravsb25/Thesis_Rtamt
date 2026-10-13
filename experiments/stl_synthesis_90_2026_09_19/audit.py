"""Invariance, run weighting, held-out reporting, and paired uncertainty."""
import json
import numpy as np
from features import HERE,ROOT,load,temporal
from synthesis import CL,initial,classify,ba,predict,fit_stage,evaluate
from run_search import subset

def bootstrap(z,seed=90):
    y=z['label'];p=z['prediction'];b=z['baseline'];ses=z['session']; groups=np.unique(ses)
    n=np.array([[np.sum((ses==s)&(y==c)) for c in CL] for s in groups])
    a=np.array([[np.sum((ses==s)&(y==c)&(p==c)) for c in CL] for s in groups])
    old=np.array([[np.sum((ses==s)&(y==c)&(b==c)) for c in CL] for s in groups])
    rng=np.random.default_rng(seed);gain=[];acc=[];missing=0
    for _ in range(4000):
        ix=rng.integers(len(groups),size=len(groups));count=n[ix].sum(axis=0);valid=count>0
        if not valid.all(): missing+=1
        gain.append(float(np.mean((a[ix].sum(axis=0)-old[ix].sum(axis=0))[valid]/count[valid])*100))
        acc.append(float(np.mean(a[ix].sum(axis=0)[valid]/count[valid])*100))
    return {'gain_95_interval':np.quantile(gain,[.025,.975]).tolist(),'accuracy_95_interval':np.quantile(acc,[.025,.975]).tolist(),
            'replicates_with_absent_classes':missing,'note':'Paired session resampling of fixed predictions; selection is not refitted; not a new-subject uncertainty estimate.'}

def main():
    # Test new feature construction on independent random signals and real recordings.
    rng=np.random.default_rng(90);raw=rng.normal(size=(5,8,768))
    a,n=temporal(raw);b,_=temporal(raw*10);np.testing.assert_allclose(a,b,rtol=1e-4,atol=2e-5)
    negative,_=temporal(-raw)
    signs=np.array([1 if name.startswith('circulation_') else -1 for name in n])
    np.testing.assert_allclose(negative,a*signs,rtol=1e-4,atol=2e-5)
    z=np.load(ROOT/'extracted_npz_dataset_v2'/'sub_01'/'ses-01'/'clean_epochs.npz',allow_pickle=True)
    a,n=temporal(z['epochs'][:5,:8]*1e6);b,_=temporal(z['epochs'][:5,:8]*1e7)
    np.testing.assert_allclose(a,b,rtol=1e-4,atol=2e-5)
    assert np.isfinite(a).all() and len(n)==280
    # Optimized Boolean search and recursive AST evaluation must agree exactly.
    tx=rng.normal(size=(200,6)).astype(np.float32);ty=np.tile(CL,50);seed=rng.random(200)>.5
    for budget in (1,2,3):
        ast,mask=fit_stage(tx,ty,rng.normal(size=200).astype(np.float32),seed,'gate',budget)
        assert np.array_equal(mask,evaluate(ast,tx,{'gate':seed}))
    X,names,m=load();ses=m['session'];y=m['label'];base=classify(initial(m));sn=np.array([int(s[-2:]) for s in ses])
    result={'new_feature_checks':'280 features: positive global gain x10 on five random and five real windows passed; sign equivariance and finiteness passed; three optimized-search/AST comparisons passed.',
            'development_first_search':bootstrap(dict(np.load(HERE/'oof_predictions.npz')))}
    first=json.loads((HERE/'results.json').read_text());ablations={}
    for keep in (('gate',),('nod',),('left',),('gate','nod'),('gate','left'),('nod','left')):
        p=base.copy()
        for block,budget in enumerate(first['selected_budgets']):
            held=(sn-1)//10==block
            model=json.loads((HERE/f'outer{block}_selected{budget}.json').read_text())
            for stage in ('gate','nod','left'):
                if stage not in keep:model[stage]=['base',stage]
            p[held]=predict(model,X[held],subset(m,held))
        ablations['+'.join(keep)]=ba(y,p)
    result['fixed_model_stage_ablations_development_only']=ablations
    if (HERE/'held_out_predictions.npz').exists():
        z=dict(np.load(HERE/'held_out_predictions.npz'));result['held_out_first_search']={}
        for subject in ('sub_01','sub_11'):
            ix=np.char.startswith(z['session'],subject)
            result['held_out_first_search'][subject]=bootstrap({k:v[ix] for k,v in z.items()})
    if (HERE/'robust_oof_predictions.npz').exists():
        z=dict(np.load(HERE/'robust_oof_predictions.npz')); result['development_robust_search']=bootstrap(z)
        recalls={}
        for source in ('prediction','baseline'):
            recalls[source]=float(np.mean([np.mean([np.mean(z[source][(z['label']==c)&(z['run']==g)]==c) for g in np.unique(z['run'][z['label']==c])]) for c in CL])*100)
        result['robust_recording_within_class_weighted_balanced_accuracy']=recalls
    if (HERE/'pruned_oof_predictions.npz').exists():
        result['development_pruned_search']=bootstrap(dict(np.load(HERE/'pruned_oof_predictions.npz')))
    model_count=0
    for path in HERE.glob('*.json'):
        data=json.loads(path.read_text())
        if set(data)=={'gate','nod','left'} and not path.name.startswith('pruned_'):model_count+=1
    result['saved_training_fits']=model_count
    (HERE/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
