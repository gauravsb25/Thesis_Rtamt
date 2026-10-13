"""Inner-selected stage subsets; no new signal tuning or held-out access."""
import itertools
import json
import numpy as np
from features import HERE,load
from synthesis import initial,classify,ba,predict
from run_search import subset,summary,verify_formulas
from run_robust import metadata,cached

def prune(model,keep):return {k:ast if k in keep else ['base',k] for k,ast in model.items()}

def main():
    X,names,m=load();y=m['label'];ses=m['session'];sn=np.array([int(s[-2:]) for s in ses]);runs=metadata(m)
    base=classify(initial(m));oof=base.copy()
    orders={'gate_first':('gate','nod','left'),'nod_first':('nod','left','gate')}
    configs=[('window','gate_first',b) for b in (1,2,3)]+[(w,o,b) for w in ('session','run') for o in orders for b in (1,2,3)]
    keeps=[v for size in (1,2,3) for v in itertools.combinations(('gate','nod','left'),size)]
    options=[(c,k) for k in keeps for c in configs]
    choices=[];records=[];inner_records=[]
    for block in (0,1,2):
        held=(sn-1)//10==block;train=~held
        inner={str(o):base.copy() for o in options}
        for fold in (0,1):
            valid=train&(((sn-1)//5)%2==fold)
            for config in configs:
                w,o,b=config
                tag=f'outer{block}_inner{fold}_budget{b}' if w=='window' else f'robust_outer{block}_inner{fold}_{w}_{o}_budget{b}'
                model=json.loads((HERE/(tag+'.json')).read_text())
                for keep in keeps:inner[str((config,keep))][valid]=predict(prune(model,keep),X[valid],subset(m,valid))
        scores={str(o):ba(y[train],inner[str(o)][train]) for o in options}
        choice=max(options,key=lambda o:scores[str(o)]);config,keep=choice;w,o,b=config
        if scores[str(choice)]<=ba(y[train],base[train]): model={k:['base',k] for k in initial(m)}
        else:
            groups=None if w=='window' else (ses if w=='session' else runs)[train]
            model=prune(cached(f'robust_outer{block}_selected_{w}_{o}_{b}',X[train],subset(m,train),b,groups,orders[o]),keep)
        oof[held]=predict(model,X[held],subset(m,held)); choices.append(choice);inner_records.append(scores)
        r=summary(y[held],oof[held],base[held],ses[held]);records.append(r)
        print('PRUNED OUTER',block,choice,json.dumps(r),flush=True)
        (HERE/f'pruned_outer{block}_model.json').write_text(json.dumps(model,indent=2))
    final=max(options,key=lambda o:np.mean([r[str(o)] for r in inner_records]));(w,o,b),keep=final
    groups=None if w=='window' else ses if w=='session' else runs
    model=prune(cached(f'robust_full_{w}_{o}_{b}',X,m,b,groups,orders[o]),keep)
    checks,formulas=verify_formulas(model,names,X,m)
    result={'development':summary(y,oof,base,ses),'outer_folds':records,'choices':choices,'frozen_choice':final,
            'rtamt_checked_windows':checks,'training_accuracy_not_validation':ba(y,predict(model,X,m)),
            'held_out':'Not evaluated; development-only protocol.'}
    (HERE/'pruned_results.json').write_text(json.dumps(result,indent=2))
    (HERE/'pruned_frozen_specification.json').write_text(json.dumps({'model':model,'feature_names':names.tolist(),'formulas':formulas},indent=2))
    np.savez_compressed(HERE/'pruned_oof_predictions.npz',label=y,session=ses,run=runs,baseline=base,prediction=oof)
    print('PRUNED FINAL',json.dumps(result),flush=True)

if __name__=='__main__':main()
