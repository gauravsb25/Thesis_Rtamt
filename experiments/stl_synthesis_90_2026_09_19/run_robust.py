"""Run/session-balanced STL synthesis. Development only; no held-out scoring."""
import json
import time
import numpy as np
from features import HERE,ROOT,load
from synthesis import initial,classify,ba,fit,predict
from run_search import subset,summary,verify_formulas

def metadata(m):
    group=[];labels=[];sessions=[]
    for path in sorted((ROOT/'extracted_npz_dataset_v2'/'sub_01').glob('ses-*/clean_epochs.npz')):
        if int(path.parent.name[-2:])>30:continue
        z=np.load(path,allow_pickle=True);s='sub_01/'+path.parent.name
        group.extend([f'{s}/{t}/{r}' for t,r in zip(z['task'].astype(str),z['run'])])
        labels.extend(z['labels'].astype(str));sessions.extend([s]*len(z['labels']))
    assert np.array_equal(labels,m['label']) and np.array_equal(sessions,m['session'])
    return np.array(group)

def cached(tag,X,m,budget,group,order):
    path=HERE/(tag+'.json')
    if path.exists():return json.loads(path.read_text())
    print('fit',tag,flush=True); start=time.time()
    model=fit(X,m['label'],m,budget,group,order)
    path.write_text(json.dumps(model,indent=2)); print('done',tag,round(time.time()-start,1),flush=True)
    return model

def main():
    X,names,m=load();y=m['label'];ses=m['session'];run=metadata(m);sn=np.array([int(s[-2:]) for s in ses])
    base=classify(initial(m));oof=base.copy()
    orders={'gate_first':('gate','nod','left'),'nod_first':('nod','left','gate')}
    configs=[('window','gate_first',b) for b in (1,2,3)]+[(w,o,b) for w in ('session','run') for o in orders for b in (1,2,3)]
    inner_records=[];outer_records=[];choices=[]
    for block in (0,1,2):
        held=(sn-1)//10==block;train=~held
        inner={str(c):base.copy() for c in configs}
        for fold in (0,1):
            valid=train&(((sn-1)//5)%2==fold);mask=train&~valid
            for w,o,b in configs:
                tag=f'outer{block}_inner{fold}_budget{b}' if w=='window' else f'robust_outer{block}_inner{fold}_{w}_{o}_budget{b}'
                groups=None if w=='window' else (ses if w=='session' else run)[mask]
                model=cached(tag,X[mask],subset(m,mask),b,groups,orders[o])
                inner[str((w,o,b))][valid]=predict(model,X[valid],subset(m,valid))
        scores={str(c):ba(y[train],inner[str(c)][train]) for c in configs}
        choice=max(configs,key=lambda c:scores[str(c)])
        inner_records.append(scores); choices.append(choice)
        w,o,b=choice
        if scores[str(choice)]<=ba(y[train],base[train]):
            model={k:['base',k] for k in initial(m)}
        else:
            groups=None if w=='window' else (ses if w=='session' else run)[train]
            model=cached(f'robust_outer{block}_selected_{w}_{o}_{b}',X[train],subset(m,train),b,groups,orders[o])
        oof[held]=predict(model,X[held],subset(m,held))
        result=summary(y[held],oof[held],base[held],ses[held]);outer_records.append(result)
        print('ROBUST OUTER',block,'choice',choice,json.dumps(result),flush=True)
    final=max(configs,key=lambda c:np.mean([r[str(c)] for r in inner_records]));w,o,b=final
    groups=None if w=='window' else ses if w=='session' else run
    model=cached(f'robust_full_{w}_{o}_{b}',X,m,b,groups,orders[o])
    checks,formulas=verify_formulas(model,names,X,m)
    result={'development':summary(y,oof,base,ses),'outer_folds':outer_records,'inner_scores':inner_records,
            'selected_configurations':choices,'frozen_configuration':final,'rtamt_checked_windows':checks,
            'training_accuracy_not_validation':ba(y,predict(model,X,m)),
            'held_out':'Not evaluated in this continuation, as specified before running.'}
    (HERE/'robust_results.json').write_text(json.dumps(result,indent=2))
    (HERE/'robust_frozen_specification.json').write_text(json.dumps({'model':model,'feature_names':names.tolist(),'formulas':formulas},indent=2))
    np.savez_compressed(HERE/'robust_oof_predictions.npz',label=y,session=ses,run=run,baseline=base,prediction=oof)
    print('ROBUST FINAL',json.dumps(result),flush=True)

if __name__=='__main__':main()
