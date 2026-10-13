"""Nested session validation, freeze, then one eligible held-out evaluation."""
import hashlib
import json
import sys
import time
import numpy as np
from features import HERE,ROOT,load
from synthesis import CL,initial,classify,ba,fit,predict,export

def subset(m,ix): return {k:v[ix] for k,v in m.items()}

def summary(y,p,base,ses):
    changes={s:ba(y[ses==s],p[ses==s])-ba(y[ses==s],base[ses==s]) for s in np.unique(ses)}
    return dict(balanced_accuracy=ba(y,p),baseline=ba(y,base),gain=ba(y,p)-ba(y,base),
                sessions_up=sum(v>1e-8 for v in changes.values()),sessions_down=sum(v< -1e-8 for v in changes.values()),
                session_gains=changes,per_class={c:float(np.mean(p[y==c]==c)*100) for c in CL},
                fixes=int(np.sum((base!=y)&(p==y))),breaks=int(np.sum((base==y)&(p!=y))))

def cached_fit(tag,X,m,budget):
    file=HERE/(tag+'.json')
    if file.exists(): return json.loads(file.read_text())
    start=time.time(); print('fit',tag,'windows',len(X),'budget',budget,flush=True)
    model=fit(X,m['label'],m,budget)
    file.write_text(json.dumps(model,indent=2))
    print('fit finished',tag,round(time.time()-start,1),'seconds',flush=True)
    return model

def verify_formulas(model,names,X,m):
    sys.path.insert(0,str(ROOT))
    from monitoring.rtamt_monitors import import_rtamt
    rtamt=import_rtamt(); formulas=export(model,names)
    used=set()
    def visit(ast):
        if ast[0]=='lit': used.add(ast[1])
        elif ast[0] in ('and','or'): visit(ast[1]);visit(ast[2])
    for ast in model.values(): visit(ast)
    # All classes and a sample of changed cases; one completed window per time step.
    seed=initial(m); expected=predict(model,X,m); base=classify(seed)
    indices=np.unique(np.concatenate([np.flatnonzero(m['label']==c)[::max(1,np.sum(m['label']==c)//200)] for c in CL]+[np.flatnonzero(expected!=base)[:300]]))
    data={'time':list(range(len(indices))),**{str(names[j]):X[indices,j].astype(float).tolist() for j in used},
          **{k+'0':v[indices].astype(float).tolist() for k,v in seed.items()}}
    results={}
    for c,formula in formulas.items():
        spec=rtamt.StlDiscreteTimeOfflineSpecification()
        for key in data:
            if key!='time':spec.declare_var(key,'float')
        spec.spec=formula;spec.parse()
        results[c]=np.array([v for _,v in spec.evaluate(data)])>=0
    # No threshold ties are permitted to make non-strict numerical robustness ambiguous.
    for k,c in enumerate(expected[indices]):
        satisfied=[label for label in CL if results[label][k]]
        if satisfied!=[c]: raise AssertionError((int(indices[k]),c,satisfied))
    print('RTAMT export matched',len(indices),'windows',flush=True)
    return len(indices),formulas

def main():
    X,names,m=load();y=m['label'];ses=m['session']; sn=np.array([int(s[-2:]) for s in ses])
    base=classify(initial(m));oof=base.copy();budgets=[];inner_results=[];outer_results=[]
    for block in (0,1,2):
        held=(sn-1)//10==block;train=~held
        inner_oof={b:base.copy() for b in (1,2,3)}
        for fold in (0,1):
            valid=train&(((sn-1)//5)%2==fold);fitmask=train&~valid
            for budget in (1,2,3):
                model=cached_fit(f'outer{block}_inner{fold}_budget{budget}',X[fitmask],subset(m,fitmask),budget)
                inner_oof[budget][valid]=predict(model,X[valid],subset(m,valid))
                print('inner score',block,fold,budget,ba(y[valid],inner_oof[budget][valid]),flush=True)
        scores={b:ba(y[train],p[train]) for b,p in inner_oof.items()}
        # Baseline is an eligible configuration; no selection forced if no improvement.
        budget=max(scores,key=lambda b:scores[b]);budgets.append(budget)
        inner_results.append(scores)
        if scores[budget]<=ba(y[train],base[train]):
            model={k:['base',k] for k in ('gate','nod','left')}
        else:model=cached_fit(f'outer{block}_selected{budget}',X[train],subset(m,train),budget)
        oof[held]=predict(model,X[held],subset(m,held))
        record=summary(y[held],oof[held],base[held],ses[held]);outer_results.append(record)
        print('OUTER',block,json.dumps(record),flush=True)
    result={'features':len(names),'new_temporal_features':280,'outer_folds':outer_results,'inner_scores':inner_results,
            'selected_budgets':budgets,'development':summary(y,oof,base,ses)}
    d=result['development']; passed=d['gain']>=.5 and d['sessions_up']>d['sessions_down'] and sum(r['gain']>0 for r in outer_results)>=2
    result['eligible_for_test']=passed
    np.savez_compressed(HERE/'oof_predictions.npz',label=y,session=ses,baseline=base,prediction=oof)
    (HERE/'development_results.json').write_text(json.dumps(result,indent=2))
    print('DEVELOPMENT',json.dumps(d),'eligible',passed,flush=True)
    # Export the best nested-selected budget even if ineligible, for inspection/verification only.
    final_budget=max((1,2,3),key=lambda b:np.mean([r[b] for r in inner_results]))
    model=cached_fit(f'full_development_budget{final_budget}',X,m,final_budget)
    checks,formulas=verify_formulas(model,names,X,m)
    frozen={'budget':final_budget,'model':model,'feature_names':names.tolist(),'formulas':formulas,
            'training_accuracy_not_validation':ba(y,predict(model,X,m)),'rtamt_checked_windows':checks,
            'eligible_for_test':passed}
    frozen_path=HERE/'frozen_specification.json';frozen_path.write_text(json.dumps(frozen,indent=2))
    result['frozen_sha256']=hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    result['training_accuracy_not_validation']=frozen['training_accuracy_not_validation'];result['rtamt_checked_windows']=checks
    # Timestamped separate freeze artifact is written before any test extraction.
    (HERE/'pretest_record.json').write_text(json.dumps(result,indent=2))
    if passed:
        tx,tn,tm=load(test=True); assert np.array_equal(tn,names)
        tp=predict(model,tx,tm);tb=classify(initial(tm)); result['held_out']={}
        for subject in ('sub_01','sub_11'):
            ix=np.char.startswith(tm['session'].astype(str),subject)
            result['held_out'][subject]=summary(tm['label'][ix],tp[ix],tb[ix],tm['session'][ix])
        np.savez_compressed(HERE/'held_out_predictions.npz',label=tm['label'],session=tm['session'],baseline=tb,prediction=tp)
        result['adoptable']=all(r['gain']>=0 for r in result['held_out'].values())
    else: result['held_out']='Not evaluated: development eligibility criterion failed.'; result['adoptable']=False
    (HERE/'results.json').write_text(json.dumps(result,indent=2))
    print('FINAL',json.dumps(result),flush=True)

if __name__=='__main__':main()
