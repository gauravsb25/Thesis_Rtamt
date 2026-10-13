"""Nested session evaluation of directional specialists and conservative monitor fusion."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[k]='1'
import sys, json, hashlib, time
from pathlib import Path
import numpy as np
from scipy.linalg import solve
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
# Reuse only data loading and pure helpers, never the previous experiment's training entry point.
source=(ROOT/'experiments/remaining_methods_2026_09_21/run.py').read_text()
prefix=source.split('cache={}; fits=0')[0]
(HERE/'input_helpers_snapshot.py').write_text(prefix,encoding='utf-8')
exec(compile(prefix,str(HERE/'input_helpers_snapshot.py'),'exec'),globals())
FAMILIES=['nod_specialist','turn_specialist','nested_specialists','agreement','prior_fusion']
CONFIGS={f:[(k,q) for k in (32,128) for q in ((.25,.5,1.) if f=='prior_fusion' else (0.,.5,.75))] for f in FAMILIES}
models={}; scores={}; fits=0

def expert(kind,k,ids):
    global fits
    key=(kind,k,tuple(sorted(ids)))
    if key in scores: return scores[key]
    train=np.logical_or.reduce([folds[i] for i in ids])
    use=train & (Y<2 if kind=='turn' else Y<3)
    labels=(Y[use]==2).astype(int) if kind=='nod' else Y[use]
    classes=3 if kind=='three' else 2
    sessions=S[use]; x=allx[use]; w=np.zeros(len(labels))
    for c in range(classes):
        groups=np.unique(sessions[labels==c])
        for s in groups:
            m=(labels==c)&(sessions==s); w[m]=1/(classes*len(groups)*m.sum())
    w*=len(w)
    mu=np.average(x,axis=0,weights=w); scale=np.maximum(np.sqrt(np.average((x-mu)**2,axis=0,weights=w)),1e-8)
    z=np.clip((x-mu)/scale,-10,10)
    means=np.array([np.average(z[labels==c],axis=0,weights=w[labels==c]) for c in range(classes)])
    cols=np.argsort(-np.mean(means**2,axis=0),kind='stable')[:k]
    z=np.column_stack([z[:,cols],np.ones(len(z))]); penalty=np.eye(z.shape[1])*100; penalty[-1,-1]=1e-8
    coef=solve((z*w[:,None]).T@z+penalty,(z*w[:,None]).T@np.eye(classes)[labels],assume_a='pos')
    model=dict(cols=cols,mu=mu[cols],scale=scale[cols],coef=coef[:-1],intercept=coef[-1])
    score=np.clip((allx[:,cols]-model['mu'])/model['scale'],-10,10)@model['coef']+model['intercept']
    models[key]=model; scores[key]=score; fits+=1
    return score

def prediction(family,config,ids,save=False):
    if config is None: return BASE.copy()
    k,q=config; train=np.logical_or.reduce([folds[i] for i in ids]); gate=BASE!=3
    n=expert('nod',k,ids); t=expert('turn',k,ids); z=expert('three',k,ids)
    nm=np.abs(n[:,1]-n[:,0]); tm=np.abs(t[:,1]-t[:,0]); zm=np.sort(z,axis=1)[:,-1]-np.sort(z,axis=1)[:,-2]
    cuts=[float(np.quantile(m[train & (Y<3)],q)) for m in (nm,tm,zm)] if family!='prior_fusion' else [0.,0.,0.]
    nh=n.argmax(1)==1; th=t.argmax(1); zh=z.argmax(1)
    base_turn=(M['turn']<=0).astype(int)
    nod_decision=np.where(nm>=cuts[0],nh,BASE==2)
    turn_decision=np.where(tm>=cuts[1],th,base_turn)
    nested=np.where(nod_decision,2,turn_decision)
    if family=='nod_specialist': result=np.where(nod_decision,2,base_turn)
    elif family=='turn_specialist': result=np.where(BASE==2,2,turn_decision)
    elif family=='nested_specialists': result=nested
    elif family=='agreement':
        # Both relevant binary specialists must be confident, not merely fall back to baseline.
        binary=np.where(nh,2,th)
        confident=(nm>=cuts[0]) & (nh | (tm>=cuts[1])) & (zm>=cuts[2])
        result=np.where(confident & (binary==zh),zh,BASE)
    elif family=='prior_fusion':
        fused=z.copy(); rows=np.flatnonzero(gate); fused[rows,BASE[rows]]+=q; result=fused.argmax(1)
    else: raise ValueError(family)
    result=np.where(gate,result,3)
    assert np.array_equal(result==3,BASE==3)
    if save:
        for kind in ('nod','turn','three'):
            np.savez_compressed(HERE/f'{family}_{kind}.npz',**models[(kind,k,tuple(sorted(ids)))])
        return result,dict(family=family,config=list(config),cuts=cuts,feature_names=allnames,baseline_fingerprint=str(M['fingerprint']))
    return result

def rec(pred,mask):
    return np.array([100*np.mean(pred[mask&(Y==i)]==i) for i in range(3)])
def objective(pred,mask):
    return float(rec(pred,mask).mean())
def select(ids,allowed):
    mask=np.logical_or.reduce([folds[i] for i in ids]); best=(objective(BASE,mask),None,None)
    base_rec=rec(BASE,mask)
    for f in allowed:
        for cfg in CONFIGS[f]:
            p=BASE.copy()
            for hold in ids:
                pr=prediction(f,cfg,[i for i in ids if i!=hold]); p[folds[hold]]=pr[folds[hold]]
            r=rec(p,mask); val=float(r.mean())
            if np.all(r>=base_rec-1.) and val>best[0]+1e-9: best=(val,f,cfg)
    return best

def detail(pred,mask):
    d=metrics(pred,mask); movement=mask&(Y<3)
    d.update(movement_macro=objective(pred,mask),movement_correct=int(np.sum(pred[movement]==Y[movement])),movement_total=int(movement.sum()),false_movements=int(np.sum(mask&(Y==3)&(pred!=3))))
    return d

def interval(pred,mask):
    sessions=np.unique(S[mask]); cells=[]
    for s in sessions:
        m=mask&(S==s); cells.append([[np.sum(m&(Y==c)&(a==c)) for c in range(3)] for a in (pred,BASE)]+[[np.sum(m&(Y==c)) for c in range(3)]])
    cells=np.array(cells); rng=np.random.default_rng(814); deltas=[]
    for _ in range(2000):
        v=cells[rng.integers(len(cells),size=len(cells))].sum(0)
        r=np.divide(v[:2],v[2],out=np.full((2,3),np.nan),where=v[2]>0)
        deltas.append(100*np.nanmean(r[0]-r[1]))
    return np.percentile(deltas,[2.5,97.5]).tolist()

report={'protocol':'PROTOCOL.md','baseline':{n:detail(BASE,m) for n,m in {'development':DEV,**splits}.items()},'families':{},'source_hash':hashlib.sha256(source.encode()).hexdigest()}
outputs={}; frozen={}
for family in FAMILIES+['selected_pipeline']:
    allowed=FAMILIES if family=='selected_pipeline' else [family]
    oof=BASE.copy(); chosen=[]
    for hold in range(3):
        ids=[i for i in range(3) if i!=hold]; val,f,cfg=select(ids,allowed)
        p=prediction(f,cfg,ids); oof[folds[hold]]=p[folds[hold]]
        chosen.append(dict(family=f,config=cfg,inner_movement_macro=val))
    val,f,cfg=select([0,1,2],allowed)
    frozen[family]=(f,cfg); outputs[family+'_oof']=oof
    report['families'][family]=dict(nested_validation=detail(oof,DEV),outer_selections=chosen,final_family=f,final_config=cfg,development_selection_score=val,validation_gain_ci=interval(oof,DEV))
    print(f'{family}: nested movement macro {objective(oof,DEV):.2f}; selected {f} {cfg}; fits={fits}',flush=True)
# All architecture choices are frozen before external scoring.
(HERE/'frozen_selections.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
for family,(f,cfg) in frozen.items():
    if cfg is None: p=BASE.copy(); manifest=dict(family='baseline',baseline_fingerprint=str(M['fingerprint']))
    else: p,manifest=prediction(f,cfg,[0,1,2],save=True)
    (HERE/f'{family}_monitor.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    outputs[family+'_final']=p
    row=report['families'][family]; row['training_fit']=detail(p,DEV)
    row['external']={n:dict(metrics=detail(p,m),movement_gain=objective(p,m)-objective(BASE,m),gain_ci=interval(p,m)) for n,m in splits.items()}
    row['eligible']=all(objective(p,m)>objective(BASE,m) and np.all(rec(p,m)>=rec(BASE,m)-1.) for m in splits.values())
    print(f'{family}: external movement macro '+', '.join(f'{n} {objective(p,m):.2f}' for n,m in splits.items())+f'; eligible={row["eligible"]}',flush=True)
report.update(fits=fits,elapsed_seconds=time.time()-START,concurrent_changes=[p for p,h in hashes().items() if initial[p]!=h])
(HERE/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
np.savez_compressed(HERE/'predictions.npz',label=Y,session=S,baseline=BASE,**outputs)
print('COMPLETE',report['fits'],report['elapsed_seconds'],report['concurrent_changes'],flush=True)
