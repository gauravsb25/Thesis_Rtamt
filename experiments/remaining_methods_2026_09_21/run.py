import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[k]='1'
import sys, json, hashlib, time
from pathlib import Path
import numpy as np
from scipy.linalg import solve
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import project_code as project
START=time.time()
CL=['LEFT','RIGHT','DOWN','STRAIGHT']
paths=[ROOT/'measurements_v2.npz',ROOT/'monitor_labels_v2.npz',ROOT/'experiments/operator_sweep_2026_09_21/bank.npz']
paths+=list(sorted((ROOT/'monitoring').glob('*.py')))
paths += [ROOT/'experiments/operator_sweep_2026_09_21/signals.py']
def hashes(): return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
initial=hashes()
M=dict(np.load(paths[0],allow_pickle=True)); L=dict(np.load(paths[1],allow_pickle=True)); B=dict(np.load(paths[2],allow_pickle=True))
assert str(M['fingerprint'])==str(L['fingerprint'])==project.fingerprint(), 'Stale original cache: stop, do not overwrite'
S=M['session'].astype(str); Y=np.array([CL.index(x) for x in M['label']]); BASE=np.array([CL.index(x) for x in L['with_history']])
for d in (L,B):
    assert np.array_equal(d['session'],S) and np.array_equal(d['label'],M['label'])
P=project.SHIPPED
move=np.where(M['background_valid']>=.5,M['background']>=P['b'],M['contrast']>=P['g']) & (M['delta']>=P['d']) & (M['deflect']>=P['q'])
nod=(M['nod']>=P['n']) & (M['sideways']<=P['v'])
arithmetic=np.where(~move,3,np.where(nod,2,np.where(M['turn']>0,0,1)))
assert np.array_equal(arithmetic,BASE),'Arithmetic/genuine RTAMT disagreement'
corrections={}
for k in B:
    if k.endswith('|resp'):
        no_trigger=B[k.replace('|resp','|max')]<2
        corrections[k]=int(np.count_nonzero(no_trigger & (B[k]!=0)))
        B[k]=np.where(no_trigger,0,B[k])
keys=list(project.KEYS)
orig=np.column_stack([M[k] for k in keys]).astype(float)
bankkeys=sorted(k for k in B if k not in ('label','session'))
bank=np.column_stack([B[k] for k in bankkeys]).astype(float)
allx=np.column_stack([orig,bank]); allnames=keys+bankkeys
invops={'absmax','dwell2','dwell3','first2','last2','run2','run3','calm','slope','sep'}
inv=[i for i,k in enumerate(allnames) if (k in keys and k!='turn') or ('|' in k and k.split('|')[1] in invops)]
assert np.isfinite(allx).all()
DEV=np.char.startswith(S,'sub_01') & np.array([int(s[-2:])<=30 for s in S])
splits={'external_sub01':np.char.startswith(S,'sub_01') & ~DEV,'external_sub11':np.char.startswith(S,'sub_11')}
folds=[DEV & np.isin(S,[f'sub_01/ses-{n:02d}' for n in range(a,a+10)]) for a in (1,11,21)]
families={
 'original_ridge':[(8,a) for a in (1.,10.,100.)],
 'operator_ridge':[(k,a) for k in (32,128) for a in (10.,100.)],
 'operator_lda':[(k,a) for k in (32,128) for a in (.5,.9)],
 'nonlinear_ridge':[(32,a) for a in (.3,1.,3.)],
 'hybrid_gate':[(k,a) for k in (32,128) for a in (10.,100.)],
 'polarity_invariant':[(k,a) for k in (32,128) for a in (10.,100.)]}

def metrics(pred,mask):
    cm=np.zeros((4,4),int); np.add.at(cm,(Y[mask],pred[mask]),1)
    den=cm.sum(1); rec=np.divide(cm.diagonal(),den,out=np.full(4,np.nan),where=den>0)
    return dict(n=int(mask.sum()),correct=int(cm.trace()),accuracy=float(100*cm.trace()/max(1,mask.sum())),balanced_accuracy=float(100*np.nanmean(rec)),recall=dict(zip(CL,(100*rec).tolist())),confusion=cm.tolist())

def fit(fam,config,train):
    count,param=config
    cols=np.arange(8) if fam=='original_ridge' else (np.array(inv) if fam=='polarity_invariant' else np.arange(allx.shape[1]))
    use=train & (Y!=3) if fam=='hybrid_gate' else train
    labels=Y[use]; sessions=S[use]; x=allx[use][:,cols]; classes=3 if fam=='hybrid_gate' else 4
    w=np.zeros(len(labels))
    for c in range(classes):
        groups=np.unique(sessions[labels==c])
        for s in groups:
            m=(labels==c)&(sessions==s); w[m]=1/(classes*len(groups)*m.sum())
    w*=len(w)
    mu=np.average(x,axis=0,weights=w); scale=np.sqrt(np.average((x-mu)**2,axis=0,weights=w)); scale=np.maximum(scale,1e-8)
    z=np.clip((x-mu)/scale,-10,10)
    means=np.array([np.average(z[labels==c],axis=0,weights=w[labels==c]) for c in range(classes)])
    rank=np.mean(means**2,axis=0)
    sel=np.argsort(-rank,kind='stable')[:min(count,len(cols))]
    cols=cols[sel]; mu=mu[sel]; scale=scale[sel]; z=z[:,sel]
    model=dict(cols=cols,mu=mu,scale=scale)
    if fam=='nonlinear_ridge':
        rng=np.random.default_rng(712)
        omega=rng.normal(size=(z.shape[1],256))*np.sqrt(2*param/z.shape[1]); phase=rng.uniform(0,2*np.pi,256)
        model.update(omega=omega,phase=phase)
        z=np.sqrt(2/256)*np.cos(z@omega+phase); alpha=10.
    else: alpha=param
    if fam=='operator_lda':
        means=np.array([np.average(z[labels==c],axis=0,weights=w[labels==c]) for c in range(classes)])
        r=z-means[labels]; cov=(r*w[:,None]).T@r/w.sum()
        cov=(1-param)*cov+param*np.diag(np.diag(cov))+np.eye(len(cols))*1e-5
        coef=solve(cov,means.T,assume_a='pos'); intercept=-.5*np.sum(means*coef.T,axis=1)
    else:
        z=np.column_stack([z,np.ones(len(z))]); target=np.eye(classes)[labels]
        penalty=np.eye(z.shape[1])*alpha; penalty[-1,-1]=1e-8
        full=solve((z*w[:,None]).T@z+penalty,(z*w[:,None]).T@target,assume_a='pos')
        coef,intercept=full[:-1],full[-1]
    model.update(coef=coef,intercept=intercept)
    return model

def predict(fam,model):
    z=np.clip((allx[:,model['cols']]-model['mu'])/model['scale'],-10,10)
    if 'omega' in model: z=np.sqrt(2/256)*np.cos(z@model['omega']+model['phase'])
    score=z@model['coef']+model['intercept']; pred=score.argmax(1)
    if fam=='hybrid_gate': pred=np.where(move,pred,3)
    ordered=np.sort(score,axis=1); margin=ordered[:,-1]-ordered[:,-2]
    return pred,margin

cache={}; fits=0
def predictions(fam,config,ids):
    global fits
    key=(fam,config,tuple(sorted(ids)))
    if key not in cache:
        train=np.logical_or.reduce([folds[i] for i in ids]); model=fit(fam,config,train)
        cache[key]=predict(fam,model)[0]; fits+=1
    return cache[key]

def choose(fam,ids):
    scores=[]
    for config in families[fam]:
        p=BASE.copy()
        for hold in ids:
            fitted=predictions(fam,config,[i for i in ids if i!=hold]); p[folds[hold]]=fitted[folds[hold]]
        scores.append(metrics(p,np.logical_or.reduce([folds[i] for i in ids]))['balanced_accuracy'])
    return families[fam][int(np.argmax(scores))],scores

def paired(pred,mask):
    sessions=np.unique(S[mask]); mats=[]
    for s in sessions:
        m=mask&(S==s); mats.append((np.array(metrics(pred,m)['confusion']),np.array(metrics(BASE,m)['confusion'])))
    mats=np.array(mats); rng=np.random.default_rng(83); ds=[]
    for _ in range(2000):
        a=mats[rng.integers(len(sessions),size=len(sessions))].sum(0)
        recalls=np.divide(np.diagonal(a,axis1=1,axis2=2),a.sum(2),out=np.full((2,4),np.nan),where=a.sum(2)>0)
        ds.append(100*(np.nanmean(recalls[0])-np.nanmean(recalls[1])))
    changes=[metrics(pred,mask&(S==s))['balanced_accuracy']-metrics(BASE,mask&(S==s))['balanced_accuracy'] for s in sessions]
    return dict(gain=metrics(pred,mask)['balanced_accuracy']-metrics(BASE,mask)['balanced_accuracy'],paired_session_bootstrap_95=np.percentile(ds,[2.5,97.5]).tolist(),sessions_improved=sum(g>1e-8 for g in changes),sessions_harmed=sum(g< -1e-8 for g in changes))

report={'protocol':'PROTOCOL.md','fingerprint':str(M['fingerprint']),'hashes':initial,'response_corrected_values':corrections,'baseline':{name:metrics(BASE,m) for name,m in {'development':DEV,**splits}.items()},'families':{}}
frozen={}; outputs={}
for fam in families:
    oof=BASE.copy(); selections=[]
    for hold in range(3):
        ids=[i for i in range(3) if i!=hold]; config,sc=choose(fam,ids)
        p=predictions(fam,config,ids); oof[folds[hold]]=p[folds[hold]]; selections.append({'config':config,'inner_scores':sc})
    config,scores=choose(fam,[0,1,2]); model=fit(fam,config,DEV); fits+=1
    # Freeze coefficients before any external scoring.
    np.savez_compressed(HERE/f'{fam}_frozen.npz',**model)
    frozen[fam]=(config,model); outputs[fam+'_oof']=oof
    report['families'][fam]={'outer_selections':selections,'full_config':config,'development_selection_scores':scores,'nested_validation':metrics(oof,DEV),'validation_comparison':paired(oof,DEV)}
    print(f'{fam}: nested BA {metrics(oof,DEV)["balanced_accuracy"]:.2f}%, gain {paired(oof,DEV)["gain"]:+.2f}; {fits} fits, {time.time()-START:.1f}s',flush=True)
(HERE/'development_frozen.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
for fam,(config,model) in frozen.items():
    pred,margin=predict(fam,model); outputs[fam+'_final']=pred
    row=report['families'][fam]; row['training_fit']=metrics(pred,DEV)
    row['external']={name:dict(metrics=metrics(pred,m),comparison=paired(pred,m)) for name,m in splits.items()}
    row['abstention']={}
    for q in (.1,.25,.5):
        threshold=float(np.quantile(margin[DEV],q)); accepted=margin>=threshold
        row['abstention'][str(q)]={'training_margin_threshold':threshold,'external':{name:dict(coverage=float(100*(m&accepted).sum()/m.sum()),metrics=metrics(pred,m&accepted)) for name,m in splits.items()}}
    print(f'{fam}: external BA '+', '.join(f'{name}={metrics(pred,m)["balanced_accuracy"]:.2f}%' for name,m in splits.items()),flush=True)
report['fits']=fits; report['elapsed_seconds']=time.time()-START
final=hashes(); report['concurrent_changes']=[p for p in initial if initial[p]!=final[p]]
(HERE/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
np.savez_compressed(HERE/'predictions.npz',label=Y,session=S,baseline=BASE,**outputs)
print('COMPLETE',fits,'fits;',round(time.time()-START,1),'seconds; concurrent changes:',report['concurrent_changes'],flush=True)
