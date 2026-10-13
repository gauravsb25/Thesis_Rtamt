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

