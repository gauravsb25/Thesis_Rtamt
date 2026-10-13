"""Independent spectral-pattern benchmark. No production files are written.

Fixed models; session-block validation; selection before external scoring.
Uses no movement annotation, cue, task or subject ID as a predictor.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import json
import hashlib
from pathlib import Path
import numpy as np
from scipy.signal import spectrogram, butter, sosfilt
from scipy.linalg import eigh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLASSES = ['LEFT', 'RIGHT', 'DOWN', 'STRAIGHT']
BANDS = [(1,4),(4,8),(8,13),(13,20),(20,30),(30,45)]
CHANNELS = ['Fp1','Fp2','F3','Fz','F4','T3','Cz','T4']

def features(x):
    """x: batch, channel, sample in uV. Only samples inside the window.

    Causal band filters start at zero; discard first second for covariance.
    STFT uses complete one-second frames: frame-end times 1,1.25,...,3 s.
    Early/late changes are NOT labelled physiological ERD (no rest baseline).
    """
    f, t, psd = spectrogram(x, fs=256, nperseg=256, noverlap=192, axis=-1)
    power = np.stack([psd[..., (f>=a)&(f<b), :].sum(axis=-2) for a,b in BANDS], axis=1)
    tiny = 1e-12
    rel = power / (power.sum(axis=1, keepdims=True) + tiny)
    lp = np.log(power + tiny)
    mean = power.mean(axis=-1)
    # Joint spatial power proportions and relative frequency power.
    static = np.concatenate([np.log(rel.mean(axis=-1)+tiny).reshape(len(x),-1),
        np.log(mean/(mean.sum(axis=-1,keepdims=True)+tiny)+tiny).reshape(len(x),-1)], axis=1)
    # Retain order, change, and sustained suppression in every band/channel.
    temporal = np.concatenate([(lp[...,-3:].mean(-1)-lp[...,:3].mean(-1)).reshape(len(x),-1),
        (lp[...,3:6].mean(-1)-lp[...,:3].mean(-1)).reshape(len(x),-1),
        (lp.max(-1)-lp.min(-1)).reshape(len(x),-1)], axis=1)
    covs = []
    for a,b in BANDS:
        z = sosfilt(butter(4,[a,b],btype='bandpass',fs=256,output='sos'), x, axis=-1)[...,256:]
        z = z-z.mean(axis=-1,keepdims=True)
        c = np.einsum('nct,ndt->ncd',z,z)/z.shape[-1]
        c /= np.trace(c,axis1=1,axis2=2)[:,None,None]+tiny
        covs.append(c)
    return static.astype('float32'), temporal.astype('float32'), np.stack(covs,axis=1).astype('float32')

def extract():
    path = HERE/'features.npz'
    if path.exists():
        return dict(np.load(path,allow_pickle=False))
    parts=[]; manifest=[]
    for p in sorted((ROOT/'extracted_npz_dataset_v2').glob('*/ses-*/clean_epochs.npz')):
        d=np.load(p,allow_pickle=False); outputs=[]
        for start in range(0,len(d['labels']),128):
            outputs.append(features(d['epochs'][start:start+128,:8]*1e6))
        static,temporal,cov=[np.concatenate([o[k] for o in outputs]) for k in range(3)]
        ses=f'{p.parent.parent.name}/{p.parent.name}'
        trial=np.array([f'{ses}/{t}/{r}' for t,r in zip(d['task'],d['run'])])
        parts.append(dict(static=static,temporal=temporal,cov=cov,label=d['labels'].astype(str),
            session=np.repeat(ses,len(static)),trial=trial))
        manifest.append(dict(path=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns))
        print('extracted',ses,len(static),flush=True)
    out={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(path,**out)
    (HERE/'inputs.json').write_text(json.dumps(manifest,indent=2))
    return out

def weights(y,trial):
    # Equal weight per class, then equal per recording within class.
    w=np.zeros(len(y))
    for c in np.unique(y):
        groups=np.unique(trial[y==c])
        for g in groups:
            ix=(y==c)&(trial==g); w[ix]=1/(len(groups)*ix.sum())
    return w/w.sum()

class LDA:
    def fit(self,x,y,w):
        self.center=np.sum(x*w[:,None],axis=0)
        self.scale=np.sqrt(np.sum((x-self.center)**2*w[:,None],axis=0)).clip(.0001)
        z=(x-self.center)/self.scale
        self.classes=np.unique(y)
        mu=np.stack([np.average(z[y==c],axis=0,weights=w[y==c]) for c in self.classes])
        residual=z-mu[np.searchsorted(self.classes,y)]
        cov=(residual*w[:,None]).T@residual
        # Fixed strong shrinkage; no hyperparameter search on external data.
        cov=.5*cov+.5*np.eye(x.shape[1])*max(np.trace(cov)/x.shape[1],1e-6)
        self.coef=np.linalg.solve(cov,mu.T)
        self.bias=-.5*np.sum(mu*self.coef.T,axis=1)
        return self
    def predict(self,x):
        return self.classes[np.argmax(((x-self.center)/self.scale)@self.coef+self.bias,axis=1)]

def fit_csp(cov,y,w):
    filters=[]
    for b in range(len(BANDS)):
        bank=[]
        for c in np.unique(y):
            a=np.average(cov[y==c,b],axis=0,weights=w[y==c])
            z=np.average(cov[y!=c,b],axis=0,weights=w[y!=c])
            a=.9*a+.1*np.eye(8)/8; z=.9*z+.1*np.eye(8)/8
            _,v=eigh(a,a+z)
            bank.extend([v[:,0],v[:,-1]])
        filters.append(np.stack(bank,axis=1))
    return filters

def csp_features(cov,filters):
    return np.concatenate([np.log(np.einsum('ci,ncd,di->ni',v,cov[:,b],v).clip(1e-12))
        for b,v in enumerate(filters)],axis=1)

def metrics(y,p):
    cm=np.array([[int(np.sum((y==a)&(p==b))) for b in range(4)] for a in range(4)])
    totals=cm.sum(1); recall=np.divide(cm.diagonal(),totals,out=np.zeros(4),where=totals>0)
    return dict(n=len(y),correct=int((y==p).sum()),accuracy=float(100*(y==p).mean()),
        balanced_accuracy=float(100*recall[totals>0].mean()),movement_macro=float(100*recall[:3].mean()),
        recall=(100*recall).tolist(),confusion=cm.tolist())

def baseline(m):
    move=np.where(m['background_valid']>=.5,m['background']>=6.61,m['contrast']>=7.81)
    move &= (m['delta']>=.208)&(m['deflect']>=2.72)
    return np.where(~move,3,np.where((m['nod']>=-.661)&(m['sideways']<=7.54),2,np.where(m['turn']>0,0,1)))

def main():
    d=extract(); y=np.array([CLASSES.index(v) for v in d['label']]); ses=d['session']; trial=d['trial']
    m=dict(np.load(ROOT/'measurements_v2.npz',allow_pickle=False))
    cached=dict(np.load(ROOT/'monitor_labels_v2.npz',allow_pickle=False))
    assert np.array_equal(m['session'],ses) and np.array_equal(m['label'],d['label'])
    assert np.array_equal(cached['session'],ses) and np.array_equal(cached['label'],d['label'])
    base=baseline(m)
    assert np.array_equal(np.array(CLASSES)[base],cached['with_history'])
    monitor=np.stack([m[k] for k in ['contrast','background','background_valid','delta','deflect','nod','sideways','turn']],axis=1)
    monitor=np.sign(monitor)*np.log1p(np.abs(monitor))
    n=np.array([int(s[-2:]) for s in ses]); source=np.char.startswith(ses,'sub_01')
    dev=source&(n<=30); external={'sub01_later':source&(n>30),'sub11':~source}
    # Fixed, small ablation set. Hybrid preserves existing movement gate; replaces
    # only LEFT/RIGHT/DOWN decision, with every movement included in the metric.
    names=['joint_bandpower','ordered_bandpower','filterbank_csp','monitor_lda_control',
           'monitor_plus_spectral','hybrid_csp','hybrid_ordered']
    def predict(name,tr,te):
        if name.startswith('hybrid_'): tr=tr&(y!=3)
        w=weights(y[tr],trial[tr])
        if 'csp' in name:
            filters=fit_csp(d['cov'][tr],y[tr],w)
            x=csp_features(d['cov'],filters)
        elif name=='monitor_lda_control': x=monitor
        elif name=='joint_bandpower': x=d['static']
        else: x=np.concatenate([d['static'],d['temporal']],axis=1)
        if name=='monitor_plus_spectral': x=np.concatenate([monitor,x],axis=1)
        model=LDA().fit(x[tr],y[tr],w); p=model.predict(x[te])
        if name.startswith('hybrid_'): p=np.where(base[te]==3,3,p)
        return p
    cv={}; results={'protocol':{'classes':CLASSES,'selection':'movement macro, then balanced accuracy; development only',
        'adopt':'CV movement gain >= 0.5 points; BA no loss; more sessions improve than worsen; both external splits no loss',
        'external_status':'Previously exposed retrospective sets, not blind tests',
        'models':'research classifiers, not RTAMT production monitors'},'development':{},'external':{}}
    results['development']['current_monitor']=metrics(y[dev],base[dev])
    for name in names:
        pred=base.copy(); foldrows=[]
        for block in range(3):
            te=dev&(n>block*10)&(n<=(block+1)*10); tr=dev&~te
            assert not set(trial[tr])&set(trial[te])
            pred[te]=predict(name,tr,te)
            foldrows.append(metrics(y[te],pred[te]))
        row=metrics(y[dev],pred[dev]); row['folds']=foldrows
        gains=[metrics(y[ses==s],pred[ses==s])['movement_macro']-metrics(y[ses==s],base[ses==s])['movement_macro'] for s in np.unique(ses[dev])]
        row['sessions_better']=sum(g>1e-8 for g in gains); row['sessions_worse']=sum(g< -1e-8 for g in gains)
        results['development'][name]=row; cv[name]=pred
        print(name,json.dumps({k:v for k,v in row.items() if k not in ['folds','confusion']}),flush=True)
    ranked=sorted(names,key=lambda k:(results['development'][k]['movement_macro'],results['development'][k]['balanced_accuracy']),reverse=True)
    winner=ranked[0]; results['selected']=winner
    # Write selection before any external scoring. Do not try other models there.
    results['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (HERE/'selection.json').write_text(json.dumps(results,indent=2))
    for name in [winner]:
        results['training_fit']=metrics(y[dev],predict(name,dev,dev))
        for key,te in external.items():
            p=predict(name,dev,te)
            results['external'][key]={'current_monitor':metrics(y[te],base[te]),name:metrics(y[te],p)}
    b=results['development']['current_monitor']; a=results['development'][winner]
    passed=a['movement_macro']>=b['movement_macro']+.5 and a['balanced_accuracy']>=b['balanced_accuracy'] and a['sessions_better']>a['sessions_worse']
    results['development_pass']=passed
    results['adopt']=passed and all(r[winner]['movement_macro']>=r['current_monitor']['movement_macro'] and r[winner]['balanced_accuracy']>=r['current_monitor']['balanced_accuracy'] for r in results['external'].values())
    np.savez_compressed(HERE/'oof_predictions.npz',y=y,session=ses,trial=trial,dev=dev,baseline=base,**cv)
    (HERE/'results.json').write_text(json.dumps(results,indent=2))
    print('SELECTED',winner,'ADOPT',results['adopt'],flush=True)
    print(json.dumps(results['external'],indent=2),flush=True)

if __name__=='__main__': main()
