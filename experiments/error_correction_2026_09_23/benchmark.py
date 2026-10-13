"""Nested session validation of targeted error correction. Writes here only."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import sys, json, pickle, time, hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'vendor'))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

CL=['LEFT','RIGHT','DOWN','STRAIGHT']
CONFIGS=[f'{kind}:{features}' for kind in ['direct','staged','correct'] for features in ['monitor','combined']]

def metrics(y,p):
    cm=np.array([[np.sum((y==a)&(p==b)) for b in range(4)] for a in range(4)])
    totals=cm.sum(1); r=np.divide(cm.diagonal(),totals,out=np.zeros(4),where=totals>0)
    return dict(n=len(y),correct=int((y==p).sum()),accuracy=100*float((y==p).mean()),
        ba=100*float(r[totals>0].mean()),movement=100*float(r[:3].mean()),recall=(r*100).tolist(),confusion=cm.tolist())

def weights(y,trial):
    w=np.zeros(len(y))
    for k in np.unique(y):
        gs=np.unique(trial[y==k])
        for g in gs:
            mask=(y==k)&(trial==g); w[mask]=1/(len(gs)*mask.sum())
    return w/w.mean()

def classifier():
    # Disable default random-window early stopping: that would leak overlaps.
    return HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=7,min_samples_leaf=35,
        l2_regularization=10,learning_rate=.06,early_stopping=False,random_state=23,max_bins=127)

def load():
    s=dict(np.load(ROOT/'experiments/spectral_patterns_2026_09_23/features.npz'))
    m=dict(np.load(ROOT/'measurements_v2.npz')); b=dict(np.load(ROOT/'monitor_labels_v2.npz'))
    assert np.array_equal(m['session'],s['session']) and np.array_equal(m['label'],s['label'])
    assert np.array_equal(b['session'],s['session']) and np.array_equal(b['label'],s['label'])
    names=['contrast','background','background_valid','delta','deflect','nod','sideways','turn']
    monitor=np.column_stack([m[k] for k in names])
    base=np.array([CL.index(x) for x in b['with_history']])
    # Exact fixed-rule margins expose which tests were close to failing.
    margins=np.column_stack([np.where(m['background_valid']>=.5,m['background']/6.61,m['contrast']/7.81)-1,
        m['delta']/.208-1,m['deflect']/2.72-1,m['nod']+.661,1-m['sideways']/7.54,m['turn']])
    monitor=np.column_stack([monitor,margins,np.eye(4)[base]])
    combined=[monitor,s['static'],s['temporal']]
    sources=['proposed_methods_2026_09_18','phase_shape_2026_09_18','fifteen_methods_2026_09_18','five_paradigms_2026_09_18']
    for source in sources:
        d=dict(np.load(ROOT/'experiments'/source/'features.npz'))
        assert np.array_equal(d['session'],s['session'])
        if 'label' in d:
            assert np.array_equal(d['label'],s['label'])
        else:
            for session in np.unique(s['session']):
                mask=s['session']==session
                assert np.array_equal(d['index'][mask],np.arange(mask.sum()))
        columns=[v for k,v in sorted(d.items()) if k not in ['index','run','task','offset','onset','dev','valid'] and v.ndim==1 and v.dtype.kind in 'fiu']
        combined.append(np.column_stack(columns))
    x={'monitor':monitor.astype('float32'),'combined':np.column_stack(combined).astype('float32')}
    for v in x.values(): assert np.isfinite(v).all()
    y=np.array([CL.index(v) for v in s['label']]); ses=s['session']; n=np.array([int(v[-2:]) for v in ses])
    dev=np.char.startswith(ses,'sub_01')&(n<=30)
    return x,y,base,ses,s['trial'],dev,n

class Candidate:
    def __init__(self,config): self.kind,self.features=config.split(':')
    def fit(self,x,y,base,trial):
        z=x[self.features]; self.models=[]
        if self.kind=='staged':
            for mask,target in [(np.ones(len(y),bool),(y!=3).astype(int)),(y!=3,(y==2).astype(int)),(y<2,y)]:
                model=classifier().fit(z[mask],target[mask],sample_weight=weights(target[mask],trial[mask]))
                self.models.append(model)
        else:
            # A fifth class explicitly means retain the current monitor answer.
            target=np.where(y==base,4,y) if self.kind=='correct' else y
            self.models=[classifier().fit(z,target,sample_weight=weights(y,trial))]
        return self
    def predict(self,x,base):
        z=x[self.features]
        if self.kind=='staged':
            move,nod,direction=[m.predict(z) for m in self.models]
            return np.where(move==0,3,np.where(nod==1,2,direction))
        if self.kind=='correct':
            model=self.models[0]; probabilities=model.predict_proba(z)
            p=np.zeros((len(z),5)); p[:,model.classes_]=probabilities
            p[np.arange(len(z)),base]+=p[:,4]
            return p[:,:4].argmax(1)
        return self.models[0].predict(z)

def subset(x,mask): return {k:v[mask] for k,v in x.items()}

def choose(x,y,base,ses,trial,training,n,prefix):
    # Two inner folds comprising alternating 5-session blocks. The outer fold
    # and all external sessions are excluded from fitting AND model selection.
    preds={k:base.copy() for k in CONFIGS}
    for fold in range(2):
        valid=training&(((n-1)//5)%2==fold); fit=training&~valid
        assert not set(trial[fit])&set(trial[valid])
        for config in CONFIGS:
            stamp=time.monotonic()
            model=Candidate(config).fit(subset(x,fit),y[fit],base[fit],trial[fit])
            preds[config][valid]=model.predict(subset(x,valid),base[valid])
            print(prefix,'inner',fold,config,'seconds',round(time.monotonic()-stamp,1),flush=True)
    scores={k:metrics(y[training],v[training]) for k,v in preds.items()}
    scores['baseline']=metrics(y[training],base[training])
    # Baseline participates in selection. A candidate must not sacrifice L/R/D.
    eligible=[k for k,v in scores.items() if v['movement']>=scores['baseline']['movement']]
    selected=max(eligible,key=lambda k:scores[k]['ba'])
    (HERE/f'{prefix}_selection.json').write_text(json.dumps(dict(selected=selected,scores=scores),indent=2))
    return selected

def main():
    started=time.monotonic(); x,y,base,ses,trial,dev,n=load()
    print('features',{k:v.shape for k,v in x.items()},flush=True)
    result={'baseline_development':metrics(y[dev],base[dev]),'folds':[]}
    oof=base.copy(); ablation={k:base.copy() for k in CONFIGS}
    for fold in range(3):
        valid=dev&((n-1)//10==fold); train=dev&~valid
        selected=choose(x,y,base,ses,trial,train,n,f'outer{fold}')
        for config in CONFIGS:
            model=Candidate(config).fit(subset(x,train),y[train],base[train],trial[train])
            ablation[config][valid]=model.predict(subset(x,valid),base[valid])
        if selected!='baseline': oof[valid]=ablation[selected][valid]
        row=dict(fold=fold,selected=selected,baseline=metrics(y[valid],base[valid]),candidate=metrics(y[valid],oof[valid]))
        result['folds'].append(row)
        print('OUTER',json.dumps(row),flush=True)
        (HERE/'progress.json').write_text(json.dumps(result,indent=2))
    result['nested_validation']=metrics(y[dev],oof[dev])
    result['exploratory_ablation']={k:metrics(y[dev],v[dev]) for k,v in ablation.items()}
    selected=choose(x,y,base,ses,trial,dev,n,'final')
    result['selected']=selected
    result['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (HERE/'pretest.json').write_text(json.dumps(result,indent=2))
    final=base.copy()
    if selected!='baseline':
        model=Candidate(selected).fit(subset(x,dev),y[dev],base[dev],trial[dev])
        final=model.predict(x,base)
        with (HERE/'candidate.pkl').open('wb') as f: pickle.dump(model,f)
    result['training_fit']=metrics(y[dev],final[dev])
    result['external']={}
    for name,mask in [('sub01_later',np.char.startswith(ses,'sub_01')&~dev),('sub11',np.char.startswith(ses,'sub_11'))]:
        result['external'][name]=dict(baseline=metrics(y[mask],base[mask]),candidate=metrics(y[mask],final[mask]))
    result['elapsed_seconds']=time.monotonic()-started
    result['achieved_90']=result['nested_validation']['ba']>=90 and all(v['candidate']['ba']>=90 for v in result['external'].values())
    result['adopt']=result['nested_validation']['ba']>result['baseline_development']['ba'] and all(v['candidate']['ba']>=v['baseline']['ba'] and v['candidate']['movement']>=v['baseline']['movement'] for v in result['external'].values())
    np.savez_compressed(HERE/'predictions.npz',y=y,session=ses,trial=trial,dev=dev,baseline=base,nested=oof,final=final,**ablation)
    (HERE/'results.json').write_text(json.dumps(result,indent=2))
    print('FINAL',json.dumps(result),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2): main()
