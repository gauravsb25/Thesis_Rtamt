"""New pair-wise temporal robustness signals, computed without labels or metadata."""
import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, lfilter

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments'/'nested_stl_2026_09_19'))
from run import past, sequence

CHANNELS=['Fp1','Fp2','F3','Fz','F4','T3','Cz','T4']

def temporal(raw):
    x=lfilter(*butter(2,[.3,6],btype='bandpass',fs=256),raw,axis=-1)[...,::8]
    x-=np.median(x,axis=-1,keepdims=True)
    values=[]; names=[]
    def add(name,v): names.append(name); values.append(v)
    for i in range(8):
        for j in range(i+1,8):
            p=x[:,i]-x[:,j]
            p-=np.median(p,axis=-1,keepdims=True)
            p/=1.4826*np.median(np.abs(p),axis=-1,keepdims=True)+1e-12
            pair=f'{CHANNELS[i]}_{CHANNELS[j]}'
            for d in (2,4,8):
                positive=past(p,0,d-1,True)[:,d-1:].max(axis=1)
                negative=past(-p,0,d-1,True)[:,d-1:].max(axis=1)
                add(f'sustained_{pair}_{d}',positive-negative)
            for d,g in ((2,16),(2,32),(4,16),(4,32)):
                add(f'ordered2_{pair}_{d}_{g}',sequence(p,d,g)-sequence(-p,d,g))
            for d in (2,4):
                add(f'ordered3_{pair}_{d}_32',sequence(p,d,32,True)-sequence(-p,d,32,True))
            a=x[:,i]; b=x[:,j]
            da=np.diff(a,axis=1); db=np.diff(b,axis=1)
            circulation=np.sum(a[:,:-1]*db-b[:,:-1]*da,axis=1)
            norm=np.sqrt(np.sum(a[:,:-1]**2+b[:,:-1]**2,axis=1)*np.sum(da**2+db**2,axis=1))+1e-12
            add(f'circulation_{pair}',circulation/norm)
    return np.stack(values,axis=1).astype(np.float32),np.array(names)

def build_temporal(test=False):
    cache=HERE/('temporal_test.npz' if test else 'temporal_dev.npz')
    if cache.exists(): return dict(np.load(cache))
    parts=[]; labels=[]; sessions=[]
    for path in sorted((ROOT/'extracted_npz_dataset_v2').glob('*/ses-*/clean_epochs.npz')):
        isdev=path.parent.parent.name=='sub_01' and int(path.parent.name[-2:])<=30
        if isdev==test: continue
        z=np.load(path,allow_pickle=True)
        vals,names=temporal(z['epochs'][:,:8]*1e6)
        parts.append(vals); labels.append(z['labels'].astype(str))
        sessions.append(np.repeat(f'{path.parent.parent.name}/{path.parent.name}',len(vals)))
        print('temporal',sessions[-1][0],len(vals),flush=True)
    out=dict(X=np.concatenate(parts),names=names,label=np.concatenate(labels),session=np.concatenate(sessions))
    np.savez_compressed(cache,**out)
    return out

def load(test=False):
    m0=dict(np.load(ROOT/'measurements_v2.npz',allow_pickle=True))
    dev=np.array([s.startswith('sub_01') and int(s[-2:])<=30 for s in m0['session']])
    use=~dev if test else dev
    m={k:v[use] for k,v in m0.items()}
    raw=np.load(ROOT/'experiments'/'ml_raw_2026_09_19'/'features_raw.npz',allow_pickle=True)
    assert np.array_equal(raw['session'],m0['session']) and np.array_equal(raw['label'],m0['label'])
    free=raw['scale_free'].astype(bool)
    arrays=[raw['X'][use][:,free]]; names=list(raw['names'][free].astype(str))
    for dirname in ('proposed_methods_2026_09_18','phase_shape_2026_09_18','fifteen_methods_2026_09_18','five_paradigms_2026_09_18'):
        z=np.load(ROOT/'experiments'/dirname/'features.npz',allow_pickle=True)
        assert np.array_equal(z['session'],m0['session'])
        if 'label' in z.files: assert np.array_equal(z['label'],m0['label'])
        else:
            for s in np.unique(m0['session']):
                ix=m0['session']==s
                assert np.array_equal(z['index'][ix],np.arange(ix.sum()))
        keys=[k for k in z.files if k not in ('session','label','task','run','onset','index')]
        arrays.append(np.stack([z[k][use] for k in keys],axis=1)); names.extend([dirname.split('_2026')[0]+'_'+k.replace('.','p') for k in keys])
    keys=[k for k in m if k not in ('session','label')]
    arrays.append(np.stack([m[k] for k in keys],axis=1)); names.extend(['monitor_'+k for k in keys])
    t=build_temporal(test)
    assert np.array_equal(t['session'],m['session']) and np.array_equal(t['label'],m['label'])
    arrays.append(t['X']); names.extend(t['names'])
    X=np.concatenate(arrays,axis=1).astype(np.float32)
    assert np.isfinite(X).all()
    assert len(names)==len(set(names))
    return X,np.array([str(name).replace('.', 'p') for name in names]),m

if __name__=='__main__':
    x,n,m=load(); print('development features',x.shape,flush=True)
