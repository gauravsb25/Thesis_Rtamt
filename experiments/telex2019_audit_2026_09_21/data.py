import sys, json, hashlib, itertools
from pathlib import Path
import numpy as np
from scipy.signal import butter, lfilter
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.dont_write_bytecode=True; sys.path.insert(0,str(ROOT))
CHANNELS=['Fp1','Fp2','F3','Fz','F4','T3','Cz','T4']
PAIRS=list(itertools.combinations(range(8),2))
NAMES=['ch_'+c for c in CHANNELS]+['pair_'+CHANNELS[i]+'_'+CHANNELS[j] for i,j in PAIRS]+['energy_'+c for c in CHANNELS]
FILTER=butter(2,[.3,6],btype='bandpass',fs=256)

def traces(raw):
    raw=np.asarray(raw,float)
    if raw.ndim==2: raw=raw[None]
    if raw.shape[1:]!=(8,768) or not np.isfinite(raw).all(): raise ValueError('Expected finite (N,8,768) microvolt windows')
    f=lfilter(*FILTER,raw,axis=-1)[...,::8]
    def norm(a):
        a=a-np.median(a,axis=-1,keepdims=True)
        return a/(1.4826*np.median(np.abs(a),axis=-1,keepdims=True)+1e-9)
    channels=norm(f)
    pairs=norm(np.stack([f[:,i]-f[:,j] for i,j in PAIRS],axis=1))
    energy=lfilter(np.ones(8)/8,[1.],f*f,axis=-1)
    energy=np.log1p(energy/(np.median(energy,axis=-1,keepdims=True)+1e-9))
    return np.concatenate([channels,pairs,energy],axis=1).astype(np.float32)

def prepare():
    import project_code
    with np.load(ROOT/'monitor_labels_v2.npz') as z: labels=dict(z)
    assert str(labels['fingerprint'])==project_code.fingerprint(),'Original cache stale; stop'
    manifest=HERE/'data_manifest.json'
    if manifest.exists():
        m=json.loads(manifest.read_text()); assert m['fingerprint']==project_code.fingerprint()
        assert m['preprocessing_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return
    arr=np.lib.format.open_memmap(HERE/'traces.npy',mode='w+',dtype='float32',shape=(len(labels['label']),44,96))
    groups=[]; offset=0; source=[]
    for path in sorted((ROOT/'extracted_npz_dataset_v2').glob('*/ses-*/clean_epochs.npz')):
        with np.load(path,allow_pickle=True) as d:
            s=f'{path.parent.parent.name}/{path.parent.name}'; lab=d['labels'].astype(str); n=len(lab)
            assert np.array_equal(labels['label'][offset:offset+n],lab)
            assert np.all(labels['session'][offset:offset+n]==s)
            raw=d['epochs'][:,:8]*1e6
            for start in range(0,n,128): arr[offset+start:offset+min(start+128,n)]=traces(raw[start:start+128])
            groups.extend([f'{s}/{task}/{run}' for task,run in zip(d['task'].astype(str),d['run'])])
            source.append({'path':str(path.relative_to(ROOT)),'size':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns})
            offset+=n; print('prepared',s,n,flush=True)
    assert offset==len(arr); arr.flush()
    np.savez_compressed(HERE/'metadata.npz',label=labels['label'],session=labels['session'],baseline=labels['with_history'],group=np.array(groups))
    manifest.write_text(json.dumps(dict(fingerprint=project_code.fingerprint(),names=NAMES,sources=source,preprocessing_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2),encoding='utf-8')

if __name__=='__main__': prepare()
