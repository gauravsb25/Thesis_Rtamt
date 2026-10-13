"""Record the exact cached feature inputs; no fitting and no cache changes."""
import hashlib,json
import numpy as np
from benchmark import HERE,ROOT,load

def main():
    x,y,base,session,trial,dev,n=load()
    files=[ROOT/'measurements_v2.npz',ROOT/'monitor_labels_v2.npz',
        ROOT/'experiments/spectral_patterns_2026_09_23/features.npz']
    files += [ROOT/'experiments'/s/'features.npz' for s in
        ['proposed_methods_2026_09_18','phase_shape_2026_09_18','fifteen_methods_2026_09_18','five_paradigms_2026_09_18']]
    result={'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'feature_matrices':{k:{'shape':v.shape,'sha256':hashlib.sha256(v.tobytes()).hexdigest()} for k,v in x.items()},
        'folds':[]}
    for fold in range(3):
        test=dev&((n-1)//10==fold); train=dev&~test
        assert not set(trial[test])&set(trial[train])
        assert not set(session[test])&set(session[train])
        result['folds'].append({'outer':fold,'train_sessions':np.unique(session[train]).tolist(),
            'test_sessions':np.unique(session[test]).tolist(),'train_recordings':len(np.unique(trial[train])),
            'test_recordings':len(np.unique(trial[test]))})
    (HERE/'input_audit.json').write_text(json.dumps(result,indent=2))
    print('Feature files hashed; every outer fold has disjoint recordings and sessions.')

if __name__=='__main__': main()
