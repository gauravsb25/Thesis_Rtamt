import json,sys
from pathlib import Path
import numpy as np
from data import HERE,ROOT,NAMES,traces
from logic import evaluate,formula,horizon
from monitor import LearnedSTLMonitor
sys.dont_write_bytecode=True; sys.path.insert(0,str(ROOT))
from monitoring.rtamt_monitors import import_rtamt
from monitoring.stream_history import windows_with_history
rtamt=import_rtamt(); X=np.load(HERE/'traces.npy',mmap_mode='r'); md=dict(np.load(HERE/'metadata.npz'))
saved=np.load(HERE/'challenger_predictions.npz'); checks=0; rawchecks=0
indices=np.array([np.flatnonzero(md['label']==c)[0] for c in ['LEFT','RIGHT','DOWN','STRAIGHT']])
summary={}
for mode in ('positive','supervised'):
    mon=LearnedSTLMonitor(mode,challenger=True); cfg=mon.manifest['config']
    if cfg is None:
        summary[mode]={'selected':'baseline'}; continue
    for model in mon.manifest['models']:
        assert len(set(model['trial_keys']))==len(model['trial_keys'])
        assert len(set(np.bincount(model['labels'],minlength=4)))==1
        for tt in model['trees']:
            tree=tt[cfg[1]-1]
            for idx in indices:
                spec=rtamt.StlDiscreteTimeOfflineSpecification()
                for name in NAMES: spec.declare_var(name,'float')
                spec.spec=formula(tree,NAMES); spec.parse()
                data={'time':list(range(96)),**{name:X[idx,j].astype(float).tolist() for j,name in enumerate(NAMES)}}
                got=np.array(spec.evaluate(data))[:,1]
                want=evaluate(tree,X[idx:idx+1])[0]
                np.testing.assert_allclose(got[:len(want)],want,atol=2e-5,rtol=2e-5); checks+=1
    # Validate exported decoder over every saved window, with bounded memory.
    encoded=np.array([['LEFT','RIGHT','DOWN','STRAIGHT'].index(x) for x in md['baseline']])
    decoded=[]
    for start in range(0,len(X),256):
        sc=mon.trace_scores(np.asarray(X[start:start+256],float))
        pred=np.where(encoded[start:start+len(sc)]==3,3,sc[:,:3].argmax(1)) if cfg[2] else sc.argmax(1)
        decoded.extend(pred.tolist())
    assert np.array_equal(decoded,saved[mode+'_final'])
    # Raw-window adapter with actual reconstructed history, one example per available class.
    path=ROOT/'extracted_npz_dataset_v2/sub_01/ses-01/clean_epochs.npz'
    with np.load(path,allow_pickle=True) as z: lab=z['labels'].astype(str)
    targets={int(np.flatnonzero(lab==c)[0]) for c in np.unique(lab)}
    start=int(np.flatnonzero(md['session']=='sub_01/ses-01')[0])
    for i,window,history in windows_with_history(path):
        if i not in targets: continue
        np.testing.assert_allclose(traces(window)[0],X[start+i],atol=2e-5,rtol=2e-5)
        label=mon.classify(window,history)
        assert label==['LEFT','RIGHT','DOWN','STRAIGHT'][saved[mode+'_final'][start+i]]
        rawchecks+=1; targets.remove(i)
        if not targets: break
    summary[mode]={'decoder_rows':len(decoded),'formulas':8}
(HERE/'verification.json').write_text(json.dumps(dict(rtamt_real_trace_checks=checks,raw_window_checks=rawchecks,modes=summary),indent=2),encoding='utf-8')
print('VERIFIED',checks,'formula/real-trace RTAMT comparisons;',rawchecks,'raw-window checks;',summary,flush=True)
