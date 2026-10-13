"""Replay the frozen experimental classifier, without training or selection.

This is a cached-feature research interface, not a live EEG monitor. The raw
feature extraction/integration latency has not been validated for deployment.
"""
import pickle
import json
import numpy as np
from benchmark import HERE, Candidate, load, CL

class ModelReader(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='__main__' and name=='Candidate': return Candidate
        return super().find_class(module,name)

def load_candidate():
    with (HERE/'candidate.pkl').open('rb') as f: return ModelReader(f).load()

def main():
    result=json.loads((HERE/'results.json').read_text())
    x,y,base,session,trial,dev,n=load()
    pred=base if result['selected']=='baseline' else load_candidate().predict(x,base)
    saved=np.load(HERE/'predictions.npz')['final']
    np.testing.assert_array_equal(pred,saved)
    print(f'Frozen candidate {result["selected"]}: all {len(pred):,} saved predictions reproduced exactly.')
    print('Research classifier only; production monitor has not been replaced.')

if __name__=='__main__': main()
