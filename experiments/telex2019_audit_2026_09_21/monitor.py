"""Experimental raw-window STL monitor. No changes to production monitoring/.

All future formulas are evaluated offline at time 0 on a completed 3-second window.
Input units: microvolts, shape (8,768). Optional history uses the original monitor convention.
"""
import json,sys
from pathlib import Path
import numpy as np
from data import HERE,ROOT,traces,NAMES
from logic import evaluate
sys.path.insert(0,str(ROOT))
from monitoring.rtamt_monitors import HeadMovementMonitor
from monitoring.rtamt_signals import RtamtConfig
CLASSES=['LEFT','RIGHT','DOWN','STRAIGHT']

class LearnedSTLMonitor:
    def __init__(self,mode='supervised',*,challenger=False):
        if mode not in ('positive','supervised'): raise ValueError(mode)
        suffix='challenger_monitor' if challenger else 'monitor'
        self.manifest=json.loads((HERE/f'{mode}_{suffix}.json').read_text())
        import project_code
        if project_code.fingerprint()!=self.manifest['fingerprint']:
            raise ValueError('Production monitor or dataset changed since the experimental model was frozen')
        self.original=HeadMovementMonitor(); self.config=RtamtConfig()

    def trace_scores(self,xx):
        cfg=self.manifest['config']
        if cfg is None: return None
        depth=cfg[1]-1
        outputs=[]
        for model in self.manifest['models']:
            outputs.append(np.column_stack([evaluate(model['trees'][c][depth],xx)[:,0]/model['scales'][c][depth] for c in range(4)]))
        return np.mean(outputs,axis=0)

    def explain(self,window,history=None):
        baseline=self.original.classify(window,history)
        if baseline=='INVALID' or self.manifest['config'] is None:
            return dict(label=baseline,original_label=baseline,mode='original fallback')
        score=self.trace_scores(traces(window))[0]; gate=self.manifest['config'][2]
        verdict=baseline if gate and baseline=='STRAIGHT' else CLASSES[int(np.argmax(score[:3] if gate else score))]
        return dict(label=verdict,original_label=baseline,normalized_robustness=dict(zip(CLASSES,score.tolist())),
            decoder='average normalized robustness across two fixed seeds, then argmax',experimental=True)

    def classify(self,window,history=None): return self.explain(window,history)['label']
    classify_window=classify
