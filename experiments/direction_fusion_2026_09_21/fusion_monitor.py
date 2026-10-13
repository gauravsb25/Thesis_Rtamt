"""Experimental fusion of an existing RTAMT verdict with measured feature specialists.

These combinations failed external adoption criteria. Not a production replacement.
Inputs are completed-window measurements, not raw EEG or prefix-causal signals.
"""
import json
from pathlib import Path
import numpy as np

CLASSES=('LEFT','RIGHT','DOWN','STRAIGHT')

class DirectionFusionMonitor:
    def __init__(self, manifest='selected_pipeline_monitor.json'):
        path=Path(manifest)
        if not path.is_absolute(): path=Path(__file__).resolve().parent/path
        self.manifest=json.loads(path.read_text())
        self.family=self.manifest['family']
        self.models={}
        if self.family!='baseline':
            for kind in ('nod','turn','three'):
                with np.load(path.parent/f'{self.family}_{kind}.npz') as data:
                    self.models[kind]=dict(data)

    def predict_many(self, features, baseline):
        baseline=np.asarray(baseline,dtype=int)
        if baseline.ndim!=1 or not np.isin(baseline,[0,1,2,3]).all():
            raise ValueError('Expected original labels encoded LEFT=0 RIGHT=1 DOWN=2 STRAIGHT=3')
        if self.family=='baseline': return baseline.copy()
        x=np.asarray(features,dtype=float)
        if x.shape!=(len(baseline),len(self.manifest['feature_names'])) or not np.isfinite(x).all():
            raise ValueError('Missing, nonfinite, or incorrectly ordered feature matrix')
        scores={k:np.clip((x[:,m['cols']]-m['mu'])/m['scale'],-10,10)@m['coef']+m['intercept'] for k,m in self.models.items()}
        n,t,z=(scores[k] for k in ('nod','turn','three'))
        nm=np.abs(n[:,1]-n[:,0]); tm=np.abs(t[:,1]-t[:,0]); zz=np.sort(z,axis=1); zm=zz[:,-1]-zz[:,-2]
        cuts=self.manifest['cuts']; nh=n.argmax(1)==1; th=t.argmax(1); zh=z.argmax(1)
        base_turn=(x[:,self.manifest['feature_names'].index('turn')]<=0).astype(int)
        nod=np.where(nm>=cuts[0],nh,baseline==2)
        turn=np.where(tm>=cuts[1],th,base_turn)
        if self.family=='nod_specialist': p=np.where(nod,2,base_turn)
        elif self.family=='turn_specialist': p=np.where(baseline==2,2,turn)
        elif self.family=='nested_specialists': p=np.where(nod,2,turn)
        elif self.family=='agreement':
            binary=np.where(nh,2,th)
            confident=(nm>=cuts[0]) & (nh | (tm>=cuts[1])) & (zm>=cuts[2])
            p=np.where(confident & (binary==zh),zh,baseline)
        elif self.family=='prior_fusion':
            fused=z.copy(); rows=np.flatnonzero(baseline!=3)
            fused[rows,baseline[rows]]+=self.manifest['config'][1]; p=fused.argmax(1)
        else: raise ValueError(self.family)
        return np.where(baseline==3,3,p)

    def classify_features(self, features, current_label):
        if current_label=='INVALID': return 'INVALID'
        baseline=CLASSES.index(current_label)
        if self.family=='baseline' or current_label=='STRAIGHT': return current_label
        x=np.array([[features[name] for name in self.manifest['feature_names']]])
        return CLASSES[int(self.predict_many(x,[baseline])[0])]
