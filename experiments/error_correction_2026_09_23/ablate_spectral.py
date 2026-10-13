"""Post-hoc diagnostic only: are spectra needed for the direct classifier?

Never used to alter nested choices or choose an external-test candidate.
"""
import json
import numpy as np
from benchmark import HERE,load,metrics,classifier,weights,threadpool_limits

def main():
    x,y,base,session,trial,dev,n=load()
    # combined = monitor(18), static spectra(96), ordered spectra(144), old shape.
    z=np.column_stack([x['combined'][:,:18],x['combined'][:,258:]])
    pred=base.copy()
    for fold in range(3):
        test=dev&((n-1)//10==fold); train=dev&~test
        model=classifier().fit(z[train],y[train],sample_weight=weights(y[train],trial[train]))
        pred[test]=model.predict(z[test])
    result={'status':'Post-hoc development ablation; not part of model selection',
        'features':z.shape[1],'no_spectral_direct':metrics(y[dev],pred[dev])}
    (HERE/'spectral_ablation.json').write_text(json.dumps(result,indent=2))
    np.save(HERE/'no_spectral_oof.npy',pred)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2): main()
