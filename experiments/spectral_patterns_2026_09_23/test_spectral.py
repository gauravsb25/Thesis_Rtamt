import numpy as np
from run import features, weights, fit_csp, csp_features, LDA

def test_frequency_and_temporal_order():
    t=np.arange(768)/256
    ramp=np.linspace(.2,2,768)
    x=np.tile((np.sin(2*np.pi*10*t)*ramp)[None,None,:],(2,8,1))
    x[1]=x[0,:,::-1]
    static,temporal,cov=features(x)
    assert np.argmax(static[0,:48].reshape(6,8).mean(1))==2
    assert temporal[0].reshape(3,6,8)[0,2,0]>0
    assert temporal[1].reshape(3,6,8)[0,2,0]<0
    scaled=features(x*10)
    for a,b in zip((static,temporal,cov),scaled):
        np.testing.assert_allclose(a,b,atol=.01,rtol=.01)
    assert np.isfinite(cov).all()

def test_recording_weighting():
    y=np.array([0,0,0,0,1,1]); trial=np.array(['a','a','a','b','c','c'])
    w=weights(y,trial)
    np.testing.assert_allclose([w[:3].sum(),w[3],w[4:].sum()],[.25,.25,.5])

def test_csp_and_lda_recover_covariance_signal():
    rng=np.random.default_rng(2); y=np.repeat([0,1],30)
    x=rng.normal(size=(60,8,768)); x[y==0,0]*=4; x[y==1,1]*=4
    _,_,cov=features(x); tr=np.r_[0:20,30:50]; te=np.r_[20:30,50:60]
    w=np.ones(len(tr))/len(tr)
    filters=fit_csp(cov[tr],y[tr],w)
    a=csp_features(cov,filters)
    model=LDA().fit(a[tr],y[tr],w)
    assert (model.predict(a[te])==y[te]).mean()>.9
