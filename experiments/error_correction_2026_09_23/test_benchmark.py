import numpy as np
from benchmark import Candidate, weights, metrics

def test_recordings_not_windows_determine_weight():
    y=np.array([0,0,0,0,1,1]); trial=np.array(['a','a','a','b','c','c'])
    w=weights(y,trial); w/=w.sum()
    np.testing.assert_allclose([w[:3].sum(),w[3],w[4:].sum()],[.25,.25,.5])

def test_keep_probability_is_assigned_to_original_decision():
    class Dummy:
        classes_=np.array([0,1,2,3,4])
        def predict_proba(self,x):
            return np.array([[.1,.1,.3,.1,.4],[.1,.6,.1,.1,.1]])
    model=Candidate('correct:monitor'); model.models=[Dummy()]
    p=model.predict({'monitor':np.zeros((2,1))},np.array([0,0]))
    np.testing.assert_array_equal(p,[0,1])

def test_movement_metric_penalizes_missed_events():
    y=np.arange(4); p=np.array([3,1,2,3]); m=metrics(y,p)
    assert m['ba']==75
    np.testing.assert_allclose(m['movement'],200/3)

def test_models_learn_known_nonlinear_problem_without_test_fitting():
    rng=np.random.default_rng(12); x=rng.normal(size=(1200,2))
    y=(x[:,0]>0).astype(int)+2*(x[:,1]>0).astype(int)
    base=np.zeros(1200,dtype=int); trial=np.arange(1200).astype(str)
    for kind in ['direct','staged','correct']:
        model=Candidate(kind+':monitor').fit({'monitor':x[:900]},y[:900],base[:900],trial[:900])
        pred=model.predict({'monitor':x[900:]},base[900:])
        assert (pred==y[900:]).mean()>.9
