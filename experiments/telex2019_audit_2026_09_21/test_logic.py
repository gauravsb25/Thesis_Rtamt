import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE.parents[1]))
from logic import peak,evaluate,formula,horizon,learn
from monitoring.rtamt_monitors import import_rtamt

def test_peak_sign_and_boundary():
    r=np.array([-10.,-1.,-.01,0.,.01,1.,10.])
    assert np.array_equal(np.sign(peak(r)),np.sign(r))
    assert peak(.1)>peak(10.)

def test_against_rtamt():
    rtamt=import_rtamt(); rng=np.random.default_rng(44)
    a=dict(op='atom',signal=0,sign=1,threshold=.5)
    b=dict(op='atom',signal=1,sign=-1,threshold=.2)
    f=dict(op='F',a=2,b=7,child=a); g=dict(op='G',a=0,b=3,child=b)
    forms=[a,b,f,g,dict(op='not',child=f),dict(op='and',left=f,right=g),dict(op='or',left=f,right=g),
        dict(op='G',a=0,b=5,child=dict(op='aug',sign=-1,child=f)),dict(op='F',a=0,b=5,child=g)]
    forms += [dict(op='U',a=lo,b=hi,left=l,right=r) for lo,hi in ((0,0),(0,5),(2,7)) for l,r in ((a,b),(f,g))]
    for x in [rng.normal(size=(2,2,32)),np.zeros((2,2,32)),np.ones((2,2,32))*.5]:
        for tree in forms:
            ours=evaluate(tree,x)
            assert ours.shape[1]==32-horizon(tree)
            for i in range(len(x)):
                spec=rtamt.StlDiscreteTimeOfflineSpecification()
                for name in ('x','y'): spec.declare_var(name,'float')
                spec.spec=formula(tree,['x','y']); spec.parse()
                theirs=np.array(spec.evaluate(dict(time=list(range(32)),x=x[i,0].tolist(),y=x[i,1].tolist())))[:,1]
                np.testing.assert_allclose(ours[i],theirs[:ours.shape[1]],atol=1e-10)

def test_horizon_rejects_incomplete():
    import pytest
    with pytest.raises(ValueError): evaluate(dict(op='F',a=0,b=32,child=dict(op='atom',signal=0,sign=1,threshold=0)),np.zeros((1,1,32)))
