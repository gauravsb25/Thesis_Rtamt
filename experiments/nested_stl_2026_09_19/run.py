"""Finite, preregistered nested-STL search. No held-out extraction unless CV passes.
Run from any directory: python run.py. Results and reproducible predictions saved here.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, lfilter
from scipy.ndimage import minimum_filter1d, maximum_filter1d

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CL = np.array(['LEFT', 'RIGHT', 'DOWN', 'STRAIGHT'])

def past(x, a, b, minimum=False):
    """Inclusive bounded past extrema; missing history has neutral robustness."""
    neutral = np.inf if minimum else -np.inf
    f = minimum_filter1d if minimum else maximum_filter1d
    size = b-a+1
    y = f(x, size=size, axis=-1, origin=(size-1)//2, mode='constant', cval=neutral)
    if a:
        y = np.concatenate([np.full((*y.shape[:-1], a), neutral), y[..., :-a]], axis=-1)
    return y

def sequence(x, duration, gap, triple=False):
    # Negative sustained lobe, then positive sustained lobe (then negative).
    p = past(x, 0, duration-1, True)
    n = past(-x, 0, duration-1, True)
    r = np.minimum(p, past(n, duration, gap))
    if triple:
        r = np.minimum(n, past(r, duration, gap))
    # Require complete history, including all inner operators.
    horizon = (2 if triple else 1)*gap + duration-1
    return r[..., horizon:].max(axis=-1)

def extract(paths):
    rows=[]
    for path in paths:
        z=np.load(path, allow_pickle=True)
        raw=z['epochs'][:, :8]*1e6
        x=lfilter(*butter(2, [0.3, 6], btype='bandpass', fs=256), raw, axis=-1)[..., ::8]
        x-=np.median(x, axis=-1, keepdims=True)
        scale=1.4826*np.median(np.abs(x), axis=-1, keepdims=True)+1e-9
        u=x/scale
        # Fourth-largest channel amplitude: at least four electrodes support a burst.
        support=np.sort(np.abs(u),axis=1)[:,4,:]
        # Robust channel pooling reduces dependence on a single huge temple channel.
        mid=np.mean(u[:,[3,6]]**2,axis=1)
        lateral=np.median(u[:,[2,4,5,7]]**2,axis=1)
        sync=np.log((mid+0.05)/(lateral+0.05))
        pair=x[:,4]-x[:,5]
        pair/=1.4826*np.median(np.abs(pair),axis=-1,keepdims=True)+1e-9
        f={}
        for d in (2,4,8):
            f[f'burst_{d}']=past(support,0,d-1,True)[:,d-1:].max(axis=1)
            # A quiet period followed within 0.25--1s by sustained multi-channel activity.
            quiet=past(1.5-support,0,3,True)
            active=past(support,0,d-1,True)
            prior=past(quiet,8,32)
            f[f'quietburst_{d}']=np.where(prior[:,35:]>=0,active[:,35:],-1e6).max(axis=1)
            f[f'nodsync_{d}']=past(sync,0,d-1,True)[:,d-1:].max(axis=1)
            for gap in (16,32):
                for triple in (False,True):
                    key=f'{"triple" if triple else "double"}_{d}_{gap}'
                    f[key]=sequence(pair,d,gap,triple)-sequence(-pair,d,gap,triple)
        f['session']=np.repeat(f'{path.parent.parent.name}/{path.parent.name}',len(raw))
        f['label']=z['labels'].astype(str)
        rows.append(f)
        print('extracted',f['session'][0],len(raw),flush=True)
    return {k:np.concatenate([r[k] for r in rows]) for k in rows[0]}

def score(y,p):
    return float(np.mean([np.mean(p[y==c]==c) for c in CL if np.any(y==c)])*100)

def baseline(m):
    move=np.where(m['background_valid']>=.5,m['background']>=6.61,m['contrast']>=7.81)&(m['delta']>=.208)&(m['deflect']>=2.72)
    nod=(m['nod']>=-.661)&(m['sideways']<=7.54)
    side=np.where(m['turn']>0,'LEFT','RIGHT')
    return move,nod,side,np.where(move,np.where(nod,'DOWN',side),'STRAIGHT')

def candidates(f,m):
    move,nod,side,base=baseline(m)
    out={'baseline':base}; families={'baseline':'baseline'}
    for key,v in f.items():
        if key in ('session','label'): continue
        if key.startswith(('double','triple')):
            for sign in (-1,1):
                for confidence in (0.,1.,2.):
                    s=np.where(np.abs(v)>confidence,np.where(sign*v>0,'LEFT','RIGHT'),side)
                    name=f'{key}:sign={sign}:margin={confidence}'
                    out[name]=np.where(move,np.where(nod,'DOWN',s),'STRAIGHT'); families[name]=key.split('_')[0]
        elif key.startswith('nodsync'):
            for t in (-1.,0.,1.,2.):
                for mode in ('replace','veto'):
                    n=(v>=t)&(nod if mode=='veto' else True)
                    name=f'{key}:{mode}:{t}'
                    out[name]=np.where(move,np.where(n,'DOWN',side),'STRAIGHT'); families[name]='nodsync'
        else:
            for t in (2.,3.,4.,6.):
                for mode in ('veto','rescue'):
                    # Rescue only if the existing energy gate is already satisfied.
                    loud=np.where(m['background_valid']>=.5,m['background']>=6.61,m['contrast']>=7.81)
                    g=move&(v>=t) if mode=='veto' else move|(loud&(v>=t))
                    name=f'{key}:{mode}:{t}'
                    out[name]=np.where(g,np.where(nod,'DOWN',side),'STRAIGHT'); families[name]=key.split('_')[0]
    return out,families

def verify():
    from monitoring.rtamt_monitors import import_rtamt
    rtamt=import_rtamt(); rng=np.random.default_rng(9)
    checks=0
    for d,g,triple in ((2,16,False),(4,16,True),(8,32,True)):
        for x in (rng.normal(size=96),np.sin(np.arange(96)/7)):
            neg=f'historically[0:{d-1}](-x >= 0)'
            pos=f'historically[0:{d-1}](x >= 0)'
            inner=f'({pos}) and once[{d}:{g}]({neg})'
            if triple: inner=f'({neg}) and once[{d}:{g}]({inner})'
            h=(2 if triple else 1)*g+d-1
            spec=rtamt.StlDiscreteTimeOfflineSpecification(); spec.declare_var('x','float')
            spec.spec=f'once[0:{95-h}]({inner})'; spec.parse()
            actual=spec.evaluate({'time':list(range(96)),'x':x.tolist()})[-1][1]
            np.testing.assert_allclose(actual,sequence(x[None],d,g,triple)[0],atol=1e-10)
            checks+=1
    print('RTAMT nested robustness checks:',checks,flush=True)
    return checks

def main():
    checks=verify()
    m0=dict(np.load(ROOT/'measurements_v2.npz',allow_pickle=True))
    dev=np.array([s.startswith('sub_01') and int(s[-2:])<=30 for s in m0['session']])
    m={k:v[dev] for k,v in m0.items()}
    paths=sorted((ROOT/'extracted_npz_dataset_v2'/'sub_01').glob('ses-*/clean_epochs.npz'))
    cache=HERE/'development_features.npz'
    if not cache.exists(): np.savez_compressed(cache,**extract([p for p in paths if int(p.parent.name[-2:])<=30]))
    f=dict(np.load(cache,allow_pickle=True))
    assert np.array_equal(f['session'],m['session']) and np.array_equal(f['label'],m['label'])
    preds,families=candidates(f,m); y=m['label']; ses=m['session']; base=preds['baseline']
    result={'protocol': '3 blocked session folds; >=0.5 point CV gain and sessions improved > harmed; family selection also nested within folds; no subject/session constants', 'rtamt_checks':checks,'baseline':score(y,base),'candidates':len(preds),'families':{}}
    oofs={}
    for family in sorted(set(families.values())-{'baseline'})+['all_families']:
        keys=['baseline']+[k for k in preds if k!='baseline' and (family=='all_families' or families[k]==family)]
        oof=base.copy(); chosen=[]
        for low in (1,11,21):
            held=np.array([low<=int(s[-2:])<low+10 for s in ses]); train=~held
            best=max(keys,key=lambda k:score(y[train],preds[k][train]))
            oof[held]=preds[best][held]; chosen.append(best)
        changes=[score(y[ses==s],oof[ses==s])-score(y[ses==s],base[ses==s]) for s in np.unique(ses)]
        gain=score(y,oof)-score(y,base); up=sum(v>1e-9 for v in changes); down=sum(v< -1e-9 for v in changes)
        record={'cv':score(y,oof),'gain':gain,'sessions_up':up,'sessions_down':down,'selected':chosen,'pass':gain>=.5 and up>down}
        result['families'][family]=record; oofs[family]=oof
        print(family,json.dumps(record),flush=True)
    # One final candidate chosen on development only, evaluated only if entire search passes.
    if result['families']['all_families']['pass']:
        best=max(preds,key=lambda k:score(y,preds[k])); result['frozen_candidate']=best
        (HERE/'frozen_selection.json').write_text(json.dumps(result,indent=2))
        testpaths=sorted((ROOT/'extracted_npz_dataset_v2').glob('*/ses-*/clean_epochs.npz'))
        testpaths=[p for p in testpaths if p.parent.parent.name!='sub_01' or int(p.parent.name[-2:])>30]
        tf=extract(testpaths); tm={k:v[~dev] for k,v in m0.items()}
        assert np.array_equal(tf['session'],tm['session']) and np.array_equal(tf['label'],tm['label'])
        tp,_=candidates(tf,tm)
        result['held_out']={}
        for subject in ('sub_01','sub_11'):
            ix=np.char.startswith(tm['session'].astype(str),subject)
            result['held_out'][subject]={'baseline':score(tm['label'][ix],tp['baseline'][ix]),'candidate':score(tm['label'][ix],tp[best][ix])}
    else: result['held_out']='Not evaluated: entire search failed development adoption rule.'
    # Relate new evidence to previous measurements, on development only.
    result['correlations']={}
    for key in ('double_4_32','triple_4_32','burst_4','quietburst_4','nodsync_4'):
        result['correlations'][key]={k:float(np.corrcoef(f[key],m[k])[0,1]) for k in ('turn','nod','background','deflect')}
    np.savez_compressed(HERE/'oof_predictions.npz',label=y,session=ses,baseline=base,**oofs)
    (HERE/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result.get('held_out')),flush=True)

if __name__=='__main__': main()
