"""Bounded STL semantics and a TeLEx-style, deliberately finite structure search."""
import copy, json, itertools
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from numpy.lib.stride_tricks import sliding_window_view
BETA=10.; GAMMA=.01
INTERVALS=[(0,15),(0,31),(0,63),(0,95),(16,47),(32,63),(64,95)]

def peak(r):
    r=np.asarray(r,float)
    return 1/(r+np.exp(np.clip(-BETA*r,-60,60)))-np.exp(np.clip(-r,-60,60))

def size(f):
    return 1+sum(size(f[k]) for k in ('child','left','right') if k in f)

def horizon(f):
    op=f['op']
    if op=='atom': return 0
    if op in ('aug','not'): return horizon(f['child'])
    if op in ('F','G'): return f['b']+horizon(f['child'])
    if op=='U': return f['b']+max(horizon(f['left']),horizon(f['right']))
    return max(horizon(f['left']),horizon(f['right']))

def evaluate(f,x,tight=False):
    op=f['op']
    if op=='atom':
        r=f['sign']*x[:,f['signal'],:]-f['threshold']
        return peak(r) if tight else r
    if op in ('aug','not'):
        r=evaluate(f['child'],x,False)*(f.get('sign',-1) if op=='aug' else -1)
        return peak(r) if tight else r
    if op in ('F','G'):
        v=evaluate(f['child'],x,tight)
        a,b=f['a'],f['b']; n=v.shape[1]-b
        if n<=0: raise ValueError('Formula horizon exceeds trace')
        windows=sliding_window_view(v,b-a+1,axis=-1)[:,a:a+n]
        out=windows.max(-1) if op=='F' else windows.min(-1)
        if tight: out=out*2*expit((-1 if op=='F' else 1)*GAMMA*(b-a+1))
        return out
    l=evaluate(f['left'],x,tight); r=evaluate(f['right'],x,tight); n=min(l.shape[1],r.shape[1])
    if op=='and': return np.minimum(l[:,:n],r[:,:n])
    if op=='or': return np.maximum(l[:,:n],r[:,:n])
    if op=='U':
        a,b=f['a'],f['b']; n-=b
        if n<=0: raise ValueError('Formula horizon exceeds trace')
        out=np.full((len(x),n),-np.inf); prefix=np.full_like(out,np.inf)
        for k in range(b+1):
            if k>=a: out=np.maximum(out,np.minimum(prefix,r[:,k:k+n]))
            prefix=np.minimum(prefix,l[:,k:k+n])
        return out*(2*expit(GAMMA*(b-a+1)) if tight else 1)
    raise ValueError(op)

def formula(f,names):
    op=f['op']
    if op=='atom':
        lhs=names[f['signal']]; threshold=f['threshold']/f['sign']; relation='>=' if f['sign']>0 else '<='
        return f'({lhs} {relation} {threshold:.17g})'
    if op in ('aug','not'):
        child=formula(f['child'],names)
        return child if op=='aug' and f['sign']>0 else f'not({child})'
    if op in ('F','G'): return f'{"eventually" if op=="F" else "always"}[{f["a"]}:{f["b"]}]({formula(f["child"],names)})'
    if op=='U': return f'({formula(f["left"],names)} until[{f["a"]}:{f["b"]}] {formula(f["right"],names)})'
    return f'({formula(f["left"],names)} {op} {formula(f["right"],names)})'

def learn(x,y,target,mode,beam=5):
    """Each class formula is consistent with all discovery positives. Negatives rank only in supervised mode."""
    positive=y==target; xp=x[positive]
    counts={'templates':0,'refinements':0,'nested_candidates':0}
    def rank(f):
        counts['templates']+=1
        rp=evaluate(f,xp)[:,0]
        if np.min(rp)<-1e-6: return None
        tight=float(np.min(evaluate(f,xp,True)[:,0]))
        if mode=='positive': return (tight/(1+.02*size(f)), -float(np.mean(np.abs(rp))),-size(f))
        r=evaluate(f,x)[:,0]; pred=r>=-1e-6
        specificity=np.mean([np.mean(~pred[y==c]) for c in np.unique(y) if c!=target])
        return (float(.5*(np.mean(pred[positive])+specificity))-.001*size(f),tight,-size(f))
    pool=[]
    for signal in range(x.shape[1]):
        for sign,op,(a,b) in itertools.product((1,-1),('F','G'),INTERVALS):
            values=sign*xp[:,signal,a:b+1]
            extrema=values.max(1) if op=='F' else values.min(1)
            limit=float(extrema.min())
            f=dict(op=op,a=a,b=b,child=dict(op='atom',signal=signal,sign=sign,threshold=limit-.1))
            r=rank(f)
            if r is not None: pool.append((r,f))
    pool.sort(key=lambda t:t[0],reverse=True)
    # Refine thresholds on shortlisted templates with the paper's max-min positive tightness objective.
    for _,f in pool[:12]:
        limit=f['child']['threshold']+.1
        def cost(t):
            f['child']['threshold']=float(t)
            return -float(np.min(evaluate(f,xp,True)[:,0]))
        result=minimize_scalar(cost,bounds=(limit-3.2,limit-1e-7),method='bounded',options={'maxiter':40,'xatol':1e-5})
        f['child']['threshold']=float(result.x); counts['refinements']+=1
    pool=[(rank(f),f) for _,f in pool]; pool=[p for p in pool if p[0] is not None]
    pool.sort(key=lambda t:t[0],reverse=True)
    current=pool[:beam]; selected=[copy.deepcopy(current[0][1])]; history=[]
    for depth in (2,3):
        candidates=[f for _,f in current]
        for _,child in current:
            for sign in (1,-1):
                aug=dict(op='aug',sign=sign,child=child)
                for op,(a,b) in itertools.product(('F','G'),((0,3),(0,7),(0,15),(0,31),(8,15))):
                    f=dict(op=op,a=a,b=b,child=aug)
                    if horizon(f)<96 and size(f)<=14: candidates.append(f)
        for (_,a),(_,b) in itertools.combinations(current,2):
            for op in ('and','or'):
                f=dict(op=op,left=a,right=b)
                if size(f)<=14: candidates.append(f)
            for lo,hi in ((0,7),(0,15),(8,15)):
                f=dict(op='U',a=lo,b=hi,left=dict(op='aug',sign=1,child=a),right=dict(op='aug',sign=1,child=b))
                if horizon(f)<96 and size(f)<=14: candidates.append(f)
        # Retemplate primitive thresholds to create complementary, individually inconsistent disjuncts.
        for (_,a),(_,b) in itertools.combinations(pool[:5],2):
            aa=copy.deepcopy(a); bb=copy.deepcopy(b)
            aa['child']['threshold']+=.5; bb['child']['threshold']+=.5
            candidates.append(dict(op='or',left=aa,right=bb))
        counts['nested_candidates']+=len(candidates)
        ranked=[]; seen=set()
        for f in candidates:
            key=json.dumps(f,sort_keys=True)
            if key in seen: continue
            seen.add(key); r=rank(f)
            if r is not None: ranked.append((r,f))
        ranked.sort(key=lambda t:t[0],reverse=True)
        # Paper-inspired decreasing robustness band, capped beam. Preserve best consistent candidate
        # if band becomes empty; this fallback and supervised ranking are explicit deviations.
        average=np.mean([np.mean(np.abs(evaluate(f,xp)[:,0])) for _,f in ranked])
        band=[p for p in ranked if np.min(np.abs(evaluate(p[1],xp)[:,0]))<=average*np.exp(-(depth-1))/size(p[1])]
        current=(band or ranked)[:beam]
        selected.append(copy.deepcopy(current[0][1])); history.append(dict(depth=depth,consistent=len(ranked),in_band=len(band)))
    return selected,dict(counts=counts,history=history)
