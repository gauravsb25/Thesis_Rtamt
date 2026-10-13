"""Constrained STL AND/OR structure synthesis. No fitted statistical classifier."""
import json
import numpy as np

CL=np.array(['LEFT','RIGHT','DOWN','STRAIGHT'])
PENALTY=.20  # balanced accuracy percentage points per added predicate

def initial(m):
    gate=np.where(m['background_valid']>=.5,m['background']>=6.61,m['contrast']>=7.81)&(m['delta']>=.208)&(m['deflect']>=2.72)
    nod=(m['nod']>=-.661)&(m['sideways']<=7.54)
    return dict(gate=gate,nod=nod,left=m['turn']>0)

def classify(state):
    return np.where(state['gate'],np.where(state['nod'],'DOWN',np.where(state['left'],'LEFT','RIGHT')),'STRAIGHT')

def ba(y,p): return float(np.mean([np.mean(p[y==c]==c) for c in CL if np.any(y==c)])*100)

def evaluate(ast,X,state):
    op=ast[0]
    if op=='base': return state[ast[1]]
    if op=='constant': return np.full(len(X),ast[1],bool)
    if op=='lit':
        mask=X[:,ast[1]]>=ast[2]
        return mask if ast[3] else ~mask
    a=evaluate(ast[1],X,state); b=evaluate(ast[2],X,state)
    return a&b if op=='and' else a|b

def predict(model,X,m):
    seed=initial(m)
    return classify({k:evaluate(v,X,seed) for k,v in model.items()})

def threshold_bank(X,y,weight,base):
    """Screen 64 features inside training only, retaining complementary relevance ranks."""
    quant=np.array([.03,.10,.25,.40,.60,.75,.90,.97])
    move=y!='STRAIGHT'
    cuts=np.concatenate([np.quantile(X,quant,axis=0),np.quantile(X[move],quant,axis=0),np.zeros((1,X.shape[1]))],axis=0)
    scores=[]; allcuts=[]
    for j in range(X.shape[1]):
        ts=np.unique(cuts[:,j]); allcuts.append(ts)
        c=X[:,j][None,:]>=ts[:,None]
        standalone=np.einsum('ij,j->i',c,weight,optimize=False)
        inside=np.einsum('ij,j->i',c,weight*base,optimize=False)
        outside=np.einsum('ij,j->i',c,weight*~base,optimize=False)
        standalone=max(standalone.max(),weight.sum()-standalone.min())
        extension=max(outside.max(),(weight*~base).sum()-outside.min(),-inside.min(),inside.max()-(weight*base).sum())
        scores.append([standalone,extension])
    scores=np.array(scores)
    keep=np.unique(np.concatenate([np.argsort(scores[:,i],kind='stable')[-32:] for i in (0,1)]))
    masks=[]; specs=[]
    for j in keep:
        for t in allcuts[j]:
            c=X[:,j]>=t
            masks.extend([c,~c]); specs.extend([['lit',int(j),float(t),True],['lit',int(j),float(t),False]])
    return np.array(masks),specs

def fit_stage(X,y,weight,base,stage,budget):
    bank,literals=threshold_bank(X,y,weight,base)
    baseline_score=float(weight@base)
    # Four best formulas per depth. Constants allow replacing the baseline entirely.
    beam=[(baseline_score,['base',stage],base), (0.,['constant',False],np.zeros(len(y),bool)),
          (float(weight.sum()),['constant',True],np.ones(len(y),bool))]
    winner=(baseline_score,['base',stage],base)
    for depth in range(1,budget+1):
        pool=[]
        for _,ast,mask in beam:
            current=float(weight@mask)
            a=np.einsum('ij,j->i',bank,weight*mask,optimize=False)
            o=current+np.einsum('ij,j->i',bank,weight*~mask,optimize=False)
            for op,score in (('and',a),('or',o)):
                top=np.argsort(score,kind='stable')[-4:]
                for idx in top:
                    val=float(score[idx])-PENALTY*depth
                    # Constants are simplified to a single literal when possible.
                    expr=[op,ast,literals[idx]]
                    out=mask&bank[idx] if op=='and' else mask|bank[idx]
                    pool.append((val,expr,out))
        pool.sort(key=lambda p:p[0],reverse=True)
        beam=[]; seen=set()
        for item in pool:
            signature=np.packbits(item[2]).tobytes()
            if signature not in seen:
                seen.add(signature); beam.append(item)
            if len(beam)==4: break
        if beam[0][0]>winner[0]+1e-9: winner=beam[0]
    return winner[1],winner[2]

def fit(X,y,m,budget,group=None,order=('gate','nod','left')):
    state=initial(m); model={k:['base',k] for k in state}
    counts={c:np.sum(y==c) for c in CL}
    weights=np.array([25./counts[c] for c in y],dtype=np.float32)
    if group is not None:
        for c in CL:
            ix=y==c
            _,inverse,number=np.unique(group[ix],return_inverse=True,return_counts=True)
            weights[ix]=25./(len(number)*number[inverse])
    for stage in order:
        yes={**state,stage:np.ones(len(y),bool)}; no={**state,stage:np.zeros(len(y),bool)}
        gain=weights*((classify(yes)==y).astype(np.int8)-(classify(no)==y).astype(np.int8))
        model[stage],state[stage]=fit_stage(X,y,gain,state[stage],stage,budget)
    return model

def formula(ast,names):
    op=ast[0]
    if op=='base': return f'({ast[1]}0 >= 0.5)'
    if op=='constant': return '(gate0 >= -1)' if ast[1] else '(gate0 < -1)'
    if op=='lit':
        # Features are float32. Move a representable threshold halfway to its predecessor,
        # preserving all Boolean decisions while avoiding zero-robustness ties in RTAMT.
        t=ast[2]
        if float(np.float32(t))==t:
            t=(t+float(np.nextafter(np.float32(t),np.float32(-np.inf))))/2
        return f'({names[ast[1]]} {">=" if ast[3] else "<"} {t:.17g})'
    return f'({formula(ast[1],names)} {op} {formula(ast[2],names)})'

def export(model,names):
    g,n,l=[formula(model[k],names) for k in ('gate','nod','left')]
    return {'STRAIGHT':f'not ({g})','DOWN':f'({g}) and ({n})',
            'LEFT':f'({g}) and (not ({n})) and ({l})',
            'RIGHT':f'({g}) and (not ({n})) and (not ({l}))'}
