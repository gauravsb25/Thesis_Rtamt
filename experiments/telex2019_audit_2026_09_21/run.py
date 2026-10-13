import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys,json,time,hashlib,itertools
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
from data import HERE,ROOT,NAMES,prepare
from logic import learn,evaluate,formula,horizon,size
prepare(); sys.path.insert(0,str(ROOT)); import project_code
START=time.time(); FP=project_code.fingerprint()
X=np.load(HERE/'traces.npy',mmap_mode='r'); meta=dict(np.load(HERE/'metadata.npz'))
CL=['LEFT','RIGHT','DOWN','STRAIGHT']; Y=np.array([CL.index(c) for c in meta['label']]); S=meta['session']; G=meta['group']; BASE=np.array([CL.index(c) for c in meta['baseline']])
DEV=np.array([s.startswith('sub_01') and int(s[-2:])<=30 for s in S]); DI=np.flatnonzero(DEV)
folds=[DEV & np.isin(S,[f'sub_01/ses-{s:02d}' for s in range(a,a+10)]) for a in (1,11,21)]
SPLITS={'external_sub01':np.char.startswith(S,'sub_01')&~DEV,'external_sub11':np.char.startswith(S,'sub_11')}
MODES=['positive','supervised']; SEEDS=[11,29]; BUDGETS=[8,16]
CACHE={}; fits=0

def balanced_sample(mask,budget,seed):
    rng=np.random.default_rng(seed)
    for count in range(budget,0,-1):
        chosen=[]; used=set(); good=True
        for c in range(4):
            eligible=np.flatnonzero(mask&(Y==c)); sessions=list(np.unique(S[eligible])); rng.shuffle(sessions)
            queue={s:list(np.unique(G[eligible[S[eligible]==s]])) for s in sessions}
            for v in queue.values(): rng.shuffle(v)
            picks=[]
            while len(picks)<count:
                before=len(picks)
                for s in sessions:
                    while queue[s] and queue[s][-1] in used: queue[s].pop()
                    if not queue[s]: continue
                    g=queue[s].pop(); used.add(g)
                    choices=eligible[G[eligible]==g]; picks.append(int(rng.choice(choices)))
                    if len(picks)==count: break
                if len(picks)==before: break
            if len(picks)<count: good=False; break
            chosen.extend(picks)
        if good:
            idx=np.array(chosen); assert len(np.unique(G[idx]))==len(idx)
            assert np.array_equal(np.bincount(Y[idx],minlength=4),np.repeat(count,4))
            return idx
    raise RuntimeError('Cannot sample four independent classes')

def score_model(model,indices):
    out=np.zeros((3,len(indices),4),float)
    for start in range(0,len(indices),256):
        xx=np.asarray(X[indices[start:start+256]],float)
        for depth in range(3):
            for c in range(4):
                out[depth,start:start+len(xx),c]=evaluate(model['trees'][c][depth],xx)[:,0]/model['scales'][c][depth]
    return out

def trained(ids,budget,seed,mode):
    global fits
    key=(tuple(sorted(ids)),budget,seed,mode)
    if key in CACHE: return CACHE[key]
    name=f'{"".join(map(str,sorted(ids)))}_n{budget}_s{seed}_{mode}'
    path=HERE/f'model_{name}.json'; predpath=HERE/f'dev_scores_{name}.npy'
    # Reuse only files from this fixed implementation/protocol and fingerprint.
    stamp=hashlib.sha256((Path(__file__).read_bytes()+ (HERE/'logic.py').read_bytes()+(HERE/'PROTOCOL.md').read_bytes()+FP.encode())).hexdigest()
    if path.exists() and predpath.exists():
        model=json.loads(path.read_text())
        if model.get('stamp')==stamp:
            result=(model,np.load(predpath,mmap_mode='r')); CACHE[key]=result; return result
    train=np.logical_or.reduce([folds[i] for i in ids]); idx=balanced_sample(train,budget,seed)
    xx=np.asarray(X[idx],float); yy=Y[idx]; trees=[]; details=[]; scales=[]
    for c in range(4):
        tree,diag=learn(xx,yy,c,mode); trees.append(tree); details.append(diag)
        scales.append([float(max(.1,np.median(np.abs((r:=evaluate(t,xx)[:,0])-np.median(r)))*1.4826)) for t in tree])
        assert all(np.min(evaluate(t,xx[yy==c])[:,0])>=-1e-6 for t in tree)
    model=dict(stamp=stamp,fingerprint=FP,mode=mode,budget=budget,seed=seed,train_folds=ids,indices=idx.tolist(),
        trial_keys=G[idx].tolist(),labels=yy.tolist(),trees=trees,scales=scales,diagnostics=details,
        formulas=[[formula(t,NAMES) for t in tt] for tt in trees])
    scores=score_model(model,DI); path.write_text(json.dumps(model,indent=2),encoding='utf-8'); np.save(predpath,scores)
    fits+=1; CACHE[key]=(model,scores)
    print('fit',name,'examples/class',len(idx)//4,'elapsed',round(time.time()-START,1),flush=True)
    return CACHE[key]

def scores_for(ids,budget,mode):
    return sum(trained(ids,budget,seed,mode)[1] for seed in SEEDS)/len(SEEDS)

def decode(scores,depth,gate,indices):
    sc=scores[depth-1]
    if gate: return np.where(BASE[indices]==3,3,sc[:,:3].argmax(1))
    return sc.argmax(1)

def metrics(pred,mask):
    cm=np.zeros((4,4),int); np.add.at(cm,(Y[mask],pred[mask]),1)
    recalls=100*np.divide(cm.diagonal(),cm.sum(1),out=np.full(4,np.nan),where=cm.sum(1)>0)
    return dict(correct=int(cm.trace()),n=int(mask.sum()),accuracy=float(100*cm.trace()/mask.sum()),balanced_accuracy=float(np.nanmean(recalls)),
        movement_macro=float(np.nanmean(recalls[:3])),recall=dict(zip(CL,recalls.tolist())),confusion=cm.tolist())

def select(ids,mode):
    mask=np.logical_or.reduce([folds[i] for i in ids]); base=metrics(BASE,mask); best=(base['movement_macro'],None); table=[]
    for budget in BUDGETS:
        scores=np.zeros((3,len(DI),4))
        for hold in ids:
            got=scores_for([i for i in ids if i!=hold],budget,mode); held=folds[hold][DI]; scores[:,held]=got[:,held]
        for depth,gate in itertools.product((1,2,3),(False,True)):
            p=BASE.copy(); p[DI]=decode(scores,depth,gate,DI); m=metrics(p,mask)
            eligible=m['recall']['STRAIGHT']>=base['recall']['STRAIGHT']-1.
            row=dict(budget=budget,depth=depth,gate=gate,metrics=m,eligible=eligible); table.append(row)
            if eligible and m['movement_macro']>best[0]+1e-9: best=(m['movement_macro'],[budget,depth,gate])
    return best[1],table

report={'protocol':'PROTOCOL.md','fingerprint':FP,'baseline':{k:metrics(BASE,m) for k,m in {'development':DEV,**SPLITS}.items()},'modes':{}}
outputs={}; selections={}
for mode in MODES:
    oof=BASE.copy(); outer=[]
    for hold in range(3):
        ids=[i for i in range(3) if i!=hold]; config,table=select(ids,mode)
        if config is not None:
            budget,depth,gate=config; sc=scores_for(ids,budget,mode); pred=decode(sc,depth,gate,DI)
            oof[DI[folds[hold][DI]]]=pred[folds[hold][DI]]
        outer.append(dict(held_fold=hold,config=config,selection=table))
        print('outer',mode,hold,config,flush=True)
    config,table=select([0,1,2],mode); selections[mode]=config; outputs[mode+'_oof']=oof
    report['modes'][mode]=dict(nested_validation=metrics(oof,DEV),outer=outer,final_config=config,development_selection=table)
    if config is not None:
        for seed in SEEDS: trained([0,1,2],config[0],seed,mode)
    print('development',mode,report['modes'][mode]['nested_validation'],flush=True)
(HERE/'frozen_development.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
# External outcomes are only computed after BOTH modes and all parameters are frozen.
for mode,config in selections.items():
    p=BASE.copy(); manifest=dict(mode=mode,config=config,names=NAMES,fingerprint=FP,seeds=SEEDS,models=[])
    if config is not None:
        budget,depth,gate=config; models=[trained([0,1,2],budget,seed,mode)[0] for seed in SEEDS]
        manifest['models']=models
        for indices in (DI,np.flatnonzero(~DEV)):
            sc=sum(score_model(model,indices) for model in models)/len(models); p[indices]=decode(sc,depth,gate,indices)
        report['modes'][mode]['discovery_fit']=[]
        for model in models:
            ii=np.array(model['indices']); one=BASE.copy(); one[ii]=decode(score_model(model,ii),depth,gate,ii)
            mask=np.zeros(len(Y),bool); mask[ii]=True
            report['modes'][mode]['discovery_fit'].append(metrics(one,mask))
    (HERE/f'{mode}_monitor.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    row=report['modes'][mode]; row['development_fit']=metrics(p,DEV)
    row['external']={k:metrics(p,m) for k,m in SPLITS.items()}
    outputs[mode+'_final']=p
    row['eligible']=config is not None and all(row['external'][k]['movement_macro']>report['baseline'][k]['movement_macro'] and
        all(row['external'][k]['recall'][c]>=report['baseline'][k]['recall'][c]-1 for c in CL) for k in SPLITS)
    print('EXTERNAL',mode,row['external'],'eligible',row['eligible'],flush=True)
report.update(fits_this_run=fits,elapsed_seconds=time.time()-START,production_unchanged=project_code.fingerprint()==FP)
(HERE/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
np.savez_compressed(HERE/'predictions.npz',label=Y,session=S,baseline=BASE,**outputs)
print('COMPLETE',report['elapsed_seconds'],flush=True)
