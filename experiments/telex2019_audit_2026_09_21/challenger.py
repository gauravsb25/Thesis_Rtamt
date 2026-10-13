"""Evaluate the learned formulas explicitly, even if deployment selection preferred baseline."""
from pathlib import Path
actual=Path(__file__).resolve(); runpath=actual.parent/'run.py'
# Trusted local helper snapshot; execute definitions only, not the runner entry point.
__file__=str(runpath)
exec(compile(runpath.read_text().split("report={'protocol'")[0],str(runpath),'exec'),globals())
old=json.loads((HERE/'results.json').read_text())
def pick(table):
    row=max((r for r in table if r['eligible']),key=lambda r:r['metrics']['movement_macro'])
    return [row['budget'],row['depth'],row['gate']]

report={'note':'Forced learned challenger; original selection may choose baseline','modes':{}}
outputs={}; final={}
for mode in MODES:
    oof=BASE.copy(); selections=[]
    for outer in old['modes'][mode]['outer']:
        hold=outer['held_fold']; ids=[i for i in range(3) if i!=hold]; cfg=pick(outer['selection'])
        scores=scores_for(ids,cfg[0],mode); pp=decode(scores,cfg[1],cfg[2],DI)
        take=folds[hold][DI]; oof[DI[take]]=pp[take]; selections.append(cfg)
    cfg=pick(old['modes'][mode]['development_selection'])
    models=[trained([0,1,2],cfg[0],seed,mode)[0] for seed in SEEDS]
    final[mode]=(cfg,models); outputs[mode+'_oof']=oof
    report['modes'][mode]=dict(config=cfg,outer_configs=selections,nested_validation=metrics(oof,DEV))
    print('forced learned validation',mode,report['modes'][mode],flush=True)
# Freeze both formula sets and selection records before external scores.
for mode,(cfg,models) in final.items():
    (HERE/f'{mode}_challenger_monitor.json').write_text(json.dumps(dict(mode=mode,config=cfg,names=NAMES,fingerprint=FP,seeds=SEEDS,models=models),indent=2),encoding='utf-8')
(HERE/'challenger_frozen.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
for mode,(cfg,models) in final.items():
    p=BASE.copy(); row=report['modes'][mode]
    for ii in (DI,np.flatnonzero(~DEV)):
        scores=sum(score_model(model,ii) for model in models)/len(models)
        p[ii]=decode(scores,cfg[1],cfg[2],ii)
    row['development_fit']=metrics(p,DEV); row['external']={k:metrics(p,m) for k,m in SPLITS.items()}
    row['seed_external']=[]; row['discovery_fit']=[]
    for model in models:
        idx=np.array(model['indices']); one=BASE.copy(); one[idx]=decode(score_model(model,idx),cfg[1],cfg[2],idx)
        mask=np.zeros(len(Y),bool); mask[idx]=True; row['discovery_fit'].append(metrics(one,mask))
        ii=np.flatnonzero(~DEV); one=BASE.copy(); one[ii]=decode(score_model(model,ii),cfg[1],cfg[2],ii)
        row['seed_external'].append({k:metrics(one,m) for k,m in SPLITS.items()})
    row['eligible']=all(row['external'][k]['movement_macro']>old['baseline'][k]['movement_macro'] and
        all(row['external'][k]['recall'][c]>=old['baseline'][k]['recall'][c]-1 for c in CL) for k in SPLITS)
    outputs[mode+'_final']=p; print('CHALLENGER',mode,row['external'],'eligible',row['eligible'],flush=True)
report.update(elapsed_seconds=time.time()-START,production_unchanged=project_code.fingerprint()==FP)
(HERE/'challenger_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
np.savez_compressed(HERE/'challenger_predictions.npz',label=Y,session=S,baseline=BASE,**outputs)
print('COMPLETE',flush=True)
