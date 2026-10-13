"""Snapshot and independently rerun original vs corrected response-family CV only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys, importlib.util, json, hashlib, time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
SOURCE=ROOT/'experiments/operator_sweep_2026_09_21'
raw=(SOURCE/'run.py').read_text(encoding='utf-8')
(HERE/'signals.py').write_bytes((SOURCE/'signals.py').read_bytes())
modified=raw.replace('HERE / "bank.npz"','ROOT / "experiments/operator_sweep_2026_09_21/bank.npz"')
(HERE/'operator_search_snapshot.py').write_text(modified,encoding='utf-8')
loader=importlib.util.spec_from_file_location('independent_response_search',HERE/'operator_search_snapshot.py')
m=importlib.util.module_from_spec(loader); loader.loader.exec_module(m)
start=time.time(); result={'source_sha256':hashlib.sha256(raw.encode()).hexdigest()}
for version in ('original','corrected'):
    if version=='corrected':
        for j,k in enumerate(m.FEATS):
            if k.endswith('|resp'):
                no_trigger=m.X[m.FEATS.index(k.replace('|resp','|max'))]<2
                m.X[j,no_trigger]=0
    g,up,down,picks=m.cv_gain(m.X,only='resp')
    result[version]=dict(balanced_accuracy=m.bal(m.BASE,m.bal_idx(m.DEV))+g,gain=g,sessions_improved=up,sessions_harmed=down,picks=picks)
    print(version,json.dumps(result[version]),flush=True)
result['elapsed_seconds']=time.time()-start
(HERE/'response_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
