"""Independent read-only audit; outputs confined to this directory. Run with python -B."""
import sys, json, hashlib, importlib.util, contextlib, io, re
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / '.runtime' / 'rtamt'))
import rtamt

source = ROOT / 'experiments/operator_sweep_2026_09_21/signals.py'
raw = source.read_bytes()
(HERE / 'signals_snapshot.py').write_bytes(raw)
loader = importlib.util.spec_from_file_location('audit_signals', HERE / 'signals_snapshot.py')
mod = importlib.util.module_from_spec(loader); loader.loader.exec_module(mod)

def evaluate(formula, x, mode='offline'):
    spec = (rtamt.StlDiscreteTimeOfflineSpecification() if mode == 'offline'
            else rtamt.StlDiscreteTimeOnlineSpecification())
    spec.declare_var('x', 'float'); spec.declare_var('y', 'float')
    spec.spec = formula
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.parse()
        if mode == 'pastified': spec.pastify()
        if mode == 'offline':
            return spec.evaluate({'time':list(range(len(x))), 'x':list(map(float,x)), 'y':[1.0]*len(x)})
        return [spec.update(i, [('x',float(v))] + ([('y',1.0)] if re.search(r'\by\b', formula) else [])) for i,v in enumerate(x)]

formulas = ['x + y >= 0','x - y >= 0','x * y >= 0','x / y >= 0',
            'abs(x) >= 0','sqrt(abs(x)) >= 0','exp(x) >= 0','pow(x, 2) >= 0',
            'x > y','x >= y','x < y','x <= y','x == y','x !== y',
            'not(x >= 0)','(x >= 0) and (y >= 0)','(x >= 0) or (y >= 0)',
            '(x >= 0) implies (y >= 0)','(x >= 0) iff (y >= 0)','(x >= 0) xor (y >= 0)',
            'rise(x >= 0)','fall(x >= 0)','prev(x >= 0)','s_prev(x >= 0)',
            'next(x >= 0)','s_next(x >= 0)']
for op in ['always','eventually','historically','once']:
    formulas += [f'{op}(x >= 0)',f'{op}[0:2](x >= 0)']
for op in ['until','unless','since']:
    formulas += [f'(x >= 0) {op} (y >= 0)',f'(x >= 0) {op}[0:2] (y >= 0)']
results=[]
for f in formulas:
    row={'formula':f}
    for mode in ['offline','online','pastified']:
        try: evaluate(f,[-3,1,2,-1,3],mode); row[mode]='PASS'
        except Exception as e: row[mode]=f'{type(e).__name__}: {e}'
    results.append(row)

counterexamples=[]
for label,x in [('quiet',np.zeros(12)),('negative_excursion',np.array([0,-3,-3,0]*3,dtype=float))]:
    out={}; mod._reduce(x,out,'s')
    response=evaluate('always[0:11]((x >= 2) implies eventually[0:3](x <= -2))',x)[0][1]
    signed=sum(r[1]>0 for r in evaluate('rise(x >= 2)',x))
    absolute=sum(r[1]>0 for r in evaluate('rise(abs(x) >= 2)',x))
    counterexamples.append(dict(trace=label,resp_feature=out['s|resp'],
        feature_response_satisfied=bool(out['s|resp']*len(x)<=3),
        actual_response_satisfied=bool(response>0),rise_feature=out['s|rise2'],
        signed_rise_events=signed,absolute_rise_events=absolute))
report={'source_sha256':hashlib.sha256(raw).hexdigest(),
        'source_unchanged':source.read_bytes()==raw,'operator_support':results,
        'counterexamples':counterexamples}
(HERE/'audit_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
for mode in ['offline','online','pastified']:
    print(mode, sum(r[mode]=='PASS' for r in results),'/',len(results))
print(json.dumps(counterexamples,indent=2))
print('Source unchanged:',report['source_unchanged'])
