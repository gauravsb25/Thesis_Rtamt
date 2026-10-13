"""Presentation report from fingerprint-checked genuine RTAMT verdicts."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
import project_code as project

def summarize(truth,prediction):
    names=project.CLASSES
    matrix=np.zeros((4,4),dtype=int)
    for a,b in zip(truth,prediction):
        if a not in names or b not in names:raise ValueError('Unknown or INVALID label in evaluation cache')
        matrix[names.index(a),names.index(b)]+=1
    counts=matrix.sum(1);correct=matrix.diagonal()
    return dict(windows=int(matrix.sum()),correct=int(correct.sum()),errors=int(matrix.sum()-correct.sum()),
                accuracy=float(correct.sum()/matrix.sum()) if matrix.sum() else None,
                balanced_accuracy=float(np.mean(correct/counts)) if np.all(counts) else None,
                per_class={c:dict(correct=int(correct[i]),total=int(counts[i]),recall=float(correct[i]/counts[i]) if counts[i] else None) for i,c in enumerate(names)},
                confusion_matrix=matrix.tolist())

def verify_live(f):
    from monitoring import HeadMovementMonitor
    from monitoring.stream_history import windows_with_history
    monitor=HeadMovementMonitor();wanted={}
    for _,mask in project.splits(f):
        for cls in project.CLASSES:
            ix=np.flatnonzero(mask&(f['label']==cls))
            if not len(ix):continue
            global_index=int(ix[len(ix)//2]);session=f['session'][global_index]
            session_indices=np.flatnonzero(f['session']==session)
            local_index=int(np.flatnonzero(session_indices==global_index)[0])
            wanted.setdefault(str(session),{})[local_index]=global_index
    checked=0
    for session,indices in wanted.items():
        for k,window,history in windows_with_history(project.DATA/session/'clean_epochs.npz'):
            if k in indices:
                expected=str(f['with_history'][indices[k]]);actual=monitor.classify(window,history)
                if actual!=expected:raise ValueError(f'Live RTAMT mismatch: {session}, window {k}: {actual} != {expected}')
                checked+=1
            if k>=max(indices):break
    return checked

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--details',action='store_true',help='Also show confusion matrices')
    parser.add_argument('--verify-live',action='store_true',help='Re-run 12 representative windows through actual RTAMT')
    parser.add_argument('--json',type=Path,help='Save the displayed metrics as JSON')
    parser.add_argument('--no-color',action='store_true')
    args=parser.parse_args();stamp=project.fingerprint()
    if project._cached(project.LABELS_CACHE,stamp) is None:
        parser.exit(1,'No matching RTAMT verdict cache. Run: python project_code.py evaluate\nThis recomputes genuine RTAMT predictions and may take about 90 minutes.\n')
    f=project.monitor_labels();checked=verify_live(f) if args.verify_live else 0
    names=['Development: sub_01 1-20','External: sub_01 21-36','External: sub_11']
    groups={name:summarize(f['label'][mask],f['with_history'][mask]) for name,(_,mask) in zip(names,project.splits(f))}
    result=dict(monitor='Original four-class RTAMT monitor with stream history',fingerprint=stamp,
                source='Fingerprint-checked cached HeadMovementMonitor.classify predictions',live_checks=checked,
                evaluation_status='Retrospective: external populations have been inspected during prior research',
                classes=project.CLASSES,groups=groups)
    color=sys.stdout.isatty() and not args.no_color and 'NO_COLOR' not in os.environ
    if color and os.name=='nt':
        import ctypes
        handle=ctypes.windll.kernel32.GetStdHandle(-11);mode=ctypes.c_uint()
        color=bool(ctypes.windll.kernel32.GetConsoleMode(handle,ctypes.byref(mode)) and ctypes.windll.kernel32.SetConsoleMode(handle,mode.value|4))
    def ink(text,code='36'):return f'\033[{code}m{text}\033[0m' if color else text
    def table(headers,rows):
        widths=[max(len(str(header)),*(len(str(row[i])) for row in rows)) for i,header in enumerate(headers)]
        border='+'+'+'.join('-'*(w+2) for w in widths)+'+'
        print(ink(border));print(ink('| '+' | '.join(str(h).ljust(w) for h,w in zip(headers,widths))+' |','1;36'));print(ink(border))
        for row in rows:print('| '+' | '.join(str(v).ljust(w) for v,w in zip(row,widths))+' |')
        print(ink(border))
    print('\n'+ink('='*86,'1;36'))
    print(ink('  EEG HEAD-MOVEMENT MONITORS  |  ACCURACY REPORT','1;37'))
    print('  LEFT / RIGHT / DOWN / STRAIGHT     |     Genuine RTAMT predictions')
    print(ink('='*86,'1;36'))
    print(f'  Dataset: {project.DATA.name}    Windows: {len(f["label"]):,}')
    print(f'  Monitor fingerprint: {stamp}    Stream history: enabled')
    print('  Verdict cache: MATCHES current monitor/settings/dataset')
    if checked:print(ink(f'  Live verification: {checked}/{checked} RTAMT predictions match the cache','32'))
    print('\n'+ink('  ACCURACY BY EVALUATION POPULATION','1;36'))
    table(['Population','Correct / total','Accuracy','Balanced accuracy'],
          [[name,f"{m['correct']:,} / {m['windows']:,}",f"{100*m['accuracy']:.2f}%",f"{100*m['balanced_accuracy']:.2f}%"] for name,m in groups.items()])
    print('\n'+ink('  CLASS RECALL  |  correct / total (percentage)','1;36'))
    rows=[]
    for cls in project.CLASSES:
        row=[cls]
        for m in groups.values():
            c=m['per_class'][cls];row.append(f"{c['correct']:,}/{c['total']:,} ({100*c['recall']:.2f}%)")
        rows.append(row)
    table(['Actual class','Development','Later sub_01','sub_11'],rows)
    if args.details:
        for name,m in groups.items():
            print('\n'+ink('  CONFUSION MATRIX: '+name,'1;36'))
            table(['Actual / predicted']+project.CLASSES,[[c]+row for c,row in zip(project.CLASSES,m['confusion_matrix'])])
    print('\n  Accuracy          = correct windows / all windows.')
    print('  Balanced accuracy = average recall across the four classes.')
    print('  Development data were used for design/threshold selection, not a blind test.')
    print('  External results are retrospective; these populations were previously inspected.')
    print('  Windows overlap. These are window scores, not independent gesture/command accuracy.')
    print(ink('='*86,'36')+'\n')
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True);args.json.write_text(json.dumps(result,indent=2))
        print('Saved:',args.json.resolve())

if __name__=='__main__':main()
