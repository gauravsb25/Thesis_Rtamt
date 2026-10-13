import sys; sys.path.insert(0, r"C:\Users\gaura\thesis_rtamt")
import numpy as np, project_code as pc
f = dict(np.load(r"C:\Users\gaura\thesis_rtamt\measurements_v2.npz", allow_pickle=True))
f['label']=f['label'].astype(str); f['session']=f['session'].astype(str)
pred = pc.predict(f); lab = f['label']
print(f"{'session':14s} {'L':>9s} {'R':>9s} {'D':>9s} {'S':>11s} | turn->D  D->turn  D->S  LR-swap | nod med(turn) nod med(down) side med(turn) side med(down)")
for s in sorted(set(f['session'])):
    m = f['session']==s
    cells=[]
    for c in pc.CLASSES:
        k = m & (lab==c)
        cells.append(f"{(pred[k]==c).sum():3d}/{k.sum():<4d}" if c!='STRAIGHT' else f"{(pred[k]==c).sum():4d}/{k.sum():<5d}")
    turn = m & np.isin(lab,['LEFT','RIGHT']); down = m & (lab=='DOWN')
    t2d = (pred[turn]=='DOWN').sum(); d2t = np.isin(pred[down],['LEFT','RIGHT']).sum(); d2s=(pred[down]=='STRAIGHT').sum()
    sw = ((lab=='LEFT')&(pred=='RIGHT')|(lab=='RIGHT')&(pred=='LEFT'))[m].sum()
    nt = np.median(f['nod'][turn]) if turn.any() else np.nan; nd = np.median(f['nod'][down]) if down.any() else np.nan
    st = np.median(f['sideways'][turn]) if turn.any() else np.nan; sd = np.median(f['sideways'][down]) if down.any() else np.nan
    print(f"{s:14s} {cells[0]} {cells[1]} {cells[2]} {cells[3]} | {t2d:6d} {d2t:7d} {d2s:5d} {sw:7d} | {nt:8.2f} {nd:10.2f} {st:10.2f} {sd:10.2f}")
