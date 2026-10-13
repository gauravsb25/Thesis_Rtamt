"""Per-class held-out recall for the key combinations (fits on sub_01 1-20 only)."""
import sys
import numpy as np
from e2e import assemble, predict_all, LAB, TEST01, TEST11, TRAIN
from group_select import build

stages = build()[:3]
combos = [{}, {"NOD": "vote_3"}, {"NOD": "vote_3@T3T4"}, {"NOD": "base_or"}, {"DETECT": "vote_3@all8"},
          {"DIRECTION": "base_and"}, {"DETECT": "vote_3@all8", "NOD": "vote_3@T3T4", "DIRECTION": "base_and"}]
for choice in combos:
    fns, _ = assemble(stages, choice, lambda st: st.train)
    p = predict_all(stages, fns, ~TRAIN)
    cells = []
    for name, m in (("sub_01 21-36", TEST01), ("sub_11", TEST11)):
        cells.append(name + " " + "  ".join(f"{c} {int((p[m & (LAB == c)] == c).sum())}/{int((m & (LAB == c)).sum())}"
                                             for c in ("LEFT", "RIGHT", "DOWN", "STRAIGHT")))
        t = m & (LAB != "DOWN") & (LAB != "STRAIGHT")
        cells[-1] += f"  | turns called DOWN {int((p[t] == 'DOWN').sum())}, DOWN called a turn {int(np.isin(p[m & (LAB == 'DOWN')], ['LEFT', 'RIGHT', 'TURN?']).sum())}"
    print(choice or "baseline")
    for c in cells:
        print("   " + c)
