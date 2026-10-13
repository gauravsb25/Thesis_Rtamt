"""Protocol-change measurement (NOT a rule-compliant candidate): per-session nod threshold from ONE known nod
and ONE known turn performed at session start (the first DOWN run and the first turn run in run order).
Threshold = midpoint of their median nod values (fallback: shipped n if a class is missing or order inverted).
Calibration runs are excluded from scoring for BOTH monitors."""
import numpy as np, harness as H
f = H.load(); lab = f["label"]; p = H.SHIPPED
m = dict(np.load("meta.npz", allow_pickle=True))
run = m["run"].astype(int); task = m["task"].astype(str)
mvm = H.move_mask(f)
thr = np.full(len(lab), p["n"]); keep = np.ones(len(lab), bool); used = {}
for s in np.unique(f["session"]):
    k = f["session"] == s
    runs = sorted(set(zip(run[k], task[k])))
    first_d = next((r for r, t in runs if t == "down" and ((run == r) & k & (lab == "DOWN")).any()), None)
    first_t = next((r for r, t in runs if t in ("left", "right") and ((run == r) & k & np.isin(lab, ["LEFT","RIGHT"])).any()), None)
    if first_d is None or first_t is None:
        used[s] = "fallback (class missing)"; continue
    cd = k & (run == first_d) & (lab == "DOWN"); ct = k & (run == first_t) & np.isin(lab, ["LEFT","RIGHT"])
    nd, nt = np.median(f["nod"][cd]), np.median(f["nod"][ct])
    keep[k & ((run == first_d) | (run == first_t))] = False
    if nd > nt:
        thr[k] = (nd + nt) / 2; used[s] = f"thr {(nd+nt)/2:5.2f}"
    else:
        used[s] = "fallback (inverted)"
isnod = (f["nod"] >= thr) & (f["sideways"] <= p["v"])
cal = np.where(~mvm, "STRAIGHT", np.where(isnod, "DOWN", np.where(f["turn"] > 0, "LEFT", "RIGHT")))
base = H.predict(f)
for split in ("dev", "t01", "s11"):
    mm = f[split] & keep
    print(f"{split}: shipped {H.balanced(f, base, mm):6.2f} -> 1-nod+1-turn calibrated {H.balanced(f, cal, mm):6.2f}   "
          f"[" + " ".join(f"{c[0]} {np.mean(base[mm & (lab==c)]==c)*100:5.1f}->{np.mean(cal[mm & (lab==c)]==c)*100:5.1f}" for c in H.CLASSES) + "]"
          f"  sessions (gain, better, worse) {H.compare(f, base, cal, mm)}")
print({k: v for k, v in used.items() if not k.startswith("sub_01") or int(k[-2:]) > 30})
