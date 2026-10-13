import numpy as np, harness as H
f = H.load(); base = H.predict(f); lab = f["label"]; p = H.SHIPPED
valid = f["background_valid"] >= 0.5
loud = np.where(valid, f["background"] >= p["b"], f["contrast"] >= p["g"])
dl = f["delta"] >= p["d"]; df = f["deflect"] >= p["q"]
for nm, g in (("dev", f["dev"]), ("t01", f["t01"]), ("s11", f["s11"])):
    k = g & (lab != "STRAIGHT") & (base == "STRAIGHT")
    n = k.sum()
    print(f"{nm}: missed movement windows {n}; with history {valid[k].sum()}; fail loud {(~loud[k]).sum()} delta {(~dl[k]).sum()} deflect {(~df[k]).sum()}; only-loud {(~loud[k] & dl[k] & df[k]).sum()} only-delta {(loud[k] & ~dl[k] & df[k]).sum()} only-deflect {(loud[k] & dl[k] & ~df[k]).sum()}")
    r = g & (lab == "STRAIGHT") & (base != "STRAIGHT")
    print(f"     false alarms {r.sum()} (history {valid[r].sum()}), as DOWN {(base[r]=='DOWN').sum()} LEFT {(base[r]=='LEFT').sum()} RIGHT {(base[r]=='RIGHT').sum()}")
    mvw = g & (lab != "STRAIGHT")
    print(f"     movement windows with valid history: {valid[mvw].mean()*100:.1f}% ; rest windows: {valid[g & (lab=='STRAIGHT')].mean()*100:.1f}%")
