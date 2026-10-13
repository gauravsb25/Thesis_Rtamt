"""Fast arithmetic scorer for the shipped monitor + candidate modifications, with the project's blocked 3-fold CV."""
import sys; sys.path.insert(0, r"C:\Users\gaura\thesis_rtamt")
import numpy as np
CLASSES = ["LEFT","RIGHT","DOWN","STRAIGHT"]
SHIPPED = dict(b=6.61, g=7.81, d=0.208, q=2.72, n=-0.661, v=7.54)

def load():
    f = dict(np.load(r"C:\Users\gaura\thesis_rtamt\measurements_v2.npz", allow_pickle=True))
    f["label"] = f["label"].astype(str); f["session"] = f["session"].astype(str)
    f["dev"] = np.array([s.startswith("sub_01") and int(s[-2:]) <= 30 for s in f["session"]])
    f["t01"] = np.array([s.startswith("sub_01") and int(s[-2:]) > 30 for s in f["session"]])
    f["s11"] = np.char.startswith(f["session"], "sub_11")
    FOLDS = [[f"sub_01/ses-{s:02d}" for s in range(a, a + 10)] for a in (1, 11, 21)]
    f["fold"] = np.full(len(f["label"]), -1)
    for k, ss in enumerate(FOLDS):
        f["fold"][np.isin(f["session"], ss)] = k
    return f

def move_mask(f, p=SHIPPED, history=True):
    valid = (f["background_valid"] >= 0.5) if history else np.zeros(len(f["label"]), bool)
    loud = np.where(valid, f["background"] >= p["b"], f["contrast"] >= p["g"])
    return loud & (f["delta"] >= p["d"]) & (f["deflect"] >= p["q"])

def predict(f, p=SHIPPED, nod=None, notnod_extra=None, direction=None, move=None, history=True):
    """nod: bool array 'looks like a nod' (default shipped). notnod_extra: bool array that vetoes DOWN.
    direction: array >0 LEFT, <0 RIGHT (default f['turn'])."""
    mv = move_mask(f, p, history) if move is None else move
    isnod = (f["nod"] >= p["n"]) & (f["sideways"] <= p["v"]) if nod is None else nod
    if notnod_extra is not None:
        isnod = isnod & ~notnod_extra
    d = f["turn"] if direction is None else direction
    return np.where(~mv, "STRAIGHT", np.where(isnod, "DOWN", np.where(d > 0, "LEFT", "RIGHT")))

def balanced(f, pred, m):
    return 100 * float(np.mean([np.mean(pred[m & (f["label"] == k)] == k) for k in CLASSES if (m & (f["label"] == k)).any()]))

def per_session(f, pred, m):
    out = {}
    for s in np.unique(f["session"][m]):
        k = m & (f["session"] == s)
        out[s] = balanced(f, pred, k)
    return out

def compare(f, base, cand, m):
    a, b = per_session(f, base, m), per_session(f, cand, m)
    better = sum(b[s] > a[s] + 1e-9 for s in a); worse = sum(b[s] < a[s] - 1e-9 for s in a)
    return balanced(f, cand, m) - balanced(f, base, m), better, worse

def cv(f, make_pred, grid):
    """make_pred(params) -> prediction over all windows. Select params maximising balanced acc on the 2 training
    folds, apply to the held fold. Returns CV prediction on dev and the chosen params per fold."""
    dev = f["dev"]; out = np.array(["?"] * len(f["label"]), dtype=object)
    preds = [make_pred(g) for g in grid]
    chosen = []
    for k in range(3):
        train = dev & (f["fold"] != k); held = dev & (f["fold"] == k)
        scores = [balanced(f, p, train) for p in preds]
        j = int(np.argmax(scores))
        chosen.append(grid[j]); out[held] = preds[j][held]
    return out.astype(str), chosen
