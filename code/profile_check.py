import numpy as np
from itertools import combinations
from math import gamma
from scipy.optimize import brentq
import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from layer3 import find_defects
from interval_orient import interval, interval_frame

def mm(f):
    g = lambda d: gamma(d+1)*gamma(d/2)/(2*gamma(1.5*d))
    try: return brentq(lambda d: g(d)-f, 1.0, 40)
    except ValueError: return float('nan')

def acyc(R):
    Rb = np.ascontiguousarray(R, dtype=bool); n = len(Rb)
    indeg = Rb.sum(axis=1).astype(np.int64)
    ready = list(np.flatnonzero(indeg == 0)); seen = 0
    while ready:
        j = ready.pop(); seen += 1
        kids = np.flatnonzero(Rb[:, j])
        if kids.size:
            indeg[kids] -= 1; ready.extend(kids[indeg[kids] == 0].tolist())
    return seen == n

def run_once(N, seed, k=4, minsz=20, rad=0.09):
    rs = np.random.default_rng(seed)
    X = rs.random((N, k))
    R = np.ones((N, N), dtype=bool)
    for m in range(k): R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False); Rb = np.asarray(R, dtype=bool)
    r2 = np.random.default_rng(seed+1); best = None
    for _ in range(6000):
        i = int(r2.integers(0, N))
        ends = [j for j in range(N) if Rb[i, j] and len(interval(Rb, i, j)) >= minsz]
        if len(ends) >= int(0.1*N): best = (i, ends); break
    if best is None: return None
    i0, ends = best
    emap = {j: t for t, j in enumerate(ends)}
    feat = np.zeros((len(ends), 4))
    for c, tri in enumerate(combinations(range(k), 3)):
        Xa = X[:, list(tri)]
        F = {}
        for j in ends:
            f = interval_frame(Xa, i0, j, interval(Rb, i0, j))
            if f is not None: F[j] = f[1]
        for j, w in find_defects(Xa, F, list(F), rad=rad): feat[emap[j], c] = w
    sel = np.flatnonzero((feat != 0).any(axis=1))
    if len(sel) < 40: return None
    Y = feat[sel]; n = len(Y)
    prof = np.column_stack([(Y != 0).sum(1), Y.sum(1), np.abs(Y).sum(1)])
    prof = prof + 1e-9*np.random.default_rng(seed+2).random(prof.shape)
    Rp = np.ones((n, n), dtype=bool)
    for m in range(3): Rp &= (prof[None, :, m] < prof[:, None, m])
    np.fill_diagonal(Rp, False)
    f = Rp.sum()/(n*(n-1)/2)
    return n, f, mm(f) if 0 < f < 1 else float('nan'), acyc(Rp), Y
