import numpy as np
from math import gamma
from scipy.optimize import brentq

def proper_time(K):
    Kb = np.ascontiguousarray(K, dtype=bool); n = len(Kb)
    indeg = Kb.sum(axis=1).astype(np.int64); D = np.zeros(n, dtype=np.int64)
    ready = list(np.flatnonzero(indeg == 0)); seen = 0
    while ready:
        j = ready.pop(); seen += 1
        kids = np.flatnonzero(Kb[:, j])
        if kids.size:
            D[kids] = np.maximum(D[kids], D[j]+1)
            indeg[kids] -= 1
            ready.extend(kids[indeg[kids] == 0].tolist())
    return D

def rel_of(X):
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k): R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return R

def mm_dim(f):
    g = lambda d: gamma(d+1)*gamma(d/2)/(2*gamma(1.5*d))
    try: return brentq(lambda d: g(d)-f, 1.0, 40)
    except ValueError: return float('nan')
