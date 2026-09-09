import numpy as np
from math import gamma
from scipy.optimize import brentq

def mm(f):
    g = lambda d: gamma(d+1)*gamma(d/2)/(2*gamma(1.5*d))
    try: return brentq(lambda d: g(d)-f, 1.0, 40)
    except ValueError: return float('nan')

def proper_time(K):
    Kb = np.ascontiguousarray(K, dtype=bool); n = len(Kb)
    indeg = Kb.sum(axis=1).astype(np.int64); D = np.zeros(n, dtype=np.int64)
    ready = list(np.flatnonzero(indeg == 0)); seen = 0
    while ready:
        j = ready.pop(); seen += 1
        kids = np.flatnonzero(Kb[:, j])
        if kids.size:
            D[kids] = np.maximum(D[kids], D[j]+1); indeg[kids] -= 1
            ready.extend(kids[indeg[kids] == 0].tolist())
    return D

def distinctness(K, samples=20000, rs=None):
    r = rs or np.random.default_rng(0)
    K = np.ascontiguousarray(K, dtype=np.float32)
    D = proper_time(K); hi = max(D.max(), 1)
    idx = np.flatnonzero((D > 0.25*hi) & (D < 0.9*hi))
    if len(idx) < 40: return float('nan')
    P = K[:, idx]; sz = P.sum(axis=0)
    rel = K[np.ix_(idx, idx)] > 0; rel = rel | rel.T
    out = []
    for _ in range(samples):
        i, j = r.integers(0, len(idx), 2)
        if i == j or rel[i, j] or abs(D[idx[i]]-D[idx[j]]) > 2: continue
        s = min(sz[i], sz[j])
        if s >= 8: out.append(float(P[:, i]@P[:, j])/s)
    return float(np.median(out)) if len(out) > 40 else float('nan')

def rel_of(X):
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k): R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return R

def rotl(v, s, bits):
    """Циклический сдвиг влево на s бит в слове длины bits."""
    mask = (1 << bits) - 1
    v = v & mask
    return ((v << s) | (v >> (bits - s))) & mask

def bit_input(N, k, bits, mode, rs):
    """Вход: ОДНО число на событие, k шкал — его циклические сдвиги."""
    if mode == 'random':
        base = rs.integers(0, 1 << bits, N, dtype=np.int64)
    elif mode == 'counter':
        base = np.arange(N, dtype=np.int64) % (1 << bits)
    elif mode == 'lcg':
        base = np.zeros(N, dtype=np.int64); x = 12345
        for i in range(N):
            x = (1103515245*x + 12345) % (1 << bits); base[i] = x
    shifts = np.linspace(0, bits, k, endpoint=False).astype(int)
    return np.column_stack([rotl(base, int(s), bits)/(1 << bits) for s in shifts])
