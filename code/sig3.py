import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as GBR

def proper_time(K):
    Kb = np.ascontiguousarray(K, dtype=bool); n = len(Kb)
    indeg = Kb.sum(axis=1).astype(np.int64)
    D = np.zeros(n, dtype=np.int64)
    ready = list(np.flatnonzero(indeg == 0)); seen = 0
    while ready:
        j = ready.pop(); seen += 1
        kids = np.flatnonzero(Kb[:, j])
        if kids.size:
            D[kids] = np.maximum(D[kids], D[j] + 1)
            indeg[kids] -= 1
            ready.extend(kids[indeg[kids] == 0].tolist())
    return D

def rel_of(X):
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k):
        R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return R

def feats(R):
    Rb = np.ascontiguousarray(np.asarray(R, dtype=bool))
    Rf = Rb.astype(np.float32); N = len(Rb)
    past = Rf.sum(axis=1); fut = Rf.sum(axis=0)
    dep = proper_time(Rf).astype(float)
    codep = proper_time(np.ascontiguousarray(Rf.T)).astype(float)
    btw = (Rf@Rf); L = Rb & (btw == 0)
    return np.column_stack([past, fut, dep, codep, L.sum(1), L.sum(0),
                            past-fut, dep-codep, past*fut/N**2])

def r2(F, y, seed=0):
    n = len(y); h = n//2
    m = GBR(max_iter=200, random_state=seed).fit(F[:h], y[:h])
    p = m.predict(F[h:]); yt = y[h:]
    return float(1 - np.var(yt-p)/max(np.var(yt), 1e-12))

def to_next(R, k, rs):
    N = len(R)
    A = np.asarray(R, dtype=np.float32)
    S = proper_time(A).astype(float) - proper_time(np.ascontiguousarray(A.T)).astype(float)
    V = np.abs(A.sum(axis=1) - A.sum(axis=0))
    S = np.argsort(np.argsort(S + 1e-9*rs.random(N)))/N
    V = np.argsort(np.argsort(V + 1e-9*rs.random(N)))/N
    sh = np.array([np.sin(2*np.pi*j/k + 0.7) for j in range(k)])
    sh -= sh.mean(); sh /= sh.std()
    return rel_of(np.mod(S[:, None] + 5.0*V[:, None]*sh[None, :], 1.0))
