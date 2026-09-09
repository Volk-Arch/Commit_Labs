import numpy as np

def rel_of(X):
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k): R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return R

def pt(K):
    Kb = np.ascontiguousarray(K, dtype=bool); n = len(Kb)
    ind = Kb.sum(axis=1).astype(np.int64); D = np.zeros(n, dtype=np.int64)
    rd = list(np.flatnonzero(ind == 0))
    while rd:
        j = rd.pop(); kd = np.flatnonzero(Kb[:, j])
        if kd.size:
            D[kd] = np.maximum(D[kd], D[j]+1); ind[kd] -= 1
            rd.extend(kd[ind[kd] == 0].tolist())
    return D

def everything(X):
    """Меряем всё, что умеем, без цели."""
    N, k = X.shape
    R = rel_of(X); Rb = np.asarray(R, dtype=bool); Rf = Rb.astype(np.float32)
    A = Rf; L = Rb & ((A@A) == 0); Lf = L.astype(np.float32)
    D = pt(np.asarray(R, dtype=np.float32))
    Dc = pt(np.ascontiguousarray(Rf.T))
    q = {}
    q['прошлое'] = Rf.sum(1)
    q['будущее'] = Rf.sum(0)
    q['глубина'] = D.astype(float)
    q['кодглубина'] = Dc.astype(float)
    q['связей вниз'] = Lf.sum(1)
    q['связей вверх'] = Lf.sum(0)
    q['сумма фаз'] = X.sum(1)
    q['разброс фаз'] = X.var(1)
    q['размах фаз'] = X.max(1)-X.min(1)
    q['медиана фаз'] = np.median(X, 1)
    s = np.ones(N)
    for i in range(k):
        for j in range(i+1, k): s = s*np.sign(X[:, i]-X[:, j])
    q['ориентация'] = s
    q['прошлое+будущее'] = q['прошлое']+q['будущее']
    q['глубина+кодглубина'] = q['глубина']+q['кодглубина']
    return q, D, Rb, Lf
