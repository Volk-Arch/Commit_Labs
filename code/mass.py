import numpy as np

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

def rel_of(X):
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k): R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return R

def local_mass(X, R, kmin=3):
    """Кандидат в массу как СВОЙСТВО СОБЫТИЯ:
    геометрическое среднее темпов вдоль локального направления.
    Темп = насколько событие продвинулось по каждой шкале относительно соседей."""
    N, k = X.shape
    Rb = np.ascontiguousarray(R, dtype=bool)
    Rf = Rb.astype(np.float64)
    npast = np.maximum(Rf.sum(axis=1), 1)[:, None]
    nfut = np.maximum(Rf.sum(axis=0), 1)[:, None]
    cp = (Rf @ X)/npast          # центр прошлого
    cf = (Rf.T @ X)/nfut         # центр будущего
    v = cf - cp                  # локальное направление (темпы по шкалам)
    ok = (Rf.sum(axis=1) >= kmin) & (Rf.sum(axis=0) >= kmin)
    pos = np.clip(v, 1e-12, None)
    geo = np.exp(np.log(pos).mean(axis=1))     # геом. среднее темпов = интервал
    ari = v.mean(axis=1)
    return geo, ari, v, ok
