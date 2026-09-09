import numpy as np

def rel_of(X):
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k): R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return R

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

def two_way(X):
    """Два порядка на одних событиях: вперёд и назад по времени."""
    Rf = rel_of(X)
    Rb = rel_of(1-X)          # обращение всех шкал = обращение времени
    return Rf, Rb
