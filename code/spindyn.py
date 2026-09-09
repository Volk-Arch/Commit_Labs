import numpy as np
import sys; sys.path.insert(0, '/home/claude')
from cliff import clifford, transposition_op

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

def transport(X, chain, g, k):
    """Перенос спинора вдоль цепи: на каждой смене ранжировки — транспозиция."""
    d = g[0].shape[0]
    M = np.eye(d, dtype=complex)
    perms = [tuple(np.argsort(X[c])) for c in chain]
    flips = 0
    for t in range(1, len(perms)):
        a, b = perms[t-1], perms[t]
        diff = [i for i in range(k) if a[i] != b[i]]
        if len(diff) == 2:
            M = M @ transposition_op(g, diff[0], diff[1]); flips += 1
        elif len(diff) > 2:
            # раскладываем в транспозиции
            pa = list(a)
            for i in range(k):
                if pa[i] != b[i]:
                    j = pa.index(b[i])
                    M = M @ transposition_op(g, i, j); flips += 1
                    pa[i], pa[j] = pa[j], pa[i]
    return M, flips
