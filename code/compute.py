import numpy as np

def rel_of(X):
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k): R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return R

def links(R):
    Rb = np.ascontiguousarray(np.asarray(R, dtype=bool))
    A = Rb.astype(np.float32)
    return Rb & ((A@A) == 0)

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

def evaluate(L, inputs, gate, order):
    """Событие вычисляет функцию от значений прямых предков."""
    N = len(L); val = np.zeros(N, dtype=np.int8)
    Lb = np.asarray(L, dtype=bool)
    for v in order:
        par = np.flatnonzero(Lb[v])
        if len(par) == 0:
            val[v] = inputs.get(v, 0)
        else:
            val[v] = gate(val[par])
    return val

def topo_order(L):
    Lb = np.ascontiguousarray(np.asarray(L, dtype=bool)); n = len(Lb)
    indeg = Lb.sum(axis=1).astype(np.int64)
    ready = list(np.flatnonzero(indeg == 0)); out = []
    while ready:
        j = ready.pop(); out.append(j)
        kids = np.flatnonzero(Lb[:, j])
        if kids.size:
            indeg[kids] -= 1
            ready.extend(kids[indeg[kids] == 0].tolist())
    return out
