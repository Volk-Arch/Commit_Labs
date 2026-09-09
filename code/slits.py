import numpy as np
import sys; sys.path.insert(0, '/home/claude')
from compute import rel_of, links, topo_order

def build(N, k, rs):
    X = rs.random((N, k))
    R = rel_of(X); L = links(R)
    return X, R, np.asarray(L, dtype=bool), topo_order(L)

def propagate(Lb, order, A, sources):
    """Амплитуда от источников по причинным связям."""
    N = len(Lb)
    V = np.zeros(N, dtype=complex)
    for v in order:
        if v in sources:
            V[v] = sources[v]; continue
        par = np.flatnonzero(Lb[v])
        if len(par): V[v] = A[v]*V[par].sum()/len(par)
    return V
