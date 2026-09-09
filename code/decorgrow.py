import numpy as np

def grow_decor(N, rs, k=4, step=0.05):
    """Рост с декорреляцией: каждая шкала наследуется от СВОЕГО предка."""
    X = np.zeros((N, k))
    for n in range(1, N):
        for m in range(k):
            q = int(rs.integers(0, n))
            X[n, m] = X[q, m] + rs.random()*step
    R = np.ones((N, N), dtype=bool)
    for m in range(k): R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool), X

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
