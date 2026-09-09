import numpy as np

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

def rel_masked(X, active):
    """Порядок при выборочно активных шкалах: сравниваем только по общим."""
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k):
        both = active[:, m][:, None] & active[:, m][None, :]
        R &= ((X[None, :, m] < X[:, None, m]) | ~both)
    np.fill_diagonal(R, False)
    return R

def evolve(X, iters=12, target=None, rs=None):
    """Шкалы ОТМИРАЮТ: на каждом шаге у события отключается та шкала,
    которая меньше всего влияет на его связи."""
    N, k = X.shape
    active = np.ones((N, k), dtype=bool)
    hist = []
    for it in range(iters):
        R = rel_masked(X, active)
        Rb = np.asarray(R, dtype=bool)
        f = Rb.sum()/(N*(N-1)/2)
        D = pt(np.asarray(R, dtype=np.float32))
        hist.append((float(active.sum(1).mean()), float(f), int(D.max())))
        # для каждого события: какая шкала даёт меньше всего связей
        for v in range(N):
            act = np.flatnonzero(active[v])
            if len(act) <= 2: continue
            # вклад шкалы: сколько связей она отсекает
            gain = []
            for m in act:
                sup = (X[:, m] < X[v, m]) & active[:, m]
                gain.append(sup.sum())
            drop = act[int(np.argmin(gain))]
            if rs is None or rs.random() < 0.3:
                active[v, drop] = False
    return active, hist
