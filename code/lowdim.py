import numpy as np

def lin_ext(R, weights, rs):
    Rb = np.ascontiguousarray(R, dtype=bool); n = len(Rb)
    indeg = Rb.sum(axis=1).astype(np.int64)
    avail = list(np.flatnonzero(indeg == 0)); out = []
    while avail:
        w = weights[np.array(avail)]
        t = int(np.argmax(w + 1e-6*rs.random(len(avail))))
        v = avail.pop(t); out.append(v)
        kids = np.flatnonzero(Rb[:, v])
        if kids.size:
            indeg[kids] -= 1
            avail.extend([int(c) for c in kids[indeg[kids] == 0]])
    pos = np.empty(n, dtype=np.int64); pos[np.array(out)] = np.arange(n)
    return pos

def order_dim_upper(R, rs, cap=40):
    """Сколько шкал нужно, чтобы восстановить порядок."""
    N = len(R); Rb = np.asarray(R, dtype=bool)
    cur = np.ones((N, N), dtype=bool); np.fill_diagonal(cur, False)
    for step in range(cap):
        bad = cur & ~Rb.T
        if not bad.any(): return step
        w = bad.sum(axis=0).astype(float) - bad.sum(axis=1).astype(float)
        p = lin_ext(R, w, rs)
        cc = (p[None, :] > p[:, None]); np.fill_diagonal(cc, False)
        cur = cur & cc
    return cap+1

def grow_lowdim(N, rs, nscales=4, burn=0.0):
    """Рост, где новое событие получает ШКАЛЬНЫЕ координаты:
    оно наследует координаты предков и добавляет свои.
    Малоразмерность встроена в правило."""
    X = np.zeros((N, nscales))
    X[0] = rs.random(nscales)
    R = np.zeros((N, N), dtype=bool)
    for n in range(1, N):
        # новое событие: координаты БОЛЬШЕ чем у выбранных предков
        cand = np.arange(n)
        m = max(1, min(3, n))
        sel = rs.choice(cand, m, replace=False)
        lo = X[sel].max(axis=0)
        X[n] = lo + rs.random(nscales)*(1.0-lo)*0.5 + 1e-6
        R[:n, n] = np.all(X[:n] < X[n], axis=1)
    return R, X
