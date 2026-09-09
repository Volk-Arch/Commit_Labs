"""Замкнутая петля из уже проверенных кусков.

1. плотность -> метрика       (число активных шкал падает с густотой)
2. метрика   -> кто выживает  (в искривлённых местах гибель иная)
3. гибель    -> плотность     (круг замыкается)
"""
import numpy as np


def rel_weighted(X, W):
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    act = W > 0
    for m in range(k):
        both = act[:, m][:, None] & act[:, m][None, :]
        R &= ((X[None, :, m] < X[:, None, m]) | ~both)
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)


def local_density(X, radius=0.15):
    N = len(X); d = np.zeros(N)
    for i in range(N):
        d[i] = int((np.linalg.norm(X - X[i], axis=1) < radius).sum()) - 1
    return d


def metric_from_density(dens, k, alpha):
    """1. Плотность -> метрика: где гуще, там меньше активных шкал."""
    N = len(dens)
    W = np.ones((N, k))
    if alpha <= 0: return W
    d = dens/max(dens.mean(), 1e-9)
    off = np.clip((alpha*(d-1.0)).astype(int), 0, k-2)
    for m in range(k):
        W[off > (k-1-m), m] = 0.0
    return W


def step_closed(X, k, rs, alpha=1.0, rate=0.02, radius=0.15):
    """Полный шаг петли."""
    N = len(X)
    dens = local_density(X, radius)
    W = metric_from_density(dens, k, alpha)          # 1
    R = rel_weighted(X, W)                            # структура
    # 2. метрика -> гибель: гибнут там, где мало активных шкал
    #    (в искривлённых местах структура нестабильна)
    active = W.sum(axis=1)
    w = 1.0/(1.0 + dens/max(dens.mean(), 1e-9))       # разреженные гибнут чаще
    w = w * (1.0 + (k - active))                      # искривление усиливает
    p = w/w.sum()
    n = max(1, int(rate*N))
    die = rs.choice(N, n, replace=False, p=p)
    keep = np.ones(N, bool); keep[die] = False
    Xn = np.vstack([X[keep], rs.random((n, X.shape[1]))])   # 3
    return Xn, R, W
