"""Полная петля: структура влияет и на гибель, И на рождение."""
import numpy as np
import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from closed import rel_weighted, local_density, metric_from_density


def step_full(X, k, rs, alpha=1.5, rate=0.02, radius=0.15, birth='free', beta=1.0):
    N = len(X)
    dens = local_density(X, radius)
    W = metric_from_density(dens, k, alpha)
    R = rel_weighted(X, W)
    active = W.sum(axis=1)
    # ГИБЕЛЬ: разреженные и искривлённые
    wd = 1.0/(1.0 + dens/max(dens.mean(), 1e-9))
    wd = wd*(1.0 + (k - active))
    pd = wd/wd.sum()
    n = max(1, int(rate*N))
    die = rs.choice(N, n, replace=False, p=pd)
    keep = np.ones(N, bool); keep[die] = False
    Xk = X[keep]
    # РОЖДЕНИЕ
    if birth == 'free':
        new = rs.random((n, X.shape[1]))
    else:
        # выбираем МЕСТО по состоянию мира
        cand = rs.random((n*20, X.shape[1]))
        dd = np.array([int((np.linalg.norm(Xk - c, axis=1) < radius).sum()) for c in cand])
        m = dd/max(dd.mean(), 1e-9)
        if birth == 'dense':      # рождаются, где густо
            w = m**beta
        elif birth == 'sparse':   # где редко
            w = 1.0/(1.0 + m)**beta
        elif birth == 'curved':   # где метрика искривлена
            act = np.array([metric_from_density(np.array([d]), k, alpha).sum()
                            for d in dd])
            w = (1.0 + (k - act))**beta
        else:
            w = np.ones(len(cand))
        if w.sum() <= 0: w = np.ones(len(cand))
        idx = rs.choice(len(cand), n, replace=False, p=w/w.sum())
        new = cand[idx]
    return np.vstack([Xk, new]), R, W
