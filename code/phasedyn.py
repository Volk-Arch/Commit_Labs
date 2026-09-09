import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/divergent.py').read())

def rel_of(X):
    """Порядок из позиций в фазовом пространстве."""
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k):
        R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return R

def phase_step(X, rate=0.05, mode='past', rs=None, noise=0.0):
    """Фазы события смещаются под действием фаз его СОСЕДЕЙ ПО ПОРЯДКУ.
    mode='past'  — тянут предки
    mode='both'  — тянут и предки, и потомки
    mode='push'  — ОТТАЛКИВАЮТ соседи"""
    r = rs if rs is not None else np.random.default_rng(0)
    R = rel_of(X)
    Rf = R.astype(np.float64)
    npast = np.maximum(Rf.sum(axis=1), 1)[:, None]
    nfut = np.maximum(Rf.sum(axis=0), 1)[:, None]
    cpast = (Rf @ X)/npast
    cfut = (Rf.T @ X)/nfut
    if mode == 'past':   target = cpast
    elif mode == 'both': target = 0.5*(cpast + cfut)
    else:                target = 2*X - 0.5*(cpast + cfut)
    Xn = X + rate*(target - X)
    if noise: Xn = Xn + r.normal(0, noise, X.shape)
    return np.clip(Xn, 0, 1), R
