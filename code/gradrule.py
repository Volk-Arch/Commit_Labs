"""Правило, чувствительное к ГРАДИЕНТУ плотности.

Прежнее правило насыщается внутри тела: плотность там ~100 при пороге ~2,
и давление гибели одинаково по всему телу. Здесь давление зависит от того,
где плотность выше — с какой стороны события.
"""
import numpy as np
import sys
import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from deathrules import local_density


def density_gradient(X, radius=0.15, chunk=256):
    """Вектор градиента плотности в каждом событии: куда плотность растёт."""
    N, k = X.shape
    g = np.zeros((N, k))
    r2 = radius*radius
    sq = (X*X).sum(1)
    for a in range(0, N, chunk):
        b = min(a+chunk, N)
        d2 = sq[a:b, None] + sq[None, :] - 2.0*(X[a:b] @ X.T)
        near = d2 < r2
        for i in range(b-a):
            m = near[i]
            m[a+i] = False
            if m.sum():
                g[a+i] = (X[m] - X[a+i]).mean(0)
    return g


def step_gradient(X, rate, rs, beta=1.0, radius=0.15, lo=0.7, hi=1.5):
    """Гибель зависит и от уровня плотности, и от её ГРАДИЕНТА.

    beta — вес градиентной части. beta=0 сводится к прежнему правилу.
    """
    N, k = X.shape
    n = max(1, int(rate*N))
    dens = local_density(X, radius)
    rel = dens/max(dens.mean(), 1e-9)
    w = np.where((rel < lo) | (rel > hi), 1.0, 0.05)
    if beta > 0:
        g = density_gradient(X, radius)
        # событие на «спуске» плотности гибнет охотнее
        gm = np.linalg.norm(g, axis=1)
        w = w*(1.0 + beta*gm/max(gm.mean(), 1e-12))
    die = rs.choice(N, n, replace=False, p=w/w.sum())
    keep = np.ones(N, bool); keep[die] = False
    return np.vstack([X[keep], rs.random((n, k))]), keep, die
