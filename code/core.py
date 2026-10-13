"""Минимальное ядро конструкции (восстановлено).

Правило: событию даны k чисел. A раньше B, если A раньше по ВСЕМ k числам.
"""
import numpy as np
import math
from scipy.optimize import brentq


def rel_of(X):
    """Частичный порядок из координат."""
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k):
        R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)


def links_of(R):
    A = np.asarray(R, dtype=np.float32)
    return np.asarray(R & ((A @ A) == 0), dtype=bool)


def frac_pairs(R):
    N = len(R)
    return float(R.sum()/(N*(N-1)/2))


def mm_dim(f):
    """Размерность Мирхейма–Мейера из доли связанных пар."""
    g = lambda d: math.gamma(d+1)*math.gamma(d/2)/(2*math.gamma(1.5*d))
    try:
        return brentq(lambda d: g(d)-f, 1.0, 40)
    except ValueError:
        return float('nan')


def dimension(X=None, R=None):
    if R is None:
        R = rel_of(X)
    f = frac_pairs(R)
    return mm_dim(f) if 0 < f < 1 else float('nan')


def height(R):
    L = links_of(R)
    N = len(R)
    ind = L.sum(axis=1).astype(np.int64)
    D = np.zeros(N, dtype=np.int64)
    rd = list(np.flatnonzero(ind == 0))
    while rd:
        j = rd.pop(); kd = np.flatnonzero(L[:, j])
        if kd.size:
            D[kd] = np.maximum(D[kd], D[j]+1); ind[kd] -= 1
            rd.extend(kd[ind[kd] == 0].tolist())
    return int(D.max())+1, D


def spectral(R):
    """Спектральный различитель: отличие от случайного графа той же плотности."""
    S = (np.asarray(R) | np.asarray(R).T).astype(np.float64)
    n = len(S)
    p = float((np.linalg.eigvalsh(S) > 0).mean())
    f = float(S.sum()/(n*(n-1)))
    rs = np.random.default_rng(9)
    G = (rs.random(S.shape) < f); G = np.triu(G, 1); G = G | G.T
    return p - float((np.linalg.eigvalsh(G.astype(np.float64)) > 0).mean())


def interval_counts(R, kmax=4):
    C = np.asarray(R, dtype=np.float32)
    btw = (C @ C).astype(np.int32)
    Rb = np.asarray(R, dtype=bool)
    return [int(((btw == m) & Rb).sum()) for m in range(kmax)]


def bd_action(R, d=4):
    """Действие Бенинказы–Дауке."""
    N = len(R)
    n = interval_counts(R, 4)
    return N - n[0] + 9*n[1] - 16*n[2] + 8*n[3]


def transitivity_violations(R):
    A = np.asarray(R, dtype=np.float32)
    return int((((A @ A) > 0) & ~np.asarray(R, dtype=bool)).sum())
