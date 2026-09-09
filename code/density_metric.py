import numpy as np

def local_density(P, radius=0.12):
    """Плотность событий вокруг каждого: сколько соседей в шаре."""
    N = len(P)
    d = np.zeros(N)
    for i in range(N):
        r = np.linalg.norm(P - P[i], axis=1)
        d[i] = int((r < radius).sum()) - 1
    return d

def effective_scales(R, idx):
    """Сколько шкал эффективно работает в подмножестве:
    из доли связанных пар обратной формулой k = 1 - log2(f)."""
    n = len(idx)
    if n < 20: return np.nan
    sub = R[np.ix_(idx, idx)]
    f = sub.sum()/(n*(n-1)/2)
    if f <= 0 or f >= 1: return np.nan
    return 1 - np.log2(f)
