"""Избыток объёма: отклонение от плоского соотношения n ~ h^d.

В плоском пространстве объём интервала связан с его высотой жёстко.
Отклонение — мера того, насколько геометрия не плоская.
"""
import numpy as np


def interval_stats(R, L, mid):
    n = len(mid)
    if n < 8: return None
    sub = np.asarray(R[np.ix_(mid, mid)], dtype=bool)
    subL = L[np.ix_(mid, mid)]
    tot = float(sub.sum())
    if tot == 0: return None
    dep = np.zeros(n)
    ind = sub.sum(axis=1).astype(np.int64)
    rd = list(np.flatnonzero(ind == 0))
    while rd:
        a = rd.pop(); kd = np.flatnonzero(sub[:, a])
        if kd.size:
            dep[kd] = np.maximum(dep[kd], dep[a]+1); ind[kd] -= 1
            rd.extend(kd[ind[kd] == 0].tolist())
    h = int(dep.max())+1
    return n, h, float(subL.sum()/tot)


def excess(n, h, alpha, beta):
    """Избыток объёма над плоским законом n = alpha * h^beta."""
    return np.log(np.maximum(n, 1)) - (np.log(alpha) + beta*np.log(np.maximum(h, 1)))
