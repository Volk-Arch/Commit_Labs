"""Мера близости на частичных порядках.

Два мира близки, если совпадают их ВНУТРЕННИЕ характеристики —
те, что вычисляются из порядка, без координат.
"""
import numpy as np


def profile(R, nmax=6):
    """Отпечаток порядка: доли пар с k событиями между ними."""
    N = len(R)
    A = np.asarray(R, dtype=np.float32)
    btw = (A @ A).astype(np.int64)
    Rb = np.asarray(R, dtype=bool)
    n = btw[Rb]
    cnt = np.bincount(n, minlength=nmax+1)[:nmax+1].astype(float)
    tot = max(Rb.sum(), 1)
    return cnt/tot


def descriptor(R):
    """Полный внутренний дескриптор мира."""
    N = len(R)
    Rb = np.asarray(R, dtype=bool)
    A = Rb.astype(np.float32)
    L = Rb & ((A @ A) == 0)
    f = float(Rb.sum()/(N*(N-1)/2))
    # высота
    ind = L.sum(axis=1).astype(np.int64)
    D = np.zeros(N, dtype=np.int64)
    rd = list(np.flatnonzero(ind == 0))
    while rd:
        j = rd.pop(); kd = np.flatnonzero(L[:, j])
        if kd.size:
            D[kd] = np.maximum(D[kd], D[j]+1); ind[kd] -= 1
            rd.extend(kd[ind[kd] == 0].tolist())
    h = float(D.max()+1)
    deg = (Rb.sum(1) + Rb.sum(0)).astype(float)
    return np.concatenate([
        [f, h/N, float(L.sum()/max(Rb.sum(), 1)),
         float(deg.std()/max(deg.mean(), 1e-9))],
        profile(Rb)])


def distance(d1, d2, w=None):
    """Расстояние между дескрипторами."""
    if w is None:
        w = np.ones(len(d1))
    return float(np.sqrt((w*(d1-d2)**2).sum()))
