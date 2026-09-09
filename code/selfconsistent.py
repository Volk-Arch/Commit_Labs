"""Замкнутая петля: плотность событий -> метрика -> распределение -> плотность.

У Эйнштейна материя говорит пространству как искривляться, пространство говорит
материи как двигаться. Здесь та же петля в дискретном виде.
"""
import numpy as np


def density_field(X, centres, radius=0.15):
    """Локальная плотность событий вокруг заданных точек."""
    d = np.zeros(len(centres))
    for i, c in enumerate(centres):
        d[i] = int((np.linalg.norm(X - c, axis=1) < radius).sum())
    return d


def active_from_density(X, dens_at, k, alpha, base_r=0.15):
    """МЕТРИКА из плотности: где гуще — там меньше активных шкал.
    Это и есть 'материя говорит пространству как искривляться'."""
    N = len(X)
    W = np.ones((N, k))
    if alpha <= 0: return W
    # нормируем плотность
    d = dens_at/max(dens_at.mean(), 1e-9)
    # число выключенных шкал растёт с плотностью
    off = np.clip((alpha*(d-1.0)).astype(int), 0, k-2)
    for m in range(k):
        W[off > (k-1-m), m] = 0.0
    return W


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
    N = len(X)
    d = np.zeros(N)
    for i in range(N):
        d[i] = int((np.linalg.norm(X - X[i], axis=1) < radius).sum()) - 1
    return d
