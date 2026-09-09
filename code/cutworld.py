"""Мир с обрезанными связями: есть внутренность у областей.

Режем ПРЯМЫЕ связи по пространственной длине, затем восстанавливаем
транзитивность замыканием — иначе это не частичный порядок.
"""
import numpy as np


def build_cut(P, cut, max_iter=60):
    N = len(P)
    dt = P[None, :, 0] - P[:, None, 0]
    dx2 = ((P[None, :, 1:] - P[:, None, 1:])**2).sum(-1)
    R0 = (dt > 0) & (dt*dt > dx2)
    A = R0.astype(np.float32)
    L0 = R0 & ((A @ A) == 0)
    del A
    if cut is not None:
        L0 = L0 & (np.sqrt(dx2) < cut)
    del dt, dx2, R0
    R = np.asarray(L0, dtype=bool)
    for _ in range(max_iter):
        Rn = R | ((R.astype(np.float32) @ R.astype(np.float32)) > 0)
        if Rn.sum() == R.sum():
            break
        R = Rn
    np.fill_diagonal(R, False)
    A = R.astype(np.float32)
    L = np.asarray(R & ((A @ A) == 0), dtype=bool)
    del A
    return np.asarray(R, dtype=bool), L


def interior(L, mask):
    """События области, у которых НЕТ прямых связей наружу."""
    out = ~mask
    touch = (L[np.ix_(mask, out)] | L[np.ix_(out, mask)].T).any(axis=1)
    return int(mask.sum()) - int(touch.sum())
