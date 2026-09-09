"""Безразмерная обрезка: по ДОЛЕ, а не по абсолютной длине.

Обрезка по dx < 0.30 значит разное в кубе и в интервале — она зависит
от формы и масштаба мира. Обрезка «оставить самые короткие q связей»
одинакова везде.
"""
import numpy as np


def build_relcut(X, keep, k=None):
    """keep — доля прямых связей, которые оставляем (по длине).
    keep=1.0 — без обрезки."""
    N, kk = X.shape
    if k is None:
        k = kk
    R0 = np.ones((N, N), dtype=bool)
    for m in range(k):
        R0 &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R0, False)
    A = R0.astype(np.float32)
    L0 = R0 & ((A @ A) == 0)
    del A, R0
    if keep < 1.0:
        dx = np.linalg.norm(X[None, :, 1:] - X[:, None, 1:], axis=-1)
        ln = dx[L0]
        if len(ln):
            thr = np.quantile(ln, keep)
            L0 = L0 & (dx < thr)
        del dx
    R = np.asarray(L0, dtype=bool)
    for _ in range(60):
        Rn = R | ((R.astype(np.float32) @ R.astype(np.float32)) > 0)
        if Rn.sum() == R.sum():
            break
        R = Rn
    np.fill_diagonal(R, False)
    A = R.astype(np.float32)
    L = np.asarray(R & ((A @ A) == 0), dtype=bool)
    del A
    return np.asarray(R, dtype=bool), L
