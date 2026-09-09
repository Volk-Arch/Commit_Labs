import numpy as np

def rel_of(X):
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k): R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return R

def interval(Rb, i, j):
    """Все события строго между i и j."""
    return np.flatnonzero(Rb[i] & Rb[:, j])

def interval_frame(X, i, j, mid):
    """Ориентация интервала: главная ось разброса внутренности
    относительно оси i->j. У интервала ЕСТЬ внутренность — в отличие от цепи."""
    if len(mid) < 4: return None
    axis = X[j] - X[i]
    n = np.linalg.norm(axis)
    if n < 1e-12: return None
    e1 = axis/n
    P = X[mid] - 0.5*(X[i]+X[j])
    P = P - np.outer(P @ e1, e1)          # убираем продольную часть
    if np.linalg.norm(P) < 1e-12: return None
    C = P.T @ P
    w, v = np.linalg.eigh(C)
    e2 = v[:, -1]                          # главная поперечная ось
    return e1, e2, w[-1]/max(w.sum(), 1e-12)
