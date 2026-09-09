import numpy as np

def rel_weighted(X, W):
    """Правило зависит от МЕСТА: у каждого события свои веса шкал.
    Событие i раньше j, если раньше по всем шкалам, где ОБА события активны.
    W[i,m] > 0 значит шкала m работает у события i."""
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    act = W > 0
    for m in range(k):
        both = act[:, m][:, None] & act[:, m][None, :]
        R &= ((X[None, :, m] < X[:, None, m]) | ~both)
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)

def metric_field(X, centre, radius, k, drop=1):
    """Поле метрики: около центра шкал МЕНЬШЕ — пространство 'сжато'."""
    N = len(X)
    W = np.ones((N, k))
    r = np.linalg.norm(X - centre, axis=1)
    near = r < radius
    for m in range(k-drop, k):
        W[near, m] = 0.0
    return W, near
