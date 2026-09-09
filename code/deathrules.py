import numpy as np


def rel_of(X):
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k): R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)


def local_density(X, radius=0.15):
    N = len(X); d = np.zeros(N)
    for i in range(N):
        d[i] = int((np.linalg.norm(X - X[i], axis=1) < radius).sum()) - 1
    return d


def step(X, rate, rs, rule, radius=0.15):
    """Рождение свободное, гибель по заданному правилу."""
    N, k = X.shape
    n = max(1, int(rate*N))
    if rule == 'случайно':
        w = np.ones(N)
    else:
        R = rel_of(X)
        deg = (R.sum(1) + R.sum(0)).astype(float)
        if rule == 'плотные':
            w = local_density(X, radius) + 1.0
        elif rule == 'связные':
            w = deg + 1.0
        elif rule == 'разреженные':
            w = 1.0/(local_density(X, radius) + 1.0)
        elif rule == 'старые':
            w = R.sum(1).astype(float) + 1.0
        elif rule == 'молодые':
            w = 1.0/(R.sum(1).astype(float) + 1.0)
        del R
    p = w/w.sum()
    die = rs.choice(N, n, replace=False, p=p)
    keep = np.ones(N, bool); keep[die] = False
    return np.vstack([X[keep], rs.random((n, k))])
