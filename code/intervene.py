import numpy as np


def add_events(X, target, count, rs, spread=0.02):
    """Вмешательство: добавляем события ВОЗЛЕ заданных (увеличиваем 'массу')."""
    base = X[target]
    new = base[rs.integers(0, len(base), count)] + rs.normal(0, spread, (count, X.shape[1]))
    return np.clip(new, 0, 1)


def rel_of(X):
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k): R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)


def frac_links(R, idx):
    if len(idx) < 15: return np.nan
    A = R.astype(np.float32)
    L = R & ((A @ A) == 0)
    sub = R[np.ix_(idx, idx)]; subL = L[np.ix_(idx, idx)]
    t = float(sub.sum())
    return float(subL.sum()/t) if t > 0 else np.nan
