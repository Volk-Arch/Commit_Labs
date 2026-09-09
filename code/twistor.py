import numpy as np

def rel_of(X):
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k): R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return R

def links_of(R):
    A = np.asarray(R, dtype=np.float32)
    return np.asarray(R, dtype=bool) & ((A@A) == 0)

def link_list(L):
    """Связи как объекты: пары (предок, потомок)."""
    return np.argwhere(np.asarray(L, dtype=bool))   # [j,i]: i -> j

def incidence(pairs, N):
    """Две связи ИНЦИДЕНТНЫ, если делят событие.
    Это и есть 'пересечение лучей' по Пенроузу."""
    M = len(pairs)
    a = pairs[:, 1]; b = pairs[:, 0]     # i -> j
    # общий конец: конец одной = начало другой, или общее начало/конец
    same_head = (b[:, None] == a[None, :])          # j одной = i другой (цепочка)
    same_start = (a[:, None] == a[None, :])
    same_end = (b[:, None] == b[None, :])
    return same_head, same_start, same_end
