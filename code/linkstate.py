"""Состояние на СВЯЗЯХ, а не только на событиях.

У Райдаута–Соркина динамика выражается через изинговы спины,
живущие на отношениях каузального множества. Здесь то же:
каждая прямая связь несёт свою переменную.
"""
import numpy as np


def rel_of(X):
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k): R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)


def links_of(R):
    A = R.astype(np.float32)
    return R & ((A @ A) == 0)


def link_state(X, L, rule='chirality'):
    """Состояние связи из свойств её концов."""
    Lb = np.asarray(L, dtype=bool)
    ij = np.argwhere(Lb)                 # [i,j]: j раньше i
    if rule == 'chirality':
        # знак: чётность перестановки приращений
        s = []
        for a, b in ij:
            d = X[a]-X[b]
            p = np.argsort(d)
            inv = sum(1 for x in range(len(p)) for y in range(x+1, len(p)) if p[x] > p[y])
            s.append(1 if inv % 2 == 0 else -1)
        return ij, np.array(s, dtype=float)
    if rule == 'phase':
        return ij, np.array([float((X[a]-X[b]).sum()) for a, b in ij])
    if rule == 'spread':
        return ij, np.array([float((X[a]-X[b]).var()) for a, b in ij])
    raise ValueError(rule)


def node_from_links(ij, s, N, mode='sum'):
    """Величина события, собранная из состояний его связей.
    Это и есть 'поле' — посредник между событиями."""
    out = np.zeros(N)
    if mode == 'sum':
        np.add.at(out, ij[:, 0], s)
        np.add.at(out, ij[:, 1], -s)      # входящие с минусом: как поток
    elif mode == 'abs':
        np.add.at(out, ij[:, 0], np.abs(s))
        np.add.at(out, ij[:, 1], np.abs(s))
    return out
