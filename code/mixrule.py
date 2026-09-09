"""Смешанные правила гибели.

'плотные'    — аттрактор есть, структура плохая (-1.43)
'разреженные'— структура хорошая (-0.95), аттрактора нет

Пробуем смесь: часть событий гибнет по одному правилу, часть по другому.
"""
import numpy as np
import sys
sys.path.insert(0, '/home/claude')
from deathrules import rel_of, local_density


def step_mix(X, rate, rs, w, radius=0.15):
    """w — доля гибнущих по правилу 'плотные' (остальные — 'разреженные').

    w=0 -> чистые 'разреженные', w=1 -> чистые 'плотные'.
    """
    N, k = X.shape
    n = max(1, int(rate*N))
    dens = local_density(X, radius)
    rel = dens/max(dens.mean(), 1e-9)
    n_dense = int(round(w*n))
    n_sparse = n - n_dense
    die = []
    if n_dense > 0:
        p = rel + 1.0
        die.extend(rs.choice(N, n_dense, replace=False, p=p/p.sum()).tolist())
    if n_sparse > 0:
        p = 1.0/(1.0 + rel)
        avail = np.setdiff1d(np.arange(N), np.array(die, dtype=int))
        pp = p[avail]
        die.extend(rs.choice(avail, n_sparse, replace=False,
                             p=pp/pp.sum()).tolist())
    keep = np.ones(N, bool); keep[np.array(die, dtype=int)] = False
    return np.vstack([X[keep], rs.random((n, k))])


def step_thresh(X, rate, rs, lo, hi, radius=0.15):
    """Гибнут те, чья плотность ВНЕ окна [lo, hi] от средней.

    То есть и слишком густые, и слишком пустые — а середина живёт.
    """
    N, k = X.shape
    n = max(1, int(rate*N))
    dens = local_density(X, radius)
    rel = dens/max(dens.mean(), 1e-9)
    w = np.where((rel < lo) | (rel > hi), 1.0, 0.05)
    p = w/w.sum()
    die = rs.choice(N, n, replace=False, p=p)
    keep = np.ones(N, bool); keep[die] = False
    return np.vstack([X[keep], rs.random((n, k))])
