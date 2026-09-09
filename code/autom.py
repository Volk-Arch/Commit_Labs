"""Автоморфизмы каузального алмаза.

Идея: калибровочные группы не вводятся, а возникают как группы симметрий
типичного локального участка — интервала между двумя событиями.
"""
import numpy as np
from itertools import permutations


def interval_of(R, i, j):
    return np.flatnonzero(R[:, j] & R[i, :])


def sub_order(R, idx):
    return np.asarray(R[np.ix_(idx, idx)], dtype=bool)


def automorphisms(S, cap=200000):
    """Все перестановки, сохраняющие порядок. Только для малых S."""
    n = len(S)
    if n > 8:
        return None
    cnt = 0
    for p in permutations(range(n)):
        P = np.array(p)
        if np.array_equal(S[np.ix_(P, P)], S):
            cnt += 1
        if cnt > cap:
            break
    return cnt


def profile_key(S):
    """Отпечаток порядка: отсортированные (прошлое, будущее) каждого элемента."""
    past = S.sum(1); fut = S.sum(0)
    return tuple(sorted(zip(past.tolist(), fut.tolist())))
