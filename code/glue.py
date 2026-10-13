"""Склейка кусков: сохраняется ли когерентность.

Малые интервалы (3-5 событий) складываются когерентно. Вопрос:
если длинную историю собрать из цепочки таких кусков,
останется ли когерентность.
"""
import numpy as np
import sys
sys.path.insert(0, '/home/claude')
from bdaction import bd_action


def chain_of_intervals(R, start, rs, n_steps, vmin=3, vmax=6, tries=400):
    """Цепочка последовательных интервалов: конец одного = начало следующего."""
    N = len(R)
    cur = start
    pieces = []
    for _ in range(n_steps):
        fut = np.flatnonzero(R[:, cur])
        if len(fut) == 0:
            break
        found = None
        for _ in range(tries):
            nxt = int(rs.choice(fut))
            mid = np.flatnonzero(R[:, cur] & R[nxt, :])
            if vmin <= len(mid) <= vmax:
                found = (nxt, mid)
                break
        if found is None:
            break
        nxt, mid = found
        pieces.append(mid)
        cur = nxt
    return pieces, cur


def piece_action(R, mid):
    sub = np.asarray(R[np.ix_(mid, mid)], dtype=bool)
    return bd_action(sub, 4)
