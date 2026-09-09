"""Два порядка на одних событиях.

У события k+1 шкал. Порядок A строится по шкалам 0..k-1,
порядок B — по шкалам 1..k. Они делят события, но различаются.

Вопрос: возникает ли что-то из их взаимодействия.
"""
import numpy as np


def order_from(X, cols):
    N = len(X)
    R = np.ones((N, N), dtype=bool)
    for m in cols:
        R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)


def links_of(R):
    A = np.asarray(R, dtype=np.float32)
    return np.asarray(R & ((A @ A) == 0), dtype=bool)


def compare(RA, RB):
    """Как соотносятся два порядка."""
    both = int((RA & RB).sum())
    onlyA = int((RA & ~RB).sum())
    onlyB = int((RB & ~RA).sum())
    # противоречие: A говорит i позже j, B говорит наоборот
    conflict = int((RA & RB.T).sum())
    return dict(both=both, onlyA=onlyA, onlyB=onlyB, conflict=conflict)
