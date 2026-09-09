"""Масса без границы: суммирование с убывающим весом.

У электрона нет размера, но масса определена. Значит масса — не число
элементов внутри границы, а интеграл по всему полю с весом.
"""
import numpy as np


def soft_mass(X, centre, value, kernel='gauss', scale=0.15):
    """Масса вокруг точки: сумма величины с весом, убывающим от центра.
    Границы нет — вклад дают все события."""
    d = np.linalg.norm(X - centre, axis=1)
    if kernel == 'gauss':
        w = np.exp(-(d/scale)**2)
    elif kernel == 'exp':
        w = np.exp(-d/scale)
    elif kernel == 'cauchy':
        w = 1.0/(1.0 + (d/scale)**2)
    elif kernel == 'step':
        w = (d < scale).astype(float)
    else:
        raise ValueError(kernel)
    return float((w*value).sum()), float(w.sum())


def local_curvature(R, L, X, centre, scale=0.3):
    """Кривизна вокруг точки: доля прямых связей с весом."""
    d = np.linalg.norm(X - centre, axis=1)
    idx = np.flatnonzero(d < scale)
    if len(idx) < 20: return np.nan
    sub = R[np.ix_(idx, idx)]; subL = L[np.ix_(idx, idx)]
    t = float(sub.sum())
    return float(subL.sum()/t) if t > 0 else np.nan
