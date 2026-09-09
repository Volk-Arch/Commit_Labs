"""Сглаженное действие Бенинказы–Дауке.

Локальные коэффициенты 1, -9, 16, -8 фиксированы требованием
континуального предела. Но нелокальная версия имеет свободный
параметр — масштаб сглаживания eps (промежуточный масштаб между
дискретностью и кривизной).

При eps = 1 сводится к локальной версии.
"""
import numpy as np


def between_counts(R, nmax=200):
    """Для каждой связанной пары — сколько элементов строго между."""
    C = np.asarray(R, dtype=np.float32)
    btw = (C @ C).astype(np.int64)
    Rb = np.asarray(R, dtype=bool)
    n = btw[Rb]
    return np.bincount(n, minlength=nmax+1)[:nmax+1]


def f4(n, eps):
    """Весовая функция сглаженного действия в 4 измерениях."""
    n = np.asarray(n, dtype=float)
    a = 1.0 - eps
    if a <= 0:
        w = np.zeros_like(n); w[n == 0] = 1.0
        return w
    base = a**n
    t1 = 9.0*eps*n/a
    t2 = 8.0*eps*eps*n*(n-1)/(a*a)
    t3 = (4.0/3.0)*eps**3*n*(n-1)*(n-2)/(a**3)
    return base*(1.0 - t1 + t2 - t3)


def smeared_action(R, eps, nmax=200):
    """S(eps). При eps=1 совпадает с локальным действием."""
    N = len(R)
    cnt = between_counts(R, nmax)
    n = np.arange(len(cnt))
    return float(eps*N - eps*eps*(cnt*f4(n, eps)).sum())
