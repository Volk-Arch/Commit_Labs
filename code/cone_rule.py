import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
from math import gamma
exec(open(os.path.join(_HERE, 'compare_f.py'), encoding='utf-8').read())

def rel_pyramid(X):
    """Наше правило: раньше по всем шкалам. Даёт ПИРАМИДАЛЬНЫЙ конус."""
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k):
        R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return R

def rel_cone(X):
    """Круглый конус: первая координата — время, остальные — пространство.
    i < j, если dt > |dx| (евклидова норма)."""
    N, k = X.shape
    dt = X[None, :, 0] - X[:, None, 0]
    dx = np.sqrt(((X[None, :, 1:] - X[:, None, 1:])**2).sum(-1))
    R = (dt > 0) & (dt > dx)
    np.fill_diagonal(R, False)
    return R

def rel_lp(X, p):
    """Промежуточное: dt > (сумма |dx|^p)^(1/p).
    p=1 — октаэдр (сумма модулей), p=2 — круглый конус, p=inf — куб/пирамида."""
    N, k = X.shape
    dt = X[None, :, 0] - X[:, None, 0]
    D = np.abs(X[None, :, 1:] - X[:, None, 1:])
    dx = D.max(-1) if np.isinf(p) else (D**p).sum(-1)**(1.0/p)
    R = (dt > 0) & (dt > dx)
    np.fill_diagonal(R, False)
    return R
