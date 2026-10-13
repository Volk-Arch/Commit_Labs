"""Доля сравнимых пар в ромбе d мер: замер против формулы Мирхейма-Мейера."""
import numpy as np
from math import gamma

def diamond(N, d, rs):
    """Ромб в d мерах: |x| <= min(t, T-t), T=1."""
    out, got = [], 0
    while got < N:
        m = (N - got) * 8 + 256
        Y = np.empty((m, d))
        Y[:, 0] = rs.uniform(0, 1, m)
        Y[:, 1:] = rs.uniform(-0.5, 0.5, (m, d - 1))
        r = np.linalg.norm(Y[:, 1:], axis=1)
        Y = Y[(Y[:, 0] >= r) & (1 - Y[:, 0] >= r)]
        if len(Y): out.append(Y); got += len(Y)
    return np.vstack(out)[:N]

def frac(X):
    dt = X[:, 0][:, None] - X[:, 0][None, :]
    dx = np.linalg.norm(X[:, None, 1:] - X[None, :, 1:], axis=2)
    rel = (dt > 0) & (dt > dx)
    n = len(X)
    return rel.sum() / (n * (n - 1) / 2)

print('d   замер      Г(d+1)Г(d/2)/(4Г(3d/2))')
for d in (2, 3, 4):
    f = np.mean([frac(diamond(900, d, np.random.default_rng(100 + s)))
                 for s in range(6)])
    mm = gamma(d + 1) * gamma(d / 2) / (4 * gamma(3 * d / 2))
    print(f'{d}   {f:.5f}    {mm:.5f}')
