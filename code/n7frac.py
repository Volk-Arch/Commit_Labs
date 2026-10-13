"""Доля пар у сприклинга N=7 в ромбе d мер: эмпирический эталон вместо
асимптотической формулы Мирхейма-Мейера, которая при N=7 неприменима."""
import numpy as np
from math import gamma
from scipy.optimize import brentq

def diamond(N, d, rs):
    out, got = [], 0
    while got < N:
        m = (N - got) * 12 + 256
        Y = np.empty((m, d)); Y[:, 0] = rs.uniform(0, 1, m)
        Y[:, 1:] = rs.uniform(-0.5, 0.5, (m, d - 1))
        r = np.linalg.norm(Y[:, 1:], axis=1)
        Y = Y[(Y[:, 0] >= r) & (1 - Y[:, 0] >= r)]
        if len(Y): out.append(Y); got += len(Y)
    return np.vstack(out)[:N]

def frac(X):
    dt = X[:, 0][:, None] - X[:, 0][None, :]
    dx = np.linalg.norm(X[:, None, 1:] - X[None, :, 1:], axis=2)
    n = len(X)
    return ((dt > 0) & (dt > dx)).sum() / (n * (n - 1) / 2)

g = lambda d: gamma(d+1)*gamma(d/2)/(2*gamma(1.5*d))
print('асимптотика (верная формула):  d=2 %.4f  d=3 %.4f  d=4 %.4f'
      % (g(2), g(3), g(4)))
print()
print('N=7, 60000 сприклингов на размерность:')
print(f"{'d':>3}{'средняя f':>11}{'медиана':>9}{'ст.откл':>9}"
      f"{'доля f<0.20':>13}{'доля f в 0.20..0.35':>20}{'асимпт.':>9}")
rs = np.random.default_rng(4242)
for d in (2, 3, 4):
    fs = np.array([frac(diamond(7, d, rs)) for _ in range(60000)])
    print(f'{d:>3}{fs.mean():>11.4f}{np.median(fs):>9.4f}{fs.std():>9.4f}'
          f'{(fs < 0.20).mean():>13.3f}'
          f'{((fs >= 0.20) & (fs < 0.35)).mean():>20.3f}{g(d):>9.4f}')
