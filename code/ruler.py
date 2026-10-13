#!/usr/bin/env python3
"""Хватает ли разрешения у линейки: сколько событий в опорной цепи.

ЗНАКОВАЯ шкала v = h - (W2-W1)/2 починила знак и расцепила направления
(корреляция поперечных +0.0891 против +0.4475), но разброс поперечного
остался вчетверо мал, и корреляция МЕЖДУ шкалами держится на +0.91.

ПОДОЗРЕНИЕ. W = 2*lam*r проверено на ПОДСАЖЕННОЙ плотной трубке
(lam = 68..136, цепь из 120..240 событий, корреляция с r до +0.982). А
естественная максимальная цепь вакуумного засева содержит всего
~m4*rho^(1/4)*T событий, то есть высоту мира: 9 при N=700. На девяти
уровнях поперечное расстояние почти не разрешается.

МЕРИМ: насколько знаковая шкала воспроизводит настоящую проекцию u*x,
при разных N и при разной плотности ОПОРНОЙ цепи (естественная против
подсаженной).
"""
import sys
import numpy as np

sys.path.insert(0, '/home/claude/code')
from fixpoint import diamond, cone_order
from coneloop import heights_of, chain_through

def Wof(R, ch):
    rel = R[:, ch] | R[ch, :].T
    return (~rel).sum(1).astype(float)

print('естественные опорные цепи: растёт ли разрешение с N')
print(f"{'N':>7}{'высота':>8}{'цепь':>7}{'станд.попереч':>15}"
      f"{'станд. h':>10}{'отнош.':>9}{'корр. с u*x':>13}")
for N in (700, 1500, 3000, 6000):
    rs = np.random.default_rng(4201)
    X = diamond(N, 2.0, rs)
    R = cone_order(X)
    h = heights_of(R)
    mid = np.flatnonzero((h > np.percentile(h, 25))
                         & (h < np.percentile(h, 75)))
    a, b = (int(x) for x in rs.choice(mid, 2, replace=False))
    ch1 = chain_through(R, a, h); ch2 = chain_through(R, b, h)
    d = (Wof(R, ch2) - Wof(R, ch1)) / 2.0
    # настоящая знаковая проекция вдоль направления между опорами
    u = X[b, 1:] - X[a, 1:]
    u = u / max(np.linalg.norm(u), 1e-9)
    true = X[:, 1:] @ u
    print(f'{N:>7}{h.max():>8}{len(ch1):>7}{d.std():>15.3f}'
          f'{h.std():>10.3f}{d.std()/h.std():>9.3f}'
          f'{np.corrcoef(d, true)[0,1]:>+13.4f}')

print('\nподсаженные плотные опорные цепи при N=1500')
print(f"{'событий в цепи':>15}{'станд.попереч':>15}{'станд. h':>10}"
      f"{'отнош.':>9}{'корр. с u*x':>13}")
for M in (12, 40, 120, 300):
    rs = np.random.default_rng(4301)
    X0 = diamond(1500, 2.0, rs)
    out = [X0]
    cents = []
    for s in range(2):
        c = rs.uniform(-0.35, 0.35, 3)
        cents.append(c)
        Y = np.empty((M, 4))
        Y[:, 0] = np.sort(rs.uniform(0.3, 1.7, M))
        Y[:, 1:] = c + 0.01 * rs.normal(size=(M, 3))
        out.append(Y)
    X = np.vstack(out)
    R = cone_order(X)
    h = heights_of(R)
    n0 = len(X0)
    ch1 = np.arange(n0, n0 + M); ch2 = np.arange(n0 + M, n0 + 2 * M)
    d = (Wof(R, ch2) - Wof(R, ch1)) / 2.0
    u = cents[1] - cents[0]
    u = u / max(np.linalg.norm(u), 1e-9)
    true = X[:, 1:] @ u
    print(f'{M:>15}{d.std():>15.3f}{h.std():>10.3f}'
          f'{d.std()/h.std():>9.3f}{np.corrcoef(d, true)[0,1]:>+13.4f}')

print('\n  эталон конуса: отношение 1.370')
print('  если отношение растёт только с ПОДСАЖЕННОЙ плотностью опоры,')
print('  то линейка требует вещества: нет материи — нет шкал')
