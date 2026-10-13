#!/usr/bin/env python3
"""Почему петля падает в цепь: шкалы слишком согласованы. И чего не хватает.

Ничьи оказались ни при чём (0.55% и 0.20%). Остаётся исходное
объяснение: v_i = h - W_i/2 у всех опор содержит ОДНО И ТО ЖЕ h, и если
разброс h больше разброса W/2, шкалы почти совпадают, а покомпонентное
доминирование по почти одинаковым шкалам есть тотальный порядок.

Сравниваем с НАСТОЯЩИМ конусом s_u = t - u*x, где
  * разбросы t и u*x сопоставимы,
  * u*x при разных u почти не коррелируют: corr = u1*u2, в среднем 0.
"""
import sys
import numpy as np

sys.path.insert(0, '/home/claude/code')
from fixpoint import diamond, cone_order
from coneloop import cone_scales, heights_of

N, T, K = 700, 2.0, 5
rs = np.random.default_rng(4001)
X = diamond(N, T, rs)
R0 = cone_order(X)
h = heights_of(R0)
cand = np.flatnonzero((h > np.percentile(h, 25)) & (h < np.percentile(h, 75)))
srcs = np.random.default_rng(77).choice(cand, K, replace=False)
V = cone_scales(R0, srcs)

# разложим нашу шкалу на части
hh = h.astype(float)
WW = 2.0 * (hh[:, None] - V)        # W_i = 2*(h - v_i)
print('НАША шкала v = h - W/2')
print(f'  разброс h          {hh.std():.3f}')
print(f'  разброс W/2        {(WW/2).std(0).mean():.3f}')
print(f'  отношение          {(WW/2).std(0).mean()/hh.std():.3f}')
C = np.corrcoef(V.T)
iu = np.triu_indices(K, 1)
print(f'  корреляция МЕЖДУ шкалами   {C[iu].mean():+.4f}')
Cw = np.corrcoef((WW/2).T)
print(f'  корреляция между W/2       {Cw[iu].mean():+.4f}')
print(f'  доля отрицательных W/2     {(WW < 0).mean():.4f}')

# настоящий конус для сравнения
u = np.random.default_rng(9).normal(size=(K, 3))
u /= np.linalg.norm(u, axis=1, keepdims=True)
proj = X[:, 1:] @ u.T
S = X[:, 0:1] - proj
print('\nНАСТОЯЩИЙ конус s_u = t - u*x')
print(f'  разброс t          {X[:,0].std():.3f}')
print(f'  разброс u*x        {proj.std(0).mean():.3f}')
print(f'  отношение          {proj.std(0).mean()/X[:,0].std():.3f}')
Cs = np.corrcoef(S.T)
print(f'  корреляция МЕЖДУ шкалами   {Cs[iu].mean():+.4f}')
Cp = np.corrcoef(proj.T)
print(f'  корреляция между u*x       {Cp[iu].mean():+.4f}')
print(f'  доля отрицательных u*x     {(proj < 0).mean():.4f}')

print('\nРАЗНИЦА, которая всё решает:')
print('  u*x ЗНАКОПЕРЕМЕННО — положительно по одну сторону линии,')
print('  отрицательно по другую, поэтому разные u почти не коррелируют.')
print('  W — РАССТОЯНИЕ, оно всегда положительно, и расстояния до разных')
print('  линий в ограниченной области коррелируют положительно.')
