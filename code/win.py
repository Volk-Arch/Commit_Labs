#!/usr/bin/env python3
"""Какая порядковая величина реально меряет расстояние ДО ЦЕПИ.

max по q провалился: он выбирает не пространственно ближайшее событие
трубки, а самое сдвинутое по времени из ещё несравнимых, то есть сидящее
у светового конуса, где разнос -> 0.

Кандидаты:
  W     = число событий трубки, НЕСРАВНИМЫХ с b  (классическое окно;
          для прямой мировой линии с плотностью lam длина окна по
          собственному времени равна 2r, значит W ~ 2*lam*r)
  q_mid = q до СЕРЕДИНЫ этого окна (середина определена относительно:
          это элемент цепи, максимально разнесённый с b)
  q_max = прежний провалившийся максимум, для сравнения

Сверяем с истинным r = |x| фонового события (трубка на оси).
"""
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, '/home/claude/code')
from weight import diamond, cone_rel

T, M, RAD = 1.8, 120, 0.02

for s in range(4):
    rs = np.random.default_rng(777 + 31 * s)
    u = rs.normal(size=(M, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    rr = RAD * rs.random(M) ** (1 / 3)
    ob = np.empty((M, 4))
    ob[:, 0] = np.sort(rs.uniform(RAD, T - RAD, M))
    ob[:, 1:] = u * rr[:, None]
    bg = diamond(T, rs)
    X = np.vstack([ob, bg])
    R = cone_rel(X)

    A = R.astype(np.float32)
    CP = A[M:] @ A[:M].T
    sb, so = A[M:].sum(1), A[:M].sum(1)
    mn = np.minimum(sb[:, None], so[None, :])
    qq = CP / np.maximum(mn, 1.0)

    rel = R[M:, :M] | R[:M, M:].T          # сравнимо с событием трубки
    inc = (~rel) & (mn >= 8)               # несравнимо и прошлое непусто

    W = inc.sum(1).astype(float)
    qm = np.where(inc, qq, 2.0)
    mid = qm.argmin(1)                     # минимум q = максимум разноса
    q_mid = np.where(W > 0, qq[np.arange(len(W)), mid], np.nan)
    q_max = np.where(inc, qq, -1.0).max(1)

    r = np.linalg.norm(X[M:, 1:], axis=1)
    t = X[M:, 0]
    keep = (W > 2) & (np.abs(t - T / 2) < 0.3 * T) & (r < 0.6)

    print(f'--- засев {s}:  {keep.sum()} событий,  '
          f'W от {W[keep].min():.0f} до {W[keep].max():.0f}')
    for nm, v in (('W', W), ('q_mid', q_mid), ('q_max', q_max)):
        rho = spearmanr(v[keep], r[keep]).correlation
        # линейность W по r
        print(f'   {nm:>6}  ранг. корр. с r = {rho:>+.3f}', end='')
        if nm == 'W':
            p = np.polyfit(r[keep], v[keep], 1)
            print(f'   W = {p[0]:.1f}*r + {p[1]:.1f}')
        else:
            print()
