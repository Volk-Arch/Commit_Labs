#!/usr/bin/env python3
"""Кластеры по УГЛУ И ВЕЛИЧИНЕ вместе — правильная версия его идеи.

layer.py: кластеры только по углу (m/k около 1/2) растянуты, потому что
угол одинаков у всех одновременных событий, как бы далеко они ни были.
Величина теперь тоже порядковая: q = |past&past|/min(|past|,|past|).

Склеиваем оба: близки, если и одновременны (угол), и перекрытие прошлых
велико (величина). Если ТОГДА кластеры станут компактны — идея работает.

КОНТРОЛИ: случайный граф того же веса; и то же на ящике (без геометрии).
"""
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from partial import diamond, scales_cone, m_matrix, spectral, modularity
from layer import corr_dim

N, K, T, NC, SEEDS = 300, 64, 2.0, 6, 6
print(f'N={N}, k={K}, кластеров {NC}, засевов {SEEDS}')
print(f"{'граф':>22}{'модуль':>9}{'протяж.':>10}{'случ.':>9}"
      f"{'сжатие':>9}{'разм.3D':>10}{'вр/пр':>8}")

for gname in ('только угол', 'только величина q', 'УГОЛ И ВЕЛИЧИНА',
              'случайный граф', 'ящик: угол и велич.'):
    Q, EX, EX0, D3, RT = [], [], [], [], []
    for t in range(SEEDS):
        rs = np.random.default_rng(1801000 + 977 * t)
        X = diamond(N, T, rs)
        box = gname.startswith('ящик')
        S = rs.random((N, K)) if box else scales_cone(X, K, rs)
        M = m_matrix(S)
        R = (M == K)                      # i раньше j по всем шкалам
        A_ = R.astype(np.float32)
        CP = A_.T @ A_                    # |past(i) & past(j)|
        sz = A_.sum(0)
        mn = np.minimum(sz[:, None], sz[None, :])
        q = CP / np.maximum(mn, 1.0)
        inc = (~R) & (~R.T) & (mn >= 6)
        ang = np.abs(M / K - 0.5)
        wa = np.exp(-((ang / 0.10) ** 2))
        wq = np.clip(q, 0, 1) ** 2
        if gname == 'только угол':
            A = np.where(inc, wa, 0.0)
        elif gname == 'только величина q':
            A = np.where(inc, wq, 0.0)
        elif gname.startswith('случайный'):
            A = (rs.random((N, N)) < 0.08).astype(float)
            A = np.maximum(A, A.T)
        else:
            A = np.where(inc, wa * wq, 0.0)
        A = np.asarray((A + A.T) / 2, float)
        np.fill_diagonal(A, 0.0)
        if A.sum() < 20:
            continue
        lab = spectral(A, NC, rs)
        Q.append(modularity(A, lab))
        for c in np.unique(lab):
            s = np.flatnonzero(lab == c)
            if len(s) < 14:
                continue
            P = X[s]
            EX.append(float(np.linalg.norm(
                P[:, 1:] - P[:, 1:].mean(0), axis=1).mean()))
            rnd = rs.choice(N, len(s), replace=False)
            EX0.append(float(np.linalg.norm(
                X[rnd][:, 1:] - X[rnd][:, 1:].mean(0), axis=1).mean()))
            D3.append(corr_dim(P[:, 1:]))
            RT.append(float(P[:, 0].std()) / max(float(np.linalg.norm(
                P[:, 1:] - P[:, 1:].mean(0), axis=1).std()), 1e-9))
    if not Q or not EX:
        continue
    print(f'{gname:>22}{np.mean(Q):>9.3f}{np.mean(EX):>10.4f}'
          f'{np.mean(EX0):>9.4f}{np.mean(EX)/np.mean(EX0):>9.3f}'
          f'{np.nanmean(D3):>10.2f}{np.mean(RT):>8.2f}', flush=True)

print('\n  сжатие много меньше 1 => кластеры КОМПАКТНЫ, идея работает')
print('  ящик даёт то же => комбинаторика')
