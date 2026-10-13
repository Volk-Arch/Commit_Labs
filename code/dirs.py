#!/usr/bin/env python3
"""РАЗМЕН: сколько направлений из скольких мировых линий.

ПОЧЕМУ ЭТО РЕШАЕТ ВСЁ. Архив: лоренц-инвариантность требует ~80
направлений (четыре шкалы отстоят от конуса на 27.7 сигмы). У нас
направление даёт ПАРА мировых линий, значит:

  непересекающимися парами  80 направлений -> 160 линий -> мир станет
                            веществом целиком, мертво
  с общими концами          C(13,2) = 78 направлений -> 13 линий,
                            вполне реально при большом N

Но за общий конец платят корреляцией: пары (i,j) и (i,l) делят W_i.
Именно это и ловилось: звезда от одной опоры дала +0.91 против +0.42 у
непересекающихся.

МЕРИМ корреляцию МЕЖДУ шкалами как функцию числа линий m, при трёх
способах выбрать пары, против конусного эталона +0.24.

    все C(m,2)          максимум направлений, максимум общих концов
    жадно малоперекрыт. отбор пар с наименьшим числом общих концов
    непересекающиеся    m/2 пар, общих концов нет
"""
import math
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, '/home/claude/code')
from fixpoint import diamond, cone_order
from coneloop import heights_of

NBG, T, M = 2000, 2.0, 120

def Wof(R, ch):
    rel = R[:, ch] | R[ch, :].T
    return (~rel).sum(1).astype(float)

def build(m, seed):
    rs = np.random.default_rng(seed)
    parts, chains, n = [diamond(NBG, T, rs)], [], NBG
    for s in range(m):
        c = rs.uniform(-0.40, 0.40, 3)
        Y = np.empty((M, 4))
        Y[:, 0] = np.sort(rs.uniform(0.25, 1.75, M))
        Y[:, 1:] = c + 0.01 * rs.normal(size=(M, 3))
        parts.append(Y); chains.append(np.arange(n, n + M)); n += M
    X = np.vstack(parts)
    R = cone_order(X)
    h = heights_of(R).astype(float)
    W = np.stack([Wof(R, c) for c in chains], 1)
    return X, R, h, W

def corr_of(h, W, pairs):
    V = np.stack([h - (W[:, b] - W[:, a]) / 2.0 for a, b in pairs], 1)
    if V.shape[1] < 2:
        return float('nan')
    iu = np.triu_indices(V.shape[1], 1)
    return float(np.corrcoef(V.T)[iu].mean())

print(f'фон {NBG}, по {M} событий в линии; эталон конуса +0.24')
print(f"{'линий m':>9}{'вещества':>10}{'все C(m,2)':>13}{'напр.':>7}"
      f"{'жадно малопер.':>16}{'напр.':>7}{'непересек.':>12}{'напр.':>7}")

for m in (3, 5, 8, 13, 20):
    rows = []
    for t in range(2):
        X, R, h, W = build(m, 4601 + 97 * t)
        allp = list(combinations(range(m), 2))
        c_all = corr_of(h, W, allp)
        # жадно: добавляем пару, минимально пересекающуюся с выбранными
        chosen, used = [], {}
        for _ in range(min(len(allp), 3 * m)):
            best, bk = None, 1e9
            for p in allp:
                if p in chosen:
                    continue
                k = used.get(p[0], 0) + used.get(p[1], 0)
                if k < bk:
                    best, bk = p, k
            if best is None:
                break
            chosen.append(best)
            used[best[0]] = used.get(best[0], 0) + 1
            used[best[1]] = used.get(best[1], 0) + 1
        c_gr = corr_of(h, W, chosen)
        dis = [(2 * i, 2 * i + 1) for i in range(m // 2)]
        c_di = corr_of(h, W, dis) if len(dis) >= 2 else float('nan')
        rows.append((c_all, len(allp), c_gr, len(chosen), c_di, len(dis)))
    A = np.array([[r[0], r[2], r[4]] for r in rows])
    n_all, n_gr, n_di = rows[0][1], rows[0][3], rows[0][5]
    phi = m * M / (NBG + m * M)
    print(f'{m:>9}{phi:>10.2f}{np.nanmean(A[:,0]):>+13.4f}{n_all:>7}'
          f'{np.nanmean(A[:,1]):>+16.4f}{n_gr:>7}'
          f'{np.nanmean(A[:,2]):>+12.4f}{n_di:>7}')

print('\n  если корреляция при m=13 и 78 направлениях остаётся около +0.24,')
print('  конструкция масштабируется; если лезет к +0.9 — мертва')
