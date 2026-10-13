#!/usr/bin/env python3
"""ЕДИНСТВЕННОСТЬ: сжимающее отображение или расходящееся.

РАМКА. Задача перестала быть «почему мир предпочитает» и стала
«есть ли у уравнения R = пересечение L_i(R) единственное решение».
Для этого есть обычный инструмент: взять два БЛИЗКИХ начала, прогнать
ОДНО И ТО ЖЕ отображение и смотреть, растёт расстояние или падает.

    падает  -> сжимающее -> решение единственно -> это и есть отбор
    растёт  -> расходящееся -> решений много, отбора нет
    а если расстояние по парам растёт, но ДИАГНОСТИКИ сходятся,
    то единственна СТАТИСТИКА, а не порядок. Физике этого хватило бы.

ПАРАМЕТРЫ ИСПРАВЛЕНЫ. Прежние прогоны шли на мире с 44% вещества
(линии по 140 при фоне 1500). Правильное требование: M ~ 4*высоты,
значит при фоне 6000 (высота ~16) достаточно M = 64, то есть 8%.

Отображение сделано ДЕТЕРМИНИРОВАННЫМ: опоры закреплены по индексу,
разрыв ничьих — один и тот же массив для обоих начал. Иначе сравнивать
нечего.

Близкое начало строится как конусный порядок чуть дрожащих координат —
это законный порядок (транзитивность цела), отличающийся на малую долю
пар.
"""
import math
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from core import frac_pairs, mm_dim, height, spectral
from fixpoint import diamond, cone_order, box_order, rel_of_int
from coneloop import heights_of

NBG, T, NW, M, ITERS = 6000, 2.0, 8, 64, 6

def Wof(R, ch):
    rel = R[:, ch] | R[ch, :].T
    return (~rel).sum(1).astype(float)

rs = np.random.default_rng(4701)
parts, chains, n = [diamond(NBG, T, rs)], [], NBG
for s in range(NW):
    c = rs.uniform(-0.40, 0.40, 3)
    Y = np.empty((M, 4))
    Y[:, 0] = np.sort(rs.uniform(0.25, 1.75, M))
    Y[:, 1:] = c + 0.01 * rs.normal(size=(M, 3))
    parts.append(Y); chains.append(np.arange(n, n + M)); n += M
X = np.vstack(parts)
N = len(X)
PAIRS = [(2*i, 2*i+1) for i in range(NW // 2)]
TIE = np.random.default_rng(999).random((N, len(PAIRS))) * 1e-9
phi = NW * M / N
print(f'N={N}, вещества {phi:.3f} ({NW} линий по {M}), шкал {len(PAIRS)}')
print('отображение детерминировано: опоры и разрыв ничьих закреплены')
print()

def step(R):
    h = heights_of(R).astype(float)
    W = np.stack([Wof(R, c) for c in chains], 1)
    V = np.stack([h - (W[:, b] - W[:, a]) / 2.0 for a, b in PAIRS], 1)
    return rel_of_int(V + TIE)

def dist(A, B):
    return float((A != B).mean())

# два близких начала
Ra = cone_order(X)
Xj = X.copy()
Xj[:, 1:] += 0.004 * np.random.default_rng(31).normal(size=(N, 3))
Rb = cone_order(Xj)
print(f'СЖАТИЕ: два близких конусных начала, расстояние {dist(Ra,Rb):.5f}')
print(f"{'итерация':>10}{'расстояние':>13}{'множитель':>12}"
      f"{'доля A':>9}{'доля B':>9}{'разм. A':>10}{'разм. B':>10}")
d0 = dist(Ra, Rb)
A, B = Ra, Rb
print(f'{0:>10}{d0:>13.5f}{"":>12}{frac_pairs(A):>9.4f}'
      f'{frac_pairs(B):>9.4f}{mm_dim(frac_pairs(A)):>10.2f}'
      f'{mm_dim(frac_pairs(B)):>10.2f}')
for it in range(1, ITERS + 1):
    A, B = step(A), step(B)
    d = dist(A, B)
    print(f'{it:>10}{d:>13.5f}{(d/max(d0,1e-12)):>12.3f}'
          f'{frac_pairs(A):>9.4f}{frac_pairs(B):>9.4f}'
          f'{mm_dim(frac_pairs(A)):>10.2f}{mm_dim(frac_pairs(B)):>10.2f}',
          flush=True)
    d0 = d

print()
print('СТАТИСТИЧЕСКАЯ СХОДИМОСТЬ: далёкие начала, сходятся ли диагностики')
print(f"{'начало':>12}{'итер':>6}{'доля пар':>10}{'разм.':>8}{'спектр':>10}"
      f"{'высота':>8}")
outs = {}
for nm, R0 in (('конус', cone_order(X)), ('ящик', box_order(N, rs)),
               ('случайный', rel_of_int(rs.random((N, 2))))):
    R = R0
    for it in range(ITERS):
        R = step(R)
    outs[nm] = R
    print(f'{nm:>12}{ITERS:>6}{frac_pairs(R):>10.4f}'
          f'{mm_dim(frac_pairs(R)):>8.2f}{spectral(R):>+10.4f}'
          f'{height(R)[0]:>8.0f}', flush=True)
ns = list(outs)
print()
for i in range(len(ns)):
    for j in range(i + 1, len(ns)):
        print(f'  {ns[i]:>10} против {ns[j]:<10} совпадение по парам '
              f'{1-dist(outs[ns[i]], outs[ns[j]]):.4f}')
print('\n  эталон конуса: доля 0.098-0.103, разм. 4.00, спектр > 0')
