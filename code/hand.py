#!/usr/bin/env python3
"""Знак суммарного индекса: физика или произвол базиса.

charge2.py: суммарный индекс заплаты +0.3145 +- 0.0648, то есть 4.9 сигмы
от нуля, и он масштабно-инвариантен (корреляция 0.897). Но знак обхода
зависит от ОРИЕНТАЦИИ плоскости (u,v), а её выдаёт eigh от проектора
I - a a^T, у которого собственное значение 1 трёхкратно: базис
произволен, хотя и детерминирован.

Если ориентацию выбирать случайно (менять u и v местами с вероятностью
1/2), среднее обязано уйти в нуль по симметрии. Это проверяет не наличие
хиральности, а то, НЕ ПРИВЯЗАН ли мой знак к постороннему. Содержательно
же надо смотреть |индекс| и разброс: они от ориентации не зависят.

Конусный порядок зеркально-симметричен, хиральности в нём нет по
построению, значит ненулевое среднее при фиксированном базисе есть
артефакт выбора базиса.
"""
import math
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from topo import diamond, cone_rel, plane_of
from charge import field_grid, loop_on_grid

N, T, G, STEP, MINS, PMIN = 3000, 2.0, 5, 0.05, 25, 0.25
print('суммарный индекс заплаты при разных способах выбрать базис')
print(f"{'базис':>22}{'заплат':>8}{'среднее':>12}{'±':>9}{'сигм':>7}"
      f"{'ср.|индекс|':>13}{'с.к.о.':>9}")

for mode in ('как в charge2 (eigh)', 'ориентация случайна', 'зеркало x1'):
    vals = []
    for t in range(8):
        rs = np.random.default_rng(2901000 + 977 * t)
        X = diamond(N, T, rs)
        if mode == 'зеркало x1':
            X = X.copy(); X[:, 1] = -X[:, 1]
        R = cone_rel(X)
        fut = R.sum(0)
        i0 = int(np.argmax(fut * (X[:, 0] < 0.25 * T)))
        cand = np.flatnonzero(R[:, i0])
        if len(cand) < 300:
            continue
        rs2 = np.random.default_rng(1300 + t)
        for j0 in rs2.choice(cand, min(22, len(cand)), replace=False):
            j0 = int(j0)
            try:
                u, v = plane_of(X, i0, j0)
            except Exception:
                continue
            if mode == 'ориентация случайна' and rs2.random() < 0.5:
                u, v = v, u
            th = field_grid(X, R, i0, j0, u, v, 2 * G, STEP, MINS,
                            cand, PMIN)
            b, _ = loop_on_grid(th, 0, 0, 4 * G)
            if b is not None:
                vals.append(b)
        del R, X
    A = np.array(vals)
    se = A.std() / math.sqrt(len(A))
    print(f'{mode:>22}{len(A):>8}{A.mean():>+12.4f}{se:>9.4f}'
          f'{abs(A.mean())/max(se,1e-12):>7.1f}'
          f'{np.abs(A).mean():>13.4f}{A.std():>9.4f}')

print('\n  если при случайной ориентации среднее уходит в нуль, а |индекс|')
print('  и с.к.о. держатся — знак был произволом базиса, а величина реальна')
