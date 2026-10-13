#!/usr/bin/env python3
"""Что такое +0.31: перевес дефектов или плавная текстура.

charge2.py: суммарный индекс заплаты +0.3145 +- 0.0648 (4.9 сигмы),
масштабно-инвариантный. А перепись клеток (charge.py) дала знаки
сбалансированными: 65 плюс / 55 минус, 0.9 сигмы от симметрии.

Числа кажутся несогласованными, и причина может быть простой: сумма по
клеткам включает НЕ ТОЛЬКО дефекты. У каждой клетки есть маленькая
дробная намотка, которая округляется в нуль, но при сложении четырёхсот
клеток даёт систематику. Это плавное закручивание поля по заплате —
текстура, а не заряд.

РАЗДЕЛЯЕМ:
    сумма округлённых  = вклад ДЕФЕКТОВ
    сумма - округлённые = ТЕКСТУРА
Если текстура около +0.31, а дефекты около нуля — вопрос закрыт.
"""
import math
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from topo import diamond, cone_rel, plane_of
from charge import field_grid, loop_on_grid, census

N, T, G, STEP, MINS, PMIN = 3000, 2.0, 5, 0.05, 25, 0.25
tot, dfc, tex, ndef = [], [], [], []
for t in range(8):
    rs = np.random.default_rng(2901000 + 977 * t)
    X = diamond(N, T, rs)
    R = cone_rel(X)
    i0 = int(np.argmax(R.sum(0) * (X[:, 0] < 0.25 * T)))
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
        th = field_grid(X, R, i0, j0, u, v, 2 * G, STEP, MINS, cand, PMIN)
        b, _ = loop_on_grid(th, 0, 0, 4 * G)
        if b is None:
            continue
        c = census(th, 2 * G)[0]
        c = c[np.isfinite(c)]
        q = np.round(c * 2) / 2
        tot.append(b); dfc.append(float(q.sum()))
        tex.append(float((c - q).sum()))
        ndef.append(int((np.abs(q) >= 0.5).sum()))
    del R, X

def rep(nm, A):
    A = np.array(A, float)
    se = A.std() / math.sqrt(len(A))
    print(f'{nm:>26}{A.mean():>+12.4f}{se:>10.4f}'
          f'{abs(A.mean())/max(se,1e-12):>8.1f}')

print(f'заплат {len(tot)};  дефектов на заплату {np.mean(ndef):.2f}')
print(f"{'':>26}{'среднее':>12}{'±':>10}{'сигм':>8}")
rep('суммарный индекс', tot)
rep('вклад ДЕФЕКТОВ (округл.)', dfc)
rep('ТЕКСТУРА (остаток)', tex)
print('\n  если текстура несёт почти всё, а дефекты около нуля —')
print('  +0.31 есть плавное закручивание, заряд же сбалансирован')
