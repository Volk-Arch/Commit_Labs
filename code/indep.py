#!/usr/bin/env python3
"""Независимая статистика: заплаты накладывались, 4.9 сигмы завышены.

ПОЙМАННАЯ ОШИБКА. В charge2.py снималось 22 заплаты с ОДНОГО мира при
радиусе большой петли 4*G*STEP = 1.0 и ширине ромба 1.0 — заплаты почти
полностью перекрываются. Значит 62 «измерения» независимы не больше, чем
8 засевов, и значимость 4.9 сигмы посчитана по завышенному N.

ЗДЕСЬ: по одной заплате на засев, засевов много. Плюс честная проверка
парности дефектов — дисперсия суммарного индекса против той, что была бы
при независимых дефектах (0.5 * sqrt(число дефектов)).
"""
import math
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from topo import diamond, cone_rel, plane_of
from charge import field_grid, loop_on_grid, census

N, T, G, STEP, MINS, PMIN = 3000, 2.0, 5, 0.05, 25, 0.25
SEEDS = 28
tot, nd, fine, coarse = [], [], [], []
for t in range(SEEDS):
    rs = np.random.default_rng(3101000 + 7919 * t)
    X = diamond(N, T, rs)
    R = cone_rel(X)
    i0 = int(np.argmax(R.sum(0) * (X[:, 0] < 0.25 * T)))
    cand = np.flatnonzero(R[:, i0])
    if len(cand) < 300:
        del R, X; continue
    rs2 = np.random.default_rng(4100 + t)
    j0 = int(rs2.choice(cand))                 # ОДНА заплата на засев
    try:
        u, v = plane_of(X, i0, j0)
    except Exception:
        del R, X; continue
    thf = field_grid(X, R, i0, j0, u, v, 2 * G, STEP, MINS, cand, PMIN)
    thc = field_grid(X, R, i0, j0, u, v, G, 2 * STEP, MINS, cand, PMIN)
    bf, _ = loop_on_grid(thf, 0, 0, 4 * G)
    bc, _ = loop_on_grid(thc, 0, 0, 2 * G)
    del R, X
    if bf is None or bc is None:
        continue
    c = census(thf, 2 * G)[0]
    c = c[np.isfinite(c)]
    q = np.round(c * 2) / 2
    tot.append(bf); fine.append(bf); coarse.append(bc)
    nd.append(int((np.abs(q) >= 0.5).sum()))

A = np.array(tot); F = np.array(fine); C = np.array(coarse)
se = A.std() / math.sqrt(len(A))
print(f'независимых заплат (по одной на засев): {len(A)}')
print(f'  дефектов на заплату {np.mean(nd):.2f}')
print(f'  суммарный индекс {A.mean():+.4f} ± {se:.4f}   '
      f'({abs(A.mean())/max(se,1e-12):.1f}σ от нуля)')
print(f'  с.к.о. суммарного индекса {A.std():.3f}')
exp = 0.5 * math.sqrt(np.mean(nd))
print(f'  ожидание при НЕЗАВИСИМЫХ дефектах 0.5*sqrt(n) = {exp:.3f}')
print(f'  отношение {A.std()/exp:.3f}  (меньше 1 => дефекты ПАРЯТСЯ)')
d = F - C
sed = d.std() / math.sqrt(len(d))
print(f'  масштаб: парная разность {d.mean():+.4f} ± {sed:.4f} '
      f'({abs(d.mean())/max(sed,1e-12):.1f}σ), '
      f'совпало точно {int((np.abs(d)<1e-6).sum())}/{len(d)}, '
      f'корр {np.corrcoef(F, C)[0,1]:+.3f}')
