#!/usr/bin/env python3
"""СКОЛЬКО ВЕЩЕСТВА нужно, чтобы мир сумел себя измерить.

РАМКА. Общая форма закона может быть только одна, потому что
локализация правила ломает транзитивность (fieldbow: 0.0022..0.0883
против нуля):

    R = пересечение k линейных порядков L_i,
    причём каждое L_i определено через сам R.

Вещество тут не слагаемое: плотная мировая линия есть свойство R, то
есть длинная цепь. Уравнение одно, на R одном.

СЛЕДСТВИЕ, которое проверяем. Решение обязано быть НЕОДНОРОДНЫМ:
линейке нужно ~lam событий на единицу собственного времени вдоль линии,
а четырёхмерная толща даёт только ~rho^(1/4) (измерено: 0.408 против
нужных 1.370). Значит у уравнения есть ДОЛЯ ВЕЩЕСТВА, при которой
самосогласование лучше всего.

МЕРИМ. Доля phi событий сидит в 8 плотных линиях (4 непересекающиеся
пары), остальное — фон. Для каждой phi:

    согласие  = доля пар, которую один шаг петли ОСТАВЛЯЕТ как была
    сдвиг     = |доля пар после шага - до шага|

Самосогласование = высокое согласие при малом сдвиге. Контроль:
тот же шаг на мире БЕЗ линий (phi = 0).
"""
import math
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from core import frac_pairs, mm_dim
from fixpoint import diamond, cone_order, rel_of_int
from coneloop import heights_of

NTOT, T, NW = 2400, 2.0, 8

def Wof(R, ch):
    rel = R[:, ch] | R[ch, :].T
    return (~rel).sum(1).astype(float)

def scales(R, chains):
    h = heights_of(R).astype(float)
    Ws = [Wof(R, c) for c in chains]
    return np.stack([h - (Ws[2*i+1] - Ws[2*i]) / 2.0
                     for i in range(len(Ws)//2)], 1)

print(f'всего событий {NTOT}, линий {NW} (4 непересекающиеся пары)')
print('согласие = доля пар, которую шаг оставляет как была')
print()
print(f"{'доля вещества':>15}{'в линии':>9}{'доля пар до':>13}"
      f"{'после':>9}{'разм. после':>13}{'согласие':>11}{'сдвиг':>9}"
      f"{'корр.шкал':>11}")

for phi in (0.0, 0.03, 0.08, 0.15, 0.25, 0.40):
    ag, sh, fr0, fr1, dm, cs = [], [], [], [], [], []
    for t in range(3):
        rs = np.random.default_rng(4501 + 101*t)
        M = int(phi * NTOT / NW)
        nbg = NTOT - M * NW
        parts, chains, n = [diamond(nbg, T, rs)], [], nbg
        for s in range(NW):
            if M < 3:
                break
            c = rs.uniform(-0.40, 0.40, 3)
            Y = np.empty((M, 4))
            Y[:, 0] = np.sort(rs.uniform(0.25, 1.75, M))
            Y[:, 1:] = c + 0.01 * rs.normal(size=(M, 3))
            parts.append(Y); chains.append(np.arange(n, n+M)); n += M
        X = np.vstack(parts)
        N = len(X)
        R = cone_order(X)
        if len(chains) < 2:
            # нет вещества: опоры берём естественными цепями
            from coneloop import chain_through
            h = heights_of(R)
            mid = np.flatnonzero((h > np.percentile(h, 25))
                                 & (h < np.percentile(h, 75)))
            src = rs.choice(mid, NW, replace=False)
            chains = [chain_through(R, int(v), h) for v in src]
        V = scales(R, chains) + 1e-9 * rs.random((N, len(chains)//2))
        Rn = rel_of_int(V)
        f0, f1 = frac_pairs(R), frac_pairs(Rn)
        ag.append(float((Rn == R).mean())); sh.append(abs(f1 - f0))
        fr0.append(f0); fr1.append(f1); dm.append(mm_dim(f1))
        iu = np.triu_indices(V.shape[1], 1)
        cs.append(float(np.corrcoef(V.T)[iu].mean()))
    print(f'{phi:>15.2f}{M:>9}{np.mean(fr0):>13.4f}{np.mean(fr1):>9.4f}'
          f'{np.mean(dm):>13.2f}{np.mean(ag):>11.4f}{np.mean(sh):>9.4f}'
          f'{np.mean(cs):>+11.4f}')

print('\n  максимум согласия при минимуме сдвига = лучшая доля вещества')
print('  эталон: доля пар 0.098-0.103, размерность 4.00')
