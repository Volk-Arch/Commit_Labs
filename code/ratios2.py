#!/usr/bin/env python3
"""ВОСЕМЬ ЧИСЕЛ ПРОТИВ ДВУХ РУЧЕК, БЕЗ АСИМПТОТИКИ.

ratios.py: анкер провалился, и по моему же правилу это остановило
сравнение. Диагноз: я приложил АСИМПТОТИЧЕСКУЮ форму
<N_m>/<N_0> -> Gamma(m+1/2)/m!/sqrt(pi) при N=2000, а она большого N.
Сессия уже ловила это однажды - подгонка `a` на изогнутом окне дала 1.21
вместо 4.1365. Расхождение растёт с m: подпись недостигнутой
асимптотики, а не ошибки счётчика.

Починка: цель берётся из ИЗМЕРЕННОГО анкера при том же N, а не из
формулы. Асимптотика тогда не нужна вовсе - сравниваются два ансамбля
при одном N по одной и той же величине.

И врагу даётся МАКСИМАЛЬНАЯ свобода: он подгоняется не по доле
связанных, а прямо по всем восьми отношениям сразу (минимум хи2 по
сетке (k, delta)). Это полная враждебная подгонка, как в census.

АРИФМЕТИКА, ради которой всё: 8 условий против 2 параметров. Если даже
с полной свободой враг не садится - дело не в типе инструмента, а в
счёте условий, и это простое правило на будущее.

В ratios.py качественная разница уже видна и асимптотики не требует:
у засева отношения УБЫВАЮТ (0.373 -> 0.072), у врага РАСТУТ
(1.38 -> 3.89).
"""
import math

import numpy as np

from sectors import sprinkle_d
from lkdeg import layered
from ratios import nm, MMAX


def avg(mk, rep):
    N, R = [], []
    for _ in range(rep):
        a, r = nm(mk())
        N.append(a)
        R.append(r)
    N = np.array(N)
    rat = N[:, 1:] / np.maximum(N[:, :1], 1.0)
    return (rat.mean(axis=0), rat.std(axis=0) / math.sqrt(rep),
            float(np.mean(R)), float(np.std(R) / math.sqrt(rep)))


def main():
    print(__doc__)
    rs = np.random.default_rng(1732050)
    n = 1200
    print(f'=== N = {n}, d = 4, цель — ИЗМЕРЕННЫЙ анкер ===')
    tgt, terr, r0, re0 = avg(lambda: sprinkle_d(n, 4, rs), 10)
    print(f'АНКЕР засев d=4: r={r0:.4f}±{re0:.4f}')
    print('  отношения: ' + ''.join(f'{v:>8.4f}' for v in tgt))
    print('  ошибки:    ' + ''.join(f'{v:>8.4f}' for v in terr))

    print('\nКОНТРОЛЬ: второй независимый засев против того же анкера')
    c, cerr, _, _ = avg(lambda: sprinkle_d(n, 4, rs), 10)
    sg = (c - tgt) / np.hypot(terr, cerr)
    print('  сигмы: ' + ''.join(f'{s:>8.1f}' for s in sg))
    print(f'  хи2 = {float((sg ** 2).sum()):.1f} при 8 условиях '
          f'=> {"ОК" if (sg ** 2).sum() < 25 else "СБОЙ"}')

    print('\nВРАГ подгоняется ПРЯМО ПО ВОСЬМИ ОТНОШЕНИЯМ:')
    best = None
    for k in (3, 4, 6, 9, 13, 18, 25, 34):
        for dl in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
            q = dl * k / n
            if q >= 1.0:
                continue
            rat, err, r, _ = avg(lambda: layered(n, k, dl * k / n, rs), 3)
            s = (rat - tgt) / np.hypot(terr, err)
            chi = float((s ** 2).sum())
            if best is None or chi < best[0]:
                best = (chi, k, dl, q, rat, err, r, s)
    chi, k, dl, q, rat, err, r, s = best
    print(f'  наилучшая подгонка: k={k}, delta={dl}, q={q:.4f}, '
          f'r={r:.4f}')
    print('  отношения: ' + ''.join(f'{v:>8.4f}' for v in rat))
    print('  сигмы:     ' + ''.join(f'{v:>8.1f}' for v in s))
    print(f'  ХИ2 = {chi:.1f} при 8 условиях и 2 ручках '
          f'(у второго засева было {float((sg ** 2).sum()):.1f})')
    print('\nЧТЕНИЕ: контроль мал, враг велик => восемь условий двумя')
    print('ручками не берутся, и правило простое: условий больше, чем ручек.')


if __name__ == '__main__':
    main()
