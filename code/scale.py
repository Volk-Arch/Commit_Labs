#!/usr/bin/env python3
"""ЗАВИСИМОСТЬ ОТ РАЗМЕРА: остаётся ли рост двумерно-многообразным, и
становится ли четырёхмерие достижимым?

ЧТО НАШЛОСЬ В `csg.py` (N = 1600, пять семейств связей):
  * `d = 4` НЕ достигается ни одним семейством (максимумы 2.81…3.84);
  * где размерность выше, там многообразность хуже (`C_4/предск` до 10);
  * где отношения садятся на единицу, размерность равна ДВУМ
    (степенное семейство при a = 0.465: d = 2.040, отношения 0.972 и
    0.967).

ВОПРОС. Райдаут и Соркин показали, что при ПОСТОЯННОМ параметре
размерность падает к единице с ростом `N`, поэтому держать `d` надо
бегущей связью. Тогда:

  1. если при настройке на `d = 2` отношения остаются единицами при
     росте `N` — рост асимптотически двумерно-многообразен;
  2. если максимальная достижимая `d` растёт с `N` — четырёхмерие
     когда-нибудь достижимо, но надо смотреть, что делается с
     отношениями ТАМ.

НАСТРОЙКА. Сетка по параметру, затем интерполяция к нужной `d` по
логарифму. Ошибки — складным ножом по РЕАЛИЗАЦИЯМ.

КОНТРОЛЬ: сприклинг 2D и 4D тем же кодом при каждом `N`.

  python3 scale.py
"""
import math
import time

import numpy as np

from grow import c_k, mm_dim, gen_sprinkle, intervals
from csg import gen_csg, measure, ratios

# степенное семейство: у него и максимум d высок, и единицы достигаются
def mk(a):
    return lambda j, rs: max(1, int(round(j ** a)))


def scan(n, iv, lo, hi, grid, seeds=4):
    pts = []
    for a in grid:
        M = measure(gen_csg, n, iv, lo, hi, seeds,
                    3000 + int(1e4 * float(a)), draw_m=mk(a))
        if len(M) < 2:
            continue
        d = mm_dim(float(M[:, 1].mean()))
        if np.isfinite(d):
            pts.append((float(a), d))
    return np.array(pts) if pts else None


def tune(n, iv, lo, hi, target, grid):
    P = scan(n, iv, lo, hi, grid)
    if P is None or not (P[:, 1].min() <= target <= P[:, 1].max()):
        return None, (P[:, 1].min(), P[:, 1].max()) if P is not None else None
    o = np.argsort(P[:, 1])
    return float(np.interp(target, P[o, 1], P[o, 0])), (P[:, 1].min(),
                                                        P[:, 1].max())


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    LO, HI, IV = 120, 900, 25
    grid = np.linspace(0.10, 0.95, 16)
    print('=' * 94)
    print('КОНТРОЛИ тем же кодом')
    print(f"{'что':>22}{'N':>7}{'d':>16}{'C_3/предск':>18}{'C_4/предск':>18}")
    for n in (1200, 2400):
        for dd in (2, 4):
            M = measure(gen_sprinkle, n, IV, LO, HI, 8, 600 + n + dd, d=dd)
            if len(M) < 3:
                continue
            f, e = ratios(M)
            print(f'{"сприклинг "+str(dd)+"D":>22}{n:>7}'
                  f'{f[0]:>10.3f}±{e[0]:<5.3f}{f[1]:>12.3f}±{e[1]:<5.3f}'
                  f'{f[2]:>12.3f}±{e[2]:<5.3f}')
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 94)
    print('РОСТ, степенное семейство m ~ j^a, НАСТРОЕННОЕ на d = 2')
    print(f"{'N':>7}{'a':>9}{'d':>16}{'C_3/предск':>18}{'C_4/предск':>18}"
          f"{'макс d на сетке':>18}")
    for n in (800, 1600, 3200, 4800):
        a2, rng = tune(n, IV, LO, HI, 2.0, grid)
        if a2 is None:
            print(f'{n:>7}   d = 2 не достигается'
                  + (f', размерность {rng[0]:.2f}…{rng[1]:.2f}' if rng else ''))
            continue
        M = measure(gen_csg, n, IV, LO, HI, 10, 90000 + n, draw_m=mk(a2))
        if len(M) < 3:
            print(f'{n:>7}{a2:>9.4f}   реализаций мало')
            continue
        f, e = ratios(M)
        print(f'{n:>7}{a2:>9.4f}{f[0]:>10.3f}±{e[0]:<5.3f}'
              f'{f[1]:>12.3f}±{e[1]:<5.3f}{f[2]:>12.3f}±{e[2]:<5.3f}'
              f'{rng[1]:>18.3f}')
        print(f'         ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('РОСТ, то же семейство в точке МАКСИМАЛЬНОЙ размерности')
    print('(там, где рост ближе всего к четырёхмерию)')
    print(f"{'N':>7}{'a':>9}{'d':>16}{'C_3/предск':>18}{'C_4/предск':>18}")
    for n in (800, 1600, 3200, 4800):
        P = scan(n, IV, LO, HI, grid)
        if P is None:
            continue
        amax = float(P[np.argmax(P[:, 1]), 0])
        M = measure(gen_csg, n, IV, LO, HI, 10, 95000 + n, draw_m=mk(amax))
        if len(M) < 3:
            continue
        f, e = ratios(M)
        print(f'{n:>7}{amax:>9.4f}{f[0]:>10.3f}±{e[0]:<5.3f}'
              f'{f[1]:>12.3f}±{e[1]:<5.3f}{f[2]:>12.3f}±{e[2]:<5.3f}')
        print(f'         ({time.time()-t0:.0f} с)', flush=True)
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
