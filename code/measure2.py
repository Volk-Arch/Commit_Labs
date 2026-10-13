#!/usr/bin/env python3
"""ГДЕ МЕРА РОСТА ДАЁТ РАЗМЕРНОСТЬ 4, И КАК ТАМ РАСТЁТ СЧЁТ.

Точное сравнение мер (`measure.py`, контроль `sum Pr = 1` сошёлся до
1e-10) показало: у ОДНОРОДНОЙ меры типичный порядок двумерен
(`d` Мирхейма = 2.22 при `N = 7`), а у меры роста есть `p` между 0.05 и
0.15, где он ЧЕТЫРЁХМЕРЕН. Эффективное число порядков при этом падает в
полторы тысячи раз (63.5 против 96428).

Это и есть содержание задачи меры: иерархия `e^{0.173 N²}` против
`e^{d N lnN}` считалась на ОДНОРОДНОЙ мере, а мера роста ансамбль резко
сужает И смещает вес к нужной размерности.

ЗДЕСЬ уточняется:
  1. где именно стоит `p*`, при котором `d` Мирхейма равна 4, и
     устойчиво ли оно по `N`;
  2. как растёт эффективное число порядков `exp(энтропия)` — у
     однородной меры оно идёт как `e^{c N²}`, и если у меры роста
     медленнее, иерархия переворачивается;
  3. каково действие `S/N²` в этой точке — то есть попадает ли мера
     роста на плато `O(1)`, где живёт пространство-время.

КОНТРОЛЬ: при `p = 1/2` мера НЕ однородна (вопреки интуиции): вес
`(1/2)^{L} (1/2)^{M-R} = (1/2)^{L+M-R}` зависит от порядка через
`L - R`. Это проверяется прямо — доля `r` при `p = 1/2` обязана
отличаться от однородной.

  python3 measure2.py
"""
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma

from ensemble import enumerate_orders


def c2(d):
    return gamma(d + 1) * gamma(d / 2) * gamma(d) / (
        4.0 * gamma(d) * gamma(3 * d / 2))


def mmdim(r):
    try:
        return brentq(lambda d: 2 * c2(d) - r, 0.4, 40.0)
    except Exception:
        return float('nan')


R4 = 2 * c2(4.0)


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    print(f'доля упорядоченных пар при d = 4: r = 2 c_2(4) = {R4:.6f}')
    print()
    print('=' * 78)
    print(f"{'N':>4}{'порядков':>10}{'p* (d=4)':>11}{'r в p*':>9}"
          f"{'экв. порядков':>15}{'S/N² в p*':>12}{'высота':>8}")
    rows = []
    for n in (4, 5, 6, 7):
        d = enumerate_orders(n)
        rel = d['rel'].astype(float)
        N0 = d['N0'].astype(float)
        hgt = d['hgt'].astype(float)
        S = d['S2']
        M = n * (n - 1) // 2

        def rof(p):
            lp = N0 * math.log(p) + (M - rel) * math.log1p(-p)
            w = np.exp(lp - lp.max()); w /= w.sum()
            return (w * rel).sum() / M, w

        try:
            ps = brentq(lambda p: rof(p)[0] - R4, 0.005, 0.95)
        except Exception:
            ps = float('nan')
        r0, w = rof(ps)
        ent = -(w[w > 0] * np.log(w[w > 0])).sum()
        rows.append((n, ps, ent))
        print(f'{n:>4}{len(rel):>10}{ps:>11.4f}{r0:>9.4f}'
              f'{math.exp(ent):>15.1f}{(w*S).sum()/n**2:>12.4f}'
              f'{(w*hgt).sum():>8.2f}', flush=True)
        # заодно однородная для сравнения
        wu = np.ones(len(rel)) / len(rel)
        if n == 7:
            print(f'{"одн.":>4}{len(rel):>10}{"—":>11}'
                  f'{(wu*rel).sum()/M:>9.4f}{len(rel):>15.1f}'
                  f'{(wu*S).sum()/n**2:>12.4f}{(wu*hgt).sum():>8.2f}')
    print()
    print('=' * 78)
    print('РОСТ ЭФФЕКТИВНОГО ЧИСЛА ПОРЯДКОВ')
    A = np.array(rows)
    ln = np.log(A[:, 0])
    for nm, y in (('в точке p*', A[:, 2]),):
        a2 = np.polyfit(A[:, 0] ** 2, y, 1)[0]
        anl = np.polyfit(A[:, 0] * np.log(A[:, 0]), y, 1)[0]
        r2_2 = np.corrcoef(A[:, 0] ** 2, y)[0, 1] ** 2
        r2_n = np.corrcoef(A[:, 0] * np.log(A[:, 0]), y)[0, 1] ** 2
        print(f'  {nm}: энтропия против N²   наклон {a2:.5f}, R² = {r2_2:.5f}')
        print(f'  {nm}: энтропия против NlnN наклон {anl:.5f}, R² = {r2_n:.5f}')
    ent_u = [math.log(len(enumerate_orders(n)['rel'])) for n in (4, 5, 6, 7)]
    Nv = np.array([4, 5, 6, 7], float)
    print(f'  однородная: против N²   R² = '
          f'{np.corrcoef(Nv**2, ent_u)[0,1]**2:.5f}')
    print(f'  однородная: против NlnN R² = '
          f'{np.corrcoef(Nv*np.log(Nv), ent_u)[0,1]**2:.5f}')
    print()
    print('КОНТРОЛЬ: p = 1/2 — мера НЕ однородна')
    d = enumerate_orders(7)
    rel = d['rel'].astype(float); N0 = d['N0'].astype(float)
    M = 21
    lp = N0 * math.log(0.5) + (M - rel) * math.log(0.5)
    w = np.exp(lp - lp.max()); w /= w.sum()
    wu = np.ones(len(rel)) / len(rel)
    print(f'  доля r: перколяция 0.5 = {(w*rel).sum()/M:.4f}, '
          f'однородная = {(wu*rel).sum()/M:.4f}  -> '
          f'{"различаются, как и должно" if abs((w*rel).sum()-(wu*rel).sum())>1e-6 else "СОВПАЛИ, ошибка"}')
    print()
    print('''ЧТЕНИЕ. Если `p*` устойчиво по `N`, а эффективное число
порядков в этой точке растёт как `N lnN`, а не как `N²` — то иерархия
счёта на мере роста ПЕРЕВЁРНУТА относительно однородной, и «подавляющее
большинство непохожих порядков» было артефактом выбора меры.''')


if __name__ == '__main__':
    main()
