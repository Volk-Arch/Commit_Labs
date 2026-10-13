#!/usr/bin/env python3
"""ЕСТЬ ЛИ СИСТЕМАТИКА У СЧЁТА: счёт против ТОЧНОГО объёма, честные ошибки.

ЗАЧЕМ. В `exact.py` счёт разошёлся с точным объёмом на 3-4 сигмы при
`quad 0.004`, но сошёлся на 0.2 сигмы при ВТРОЕ большей кривизне. От
физики так не бывает — значит подозрение на мою же ошибку в ошибке.

ПОЙМАННАЯ ОШИБКА. Ромбы внутри одного засева сильно перекрываются и
делят одни и те же события, то есть НЕ независимы. Я же делил разброс по
ромбам на `sqrt(число ромбов)`, как будто независимы. При 16 ромбах на
засев это занижает ошибку до 4 раз, и 1 сигма выглядит как 4.

ПОЧИНКА. Единица независимости — ЗАСЕВ, а не ромб. Разность
`счёт − точный` усредняется внутри засева, и ошибка берётся по разбросу
МЕЖДУ засевами (складной нож). Все корреляции внутри засева при этом
учтены автоматически, какими бы они ни были.

ПАРНОСТЬ. Разность берётся поромбно, то есть точный объём вычитается из
счёта на ТОМ ЖЕ ромбе. Общие случайные числа убирают разброс геометрии и
оставляют только то, что нас интересует, — смещение счёта.

КОНТРОЛЬ ПЛОСКОГО (par = 0). Тогда точный объём равен `(pi/24)tau⁴`
тождественно, и любое отклонение счёта есть чистая систематика
процедуры: сприклинга, нормировки плотности, условия попадания в ромб.
Это самая строгая проверка, какая у нас есть, потому что сравнивать не с
чем, кроме точного числа.

  python3 bias.py
"""
import math
import time

import numpy as np

from twoterm import LELL, TELL, ETA0, sprinkle, chain, count
from exact import exact_volume


def run(fam, par, seeds=48, step=1.4, nchain=9):
    """Возвращает {(v,g): массив (засевы) средних разностей и средних долей}."""
    gaps = (3, 4)
    vs = (0.0, 0.45, 0.70)
    xoffs = (-4.5, -1.5, 1.5, 4.5)
    per = {}
    for sd in range(seeds):
        X, rho = sprinkle(fam, par, np.random.default_rng(9100 + 53 * sd))
        for v in vs:
            for xo in xoffs:
                C = chain(fam, par, v, step, nchain, xo)
                for g in gaps:
                    tau = g * step
                    base = math.pi / 24 * tau ** 4
                    for i in range(nchain - g):
                        mid = 0.5 * (C[i, 0] + C[i + g, 0])
                        if abs(mid - ETA0) > 4.0:
                            continue
                        de = C[i + g, 0] - C[i, 0]
                        xlo = min(C[i, 1], C[i + g, 1]) - de / 2
                        xhi = max(C[i, 1], C[i + g, 1]) + de / 2
                        if xlo < 0.5 or xhi > LELL - 0.5:
                            continue
                        if C[i, 0] < 0.5 or C[i + g, 0] > TELL - 0.5:
                            continue
                        Vc = count(X, C[i], C[i + g]) / rho
                        Ve = exact_volume(C[i], C[i + g], fam, par)
                        per.setdefault((v, g), {}).setdefault(sd, []).append(
                            ((Vc - Ve) / base, Vc / Ve - 1.0))
        del X
    out = {}
    for k, d in per.items():
        out[k] = np.array([np.mean(d[s], axis=0) for s in sorted(d)])
    return out


def jack(y):
    """Среднее и ошибка складным ножом по засевам."""
    n = len(y)
    m = y.mean()
    jk = np.array([np.delete(y, i).mean() for i in range(n)])
    se = math.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum())
    return m, se


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    print('=' * 72)
    for fam, par, tag in (('lin', 0.0, 'КОНТРОЛЬ ПЛОСКОГО'),
                          ('lin', 0.06, 'линейное (R = 0)'),
                          ('quad', 0.004, 'квадратичное слабое'),
                          ('quad', 0.012, 'квадратичное сильное')):
        print(f'--- {tag}: fam = {fam}, параметр = {par} ---')
        out = run(fam, par)
        print(f"{'v':>6}{'g':>3}{'засевов':>9}{'(счёт−точн)/база':>19}"
              f"{'ст.ош':>10}{'сигм':>7}{'счёт/точн−1':>14}")
        allrows = []
        for k in sorted(out):
            y = out[k]
            m, se = jack(y[:, 0])
            r, rse = jack(y[:, 1])
            allrows.append((m, se))
            print(f'{k[0]:>6.2f}{k[1]:>3}{len(y):>9}{m:>+19.5f}{se:>10.5f}'
                  f'{abs(m)/max(se,1e-12):>7.1f}{r:>+13.4f}')
        worst = max(abs(m) / max(se, 1e-12) for m, se in allrows)
        print(f'  худшее отклонение: {worst:.1f} сигм'
              f'   ({time.time()-t0:.0f} с)')
        print(flush=True)
    print('=' * 72)
    print('ЧТЕНИЕ. Если худшее по всем строкам около 1-2 сигм, то счёт')
    print('воспроизводит ТОЧНЫЙ непрерывный объём без систематики, и весь')
    print('промах в A и B идёт от обрезания формулы, а не от причинного')
    print('множества. Если же систематика есть и в плоском контроле — она')
    print('в самой процедуре, и тогда чинить надо её.')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
