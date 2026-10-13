#!/usr/bin/env python3
"""ТРЕТЬЕ СЕМЕЙСТВО: набор кусками, чтобы ломать вырождение плана.

ЗАЧЕМ. Компоненты `R_etaeta` и `R_xx` порознь у нас не разводятся:
столбцы плана дают `g1 − g2 = sqrt(C1)` ТОЖДЕСТВЕННО, число
обусловленности 3.7e8, и отношение у линейного семейства выходит с
НЕВЕРНЫМ знаком (−0.299 ± 0.247 против истины +1/3). Статистикой это не
лечится — нужен другой план.

ОТКУДА ТРЕТЬЕ СЕМЕЙСТВО. При `u = a''/a` и `v = a'^2/a^2`:
`R_etaeta = 3(v − u)`, `R_xx = u + v`. Наши два семейства дают `u = 0`
(линейное, отношение +1/3) и `v = 0` в центре (квадратичное, −1/3), то
есть рычаг одинаков по модулю. А при `u = v` отношение уходит в
бесконечность, и условие решается точно: `a''a = a'^2` даёт

    a = e^{k eta}   =>   R_etaeta = 0 ТОЖДЕСТВЕННО,  R_xx = 2k^2

Проверено символьно от метрики (Кристоффель -> Риччи), расхождение 0.

ЧЕМ ЭТО ЛУЧШЕ. Отклик `R_uu = [R_etaeta + R_xx v^2]/(a^2(1-v^2))` у
экспоненты обращается в НУЛЬ при `v = 0` и растёт как `v^2/(1-v^2)`, а у
линейного семейства при `v = 0` он МАКСИМАЛЕН. Ход по наклону
качественно другой, поэтому вырождение ломается формой, а не диапазоном
`C1`. И строки этого плеча ограничивают ТОЛЬКО `beta`, потому что
`alpha = 0` по построению.

ВЫБОР k. Берётся `k = 0.03`, то есть `R_xx = 0.0018` — вдвое больше, чем
у линейного семейства (0.00123), и это как раз та компонента, которая
меряется хуже всех. Проверка применимости: при `C1 = 900` собственное
время `T = 9.1`, отклик при `v^2 = 0.77` равен 0.006, значит
`T^2 R_uu = 0.50` — внутри окна первого порядка (предел около 0.8).

  python3 expfam.py --seeds 30 --seed0 0
"""
import argparse
import math
import pickle
import time

import numpy as np

from pencil import SCRATCH, VB, harvest, truth

CFG = (26.0, 20.0, 10.0, 1.6, 200, 900)
# ШИРОКОЕ ОКНО по C_1, добавлено 8 октября. Вырождение плана идёт от
# того, что sqrt(C1) на узком диапазоне плохо отличима от комбинации
# столбцов помехи {1, 1/sqrt(C1)}. Арифметика плана (без набора) даёт
# ошибку отношения beta/alpha: 0.172 при 200..900, 0.044 при 50..900,
# 0.021 при 50..2000 — то есть восьмикратный выигрыш. При C_1 = 2000
# собственное время T = 11.1, что влезает и по времени (ящик 20), и по
# пространству (26).
CFGW = (26.0, 20.0, 10.0, 1.6, 50, 2000)
KEXP = 0.03
# четвёртое семейство: R_xx = 0 тождественно, R_etaeta = 3c^2/2.
# c = 0.035 даёт R_etaeta = 0.00184, и при v^2 = 0.77 выходит
# T^2 R_uu = 0.66 — внутри окна первого порядка (предел около 0.8).
KROOT = 0.035


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--per', type=int, default=420)
    ap.add_argument('--fam', default='exp',
                    choices=('exp', 'root', 'lin', 'quad', 'flat'))
    ap.add_argument('--wide', action='store_true',
                    help='широкое окно C_1 = 50..2000 вместо 200..900')
    a = ap.parse_args()
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    PARS = {'exp': KEXP, 'root': KROOT, 'lin': 0.035, 'quad': 0.0012,
            'flat': 0.0}
    par = PARS[a.fam]
    cfg = CFGW if a.wide else CFG
    print(f'окно C_1 = {cfg[4]}..{cfg[5]}'
          + ('  (ШИРОКОЕ)' if a.wide else ''))
    print(f'семейство {a.fam}, параметр {par}, '
          f'истина (R_etaeta, R_xx, R) = {truth(a.fam, par)}')
    print(f'засевы {a.seed0}..{a.seed0 + a.seeds - 1}, по {a.per} интервалов')
    rows, sdi, tot = [], [], None
    for s in range(a.seed0, a.seed0 + a.seeds):
        rs = np.random.default_rng(310000 + 1931 * s + int(1e7 * par))
        r, got = harvest(a.fam, par, cfg, rs, a.per)
        rows += r
        sdi += [s] * len(r)
        tot = got if tot is None else [x + y for x, y in zip(tot, got)]
        if (s - a.seed0 + 1) % 10 == 0:
            print(f'  засевов {s - a.seed0 + 1}, интервалов {len(rows)}, '
                  f'{time.time()-t0:.0f} с', flush=True)
    A = np.array(rows)
    tagw = 'w' if a.wide else ''
    out = f'{SCRATCH}/{a.fam}fam{tagw}_{a.seed0}.pkl'
    with open(out, 'wb') as fh:
        pickle.dump({'A': A, 'sd': np.array(sdi), 'k': par,
                     'fam': a.fam,
                     'cfg': cfg, 'per': a.per,
                     'wide': bool(a.wide)}, fh)
    print()
    print(f'{len(A)} интервалов, по полосам v²: {tot}')
    print(f"{'v²':>8}{'шт':>7}{'C_1':>7}{'X_2':>10}{'ст.ош':>9}"
          f"{'X_3':>10}{'ст.ош':>9}")
    sd = np.array(sdi)
    for i in range(len(VB) - 1):
        m = (A[:, 3] >= VB[i]) & (A[:, 3] < VB[i + 1])
        if m.sum() < 20:
            continue
        us = np.unique(sd[m])
        j2 = np.array([A[m & (sd != u), 1].mean() for u in us])
        j3 = np.array([A[m & (sd != u), 2].mean() for u in us])
        n = len(us)
        e2 = math.sqrt((n - 1) / n * ((j2 - j2.mean()) ** 2).sum())
        e3 = math.sqrt((n - 1) / n * ((j3 - j3.mean()) ** 2).sum())
        print(f'{A[m,3].mean():>8.3f}{m.sum():>7}{A[m,0].mean():>7.0f}'
              f'{A[m,1].mean():>+10.4f}{e2:>9.4f}'
              f'{A[m,2].mean():>+10.4f}{e3:>9.4f}')
    print()
    print('ОЖИДАНИЕ: отклик обязан РАСТИ с v² от почти нуля при малых')
    print('наклонах — это подпись R_etaeta = 0, и её нечем подделать.')
    print(f'  сохранено: {out}, время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
