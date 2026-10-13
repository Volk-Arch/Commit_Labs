#!/usr/bin/env python3
"""СИСТЕМАТИКА ОПОРЫ: насколько её шум сдвигает `w`.

Два прогона с РАЗНЫМИ засевами опоры дали `w` = +0.3498 (`wreal.py`) и
+0.3991 (`wdot.py`, обе половины) на одних и тех же данных. Сдвиг 0.049
втрое больше заявленной ошибки +-0.0160. И `chi²` не сходится: сумма
половин 41.8 против 12.0 у полной подгонки, чего на одних данных с одной
опорой быть не может.

Разница между прогонами одна: засев опоры. Значит шум квадратуры в
опорной кривой переносится в `w`, а я его не переносил — оценивал по
ДВУМ блокам, по которым и сама оценка погрешности негодна.

ЗДЕСЬ перенос считается прямо, без новых прогонов. У минимума `chi²`

    delta p = sum_i u_i dR_i / sum_i u_i (dR_i/dp),   u_i = (dR_i/dp)/s_i²

откуда вклад шума опоры `E_i` в ошибку `p`:

    sigma_p(опора)² = sum_i u_i² E_i² / ( sum_i u_i dR_i/dp )²

Сравнивается с ошибкой от данных, полученной складным ножом по мирам.

  python3 refsys.py
"""
import math

import numpy as np

from pencil import KFAC, IX2, IX3
from refit import load_all
from wreal import ETA0, PGRID, cells, ref_grid, fit_p, w_of


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    arms = load_all()
    eta_s = ETA0 - 1.0 / 0.035
    A, sd, par = arms['lin']
    cl = cells(A, sd)
    err = np.array([c['err'] for c in cl])
    print(f'плечо lin, ячеек {len(cl)}; типичная ошибка ДАННЫХ в ячейке: '
          f'X_2 {np.median(err[:,0]):.5f}, X_3 {np.median(err[:,1]):.5f}')
    print()
    print('опора с двумя засевами и большим числом блоков...', flush=True)
    res = []
    for tag, seed, nn, bl in (('засев A', 777111, 1100000, 4),
                              ('засев B', 999222, 1100000, 4)):
        rs = np.random.default_rng(seed)
        R, E = ref_grid(cl, eta_s, nn, rs, blocks=bl)
        obs = np.array([c['obs'] for c in cl])
        pp, chi, _ = fit_p(obs, err, R, E)
        res.append((tag, pp, chi, R, E))
        print(f'  {tag}: p = {pp:.4f}, w = {w_of(pp):+.4f}, '
              f'chi² = {chi:.1f}; типичный шум опоры '
              f'{np.median(E):.5f}', flush=True)
    print()
    print('=' * 74)
    print('РАЗБРОС МЕЖДУ ЗАСЕВАМИ ОПОРЫ')
    d = w_of(res[0][1]) - w_of(res[1][1])
    print(f'  w(A) = {w_of(res[0][1]):+.4f}, w(B) = {w_of(res[1][1]):+.4f}, '
          f'разность {d:+.4f}')
    print()
    print('ПЕРЕНОС ШУМА ОПОРЫ АНАЛИТИЧЕСКИ')
    R, E = res[0][3], res[0][4]
    pp = res[0][1]
    k = int(np.argmin(np.abs(PGRID - pp)))
    k = min(max(k, 1), len(PGRID) - 2)
    dR = (R[k + 1] - R[k - 1]) / (PGRID[k + 1] - PGRID[k - 1])
    s2 = err ** 2 + E[k] ** 2
    u = dR / s2
    den = (u * dR).sum()
    sp_ref = math.sqrt(((u ** 2) * E[k] ** 2).sum()) / abs(den)
    sp_dat = math.sqrt(((u ** 2) * err ** 2).sum()) / abs(den)
    for nm, sp in (('от данных', sp_dat), ('от шума опоры', sp_ref)):
        ew = abs(w_of(pp + sp) - w_of(pp - sp)) / 2
        print(f'  sigma_p {nm:>14} = {sp:.4f}  ->  sigma_w = {ew:.4f}')
    tot = math.sqrt(sp_dat ** 2 + sp_ref ** 2)
    ewt = abs(w_of(pp + tot) - w_of(pp - tot)) / 2
    print(f'  полная: sigma_p = {tot:.4f}  ->  sigma_w = {ewt:.4f}')
    print()
    print(f'''ИТОГ. Заявленная вчера ошибка +-0.0160 учитывала только
данные. Полная, с переносом шума опоры, равна +-{ewt:.4f}. Чтобы шум
опоры стал пренебрежим, нужно поднять число точек квадратуры в
{(sp_ref/sp_dat)**2 * 4:.0f} раз относительно нынешних {nn} x {bl}.''')


if __name__ == '__main__':
    main()
