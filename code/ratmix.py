#!/usr/bin/env python3
"""СМЕСИ ВРАГА ПРОТИВ ВОСЬМИ ОТНОШЕНИЙ: ЛИНЕЙНАЯ ЗАДАЧА.

ratios2.py: враг с полной подгонкой по восьми отношениям дал хи2 = 979.7
против 9.7 у второго независимого засева, и промахнулся вдобавок по доле
связанных (0.0015 против 0.1000). Но два возражения я обязан снять сам:

  (1) оптимум сел на границу сетки (k=34, delta=0.5) - трижды за день
      граница сетки уже делала из артефакта «вывод»;
  (2) у врага отнята свобода СМЕСЕЙ, которой он спасся в census. Это
      возражение серьёзнее: отношение есть отношение средних, и смесь
      достигает значений, недостижимых ни одним членом.

Второе сводится к ЛИНЕЙНОЙ задаче. Смесь воспроизводит отношения, если
    sum_i lambda_i N^(i) = c * цель,   lambda >= 0, sum lambda = 1, c > 0
- линейно по (lambda, c). Масштаб c свободен, значит из девяти равенств
(m = 0..8) остаётся ВОСЕМЬ условий, против смеси по всему семейству.

Решаем как ЛП на минимум максимального отклонения в единицах сигм:
    min s  при  |sum_i lambda_i N^(i) - c * цель|_m <= s * sigma_m
Тогда s есть ответ в сигмах, а не «совместимо/нет».

КОНТРОЛЬ, обязательный: в набор столбцов добавляется НЕЗАВИСИМЫЙ ЗАСЕВ.
Если ЛП не находит s ~ 1, сломан код, а не физика.
"""
import math

import numpy as np
from scipy.optimize import linprog

from sectors import sprinkle_d
from lkdeg import layered
from ratios import nm, MMAX

M = MMAX + 1


def vec(mk, rep):
    V = np.array([nm(mk())[0] for _ in range(rep)])
    return V.mean(axis=0), V.std(axis=0) / math.sqrt(rep)


def lp(cols, tgt, sig):
    """min s : |sum l_i c_i - c tgt| <= s sig, l>=0, sum l = 1, c>=0."""
    nc = len(cols)
    A = np.array(cols).T                      # M x nc
    nv = nc + 2                               # lambda, c, s
    ub_A, ub_b = [], []
    for m in range(M):
        row = np.zeros(nv)
        row[:nc] = A[m]
        row[nc] = -tgt[m]
        row[nc + 1] = -sig[m]
        ub_A.append(row.copy())
        ub_b.append(0.0)
        ub_A.append(-row + 2 * np.array(
            [0.0] * nc + [0.0, -sig[m]]) * 0)
        ub_b.append(0.0)
    for i in range(M):
        ub_A[2 * i + 1] = -ub_A[2 * i].copy()
        ub_A[2 * i + 1][nc + 1] = -sig[i]
    eq = np.zeros((1, nv))
    eq[0, :nc] = 1.0
    c = np.zeros(nv)
    c[nc + 1] = 1.0
    res = linprog(c, A_ub=np.array(ub_A), b_ub=np.array(ub_b),
                  A_eq=eq, b_eq=[1.0],
                  bounds=[(0, None)] * nc + [(1e-6, None), (0, None)],
                  method='highs')
    return res


def main():
    print(__doc__)
    rs = np.random.default_rng(2236067)
    n = 1200
    tgt, terr = vec(lambda: sprinkle_d(n, 4, rs), 10)
    ctl, cerr = vec(lambda: sprinkle_d(n, 4, rs), 10)
    sig = np.hypot(terr, cerr)
    print(f'цель (засев d=4, N={n}): N_m = '
          + ' '.join(f'{v:.0f}' for v in tgt))

    cols, labs = [], []
    for k in (3, 4, 6, 9, 13, 18, 25, 34, 48, 64):
        for dl in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
            q = dl * k / n
            if q >= 1.0:
                continue
            v, _ = vec(lambda: layered(n, k, q, rs), 2)
            if v[0] < 1.0:
                continue
            cols.append(v)
            labs.append(f'k={k} d={dl}')
    print(f'столбцов врага: {len(cols)}')

    r = lp(cols, tgt, sig)
    print(f'\nВРАГ, смесь по всему семейству: '
          f'{"решено" if r.success else "ЛП не сошлась"}, '
          f's = {r.x[-1]:.1f}σ' if r.success else '')
    if r.success:
        lam = r.x[:len(cols)]
        top = np.argsort(lam)[::-1][:4]
        print('  вес смеси на: '
              + ', '.join(f'{labs[i]} ({lam[i]:.2f})'
                          for i in top if lam[i] > 1e-3))

    r2 = lp(cols + [ctl], tgt, sig)
    print(f'\nКОНТРОЛЬ, тот же набор ПЛЮС независимый засев: '
          f's = {r2.x[-1]:.1f}σ'
          f'  => {"ОК" if r2.success and r2.x[-1] < 3 else "СБОЙ"}')
    if r2.success:
        lam = r2.x[:len(cols) + 1]
        print(f'  вес на засеве: {lam[-1]:.3f} '
              f'(остальное: {1 - lam[-1]:.3f})')
    print('\nЧТЕНИЕ: контроль около 1σ, а враг велик => восемь условий')
    print('не берутся даже СМЕСЬЮ, то есть дело в счёте условий.')


if __name__ == '__main__':
    main()
