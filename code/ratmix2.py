#!/usr/bin/env python3
"""СМЕСИ ВРАГА: ЛП С ЗАКРЫТОЙ ЛАЗЕЙКОЙ МАСШТАБА.

ratmix.py дал s = 0.0 сигм, и это была МОЯ ошибка, а не победа врага.
Контроль её и выдал: он поставил на столбец независимого засева вес
0.000 и всё равно отчитался о точном совпадении - чего быть не может.

Диагноз: условие на ОТНОШЕНИЯ я наложил в АБСОЛЮТНЫХ единицах с
абсолютными ошибками, при свободной переменной масштаба c. ЛП уводил c
к нулю: тогда все N_m смеси малы, невязки в единицы отсчётов, а ошибки
цели - сотни. Лазейка масштаба.

Починка: переменную c и нормировку sum lambda = 1 убрать, оставив
НЕОТРИЦАТЕЛЬНУЮ комбинацию против ФИКСИРОВАННОЙ цели:
    min s  при  |sum_i lambda_i N^(i)_m - N^цель_m| <= s * sigma_m,
               lambda >= 0
Масштаб поглощается самой суммой весов (это и есть «с точностью до
общего множителя»), а увести его к нулю нельзя: цель стоит справа и
никуда не уходит.

И решение проверяется ПОДСТАНОВКОЙ - иначе это не результат.

КОНТРОЛИ:
  (a) в набор добавляется независимый засев: ЛП обязано найти s ~ 1 и
      положить вес ~1 именно на него;
  (b) невязки печатаются по каждому m из возвращённого решения.
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
    nc = len(cols)
    A = np.array(cols).T
    ub_A, ub_b = [], []
    for m in range(M):
        row = np.zeros(nc + 1)
        row[:nc] = A[m]
        row[nc] = -sig[m]
        ub_A.append(row)
        ub_b.append(tgt[m])
        row2 = np.zeros(nc + 1)
        row2[:nc] = -A[m]
        row2[nc] = -sig[m]
        ub_A.append(row2)
        ub_b.append(-tgt[m])
    c = np.zeros(nc + 1)
    c[nc] = 1.0
    return linprog(c, A_ub=np.array(ub_A), b_ub=np.array(ub_b),
                   bounds=[(0, None)] * nc + [(0, None)], method='highs')


def check(r, cols, tgt, sig, labs):
    lam = r.x[:len(cols)]
    mix = (np.array(cols).T @ lam)
    res = (mix - tgt) / sig
    print('    невязки по m (подстановка): '
          + ' '.join(f'{v:+.1f}' for v in res))
    print(f'    наибольшая |невязка| = {np.abs(res).max():.1f}σ, '
          f'заявлено s = {r.x[-1]:.1f}σ')
    top = np.argsort(lam)[::-1][:3]
    print('    вес на: ' + ', '.join(f'{labs[i]} ({lam[i]:.3f})'
                                     for i in top if lam[i] > 1e-6))
    return float(np.abs(res).max())


def main():
    print(__doc__)
    rs = np.random.default_rng(2645751)
    n = 1200
    tgt, terr = vec(lambda: sprinkle_d(n, 4, rs), 10)
    ctl, cerr = vec(lambda: sprinkle_d(n, 4, rs), 10)
    sig = np.hypot(terr, cerr)
    print('цель N_m: ' + ' '.join(f'{v:.0f}' for v in tgt))
    print('сигмы:    ' + ' '.join(f'{v:.0f}' for v in sig))

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
    print(f'\nстолбцов врага: {len(cols)}')

    print('\nКОНТРОЛЬ (a): набор ПЛЮС независимый засев')
    r2 = lp(cols + [ctl], tgt, sig, )
    print(f'  ЛП: {"решено" if r2.success else "НЕ СОШЛАСЬ"}')
    w = check(r2, cols + [ctl], tgt, sig, labs + ['ЗАСЕВ'])
    print(f'  => {"ОК" if w < 3 else "СБОЙ"}')

    print('\nВРАГ, смесь по всему семейству (без засева):')
    r = lp(cols, tgt, sig)
    print(f'  ЛП: {"решено" if r.success else "НЕ СОШЛАСЬ"}')
    w2 = check(r, cols, tgt, sig, labs)
    print(f'\nИТОГ: контроль {w:.1f}σ, враг {w2:.1f}σ на {M} условиях')
    print('ЧТЕНИЕ: контроль ~1, враг велик => девять чисел не берутся')
    print('даже смесью по всему семейству.')


if __name__ == '__main__':
    main()
