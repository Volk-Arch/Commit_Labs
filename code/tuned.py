#!/usr/bin/env python3
"""НАСТРОЕННАЯ ПЕРКОЛЯЦИЯ: подогнать размерность по C_2 и проверить C_3, C_4.

ЗАМЫСЕЛ. Размерность Мирхейма–Мейера подгоняется ОДНИМ числом `p` так,
чтобы вышло `d = 4`. После этого `C_3` и `C_4` — чистая проверка без
свободных чисел: если множество похоже на четырёхмерное многообразие,
отношения замер/предсказание обязаны быть единицами.

Это и есть количественная мера удалённости от многообразности. По
отчётам поиска литературы, обилия цепей к множествам из роста не
прикладывал никто.

ИСПРАВЛЕНИЕ ОШИБКИ ПРЕДЫДУЩЕГО ПРОГОНА. В `grow.py` ошибка считалась по
разбросу интервалов ВНУТРИ одного множества, а они делят одни и те же
события и независимыми не являются. Здесь единица независимости —
РЕАЛИЗАЦИЯ, и ошибка берётся складным ножом по ним.

ЧТО ПРОВЕРЯЕТСЯ:
  1. КОНТРОЛЬ: сприклинг 4D, прогнанный тем же кодом, обязан дать
     единицы;
  2. при настроенном `p` — отношения `C_3`, `C_4`;
  3. ЗАВИСИМОСТЬ ОТ РАЗМЕРА. Если отклонение есть краевой эффект, оно
     обязано убывать с ростом `N`. Если оно структурно — останется.
     Это и решает, многообразна ли перколяция асимптотически.

  python3 tuned.py
"""
import math
import time

import numpy as np

from grow import (c_k, mm_dim, spectrum, gen_sprinkle, gen_percolation,
                  intervals)


def measure(gen, n, iv, lo, hi, seeds, root, **kw):
    """Спектр по РЕАЛИЗАЦИЯМ: список (d, h2, h3, h4) на реализацию."""
    out = []
    for s in range(seeds):
        A = gen(n, rs=np.random.default_rng(root + 17 * s), **kw)
        rows = intervals(A, iv, lo, hi, np.random.default_rng(root + 999 + s))
        del A
        if not rows:
            continue
        R = np.array(rows, dtype=float)
        c1 = R[:, 0]
        out.append((c1.mean(),
                    float((R[:, 1] / (c1 * (c1 - 1))).mean()),
                    float((R[:, 2] / (c1 * (c1 - 1) * (c1 - 2))).mean()),
                    float((R[:, 3] / (c1 * (c1 - 1) * (c1 - 2)
                                      * (c1 - 3))).mean())))
    return np.array(out)


def ratios(M):
    """(d, C_3/предск, C_4/предск) со складным ножом по реализациям."""
    def one(S):
        d = mm_dim(float(S[:, 1].mean()))
        if not np.isfinite(d):
            return np.array([np.nan] * 3)
        return np.array([d,
                         float(S[:, 2].mean()) / c_k(3, d),
                         float(S[:, 3].mean()) / c_k(4, d)])
    full = one(M)
    k = len(M)
    jk = np.array([one(np.delete(M, i, axis=0)) for i in range(k)])
    se = np.sqrt((k - 1) / k * ((jk - jk.mean(0)) ** 2).sum(0))
    return full, se


def show(tag, M, c1=None):
    if len(M) < 3:
        print(f'  {tag:>34}: реализаций мало ({len(M)})')
        return None
    f, e = ratios(M)
    if not np.isfinite(f[0]):
        print(f'  {tag:>34}: C_2/C_1² = {M[:,1].mean():.5f} вне области')
        return None
    print(f'  {tag:>34}: C_1 {M[:,0].mean():>6.0f}  d {f[0]:>6.3f}±{e[0]:<5.3f}'
          f'  C_3/предск {f[1]:>6.3f}±{e[1]:<5.3f}'
          f'  C_4/предск {f[2]:>6.3f}±{e[2]:<5.3f}')
    return f, e


def find_p(n, target, iv, lo, hi, seeds=4, root=11):
    """Подбор p под заданную размерность: d убывает с p, значит деление."""
    plo, phi = 2e-4, 0.5
    for _ in range(18):
        pm = math.sqrt(plo * phi)
        M = measure(gen_percolation, n, iv, lo, hi, seeds, root, p=pm)
        if len(M) < 2:
            plo = pm
            continue
        d = mm_dim(float(M[:, 1].mean()))
        if not np.isfinite(d):
            plo = pm
            continue
        if d > target:
            plo = pm
        else:
            phi = pm
    return math.sqrt(plo * phi)


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    LO, HI, IV = 120, 900, 30
    print('=' * 86)
    print('КОНТРОЛЬ: сприклинг 4D тем же кодом, ошибка по реализациям')
    for n in (900, 1800):
        M = measure(gen_sprinkle, n, IV, LO, HI, 10, 400 + n, d=4)
        show(f'сприклинг 4D, N = {n}', M)
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 86)
    print('НАСТРОЙКА p ПОД d = 4 И ПРОВЕРКА C_3, C_4')
    print('p подбирается по C_2; C_3 и C_4 после этого — чистая проверка.')
    print()
    for n in (800, 1600, 3200):
        p = find_p(n, 4.0, IV, LO, HI)
        M = measure(gen_percolation, n, IV, LO, HI, 12, 5000 + n, p=p)
        show(f'перколяция N = {n}, p = {p:.5f}', M)
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('ЧТЕНИЕ. Если C_3 и C_4 близки к единице и отклонение убывает с')
    print('N — перколяция асимптотически похожа на четырёхмерное')
    print('многообразие. Если отклонение держится или растёт — она')
    print('структурно не многообразна, и величина отклонения есть мера.')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
