#!/usr/bin/env python3
"""Что порядок помнит о своём входе: ящик против конуса.

ORDER-final.md, часть VI: на НЕЗАВИСИМЫХ шкалах порядок помнит о входе
ровно два числа — сумму шкал (R² = 0.948, истолкована как время) и
разброс шкал (R² = 0.909, истолкован как заряд). Отдельные шкалы не
восстанавливаются (0.121, 0.045), третьей величины нет.

На СВЯЗАННЫХ шкалах вход другой: точка в m измерениях, а шкала
направления u есть t − u·x. Тогда симметричные функции шкал имеют прямой
геометрический смысл:

    среднее по направлениям   = t            (средний u·x равен нулю)
    разброс по направлениям   ~ |x|²         (дисперсия проекций)

То есть второе запомненное число должно оказаться не зарядом, а
РАССТОЯНИЕМ ОТ НАЧАЛА. Это проверяется той же процедурой, что в архиве:
восстановить величину из признаков, посчитанных ТОЛЬКО из порядка.

Контроль обязателен: та же область, та же граница, два правила. Иначе
восстановимость может идти от края мира, а не от памяти порядка.

  python3 inputmem.py
"""
import argparse
import math

import numpy as np

from core import rel_of, links_of, height
from coneworld import sprinkle, cone_rel


def order_features(R):
    """Признаки события, считанные только из порядка."""
    Rb = np.asarray(R, dtype=bool)
    past = Rb.sum(1).astype(float)          # R[i,j]: j раньше i
    fut = Rb.sum(0).astype(float)
    L = links_of(Rb)
    lin = L.sum(1).astype(float)
    lout = L.sum(0).astype(float)
    depth = height(Rb)[1].astype(float)
    n = len(Rb)
    F = np.column_stack([
        past, fut, past - fut, past + fut, past * fut / max(n, 1),
        lin, lout, lin - lout, lin + lout, depth,
        np.sqrt(past + 1), np.sqrt(fut + 1), depth ** 2,
    ])
    return F


def r2(F, y):
    """Доля объяснённой дисперсии при линейной регрессии на признаки."""
    y = np.asarray(y, float)
    A = np.column_stack([F, np.ones(len(F))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss = ((y - y.mean()) ** 2).sum()
    return 1.0 - (resid ** 2).sum() / ss if ss > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser(description='Память порядка о входе')
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--seeds', type=int, default=4)
    a = ap.parse_args()

    print(f'N={a.n}, k={a.k}, семян {a.seeds}')
    print('признаки считаются ТОЛЬКО из порядка; восстанавливаются величины входа')
    print()

    rows = {}
    for world in ('ящик (независимые шкалы)', 'конус (проекции точки)'):
        acc = {}
        for s in range(a.seeds):
            rs = np.random.default_rng(100 + 17 * s)
            if world.startswith('ящик'):
                X = rs.random((a.n, a.k))
                R = rel_of(X)
                S = X                                  # шкалы = координаты
                t_true = X.sum(1)                      # «время» = сумма шкал
                r_true = X.var(1)                      # «заряд» = разброс
                geo_t, geo_r = t_true, r_true
            else:
                X = sprinkle(a.n, a.k, rs)
                R = cone_rel(X)
                # шкалы направления u: t - u*x; берём много направлений
                D = rs.normal(size=(60, a.k - 1))
                D /= np.linalg.norm(D, axis=1, keepdims=True)
                S = X[:, 0:1] - X[:, 1:] @ D.T
                t_true = S.sum(1)
                r_true = S.var(1)
                geo_t = X[:, 0]                        # настоящее время
                geo_r = np.linalg.norm(X[:, 1:], axis=1)   # настоящий радиус
            F = order_features(R)
            for nm, y in (('сумма шкал', t_true), ('разброс шкал', r_true),
                          ('одна шкала', S[:, 0]), ('вторая шкала', S[:, 1]),
                          ('координата t', geo_t), ('радиус |x|', geo_r)):
                acc.setdefault(nm, []).append(r2(F, y))
        rows[world] = {k: (float(np.mean(v)), float(np.std(v)))
                       for k, v in acc.items()}

    names = ['сумма шкал', 'разброс шкал', 'одна шкала', 'вторая шкала',
             'координата t', 'радиус |x|']
    print(f"{'величина входа':>18}", end='')
    for w in rows:
        print(f'{w:>28}', end='')
    print()
    for nm in names:
        print(f'{nm:>18}', end='')
        for w in rows:
            m, sd = rows[w][nm]
            print(f'{f"R² = {m:.3f} ± {sd:.3f}":>28}', end='')
        print()
    print()
    print('  архив (независимые шкалы): сумма 0.948, разброс 0.909,')
    print('  отдельные шкалы 0.121 и 0.045')


if __name__ == '__main__':
    main()
