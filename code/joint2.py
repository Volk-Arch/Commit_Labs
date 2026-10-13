#!/usr/bin/env python3
"""ПЛОЩАДЬ СТЫКА ИЗ СГЛАЖЕННОГО ДЕЙСТВИЯ (пункт B1 плана).

ОТКУДА. Бак, Доукер, Джабб и Сурья ([1502.05388]) и Доукер
([2007.13206]): в `d = 4` предел действия БД на причинном ромбе равен
действию Эйнштейна–Гильберта ПЛЮС площадь стыка. У ПЛОСКОГО ромба
`∫R = 0`, значит действие БД есть ровно площадь стыка.

ПОЧЕМУ РАНЬШЕ БЫЛО НЕЛЬЗЯ. У резкого действия разброс 943 при `N = 1200`
(`bd.py`), а площадь стыка в единицах `ℓ²` идёт как `N^{1/2}` ≈ 35, то
есть сигнал в 27 раз меньше шума. У сглаженного разброс 4.9, то есть
сигнал к шуму около семи НА ОДНУ РЕАЛИЗАЦИЮ (`smear-doc.md`).

ПРЕДСКАЗАНИЕ ПОКАЗАТЕЛЯ, и оно жёсткое. Объём ромба `V = πτ⁴/24`,
`N = ρV`, поэтому `τ ∝ (N/ρ)^{1/4}`, площадь талии `∝ τ² ∝ (N/ρ)^{1/2}`,
а в единицах `ℓ² = ρ^{−1/2}` это

    площадь/ℓ² ∝ N^{1/2}.

Значит `S_ε(плоский ромб)` обязано идти как `N^{1/2}`. Это проверка без
единого свободного числа: объёмный член идёт как `N`, краевой как
`N^{3/4}`, площадь стыка как `N^{1/2}` — показатели разные, и подделать
их нечем.

КОНТРОЛЬ: тот же замер в двух измерениях, где площадь стыка есть
ТОЧКА, то есть `N^0` — показатель обязан упасть до нуля.

  python3 joint2.py
"""
import math
import time

import numpy as np

from sprinkle import sprinkle_d
from smear import smeared
from bd import action


def run(n, dim, eps, seeds, root):
    v = []
    for s in range(seeds):
        A = sprinkle_d(n, dim, np.random.default_rng(root + 17 * s))
        v.append(smeared(A, eps, dim) if eps > 0 else action(A, dim)[0])
        del A
    v = np.array(v, float)
    return v.mean(), v.std(ddof=1) / math.sqrt(len(v))


def slope(NS, Y, E):
    x = np.log(np.asarray(NS, float))
    y = np.log(np.abs(np.asarray(Y, float)))
    e = np.asarray(E, float) / np.abs(np.asarray(Y, float))
    w = 1.0 / np.maximum(e, 1e-12) ** 2
    M = np.vstack([np.ones_like(x), x]).T
    C = np.linalg.inv(M.T @ (w[:, None] * M))
    b = C @ (M.T @ (w * y))
    jk = []
    for i in range(len(x)):
        k = np.arange(len(x)) != i
        Mk, wk = M[k], w[k]
        Ck = np.linalg.inv(Mk.T @ (wk[:, None] * Mk))
        jk.append((Ck @ (Mk.T @ (wk * y[k])))[1])
    jk = np.array(jk)
    se = math.sqrt((len(x) - 1) / len(x) * ((jk - jk.mean()) ** 2).sum())
    return float(b[1]), se, math.exp(float(b[0]))


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    NS = (300, 600, 1200, 2400, 4800)
    for dim, tag, pred in ((4, 'ЧЕТЫРЁХМЕРНЫЙ ромб: стык — ПОВЕРХНОСТЬ', 0.5),
                           (2, 'ДВУМЕРНЫЙ ромб: стык — ТОЧКА (контроль)', 0.0)):
        print('=' * 96)
        print(tag + f', ожидаемый показатель {pred}')
        print(f"{'ε':>8}" + ''.join(f'{n:>15}' for n in NS)
              + f"{'показатель':>14}{'±':>8}{'коэф.':>10}")
        for eps in (0.3, 0.1, 0.03):
            Y, E = [], []
            for n in NS:
                m, e = run(n, dim, eps, 12, 700 + n + int(100 * eps))
                Y.append(m); E.append(e)
            k, se, c = slope(NS, Y, E)
            print(f'{eps:>8.2f}' + ''.join(f'{y:>9.2f}±{e:<5.2f}'
                                           for y, e in zip(Y, E))
                  + f'{k:>14.3f}{se:>8.3f}{c:>10.3f}')
            print(f'    ({time.time()-t0:.0f} с)', flush=True)
        print()
    print('=' * 96)
    print('РЕЗКОЕ действие для сравнения (шум обязан всё забить)')
    print(f"{'':>8}" + ''.join(f'{n:>15}' for n in NS))
    Y, E = [], []
    for n in NS:
        m, e = run(n, 4, 0.0, 12, 1500 + n)
        Y.append(m); E.append(e)
    print(f'{"резкое":>8}' + ''.join(f'{y:>9.0f}±{e:<5.0f}'
                                     for y, e in zip(Y, E)))
    print(f'    площадь ~ N^0.5 была бы {[f"{math.sqrt(n):.0f}" for n in NS]}')
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
