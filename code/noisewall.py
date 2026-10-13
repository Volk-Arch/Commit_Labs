#!/usr/bin/env python3
"""ПРЕПЯТСТВИЕ ВЫЧИСЛИТЕЛЬНОЕ ИЛИ СТАТИСТИЧЕСКОЕ.

Час назад я заявил: в 1+1 материя достижима, потому что препятствие
вычислительное (матрица N^2) и снимается алгоритмом O(N). Алгоритм
написан и сверен до 1e-14. Но прогон до N=3e5 дал значения, СОВМЕСТИМЫЕ
С НУЛЁМ при растущих ошибках (±13, ±156, ±1548, ±1725, ±11173) - то есть
шум, а не расходимость.

Значит надо проверить саму посылку. Если разброс НА ЭЛЕМЕНТ растёт с N,
препятствие статистическое, и ускорение его не снимает: нужное число
точек растёт как квадрат разброса.

Мера: разброс sigma значений B f по отдельным элементам ядра, и отсюда
нужное число точек M = (sigma/0.5)^2 для ошибки 0.5 при цели 2.

ЧТО РАЗЛИЧАЕТ ИСХОДЫ: sigma стоит => дело было в алгоритме, и я прав;
sigma растёт как N => препятствие статистическое, моя стратегическая
посылка НЕВЕРНА, и это надо записать.
"""
import math

import numpy as np

from fastwave import sprinkle_uv, tx, bop_at, CSELF, W


def spread(n, rs, npts=400):
    P = sprinkle_uv(n, rs)
    U, V = P[:, 0], P[:, 1]
    t, x = tx(P)
    order_u = np.argsort(U)
    pos = np.empty(n, int)
    pos[order_u] = np.arange(n)
    ok = np.nonzero((t > 0.3) & (t < 0.7) & (np.abs(x) < 0.15))[0]
    pick = rs.choice(ok, size=min(npts, len(ok)), replace=False)
    rho = 2.0 * n
    f = x * x
    vals = []
    for idx in pick:
        ys, ws = bop_at(U, V, idx, order_u, pos)
        s = float((ws * f[ys]).sum()) if len(ys) else 0.0
        vals.append(rho * (CSELF * f[idx] + s))
    v = np.array(vals)
    return float(v.mean()), float(v.std()), len(ok)


def main():
    print(__doc__)
    print(f"{'N':>8}{'ядро':>9}{'среднее':>11}{'sigma':>12}"
          f"{'M для ошибки 0.5':>19}{'доля ядра':>12}")
    ns, sg = [], []
    for n in (1000, 3000, 10000, 30000, 100000):
        rs = np.random.default_rng(4242 + n)
        m, s, nc = spread(n, rs)
        need = (s / 0.5) ** 2
        ns.append(n)
        sg.append(s)
        print(f'{n:>8}{nc:>9}{m:>11.1f}{s:>12.1f}{need:>19.3g}'
              f'{need / nc:>12.3g}', flush=True)
    p = np.polyfit(np.log(ns), np.log(sg), 1)[0]
    print(f'\nпоказатель роста разброса: sigma ~ N^{p:.3f}')
    print(f'нужное число точек растёт как N^{2 * p:.2f}, а ядро как N^1')
    print(f'=> разрыв растёт как N^{2 * p - 1:.2f}')
    for N in (1e6, 1e9):
        print(f'   при N={N:.0g}: нужно ~{(sg[-1] * (N / ns[-1]) ** p / 0.5) ** 2:.2g} '
              f'точек, а в ядре их ~{0.2 * N:.1g}')


if __name__ == '__main__':
    main()
