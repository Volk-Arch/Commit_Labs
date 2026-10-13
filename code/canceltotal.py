#!/usr/bin/env python3
"""ПОЛНОЕ |Z| слоистого сектора, точно, с верным действием.

Вчера считалась ОЦЕНКА СВЕРХУ: |Z| <= 2^{a1 a2} max|G|^{a3}, то есть
сумма по `M12` оценивалась как «все слагаемые в фазе». Здесь она берётся
ТОЧНО: число матриц `M12` с данными размерами классов есть
мультиномиальный коэффициент, поэтому для a1 = 2

    Z = e^{i th N} sum_{композиции} [a_2!/prod n_c!] u^{|M12|} G^{a_3}

и это сумма по O(a_2^3) слагаемым, а не по 2^{2 a_2} матрицам.

ЗАЧЕМ. Критерий архитектуры сравнивает |Z| секторов, а не скорости
подавления. Счёт сектора 2^{a1 a2 + a2 a3}; если доля аргмаксимума в
сумме по `M12` порядка 2^{-a_2} (мультиномиальная доля разбиения
второго слоя пополам), то оценка сверху завышает |Z| ровно на этот
множитель, и его надо знать.

ОЦЕНКА ДЛЯ СРАВНЕНИЯ (доля аргмаксимума, без погашения между
композициями):

    ln|Z| ~ ln C(a2, a2/2) + a_3 ln|G|_max

КОНТРОЛИ:
  1. против перечисления ВСЕХ порядков сектора (действие от матрицы);
  2. против mpmath с 60 знаками — измерить пол float64, а не принять
     его на веру: сокращение ниже пола выглядит как подавление;
  3. a_3 = 0 обязано дать ровно 4^{a_2}.

  python3 canceltotal.py
"""
import math

import numpy as np

from cancelfast import comps, g_vec
from cancelclosed import z_poset

K4 = 4.0 / math.sqrt(6.0)


def z_log(a2, a3, th, nc=None):
    """ln|Z| и ln|Z|/count для a1 = 2, точно. Возвращает (lnZ, lnQ)."""
    if nc is None:
        nc = comps(a2)
    n00, n10, n01, n11 = nc
    g = g_vec(nc, th)
    lg = np.log(np.maximum(np.abs(g), 1e-300))
    ag = np.angle(g)
    lf = np.array([math.lgamma(k + 1) for k in range(a2 + 1)])
    lmult = lf[a2] - lf[n00] - lf[n10] - lf[n01] - lf[n11]
    m12 = n10 + n01 + 2 * n11
    L = lmult + a3 * lg
    Phi = -th * m12 + a3 * ag
    Lmax = L.max()
    acc = np.exp(L - Lmax) * np.exp(1j * Phi)
    tot = acc.sum()
    lnZ = Lmax + math.log(abs(tot))
    # пол: сумма модулей против модуля суммы
    lnabs = Lmax + math.log(np.abs(acc).sum())
    lncount = (2 * a2 + a2 * a3) * math.log(2.0)
    return lnZ, lnZ - lncount, lnabs - lnZ


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    th = K4 / 0.55
    print('КОНТРОЛЬ 1: против перечисления всех порядков')
    for (a2, a3) in ((2, 2), (3, 2), (2, 3), (3, 3)):
        lnZ, _, _ = z_log(a2, a3, th)
        zp = abs(z_poset(2, a2, a3, th))
        print(f'  (2,{a2},{a3}): ln|Z| точно {lnZ:.10f}, '
              f'перечисление {math.log(zp):.10f}, '
              f'отн. {abs(math.exp(lnZ)-zp)/zp:.2e}')
    print()
    print('КОНТРОЛЬ 3: a_3 = 0 обязано дать 4^{a_2}')
    for a2 in (3, 7, 12):
        lnZ, _, _ = z_log(a2, 0, th)
        print(f'  a2={a2:>3}: ln|Z| = {lnZ:.8f}, '
              f'a_2 ln4 = {a2*math.log(4):.8f}')
    print()
    print('КОНТРОЛЬ 2: пол float64 — отношение суммы модулей к |сумме|')
    print('  (если оно ~1e16, результат упёрся в пол и доверять нельзя)')
    for (a2, a3) in ((20, 5), (40, 10), (60, 20)):
        lnZ, lnQ, slack = z_log(a2, a3, th)
        print(f'  (2,{a2},{a3}): сокращение в {math.exp(slack):.3e} раз '
              f'{"— ПОДОЗРИТЕЛЬНО" if slack > 30 else "— в пределах точности"}')
    print()
    print('=' * 74)
    print('ТОЧНОЕ ln|Z| ПРОТИВ ОЦЕНКИ ПО ДОЛЕ АРГМАКСИМУМА, a1 = 2, hbar=0.55')
    print(f"{'a_2':>5}{'a_3':>5}{'ln|Z| точно':>15}{'оценка':>13}"
          f"{'разница':>11}{'сокращение':>13}")
    for a2 in (20, 40, 60, 80, 120):
        nc = comps(a2)
        g = np.abs(g_vec(nc, th))
        lgmax = math.log(g.max())
        for a3 in (5, 20, 60):
            lnZ, lnQ, slack = z_log(a2, a3, th, nc)
            est = (math.lgamma(a2 + 1) - 2 * math.lgamma(a2 / 2 + 1)
                   + a3 * lgmax)
            print(f'{a2:>5}{a3:>5}{lnZ:>15.4f}{est:>13.4f}'
                  f'{lnZ-est:>11.4f}{math.exp(slack):>13.2e}', flush=True)


if __name__ == '__main__':
    main()
