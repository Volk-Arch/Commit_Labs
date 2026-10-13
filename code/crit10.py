#!/usr/bin/env python3
"""КРИВАЯ ТЕПЛОЁМКОСТИ целиком, а не argmax.

crit9.py прошёл контроль (2.0 сигмы блочной ошибкой; наивная была втрое
занижена — отсюда и прежние «8.1 сигмы»). Но скан читать было нельзя:
beta пика вышло 0.080, 0.300, 0.200 при N = 6, 9, 12, причём 0.300 —
КРАЙ лестницы, а точное перечисление даёт убывание (0.070 при N=6,
0.060 при N=7). Плюс минимальная доля принятых ходов была 0.000, то
есть какая-то цепь стояла.

Здесь: лестница продлена, и печатается ВСЯ кривая C(beta) вместе с
долей принятых ходов на каждом beta — чтобы видеть, где числам можно
верить, а где цепь стоит.

КОНТРОЛЬ: N=6 против точного перечисления на всей кривой.
"""
import math

import numpy as np

from crit import enumerate_exact
from crit9 import blockerr, run

BET = [0.0, 0.02, 0.04, 0.055, 0.07, 0.085, 0.10, 0.125, 0.15, 0.2,
       0.3, 0.45, 0.7, 1.0]


def main():
    print(__doc__)
    rs = np.random.default_rng(1001)

    print('КОНТРОЛЬ: N = 6, вся кривая')
    S6 = enumerate_exact(6)
    Ss, Rs, accr, swr, bad = run(6, BET, 2000, rs)
    worst = 0.0
    print(f"{'beta':>6}{'C точно':>10}{'C отжиг':>10}"
          f"{'<S> расх.':>11}{'принят.':>9}")
    for k, b in enumerate(BET):
        w = np.exp(-b * (S6 - S6.min())); w /= w.sum()
        ex = float((S6 * w).sum())
        vx = float((S6 ** 2 * w).sum() - ex ** 2) * b * b
        m = Ss[k].mean()
        be = max(blockerr(Ss[k]), 0.02)
        dv = abs(m - ex) / be
        worst = max(worst, dv)
        print(f'{b:>6.3f}{vx:>10.3f}{b*b*Ss[k].var():>10.3f}'
              f'{dv:>10.1f}с{accr[k]:>9.3f}')
    print(f'  наибольшее {worst:.1f}с => '
          f'{"ПРОЙДЕН" if worst < 5 else "ПРОВАЛЕН"}')
    if worst >= 5:
        return

    for n in (9, 12, 15):
        Ss, Rs, accr, swr, bad = run(n, BET, 1100, rs)
        npair = n * (n - 1) / 2
        print(f'\nN = {n}: обменов {swr:.3f}, сбоев {bad}')
        print(f"{'beta':>6}{'C':>9}{'<S>':>10}{'доля связ.':>12}"
              f"{'принят.':>9}{'верить?':>9}")
        cs = []
        for k, b in enumerate(BET):
            c = b * b * Ss[k].var()
            cs.append(c if accr[k] > 0.002 else -1.0)
            ok = 'да' if accr[k] > 0.002 else 'НЕТ'
            print(f'{b:>6.3f}{c:>9.2f}{Ss[k].mean():>10.1f}'
                  f'{Rs[k].mean()/npair:>12.4f}{accr[k]:>9.4f}{ok:>9}')
        k = int(np.argmax(cs))
        print(f'  пик: beta = {BET[k]}, C = {cs[k]:.2f}, '
              f'доля связанных {Rs[k].mean()/npair:.4f} '
              f'(d=4 ждёт 0.1, d=2 ждёт 0.5)', flush=True)


if __name__ == '__main__':
    main()
