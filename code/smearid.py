#!/usr/bin/env python3
"""ТОЖДЕСТВО НА ЭЛЕМЕНТ ДЛЯ РАЗМАЗАННОГО ОПЕРАТОРА.

smearop.py: размазка погасила шум (ошибки 0.155, 0.020, 0.150 против
целей 2, -2, 4), и под ним вылезло смещение - средние -0.52, +0.45,
-0.97, то есть 16-122 сигмы мимо, со неверными знаками. При eps=0.1
значение B x^2 = +0.44 одинаково при N=900 и N=1800: смещение не
зависит от N, значит это НОРМИРОВКА, а не предел.

Проверяю тем же способом, каким утром нашёлся собственный
коэффициент -1/2: не вспоминая коэффициенты, а измеряя условие,
которому они обязаны удовлетворять. Скобка при f(x) обязана гаснуть:

    eps * c_self + sum_m W_m(eps) * n_m(x) = 0,   c_self = -1/2
    то есть   sum_m W_m(eps) n_m  =  eps/2

где n_m(x) - сколько y < x имеют интервал размера m. Для резкого
оператора (eps=1) это и есть измеренное утром <sum w n> = 0.5.

ЧТО РАЗЛИЧАЕТ ИСХОДЫ:
  * отношение (sum W n)/(eps/2) = 1  => нормировка верна, смещение
    из двойного предела, и материя при этих N недостижима;
  * отношение НЕ 1 и видно какое  => нормировка моя неверна, видно во
    сколько раз, и тест продолжается.
"""
import math

import numpy as np

from waveop import causal, sprinkle_11
from smearop import smeared_w


def ident(n, reps, eps, rs):
    vals = []
    for _ in range(reps):
        P = sprinkle_11(n, rs)
        t, x = P[:, 0], P[:, 1]
        Rc = causal(P)
        core = (t > 0.3) & (t < 0.7) & (np.abs(x) < 0.15)
        if core.sum() < 5:
            continue
        F = Rc.astype(np.float32)
        Iv = np.rint(F @ F).astype(np.int32)
        Warr = smeared_w(eps, int(Iv.max()) + 1)
        M = np.where(Rc.T, Warr[Iv.T], 0.0)
        vals.append(float(M.sum(axis=1)[core].mean()))
    return float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals)))


def main():
    print(__doc__)
    rs = np.random.default_rng(31415926)
    print(f"{'N':>6}{'eps':>7}{'sum W n':>12}{'ошибка':>9}"
          f"{'eps/2':>9}{'отношение':>11}")
    for n, reps in ((900, 30), (1800, 14)):
        for eps in (1.0, 0.5, 0.25, 0.1):
            m, e = ident(n, reps, eps, rs)
            print(f'{n:>6}{eps:>7.2f}{m:>12.4f}{e:>9.4f}'
                  f'{eps / 2:>9.4f}{m / (eps / 2):>11.3f}', flush=True)
    print('\nпри eps=1 отношение обязано быть около 1 - это утреннее 0.5')


if __name__ == '__main__':
    main()
