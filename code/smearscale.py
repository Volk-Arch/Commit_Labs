#!/usr/bin/env python3
"""ЧТО УПРАВЛЯЕТ ОСЛАБЛЕНИЕМ: N ИЛИ eps*N.

smearop.py + smearid.py: нормировка верна (тождество на элемент даёт
1.006 при eps=0.1), но оператор ослаблен в ~4.5 раза, и множитель
0.222 +- 0.009 одинаков при eps = 0.25 и 0.1 и при N = 900 и 1800.

ГИПОТЕЗА. Размазанный оператор по построению есть среднее РЕЗКОГО
оператора при плотности eps*rho. Значит к пределу его ведёт не N, а
число эффективных элементов eps*N, и ослабление 0.22 - это конечность
eps*N ~ 90..180, которую у резкого оператора не видно под шумом
+-8..47, а размазка её показывает с ошибкой +-0.01.

ЧТО РАЗЛИЧАЕТ ИСХОДЫ:
  * пары с равным eps*N совпадают, и множитель растёт с eps*N
    => управляет eps*N, конструкция верна, видно какое N нужно;
  * пары не совпадают / множитель стоит
    => дело не в eps*N, и материя при доступных N закрыта.

Мера - самый тихий столбец B x^2 (ошибка 0.5% при eps=0.1), цель по
модулю 2; знак есть соглашение сигнатуры и для всех трёх функций общий.
"""
import math

import numpy as np

from waveop import causal, sprinkle_11
from smearop import smeared_w, CSELF


def factor(n, reps, eps, rs):
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
        f = x ** 2
        out = (n / 0.5) * (eps * CSELF * f + M @ f)
        vals.append(float(out[core].mean()) / -2.0)
    return float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals)))


def main():
    print(__doc__)
    rs = np.random.default_rng(14142135)
    print(f"{'eps*N':>7}" + ''.join(f'{f"N={n}":>20}'
                                    for n in (450, 900, 1800)))
    for en in (45, 90, 180, 360):
        row = f'{en:>7}'
        for n, reps in ((450, 60), (900, 40), (1800, 16)):
            eps = en / n
            if eps > 1.0:
                row += f'{"-":>20}'
                continue
            m, e = factor(n, reps, eps, rs)
            row += f'{m:>13.3f} {e:>6.3f}'
        print(row, flush=True)
    print('\nстолбцы - разные N, строки - равное число эффективных')
    print('элементов. Совпадение ВДОЛЬ СТРОКИ = управляет eps*N.')


if __name__ == '__main__':
    main()
