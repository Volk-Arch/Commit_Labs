#!/usr/bin/env python3
"""ДОСТАТОЧНО ЛИ ДВУМЕРИЯ В ОСНОВЕ: может ли dim=2 выглядеть как 4d.

Мысль: может, четырёхмерие не нужно, а достаточно двумерия в основе.
Повод думать, что да: доля связанных у двумерного порядка (перестановки)
бывает ЛЮБОЙ от 0 (обратная перестановка) до 1 (тождественная). Значит
можно настроить её на 0.1 - значение Майрхейма-Майера для d=4, - и такой
порядок будет выглядеть ЧЕТЫРЁХМЕРНЫМ по этой статистике, оставаясь
двумерным по порядковой размерности.

Геометрически это понятно: двумерный порядок есть точки в квадрате с
покоординатным порядком, и малая доля связанных означает сгущение у
антидиагонали, то есть ТОНКИЙ ПЛАСТ пространства-времени 1+1 - широкий
по пространству, короткий по времени. Оценщик размерности через долю
связанных на форму и размерность не различает (это же мы видели утром в
`crit-geom-doc.md`: форма и размерность сцеплены).

Поэтому спрашиваем ПРОФИЛЕМ ОБИЛИЙ - единственной величиной, которая
подгонке не поддаётся (`prostoe-doc.md`): ни параметрами, ни смешиванием.

Семейство: перестановки по Маллоусу, P(tau) ~ q^{inv(tau)} при
q = exp(-b/n). Параметр b даёт весь диапазон доли связанных, и врагу
даётся ПОЛНАЯ свобода по b - минимум хи2 по сетке.

ПРЕДСКАЗАНИЕ до замера: не пройдёт, потому что профиль - инструмент
тоньше одной статистики, и в этом был весь смысл сегодняшнего дня.

КОНТРОЛИ:
  (a) при b=0 (равномерные перестановки) доля связанных обязана быть 0.5
      и порядок обязан лежать в области засева 1+1;
  (b) засев 4d обязан лежать в своей области.
"""
import math

import numpy as np

from sectors import sprinkle_d
from waveop import causal, sprinkle_11
from ratios import nm


def mallows(n, b, rs):
    """перестановка по Маллоусу, q = exp(-b/n), вставками."""
    q = math.exp(-b / n)
    out = []
    for i in range(1, n + 1):
        w = q ** np.arange(i)
        w /= w.sum()
        j = int(rs.choice(i, p=w))
        out.insert(i - 1 - j, i - 1)
    return np.array(out)


def order_from_perm(t):
    n = len(t)
    T = np.empty(n, int)
    T[t] = np.arange(n)          # позиция элемента = первая шкала
    A = np.argsort(t)
    pos = np.empty(n, int)
    pos[A] = np.arange(n)
    R = np.triu(np.ones((n, n), bool), 1) & (pos[:, None] < pos[None, :])
    return R


def prof(R):
    v, _ = nm(R)
    return v[1:] / max(v[0], 1.0)


def region(mk, rep, rs):
    A = np.array([prof(mk()) for _ in range(rep)])
    mu, sd = A.mean(axis=0), A.std(axis=0, ddof=1)
    chi = lambda X: (((X - mu) / sd) ** 2).sum(axis=-1)
    return chi, float(np.percentile(chi(A), 95))


def rfrac(R):
    n = R.shape[0]
    return float(R.sum()) / (n * (n - 1) / 2)


def main():
    print(__doc__)
    rs = np.random.default_rng(141421)
    n = 600

    chi4, thr4 = region(lambda: sprinkle_d(n, 4, rs), 25, rs)
    chi2, thr2 = region(lambda: causal(sprinkle_11(n, rs)), 25, rs)
    print(f'области при N={n}: порог 4d {thr4:.1f}, порог 1+1 {thr2:.1f}')
    print(f'КОНТРОЛЬ (b): засев 4d в своей области -> '
          f'{float(chi4(prof(sprinkle_d(n, 4, rs)))):.1f}')

    print(f'\nКОНТРОЛЬ (a): b=0 обязано дать r=0.5 и лечь в область 1+1')
    R0 = order_from_perm(mallows(n, 0.0, rs))
    print(f'  r={rfrac(R0):.4f}, хи2 к области 1+1 = '
          f'{float(chi2(prof(R0))):.1f} '
          f'=> {"ОК" if float(chi2(prof(R0))) < 3 * thr2 else "СБОЙ"}')

    print(f'\nдвумерный порядок с полной свободой по b, против области 4d:')
    print(f"{'b':>8}{'доля связ.':>12}{'хи2 к 4d':>12}{'хи2 к 1+1':>12}")
    best = None
    for b in (-60, -40, -25, -15, -8, -4, 0, 4, 8):
        R = order_from_perm(mallows(n, b, rs))
        r = rfrac(R)
        p = prof(R)
        c4, c2 = float(chi4(p)), float(chi2(p))
        print(f'{b:>8}{r:>12.4f}{c4:>12.1f}{c2:>12.1f}', flush=True)
        if best is None or c4 < best[0]:
            best = (c4, b, r)
    print(f'\nнаилучшее против 4d: b={best[1]}, r={best[2]:.4f}, '
          f'хи2={best[0]:.1f} при пороге {thr4:.1f} '
          f'=> превышение {best[0] / thr4:.0f}x')


if __name__ == '__main__':
    main()
