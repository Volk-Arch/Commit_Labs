#!/usr/bin/env python3
"""Заряд на настоящем причинном порядке.

Пересчёт antimatter-doc.md. На покомпонентном порядке было два правила:
переворот ВСЕХ шкал (ломал причинность) и переворот ОДНОЙ шкалы
(причинность сохранял, но не имел устойчивой размерности).

На конусном порядке «шкала» — это направление, и их континуум. Поэтому
прежние два правила переводятся так:

  A. переворот всех шкал  ->  обращение ВРЕМЕНИ для пар разных знаков.
     Это в точности картина Фейнмана–Штюкельберга: античастица как
     частица, идущая назад по времени.

  B. переворот одной шкалы -> пространственное отражение для пар разных
     знаков. Но конус инвариантен к отражениям, поэтому предсказание:
     отношение НЕ изменится вовсе, и прежнее «правило B» было артефактом
     независимости шкал.

  C. осмысленный остаток: у пар разных знаков ДРУГАЯ предельная скорость,
     то есть конус другой ширины, dt > c*|dx| с c != 1. При c > 1 такое
     отношение вложено в обычный причинный порядок, значит ациклично.

Проверяется: цикличность у A, тождественность у B, и у C — ацикличность
плюс устойчивость размерности по размеру мира (именно на устойчивости
прежнее правило B и умерло).

  python3 chargecone.py
"""
import argparse
import math
import time

import numpy as np

from coneworld import sprinkle
from core import dimension, height, spectral, frac_pairs
from majority import has_cycle, shortest_cycle


def charged_rel(X, q, mode, c=1.0):
    """Причинное отношение с зарядом. R[i,j] = True, если j раньше i."""
    dt = X[:, None, 0] - X[None, :, 0]
    dx = np.linalg.norm(X[:, None, 1:] - X[None, :, 1:], axis=-1)
    same = (q[:, None] * q[None, :]) > 0
    if mode == 'время':
        R = np.where(same, dt > dx, -dt > dx)
    elif mode == 'отражение':
        Y = X.copy()
        Y[:, 1] = -Y[:, 1]                      # отражение одной оси
        dxr = np.linalg.norm(X[:, None, 1:] - Y[None, :, 1:], axis=-1)
        R = np.where(same, dt > dx, dt > dxr)
    elif mode == 'скорость':
        R = np.where(same, dt > dx, dt > c * dx)
    else:
        R = dt > dx
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)


def main():
    ap = argparse.ArgumentParser(description='Заряд на конусном порядке')
    ap.add_argument('--n', type=int, default=900)
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--f', type=float, default=0.5)
    a = ap.parse_args()

    t0 = time.time()

    def world(N, seed):
        rs = np.random.default_rng(seed)
        X = sprinkle(N, 4, rs)
        q = np.where(rs.random(N) < a.f, -1, 1)
        return X, q

    print('A. ОБРАЩЕНИЕ ВРЕМЕНИ для пар разных знаков')
    print('   (Фейнман–Штюкельберг: античастица идёт назад по времени)')
    print(f"{'доля анти':>11}{'ациклично':>12}{'крат. цикл':>12}{'доля пар':>11}")
    for f in (0.0, 0.001, 0.01, 0.1, 0.5):
        bad = 0
        shortest = []
        for s in range(a.seeds):
            rs = np.random.default_rng(200 + 17 * s)
            X = sprinkle(a.n, 4, rs)
            q = np.where(rs.random(a.n) < f, -1, 1)
            R = charged_rel(X, q, 'время')
            if has_cycle(R):
                bad += 1
                shortest.append(shortest_cycle(R))
            fr = frac_pairs(R)
        sh = min([s for s in shortest if s > 0], default='-')
        print(f'{f:>11.3f}{f"{a.seeds-bad} из {a.seeds}":>12}{str(sh):>12}'
              f'{fr:>11.4f}')

    print()
    print('B. ПРОСТРАНСТВЕННОЕ ОТРАЖЕНИЕ для пар разных знаков')
    print('   предсказание: конус к отражениям инвариантен, отношение то же')
    for N in (600, 1200):
        X, q = world(N, 7)
        R0 = charged_rel(X, q, 'нет')
        R1 = charged_rel(X, q, 'отражение')
        print(f'  N={N}: различий {int((R0 != R1).sum())} из {R0.size}')

    print()
    print('C. ДРУГАЯ ПРЕДЕЛЬНАЯ СКОРОСТЬ у пар разных знаков')
    print('   устойчива ли размерность по размеру мира — на этом')
    print('   прежнее правило B и умерло (3.054 -> 3.001 -> 2.960)')
    print(f"{'c':>6}{'N':>7}{'ациклично':>11}{'доля пар':>11}"
          f"{'размерность':>16}{'спектр':>11}")
    for c in (1.0, 1.5, 3.0):
        for N in (600, 1200, 2400):
            V, acy = [], True
            for s in range(a.seeds):
                rs = np.random.default_rng(300 + 17 * s)
                X = sprinkle(N, 4, rs)
                q = np.where(rs.random(N) < a.f, -1, 1)
                R = charged_rel(X, q, 'скорость', c)
                if has_cycle(R):
                    acy = False
                V.append((frac_pairs(R), dimension(R=R), spectral(R)))
            V = np.array(V)
            m, sd = V.mean(0), V.std(0) / math.sqrt(a.seeds)
            print(f'{c:>6.1f}{N:>7}{("да" if acy else "НЕТ"):>11}'
                  f'{m[0]:>11.4f}{f"{m[1]:.3f} ± {sd[1]:.3f}":>16}'
                  f'{m[2]:>+11.4f}')
        print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
