#!/usr/bin/env python3
"""КРУПНЫЙ ФРАКТАЛ: хватает ли сильного огрубления.

fracscale.py: огрубление улучшает фрактал на ДВА-ТРИ ПОРЯДКА (у затравки
с 2 связями 15264 -> 70.2 при пороге 19.5) и ЗАПОЛНЯЕТ ПРОВАЛЫ (нулей
среди m=1..8 было 2, стало 0). Направление догадки подтверждено. Но до
порога не дотянуло, и мешает скорее размер: при сильном прореживании N'
падало до 194, где выборка тонкая (8 различных |I|).

Починка: фрактал в шесть раз крупнее, N = 6^5 = 7776. Матрица строится
дёшево (кронекер, 60 МБ bool), а дорогой шаг - матричное умножение для
|I| - считается УЖЕ ПОСЛЕ прореживания, то есть на N' ~ 300..1500.
Значит можно огрублять в 5..25 раз, сохраняя выборку.

Отбор затравок двухступенчатый: сперва просеиваем все 318 при k=4
(дёшево), лучшие проверяем при k=5.

КОНТРОЛЬ прежний и точный: прореженный засев обязан оставаться в области
(прореживание пуассоновского процесса даёт пуассоновский процесс).
"""
import gc
import math

import numpy as np

from compose2 import posets
from sectors import sprinkle_d
from fractal import frac_poset, prof, region
from fracscale import thin, gaps

B = 6


def build(RB, k):
    R = RB.copy()
    for _ in range(k - 1):
        m = R.shape[0]
        R = (np.kron(RB, np.ones((m, m), bool))
             | np.kron(np.eye(B, dtype=bool), R))
        gc.collect()
    return R


def main():
    print(__doc__)
    rs = np.random.default_rng(5566778)

    mu, sd, chi4, thr4 = region(1296, rs, 25)
    scr = []
    for RB in posets(B):
        if RB.shape[0] != B:
            continue
        nr = int(RB.sum())
        if nr == 0 or nr == 15:
            continue
        R = build(RB, 4)
        T = thin(R, 0.3, rs)
        scr.append((float(chi4(prof(T))), nr, RB.copy()))
    scr.sort()
    print(f'просеяно затравок: {len(scr)} (при k=4, p=0.3, '
          f'порог {thr4:.1f})')
    print('  лучшие пять: ' + ', '.join(f'{c:.0f} ({nr}св)'
                                        for c, nr, _ in scr[:5]))

    N0 = B ** 5
    print(f'\nкрупный фрактал N={N0}')
    print(f"{'p':>6}{'N′':>7}{'порог':>8}{'ЗАСЕВ':>9}"
          + ''.join(f'{f"фр {nr}св":>12}' for _, nr, _ in scr[:4]))
    big = [build(RB, 5) for _, _, RB in scr[:4]]
    for p in (0.20, 0.12, 0.07, 0.04):
        n2 = int(N0 * p)
        mu, sd, chi, thr = region(n2, rs, 25)
        S = sprinkle_d(N0 // 2, 4, rs)
        cs = float(chi(prof(thin(S, p * 2, rs))))
        cells = [float(chi(prof(thin(R, p, rs)))) for R in big]
        print(f'{p:>6.2f}{n2:>7}{thr:>8.1f}{cs:>9.1f}'
              + ''.join(f'{v:>12.1f}' for v in cells), flush=True)

    print('\nспектр лучшего при сильном огрублении:')
    R = big[0]
    for p in (1.0, 0.2, 0.07):
        T = thin(R, p, rs) if p < 1 else R
        if T.shape[0] > 2500:
            T = thin(R, 2500 / R.shape[0], rs)
        h, nz, nu = gaps(T)
        print(f'  p={p:>4.2f} (N′={T.shape[0]}): '
              + ' '.join(f'{v:>6}' for v in h)
              + f'   нулей {nz}, различных |I| {nu}')


if __name__ == '__main__':
    main()
