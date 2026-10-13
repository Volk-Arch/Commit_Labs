#!/usr/bin/env python3
"""СТАНОВИТСЯ ЛИ ФРАКТАЛ ЗАСЕВОМ НА ДРУГОМ МАСШТАБЕ.

Мысль: провалы в спектре интервалов - по форме ровно то, из чего в
программе Соркина читается КРИВИЗНА (всё действие БД есть взвешенная
недостача N_0..N_3). И главное - фундаментальный масштаб ненаблюдаем, мы
видим множество ЧЕРЕЗ ОГРУБЛЕНИЕ. А огрубление СЛУЧАЙНО, и случайное
прореживание детерминированного фрактала может ЗАПОЛНИТЬ провалы. Тогда
фрактал, негодный на своём масштабе, окажется засевом на наблюдаемом.

Огрубление: оставляем каждый элемент с вероятностью p и берём
ИНДУЦИРОВАННЫЙ порядок - для транзитивного отношения это просто
подматрица R[idx][:,idx].

КОНТРОЛЬ ИДЕАЛЬНЫЙ И ВСТРОЕННЫЙ: прореживание ЗАСЕВА даёт засев меньшей
плотности - это свойство пуассоновского процесса. Значит прореженный
засев обязан оставаться в области засева при всех p. Если нет - сломана
область или прореживание, и про фрактал говорить нельзя.

ЧТО РАЗЛИЧАЕТ ИСХОДЫ:
  * фрактал входит в область при каком-то p => догадка верна,
    самоподобие годится, провалы были артефактом масштаба;
  * не входит ни при каком p, а провалы лишь редеют => детерминированное
    самоподобие не спасается огрублением.
"""
import math

import numpy as np

from compose2 import posets
from sectors import sprinkle_d
from fractal import frac_poset, prof, region

B, K = 6, 4


def thin(R, p, rs):
    keep = np.nonzero(rs.random(R.shape[0]) < p)[0]
    return R[np.ix_(keep, keep)]


def gaps(R):
    F = R.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    sz = Iv[R]
    h = np.array([int((sz == m).sum()) for m in range(9)])
    return h, int((h[1:] == 0).sum()), len(np.unique(sz))


def main():
    print(__doc__)
    rs = np.random.default_rng(1122334)

    seeds = []
    for RB in posets(B):
        if RB.shape[0] != B:
            continue
        nr = int(RB.sum())
        if nr in (1, 2, 4) and len(seeds) < 3:
            seeds.append((nr, RB.copy()))
    print(f'затравки: ' + ', '.join(f'{n} связей' for n, _ in seeds))

    N0 = B ** K
    print(f'\nфрактал N={N0}, прореживание до p·N')
    print(f"{'p':>6}{'N′':>7}{'порог':>8}{'ЗАСЕВ (контроль)':>20}"
          + ''.join(f'{f"фрактал {n}св":>16}' for n, _ in seeds))
    for p in (1.0, 0.75, 0.5, 0.3, 0.15):
        n2 = int(N0 * p)
        if n2 < 60:
            continue
        mu, sd, chi, thr = region(n2, rs, 30)
        S = sprinkle_d(N0, 4, rs)
        cs = float(chi(prof(thin(S, p, rs)))) if p < 1.0 else \
            float(chi(prof(sprinkle_d(n2, 4, rs))))
        cells = []
        for nr, RB in seeds:
            R = frac_poset(RB, K)
            T = thin(R, p, rs) if p < 1.0 else R
            cells.append(float(chi(prof(T))))
        print(f'{p:>6.2f}{n2:>7}{thr:>8.1f}{cs:>20.1f}'
              + ''.join(f'{v:>16.1f}' for v in cells), flush=True)

    print('\nредеют ли провалы при огрублении (затравка с 2 связями):')
    RB = [RB for nr, RB in seeds if nr == 2][0]
    R = frac_poset(RB, K)
    for p in (1.0, 0.75, 0.5, 0.3, 0.15):
        T = thin(R, p, rs) if p < 1.0 else R
        h, nz, nu = gaps(T)
        print(f'  p={p:>4.2f}: |I|=0..8 -> ' + ' '.join(f'{v:>6}' for v in h)
              + f'   нулей среди m=1..8: {nz}, различных |I|: {nu}')
    S = sprinkle_d(N0, 4, rs)
    h, nz, nu = gaps(S)
    print(f'  засев   : |I|=0..8 -> ' + ' '.join(f'{v:>6}' for v in h)
          + f'   нулей: {nz}, различных |I|: {nu}')


if __name__ == '__main__':
    main()
